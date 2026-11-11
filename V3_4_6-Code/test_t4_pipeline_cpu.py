"""
CPU-only tests for t4_pipeline.py using synthetic fixtures (fake bundle ZIP, fake
safetensors file, fake trainer_state.json). No torch / transformers / GPU needed.

Run:  python -m unittest -v test_t4_pipeline_cpu
"""

import io
import sys
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import t4_pipeline as tp


def _png_bytes():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 10, 10)).save(buf, format="JPEG")
    return buf.getvalue()


def make_bundle(path: Path, sizes=None, overlap=False):
    """Synthetic bundle with the same layout as V3.3.2_pilot_bundle.zip."""
    sizes = sizes or {"train": (250, 6), "validation": (62, 2), "extracted_diagnostic": (0, 66)}
    rng = np.random.default_rng(0)
    img = _png_bytes()
    manifests = {}
    counter = 0
    with zipfile.ZipFile(path, "w") as z:
        for split, (n_gen, n_ext) in sizes.items():
            rows = []
            for src, n in (("generated", n_gen), ("extracted", n_ext)):
                for _ in range(n):
                    cid = f"c{counter:05d}"
                    counter += 1
                    k = int(rng.integers(3, 8))
                    ann = {"chart-type": "vertical_bar", "source": src,
                           "data-series": [{"x": f"L{j}", "y": float(rng.uniform(1, 100))} for j in range(k)]}
                    z.writestr(f"train/annotations/{cid}.json", json.dumps(ann))
                    z.writestr(f"train/images/{cid}.jpg", img)
                    rows.append({"chart_id": cid, "chart_type": "vertical_bar", "source": src, "n_points": k,
                                 "image_member": f"train/images/{cid}.jpg",
                                 "annotation_member": f"train/annotations/{cid}.json"})
            df = pd.DataFrame(rows)
            manifests[split] = df
        if overlap:
            manifests["validation"].iloc[0] = manifests["train"].iloc[0]
        for split, member in tp.MANIFESTS.items():
            z.writestr(member, manifests[split].to_csv(index=False))
    return manifests


def write_fake_safetensors(path: Path, truncate=0):
    header = {"a.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
              "b.bias": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]},
              "__metadata__": {"format": "pt"}}
    h = json.dumps(header).encode()
    data = struct.pack("<Q", len(h)) + h + bytes(24)
    path.write_bytes(data[:len(data) - truncate] if truncate else data)


def write_ckpt(run_dir: Path, step, epoch, eval_loss, complete=True, resume=False):
    ck = run_dir / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    (ck / "config.json").write_text("{}")
    write_fake_safetensors(ck / "model.safetensors", truncate=0 if complete else 5)
    hist = [{"epoch": e + 1, "loss": 5.0 - e, "step": (e + 1) * 64} for e in range(epoch)]
    hist += [{"epoch": e + 1, "eval_loss": l, "eval_benetech_score": 0.0, "step": (e + 1) * 64}
             for e, l in enumerate(eval_loss)]
    (ck / "trainer_state.json").write_text(json.dumps(
        {"global_step": step, "epoch": float(epoch), "log_history": hist,
         "best_model_checkpoint": str(run_dir / "checkpoint-64")}))
    if resume:
        for f in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (ck / f).write_bytes(b"x")
    return ck


class TestManifests(unittest.TestCase):
    def test_verified_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "b.zip"
            make_bundle(zp)
            dfs = tp.load_and_verify_manifests(zp)
            self.assertEqual(len(dfs["validation"]), 64)

    def test_overlap_detected(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "b.zip"
            make_bundle(zp, overlap=True)
            with self.assertRaises(AssertionError):
                tp.load_and_verify_manifests(zp)

    def test_wrong_counts_detected(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "b.zip"
            make_bundle(zp, sizes={"train": (250, 5), "validation": (62, 2), "extracted_diagnostic": (0, 66)})
            with self.assertRaises(AssertionError):
                tp.load_and_verify_manifests(zp)

    def test_image_loader_matches_pil(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "b.zip"
            m = make_bundle(zp)
            with zipfile.ZipFile(zp) as z:
                arr = tp.load_rgb_array(z, m["train"]["image_member"].iloc[0])
            self.assertEqual(arr.shape, (30, 40, 3))
            self.assertTrue(arr.flags.writeable)


class TestCheckpoints(unittest.TestCase):
    def test_integrity_ok_and_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.safetensors"
            write_fake_safetensors(p)
            self.assertTrue(tp.safetensors_integrity(p)["ok"])
            write_fake_safetensors(p, truncate=5)
            self.assertFalse(tp.safetensors_integrity(p)["ok"])

    def test_inventory_and_selection(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "deplot_pilot"
            write_ckpt(run, 64, 1, [1.634054], resume=True)
            write_ckpt(run, 128, 2, [1.634054, 1.557504])
            inv = tp.inventory_checkpoints(run)
            self.assertEqual(list(inv["checkpoint"]), ["checkpoint-64", "checkpoint-128"])
            self.assertTrue(inv["resume_ready"].iloc[0])
            self.assertFalse(inv["resume_ready"].iloc[1])
            self.assertTrue(inv["weights_complete"].all())
            self.assertAlmostEqual(inv["eval_loss_at_ckpt"].iloc[1], 1.557504)
            sel = tp.select_checkpoint_by_eval_loss(run)
            self.assertEqual(sel["checkpoint"], "checkpoint-128")
            hist = tp.training_history(run / "checkpoint-128" / "trainer_state.json")
            self.assertEqual(list(hist["eval_loss"]), [1.634054, 1.557504])

    def test_truncated_download_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "r"
            write_ckpt(run, 64, 1, [1.6], complete=False)
            self.assertFalse(tp.inventory_checkpoints(run)["weights_complete"].iloc[0])

    def test_numeric_sort_of_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "r"
            for s in (1000, 64, 128):
                write_ckpt(run, s, 1, [1.0])
            self.assertEqual(list(tp.inventory_checkpoints(run)["checkpoint"]),
                             ["checkpoint-64", "checkpoint-128", "checkpoint-1000"])

    def test_mid_epoch_checkpoint_not_credited_with_previous_eval(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "r"
            write_ckpt(run, 64, 1, [1.6])
            write_ckpt(run, 100, 1, [1.6])          # time-limit save: last eval was at step 64
            inv = tp.inventory_checkpoints(run)
            self.assertTrue(np.isnan(inv["eval_loss_at_ckpt"].iloc[1]))
            self.assertEqual(tp.select_checkpoint_by_eval_loss(run)["checkpoint"], "checkpoint-64")

    def test_selection_without_eval_loss_raises(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "r"
            run.mkdir()
            with self.assertRaises(RuntimeError):
                tp.select_checkpoint_by_eval_loss(run)


class TestScoringPipeline(unittest.TestCase):
    def test_score_predictions_complete_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            zp = d / "b.zip"
            m = make_bundle(zp)
            val = m["validation"]
            targets = tp.read_targets(zp, val)
            # Perfect predictions in byte-token form for all but the last 3 charts;
            # chart 0 gets a missing row.
            rows = []
            for i, (cid, src, t) in enumerate(zip(val["chart_id"], val["source"], targets)):
                if i >= len(val) - 3:
                    continue
                text = t.replace("\n", "<0x0A>")
                if i == 0:
                    text = "<0x0A>".join(text.split("<0x0A>")[:-1])
                rows.append({"chart_id": cid, "source": src, "prediction_text": text,
                             "hit_token_limit": False})
            pd.DataFrame(rows).to_csv(d / "p.csv", index=False)
            out = tp.score_predictions(d / "p.csv", val, zp, header_mode="none", out_prefix=d / "exp")
            s = out["summary"]
            self.assertFalse(s["complete"])
            self.assertEqual(s["n_missing_predictions"], 3)
            self.assertEqual(s["n_scored"], 61)
            self.assertAlmostEqual(s["overall"]["strict_benetech_score"], 60 / 61)
            self.assertAlmostEqual(s["overall"]["row_count_accuracy"], 60 / 61)
            self.assertTrue((d / "exp_summary.json").is_file())
            table = tp.results_table(d)
            self.assertEqual(len(table), 1)
            self.assertFalse(table["complete"].iloc[0])

    def test_generation_cap_rule(self):
        r = tp.choose_generation_cap([100, 400, 600], [500, 700])
        self.assertEqual(r["max_new_tokens"], 928)  # 1.5*600=900 -> 928
        # Evaluation lengths must NOT move the cap; they are only reported, per split.
        r = tp.choose_generation_cap([100, 400, 600], {"validation": [500, 1000], "extracted_diagnostic": [1145]})
        self.assertEqual(r["max_new_tokens"], 928)
        self.assertEqual(r["eval_refs_over_cap"]["validation"]["n_over_cap"], 1)
        self.assertEqual(r["eval_refs_over_cap"]["extracted_diagnostic"]["longest_ref_tokens"], 1145)
        self.assertEqual(tp.choose_generation_cap([10_000])["max_new_tokens"], 1152)   # ceiling


class _FakeTokenizer:
    """Character-level stand-in for the Pix2Struct tokenizer (decision logic test only).
    mode: 'byte'  -> '\\n' encoded as the <0x0A> byte token (decodes as literal '<0x0A>')
          'drop'  -> '\\n' normalized to a space (row structure lost)
          'lossy' -> like 'byte' but the character '~' is not in the vocabulary (<unk>)"""
    pad_token_id, eos_token_id, unk_token_id, NL = 0, 1, 2, 5

    def __init__(self, mode):
        self.mode = mode

    def get_vocab(self):
        return {"<0x0A>": self.NL}

    def convert_tokens_to_ids(self, tok):
        return self.NL if tok == "<0x0A>" else self.unk_token_id

    def convert_ids_to_tokens(self, ids):
        return ["<0x0A>" if i == self.NL else chr(i) if i > 31 else f"<{i}>" for i in ids]

    def __call__(self, text, add_special_tokens=True, truncation=False, **kw):
        ids, i = [], 0
        while i < len(text):
            if text.startswith("<0x0A>", i):
                ids.append(self.NL); i += 6; continue
            ch = text[i]
            if ch == "\n":
                ids.append(self.NL if self.mode in ("byte", "lossy") else ord(" "))
            elif ch == "~" and self.mode == "lossy":
                ids.append(self.unk_token_id)
            else:
                ids.append(ord(ch))
            i += 1
        return {"input_ids": ids + ([self.eos_token_id] if add_special_tokens else [])}

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            if i in (0, 1, 2):
                continue
            out.append("<0x0A>" if i == self.NL else chr(i))
        return "".join(out)


class _FakeProcessor:
    def __init__(self, mode):
        self.tokenizer = _FakeTokenizer(mode)
        self.image_processor = type("IP", (), {"is_vqa": True, "max_patches": 2048})()


class TestTokenizerDiagnosticLogic(unittest.TestCase):
    TARGETS = ["Vanuatu | 1\nVenezuela,\nRB | 2\nZambia | 3", "a | 1.5\nb | 2"]

    def test_branch_a_with_multiline_labels_exact(self):
        r = tp.tokenizer_diagnostic(_FakeProcessor("byte"), self.TARGETS)
        self.assertEqual(r["BRANCH"], "A_NEWLINE_ENCODED")
        self.assertEqual(r["real_target_roundtrip_exact"], "2/2")

    def test_branch_b_when_newline_dropped(self):
        r = tp.tokenizer_diagnostic(_FakeProcessor("drop"), self.TARGETS)
        self.assertEqual(r["BRANCH"], "B_NEWLINE_LOST")
        self.assertEqual(r["real_target_roundtrip_row_preserved"], "0/2")

    def test_text_change_with_intact_rows_is_reported_not_branching(self):
        r = tp.tokenizer_diagnostic(_FakeProcessor("lossy"), ["a~b | 1\nc | 2"])
        self.assertEqual(r["BRANCH"], "A_NEWLINE_ENCODED")
        self.assertEqual(r["real_target_roundtrip_exact"], "0/1")
        self.assertEqual(r["roundtrip_examples_of_changed_text"][0]["label_changes"], [("a~b", "ab")])


class TestLabelInventory(unittest.TestCase):
    def test_label_issue_report_finds_planted_newline_label(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src, dst = d / "clean.zip", d / "b.zip"
            m = make_bundle(src)
            target = m["validation"]["annotation_member"].iloc[0]
            with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w") as zo:
                for n in zi.namelist():
                    b = zi.read(n)
                    if n == target:
                        a = json.loads(b)
                        a["data-series"][0]["x"] = "West Bank\nand Gaza"
                        b = json.dumps(a).encode()
                    zo.writestr(n, b)
            dfs = tp.load_and_verify_manifests(dst)
            rep = tp.label_issue_report(dst, dfs)
            self.assertEqual(len(rep), 1)
            self.assertEqual(rep["split"].iloc[0], "validation")
            self.assertEqual(rep["n_newline_labels"].iloc[0], 1)

    def test_label_issue_report_empty(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "b.zip"
            make_bundle(zp)
            self.assertEqual(len(tp.label_issue_report(zp, tp.load_and_verify_manifests(zp))), 0)


class TestReportingHelpers(unittest.TestCase):
    def test_step_log_captures_output_and_traceback(self):
        import contextlib
        with tempfile.TemporaryDirectory() as d:
            rep = Path(d) / "run_report.txt"
            with contextlib.redirect_stdout(io.StringIO()):
                with tp.step_log(rep, "Step A"):
                    print("hello from step A")
                with self.assertRaises(ZeroDivisionError):
                    with tp.step_log(rep, "Step B"):
                        print("about to fail")
                        1 / 0
            text = rep.read_text(encoding="utf-8")
            self.assertIn("hello from step A", text)
            self.assertIn("Step A: completed", text)
            self.assertIn("STEP FAILED", text)
            self.assertIn("ZeroDivisionError", text)

    def test_tee_is_safe_after_report_closed(self):
        import contextlib, gc, warnings
        with tempfile.TemporaryDirectory() as d:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                log = tp.step_log(Path(d) / "r.txt", "S")
                with log:
                    print("x")
                tee = sys.stdout       # restored stream, not the tee
            leftover = tp._Tee(io.StringIO(), open(Path(d) / "c.txt", "w"))
            leftover.streams[1].close()
            leftover.write("ignored"); leftover.flush()   # must not raise on a closed file
            del leftover; gc.collect()

    def test_show_prints_dataframe(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tp.show(pd.DataFrame({"a": [1, 2]}))
        self.assertIn("a", buf.getvalue())

    def test_inventory_only_lists_given_roots(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "proj"
            (root / "sub").mkdir(parents=True)
            (root / "sub" / "f.txt").write_text("x")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "junk.pyc").write_text("x")
            text = tp.inventory_locations({"project": root, "missing": Path(d) / "nope"})
            self.assertIn("f.txt", text)
            self.assertNotIn("junk.pyc", text)
            self.assertIn("exists=False", text)

    def test_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.py").write_text("print(1)")
            (d / "SHA256SUMS.txt").write_text(f"{tp.file_sha256(d / 'a.py')}  a.py\n")
            self.assertEqual(tp.verify_release_checksums(d), [])
            (d / "a.py").write_text("print(2)")
            self.assertEqual(tp.verify_release_checksums(d), ["changed or corrupted: a.py"])
            (d / "a.py").unlink()
            self.assertEqual(tp.verify_release_checksums(d), ["missing: a.py"])

    def test_checksums_suffix_filter_ignores_notebook(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.py").write_text("x")
            (d / "n.ipynb").write_text("{}")
            (d / "SHA256SUMS.txt").write_text(
                f"{tp.file_sha256(d / 'a.py')}  a.py\n{tp.file_sha256(d / 'n.ipynb')}  n.ipynb\n")
            (d / "n.ipynb").write_text('{"executed": true}')       # notebooks change when run
            self.assertEqual(tp.verify_release_checksums(d, only_suffixes=(".py",)), [])
            self.assertEqual(tp.verify_release_checksums(d), ["changed or corrupted: n.ipynb"])

    def test_bundle_report(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "predictions").mkdir()
            (d / "predictions" / "p.csv").write_text("x")
            (d / "summary.json").write_text("{}")
            out = tp.bundle_report(d, d / "report.zip")
            names = zipfile.ZipFile(out).namelist()
            self.assertIn("summary.json", names)
            self.assertTrue(any(n.endswith("p.csv") for n in names))


class TestStorageBackupGate(unittest.TestCase):
    def test_free_space_for_user(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertGreater(tp.free_space_for_user(d), 0)
            self.assertGreater(tp.free_space_for_user(Path(d) / "not" / "yet"), 0)   # nearest existing parent

    def test_launch_environment_check(self):
        """Deterministic: Python caches tempfile.gettempdir() per process, so the temp folder
        is passed explicitly. The production check still uses the real gettempdir()."""
        import os as _os
        saved = {k: _os.environ.get(k) for k in tp.LAUNCH_VARS}
        try:
            for k in tp.LAUNCH_VARS:
                _os.environ[k] = "C:\\Users\\x\\AppData\\Local\\Temp"
            probs = tp.check_launch_environment("D:", tempdir="C:\\Users\\x\\AppData\\Local\\Temp")
            self.assertEqual(len(probs), len(tp.LAUNCH_VARS) + 1)          # every variable + temp folder
            for k in tp.LAUNCH_VARS:
                _os.environ[k] = "D:\\csci7090\\cache\\x"
            self.assertEqual(tp.check_launch_environment("D:", tempdir="D:\\csci7090\\cache\\tmp"), [])
            self.assertEqual(len(tp.check_launch_environment("D:", tempdir="C:\\Temp")), 1)  # temp folder alone
            _os.environ.pop("JUPYTER_CONFIG_DIR")
            self.assertEqual(len(tp.check_launch_environment("D:", tempdir="D:\\t")), 1)    # missing variable
            self.assertIn("JUPYTER_CONFIG_DIR", tp.check_launch_environment("D:", tempdir="D:\\t")[0])
        finally:
            for k, v in saved.items():
                if v is None:
                    _os.environ.pop(k, None)
                else:
                    _os.environ[k] = v

    def test_backup_roundtrip_verified(self):
        import shutil as _sh
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            m = tp.backup_check_create(d / "work" / "backup_check", "V_TEST", size_mb=1)
            self.assertEqual(tp.backup_check_create(d / "work" / "backup_check", "V_TEST", size_mb=1)["sha256"],
                             m["sha256"])                               # re-run keeps the same file
            dl = d / "downloads"; dl.mkdir()
            out = d / "verdict.json"
            self.assertFalse(tp.backup_check_verify(m, [dl], out)["verified"])        # nothing downloaded yet
            _sh.copy(m["file"], dl / "BACKUP_TEST_V_TEST (1).bin")                    # browser-renamed copy
            r = tp.backup_check_verify(m, [dl], out)
            self.assertTrue(r["verified"])
            self.assertTrue(out.is_file())

    def test_backup_corrupted_copy_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            m = tp.backup_check_create(d / "bc", "V_TEST", size_mb=1)
            dl = d / "downloads"; dl.mkdir()
            data = bytearray(Path(m["file"]).read_bytes()); data[100] ^= 0xFF
            (dl / "BACKUP_TEST_V_TEST.bin").write_bytes(bytes(data))
            self.assertFalse(tp.backup_check_verify(m, [dl], d / "v.json")["verified"])

    def test_original_file_is_not_its_own_proof(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            m = tp.backup_check_create(d / "bc", "V_TEST", size_mb=1)
            self.assertFalse(tp.backup_check_verify(m, [d / "bc"], d / "v.json")["verified"])

    def _prior_proof(self, d, tag="V3_4_5"):
        import shutil as _sh
        m = tp.backup_check_create(d / "backup_check", tag, size_mb=1)
        dl = d / "downloads"; dl.mkdir(exist_ok=True)
        _sh.copy(m["file"], dl / f"BACKUP_TEST_{tag} (1).bin")
        (d / "results" / tag).mkdir(parents=True, exist_ok=True)
        r = tp.backup_check_verify(m, [dl], d / "results" / tag / "backup_check_result.json")
        self.assertTrue(r["verified"])
        return m, r

    def test_backup_reuse_of_earlier_release(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._prior_proof(d)
            out = d / "results" / "V3_4_6" / "backup_check_result.json"
            out.parent.mkdir(parents=True)
            r = tp.backup_check_reuse(d / "backup_check", d / "results", out, "V3_4_6")
            self.assertTrue(r["verified"])
            self.assertEqual(r["reused_from_release"], "V3_4_5")
            self.assertTrue(out.is_file())

    def test_backup_reuse_rejected_if_recovered_copy_changed_or_missing(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            m, r = self._prior_proof(d)
            rec = Path(r["recovered_copy"])
            data = bytearray(rec.read_bytes()); data[0] ^= 0xFF; rec.write_bytes(bytes(data))
            self.assertFalse(tp.backup_check_reuse(d / "backup_check", d / "results", d / "o.json", "V3_4_6")["verified"])
            rec.unlink()
            self.assertFalse(tp.backup_check_reuse(d / "backup_check", d / "results", d / "o.json", "V3_4_6")["verified"])

    def test_backup_reuse_rejected_if_original_missing_or_no_prior(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "results").mkdir()
            self.assertFalse(tp.backup_check_reuse(d / "backup_check", d / "results", d / "o.json", "V3_4_6")["verified"])
            m, r = self._prior_proof(d)
            Path(m["file"]).unlink()
            self.assertFalse(tp.backup_check_reuse(d / "backup_check", d / "results", d / "o.json", "V3_4_6")["verified"])

    def test_training_gate(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                tp.training_gate(d)
            tp.write_gate(d, "real_data_tests", True)
            with self.assertRaises(RuntimeError):
                tp.training_gate(d)
            tp.write_gate(d, "backup_recovery", False)
            with self.assertRaises(RuntimeError):
                tp.training_gate(d)
            tp.write_gate(d, "backup_recovery", True)
            tp.training_gate(d)


class TestMisc(unittest.TestCase):
    def test_configure_environment_rerun_is_quiet(self):
        import contextlib, io, sys as _sys, types
        with tempfile.TemporaryDirectory() as d:
            had = "transformers" in _sys.modules
            _sys.modules.setdefault("transformers", types.ModuleType("transformers"))
            try:
                os_env_before = dict(__import__("os").environ)
                tp.configure_environment(d)                 # may warn (first call in this process)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    tp.configure_environment(d)             # re-run with same value: must be quiet
                self.assertEqual(buf.getvalue(), "")
            finally:
                if not had:
                    del _sys.modules["transformers"]
                __import__("os").environ.clear()
                __import__("os").environ.update(os_env_before)

    def test_storage_report_flags_d_drive_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            rep = tp.storage_report({"here": Path(d), "nested_new": Path(d) / "a" / "b"})
            self.assertEqual(len(rep), 2)
            self.assertIn("free_GB", rep.columns)

    def test_registry(self):
        with tempfile.TemporaryDirectory() as d:
            tp.register(Path(d), "x", status="partial", n_done=10)
            r = tp.register(Path(d), "x", status="complete")
            self.assertEqual(r["status"], "complete")
            self.assertEqual(r["n_done"], 10)

    def test_environment_report_without_torch_is_safe(self):
        rep = tp.environment_report()
        self.assertIn("READY_FOR_GPU_WORK", rep)


if __name__ == "__main__":
    unittest.main(verbosity=2)
