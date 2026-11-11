"""
t4_pipeline.py — recovery / evaluation / matched-pilot helpers (release: see __version__) for the
university Windows Azure Virtual Desktop (NVIDIA Tesla T4).

Design rules
------------
* Nothing in this module starts training or long generation on import.
* torch / transformers are imported lazily, so the CPU-only parts (manifest checks,
  checkpoint inventory, scoring of saved predictions) work without a GPU.
* Every long-running function prints a projected duration after its first unit of
  work and honours a hard `max_minutes` budget, writing partial results that can be
  resumed.
* Ground truth for scoring is always read from the annotation JSON in the pilot ZIP.

Test status (as delivered): the CPU-only functions in this file were executed in a
test environment with synthetic ZIP / safetensors / trainer_state fixtures
(see test_t4_pipeline_cpu.py). Functions that need torch/transformers/CUDA were
syntax-checked only — they have NOT been executed by the author and must be run on
the T4 machine, where each prints what it verified.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import platform
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import chart_eval as ce

__version__ = "3.4.6"

TASK_PROMPT = "Generate underlying data table of the figure below:"   # unchanged from V3.3.2
MODEL_CONFIGS = {"DePlot": "google/deplot", "MatCha": "google/matcha-base"}

# Historical V3.3.2 configuration (kept for the matched comparison).
V332 = dict(
    MAX_PATCHES=2048,
    MAX_TARGET_LENGTH=1152,
    GENERATION_MAX_LENGTH=1152,
    LEARNING_RATE=2e-5,
    PILOT_MAX_EPOCHS=100,          # defines the linear LR schedule length (6400 steps)
    TRAIN_BATCH_SIZE=2,
    GRADIENT_ACCUMULATION_STEPS=2,
    SEED=42,
)

MANIFESTS = {
    "train": "manifests/V3.3.2_pilot_train.csv",
    "validation": "manifests/V3.3.2_pilot_validation.csv",
    "extracted_diagnostic": "manifests/V3.3.2_extracted_diagnostic.csv",
}


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)   # atomic on Windows and POSIX
    return path


# ---------------------------------------------------------------------------
# 0. Environment configuration (call BEFORE importing transformers)
# ---------------------------------------------------------------------------

def configure_environment(hf_home: Optional[str] = None):
    """Point the Hugging Face cache at a disposable location (re-downloadable files
    only) and silence Windows symlink warnings. Must run before transformers import."""
    already_set = bool(hf_home) and os.environ.get("HF_HOME") == str(hf_home)
    if hf_home:
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if "transformers" in sys.modules and not already_set:
        print("WARNING: transformers was imported before configure_environment(); "
              "HF_HOME change will not apply until the kernel is restarted.")
    return {"HF_HOME": os.environ.get("HF_HOME")}


# ---------------------------------------------------------------------------
# 0b. Machine-generated run report (replaces screenshots)
# ---------------------------------------------------------------------------

V332_COMPLETED_EPOCHS = 2   # from the V3.3.2 notebook's saved Trainer table (epochs 1-2 logged; epoch 3 interrupted)


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            if not getattr(s, "closed", False):
                s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            if not getattr(s, "closed", False):
                s.flush()

    def close(self):   # never close the notebook's own streams or the report file from here
        pass


class step_log:
    """Context manager: everything printed inside the block, plus the full traceback
    if it fails, is appended to the run report file (and still shown in the notebook).

        with tp.step_log(REPORT_FILE, "Step 3 - manifests"):
            ...
    """

    def __init__(self, report_file, title):
        self.report_file = Path(report_file)
        self.title = title

    def __enter__(self):
        import traceback  # noqa: F401  (imported for __exit__)
        self.report_file.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.report_file, "a", encoding="utf-8")
        self.fh.write(f"\n{'=' * 100}\n{self.title}   [{_now()}]\n{'=' * 100}\n")
        self._out, self._err = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._out, self.fh)
        sys.stderr = _Tee(self._err, self.fh)
        return self

    def __exit__(self, exc_type, exc, tb):
        import traceback
        sys.stdout, sys.stderr = self._out, self._err
        if exc_type is not None:
            self.fh.write("\n!!! STEP FAILED !!!\n" + "".join(traceback.format_exception(exc_type, exc, tb)))
        else:
            self.fh.write(f"\n--- {self.title}: completed ---\n")
        self.fh.close()
        return False   # never swallow the error


def show(df, max_rows=60):
    """Print a DataFrame as plain text so it is captured by step_log (display() is not)."""
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", 50, "display.width", 250,
                           "display.max_colwidth", 120):
        print(df.to_string() if len(df) <= max_rows else df.head(max_rows).to_string() + f"\n... ({len(df)} rows)")


def inventory_locations(roots: Dict[str, Path], max_depth=4, max_entries=300, skip_dirs=("cache",)) -> str:
    """List files (with sizes) under ONLY the given project roots, depth-limited.
    Large or irrelevant folders (__pycache__, .ipynb_checkpoints, caches) are skipped."""
    skip = {"__pycache__", ".ipynb_checkpoints", ".git", *skip_dirs}
    lines = []
    for name, root in roots.items():
        root = Path(root)
        lines.append(f"[{name}] {root}  exists={root.exists()}")
        if not root.exists():
            continue
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            dirnames[:] = sorted(d for d in dirnames if d not in skip and depth < max_depth)
            for fn in sorted(filenames):
                fp = Path(dirpath) / fn
                try:
                    size = fp.stat().st_size
                except OSError:
                    size = -1
                lines.append(f"   {str(fp.relative_to(root)):90s} {size / 1024**2:10.2f} MB")
                count += 1
                if count >= max_entries:
                    lines.append(f"   ... truncated after {max_entries} files")
                    break
            if count >= max_entries:
                break
    return "\n".join(lines)


def verify_release_checksums(code_dir: Path, sums_file: str = "SHA256SUMS.txt",
                             only_suffixes: Optional[Sequence[str]] = None) -> List[str]:
    """Compare files listed in SHA256SUMS.txt with the copies in code_dir (optionally only
    those with the given suffixes - the notebook itself legitimately changes when run).
    Returns a list of problems (empty list = release intact)."""
    code_dir = Path(code_dir)
    listing = code_dir / sums_file
    if not listing.is_file():
        return [f"{sums_file} not found in {code_dir}"]
    problems = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        if only_suffixes and Path(name).suffix not in only_suffixes:
            continue
        f = code_dir / name
        if not f.is_file():
            problems.append(f"missing: {name}")
        elif file_sha256(f) != digest:
            problems.append(f"changed or corrupted: {name}")
    return problems


def bundle_report(results_dir: Path, out_zip: Path, max_mb: float = 20.0) -> Path:
    """Zip the small result/report files of this release (predictions, JSON, CSV, PNG,
    the run report) into one file to upload. Files above max_mb are listed, not included."""
    results_dir, out_zip = Path(results_dir), Path(out_zip)
    skipped = []
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(results_dir.rglob("*")):
            if not f.is_file() or f.resolve() == out_zip.resolve() or f.suffix == ".zip":
                continue
            if f.stat().st_size > max_mb * 1024**2:
                skipped.append(str(f.relative_to(results_dir)))
                continue
            z.write(f, f.relative_to(results_dir))
        if skipped:
            z.writestr("SKIPPED_LARGE_FILES.txt", "\n".join(skipped))
    return out_zip


# ---------------------------------------------------------------------------
# 0c. Storage limits, launch environment, backup/recovery check, training gate
# ---------------------------------------------------------------------------

def free_space_for_user(path) -> Optional[float]:
    """GB that THIS user can still write at `path`. On Windows this is
    GetDiskFreeSpaceExW's 'free bytes available to caller', which honours per-user
    disk quotas (shutil.disk_usage reports the whole volume's free space instead)."""
    path = Path(path)
    probe = path if path.exists() else next((q for q in path.parents if q.exists()), None)
    if probe is None:
        return None
    if os.name == "nt":
        import ctypes
        caller, total, free = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
        ok = ctypes.windll.kernel32.GetDiskFreeSpaceExW(str(probe), ctypes.byref(caller),
                                                        ctypes.byref(total), ctypes.byref(free))
        return round(caller.value / 1024**3, 2) if ok else None
    st = os.statvfs(probe)
    return round(st.f_bavail * st.f_frsize / 1024**3, 2)


LAUNCH_VARS = ("TEMP", "TMP", "HF_HOME", "MPLCONFIGDIR", "IPYTHONDIR", "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR")


def check_launch_environment(work_drive: str = "D:", tempdir: Optional[str] = None) -> List[str]:
    """Every cache/temp location must be on the work drive (set by START_JUPYTER_*.bat),
    so nothing is written into the restricted user profile.
    tempdir: Python's temp folder to check; default tempfile.gettempdir() (the real one in use).
    Tests pass it explicitly because Python caches gettempdir() for the whole process."""
    import tempfile
    problems = []
    for var in LAUNCH_VARS:
        val = os.environ.get(var, "")
        if not val.upper().startswith(work_drive.upper()):
            problems.append(f"{var}={val or '(not set)'} is not on {work_drive}")
    tmp = tempdir if tempdir is not None else tempfile.gettempdir()
    if not tmp.upper().startswith(work_drive.upper()):
        problems.append(f"Python temp folder {tmp} is not on {work_drive}")
    return problems


def backup_check_create(check_dir: Path, release_tag: str, size_mb: int = 5) -> dict:
    """Write a random test file plus its SHA-256. You upload it to Google Drive in the
    browser, download it back, and backup_check_verify() compares the copies."""
    check_dir = Path(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = check_dir / f"backup_check_{release_tag}.json"
    if manifest_path.is_file():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if Path(m["file"]).is_file():
            return m          # keep the same file across re-runs
    f = check_dir / f"BACKUP_TEST_{release_tag}.bin"
    f.write_bytes(os.urandom(size_mb * 1024 * 1024))
    m = {"file": str(f), "name": f.name, "sha256": file_sha256(f), "bytes": f.stat().st_size, "created": _now()}
    save_json(m, manifest_path)
    return m


def backup_check_verify(manifest: dict, search_dirs: Sequence[Path], out_json: Path) -> dict:
    """Look for the round-tripped copy (browser may add ' (1)') in search_dirs and
    compare its SHA-256 with the original. Writes the verdict to out_json."""
    stem, suffix = Path(manifest["name"]).stem, Path(manifest["name"]).suffix
    original = Path(manifest["file"]).resolve()
    candidates = []
    for d in search_dirs:
        d = Path(d)
        if d.is_dir():
            candidates += [p for p in d.glob(f"{stem}*{suffix}") if p.resolve() != original]
    result = {"checked": _now(), "expected_sha256": manifest["sha256"],
              "candidates": [str(c) for c in candidates], "verified": False}
    for c in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        if c.stat().st_size == manifest["bytes"] and file_sha256(c) == manifest["sha256"]:
            result.update(verified=True, recovered_copy=str(c))
            break
    save_json(result, out_json)
    return result


def backup_check_reuse(backup_dir: Path, results_root: Path, out_json: Path, current_tag: str) -> dict:
    """Reuse a round trip already verified under an EARLIER release, without weakening it:
    the earlier result must say verified, and BOTH files it compared - the original test file
    (from its manifest) and the copy downloaded back from Google Drive - must still exist and
    still hash to the recorded SHA-256 now. Otherwise nothing is reused."""
    backup_dir, results_root = Path(backup_dir), Path(results_root)
    manifests = {}
    for mf in backup_dir.glob("backup_check_*.json"):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            manifests[m["sha256"]] = m
        except Exception:  # noqa: BLE001
            continue
    tried = []
    for res_file in sorted(results_root.glob("*/backup_check_result.json"), reverse=True):
        tag = res_file.parent.name
        if tag == current_tag:
            continue
        try:
            prior = json.loads(res_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not prior.get("verified"):
            continue
        m = manifests.get(prior.get("expected_sha256"))
        original = Path(m["file"]) if m else None
        recovered = Path(prior.get("recovered_copy", ""))
        ok = (m is not None and original.is_file() and recovered.is_file()
              and original.resolve() != recovered.resolve()
              and file_sha256(original) == prior["expected_sha256"]
              and file_sha256(recovered) == prior["expected_sha256"])
        tried.append({"release": tag, "ok": bool(ok)})
        if ok:
            result = {"checked": _now(), "verified": True, "reused_from_release": tag,
                      "original_file": str(original), "recovered_copy": str(recovered),
                      "expected_sha256": prior["expected_sha256"],
                      "originally_verified": prior.get("checked"),
                      "method": "both files re-hashed now; both match the recorded SHA-256"}
            save_json(result, out_json)
            return result
    result = {"checked": _now(), "verified": False, "prior_proofs_examined": tried}
    return result


def write_gate(results_dir: Path, name: str, passed: bool, **info) -> Path:
    return save_json({"passed": bool(passed), "time": _now(), **info}, Path(results_dir) / f"gate_{name}.json")


def training_gate(results_dir: Path, required=("real_data_tests", "backup_recovery")) -> None:
    """Raise unless every required gate file exists and passed."""
    missing = []
    for name in required:
        f = Path(results_dir) / f"gate_{name}.json"
        if not f.is_file() or not json.loads(f.read_text(encoding="utf-8")).get("passed"):
            missing.append(name)
    if missing:
        raise RuntimeError(f"Training blocked: these checks have not passed yet: {missing}. "
                           "Run Step 2 (real-data tests) and Step 1b (backup check) first.")


# ---------------------------------------------------------------------------
# 1. Storage and environment checks
# ---------------------------------------------------------------------------

def storage_report(paths: Dict[str, Path], warn_drive_letters=("D",)) -> pd.DataFrame:
    """Free space per path (reflects per-user quota on FSLogix profile disks) and a
    warning for paths on temporary Azure storage (D: by default)."""
    rows = []
    for name, p in paths.items():
        p = Path(p)
        probe = p if p.exists() else next((q for q in p.parents if q.exists()), None)
        if probe is None:
            rows.append({"name": name, "path": str(p), "exists": False})
            continue
        usage = shutil.disk_usage(probe)
        drive = (p.drive or "").rstrip(":").upper()
        rows.append({
            "name": name, "path": str(p), "exists": p.exists(),
            "free_GB": round(usage.free / 1024**3, 1),
            "free_for_you_GB": free_space_for_user(p),
            "total_GB": round(usage.total / 1024**3, 1),
            "TEMPORARY_STORAGE_WARNING": drive in warn_drive_letters,
        })
    return pd.DataFrame(rows)


def environment_report(out_json: Optional[Path] = None) -> dict:
    """Verify PyTorch import, CUDA availability, GPU identity, FREE GPU memory and an
    actual CUDA tensor operation (checked against a CPU result)."""
    rep = {"time": _now(), "python": sys.version.split()[0], "platform": platform.platform(),
           "executable": sys.executable}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        rep["torch_import_error"] = repr(exc)
        rep["READY_FOR_GPU_WORK"] = False
        return rep
    rep["torch"] = torch.__version__
    rep["torch_cuda_build"] = torch.version.cuda
    rep["cuda_available"] = torch.cuda.is_available()
    for mod in ("transformers", "accelerate", "tokenizers", "safetensors"):
        try:
            rep[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001
            rep[mod] = f"NOT IMPORTABLE: {exc!r}"
    if not rep["cuda_available"]:
        rep["READY_FOR_GPU_WORK"] = False
        return rep

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    free_b, total_b = torch.cuda.mem_get_info(dev)
    rep.update({
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_total_GiB": round(total_b / 1024**3, 2),
        "gpu_free_GiB_now": round(free_b / 1024**3, 2),
        "gpu_used_by_others_GiB": round((total_b - free_b) / 1024**3, 2),
    })
    # Real CUDA work, compared with CPU.
    g = torch.Generator().manual_seed(0)
    a = torch.randn(512, 512, generator=g)
    b = torch.randn(512, 512, generator=g)
    ref = a @ b
    t0 = time.time()
    out32 = (a.to(dev) @ b.to(dev)).cpu()
    with torch.autocast("cuda", dtype=torch.float16):
        out16 = (a.to(dev) @ b.to(dev)).float().cpu()
    torch.cuda.synchronize()
    rep["cuda_matmul_fp32_max_abs_err"] = float((out32 - ref).abs().max())
    rep["cuda_matmul_fp16_max_rel_err"] = float(((out16 - ref).abs().max() / ref.abs().max()))
    rep["cuda_matmul_seconds"] = round(time.time() - t0, 3)
    rep["fp16_ok"] = rep["cuda_matmul_fp16_max_rel_err"] < 1e-2
    rep["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    rep["READY_FOR_GPU_WORK"] = bool(rep["cuda_matmul_fp32_max_abs_err"] < 1e-2 and rep["fp16_ok"])
    if out_json:
        save_json(rep, out_json)
    return rep


# ---------------------------------------------------------------------------
# 2. Manifests and data
# ---------------------------------------------------------------------------

def file_sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_and_verify_manifests(bundle_zip: Path, expected=None) -> Dict[str, pd.DataFrame]:
    """Load the three ORIGINAL pilot manifests from the bundle (read-only) and verify
    sizes, source counts, uniqueness, chart-ID disjointness and member presence."""
    expected = expected or {
        "train": {"generated": 250, "extracted": 6},
        "validation": {"generated": 62, "extracted": 2},
        "extracted_diagnostic": {"extracted": 66},
    }
    bundle_zip = Path(bundle_zip)
    if not bundle_zip.is_file():
        raise FileNotFoundError(f"Pilot bundle not found: {bundle_zip}")
    with zipfile.ZipFile(bundle_zip) as z:
        names = set(z.namelist())
        dfs = {k: pd.read_csv(z.open(v)) for k, v in MANIFESTS.items()}
    problems = []
    for k, df in dfs.items():
        counts = df["source"].value_counts().to_dict()
        if counts != expected[k]:
            problems.append(f"{k}: source counts {counts} != expected {expected[k]}")
        if not df["chart_id"].is_unique:
            problems.append(f"{k}: duplicate chart_id")
        for col in ("image_member", "annotation_member"):
            missing = set(df[col]) - names
            if missing:
                problems.append(f"{k}: {len(missing)} {col} entries missing from bundle")
    ids = {k: set(df["chart_id"]) for k, df in dfs.items()}
    keys = list(ids)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = ids[keys[i]] & ids[keys[j]]
            if overlap:
                problems.append(f"{keys[i]} / {keys[j]} overlap: {len(overlap)} chart IDs")
    if problems:
        raise AssertionError("Manifest verification FAILED:\n  " + "\n  ".join(problems))
    print("Manifests verified: sizes, source counts, uniqueness, disjointness, members present.")
    for k, df in dfs.items():
        print(f"  {k:22s} n={len(df):4d}  {df['source'].value_counts().to_dict()}")
    return dfs


def label_issue_report(bundle_zip: Path, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Charts whose category labels contain a newline or '|' (cannot be written
    verbatim as one 'x | y' row). These are the charts whose V3.3.2 targets were split."""
    rows = []
    with zipfile.ZipFile(bundle_zip) as z:
        for split, df in dfs.items():
            for cid, src, member in zip(df["chart_id"], df["source"], df["annotation_member"]):
                issues = ce.label_issues(z.read(member))
                if issues["n_newline_labels"] or issues["n_pipe_labels"]:
                    rows.append({"split": split, "chart_id": cid, "source": src, **issues})
    return pd.DataFrame(rows, columns=["split", "chart_id", "source", "n_labels", "n_newline_labels",
                                       "n_pipe_labels", "examples"])


def read_ground_truth(bundle_zip: Path, df: pd.DataFrame) -> Dict[str, tuple]:
    out = {}
    with zipfile.ZipFile(bundle_zip) as z:
        for cid, member in zip(df["chart_id"], df["annotation_member"]):
            out[cid] = ce.ground_truth_from_annotation(z.read(member))
    return out


def read_targets(bundle_zip: Path, df: pd.DataFrame, **target_kwargs) -> List[str]:
    with zipfile.ZipFile(bundle_zip) as z:
        return [ce.build_linearized_target(z.read(m), **target_kwargs) for m in df["annotation_member"]]


def load_rgb_array(zip_file: zipfile.ZipFile, member: str) -> np.ndarray:
    """Identical pixels to the V3.3.2 Colab loader (PIL RGB -> numpy), but writable."""
    from PIL import Image
    return np.array(Image.open(io.BytesIO(zip_file.read(member))).convert("RGB"))


# ---------------------------------------------------------------------------
# 3. Tokenizer / processor diagnostics (CPU; downloads ~3 MB of tokenizer files)
# ---------------------------------------------------------------------------

def tokenizer_diagnostic(processor, sample_targets: Sequence[str]) -> dict:
    """Decide whether the V3.3.2 training targets kept their row separators.

    Branch A ("NEWLINE_ENCODED"): '\\n' is encoded as the <0x0A> byte token -> the
      completed DePlot training used valid targets; only evaluation was broken.
    Branch B ("NEWLINE_LOST"): '\\n' is dropped / mapped to space / <unk> -> the
      completed DePlot training learned tables WITHOUT row separators.
    """
    tok = processor.tokenizer
    nl_id = tok.convert_tokens_to_ids("<0x0A>")
    unk_id = tok.unk_token_id
    probe = "Alpha | 1.5\nBeta | 2"
    ids = tok(probe, add_special_tokens=True)["input_ids"]
    pieces = tok.convert_ids_to_tokens(ids)
    literal_ids = tok("Alpha | 1.5<0x0A>Beta | 2", add_special_tokens=True)["input_ids"]

    rep = {
        "tokenizer_class": type(tok).__name__,
        "byte_newline_token_id": nl_id,
        "unk_token_id": unk_id,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "probe_pieces": pieces,
        "probe_decoded": tok.decode(ids, skip_special_tokens=True),
        "newline_encoded_as_byte_token": bool(nl_id is not None and nl_id != unk_id and nl_id in ids),
        "newline_became_unk": bool(unk_id is not None and unk_id in ids),
        "literal_0x0A_text_encodes_to_byte_token": bool(nl_id in literal_ids and
                                                         literal_ids.count(nl_id) == 1),
        "is_vqa": bool(getattr(processor.image_processor, "is_vqa", False)),
        "processor_max_patches": getattr(processor.image_processor, "max_patches", None),
    }
    # Round trip on REAL targets: encode -> decode -> exact parse. Multi-line labels make
    # this stricter than a row count: a newline INSIDE a label must also survive.
    rows_ok, exact_ok, changes = 0, 0, []
    for t in sample_targets:
        dec = tok.decode(tok(t, add_special_tokens=True)["input_ids"], skip_special_tokens=True)
        p_orig, p_dec = ce.parse_table(t), ce.parse_table(dec)
        if p_dec.n_rows == p_orig.n_rows:
            rows_ok += 1
            same_y = np.allclose(p_dec.y_array(), p_orig.y_array(), equal_nan=True)
            if p_dec.x == p_orig.x and same_y:
                exact_ok += 1
            elif len(changes) < 5:
                diff = [(a, b) for a, b in zip(p_orig.x, p_dec.x) if a != b][:2]
                changes.append({"label_changes": diff, "values_equal": bool(same_y)})
    n = len(sample_targets)
    rep["real_target_roundtrip_row_preserved"] = f"{rows_ok}/{n}"
    rep["real_target_roundtrip_exact"] = f"{exact_ok}/{n}"
    rep["roundtrip_examples_of_changed_text"] = changes
    # Branch depends ONLY on whether row structure survives tokenization (newline handling).
    # Exact-text changes with intact rows (e.g. a character the vocabulary cannot encode)
    # are reported separately: they cap both models equally because the vocabularies are shared.
    rep["BRANCH"] = ("A_NEWLINE_ENCODED" if rep["newline_encoded_as_byte_token"] and rows_ok == n
                     else "B_NEWLINE_LOST")
    return rep


def target_token_lengths(processor, targets: Sequence[str]) -> np.ndarray:
    tok = processor.tokenizer
    return np.array([len(tok(t, add_special_tokens=True, truncation=False)["input_ids"]) for t in targets])


def choose_generation_cap(train_lengths, eval_lengths=None, factor=1.5, ceiling=1152, multiple=32) -> dict:
    """Generation cap fixed in advance from TRAINING targets only, identical for every model:
    min(ceiling, roundup_multiple(factor x longest training target)).
    Evaluation-reference lengths NEVER change the cap; they are only reported (per split if a
    dict is given), so charts whose reference is longer than the cap are disclosed separately."""
    cap = int(min(ceiling, math.ceil(factor * int(np.max(train_lengths)) / multiple) * multiple))
    if eval_lengths is None:
        groups = {}
    elif isinstance(eval_lengths, dict):
        groups = eval_lengths
    else:
        groups = {"evaluation": eval_lengths}
    over = {name: {"n_refs": int(len(l)), "n_over_cap": int(np.sum(np.asarray(l) > cap)),
                   "longest_ref_tokens": int(np.max(l)) if len(l) else 0}
            for name, l in groups.items()}
    return {"train_max_tokens": int(np.max(train_lengths)), "max_new_tokens": cap,
            "eval_refs_over_cap": over,
            "rule": f"min({ceiling}, roundup{multiple}({factor} x longest training target)); "
                    "fixed before evaluation, same for all models; evaluation lengths only reported"}


# ---------------------------------------------------------------------------
# 4. Checkpoint inventory (CPU only; no torch needed)
# ---------------------------------------------------------------------------

INFERENCE_FILES = ("config.json", "model.safetensors")
RESUME_FILES = ("optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth")


def safetensors_integrity(path: Path) -> dict:
    """Parse the safetensors header and confirm the file is complete (detects a
    truncated browser download) without loading weights."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        if header_len > 100 * 1024 * 1024:
            return {"ok": False, "reason": "implausible header length"}
        header = json.loads(f.read(header_len))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    end = max((v["data_offsets"][1] for v in tensors.values()), default=0)
    expected = 8 + header_len + end
    return {"ok": expected == size, "n_tensors": len(tensors), "file_bytes": size,
            "expected_bytes": expected, "dtypes": sorted({v["dtype"] for v in tensors.values()})}


def inventory_checkpoints(run_dir: Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    rows = []
    ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    for ck in ckpts:
        row = {"checkpoint": ck.name, "path": str(ck)}
        for f in INFERENCE_FILES + RESUME_FILES + ("generation_config.json",):
            row[f] = (ck / f).is_file()
        row["inference_ready"] = all(row[f] for f in INFERENCE_FILES)
        row["resume_ready"] = row["inference_ready"] and all(row[f] for f in RESUME_FILES)
        if row["model.safetensors"]:
            integ = safetensors_integrity(ck / "model.safetensors")
            row["weights_complete"] = integ["ok"]
            row["n_tensors"] = integ.get("n_tensors")
        ts = ck / "trainer_state.json"
        if ts.is_file():
            st = json.loads(ts.read_text(encoding="utf-8"))
            row["global_step"] = st.get("global_step")
            row["epoch"] = st.get("epoch")
            # Only an evaluation logged AT this checkpoint's step belongs to it. A mid-epoch
            # checkpoint saved by the time limit has no evaluation of its own.
            evals = [h for h in st.get("log_history", [])
                     if "eval_loss" in h and h.get("step") == st.get("global_step")]
            row["eval_loss_at_ckpt"] = evals[-1].get("eval_loss") if evals else float("nan")
            row["eval_benetech_score_logged"] = evals[-1].get("eval_benetech_score") if evals else None
            row["best_model_checkpoint_logged"] = st.get("best_model_checkpoint")
        rows.append(row)
    if not rows:
        print(f"No checkpoint-* folders found in {run_dir}")
    return pd.DataFrame(rows)


def training_history(trainer_state_json: Path) -> pd.DataFrame:
    st = json.loads(Path(trainer_state_json).read_text(encoding="utf-8"))
    by_epoch = {}
    for h in st.get("log_history", []):
        ep = round(float(h.get("epoch", 0)), 3)
        by_epoch.setdefault(ep, {"epoch": ep})
        for k in ("loss", "eval_loss", "eval_benetech_score", "learning_rate", "eval_runtime", "step"):
            if k in h:
                by_epoch[ep][k] = h[k]
    return pd.DataFrame(sorted(by_epoch.values(), key=lambda r: r["epoch"]))


# ---------------------------------------------------------------------------
# 5. Model loading and batched, resumable generation (GPU)
# ---------------------------------------------------------------------------

def load_processor(model_name: str):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_name)


def load_model(path_or_name: str, device="cuda"):
    from transformers import Pix2StructForConditionalGeneration
    model = Pix2StructForConditionalGeneration.from_pretrained(str(path_or_name))
    model.to(device).eval()
    return model


def _encode_images(processor, images, max_patches):
    import torch
    feats = [processor(images=img, text=TASK_PROMPT, return_tensors="pt", max_patches=max_patches)
             for img in images]
    return (torch.cat([f["flattened_patches"] for f in feats]),
            torch.cat([f["attention_mask"] for f in feats]))


PRED_COLUMNS = ["chart_id", "source", "prediction_text", "n_generated_tokens", "hit_token_limit",
                "batch_seconds", "model", "max_new_tokens", "timestamp"]


def generate_predictions(model, processor, df: pd.DataFrame, bundle_zip: Path, out_csv: Path,
                         model_label: str, max_new_tokens: int, batch_size: int = 8,
                         max_patches: int = 2048, fp16: bool = True, max_minutes: float = 45.0,
                         confirm_after_first_batch: bool = False) -> dict:
    """Greedy generation for every chart in df, appended to out_csv after each batch.
    Re-running skips charts already in out_csv (resume). Stops cleanly at max_minutes."""
    import torch
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_csv.is_file():
        done = set(pd.read_csv(out_csv, usecols=["chart_id"])["chart_id"])
    todo = df[~df["chart_id"].isin(done)].reset_index(drop=True)
    print(f"[{model_label}] {len(done)} charts already done, {len(todo)} to generate "
          f"(max_new_tokens={max_new_tokens}, batch={batch_size}, fp16={fp16})")
    if todo.empty:
        return {"status": "complete", "n_done": len(done), "n_total": len(df)}

    eos = processor.tokenizer.eos_token_id
    t_start = time.time()
    write_header = not out_csv.is_file()
    i = 0
    first = True
    with zipfile.ZipFile(bundle_zip) as z, open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PRED_COLUMNS)
        if write_header:
            writer.writeheader()
        while i < len(todo):
            chunk = todo.iloc[i:i + batch_size]
            t0 = time.time()
            try:
                images = [load_rgb_array(z, m) for m in chunk["image_member"]]
                patches, mask = _encode_images(processor, images, max_patches)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                    gen = model.generate(flattened_patches=patches.to(model.device),
                                         attention_mask=mask.to(model.device),
                                         max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"  OOM -> retrying with batch_size={batch_size}")
                continue
            gen = gen.cpu()
            texts = processor.batch_decode(gen, skip_special_tokens=True)
            secs = time.time() - t0
            for (_, row), ids, text in zip(chunk.iterrows(), gen.tolist(), texts):
                writer.writerow({
                    "chart_id": row["chart_id"], "source": row["source"], "prediction_text": text,
                    "n_generated_tokens": int(sum(1 for t in ids[1:] if t != processor.tokenizer.pad_token_id)),
                    "hit_token_limit": bool(eos not in ids[1:]),
                    "batch_seconds": round(secs, 2), "model": model_label,
                    "max_new_tokens": max_new_tokens, "timestamp": _now(),
                })
            f.flush()
            i += len(chunk)
            if first:
                n_batches = math.ceil(len(todo) / batch_size)
                print(f"  first batch: {secs:.1f} s -> projected total ~{secs * n_batches / 60:.1f} min "
                      f"for {len(todo)} charts (upper bound if later batches stop earlier at EOS)")
                first = False
                if confirm_after_first_batch:
                    ans = input("  Continue? [y/N] ").strip().lower()
                    if ans != "y":
                        return {"status": "stopped_by_user", "n_done": len(done) + i, "n_total": len(df)}
            elapsed = (time.time() - t_start) / 60
            if elapsed > max_minutes and i < len(todo):
                print(f"  Time budget {max_minutes} min reached after {i}/{len(todo)} charts. "
                      "Re-run the same cell to resume.")
                return {"status": "partial", "n_done": len(done) + i, "n_total": len(df)}
    total = (time.time() - t_start) / 60
    print(f"  done in {total:.1f} min")
    return {"status": "complete", "n_done": len(df), "n_total": len(df), "minutes": round(total, 2)}


# ---------------------------------------------------------------------------
# 6. Scoring saved predictions (CPU)
# ---------------------------------------------------------------------------

def score_predictions(pred_csv: Path, df: pd.DataFrame, bundle_zip: Path, header_mode: str,
                      out_prefix: Optional[Path] = None) -> dict:
    """Score a predictions CSV against annotation ground truth. Charts in df that have
    no prediction are reported as missing (never silently scored)."""
    preds = pd.read_csv(pred_csv, keep_default_na=False).drop_duplicates("chart_id", keep="last")
    gt = read_ground_truth(bundle_zip, df)
    pred_map = dict(zip(preds["chart_id"], preds["prediction_text"]))
    limit_map = dict(zip(preds["chart_id"], preds.get("hit_token_limit", pd.Series(dtype=bool))))
    rows, results, sources = [], [], []
    missing = []
    for cid, src in zip(df["chart_id"], df["source"]):
        if cid not in pred_map:
            missing.append(cid)
            continue
        x_true, y_true = gt[cid]
        r = ce.evaluate_chart(x_true, y_true, pred_map[cid], header_mode=header_mode)
        results.append(r)
        sources.append(src)
        d = {"chart_id": cid, "source": src, **r.to_dict(),
             "hit_token_limit": str(limit_map.get(cid, "")) in ("True", "true", "1")}
        rows.append(d)
    per_chart = pd.DataFrame(rows)
    summary = ce.summarize(results, sources) if results else {"overall": {"n": 0}}
    summary["n_expected"] = int(len(df))
    summary["n_scored"] = int(len(results))
    summary["n_missing_predictions"] = int(len(missing))
    summary["complete"] = len(missing) == 0
    summary["header_mode"] = header_mode
    if not per_chart.empty:
        summary["overall"]["hit_token_limit_rate"] = float(per_chart["hit_token_limit"].mean())
    summary["scorer"] = f"chart_eval {ce.__version__}"
    if out_prefix is not None:
        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        per_chart.to_csv(f"{out_prefix}_per_chart.csv", index=False)
        save_json(summary, f"{out_prefix}_summary.json")
    return {"summary": summary, "per_chart": per_chart}


def show_examples(pred_csv: Path, df: pd.DataFrame, bundle_zip: Path, header_mode: str,
                  k: int = 3, max_chars: int = 400):
    """Print k raw prediction / reference pairs with their parsed row counts and score,
    so the evaluation can be checked by eye against actual model output."""
    preds = pd.read_csv(pred_csv, keep_default_na=False).drop_duplicates("chart_id", keep="last")
    sub = df[df["chart_id"].isin(preds["chart_id"])].head(k)
    targets = dict(zip(sub["chart_id"], read_targets(bundle_zip, sub)))
    gt = read_ground_truth(bundle_zip, sub)
    pmap = dict(zip(preds["chart_id"], preds["prediction_text"]))
    for cid in sub["chart_id"]:
        r = ce.evaluate_chart(*gt[cid], pmap[cid], header_mode=header_mode)
        print("=" * 100)
        print(f"chart {cid}  rows true={r.n_true} pred={r.n_pred}  strict={r.strict_score:.4f}  "
              f"[debug label_sim={r.aligned_x_similarity:.3f} value_sim={r.aligned_y_similarity:.3f}]")
        print("REFERENCE :", targets[cid].replace("\n", " ⏎ ")[:max_chars])
        print("PREDICTION:", str(pmap[cid])[:max_chars])


# ---------------------------------------------------------------------------
# 7. Training: dataset, collator, memory-fit test, matched trainer (GPU)
# ---------------------------------------------------------------------------

try:  # torch is optional for the CPU-only parts of this module
    import torch as _torch
    from torch.utils.data import Dataset as _Dataset
except Exception:  # noqa: BLE001
    _torch = None
    _Dataset = object


class ChartTableDataset(_Dataset):
    """Same inputs/labels as V3.3.2 BenetechPilotDataset, with an optional in-RAM cache
    of processed patches (processing is deterministic; no augmentation is used)."""

    def __init__(self, df, bundle_zip, processor, max_patches=2048, max_target_length=1152,
                 row_separator="\n", cache=True, sanitize_labels=False):
        self.df = df.reset_index(drop=True).copy()
        self.bundle_zip = Path(bundle_zip)
        self.processor = processor
        self.max_patches = max_patches
        self.max_target_length = max_target_length
        self.row_separator = row_separator
        self.sanitize_labels = sanitize_labels
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        row = self.df.iloc[idx]
        with zipfile.ZipFile(self.bundle_zip) as z:
            image = load_rgb_array(z, row["image_member"])
            target = ce.build_linearized_target(z.read(row["annotation_member"]),
                                                row_separator=self.row_separator,
                                                sanitize_labels=self.sanitize_labels)
        enc = self.processor(images=image, text=TASK_PROMPT, return_tensors="pt",
                             max_patches=self.max_patches)
        labels = self.processor.tokenizer(target, return_tensors="pt", padding=False, truncation=True,
                                          max_length=self.max_target_length)["input_ids"].squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        item = {"flattened_patches": enc["flattened_patches"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": labels}
        if self.cache is not None:
            self.cache[idx] = item
        return item


def chart_collator(features):
    import torch
    return {
        "flattened_patches": torch.stack([f["flattened_patches"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.nn.utils.rnn.pad_sequence([f["labels"] for f in features],
                                                  batch_first=True, padding_value=-100),
    }


def memory_fit_test(model_name, processor, train_df, bundle_zip, batch_size=2,
                    gradient_checkpointing=False, max_patches=2048, max_target_length=1152) -> dict:
    """One fwd+bwd+AdamW step on the LONGEST pilot targets (worst case), on the GPU
    memory actually free now. Mirrors V3.3.2 cell 11. ~1-3 minutes."""
    import gc
    import torch
    from transformers import Pix2StructForConditionalGeneration
    targets = read_targets(bundle_zip, train_df)
    lengths = target_token_lengths(processor, targets)
    worst = train_df.iloc[np.argsort(-lengths)[:batch_size]]
    ds = ChartTableDataset(worst, bundle_zip, processor, max_patches, max_target_length, cache=False)
    batch = chart_collator([ds[i] for i in range(len(ds))])
    free_before, total = torch.cuda.mem_get_info()
    model = optimizer = None
    try:
        model = Pix2StructForConditionalGeneration.from_pretrained(model_name).to("cuda")
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.train()
        batch = {k: v.to("cuda") for k, v in batch.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=V332["LEARNING_RATE"])
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        free_after, _ = torch.cuda.mem_get_info()
        res = {"passed": True, "batch_size": batch_size, "gradient_checkpointing": gradient_checkpointing,
               "worst_case_target_tokens": int(lengths.max()),
               "free_GiB_before": round(free_before / 1024**3, 2), "peak_allocated_GiB": round(peak, 2),
               "peak_reserved_GiB": round(reserved, 2),
               "free_GiB_at_peak_end": round(free_after / 1024**3, 2),
               "loss_is_finite": bool(torch.isfinite(loss).item())}
        res["headroom_GiB"] = round(res["free_GiB_before"] - reserved, 2)
        if res["headroom_GiB"] < 0.75:
            res["warning"] = "less than 0.75 GiB headroom; other GPU users could cause OOM mid-run"
        return res
    except torch.OutOfMemoryError:
        return {"passed": False, "batch_size": batch_size, "gradient_checkpointing": gradient_checkpointing,
                "free_GiB_before": round(free_before / 1024**3, 2), "reason": "CUDA out of memory"}
    finally:
        del model, optimizer, batch
        gc.collect()
        torch.cuda.empty_cache()


def _make_callbacks():
    from transformers import TrainerCallback

    class StopAfterEpochs(TrainerCallback):
        """Stop after N epochs while keeping the V3.3.2 100-epoch LR schedule, so the
        learning-rate trajectory is identical to the completed DePlot run."""

        def __init__(self, n_epochs):
            self.n_epochs = n_epochs

        def on_epoch_end(self, args, state, control, **kwargs):
            if state.epoch is not None and state.epoch >= self.n_epochs - 1e-6:
                control.should_training_stop = True
            return control

    class TimeLimit(TrainerCallback):
        """Hard wall-clock budget: save and stop at the first step past the limit."""

        def __init__(self, max_minutes):
            self.max_seconds = max_minutes * 60
            self.t0 = None
            self.triggered = False

        def on_train_begin(self, args, state, control, **kwargs):
            self.t0 = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if self.t0 and time.time() - self.t0 > self.max_seconds and not self.triggered:
                self.triggered = True
                print(f"\n[TimeLimit] {self.max_seconds / 60:.0f} min reached at step "
                      f"{state.global_step}; saving checkpoint and stopping.")
                control.should_save = True
                control.should_training_stop = True
            return control

    class EpochTimer(TrainerCallback):
        def __init__(self):
            self.t = None
            self.epoch_minutes = []

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.t = time.time()

        def on_epoch_end(self, args, state, control, **kwargs):
            mins = (time.time() - self.t) / 60
            self.epoch_minutes.append(round(mins, 2))
            print(f"\n[EpochTimer] epoch {state.epoch:.0f} training time {mins:.1f} min")

    return StopAfterEpochs, TimeLimit, EpochTimer


def _filtered_args(cls, required, optional):
    import inspect
    params = inspect.signature(cls.__init__).parameters
    missing = [k for k in required if k not in params]
    if missing:
        raise RuntimeError(f"{cls.__name__} in this transformers version lacks required args: {missing}")
    dropped = [k for k in optional if k not in params]
    if dropped:
        print(f"Note: optional TrainingArguments not supported here and skipped: {dropped}")
    kw = dict(required)
    kw.update({k: v for k, v in optional.items() if k in params})
    return cls(**kw)


def build_matched_trainer(model_name, processor, train_df, val_df, bundle_zip, output_dir,
                          n_epochs=2, max_minutes=60, row_separator="\n", sanitize_labels=False,
                          gradient_checkpointing=False, save_optimizer_state=True,
                          batch_size=V332["TRAIN_BATCH_SIZE"],
                          grad_accum=V332["GRADIENT_ACCUMULATION_STEPS"]):
    """Seq2SeqTrainer with the V3.3.2 optimisation settings, but:
       * validation = teacher-forced eval_loss only (no generation inside training);
       * checkpoint choice is made afterwards from eval_loss (no Benetech-in-the-loop);
       * stops after n_epochs (same LR schedule as V3.3.2) or at max_minutes."""
    from transformers import (Pix2StructForConditionalGeneration, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments, set_seed)
    set_seed(V332["SEED"])
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    train_ds = ChartTableDataset(train_df, bundle_zip, processor, V332["MAX_PATCHES"],
                                 V332["MAX_TARGET_LENGTH"], row_separator=row_separator,
                                 sanitize_labels=sanitize_labels)
    val_ds = ChartTableDataset(val_df, bundle_zip, processor, V332["MAX_PATCHES"],
                               V332["MAX_TARGET_LENGTH"], row_separator=row_separator,
                               sanitize_labels=sanitize_labels)
    required = dict(
        output_dir=str(output_dir),
        num_train_epochs=V332["PILOT_MAX_EPOCHS"],
        learning_rate=V332["LEARNING_RATE"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        predict_with_generate=False,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        seed=V332["SEED"],
    )
    optional = dict(dataloader_num_workers=0,          # Windows + Jupyter: workers can't pickle notebook objects
                    save_only_model=not save_optimizer_state,
                    save_total_limit=None)
    args = _filtered_args(Seq2SeqTrainingArguments, required, optional)
    StopAfterEpochs, TimeLimit, EpochTimer = _make_callbacks()
    timer = EpochTimer()
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                             data_collator=chart_collator,
                             callbacks=[StopAfterEpochs(n_epochs), TimeLimit(max_minutes), timer])
    steps_per_epoch = math.ceil(len(train_df) / (batch_size * grad_accum))
    print(f"Matched trainer ready: {model_name}, {n_epochs} epoch(s) x {steps_per_epoch} optimizer steps, "
          f"LR schedule length {V332['PILOT_MAX_EPOCHS'] * steps_per_epoch} steps (as V3.3.2), "
          f"time limit {max_minutes} min, sanitize_labels={sanitize_labels}, output {output_dir}")
    return trainer, timer


def select_checkpoint_by_eval_loss(run_dir: Path) -> dict:
    inv = inventory_checkpoints(run_dir)
    if inv.empty or "eval_loss_at_ckpt" not in inv.columns:
        raise RuntimeError(f"No checkpoint with a trainer_state.json eval_loss in {run_dir}")
    inv = inv[inv["inference_ready"].astype(bool) & inv["eval_loss_at_ckpt"].notna()]
    if inv.empty:
        raise RuntimeError(f"No inference-ready checkpoint with a logged eval_loss in {run_dir}")
    best = inv.sort_values(["eval_loss_at_ckpt", "global_step"]).iloc[0]
    return {"checkpoint": best["checkpoint"], "path": best["path"], "epoch": best.get("epoch"),
            "eval_loss": float(best["eval_loss_at_ckpt"]),
            "rule": "lowest teacher-forced validation loss on the 64-chart primary validation set"}


# ---------------------------------------------------------------------------
# 8. Experiment registry (what was actually run)
# ---------------------------------------------------------------------------

def register(results_dir: Path, name: str, **info):
    """Record an experiment's status. Only call AFTER the thing has actually happened."""
    path = Path(results_dir) / "experiment_registry.json"
    reg = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    reg[name] = {**reg.get(name, {}), **info, "updated": _now()}
    save_json(reg, path)
    return reg[name]


def results_table(results_dir: Path) -> pd.DataFrame:
    """Build the comparison table ONLY from summary files that exist on disk."""
    rows = []
    for p in sorted(Path(results_dir).glob("*_summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        o = s.get("overall", {})
        rows.append({
            "experiment": p.name.replace("_summary.json", ""),
            "complete": s.get("complete"), "n_scored": s.get("n_scored"), "n_expected": s.get("n_expected"),
            "strict_benetech": o.get("strict_benetech_score"),
            "parse_success": o.get("parse_success_rate"),
            "row_count_acc": o.get("row_count_accuracy"),
            "dbg_label_sim": o.get("debug_aligned_label_similarity"),
            "dbg_value_sim": o.get("debug_aligned_value_similarity"),
            "dbg_median_APE": o.get("debug_median_abs_pct_error"),
            "hit_limit_rate": o.get("hit_token_limit_rate"),
            "generated": s.get("generated", {}).get("strict_benetech_score"),
            "extracted": s.get("extracted", {}).get("strict_benetech_score"),
            "n_extracted": s.get("extracted", {}).get("n"),
        })
    return pd.DataFrame(rows)
