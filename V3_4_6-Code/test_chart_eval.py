"""
Automated tests for chart_eval.py (stdlib unittest; no pytest/torch needed).

Run:   python -m unittest -v test_chart_eval
       (from the folder containing chart_eval.py)

Optional data tests run automatically when the pilot bundle is found via the
environment variable CSCI7090_BUNDLE (path to V3.3.2_pilot_bundle.zip).
"""

import json
import math
import os
import unittest
import zipfile
from pathlib import Path

import numpy as np

import chart_eval as ce

HERE = Path(__file__).resolve().parent

# Real DePlot output copied verbatim from the V3.3.2 notebook smoke test (cell 8),
# chart 8b1bca06d728, max_new_tokens=128.
REAL_DEPLOT_SMOKE_OUTPUT = (
    "TITLE | Sex ratio by age (100 year olds) (UNWPP, 2017) in Central African Republic"
    "<0x0A>Years | Sex ratio<0x0A>1990 | 29.10<0x0A>1991 | 25.30<0x0A>1992 | 32.10"
    "<0x0A>1993 | 30.10<0x0A>1994 | 31.10<0x0A>1995 | 26<0x0A>1996 | 34.10<0x0A>1997 | 30.10"
)

# Real MatCha-base output copied verbatim from the V3.3.2 notebook smoke test (cell 10).
REAL_MATCHA_SMOKE_OUTPUT = (
    "# Set the figure size<0x0A>plt.figure(figsize=(6,6))<0x0A><0x0A># Generate a bar plot of "
    "sex ratio by year<0x0A>sns.barplot(x='Years', y='Sex Ratio', data=df, palette='GnBu_d', "
    "color='grey')<0x0A><0x0A># Set the yticks for the x axis<0x0A>plt.yticks(np.arange(0, 41, "
    "step=2.0))<0x0A><0x0A># Set the title<0x0A>plt.title('Sex ratio by age (100 year olds)"
)

# Real target printed by the V3.3.2 notebook (cell 37, extracted chart).
REAL_EXTRACTED_TARGET = """Buprestidae | 4.442912028371576
Curculionidae | 53.50211801792925
Ixodidae | 66.66831194471865
Glossinidae | 68.88943731490622
Formicidae | 76.17647058823529
Culicidae | 82.12121212121212
Cerambycidae | 85.15151515151516
Lymantriinae | 88.18181818181819
Chrysomelidae | 91.93548387096774
Tephritidae | 93.2258064516129"""


def as_byte_token_form(text):
    """What batch_decode returns when '\\n' was encoded as the <0x0A> byte token."""
    return text.replace("\n", "<0x0A>")


class TestOriginalDefectReproduction(unittest.TestCase):
    """Proves, with the ORIGINAL V3.3.2 code, why every Benetech score was 0."""

    @classmethod
    def setUpClass(cls):
        import _v332_metric_original as old
        cls.old = old

    def test_old_parser_reads_zero_rows_from_real_deplot_output(self):
        x, y = self.old.parse_table_text(REAL_DEPLOT_SMOKE_OUTPUT)
        self.assertEqual(len(x), 0)

    def test_old_metric_scores_perfect_byte_token_prediction_as_zero(self):
        ref = as_byte_token_form(REAL_EXTRACTED_TARGET)
        self.assertEqual(self.old.benetech_vertical_score(ref, ref), 0.0)

    def test_old_metric_scores_perfect_prediction_zero_if_newlines_lost(self):
        ref = REAL_EXTRACTED_TARGET.replace("\n", " ")
        self.assertEqual(self.old.benetech_vertical_score(ref, ref), 0.0)

    def test_old_metric_works_only_with_real_newlines(self):
        self.assertEqual(self.old.benetech_vertical_score(REAL_EXTRACTED_TARGET, REAL_EXTRACTED_TARGET), 1.0)


class TestNormalization(unittest.TestCase):
    def test_byte_newline(self):
        self.assertEqual(ce.normalize_table_text("a | 1<0x0A>b | 2"), "a | 1\nb | 2")

    def test_byte_newline_with_spaces(self):
        # Spaces around <0x0A> belong to the neighbouring cells; values are space-tolerant
        # and V3.3.2 labels are kept exactly, so the second label keeps its leading space.
        t = ce.parse_table("a | 1 <0x0A> b | 2")
        self.assertEqual(t.x, ["a", " b"])
        self.assertEqual(t.y, [1.0, 2.0])

    def test_crlf_and_literal_backslash_n(self):
        self.assertEqual(ce.parse_table("a | 1\r\nb | 2\\nc | 3").x, ["a", "b", "c"])

    def test_multibyte_utf8_byte_tokens(self):
        # en dash U+2013 = E2 80 93
        t = ce.parse_table("1990<0xE2><0x80><0x93>2000 | 5")
        self.assertEqual(t.x, ["1990–2000"])

    def test_none_and_nan(self):
        self.assertEqual(ce.parse_table(None).n_rows, 0)
        self.assertEqual(ce.parse_table(float("nan")).n_rows, 0)

    def test_blank_lines_between_rows_belong_to_next_label_trailing_ignored(self):
        t = ce.parse_table("a | 1<0x0A><0x0A>b | 2<0x0A>")
        self.assertEqual(t.n_rows, 2)
        self.assertEqual(t.x, ["a", "\nb"])      # V3.3.2 format: every line break belongs to a label


class TestParsing(unittest.TestCase):
    def test_real_deplot_output_native_mode(self):
        t = ce.parse_table(REAL_DEPLOT_SMOKE_OUTPUT, header_mode="deplot_native")
        self.assertTrue(t.dropped_title)
        self.assertEqual(t.dropped_header, "Years | Sex ratio")
        self.assertEqual(t.x, [str(y) for y in range(1990, 1998)])
        self.assertEqual(t.y, [29.10, 25.30, 32.10, 30.10, 31.10, 26.0, 34.10, 30.10])
        self.assertEqual(t.n_unparseable_values, 0)

    def test_real_deplot_output_none_mode_keeps_rows_visible(self):
        # In 'none' mode the TITLE and header lines are NOT silently dropped:
        # they are counted as rows with unparseable values, so the row count is honest.
        t = ce.parse_table(REAL_DEPLOT_SMOKE_OUTPUT, header_mode="none")
        self.assertEqual(t.n_rows, 10)
        self.assertEqual(t.n_unparseable_values, 2)

    def test_real_matcha_code_output_is_not_a_table(self):
        t = ce.parse_table(REAL_MATCHA_SMOKE_OUTPUT, header_mode="none")
        self.assertEqual(t.n_rows, 0)
        self.assertGreater(t.n_malformed_lines, 0)

    def test_real_target_roundtrip(self):
        t = ce.parse_table(as_byte_token_form(REAL_EXTRACTED_TARGET))
        self.assertEqual(t.n_rows, 10)
        self.assertEqual(t.x[0], "Buprestidae")
        self.assertAlmostEqual(t.y[0], 4.442912028371576, places=12)

    def test_numeric_header_in_native_mode_is_dropped(self):
        t = ce.parse_table("TITLE | GDP<0x0A>Country | 2019<0x0A>A | 1<0x0A>B | 2", "deplot_native")
        self.assertEqual(t.x, ["A", "B"])

    def test_native_mode_without_title(self):
        t = ce.parse_table("Year | Value<0x0A>2001 | 3", "deplot_native")
        self.assertEqual((t.x, t.y), (["2001"], [3.0]))

    def test_number_formats(self):
        for cell, expected in [("1,234", 1234.0), ("45%", 45.0), ("$12.5", 12.5),
                               ("−3.2", -3.2), (" 7 ", 7.0), ("1e3", 1000.0)]:
            self.assertEqual(ce.parse_number(cell), expected, cell)
        for cell in ["abc", "", "nan", "inf", "12 apples"]:
            self.assertTrue(math.isnan(ce.parse_number(cell)), cell)

    def test_unparseable_value_row_is_kept(self):
        t = ce.parse_table("a | 1<0x0A>b | n/a<0x0A>c | 3")
        self.assertEqual(t.n_rows, 3)
        self.assertEqual(t.n_unparseable_values, 1)

    def test_native_multi_column_uses_first_value(self):
        t = ce.parse_table("Year | A | B<0x0A>a | 1 | 9<0x0A>b | 2 | 8", "deplot_native")
        self.assertEqual(t.y, [1.0, 2.0])
        self.assertEqual(t.n_multi_column_rows, 2)

    def test_native_markdown_edge_pipes(self):
        t = ce.parse_table("| k | v |<0x0A>| a | 1 |<0x0A>| b | 2 |", "deplot_native")
        self.assertEqual((t.x, t.y), (["a", "b"], [1.0, 2.0]))

    def test_v332_value_is_after_last_separator(self):
        # In the V3.3.2 format a row has exactly one value, so '|' inside a label survives.
        t = ce.parse_table("A | B | 4<0x0A>c | 5")
        self.assertEqual((t.x, t.y), (["A | B", "c"], [4.0, 5.0]))
        self.assertEqual(t.n_multi_column_rows, 1)

    def test_pipeless_line_joins_next_label(self):
        # V3.3.2 wrote a multi-line label as pipe-less line(s) before ' | value'.
        t = ce.parse_table("a | 1<0x0A>West Bank<0x0A>and Gaza | 2")
        self.assertEqual(t.x, ["a", "West Bank\nand Gaza"])
        self.assertEqual(t.n_continuation_lines, 1)
        self.assertEqual(t.n_malformed_lines, 0)

    def test_trailing_pipeless_line_is_malformed_not_row(self):
        t = ce.parse_table("a | 1<0x0A>b | 2<0x0A>garbage")
        self.assertEqual(t.n_rows, 2)
        self.assertEqual(t.n_malformed_lines, 1)

    def test_loose_separator_fallback(self):
        t = ce.parse_table("a|1<0x0A>b |2")
        self.assertEqual((t.x, t.y), (["a", "b"], [1.0, 2.0]))
        self.assertEqual(t.n_loose_separator_rows, 2)

    def test_label_whitespace_preserved(self):
        labels = ["  padded  ", " lead", "trail ", "in  side"]
        ann = {"data-series": [{"x": l, "y": i} for i, l in enumerate(labels)]}
        for form in (ce.build_linearized_target(ann), as_byte_token_form(ce.build_linearized_target(ann))):
            self.assertEqual(ce.parse_table(form).x, labels)

    def test_bad_header_mode(self):
        with self.assertRaises(ValueError):
            ce.parse_table("a | 1", header_mode="auto")


class TestGroundTruth(unittest.TestCase):
    ANN = {"chart-type": "vertical_bar", "source": "generated",
           "data-series": [{"x": 1990, "y": 29.1}, {"x": "Beta", "y": 3}, {"x": "C", "y": "4.5"}]}

    def test_from_dict_and_json_and_bytes(self):
        for ann in (self.ANN, json.dumps(self.ANN), json.dumps(self.ANN).encode()):
            x, y = ce.ground_truth_from_annotation(ann)
            self.assertEqual(x, ["1990", "Beta", "C"])
            np.testing.assert_allclose(y, [29.1, 3.0, 4.5])

    def test_target_matches_v332_format(self):
        self.assertEqual(ce.build_linearized_target(self.ANN), "1990 | 29.1\nBeta | 3\nC | 4.5")

    def test_target_alternative_separator_and_rounding(self):
        t = ce.build_linearized_target({"data-series": [{"x": "a", "y": 4.442912028371576}]},
                                       row_separator="<0x0A>", y_format=lambda v: f"{v:.4g}")
        self.assertEqual(t, "a | 4.443")

    def test_ground_truth_equals_parse_of_target(self):
        x, y = ce.ground_truth_from_annotation(self.ANN)
        t = ce.parse_table(as_byte_token_form(ce.build_linearized_target(self.ANN)))
        self.assertEqual(t.x, x)
        np.testing.assert_allclose(t.y, y)


class TestNewlineLabels(unittest.TestCase):
    """Real Benetech labels can contain newlines (found in the real bundle on 13 Oct)."""
    ANN = {"data-series": [{"x": "Vanuatu", "y": 1}, {"x": "Venezuela,\nRB", "y": 2},
                           {"x": "Vietnam", "y": 2.5}, {"x": "West Bank\nand Gaza", "y": 3},
                           {"x": "Yemen\nRep.", "y": 3.5}, {"x": "Zambia", "y": 4}]}
    LABELS = ["Vanuatu", "Venezuela,\nRB", "Vietnam", "West Bank\nand Gaza", "Yemen\nRep.", "Zambia"]

    def test_old_v332_parser_split_these_labels(self):
        import _v332_metric_original as old
        x, _ = old.parse_table_text(ce.build_linearized_target(self.ANN))
        self.assertEqual(x, ["Vanuatu", "RB", "Vietnam", "and Gaza", "Rep.", "Zambia"])  # the user's failure

    def test_exact_reconstruction_real_newlines(self):
        t = ce.parse_table(ce.build_linearized_target(self.ANN))
        self.assertEqual(t.x, self.LABELS)
        self.assertEqual(t.y, [1.0, 2.0, 2.5, 3.0, 3.5, 4.0])

    def test_exact_reconstruction_byte_tokens(self):
        t = ce.parse_table(as_byte_token_form(ce.build_linearized_target(self.ANN)))
        self.assertEqual(t.x, self.LABELS)

    def test_v332_format_prediction_scores_perfect(self):
        x, y = ce.ground_truth_from_annotation(self.ANN)
        pred = as_byte_token_form(ce.build_linearized_target(self.ANN))
        self.assertEqual(ce.evaluate_chart(x, y, pred).strict_score, 1.0)

    def test_space_instead_of_newline_is_scored_exactly_not_normalized(self):
        x, y = ce.ground_truth_from_annotation(self.ANN)
        pred = "<0x0A>".join(f"{l.replace(chr(10), ' ')} | {v}" for l, v in zip(self.LABELS, y))
        r = ce.evaluate_chart(x, y, pred)
        self.assertTrue(r.row_count_match)
        self.assertLess(r.x_score, 1.0)          # exact strings: 3 substituted characters
        self.assertGreater(r.x_score, 0.9)

    def test_label_issues(self):
        i = ce.label_issues(self.ANN)
        self.assertEqual((i["n_newline_labels"], i["n_pipe_labels"]), (3, 0))

    def test_consecutive_newlines_regression_real_case(self):
        # Regression for the VM failure on train/annotations/0aa21c0d8302.json:
        # labels such as "Week\n\n1" came back as "Week\n1" in V3.4.5.
        labels = ["Week\n\n1", "Week\n\n2", "Week\n  \n3", "A\n\n\nB", "\nLeading", "Trailing\n", "plain"]
        ann = {"data-series": [{"x": l, "y": i + 0.5} for i, l in enumerate(labels)]}
        target = ce.build_linearized_target(ann)
        for form in (target, as_byte_token_form(target)):
            t = ce.parse_table(form)
            self.assertEqual(t.x, labels)
            self.assertEqual(t.y, [i + 0.5 for i in range(len(labels))])
        x, y = ce.ground_truth_from_annotation(ann)
        self.assertEqual(ce.evaluate_chart(x, y, as_byte_token_form(target)).strict_score, 1.0)

    def test_randomized_exact_reversibility(self):
        """3,000 random tables: every label the documented rule calls reversible must
        round-trip exactly in both newline forms."""
        import random
        rng = random.Random(7)
        alphabet = ["a", "B", "7", " ", "\n", "\n", "|", ".", ","]
        checked = 0
        for _ in range(3000):
            labels = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8))) for _ in range(rng.randint(1, 6))]
            if not all(ce.v332_label_reversibility(l)[0] for l in labels):
                continue
            ann = {"data-series": [{"x": l, "y": float(i)} for i, l in enumerate(labels)]}
            target = ce.build_linearized_target(ann)
            for form in (target, as_byte_token_form(target)):
                self.assertEqual(ce.parse_table(form).x, labels, repr(labels))
            checked += 1
        self.assertGreater(checked, 500)

    def test_irreversible_labels_are_identified(self):
        self.assertEqual(ce.v332_label_reversibility("A | B\nC")[0], False)   # pipe on a non-final line
        self.assertEqual(ce.v332_label_reversibility("A\r\nB")[0], False)
        self.assertEqual(ce.v332_label_reversibility("x<0x0A>y")[0], False)
        self.assertEqual(ce.v332_label_reversibility("Week\n\n1")[0], True)

    def test_model_blank_line_between_rows_costs_one_label_edit(self):
        x, y = ["a", "b"], [1.0, 2.0]
        r = ce.evaluate_chart(x, y, "a | 1<0x0A><0x0A>b | 2")
        self.assertTrue(r.row_count_match)
        self.assertLess(r.x_score, 1.0)
        self.assertEqual(ce.parse_table("a | 1<0x0A><0x0A>b | 2").n_blank_label_lines, 1)
        self.assertEqual(ce.parse_table("a | 1<0x0A>b | 2<0x0A><0x0A>").x, ["a", "b"])   # trailing blanks ignored

    def test_branch_b_separator_also_encodes_label_newlines(self):
        t = ce.build_linearized_target(self.ANN, row_separator="<0x0A>")
        self.assertNotIn("\n", t)                      # nothing a newline-dropping tokenizer can erase
        self.assertEqual(ce.parse_table(t).x, self.LABELS)


class TestMetric(unittest.TestCase):
    X = ["January", "February", "March"]
    Y = [42.0, 31.0, 50.0]

    def score(self, text):
        return ce.evaluate_chart(self.X, self.Y, text).strict_score

    def test_perfect(self):
        self.assertEqual(self.score("January | 42<0x0A>February | 31<0x0A>March | 50"), 1.0)

    def test_perfect_with_real_newlines(self):
        self.assertEqual(self.score("January | 42\nFebruary | 31\nMarch | 50"), 1.0)

    def test_small_label_error_matches_v332_value(self):
        # V3.3.2 notebook reported 0.9875026035157897 for this exact case.
        s = self.score("January | 42<0x0A>Febuary | 31<0x0A>March | 50")
        self.assertAlmostEqual(s, 0.9875026035157897, places=12)

    def test_small_numeric_error_matches_v332_value(self):
        # V3.3.2 notebook reported 0.9679469798444224 for this exact case.
        s = self.score("January | 41<0x0A>February | 32<0x0A>March | 49")
        self.assertAlmostEqual(s, 0.9679469798444224, places=12)

    def test_missing_row_is_zero_but_diagnostics_nonzero(self):
        r = ce.evaluate_chart(self.X, self.Y, "January | 42<0x0A>February | 31")
        self.assertEqual(r.strict_score, 0.0)
        self.assertFalse(r.row_count_match)
        self.assertTrue(r.parse_ok)
        self.assertEqual(r.aligned_n, 2)
        self.assertEqual(r.aligned_exact_label_rate, 1.0)

    def test_extra_row_is_zero(self):
        self.assertEqual(self.score("January | 42<0x0A>February | 31<0x0A>March | 50<0x0A>April | 1"), 0.0)

    def test_malformed_output_is_zero(self):
        r = ce.evaluate_chart(self.X, self.Y, REAL_MATCHA_SMOKE_OUTPUT)
        self.assertEqual(r.strict_score, 0.0)
        self.assertFalse(r.parse_ok)

    def test_unparseable_value_zeroes_numeric_half_only(self):
        r = ce.evaluate_chart(self.X, self.Y, "January | 42<0x0A>February | ?<0x0A>March | 50")
        self.assertEqual(r.y_score, 0.0)
        self.assertEqual(r.x_score, 1.0)
        self.assertEqual(r.strict_score, 0.5)

    def test_label_order_matters(self):
        s = self.score("March | 50<0x0A>February | 31<0x0A>January | 42")
        self.assertLess(s, 0.8)

    def test_numeric_formula_equals_v332_for_nonconstant_series(self):
        import _v332_metric_original as old
        rng = np.random.default_rng(0)
        for _ in range(500):
            n = int(rng.integers(2, 30))
            yt = rng.normal(50, 20, n)
            yp = yt + rng.normal(0, rng.uniform(0.1, 40), n)
            self.assertAlmostEqual(ce.score_numeric_series(yt, yp), old.numerical_similarity(yt, yp), places=10)

    def test_text_formula_equals_v332(self):
        import _v332_metric_original as old
        pairs = [(["abc", "de"], ["abd", "de"]), (["x"], ["yyyy"]), (["Mon", "Tue"], ["", "Tue"])]
        for t, p in pairs:
            self.assertAlmostEqual(ce.score_text_series(t, p), old.categorical_similarity(t, p), places=12)

    def test_constant_series_follows_r2_convention(self):
        self.assertEqual(ce.score_numeric_series([5, 5, 5], [5, 5, 5]), 1.0)
        self.assertAlmostEqual(ce.score_numeric_series([5, 5, 5], [5, 6, 5]), ce.sigmoid2(1.0))

    def test_huge_error_does_not_overflow(self):
        self.assertEqual(ce.score_numeric_series([1, 2, 3], [1e300, -1e300, 0]), 0.0)

    def test_sigmoid_endpoints(self):
        self.assertEqual(ce.sigmoid2(0.0), 1.0)
        self.assertAlmostEqual(ce.sigmoid2(1.0), 0.5378828427399902)
        self.assertEqual(ce.sigmoid2(float("inf")), 0.0)

    def test_levenshtein(self):
        self.assertEqual(ce.levenshtein("kitten", "sitting"), 3)
        self.assertEqual(ce.levenshtein("", "abc"), 3)


class TestAggregation(unittest.TestCase):
    def test_summary_and_sources(self):
        rs = [ce.evaluate_chart(["a"], [1], "a | 1"), ce.evaluate_chart(["a", "b"], [1, 2], "a | 1")]
        s = ce.summarize(rs, ["generated", "extracted"])
        self.assertEqual(s["overall"]["n"], 2)
        self.assertEqual(s["overall"]["strict_benetech_score"], 0.5)
        self.assertEqual(s["overall"]["row_count_accuracy"], 0.5)
        self.assertEqual(s["generated"]["strict_benetech_score"], 1.0)
        self.assertEqual(s["extracted"]["strict_benetech_score"], 0.0)

    def test_paired_bootstrap(self):
        a = np.full(64, 0.6)
        b = np.full(64, 0.4)
        res = ce.paired_bootstrap_difference(a, b, n_boot=2000)
        self.assertAlmostEqual(res["mean_difference"], 0.2)
        self.assertAlmostEqual(res["ci95_low"], 0.2)
        self.assertEqual(res["fraction_boot_a_better"], 1.0)
        with self.assertRaises(ValueError):
            ce.paired_bootstrap_difference([1, 2], [1])


@unittest.skipUnless(os.environ.get("CSCI7090_BUNDLE") and Path(os.environ["CSCI7090_BUNDLE"]).is_file(),
                     "set CSCI7090_BUNDLE=<path to V3.3.2_pilot_bundle.zip> to run data tests")
class TestRealBundle(unittest.TestCase):
    """Runs on the student's machine against the real pilot bundle."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.zip_path = Path(os.environ["CSCI7090_BUNDLE"])
        with zipfile.ZipFile(cls.zip_path) as z:
            cls.names = set(z.namelist())
            cls.train = pd.read_csv(z.open("manifests/V3.3.2_pilot_train.csv"))
            cls.val = pd.read_csv(z.open("manifests/V3.3.2_pilot_validation.csv"))
            cls.diag = pd.read_csv(z.open("manifests/V3.3.2_extracted_diagnostic.csv"))

    def test_split_sizes_and_sources(self):
        self.assertEqual(len(self.train), 256)
        self.assertEqual(len(self.val), 64)
        self.assertEqual(len(self.diag), 66)
        self.assertEqual(self.train["source"].value_counts().to_dict(), {"generated": 250, "extracted": 6})
        self.assertEqual(self.val["source"].value_counts().to_dict(), {"generated": 62, "extracted": 2})
        self.assertEqual(set(self.diag["source"]), {"extracted"})

    def test_chart_id_disjointness(self):
        t, v, d = (set(df["chart_id"]) for df in (self.train, self.val, self.diag))
        self.assertTrue(self.train["chart_id"].is_unique and self.val["chart_id"].is_unique)
        self.assertFalse(t & v)
        self.assertFalse(t & d)
        self.assertFalse(v & d)

    def test_all_members_present(self):
        for df in (self.train, self.val, self.diag):
            for col in ("image_member", "annotation_member"):
                missing = set(df[col]) - self.names
                self.assertFalse(missing, f"{len(missing)} missing {col}")

    def test_every_target_roundtrips_through_parser(self):
        """UNCHANGED from V3.4.0: every V3.3.2 target, with real newlines and with
        <0x0A> tokens, must reproduce the annotation labels and values EXACTLY."""
        with zipfile.ZipFile(self.zip_path) as z:
            for df in (self.train, self.val, self.diag):
                for member in df["annotation_member"]:
                    ann = json.loads(z.read(member))
                    x, y = ce.ground_truth_from_annotation(ann)
                    target = ce.build_linearized_target(ann)
                    for form in (target, as_byte_token_form(target)):
                        t = ce.parse_table(form)
                        self.assertEqual(t.x, x, member)
                        np.testing.assert_allclose(t.y, y, err_msg=member)

    def test_labels_with_separator_characters_roundtrip_exactly(self):
        """Replaces test_no_label_contains_separator_characters, whose premise (no label
        contains a newline or '|') is false for this dataset. Stronger check: every such
        label must be reconstructed character-for-character. Prints the inventory."""
        found = []
        with zipfile.ZipFile(self.zip_path) as z:
            for split, df in (("train", self.train), ("validation", self.val), ("diagnostic", self.diag)):
                for member in df["annotation_member"]:
                    ann = json.loads(z.read(member))
                    issues = ce.label_issues(ann)
                    if not (issues["n_newline_labels"] or issues["n_pipe_labels"]):
                        continue
                    found.append((split, member, issues["n_newline_labels"], issues["n_pipe_labels"]))
                    x, _ = ce.ground_truth_from_annotation(ann)
                    for form in (ce.build_linearized_target(ann),
                                 as_byte_token_form(ce.build_linearized_target(ann))):
                        self.assertEqual(ce.parse_table(form).x, x, member)
        print("\n[label inventory] charts with newline or '|' in a label (all reconstructed exactly):")
        for split, member, n_nl, n_pipe in found:
            print(f"   {split:10s} {member}  newline-labels={n_nl} pipe-labels={n_pipe}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
