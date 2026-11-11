"""
chart_eval.py — table normalization, parsing and Benetech-style scoring (release: see __version__)
CSCI 7090 — Automated Extraction of Structured Data from Vertical Bar Charts

Pure Python + NumPy (no torch / transformers), so it can be tested on any CPU.

What changed relative to the V3.3.2 notebook (Section 17, cell `parse_table_text` /
`benetech_vertical_score`):

1. Row separators. Pix2Struct-family decoders (DePlot, MatCha) emit the newline as the
   SentencePiece byte token "<0x0A>", and `batch_decode(..., skip_special_tokens=True)`
   returns it as the literal 6-character string "<0x0A>". V3.3.2 split rows with
   `str.splitlines()` only, so every decoded table (predictions AND references decoded
   from the label ids) parsed to 0 rows, which forced every chart score to 0.
   `normalize_table_text()` converts "<0x0A>", real newlines, and the literal two-char
   sequence "\\n" into one canonical newline before parsing.

2. Ground truth comes from the annotation JSON, not from decoding tokenized labels.
   Decoding labels is a lossy round trip (tokenizer normalization, truncation at
   MAX_TARGET_LENGTH, possible <unk>). `ground_truth_from_annotation()` reads the
   Benetech `data-series` directly.

3. Rows are never silently dropped. A row whose value cannot be parsed is kept with
   y = NaN (counted as a predicted row, scored conservatively) and reported in
   diagnostics. V3.3.2 silently skipped such rows, which changed the row count.

4. Header handling is explicit (`header_mode`):
     "none"          — the fine-tuning target format used in V3.3.2 (`x | y` rows only).
     "deplot_native" — zero-shot DePlot format: optional "TITLE | ..." line, then a
                       header row, then data rows.
   No automatic header guessing is used for the strict score.

5. Numerical similarity matches the published Benetech formula
   sigmoid(sqrt(1 - R^2)) with sklearn's r2_score conventions, including the constant
   ground-truth case (R^2 := 1 if the prediction is exact, else 0). V3.3.2 returned 0.0
   for an imperfect prediction of a constant series instead of sigmoid(1) ~= 0.538.
   For every non-constant series the two implementations are numerically identical
   (tested).

The STRICT score is unchanged in spirit: a length mismatch between the predicted and
true series scores 0 for that series, as in the competition rule. Separate debugging
measures (parse success, row-count accuracy, aligned-prefix similarities, numerical
error) are reported alongside and are NEVER mixed into the strict score.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__version__ = "3.4.6"

# ---------------------------------------------------------------------------
# 1. Normalization and parsing
# ---------------------------------------------------------------------------

_BYTE_NEWLINE = "<0x0A>"
# Other SentencePiece byte-fallback tokens occasionally seen in Pix2Struct output.
_BYTE_TOKEN_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")
_BYTE_RUN_RE = re.compile(r"(?:<0x[0-9A-Fa-f]{2}>)+")

HEADER_MODES = ("none", "deplot_native")


def normalize_table_text(text) -> str:
    """Return a table string that uses '\n' as its only row separator.

    Handles: literal "<0x0A>" byte tokens (with or without surrounding spaces),
    other "<0xNN>" byte tokens (decoded when printable), Windows/old-Mac line
    endings, and a literal backslash-n sequence. Pipes and cell text are untouched.
    """
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):  # pandas NaN from CSV
        return ""
    s = str(text)

    def _bytes(m):
        # A run of consecutive byte tokens is one UTF-8 sequence (e.g. an en dash is
        # <0xE2><0x80><0x93>); decode the run as a whole.
        raw = bytes(int(h, 16) for h in _BYTE_TOKEN_RE.findall(m.group(0)))
        decoded = raw.decode("utf-8", errors="replace").replace("\t", " ")
        return "".join(ch for ch in decoded if ch == "\n" or ch.isprintable())

    s = _BYTE_RUN_RE.sub(_bytes, s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\n", "\n")
    return s


_NUMBER_STRIP_RE = re.compile(r"[\s,$€£¥%]")


def parse_number(cell: str) -> float:
    """Parse a numeric cell; returns NaN when it is not a finite number.

    Accepts thousands separators, currency signs, percent signs, a unicode
    minus sign, and surrounding whitespace. Does NOT guess units such as 'k' or 'M'.
    """
    if cell is None:
        return float("nan")
    s = str(cell).strip().replace("−", "-").replace("–", "-")
    s = _NUMBER_STRIP_RE.sub("", s)
    if not s:
        return float("nan")
    try:
        value = float(s)
    except ValueError:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


@dataclass
class ParsedTable:
    x: List[str] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    n_lines: int = 0
    n_malformed_lines: int = 0      # non-empty lines with no '|' after the last row
    n_continuation_lines: int = 0   # pipe-less text lines joined into the next row's label (multi-line labels)
    n_blank_label_lines: int = 0    # blank lines that are part of a label (e.g. "Week\n\n1")
    n_unparseable_values: int = 0   # rows kept with y = NaN
    n_multi_column_rows: int = 0    # deplot_native: rows with >2 cells; none: labels containing '|'
    n_loose_separator_rows: int = 0 # none mode: rows using '|' without the ' | ' spacing
    dropped_title: bool = False
    dropped_header: Optional[str] = None

    @property
    def n_rows(self) -> int:
        return len(self.x)

    def y_array(self) -> np.ndarray:
        return np.asarray(self.y, dtype=float)


def _parse_v332_rows(text, result: ParsedTable) -> ParsedTable:
    """Exact inverse of the V3.3.2 serializer  rows = f"{x} | {y}", joined by "\n".

    Format facts this relies on:
    * The serializer never writes a blank line between rows. Therefore EVERY line break in
      a target belongs to a label, and every line between two rows - including blank and
      whitespace-only lines - is part of the next row's label. "Week\n\n1 | 5" is the
      label "Week\n\n1"; "a | 1\n\nRB | 2" gives the label "\nRB".
    * The value is the text after the LAST " | " (values never contain '|'), so a label
      containing '|' on its final line is recovered intact.
    * Lines are not stripped, so label whitespace is preserved exactly.

    Inherent limits (labels for which the V3.3.2 target text is ambiguous; see
    v332_label_reversibility): '|' on a non-final line of a label, a carriage return, or
    literal text that normalize_table_text() rewrites ("<0xNN>" or a backslash-n pair).

    For MODEL OUTPUT the same rule applies: a blank line a model emits between rows is,
    in this format, the start of the next label, so it costs one edit in the label score
    (counted in n_blank_label_lines). Blank lines after the last row are ignored.
    Model output that uses '|' without the spaces falls back to splitting on the last '|'
    with the label stripped (counted in n_loose_separator_rows).
    """
    pending: List[str] = []
    for line in normalize_table_text(text).split("\n"):
        if line.strip():
            result.n_lines += 1
        if "|" not in line:
            pending.append(line)          # part of the next label (blank or not)
            continue
        if " | " in line:
            label, value = line.rsplit(" | ", 1)
        else:
            label, value = line.rsplit("|", 1)
            label = label.strip()
            result.n_loose_separator_rows += 1
        if "|" in label:
            result.n_multi_column_rows += 1
        if pending:
            result.n_continuation_lines += sum(1 for p in pending if p.strip())
            result.n_blank_label_lines += sum(1 for p in pending if not p.strip())
            label = "\n".join(pending + [label])
            pending = []
        y_value = parse_number(value)
        if math.isnan(y_value):
            result.n_unparseable_values += 1
        result.x.append(label)
        result.y.append(y_value)
    result.n_malformed_lines += sum(1 for p in pending if p.strip())   # text after the last row
    return result


def v332_label_reversibility(label) -> Tuple[bool, str]:
    """Can this label be recovered exactly from a V3.3.2 target? (bool, reason)."""
    s = str(label)
    parts = s.split("\n")
    if any("|" in p for p in parts[:-1]):
        return False, "'|' on a non-final line of the label"
    if "\r" in s:
        return False, "carriage return in the label"
    if _BYTE_TOKEN_RE.search(s) or "\\n" in s:
        return False, "label contains '<0xNN>' or a backslash-n text sequence"
    return True, ""


def parse_table(text, header_mode: str = "none") -> ParsedTable:
    """Parse a linearized table into category labels and values.

    header_mode:
      "none"          the V3.3.2 fine-tuning target format ('x | y' rows, no header);
                      parsed with the exact inverse of the V3.3.2 serializer, which
                      reconstructs multi-line and '|'-containing labels verbatim.
      "deplot_native" zero-shot DePlot output: drop a leading 'TITLE | ...' line, treat
                      the first remaining '|' line as the column header, take the first
                      value column. Cells are whitespace-stripped.
    """
    if header_mode not in HEADER_MODES:
        raise ValueError(f"header_mode must be one of {HEADER_MODES}, got {header_mode!r}")

    result = ParsedTable()
    if header_mode == "none":
        return _parse_v332_rows(text, result)

    lines = [ln.strip() for ln in normalize_table_text(text).split("\n")]
    lines = [ln for ln in lines if ln]
    result.n_lines = len(lines)

    pipe_lines = []
    for ln in lines:
        if "|" in ln:
            pipe_lines.append(ln)
        else:
            result.n_malformed_lines += 1

    if pipe_lines and pipe_lines[0].split("|", 1)[0].strip().upper() == "TITLE":
        pipe_lines = pipe_lines[1:]
        result.dropped_title = True
    if pipe_lines:
        result.dropped_header = pipe_lines[0]
        pipe_lines = pipe_lines[1:]

    for ln in pipe_lines:
        cells = [c.strip() for c in ln.split("|")]
        # Leading/trailing pipes (markdown style "| a | b |") produce empty edge cells.
        if len(cells) > 2 and cells[0] == "":
            cells = cells[1:]
        if len(cells) > 2 and cells[-1] == "":
            cells = cells[:-1]
        if len(cells) > 2:
            result.n_multi_column_rows += 1
        x_cell = cells[0]
        y_cell = cells[1] if len(cells) > 1 else ""
        y_value = parse_number(y_cell)
        if math.isnan(y_value):
            result.n_unparseable_values += 1
        result.x.append(x_cell)
        result.y.append(y_value)

    return result


# ---------------------------------------------------------------------------
# 2. Ground truth
# ---------------------------------------------------------------------------

def canonical_label(label) -> str:
    """Collapse every whitespace run (including newlines inside a label, e.g. the
    Benetech tick label "West Bank\\nand Gaza") to one space and strip the ends.

    Used only by build_linearized_target(sanitize_labels=True), an optional FUTURE
    target format. Scoring compares exact annotation strings; the V3.3.2 format is
    lossless with the exact parser (_parse_v332_rows), so no normalization is needed."""
    return " ".join(str(label).split())


def label_issues(annotation) -> dict:
    """Labels that cannot be written verbatim into an 'x | y' row."""
    if isinstance(annotation, (bytes, bytearray)):
        annotation = annotation.decode("utf-8")
    if isinstance(annotation, str):
        annotation = json.loads(annotation)
    labels = [str(p.get("x", "")) for p in annotation.get("data-series", [])]
    newline = [l for l in labels if "\n" in l or "\r" in l]
    pipe = [l for l in labels if "|" in l]
    return {"n_labels": len(labels), "n_newline_labels": len(newline), "n_pipe_labels": len(pipe),
            "examples": (newline + pipe)[:3]}


def ground_truth_from_annotation(annotation) -> Tuple[List[str], np.ndarray]:
    """Return (x_labels, y_values) from a Benetech annotation dict or JSON string/bytes.

    Uses the same str() conversion as build_linearized_target() so labels match the
    training target text exactly (e.g. int 1990 -> "1990").
    """
    if isinstance(annotation, (bytes, bytearray)):
        annotation = annotation.decode("utf-8")
    if isinstance(annotation, str):
        annotation = json.loads(annotation)
    xs, ys = [], []
    for point in annotation.get("data-series", []):
        xs.append(f"{point.get('x', '')}")
        try:
            ys.append(float(point.get("y", float("nan"))))
        except (TypeError, ValueError):
            ys.append(float("nan"))
    return xs, np.asarray(ys, dtype=float)


def build_linearized_target(annotation, row_separator: str = "\n", y_format=None,
                            sanitize_labels: bool = False) -> str:
    """Same target text as V3.3.2 `build_linearized_target` when called with defaults.

    sanitize_labels: False (default, used in every V3.4.x release) reproduces V3.3.2 exactly.
                   A multi-line label is written across lines; parse_table() reverses
                   this exactly. True (future option only) writes canonical_label(x) and
                   replaces '|' with '/'.

    row_separator: "\n" reproduces V3.3.2. Use "<0x0A>" only if the tokenizer
                   diagnostic shows "\n" is not encoded as the <0x0A> byte token.
    y_format:      None reproduces V3.3.2 (Python str of the stored value). A callable
                   (e.g. lambda v: f"{v:.4g}") can be used in FUTURE runs to shorten
                   targets; changing it changes the experimental protocol.
    """
    if isinstance(annotation, (bytes, bytearray)):
        annotation = annotation.decode("utf-8")
    if isinstance(annotation, str):
        annotation = json.loads(annotation)
    rows = []
    for point in annotation.get("data-series", []):
        x_value = point.get("x", "")
        y_value = point.get("y", "")
        if sanitize_labels:
            x_value = canonical_label(x_value).replace("|", "/")
        elif row_separator != "\n":
            # A newline inside a label must use the same encoding as a row break, otherwise
            # a tokenizer that drops raw "\n" (Branch B) would erase it. parse_table() rejoins it.
            x_value = str(x_value).replace("\n", row_separator)
        if y_format is not None and isinstance(y_value, (int, float)):
            y_value = y_format(y_value)
        rows.append(f"{x_value} | {y_value}")
    return row_separator.join(rows)


# ---------------------------------------------------------------------------
# 3. Benetech metric (published formula)
# ---------------------------------------------------------------------------

def sigmoid2(x: float) -> float:
    """Benetech transform: 2 - 2/(1+exp(-x)); 1 at x=0, -> 0 as x -> inf."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if x > 700:  # avoid overflow warnings; value is 0 to machine precision
        return 0.0
    return 2.0 - 2.0 / (1.0 + math.exp(-x))


def levenshtein(a: str, b: str) -> int:
    a, b = str(a), str(b)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[j - 1] + 1,
                               previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def normalized_levenshtein(true_values: Sequence[str], pred_values: Sequence[str]) -> float:
    """sum(lev(t,p)) / sum(len(t)). Caller guarantees equal, non-zero length."""
    total_len = sum(len(str(t)) for t in true_values)
    total_dist = sum(levenshtein(t, p) for t, p in zip(true_values, pred_values))
    if total_len == 0:
        return 0.0 if total_dist == 0 else float("inf")
    return total_dist / total_len


def r2_score_sklearn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sklearn.metrics.r2_score (force_finite=True) for 1-D input, without sklearn."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if math.isnan(ss_res):
        return float("-inf")
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def normalized_rmse(y_true, y_pred) -> float:
    """sqrt(max(0, 1 - R^2)) == RMSE / RMSE(predict-the-mean) for non-constant y_true."""
    return math.sqrt(max(0.0, 1.0 - r2_score_sklearn(y_true, y_pred)))


def score_numeric_series(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return 0.0
    if not np.all(np.isfinite(y_pred)):  # unparseable predicted value -> conservative 0
        return 0.0
    return sigmoid2(normalized_rmse(y_true, y_pred))


def score_text_series(x_true, x_pred) -> float:
    if len(x_true) == 0 or len(x_true) != len(x_pred):
        return 0.0
    return sigmoid2(normalized_levenshtein(x_true, x_pred))


def benetech_strict_score(x_true, y_true, x_pred, y_pred) -> float:
    """Chart score = mean(x-series score, y-series score); each series is 0 on a
    length mismatch (competition rule). Chart type is fixed (vertical_bar) here."""
    return 0.5 * (score_text_series(x_true, x_pred) + score_numeric_series(y_true, y_pred))


# ---------------------------------------------------------------------------
# 4. Per-chart evaluation with diagnostics
# ---------------------------------------------------------------------------

@dataclass
class ChartResult:
    strict_score: float
    x_score: float
    y_score: float
    parse_ok: bool                 # >= 1 data row parsed from the prediction
    n_true: int
    n_pred: int
    row_count_match: bool
    # Debug-only measures over the aligned prefix min(n_true, n_pred). NOT the task score.
    aligned_n: int
    aligned_x_similarity: float
    aligned_y_similarity: float
    aligned_exact_label_rate: float
    aligned_mean_abs_pct_error: float
    n_malformed_lines: int
    n_unparseable_values: int
    n_multi_column_rows: int
    dropped_header: Optional[str]

    def to_dict(self):
        return asdict(self)


def evaluate_chart(x_true: Sequence[str], y_true, prediction_text, header_mode: str = "none") -> ChartResult:
    parsed = parse_table(prediction_text, header_mode=header_mode)
    y_true = np.asarray(y_true, dtype=float)
    x_true = [str(x) for x in x_true]                 # exact annotation strings (no normalization)
    x_pred, y_pred = parsed.x, parsed.y_array()

    x_score = score_text_series(x_true, x_pred)
    y_score = score_numeric_series(y_true, y_pred)
    strict = 0.5 * (x_score + y_score)

    k = min(len(x_true), len(x_pred))
    if k > 0:
        xt, xp = list(x_true)[:k], list(x_pred)[:k]
        yt, yp = y_true[:k], y_pred[:k]
        a_x = sigmoid2(normalized_levenshtein(xt, xp))
        a_y = sigmoid2(normalized_rmse(yt, yp)) if np.all(np.isfinite(yp)) else 0.0
        exact = float(np.mean([str(a) == str(b) for a, b in zip(xt, xp)]))
        finite = np.isfinite(yp) & (yt != 0)
        mape = float(np.mean(np.abs((yp[finite] - yt[finite]) / yt[finite]))) if finite.any() else float("nan")
    else:
        a_x = a_y = exact = 0.0
        mape = float("nan")

    return ChartResult(
        strict_score=float(strict), x_score=float(x_score), y_score=float(y_score),
        parse_ok=parsed.n_rows > 0, n_true=len(x_true), n_pred=parsed.n_rows,
        row_count_match=len(x_true) == parsed.n_rows,
        aligned_n=k, aligned_x_similarity=float(a_x), aligned_y_similarity=float(a_y),
        aligned_exact_label_rate=float(exact), aligned_mean_abs_pct_error=mape,
        n_malformed_lines=parsed.n_malformed_lines,
        n_unparseable_values=parsed.n_unparseable_values,
        n_multi_column_rows=parsed.n_multi_column_rows,
        dropped_header=parsed.dropped_header,
    )


def summarize(results: Iterable[ChartResult], sources: Optional[Sequence[str]] = None) -> dict:
    """Aggregate chart results; optionally per source. Returns plain floats/ints."""
    results = list(results)

    def _agg(rs):
        if not rs:
            return {"n": 0}
        mape = [r.aligned_mean_abs_pct_error for r in rs if not math.isnan(r.aligned_mean_abs_pct_error)]
        return {
            "n": len(rs),
            "strict_benetech_score": float(np.mean([r.strict_score for r in rs])),
            "parse_success_rate": float(np.mean([r.parse_ok for r in rs])),
            "row_count_accuracy": float(np.mean([r.row_count_match for r in rs])),
            "debug_aligned_label_similarity": float(np.mean([r.aligned_x_similarity for r in rs])),
            "debug_aligned_value_similarity": float(np.mean([r.aligned_y_similarity for r in rs])),
            "debug_aligned_exact_label_rate": float(np.mean([r.aligned_exact_label_rate for r in rs])),
            "debug_median_abs_pct_error": float(np.median(mape)) if mape else float("nan"),
            "mean_rows_true": float(np.mean([r.n_true for r in rs])),
            "mean_rows_pred": float(np.mean([r.n_pred for r in rs])),
        }

    out = {"overall": _agg(results)}
    if sources is not None:
        if len(sources) != len(results):
            raise ValueError("sources and results differ in length")
        for name in sorted(set(sources)):
            out[name] = _agg([r for r, s in zip(results, sources) if s == name])
    return out


def paired_bootstrap_difference(scores_a, scores_b, n_boot: int = 10000, seed: int = 42):
    """Mean(a - b) with a percentile 95% CI from a paired bootstrap over charts."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("paired bootstrap needs two equal-length 1-D score arrays")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_difference": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "n_charts": int(len(diff)), "n_boot": int(n_boot),
            "fraction_boot_a_better": float(np.mean(boots > 0))}
