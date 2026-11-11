# VERBATIM copy of V3.3.2 notebook cell 71 (Section 17), kept only so tests can reproduce the old behaviour.
import math
import numpy as np


def levenshtein_distance(a, b):
    a = str(a)
    b = str(b)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)

            current.append(
                min(insert_cost, delete_cost, replace_cost)
            )

        previous = current

    return previous[-1]


def similarity_transform(distance_ratio):
    return 2.0 - (2.0 / (1.0 + math.exp(-distance_ratio)))


def parse_table_text(text):
    x_values = []
    y_values = []

    for raw_line in str(text).splitlines():
        line = raw_line.strip()

        if not line or "|" not in line:
            continue

        left, right = line.split("|", 1)
        x_value = left.strip()
        y_text = right.strip()

        try:
            y_value = float(y_text)
        except ValueError:
            continue
        if not np.isfinite(y_value):
            continue

        x_values.append(x_value)
        y_values.append(y_value)

    return x_values, np.asarray(y_values, dtype=float)


def categorical_similarity(true_values, pred_values):
    if len(true_values) != len(pred_values):
        return 0.0

    if len(true_values) == 0:
        return 0.0

    total_chars = sum(max(len(str(v)), 1) for v in true_values)

    total_distance = sum(
        levenshtein_distance(t, p)
        for t, p in zip(true_values, pred_values)
    )

    ratio = total_distance / total_chars
    return similarity_transform(ratio)


def numerical_similarity(true_values, pred_values):
    true_values = np.asarray(true_values, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    if len(true_values) != len(pred_values):
        return 0.0

    if len(true_values) == 0:
        return 0.0

    rmse = np.sqrt(
        np.mean((true_values - pred_values) ** 2)
    )

    baseline_rmse = np.sqrt(
        np.mean(
            (true_values - np.mean(true_values)) ** 2
        )
    )

    if baseline_rmse == 0:
        return 1.0 if rmse == 0 else 0.0

    ratio = rmse / baseline_rmse
    return similarity_transform(ratio)


def benetech_vertical_score(true_text, pred_text):
    true_x, true_y = parse_table_text(true_text)
    pred_x, pred_y = parse_table_text(pred_text)

    if len(true_x) != len(pred_x):
        return 0.0

    x_score = categorical_similarity(true_x, pred_x)
    y_score = numerical_similarity(true_y, pred_y)

    return (x_score + y_score) / 2.0
