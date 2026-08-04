"""Skew & prevalence analysis — MCC/confusion counts and prevalence-weighted
accuracy/cost, computed over scored non-infrastructure runs.

This module is pure aggregation over already-computed per-run dicts (as
produced by RunResult.to_dict()); it makes no LLM calls and performs no I/O
other than the CSV writer at the bottom.

Design note: MCC does not post-stratify (it is nonlinear in prevalence) —
Acc(pi) and cost(pi) carry the prevalence sweep instead. MCC is reported at
the as-measured balance only, with that balance stated alongside the number.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from csco.evaluation.scoring import _cell_stats

DEFAULT_PI_GRID = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95


# ---------------------------------------------------------------------------
# Confusion counts + MCC (Task 3)
# ---------------------------------------------------------------------------

def confusion_counts(runs: list[dict[str, Any]]) -> dict[str, int]:
    """Tally act_axis cells across runs. Runs without an act_axis key at all
    (legacy runs scored before this key existed) are tracked separately under
    "no_act_axis" so they can be reported as "no data" rather than silently
    dropped or miscounted as zero."""
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "unbinarizable": 0, "no_act_axis": 0}
    for r in runs:
        axis = r.get("scores", {}).get("act_axis")
        if axis is None:
            counts["no_act_axis"] += 1
            continue
        cell = axis.get("cell")
        if cell in counts:
            counts[cell] += 1
    return counts


def mcc_from_counts(tp: int, tn: int, fp: int, fn: int) -> float | str:
    """Matthews Correlation Coefficient. Returns "n/a (single-class)" if any
    marginal sum is zero (division would be zero/undefined), never 0 and
    never a crash.

    MCC does not post-stratify (nonlinear in prevalence); Acc(pi) carries the
    prevalence sweep.
    """
    marginals = [tp + fp, tp + fn, tn + fp, tn + fn]
    if any(m == 0 for m in marginals):
        return "n/a (single-class)"
    numerator = (tp * tn) - (fp * fn)
    denominator = (marginals[0] * marginals[1] * marginals[2] * marginals[3]) ** 0.5
    return numerator / denominator


def false_alarm_rate(fp: int, tn: int) -> float | None:
    """FP / (FP + TN), computed over oracle-null cases. None if no null cases."""
    denom = fp + tn
    return fp / denom if denom > 0 else None


def miss_rate(fn: int, tp: int) -> float | None:
    """FN / (FN + TP), computed over action-required cases. None if no such cases."""
    denom = fn + tp
    return fn / denom if denom > 0 else None


# ---------------------------------------------------------------------------
# Stratum helpers (Task 4)
# ---------------------------------------------------------------------------

def stratum_of(run: dict[str, Any]) -> str | None:
    """"null" | "action" | None (unknown/legacy run without fixture_meta)."""
    return run.get("fixture_meta", {}).get("stratum")


def suite_of(run: dict[str, Any]) -> str | None:
    return run.get("fixture_meta", {}).get("suite")


def _overall_correct(run: dict[str, Any], metric: str = "overall_correct") -> bool:
    return bool(run.get("scores", {}).get(metric, False))


# SDM-3 (overall_correct: rule_id + disposition + escalation) is the primary
# headline metric. SDM-full (overall_correct_full: SDM-3 AND permitted_actions
# set equality) is reported alongside it — both weighted to the null share pi.
ACC_PI_METRICS = ("overall_correct", "overall_correct_full")


def acc_pi_rows(
    runs: list[dict[str, Any]],
    arm: str,
    pi_values: list[float] | None = None,
    metric: str = "overall_correct",
) -> list[dict[str, Any]]:
    """Acc(pi) = pi*Acc_null + (1-pi)*Acc_action, swept over pi_values.

    metric: "overall_correct" (SDM-3, default) or "overall_correct_full"
    (SDM-3 AND permitted_actions set equality). Runs missing the requested
    metric key (e.g. legacy runs scored before overall_correct_full existed)
    are excluded from that stratum's mean rather than counted as incorrect.

    Rows always include acc_null/acc_action/n_null/n_action so callers can
    label "n/a" when a stratum has zero runs, without a crash.
    """
    pi_values = pi_values if pi_values is not None else DEFAULT_PI_GRID

    null_runs = [r for r in runs if stratum_of(r) == "null" and metric in r.get("scores", {})]
    action_runs = [r for r in runs if stratum_of(r) == "action" and metric in r.get("scores", {})]

    n_null, n_action = len(null_runs), len(action_runs)
    acc_null = mean(_overall_correct(r, metric) for r in null_runs) if n_null else None
    acc_action = mean(_overall_correct(r, metric) for r in action_runs) if n_action else None

    rows = []
    for pi in pi_values:
        if acc_null is None or acc_action is None:
            acc_pi = None
        else:
            acc_pi = pi * acc_null + (1 - pi) * acc_action
        rows.append({
            "arm": arm,
            "metric": metric,
            "pi": pi,
            "acc_pi": acc_pi,
            "acc_null": acc_null,
            "acc_action": acc_action,
            "n_null": n_null,
            "n_action": n_action,
        })
    return rows


def _mean_tokens(runs: list[dict[str, Any]]) -> float | None:
    vals = [
        r["scores"]["approx_policy_content_tokens"]
        for r in runs
        if "approx_policy_content_tokens" in r.get("scores", {})
    ]
    return mean(vals) if vals else None


def cost_pi_rows(
    runs: list[dict[str, Any]],
    arm: str,
    pi_values: list[float] | None = None,
) -> list[dict[str, Any]]:
    """cost(pi) = pi*mean_tokens_null + (1-pi)*mean_tokens_action."""
    pi_values = pi_values if pi_values is not None else DEFAULT_PI_GRID

    null_runs = [r for r in runs if stratum_of(r) == "null"]
    action_runs = [r for r in runs if stratum_of(r) == "action"]

    mean_tokens_null = _mean_tokens(null_runs)
    mean_tokens_action = _mean_tokens(action_runs)

    rows = []
    for pi in pi_values:
        if mean_tokens_null is None or mean_tokens_action is None:
            cost_pi = None
        else:
            cost_pi = pi * mean_tokens_null + (1 - pi) * mean_tokens_action
        rows.append({
            "arm": arm,
            "pi": pi,
            "cost_pi": cost_pi,
            "mean_tokens_null": mean_tokens_null,
            "mean_tokens_action": mean_tokens_action,
            "n_null": len(null_runs),
            "n_action": len(action_runs),
        })
    return rows


def pass_k_by_stratum(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """pass^k split by stratum ("null"/"action"), reusing the same pass_all_k
    semantics as csco.evaluation.scoring._cell_stats (all reps of a fixture×arm cell
    correct, excluding infrastructure failures), grouped by scenario_id."""
    result: dict[str, dict[str, Any]] = {}
    for stratum in ("null", "action"):
        stratum_runs = [r for r in runs if stratum_of(r) == stratum]
        by_scenario: dict[str, list[dict]] = defaultdict(list)
        for r in stratum_runs:
            by_scenario[r["scenario_id"]].append(r)
        if not by_scenario:
            result[stratum] = {"n_cells": 0, "pass_k": None, "k": 0}
            continue
        cell_stats = [_cell_stats(cell_runs) for cell_runs in by_scenario.values()]
        n_cells = len(cell_stats)
        pk_k = sum(1 for cs in cell_stats if cs["pass_all_k"])
        result[stratum] = {"n_cells": n_cells, "pass_k": pk_k / n_cells if n_cells else None, "k": pk_k}
    return result


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_acc_pi_csv(path: str | Path, all_rows: list[dict[str, Any]]) -> None:
    """Write the machine-readable acc_pi.csv (columns: arm, metric, pi, acc_pi,
    acc_null, acc_action, n_null, n_action). metric is "overall_correct"
    (SDM-3) or "overall_correct_full" (SDM-full)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["arm", "metric", "pi", "acc_pi", "acc_null", "acc_action", "n_null", "n_action"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "overall_correct" if k == "metric" else None) for k in fieldnames})
