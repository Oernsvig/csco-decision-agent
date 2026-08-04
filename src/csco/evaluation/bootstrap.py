"""Scenario-cluster bootstrap and workload-crossover analysis.

Implements the uncertainty machinery reported in the dissertation Results
section (Tables 2/3 and Figure 20):

  - Scenario-cluster bootstrap CIs for run-level correctness, per arm × suite.
    Repetitions of the same scenario share identical inputs and governing policy,
    so they are treated as clustered (not independent): each resample draws
    *scenarios* with replacement and retains all repetitions within a scenario
    (Davison & Hinkley, 1997).
  - Paired scenario-cluster differences between two arms (same sampled scenario
    indices applied to both).
  - pass^3 (three-run all-correct) and scenario profiles (3/3, mixed, 0/3).
  - Prevalence-weighted correctness crossover pi* between two arms, with the
    crossover direction/existence propagated through a paired scenario-cluster
    bootstrap.

Because the scenarios were constructed for procedural-policy coverage rather
than sampled from an operational population, these intervals quantify sensitivity
to the tested scenario composition — not population-general performance.
"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean, median
from typing import Any, Callable

from csco.evaluation.skew_analysis import confusion_counts, mcc_from_counts
from csco.evaluation.skew_analysis import _overall_correct, stratum_of

DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 20240

# A run is a "null" (no-action) case when its fixture stratum is "null"; every
# other stratum ("action") is a prescribed-action case.
Run = dict[str, Any]


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def group_by_scenario(
    runs: list[Run],
    metric: str = "overall_correct",
) -> dict[str, list[bool]]:
    """Map scenario_id -> list of per-repetition correctness booleans."""
    clusters: dict[str, list[bool]] = {}
    for r in runs:
        sid = r.get("scenario_id")
        if sid is None:
            continue
        clusters.setdefault(sid, []).append(_overall_correct(r, metric))
    return clusters


def _run_level_correctness(clusters: dict[str, list[bool]]) -> float | None:
    """Mean correctness across all runs (every repetition of every scenario)."""
    total = sum(len(v) for v in clusters.values())
    if total == 0:
        return None
    correct = sum(sum(v) for v in clusters.values())
    return correct / total


def filter_stratum(runs: list[Run], stratum: str) -> list[Run]:
    """Return only runs whose fixture stratum matches (\"action\" or \"null\")."""
    return [r for r in runs if stratum_of(r) == stratum]


# ---------------------------------------------------------------------------
# Single-arm scenario-cluster bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_correctness_ci(
    runs: list[Run],
    *,
    metric: str = "overall_correct",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run-level correctness point estimate + 95% percentile interval.

    Resamples scenarios (clusters) with replacement, retaining all repetitions
    within each drawn scenario.
    """
    clusters = group_by_scenario(runs, metric)
    ids = list(clusters.keys())
    point = _run_level_correctness(clusters)
    if not ids or point is None:
        return {"point": None, "ci_low": None, "ci_high": None, "n_scenarios": 0, "n_runs": 0}

    rng = random.Random(seed)
    n = len(ids)
    draws: list[float] = []
    for _ in range(n_boot):
        picked = [clusters[ids[rng.randrange(n)]] for _ in range(n)]
        total = sum(len(v) for v in picked)
        correct = sum(sum(v) for v in picked)
        draws.append(correct / total if total else 0.0)
    draws.sort()
    return {
        "point": point,
        "ci_low": draws[int(0.025 * n_boot)],
        "ci_high": draws[int(0.975 * n_boot)],
        "n_scenarios": n,
        "n_runs": sum(len(v) for v in clusters.values()),
    }


# ---------------------------------------------------------------------------
# Paired difference between two arms (same sampled scenarios)
# ---------------------------------------------------------------------------

def paired_difference_ci(
    runs_a: list[Run],
    runs_b: list[Run],
    *,
    metric: str = "overall_correct",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Paired run-level correctness difference (A - B) + 95% interval.

    Only scenarios present in both arms are used; each resample applies the same
    drawn scenario indices to both arms.
    """
    ca = group_by_scenario(runs_a, metric)
    cb = group_by_scenario(runs_b, metric)
    ids = [i for i in ca.keys() if i in cb]
    if not ids:
        return {"point": None, "ci_low": None, "ci_high": None, "n_scenarios": 0}

    def _rate(clusters, picked_ids):
        picked = [clusters[i] for i in picked_ids]
        total = sum(len(v) for v in picked)
        return (sum(sum(v) for v in picked) / total) if total else 0.0

    point = _rate(ca, ids) - _rate(cb, ids)
    rng = random.Random(seed)
    n = len(ids)
    draws: list[float] = []
    for _ in range(n_boot):
        picked_ids = [ids[rng.randrange(n)] for _ in range(n)]
        draws.append(_rate(ca, picked_ids) - _rate(cb, picked_ids))
    draws.sort()
    return {
        "point": point,
        "ci_low": draws[int(0.025 * n_boot)],
        "ci_high": draws[int(0.975 * n_boot)],
        "n_scenarios": n,
    }


def _cluster_runs_by_scenario(runs: list[Run]) -> dict[str, list[Run]]:
    clusters: dict[str, list[Run]] = {}
    for r in runs:
        if r.get("run_status") == "infrastructure_failure":
            continue
        sid = r.get("scenario_id")
        if sid is None:
            continue
        clusters.setdefault(sid, []).append(r)
    return clusters


def _mcc_from_runs(runs: list[Run]) -> float | None:
    scored = [r for r in runs if r.get("run_status") != "infrastructure_failure"]
    counts = confusion_counts(scored)
    mcc = mcc_from_counts(counts["TP"], counts["TN"], counts["FP"], counts["FN"])
    return mcc if isinstance(mcc, float) else None


def paired_mcc_difference_ci(
    runs_a: list[Run],
    runs_b: list[Run],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Paired scenario-cluster bootstrap for ΔMCC = MCC_A − MCC_B.

    The same sampled scenario indices are applied to both arms. Resamples that
    collapse to a single-class confusion matrix are skipped rather than forced
    to zero.
    """
    ca = _cluster_runs_by_scenario(runs_a)
    cb = _cluster_runs_by_scenario(runs_b)
    ids = [i for i in ca.keys() if i in cb]
    if not ids:
        return {"point": None, "ci_low": None, "ci_high": None, "n_scenarios": 0, "n_valid_resamples": 0}

    def _sample_mcc(clusters: dict[str, list[Run]], picked_ids: list[str]) -> float | None:
        picked: list[Run] = []
        for sid in picked_ids:
            picked.extend(clusters[sid])
        return _mcc_from_runs(picked)

    point_a = _sample_mcc(ca, ids)
    point_b = _sample_mcc(cb, ids)
    if point_a is None or point_b is None:
        return {"point": None, "ci_low": None, "ci_high": None, "n_scenarios": len(ids), "n_valid_resamples": 0}

    rng = random.Random(seed)
    n = len(ids)
    draws: list[float] = []
    attempts = 0
    max_attempts = n_boot * 20
    while len(draws) < n_boot and attempts < max_attempts:
        attempts += 1
        picked_ids = [ids[rng.randrange(n)] for _ in range(n)]
        a = _sample_mcc(ca, picked_ids)
        b = _sample_mcc(cb, picked_ids)
        if a is None or b is None:
            continue
        draws.append(a - b)

    if not draws:
        return {"point": point_a - point_b, "ci_low": None, "ci_high": None, "n_scenarios": len(ids), "n_valid_resamples": 0}

    draws.sort()
    return {
        "point": point_a - point_b,
        "ci_low": draws[int(0.025 * len(draws))],
        "ci_high": draws[int(0.975 * len(draws))],
        "n_scenarios": len(ids),
        "n_valid_resamples": len(draws),
    }


# ---------------------------------------------------------------------------
# pass^3 and scenario profiles
# ---------------------------------------------------------------------------

def pass_cubed_profile(
    runs: list[Run],
    *,
    metric: str = "overall_correct",
) -> dict[str, Any]:
    """Three-run all-correct rate (pass^3) and 3/3 · mixed · 0/3 scenario counts."""
    clusters = group_by_scenario(runs, metric)
    full = mixed = zero = 0
    for reps in clusters.values():
        c = sum(reps)
        if c == len(reps) and len(reps) > 0:
            full += 1
        elif c == 0:
            zero += 1
        else:
            mixed += 1
    n = len(clusters)
    return {
        "n_scenarios": n,
        "all_correct": full,
        "mixed": mixed,
        "all_wrong": zero,
        "pass_cubed_rate": (full / n) if n else None,
    }


# ---------------------------------------------------------------------------
# Workload crossover pi*
# ---------------------------------------------------------------------------

def crossover_pi_star(a1: float, n1: float, a2: float, n2: float) -> float | None:
    """Prevalence (null-share) at which prevalence-weighted correctness of two
    architectures is equal.

    OC_d(pi) = (1 - pi) * A_d + pi * N_d. Solving OC_d1 = OC_d2 gives:
        pi* = (A_d1 - A_d2) / ((A_d1 - A_d2) + (N_d2 - N_d1))
    Returns pi* only when it lies strictly inside (0, 1) AND the two arms cross
    (one better on action, the other on null); otherwise None.
    """
    denom = (a1 - a2) + (n2 - n1)
    if denom == 0:
        return None
    pi = (a1 - a2) / denom
    interior = (a1 - a2) * (n2 - n1) > 0  # opposite-signed advantages -> a crossing
    if interior and 0.0 < pi < 1.0:
        return pi
    return None


def crossover_analysis(
    runs_a: list[Run],
    runs_b: list[Run],
    *,
    metric: str = "overall_correct",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Observed pi* between arm A and arm B, with a paired scenario-cluster
    bootstrap over the action and null suites (resampled independently).

    Classifies each resample as:
      - "crossover_a_dominates_action" : interior crossover, A better on action
      - "crossover_b_dominates_action" : interior crossover, B better on action
      - "a_weakly_dominates"           : A >= B on both suites (no interior crossover)
      - "b_weakly_dominates"           : B >= A on both suites
      - "equal"                        : identical on both suites
    """
    a_action = group_by_scenario(filter_stratum(runs_a, "action"), metric)
    a_null = group_by_scenario(filter_stratum(runs_a, "null"), metric)
    b_action = group_by_scenario(filter_stratum(runs_b, "action"), metric)
    b_null = group_by_scenario(filter_stratum(runs_b, "null"), metric)

    action_ids = [i for i in a_action if i in b_action]
    null_ids = [i for i in a_null if i in b_null]

    def _rate(clusters, ids):
        picked = [clusters[i] for i in ids]
        total = sum(len(v) for v in picked)
        return (sum(sum(v) for v in picked) / total) if total else 0.0

    A1, N1 = _rate(a_action, action_ids), _rate(a_null, null_ids)
    A2, N2 = _rate(b_action, action_ids), _rate(b_null, null_ids)
    observed = crossover_pi_star(A1, N1, A2, N2)

    rng = random.Random(seed)
    na, nn = len(action_ids), len(null_ids)
    categories: dict[str, int] = {}
    pi_stars: list[float] = []
    for _ in range(n_boot):
        sa = [action_ids[rng.randrange(na)] for _ in range(na)] if na else []
        sn = [null_ids[rng.randrange(nn)] for _ in range(nn)] if nn else []
        a1, n1 = _rate(a_action, sa), _rate(a_null, sn)
        a2, n2 = _rate(b_action, sa), _rate(b_null, sn)
        pi = crossover_pi_star(a1, n1, a2, n2)
        if pi is not None:
            key = "crossover_a_dominates_action" if a1 >= a2 else "crossover_b_dominates_action"
            pi_stars.append(pi)
        elif a1 == a2 and n1 == n2:
            key = "equal"
        elif a1 >= a2 and n1 >= n2:
            key = "a_weakly_dominates"
        elif a2 >= a1 and n2 >= n1:
            key = "b_weakly_dominates"
        else:  # numeric edge (e.g. pi* exactly at a bound) — record as no interior crossover
            key = "a_weakly_dominates" if (a1 + n1) >= (a2 + n2) else "b_weakly_dominates"
        categories[key] = categories.get(key, 0) + 1

    pi_summary = None
    if pi_stars:
        pi_stars.sort()
        pi_summary = {
            "median": median(pi_stars),
            "ci_low": pi_stars[int(0.025 * len(pi_stars))],
            "ci_high": pi_stars[int(0.975 * len(pi_stars))],
            "n_resamples_with_crossover": len(pi_stars),
        }
    return {
        "observed": {"A_action": A1, "A_null": N1, "B_action": A2, "B_null": N2, "pi_star": observed},
        "resample_categories": {k: v / n_boot for k, v in categories.items()},
        "pi_star_summary": pi_summary,
        "n_action_scenarios": na,
        "n_null_scenarios": nn,
    }


# ---------------------------------------------------------------------------
# Convenience: per-arm × per-suite table (Table 2 / Table 3 Panel A)
# ---------------------------------------------------------------------------

def suite_table(
    runs_by_arm: dict[str, list[Run]],
    stratum: str,
    *,
    metric: str = "overall_correct",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    """Per-arm correctness CI + pass^3 for one suite (stratum)."""
    out: dict[str, dict[str, Any]] = {}
    for arm, runs in runs_by_arm.items():
        suite_runs = filter_stratum(runs, stratum)
        ci = bootstrap_correctness_ci(suite_runs, metric=metric, n_boot=n_boot, seed=seed)
        p3 = pass_cubed_profile(suite_runs, metric=metric)
        out[arm] = {**ci, **{f"pass3_{k}": v for k, v in p3.items()}}
    return out
