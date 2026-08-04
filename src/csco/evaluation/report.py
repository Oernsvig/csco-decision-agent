"""Batch-evaluation report generation.

Pure text/aggregation over already-collected run dicts (as produced by
RunResult.to_dict(), grouped by arm) — no run orchestration, no CLI, no I/O
beyond returning strings for the caller to print/save. Split out of
batch_run.py, which retains fixture discovery, run orchestration, and the
CLI entrypoint.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev
from typing import Any

from csco.evaluation import skew_analysis
from csco.evaluation.scoring import _cell_stats


# ---------------------------------------------------------------------------
# Top-level summary report
# ---------------------------------------------------------------------------

def _generate_summary_report(
    results_by_arm: dict[str, list[dict[str, Any]]],
    n_runs: int = 1,
) -> str:
    _PRIMARY_ARMS   = ["static", "vector", "lexical"]

    primary_arms  = [a for a in _PRIMARY_ARMS  if results_by_arm.get(a)]
    arms = primary_arms

    lines: list[str] = [
        "═" * 80,
        "CSCO DECISION AGENT — BATCH EVALUATION SUMMARY",
        "  Primary comparison: static | vector | lexical",
        "  (Three prose-identical policy-grounding architectures)",
        f"  runs per cell: {n_runs}",
        "═" * 80,
        "",
    ]

    for arm in arms:
        runs = results_by_arm.get(arm, [])
        if not runs:
            lines += [f"\n{arm.upper()}: NO RUNS"]
            continue

        lines += [f"\n{'=' * 80}"]
        lines += [f"ARM: {arm.upper()}"]
        lines += [f"{'=' * 80}"]
        lines += [f"Total runs: {len(runs)}  (across all cells)\n"]

        # ================================================================
        # FAILURE HANDLING & TRIAGE
        # ================================================================

        # Classify runs by status
        successful_runs = [r for r in runs if r.get("run_status", "success") == "success"]
        infrastructure_failures = [r for r in runs if r.get("run_status") == "infrastructure_failure"]
        model_output_failures = [r for r in runs if r.get("run_status") == "model_output_failure"]
        tool_loop_failures = [r for r in runs if r.get("run_status") == "tool_loop_failure"]
        extraction_failures = [r for r in runs if r.get("run_status") == "extraction_failure"]
        unknown_failures = [r for r in runs if r.get("run_status") == "unknown_failure"]

        if any([infrastructure_failures, model_output_failures, tool_loop_failures, extraction_failures, unknown_failures]):
            lines += ["", "RUN STATUS & FAILURE TRIAGE", "─" * 80]
            lines += [f"  successful_runs: {len(successful_runs)}/{len(runs)}"]
            if infrastructure_failures:
                lines += [f"  infrastructure_failures (excluded from accuracy): {len(infrastructure_failures)}"]
            if model_output_failures:
                lines += [f"  model_output_failures (counted incorrect): {len(model_output_failures)}"]
            if tool_loop_failures:
                lines += [f"  tool_loop_failures (counted incorrect): {len(tool_loop_failures)}"]
            if extraction_failures:
                lines += [f"  extraction_failures (counted incorrect): {len(extraction_failures)}"]
            if unknown_failures:
                lines += [f"  unknown_failures (excluded from accuracy): {len(unknown_failures)}"]

        # ================================================================
        # CORRECTNESS METRICS
        # ================================================================

        lines += ["", "CORRECTNESS METRICS", "─" * 80]

        # For accuracy calculation, only use runs that should be included
        accuracy_runs = [r for r in runs if r.get("included_in_accuracy", True)]

        # For incorrect counts, use successful runs but count failures as incorrect
        overall_correct_vals = []
        for r in accuracy_runs:
            if r in successful_runs:
                overall_correct_vals.append(r["scores"].get("overall_correct", False))
            else:
                # Architecture failures are counted as incorrect
                if r.get("counted_as_incorrect", False):
                    overall_correct_vals.append(False)

        if overall_correct_vals:
            acc_mean = mean(overall_correct_vals)
            acc_std = stdev(overall_correct_vals) if len(overall_correct_vals) > 1 else 0.0
            correct_count = sum(overall_correct_vals)

            if n_runs > 1:
                lines += [
                    f"  overall_correct: {correct_count}/{len(overall_correct_vals)} "
                    f"  mean={acc_mean:.3f}, σ={acc_std:.3f}"
                ]
            else:
                lines += [
                    f"  overall_correct: {correct_count}/{len(overall_correct_vals)} "
                    f"({100*acc_mean:.1f}%)"
                ]
        else:
            lines += ["  overall_correct: [no runs to analyze]"]

        for key, label in [
            ("correctness_disposition",       "correctness_disposition"),
            ("correctness_escalation",        "correctness_escalation"),
            ("correctness_rule_id",           "correctness_rule_id"),
            ("correctness_permitted_actions", "correctness_permitted_actions"),
        ]:
            vals = [r["scores"][key] for r in successful_runs if key in r["scores"]]
            if vals:
                lines += [f"  {label}: {mean(vals):.3f} (avg across suppliers × runs)"]

        # Direct band derivation metric
        band_vals = [r["scores"]["assigned_risk_level_correct"] for r in successful_runs if "assigned_risk_level_correct" in r["scores"]]
        if band_vals:
            lines += [f"  assigned_risk_level_correct: {mean(band_vals):.3f} (direct band derivation)"]

        # ================================================================
        # DEFERRED ACTION VIOLATIONS
        # ================================================================

        deferred_flags = [
            r["scores"].get("deferred_action_recommended", False)
            for r in successful_runs
            if "deferred_action_recommended" in r["scores"]
        ]
        if deferred_flags:
            lines += ["", "POLICY FAITHFULNESS", "─" * 80]
            violation_rate = mean(deferred_flags)
            lines += [
                f"  deferred_action_recommended: "
                f"{sum(deferred_flags)}/{len(deferred_flags)} "
                f"({100*violation_rate:.1f}% of supplier runs had ≥1 deferred action)"
            ]

        # ================================================================
        # RETRIEVAL QUALITY
        # ================================================================

        retrieval_ranks = [
            r["scores"]["retrieval_rank_of_rule_id_mention"]
            for r in successful_runs
            if r["scores"].get("retrieval_rank_of_rule_id_mention") is not None
        ]
        # STRICT: chunk IS the actual provision for the governing rule
        retrieval_strict = [
            r["scores"]["retrieval_governing_provision_retrieved"]
            for r in successful_runs
            if "retrieval_governing_provision_retrieved" in r["scores"]
        ]
        # LOOSE: rule ID mentioned anywhere in retrieved chunks (diagnostic)
        retrieval_loose = [
            r["scores"]["retrieval_rule_id_mentioned"]
            for r in successful_runs
            if "retrieval_rule_id_mentioned" in r["scores"]
        ]
        steps_to_rule = [
            r["scores"]["steps_to_correct_rule"]
            for r in successful_runs
            if r["scores"].get("steps_to_correct_rule") is not None
        ]
        steps_efficiency = [
            r["scores"]["steps_efficiency"]
            for r in successful_runs
            if r["scores"].get("steps_efficiency") is not None and r["scores"]["steps_efficiency"] > 0
        ]

        if retrieval_ranks or retrieval_strict or retrieval_loose or steps_to_rule or steps_efficiency:
            lines += ["", "RETRIEVAL QUALITY METRICS", "─" * 80]
            if retrieval_strict:
                strict_rate = mean(retrieval_strict)
                lines += [f"  retrieval_governing_provision_retrieved (STRICT): {sum(retrieval_strict)}/{len(retrieval_strict)} ({100*strict_rate:.1f}%)"]
            if retrieval_loose:
                loose_rate = mean(retrieval_loose)
                lines += [f"  retrieval_rule_id_mentioned (loose diagnostic): {sum(retrieval_loose)}/{len(retrieval_loose)} ({100*loose_rate:.1f}%)"]
            if retrieval_ranks:
                r_mean = mean(retrieval_ranks)
                r_std = stdev(retrieval_ranks) if len(retrieval_ranks) > 1 else 0
                lines += [f"  retrieval_rank (when hit): μ={r_mean:.2f}, σ={r_std:.2f}"]

            # Vector failure-mode diagnostics
            retrieved_but_wrong = [
                r["scores"]["governing_provision_retrieved_but_wrong_decision"]
                for r in successful_runs
                if "governing_provision_retrieved_but_wrong_decision" in r["scores"]
            ]
            wrong_pass = [
                r["scores"]["wrong_pass_rule_selected"]
                for r in successful_runs
                if "wrong_pass_rule_selected" in r["scores"]
            ]
            supplier_as_routing = [
                r["scores"]["supplier_rule_used_as_routing_error"]
                for r in successful_runs
                if "supplier_rule_used_as_routing_error" in r["scores"]
            ]
            routing_as_supplier = [
                r["scores"]["routing_rule_used_as_supplier_error"]
                for r in successful_runs
                if "routing_rule_used_as_supplier_error" in r["scores"]
            ]
            correct_pass = [
                r["scores"]["retrieved_correct_pass"]
                for r in successful_runs
                if "retrieved_correct_pass" in r["scores"]
            ]
            def_band = [
                r["scores"]["retrieved_definition_band"]
                for r in successful_runs
                if "retrieved_definition_band" in r["scores"]
            ]
            def_tier = [
                r["scores"]["retrieved_definition_tier"]
                for r in successful_runs
                if "retrieved_definition_tier" in r["scores"]
            ]
            if retrieved_but_wrong:
                lines += [
                    f"  governing_provision_retrieved_but_wrong_decision: "
                    f"{sum(retrieved_but_wrong)}/{len(retrieved_but_wrong)} "
                    f"({100*mean(retrieved_but_wrong):.1f}%) "
                    f"  ← retrieval OK but rule misapplied"
                ]
            if wrong_pass:
                lines += [
                    f"  wrong_pass_rule_selected (any): "
                    f"{sum(wrong_pass)}/{len(wrong_pass)} ({100*mean(wrong_pass):.1f}%)"
                ]
            if supplier_as_routing:
                lines += [
                    f"    supplier_rule_used_as_routing_error: "
                    f"{sum(supplier_as_routing)}/{len(supplier_as_routing)} ({100*mean(supplier_as_routing):.1f}%)"
                ]
            if routing_as_supplier:
                lines += [
                    f"    routing_rule_used_as_supplier_error: "
                    f"{sum(routing_as_supplier)}/{len(routing_as_supplier)} ({100*mean(routing_as_supplier):.1f}%)"
                ]
            if correct_pass:
                lines += [
                    f"  retrieved_correct_pass: "
                    f"{sum(correct_pass)}/{len(correct_pass)} ({100*mean(correct_pass):.1f}%)"
                ]
            if def_band:
                lines += [f"  retrieved_definition_band: {sum(def_band)}/{len(def_band)} ({100*mean(def_band):.1f}%)"]
            if def_tier:
                lines += [f"  retrieved_definition_tier: {sum(def_tier)}/{len(def_tier)} ({100*mean(def_tier):.1f}%)"]
            sp_retrieved = [
                r["scores"]["retrieved_strategic_priority_section"]
                for r in successful_runs
                if "retrieved_strategic_priority_section" in r["scores"]
            ]
            if sp_retrieved:
                lines += [f"  retrieved_strategic_priority_section: {sum(sp_retrieved)}/{len(sp_retrieved)} ({100*mean(sp_retrieved):.1f}%)"]

            if steps_to_rule:
                s_mean = mean(steps_to_rule)
                s_std = stdev(steps_to_rule) if len(steps_to_rule) > 1 else 0
                lines += [f"  steps_to_correct_rule: μ={s_mean:.2f}, σ={s_std:.2f}"]
            if steps_efficiency:
                e_mean = mean(steps_efficiency)
                e_std = stdev(steps_efficiency) if len(steps_efficiency) > 1 else 0
                lines += [f"  steps_efficiency (1/steps): μ={e_mean:.3f}, σ={e_std:.3f}"]

        # ================================================================
        # STRATEGIC-PRIORITY ALIGNMENT
        # ================================================================

        strategic_mentions = [
            r["scores"]["strategic_priority_mentioned"]
            for r in runs
            if "strategic_priority_mentioned" in r["scores"]
        ]
        strategic_correct = [
            r["scores"]["strategic_priority_correct"]
            for r in runs
            if "strategic_priority_correct" in r["scores"]
        ]
        strategic_in_rationale = [
            r["scores"]["strategic_priority_in_rationale"]
            for r in runs
            if "strategic_priority_in_rationale" in r["scores"]
        ]
        wrong_priority = [
            r["scores"]["wrong_strategic_priority_used"]
            for r in runs
            if "wrong_strategic_priority_used" in r["scores"]
        ]
        if strategic_mentions:
            lines += ["", "STRATEGIC-PRIORITY ALIGNMENT", "─" * 80]
            lines += [
                "(strategic_priority_mentioned/correct = correct priority anywhere in answer;",
                " strategic_priority_in_rationale = integrated into per-supplier rationale [stricter])\n",
            ]
            lines += [
                f"  strategic_priority_mentioned: "
                f"{sum(strategic_mentions)}/{len(strategic_mentions)} "
                f"({100*mean(strategic_mentions):.1f}%)"
            ]
            if strategic_correct:
                lines += [
                    f"  strategic_priority_correct: "
                    f"{sum(strategic_correct)}/{len(strategic_correct)} "
                    f"({100*mean(strategic_correct):.1f}%)"
                ]
            if strategic_in_rationale:
                lines += [
                    f"  strategic_priority_in_rationale: "
                    f"{sum(strategic_in_rationale)}/{len(strategic_in_rationale)} "
                    f"({100*mean(strategic_in_rationale):.1f}%)  ← stricter"
                ]
            if wrong_priority:
                lines += [
                    f"  wrong_strategic_priority_used: "
                    f"{sum(wrong_priority)}/{len(wrong_priority)} "
                    f"({100*mean(wrong_priority):.1f}%)"
                ]

        # ================================================================
        # EFFICIENCY METRICS
        # ================================================================

        lines += ["", "EFFICIENCY METRICS", "─" * 80]
        tool_calls = [r["scores"].get("tool_calls_count", 0) for r in runs]
        llm_calls  = [r["scores"].get("llm_calls_total",  0) for r in runs]
        elapsed    = [r["scores"].get("elapsed_seconds",  0) for r in runs]
        policy_tokens = [r["scores"]["policy_tokens"] for r in runs if r["scores"].get("policy_tokens", 0) > 0]

        if tool_calls:
            lines += [f"  tool_calls_count: μ={mean(tool_calls):.2f} (avg per run)"]
        if llm_calls:
            lines += [f"  llm_calls_total: μ={mean(llm_calls):.2f} (avg per run)"]
        if elapsed:
            e_mean = mean(elapsed)
            e_std = stdev(elapsed) if len(elapsed) > 1 else 0
            lines += [f"  elapsed_seconds: μ={e_mean:.2f}s, σ={e_std:.2f}s"]
        if policy_tokens:
            lines += [f"  policy_tokens (Arm 1): μ={mean(policy_tokens):.0f}"]

        # ================================================================
        # REPEATABILITY (only meaningful when n_runs > 1)
        # ================================================================

        if n_runs > 1:
            fixtures_by_scenario: dict[str, list[dict]] = defaultdict(list)
            for r in runs:
                fixtures_by_scenario[r["scenario_id"]].append(r)

            n_cells = len(fixtures_by_scenario)
            cell_results = {
                sid: _cell_stats(cell_runs)
                for sid, cell_runs in fixtures_by_scenario.items()
            }
            consistent_cells = sum(1 for cs in cell_results.values() if cs["consistent"])

            lines += ["", "REPEATABILITY", "─" * 80]
            lines += [
                f"  fully consistent cells: {consistent_cells}/{n_cells} "
                f"({100*consistent_cells/n_cells:.1f}%) "
                f"— all {n_runs} reps agree on overall_correct"
            ]

            # Per-cell accuracy rates for inconsistent cells
            inconsistent = {
                sid: cs for sid, cs in cell_results.items() if not cs["consistent"]
            }
            if inconsistent:
                lines += [f"  inconsistent cells ({len(inconsistent)}):"]
                for sid in sorted(inconsistent):
                    cs = inconsistent[sid]
                    lines += [f"    {sid:25s}: {cs['k']}/{cs['n']} correct"]

        # ================================================================
        # PER-FIXTURE BREAKDOWN
        # ================================================================

        lines += ["", "PER-FIXTURE BREAKDOWN", "─" * 80]
        fixtures_by_scenario2: dict[str, list[dict]] = defaultdict(list)
        for r in runs:
            fixtures_by_scenario2[r["scenario_id"]].append(r)

        for scenario_id in sorted(fixtures_by_scenario2.keys()):
            cell_runs = fixtures_by_scenario2[scenario_id]
            cs = _cell_stats(cell_runs)
            consistency_tag = "" if cs["consistent"] or n_runs == 1 else "  [inconsistent]"
            lines += [f"  {scenario_id:25s}: {cs['k']}/{cs['n']} correct{consistency_tag}"]

    # ================================================================
    # COMPARATIVE SUMMARY (ACROSS ARMS)
    # ================================================================

    lines += ["\n" + "=" * 80]
    lines += ["PRIMARY COMPARISON — THREE PROSE-IDENTICAL ARCHITECTURES"]
    lines += ["=" * 80]
    lines += ["(static | vector | lexical — same generated prose corpus, different delivery structure)\n"]

    def _arm_stats_line(arm: str, n_runs: int) -> list[str]:
        runs = results_by_arm.get(arm, [])
        if not runs:
            return []
        flags = [r["scores"].get("overall_correct", False) for r in runs]
        k, n = sum(flags), len(flags)
        acc = k / n
        if n_runs > 1:
            sd = stdev(flags) if n > 1 else 0.0
            return [f"  {arm:10s}: {k:3d}/{n:3d}  mean={acc:.3f}, σ={sd:.3f}"]
        return [f"  {arm:10s}: {k:2d}/{n}  ({100*acc:5.1f}%)"]

    lines += ["Overall Correctness (primary arms)", "─" * 80]
    for arm in primary_arms:
        lines += _arm_stats_line(arm, n_runs)

    if n_runs > 1:
        lines += ["", "Repeatability — primary arms (fully consistent cells)", "─" * 80]
        for arm in primary_arms:
            runs = results_by_arm.get(arm, [])
            if not runs:
                continue
            by_scenario: dict[str, list] = defaultdict(list)
            for r in runs:
                by_scenario[r["scenario_id"]].append(r)
            n_cells = len(by_scenario)
            consistent = sum(1 for cell_runs in by_scenario.values() if _cell_stats(cell_runs)["consistent"])
            lines += [f"  {arm:10s}: {consistent}/{n_cells} cells ({100*consistent/n_cells:.1f}%)"]

        lines += ["", "Repeatability interpretation:", "─" * 80]
        lines += [
            "  Static may be stable (all content present) but may suffer from",
            "    cross-policy contamination when irrelevant types are in the prompt.",
            "  Vector may vary because retrieved chunks depend on query formulation",
            "    and semantic similarity — different queries hit different chunks.",
            "  Lexical graph may improve repeatability via deterministic playcard/",
            "    section/provision navigation and explicit reference edges over the",
            "    same prose.",
        ]

    lines += ["", "Deferred-Action Violations (primary arms)", "─" * 80]
    for arm in primary_arms:
        runs = results_by_arm.get(arm, [])
        flags = [r["scores"].get("deferred_action_recommended", False) for r in runs
                 if "deferred_action_recommended" in r["scores"]]
        if flags:
            lines += [f"  {arm:10s}: {sum(flags)}/{len(flags)} runs ({100*mean(flags):.1f}%)"]

    lines += ["", "Policy Content Tokens — primary arms (single tokenizer)", "─" * 80]
    for arm in primary_arms:
        runs = results_by_arm.get(arm, [])
        tok_vals = [r["scores"]["approx_policy_content_tokens"] for r in runs
                    if "approx_policy_content_tokens" in r["scores"]]
        if tok_vals:
            tok_mean = mean(tok_vals)
            tok_std  = stdev(tok_vals) if len(tok_vals) > 1 else 0.0
            if tok_std < 1:
                lines += [f"  {arm:10s}: {tok_mean:.0f} tokens (constant)"]
            else:
                lines += [f"  {arm:10s}: μ={tok_mean:.0f}, σ={tok_std:.0f} tokens"]

    lines += ["", "Total Model Token Spend — primary arms (billed input+output, all turns)", "─" * 80]
    for arm in primary_arms:
        runs = results_by_arm.get(arm, [])
        tot = [r["total_tokens"] for r in runs if r.get("total_tokens")]
        inp = [r["input_tokens"] for r in runs if r.get("input_tokens")]
        out = [r["output_tokens"] for r in runs if r.get("output_tokens")]
        if tot:
            parts = [f"total μ={mean(tot):.0f}"]
            if inp:
                parts.append(f"input μ={mean(inp):.0f}")
            if out:
                parts.append(f"output μ={mean(out):.0f}")
            lines += [f"  {arm:10s}: " + ", ".join(parts) + " tokens/run"]
        else:
            lines += [f"  {arm:10s}: (no usage_metadata captured)"]

    # Within-run uncached input: input the model processes fresh, deduplicating the
    # re-sent ReAct prefix. Computed analytically (largest reasoning turn + extractor
    # call) so it is independent of the provider's cross-run cache — a provider-agnostic
    # fresh-compute proxy relevant to self-hosted deployment.
    lines += ["", "Within-run uncached input — primary arms (fresh-compute proxy; excludes re-sent prefix)", "─" * 80]
    any_val = False
    for arm in primary_arms:
        runs = results_by_arm.get(arm, [])
        inp = [r["input_tokens"] for r in runs if r.get("input_tokens")]
        unc = [r.get("uncached_input_within_run") for r in runs
               if r.get("uncached_input_within_run") is not None]
        if not inp or not unc:
            continue
        any_val = True
        mean_in, mean_unc = mean(inp), mean(unc)
        cached_share = (1 - mean_unc / mean_in) * 100 if mean_in else 0.0
        lines += [
            f"  {arm:10s}: input μ={mean_in:.0f} → uncached μ={mean_unc:.0f} "
            f"({cached_share:.0f}% re-sent/cached within run)"
        ]
    if not any_val:
        lines += ["  (analytical within-run uncached not recorded — runs predate the per-turn capture)"]

    # === Dissertation tables — scenario-cluster bootstrap (Tables 2/3 + crossover) ===
    from csco.evaluation import bootstrap as _bs
    if any(results_by_arm.get(a) for a in primary_arms):
        for stratum, label in (
            ("action", "Table 2 — Prescribed-action cases (Suite A): scenario-cluster bootstrap"),
            ("null",   "Table 3 — Null cases (Suite B): scenario-cluster bootstrap"),
        ):
            tbl = _bs.suite_table(results_by_arm, stratum)
            lines += ["", label, "─" * 80,
                      f"  {'arm':10s} {'correct':>9s}  {'95% CI':>18s}  {'pass^3':>8s}"]
            for arm in primary_arms:
                t = tbl.get(arm)
                if not t or t.get("point") is None:
                    continue
                ci = f"[{t['ci_low']*100:.1f}, {t['ci_high']*100:.1f}]"
                p3 = t.get("pass3_pass_cubed_rate")
                p3s = f"{p3*100:.1f}%" if p3 is not None else "n/a"
                lines += [f"  {arm:10s} {t['point']*100:8.1f}%  {ci:>18s}  {p3s:>8s}"]
            if results_by_arm.get("lexical"):
                for other in ("static", "vector"):
                    if not results_by_arm.get(other):
                        continue
                    d = _bs.paired_difference_ci(
                        _bs.filter_stratum(results_by_arm["lexical"], stratum),
                        _bs.filter_stratum(results_by_arm[other], stratum),
                    )
                    if d["point"] is None:
                        continue
                    lines += [
                        f"    Lexical − {other:7s}: {d['point']*100:+.1f} pp  "
                        f"[{d['ci_low']*100:+.1f}, {d['ci_high']*100:+.1f}] pp"
                    ]
        if results_by_arm.get("static") and results_by_arm.get("lexical"):
            ca = _bs.crossover_analysis(results_by_arm["lexical"], results_by_arm["static"])
            obs = ca["observed"]["pi_star"]
            lines += ["", "Workload crossover — Lexical vs Static (null-share pi*)", "─" * 80,
                      f"  observed pi*: {('%.3f' % obs) if obs is not None else 'no interior crossover'}"]
            for k, v in sorted(ca["resample_categories"].items(), key=lambda x: -x[1]):
                lines += [f"    {k:32s}: {v*100:.1f}% of resamples"]
            ps = ca["pi_star_summary"]
            if ps:
                lines += [f"  pi* median {ps['median']:.3f}  central-95% "
                          f"[{ps['ci_low']:.3f}, {ps['ci_high']:.3f}]"]

    lines += ["", "Average Execution Time (seconds) — primary arms", "─" * 80]
    elapsed_by_arm = {
        arm: mean([r["scores"].get("elapsed_seconds", 0) for r in results_by_arm.get(arm, [])])
        for arm in primary_arms
        if results_by_arm.get(arm)
    }
    for arm in sorted(elapsed_by_arm, key=lambda a: elapsed_by_arm[a]):
        lines += [f"  {arm:10s}: {elapsed_by_arm[arm]:6.2f}s"]

    lines += ["", "Tool Call Efficiency (retrieval arms)", "─" * 80]
    for arm in ["vector", "lexical"]:
        runs = results_by_arm.get(arm, [])
        if not runs:
            continue
        tool_calls = [r["scores"].get("tool_calls_count", 0) for r in runs]
        correct = sum(r["scores"].get("overall_correct", False) for r in runs)
        if tool_calls:
            tc_mean = mean(tool_calls)
            eff = correct / tc_mean if tc_mean > 0 else 0
            lines += [f"  {arm:10s}: {tc_mean:.2f} tool calls/run, {eff:.3f} correct/tool call"]

    # ====================================================================
    # CLUSTER-LEVEL PERFORMANCE (requires spec metadata to be loaded)
    # ====================================================================

    lines += ["", "=" * 80]
    lines += ["PERFORMANCE BY DIAGNOSTIC CLUSTER"]
    lines += ["=" * 80]
    lines += [
        "(Cluster scores = overall_correct filtered to scenarios tagged with that cluster.",
        " They measure policy-fidelity correctness, NOT direct mention of cluster-specific",
        " features. In particular, C10_STRATEGIC_POSTURE cluster score ≠",
        " strategic_priority_mentioned — use the STRATEGIC-PRIORITY ALIGNMENT section",
        " for the direct mention metric.)\n",
    ]

    cluster_data = _collect_cluster_data(results_by_arm)
    if cluster_data:
        from csco.evaluation.clusters import TAXONOMY
        all_cluster_ids = sorted(cluster_data.keys())

        for cluster_id in all_cluster_ids:
            desc = TAXONOMY.get(cluster_id, {}).get("description", "")[:70]
            lines += [f"\n{cluster_id}  —  {desc}"]
            lines += ["─" * 80]
            by_arm = cluster_data[cluster_id]
            for arm in arms:
                cluster_runs = by_arm.get(arm, [])
                if not cluster_runs:
                    continue
                flags = [r["scores"].get("overall_correct", False) for r in cluster_runs]
                k, n = sum(flags), len(flags)
                acc = k / n if n else 0.0
                lines += [f"  {arm:10s}: {k:3d}/{n:3d} ({100*acc:.1f}%)"]
    else:
        lines += ["  (No cluster metadata available — run with fixtures that have oracle entries)"]

    # ====================================================================
    # FAILURE-MODE SUMMARY
    # ====================================================================

    lines += ["", "=" * 80]
    lines += ["FAILURE-MODE SUMMARY (supplier scenarios only)"]
    lines += ["=" * 80, ""]

    failure_keys = [
        ("deferred_action_recommended",             "Deferred action as live action"),
        ("any_over_escalation",                     "Over-escalation"),
        ("any_under_escalation",                    "Under-escalation"),
        ("any_over_replace",                        "Over-replace (replace when should monitor/maintain)"),
        ("action_set_superset_error",               "Action set superset error"),
        ("supplier_assessed_when_routing_should_stop", "Supplier assessed when routing should have stopped"),
        ("reasoning_answer_contains_correct_rule",  "Correct rule appeared in raw reasoning"),
        ("extractor_preserved_rule",                "Extractor preserved rule from reasoning"),
    ]

    lines += ["─" * 80]
    for key, label in failure_keys:
        lines += [f"  {label}:"]
        for arm in arms:
            runs = results_by_arm.get(arm, [])
            flags = [r["scores"].get(key) for r in runs if key in r["scores"]]
            if not flags:
                continue
            k = sum(bool(f) for f in flags)
            n = len(flags)
            lines += [f"    {arm:10s}: {k}/{n} ({100*k/n:.1f}%)"]

    # ====================================================================
    # LEXICAL REFERENCE METRICS
    # ====================================================================

    lexical_runs_all = results_by_arm.get("lexical", [])
    ref_keys = [
        ("reference_opened_band",                 "reference_used_correctly_band"),
        ("reference_opened_action",               "reference_used_correctly_action"),
        ("reference_opened_escalation",           "reference_used_correctly_escalation"),
        ("reference_opened_tier",                 "reference_used_correctly_tier"),
        ("reference_opened_strategic_priority",   "reference_used_correctly_strategic_priority"),
    ]
    lexical_ref_data = any(
        key in r["scores"] for r in lexical_runs_all for key, _ in ref_keys
    )
    if lexical_ref_data:
        lines += ["", "=" * 80]
        lines += ["LEXICAL GRAPH REFERENCE METRICS"]
        lines += ["=" * 80]
        lines += [
            "(Counts sections opened at any point during the run that match a REFERENCES edge",
            " on the governing provision. 'used_correctly' is a behavioural proxy — opening",
            " a section is necessary but not sufficient; it does not prove causal benefit.)",
            "",
        ]
        for opened_key, used_key in ref_keys:
            ref_type = opened_key.replace("reference_opened_", "")
            opened_flags = [r["scores"][opened_key] for r in lexical_runs_all if opened_key in r["scores"]]
            used_flags   = [r["scores"][used_key]   for r in lexical_runs_all
                            if used_key in r["scores"] and r["scores"][used_key] is not None]
            if opened_flags:
                k_open = sum(opened_flags)
                lines += [
                    f"  {ref_type:20s}: opened={k_open}/{len(opened_flags)} "
                    f"({100*k_open/len(opened_flags):.1f}%)"
                    + (
                        f", used_correctly={sum(used_flags)}/{len(used_flags)} "
                        f"({100*sum(used_flags)/len(used_flags):.1f}%)"
                        if used_flags else ""
                    )
                ]

    lines += ["", "=" * 80]

    # ====================================================================
    # EXTENDED SCORING METRICS (SDM-3 vs SDM-full, pass^k, grounding
    # cross-tab, failure-stage distribution) — appended section, does not
    # alter any line above.
    # ====================================================================
    lines += _generate_extended_metrics_section(results_by_arm, primary_arms, n_runs)

    # ====================================================================
    # SKEW & PREVALENCE ANALYSIS (Suite B null-region + Act/No-Act binarization)
    # ====================================================================
    lines += _generate_skew_prevalence_section(results_by_arm, primary_arms)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extended scoring metrics — Task 2/3/4 additions
# ---------------------------------------------------------------------------

def _scored_non_infra_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs that actually reached score() (i.e. not an infra failure / exception)."""
    return [
        r for r in runs
        if r.get("run_status") != "infrastructure_failure"
        and "overall_correct" in r.get("scores", {})
    ]


def _run_rule_id_correct(r: dict[str, Any]) -> bool | None:
    """Per-run rule_id correctness, unified across the routing and supplier cases."""
    sc = r.get("scores", {})
    if "routing_rule_correct" in sc:
        return bool(sc["routing_rule_correct"])
    if "correctness_rule_id" in sc:
        return sc["correctness_rule_id"] == 1.0
    return None


def _run_outcome_correct(r: dict[str, Any]) -> bool | None:
    """Per-run outcome correctness (disposition AND escalation), unified across cases."""
    sc = r.get("scores", {})
    if "routing_disposition_correct" in sc:
        return bool(sc["routing_disposition_correct"]) and bool(sc.get("routing_escalation_correct", False))
    if "correctness_disposition" in sc:
        return sc["correctness_disposition"] == 1.0 and sc.get("correctness_escalation") == 1.0
    return None


def _generate_extended_metrics_section(
    results_by_arm: dict[str, list[dict[str, Any]]],
    primary_arms: list[str],
    n_runs: int,
) -> list[str]:
    """SDM-3 vs SDM-full, pass^1 vs pass^k, grounding cross-tab, failure-stage
    distribution — per primary arm, over scored non-infrastructure runs."""
    lines: list[str] = ["", "=" * 80]
    lines += ["EXTENDED SCORING METRICS"]
    lines += ["=" * 80]

    # ----------------------------------------------------------------
    # SDM-3 vs SDM-full
    # ----------------------------------------------------------------
    lines += [
        "",
        "SDM-3 vs SDM-full",
        "─" * 80,
        "(SDM-3 = overall_correct: rule_id + disposition + escalation [primary headline,",
        " unchanged]. SDM-full = SDM-3 AND permitted_actions set equality.)",
        "",
    ]
    for arm in primary_arms:
        scored = _scored_non_infra_runs(results_by_arm.get(arm, []))
        if not scored:
            continue
        sdm3 = sum(bool(r["scores"].get("overall_correct")) for r in scored)
        n = len(scored)
        # overall_correct_full is absent on runs scored before this key existed.
        # Report those separately rather than as 0%.
        full_scored = [r for r in scored if "overall_correct_full" in r["scores"]]
        if full_scored:
            sdmf = sum(bool(r["scores"]["overall_correct_full"]) for r in full_scored)
            nf = len(full_scored)
            sdmf_str = f"SDM-full={sdmf}/{nf} ({100*sdmf/nf:.1f}%)"
        else:
            sdmf_str = "SDM-full=n/a (legacy runs predate this key)"
        lines += [f"  {arm:10s}: SDM-3={sdm3}/{n} ({100*sdm3/n:.1f}%)   {sdmf_str}"]

    # ----------------------------------------------------------------
    # pass^1 vs pass^k
    # ----------------------------------------------------------------
    lines += [
        "",
        f"pass^1 vs pass^{n_runs}",
        "─" * 80,
        "(pass^1 = existing mean accuracy over all runs [unchanged]. "
        f"pass^{n_runs} = fraction of fixture×arm cells where ALL {n_runs} rep(s) "
        "were overall_correct, excluding infrastructure-failure reps.)",
        "",
    ]
    for arm in primary_arms:
        runs = results_by_arm.get(arm, [])
        if not runs:
            continue
        flags = [r["scores"].get("overall_correct", False) for r in runs]
        p1_k, p1_n = sum(flags), len(flags)
        p1_rate = p1_k / p1_n if p1_n else 0.0

        by_scenario: dict[str, list[dict]] = defaultdict(list)
        for r in runs:
            by_scenario[r["scenario_id"]].append(r)
        cell_stats = [_cell_stats(cell_runs) for cell_runs in by_scenario.values()]
        n_cells = len(cell_stats)
        pk_k = sum(1 for cs in cell_stats if cs["pass_all_k"])
        pk_rate = pk_k / n_cells if n_cells else 0.0

        lines += [
            f"  {arm:10s}: pass^1={p1_k}/{p1_n} ({100*p1_rate:.1f}%)   "
            f"pass^{n_runs}={pk_k}/{n_cells} cells ({100*pk_rate:.1f}%)"
        ]

    # ----------------------------------------------------------------
    # Conditional grounding cross-tab
    # ----------------------------------------------------------------
    lines += [
        "",
        "GROUNDING CROSS-TAB (rule_id correctness vs. outcome correctness)",
        "─" * 80,
        "(outcome = disposition AND escalation correct. 'lucky guess rate' =",
        " P(outcome correct | rule_id wrong).)",
        "",
    ]
    for arm in primary_arms:
        scored = _scored_non_infra_runs(results_by_arm.get(arm, []))
        if not scored:
            continue

        rule_flags = [(r, _run_rule_id_correct(r)) for r in scored]
        rule_flags = [(r, f) for r, f in rule_flags if f is not None]
        n_rule = len(rule_flags)
        rule_correct_n = sum(1 for _, f in rule_flags if f)

        lines += [f"  {arm}:"]
        if n_rule == 0:
            lines += ["    P(rule_id correct): n=0 (no scored runs with rule_id metric)"]
            continue
        lines += [
            f"    P(rule_id correct): {rule_correct_n}/{n_rule} "
            f"({100*rule_correct_n/n_rule:.1f}%)"
        ]

        given_correct = [_run_outcome_correct(r) for r, f in rule_flags if f]
        given_correct = [o for o in given_correct if o is not None]
        if given_correct:
            k_gc = sum(given_correct)
            lines += [
                f"    P(outcome correct | rule_id correct): {k_gc}/{len(given_correct)} "
                f"({100*k_gc/len(given_correct):.1f}%)"
            ]
        else:
            lines += ["    P(outcome correct | rule_id correct): n=0"]

        given_wrong = [_run_outcome_correct(r) for r, f in rule_flags if not f]
        given_wrong = [o for o in given_wrong if o is not None]
        if given_wrong:
            k_gw = sum(given_wrong)
            lines += [
                f"    P(outcome correct | rule_id wrong)  : {k_gw}/{len(given_wrong)} "
                f"({100*k_gw/len(given_wrong):.1f}%)  ← lucky-guess rate"
            ]
        else:
            lines += ["    P(outcome correct | rule_id wrong)  : n=0"]

    # ----------------------------------------------------------------
    # Failure-stage distribution
    # ----------------------------------------------------------------
    lines += [
        "",
        "FAILURE-STAGE DISTRIBUTION (first-failure attribution, failed runs only)",
        "─" * 80,
    ]
    _STAGE_ORDER = [
        "extraction", "access", "classification", "pass_structure",
        "rule_selection", "execution", "unclassified",
    ]
    for arm in primary_arms:
        scored = _scored_non_infra_runs(results_by_arm.get(arm, []))
        # failure_stage is absent on runs scored before this key existed — use
        # presence, not just non-"none", as the denominator so older runs don't
        # silently read as 0 failures.
        with_stage = [r for r in scored if "failure_stage" in r["scores"]]
        if not with_stage:
            if scored:
                lines += [f"  {arm}: no failure_stage data (legacy runs predate this key)"]
            continue
        failed = [r for r in with_stage if r["scores"]["failure_stage"] != "none"]
        lines += [f"  {arm}: {len(failed)} failed / {len(with_stage)} scored"]
        if not failed:
            continue
        stage_counts: dict[str, int] = defaultdict(int)
        for r in failed:
            stage_counts[r["scores"]["failure_stage"]] += 1
        total_failed = len(failed)
        for stage in _STAGE_ORDER:
            if stage in stage_counts:
                c = stage_counts[stage]
                lines += [f"    {stage:16s}: {c}/{total_failed} ({100*c/total_failed:.1f}%)"]

    return lines


# ---------------------------------------------------------------------------
# SKEW & PREVALENCE ANALYSIS — Task 5 (uses csco.evaluation.skew_analysis for Task 3/4 math)
# ---------------------------------------------------------------------------

def _format_confusion_block(runs: list[dict[str, Any]], indent: str = "  ") -> list[str]:
    """2x2 + unbinarizable/legacy counts, MCC (with balance stated), FA/miss rates."""
    counts = skew_analysis.confusion_counts(runs)
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    lines = [
        f"{indent}TP={tp}  TN={tn}  FP={fp}  FN={fn}"
        f"   unbinarizable={counts['unbinarizable']}"
        + (f"  no_act_axis(legacy)={counts['no_act_axis']}" if counts["no_act_axis"] else ""),
    ]
    n_pos, n_neg = tp + fn, tn + fp
    mcc = skew_analysis.mcc_from_counts(tp, tn, fp, fn)
    mcc_str = f"{mcc:.3f}" if isinstance(mcc, float) else mcc
    lines += [f"{indent}MCC: {mcc_str}   (balance: n_positive={n_pos} [action-required], n_negative={n_neg} [null])"]
    # MCC does not post-stratify (nonlinear in prevalence); Acc(pi) below carries the prevalence sweep.
    fa = skew_analysis.false_alarm_rate(fp, tn)
    miss = skew_analysis.miss_rate(fn, tp)
    fa_str = f"{fa:.3f} ({fp}/{fp+tn})" if fa is not None else "n/a (no null cases)"
    miss_str = f"{miss:.3f} ({fn}/{fn+tp})" if miss is not None else "n/a (no action-required cases)"
    lines += [f"{indent}False-Alarm Rate (FP/[FP+TN], on oracle-null cases): {fa_str}"]
    lines += [f"{indent}Miss Rate        (FN/[FN+TP], on action-required cases): {miss_str}"]
    return lines


def _format_pi_table(rows: list[dict[str, Any]], value_key: str, label: str, indent: str = "  ") -> list[str]:
    n_null = rows[0]["n_null"] if rows else 0
    n_action = rows[0]["n_action"] if rows else 0
    lines = [f"{indent}{label}  (n_null={n_null}, n_action={n_action})"]
    if n_null == 0 or n_action == 0:
        lines += [f"{indent}  n/a — need at least one run in each stratum"]
        return lines
    header = f"{indent}  pi     {value_key}"
    lines += [header]
    for row in rows:
        val = row[value_key]
        val_str = f"{val:.4f}" if isinstance(val, float) else "n/a"
        lines += [f"{indent}  {row['pi']:.2f}   {val_str}"]
    return lines


def _generate_skew_prevalence_section(
    results_by_arm: dict[str, list[dict[str, Any]]],
    primary_arms: list[str],
) -> list[str]:
    """Task 5: append the SKEW & PREVALENCE ANALYSIS section. Legacy result
    files lacking act_axis/stratum render "n/a"/"no data" rather than 0% or
    crashing (Task 3/4 helpers already guard for zero marginals/empty strata)."""
    lines: list[str] = ["", "=" * 80, "SKEW & PREVALENCE ANALYSIS", "=" * 80]
    from csco.evaluation import bootstrap as _bs

    lines += [
        "(Real-world case-mix is dominated by No-Action-Required 'null' cases; the",
        " agent is stateless so the test mix is a weighting choice, not a behaviour.",
        " Tested stratified (Suite A + Suite B null-region/boundary fixtures),",
        " reported prevalence-weighted below.)",
        "",
    ]

    any_data = False
    for arm in primary_arms:
        scored = _scored_non_infra_runs(results_by_arm.get(arm, []))
        if not scored:
            continue
        any_data = True

        lines += ["", f"ARM: {arm.upper()}", "-" * 80]
        lines += ["Act/No-Act confusion (overall, all suites):"]
        lines += _format_confusion_block(scored)
        if results_by_arm.get("lexical"):
            lines += ["", "Paired ΔMCC (Lexical vs other primary arms):"]
            for other in ("static", "vector"):
                other_runs = results_by_arm.get(other, [])
                if not other_runs:
                    continue
                d = _bs.paired_mcc_difference_ci(results_by_arm["lexical"], other_runs)
                if d["point"] is None:
                    continue
                lines += [
                    f"  Lexical − {other:7s}: ΔMCC={d['point']:+.3f}  "
                    f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
                ]

        suites_present = {skew_analysis.suite_of(r) for r in scored} - {None}
        if len(suites_present) > 1:
            for suite_id in sorted(suites_present):
                suite_runs = [r for r in scored if skew_analysis.suite_of(r) == suite_id]
                lines += ["", f"  -- Suite {suite_id} only --"]
                lines += _format_confusion_block(suite_runs, indent="    ")

        lines += ["", "Acc(pi) = pi*Acc_null + (1-pi)*Acc_action:"]
        lines += [
            "(SDM-3 = overall_correct [primary headline]; SDM-full = SDM-3 AND "
            "permitted_actions set equality — both weighted to the same null "
            "share pi. SDM-full renders n/a where legacy runs predate that key.)",
        ]
        for metric in skew_analysis.ACC_PI_METRICS:
            metric_label = "SDM-3 (overall_correct)" if metric == "overall_correct" else "SDM-full (overall_correct_full)"
            acc_rows = skew_analysis.acc_pi_rows(scored, arm, metric=metric)
            lines += [f"  -- {metric_label} --"]
            lines += _format_pi_table(acc_rows, "acc_pi", "Acc(pi)")

        cost_rows = skew_analysis.cost_pi_rows(scored, arm)
        lines += ["", "cost(pi) = pi*mean_tokens_null + (1-pi)*mean_tokens_action (approx tokens/case):"]
        lines += _format_pi_table(cost_rows, "cost_pi", "cost(pi)")

        pk = skew_analysis.pass_k_by_stratum(scored)
        lines += ["", "pass^k by stratum (k = reps per fixture x arm cell in this run):"]
        for stratum in ("null", "action"):
            info = pk[stratum]
            if info["n_cells"] == 0:
                lines += [f"  {stratum:8s}: no data"]
            else:
                lines += [f"  {stratum:8s}: {info['k']}/{info['n_cells']} cells ({100*info['pass_k']:.1f}%)"]

    if not any_data:
        lines += ["  (no scored runs available for skew/prevalence analysis)"]

    return lines


# ---------------------------------------------------------------------------
# Cluster data collector
# ---------------------------------------------------------------------------

def _collect_cluster_data(
    results_by_arm: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict]]]:
    """Group runs by diagnostic cluster using oracle metadata.

    Returns {cluster_id: {arm: [run_dict, ...]}}
    """
    from csco.oracle.manual import get_oracle
    from csco.specs.loader import load_all_specs

    # Build scenario → cluster mapping from spec evaluation metadata
    specs = load_all_specs()
    scenario_clusters: dict[str, list[str]] = {}

    for dtype, spec in specs.items():
        all_rules = list(spec.routing_rules) + list(spec.supplier_rules)
        rule_clusters: dict[str, list[str]] = {}
        for rule in all_rules:
            eval_meta = getattr(rule, "evaluation", None)
            rule_clusters[rule.rule_id] = eval_meta.clusters if eval_meta else []

    # Resolve each run's scenario to its oracle and map to the fired rule's clusters.
    seen_ids = {r["scenario_id"] for runs in results_by_arm.values() for r in runs}
    for scenario_id in seen_ids:
        oracle = get_oracle(scenario_id)
        if oracle is None:
            continue
        fired_rules = []
        if oracle.routing:
            fired_rules.append(oracle.routing.fired_rule_id)
        for s in oracle.suppliers:
            fired_rules.append(s.fired_rule_id)

        clusters: set[str] = set()
        for dtype, spec in specs.items():
            if spec.disruption_type != oracle.disruption_type:
                continue
            for rule_id in fired_rules:
                rule_obj = spec.get_rule(rule_id)
                if rule_obj:
                    eval_meta = getattr(rule_obj, "evaluation", None)
                    if eval_meta:
                        clusters.update(eval_meta.clusters)
        scenario_clusters[scenario_id] = sorted(clusters)

    # Group runs by cluster
    cluster_data: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for arm, runs in results_by_arm.items():
        for r in runs:
            sid = r["scenario_id"]
            for cluster_id in scenario_clusters.get(sid, []):
                cluster_data[cluster_id][arm].append(r)

    return dict(cluster_data)
