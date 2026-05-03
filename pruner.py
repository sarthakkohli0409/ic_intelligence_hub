"""
ProcDNA Intelligence — Pruning Engine v2
==========================================
Reduces scored scenarios from up to 186 → 15-25 curated results.

Steps (in order):
  1. Structural filter   — remove extreme splits / weight combos
  2. Dominance filter    — Pareto dominance: drop B if A >= B on all metrics & A > B on one
  3. Similarity cluster  — L1 distance < ε → keep higher-ranked only
  4. Top-25 selection    — final cap with diversity guarantee

"The One" selection (Section VII):
  1. Build Pareto frontier across all 7 metrics
  2. Rank within frontier by preset weighted score
  3. Tie-breakers: confidence → cost variance → structural simplicity
"""

import numpy as np
from typing import List, Dict, Any, Tuple

METRIC_KEYS   = ["P", "A", "C", "Ga", "V", "K", "S"]
EPSILON       = 0.05   # similarity threshold
TOP_K         = 25     # max scenarios kept after pruning
PARETO_MAX    = 50     # frontier search pool


# ── Structural filter ─────────────────────────────────────────────────────────
def structural_filter(results: List[Dict]) -> List[Dict]:
    """
    Remove scenarios with structurally extreme parameters:
      - Any individual split > 95% (degenerate)
      - Quarter weights where any single quarter > 90%
    """
    def ok(r):
        sc = r["scenario"]
        # Degenerate split check
        if sc["rs"] > 0.95 or sc["is_"] > 0.95:
            return False
        # Extreme quarter weight check (only applies to quarter-wise)
        if sc["k3"] > 0:
            if max(sc["w1"], sc["w2"], sc["w3"]) > 0.90:
                return False
        return True
    kept    = [r for r in results if ok(r)]
    dropped = len(results) - len(kept)
    return kept, dropped


# ── Dominance filter ──────────────────────────────────────────────────────────
def _dominates(a_metrics: Dict, b_metrics: Dict) -> bool:
    """
    A dominates B if A >= B on every metric AND A > B on at least one.
    """
    ge_all = all(a_metrics.get(k, 0) >= b_metrics.get(k, 0) for k in METRIC_KEYS)
    gt_one = any(a_metrics.get(k, 0) >  b_metrics.get(k, 0) for k in METRIC_KEYS)
    return ge_all and gt_one


def dominance_filter(results: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Remove dominated scenarios.
    O(n²) — fine for n ≤ 186.
    """
    n = len(results)
    dominated = set()
    metrics   = [r["fairness"] for r in results]

    for i in range(n):
        if i in dominated:
            continue
        for j in range(n):
            if i == j or j in dominated:
                continue
            if _dominates(metrics[i], metrics[j]):
                dominated.add(j)

    kept    = [r for i, r in enumerate(results) if i not in dominated]
    dropped = len(dominated)
    return kept, dropped


# ── Similarity cluster ────────────────────────────────────────────────────────
def similarity_filter(results: List[Dict], epsilon: float = EPSILON) -> Tuple[List[Dict], int]:
    """
    For any two scenarios with L1 metric distance < epsilon,
    keep only the one with the higher boosted_score.
    Results must be pre-sorted descending by boosted_score.
    """
    kept_idx = []
    dropped  = 0

    for i, r in enumerate(results):
        mv_i = r["fairness"]
        too_close = False
        for j in kept_idx:
            mv_j = results[j]["fairness"]
            l1 = sum(abs(mv_i.get(k, 0) - mv_j.get(k, 0)) for k in METRIC_KEYS)
            if l1 < epsilon:
                too_close = True
                break
        if too_close:
            dropped += 1
        else:
            kept_idx.append(i)

    return [results[i] for i in kept_idx], dropped


# ── Pareto frontier ───────────────────────────────────────────────────────────
def pareto_frontier(results: List[Dict]) -> List[Dict]:
    """
    Return the non-dominated subset of results.
    Works on the top PARETO_MAX by boosted_score to keep it fast.
    """
    pool    = results[:PARETO_MAX]
    n       = len(pool)
    metrics = [r["fairness"] for r in pool]
    non_dom = []

    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i == j:
                continue
            if _dominates(metrics[j], metrics[i]):
                is_dominated = True
                break
        if not is_dominated:
            non_dom.append(pool[i])

    return non_dom if non_dom else pool[:1]


# ── Structural simplicity score ───────────────────────────────────────────────
def simplicity_score(scenario: Dict) -> float:
    """
    Simpler = better tie-breaker.
    Historical Total (k3=0) > Quarter-wise (active weights).
    Pure splits (Regional/Individual) < Hybrid.
    Score ranges 0-1 where 1 = simplest.
    """
    score = 1.0
    if scenario["k3"] > 0:
        score -= 0.3            # quarter-wise adds complexity
        # Penalise uneven weights
        ws = sorted([scenario["w1"], scenario["w2"], scenario["w3"]], reverse=True)
        score -= max(0, (ws[0] - ws[2]) - 0.2) * 0.5
    if "hybrid" in scenario["mode"].lower():
        score -= 0.1            # hybrid slightly more complex than pure
    return round(max(0.0, score), 4)


# ── Cost variance (tie-breaker) ───────────────────────────────────────────────
def cost_variance(goals: List[Dict], national_forecast: float) -> float:
    """Lower = less budget variance = preferred in tie-break."""
    total = sum(g["fg"] for g in goals)
    return abs(total - national_forecast) / max(national_forecast, 1)


# ── "The One" selection ───────────────────────────────────────────────────────
def select_the_one(results: List[Dict], national_forecast: float) -> Dict:
    """
    Full 3-step Pareto selection:
      1. Pareto frontier from top-50 by boosted_score
      2. Rank within frontier by boosted_score
      3. Tie-break: confidence → cost_variance → simplicity
    """
    frontier = pareto_frontier(results)

    def sort_key(r):
        conf = r["fairness"].get("confidence", 0.5)
        cv   = cost_variance(r["goals"], national_forecast)
        simp = simplicity_score(r["scenario"])
        return (-r["fairness"]["boosted_score"], -conf, cv, -simp)

    frontier.sort(key=sort_key)
    return frontier[0]


# ── Master prune pipeline ─────────────────────────────────────────────────────
def prune(results: List[Dict], national_forecast: float) -> Tuple[List[Dict], Dict, str]:
    """
    Run full pruning pipeline.
    Returns: (pruned_results, pruning_stats, best_scenario_id)

    pruned_results is sorted descending by boosted_score.
    best_scenario_id is set via Pareto selection (may differ from rank 1).
    """
    # Sort by boosted score first
    results = sorted(results, key=lambda r: r["fairness"]["boosted_score"], reverse=True)

    n0 = len(results)

    # Step 1: structural
    results, d_struct = structural_filter(results)
    n1 = len(results)

    # Step 2: dominance
    results, d_dom = dominance_filter(results)
    n2 = len(results)

    # Re-sort after dominance filter
    results = sorted(results, key=lambda r: r["fairness"]["boosted_score"], reverse=True)

    # Step 3: similarity
    results, d_sim = similarity_filter(results)
    n3 = len(results)

    # Step 4: Top-K
    results = results[:TOP_K]
    n4 = len(results)

    # Assign ranks
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["is_best"] = False

    # Step 5: Pareto-based "The One"
    the_one = select_the_one(results, national_forecast)
    for r in results:
        r["is_best"] = (r["scenario"]["id"] == the_one["scenario"]["id"])

    stats = {
        "total_before_pruning":  n0,
        "after_structural":      n1,
        "after_dominance":       n2,
        "after_similarity":      n3,
        "final_count":           n4,
        "dropped_structural":    d_struct,
        "dropped_dominance":     d_dom,
        "dropped_similarity":    d_sim,
    }

    return results, stats, the_one["scenario"]["id"]
