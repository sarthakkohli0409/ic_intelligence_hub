"""
ProcDNA Intelligence — Tool Functions
======================================
All 35 functions Claude can call via tool use.
Each function takes session_id (+ optional params) and returns a dict.
The dict is sent back to Claude as the tool result.

Groups:
  Group 1  — Data retrieval          (15 functions)
  Group 2  — Comparison and delta    (8 functions)
  Group 3  — Guardrails              (6 functions)
  Group 4  — What-if and re-scoring  (7 functions)  partial — needs session territories
  Group 5  — File and upload info    (5 functions)
  Group 6  — Export and output       (4 functions)
  Group 7  — IC Advisory             (2 functions)
"""

import numpy as np
from typing import Any, Dict, List, Optional

# Sessions dict is imported from main at runtime to avoid circular import
# Each function receives it as a parameter

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _get_sess(sessions: dict, session_id: str) -> dict:
    s = sessions.get(session_id)
    if not s:
        raise ValueError(f"Session '{session_id}' not found. Run Goal Setting first.")
    return s

def _get_best(sessions: dict, session_id: str) -> dict:
    sess = _get_sess(sessions, session_id)
    best = next((r for r in sess["results"] if r["is_best"]), None)
    if not best:
        raise ValueError("No best scenario found in session.")
    return best

def _get_scenario(sessions: dict, session_id: str, scenario_id: str) -> dict:
    sess = _get_sess(sessions, session_id)
    sc = next((r for r in sess["results"] if r["scenario"]["id"] == scenario_id), None)
    if not sc:
        raise ValueError(f"Scenario '{scenario_id}' not found.")
    return sc


# ═════════════════════════════════════════════════════════════════
# GROUP 1 — Data Retrieval
# ═════════════════════════════════════════════════════════════════

def get_territory_goals(
    sessions: dict, session_id: str,
    scenario_id: str = None,
    region_filter: str = None,
) -> Dict[str, Any]:
    """Quarterly sales history and final goals for every territory."""
    sess = _get_sess(sessions, session_id)
    target_id = scenario_id or sess["best_id"]
    result = _get_scenario(sessions, session_id, target_id)
    territories = {t["territory_id"]: t for t in sess.get("territories", [])}

    rows = []
    for g in result["goals"]:
        t = territories.get(g["tid"], {})
        row = {
            "territory_id":   g["tid"],
            "territory_name": g["tname"],
            "region_id":      g["rid"],
            "region_name":    g["rname"],
            "q1_sales":       round(t.get("q1", 0), 2),
            "q2_sales":       round(t.get("q2", 0), 2),
            "q3_sales":       round(t.get("q3", 0), 2),
            "q4_sales":       round(t.get("q4", 0), 2),
            "final_goal":     g["fg"],
            "forecast_fi":    g["fc"],
            "attainment_pct": round(g["att"] * 100, 1),
            "on_target":      0.85 <= g["att"] <= 1.15,
        }
        if region_filter and g["rid"] != region_filter:
            continue
        rows.append(row)

    return {
        "scenario_id":     target_id,
        "territory_count": len(rows),
        "territories":     rows,
    }


def get_territory_sales(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Raw quarterly sales history from the uploaded file."""
    sess = _get_sess(sessions, session_id)
    territories = sess.get("territories", [])
    if not territories:
        return {"error": "Territory data not in session. Re-run Goal Setting."}
    rows = [{
        "territory_id":   t["territory_id"],
        "territory_name": t["territory_name"],
        "region_id":      t["region_id"],
        "region_name":    t["region_name"],
        "q1_sales":       round(t.get("q1", 0), 2),
        "q2_sales":       round(t.get("q2", 0), 2),
        "q3_sales":       round(t.get("q3", 0), 2),
        "q4_sales":       round(t.get("q4", 0), 2),
        "hist_12m":       round(t.get("h12", 0), 2),
        "trend_score":    round(t.get("trend", 0), 4),
        "volatility":     t.get("vband", "—"),
        "data_tier":      t.get("data_tier", "—"),
        "data_months":    t.get("data_months", 0),
    } for t in territories]
    return {"territory_count": len(rows), "territories": rows}


def get_best_scenario(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Full details of the best (Pareto-selected) scenario."""
    best = _get_best(sessions, session_id)
    sess = _get_sess(sessions, session_id)
    f = best["fairness"]
    return {
        "scenario_id":  best["scenario"]["id"],
        "method":       best["scenario"]["method"],
        "mode":         best["scenario"]["mode"],
        "w1":           best["scenario"]["w1"],
        "w2":           best["scenario"]["w2"],
        "w3":           best["scenario"]["w3"],
        "composite":    f["boosted_score"],
        "confidence":   f.get("confidence", "—"),
        "on_target":    f["on_target"],
        "above_target": f["above"],
        "below_target": f["below"],
        "total_goals":  f["total_goals"],
        "forecast_gap": f["gap"],
        "preset_used":  sess.get("preset", "fairness"),
        "metrics": {
            "P":  f["P"],  "A": f["A"],  "C": f["C"],
            "Ga": f["Ga"], "V": f["V"],  "K": f["K"], "S": f["S"],
        },
    }


def get_top_scenarios(
    sessions: dict, session_id: str,
    n: int = 5,
) -> Dict[str, Any]:
    """Top N scenarios ranked by composite score."""
    sess = _get_sess(sessions, session_id)
    top = sess["results"][:min(n, len(sess["results"]))]
    return {
        "count": len(top),
        "scenarios": [{
            "rank":        r["rank"],
            "scenario_id": r["scenario"]["id"],
            "method":      r["scenario"]["method"],
            "mode":        r["scenario"]["mode"],
            "composite":   r["fairness"]["boosted_score"],
            "confidence":  r["fairness"].get("confidence", "—"),
            "on_target_pct": round(r["fairness"]["pct_on"] * 100, 1),
            "total_goals": r["fairness"]["total_goals"],
            "is_best":     r["is_best"],
            "is_preferred":r.get("is_pref", False),
        } for r in top],
    }


def get_fairness_breakdown(
    sessions: dict, session_id: str,
    scenario_id: str = None,
) -> Dict[str, Any]:
    """All 7 metric scores for a scenario with explanations."""
    sess  = _get_sess(sessions, session_id)
    target = scenario_id or sess["best_id"]
    r = _get_scenario(sessions, session_id, target)
    f = r["fairness"]
    EXPLAIN = {
        "P":  "Proportionality — how well goals match territory potential (Pearson r).",
        "A":  "Attainability — % territories where forecast/goal is within ±15%.",
        "C":  "Consistency — how uniform goals are within each region (1 - CV).",
        "Ga": "Growth Alignment — trending territories get proportionally higher goals.",
        "V":  "Volatility Penalty — high-volatility territories have conservative goals.",
        "K":  "Cost Efficiency — total goals vs national forecast budget.",
        "S":  "Goal Spread — P90/P10 ratio. Punishes extreme inequality between territories.",
    }
    return {
        "scenario_id": target,
        "composite":   f["boosted_score"],
        "preset":      f.get("preset", "fairness"),
        "confidence":  f.get("confidence", "—"),
        "metrics": {k: {"score": f[k], "explanation": EXPLAIN[k]} for k in EXPLAIN},
    }


def get_scenario_metrics(
    sessions: dict, session_id: str,
    scenario_id: str,
) -> Dict[str, Any]:
    """All 7 metric values for a specific scenario."""
    return get_fairness_breakdown(sessions, session_id, scenario_id)


def get_territories_by_attainment(
    sessions: dict, session_id: str,
    filter: str = "below",
    scenario_id: str = None,
) -> Dict[str, Any]:
    """
    Filter territories by attainment status.
    filter: 'below' (<85%), 'above' (>115%), 'on_target' (85-115%)
    """
    sess = _get_sess(sessions, session_id)
    target = scenario_id or sess["best_id"]
    result = _get_scenario(sessions, session_id, target)

    def flt(g):
        if filter == "below":    return g["att"] < 0.85
        if filter == "above":    return g["att"] > 1.15
        return 0.85 <= g["att"] <= 1.15

    rows = [{
        "territory_id":   g["tid"],
        "territory_name": g["tname"],
        "region_name":    g["rname"],
        "final_goal":     g["fg"],
        "forecast":       g["fc"],
        "attainment_pct": round(g["att"] * 100, 1),
    } for g in result["goals"] if flt(g)]

    return {
        "scenario_id": target,
        "filter":      filter,
        "count":       len(rows),
        "territories": rows,
    }


def get_forecast_gap(
    sessions: dict, session_id: str,
    scenario_id: str = None,
) -> Dict[str, Any]:
    """Total goals vs national forecast — the budget gap."""
    sess   = _get_sess(sessions, session_id)
    target = scenario_id or sess["best_id"]
    r      = _get_scenario(sessions, session_id, target)
    f      = r["fairness"]
    nf     = sess["nf"]
    gap    = f["gap"]
    pct    = round(gap / nf * 100, 2) if nf else 0
    return {
        "scenario_id":      target,
        "national_forecast": nf,
        "total_goals":       f["total_goals"],
        "gap":               gap,
        "gap_pct":           pct,
        "status": "over budget" if gap > 0 else "under budget" if gap < 0 else "exactly on budget",
    }


def get_forecast_detail(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Forecast model and confidence per territory."""
    sess = _get_sess(sessions, session_id)
    territories = sess.get("territories", [])
    rows = [{
        "territory_id":   t["territory_id"],
        "territory_name": t["territory_name"],
        "best_model":     t.get("best_model", "—"),
        "wmape":          t.get("wmape", "—"),
        "confidence_pct": round(t.get("confidence", 0.5) * 100, 1),
        "forecast_q1":    t.get("forecast_q1", "—"),
        "data_tier":      t.get("data_tier", "—"),
        "data_months":    t.get("data_months", 0),
    } for t in territories]
    avg_conf = round(float(np.mean([t.get("confidence", 0.5) for t in territories])) * 100, 1) if territories else 0
    return {
        "territory_count":   len(rows),
        "avg_confidence_pct": avg_conf,
        "territories":       rows,
    }


def get_model_wins(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Which forecast model won for the most territories."""
    sess = _get_sess(sessions, session_id)
    fc   = sess.get("forecast_summary", {})
    wins = fc.get("model_wins", {})
    total = sum(wins.values()) if wins else 0
    ranked = sorted(wins.items(), key=lambda x: -x[1])
    return {
        "top_model":   fc.get("top_model", "—"),
        "avg_wmape":   fc.get("avg_wmape", "—"),
        "avg_confidence_pct": round(fc.get("avg_confidence", 0.5) * 100, 1),
        "total_territories": total,
        "model_wins": [{"model": k, "territories": v,
                        "share_pct": round(v/total*100, 1) if total else 0}
                       for k, v in ranked],
    }


def get_data_tier(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Data tier and month counts for the uploaded file."""
    sess = _get_sess(sessions, session_id)
    tc   = sess.get("tier_counts", {})
    return {
        "overall_tier":    sess.get("overall_tier", "full"),
        "limited_count":   tc.get("limited", 0),
        "moderate_count":  tc.get("moderate", 0),
        "full_count":      tc.get("full", 0),
        "advice": (
            "Data is sufficient for all 15 models." if sess.get("overall_tier") == "full"
            else "Some territories have limited history — confidence scores are capped."
        ),
    }


def get_pruning_stats(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """How many scenarios were pruned and why."""
    sess = _get_sess(sessions, session_id)
    ps   = sess.get("prune_stats", {})
    return {
        "total_before_pruning": ps.get("total_before_pruning", "—"),
        "dropped_structural":   ps.get("dropped_structural", 0),
        "dropped_dominance":    ps.get("dropped_dominance", 0),
        "dropped_similarity":   ps.get("dropped_similarity", 0),
        "final_count":          ps.get("final_count", "—"),
        "explanation": (
            f"Started with {ps.get('total_before_pruning','?')} valid scenarios. "
            f"Removed {ps.get('dropped_structural',0)} structurally extreme, "
            f"{ps.get('dropped_dominance',0)} dominated, "
            f"{ps.get('dropped_similarity',0)} near-identical. "
            f"Final: {ps.get('final_count','?')} curated scenarios."
        ),
    }


def get_preferred_scenario(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """The scenario that best matches the user's stated preferences."""
    sess = _get_sess(sessions, session_id)
    pref = next((r for r in sess["results"] if r.get("is_pref")), None)
    if not pref:
        return {"message": "No preferred scenario found. The best scenario is being used."}
    f = pref["fairness"]
    return {
        "scenario_id": pref["scenario"]["id"],
        "method":      pref["scenario"]["method"],
        "mode":        pref["scenario"]["mode"],
        "composite":   f["boosted_score"],
        "on_target":   f["on_target"],
        "total_goals": f["total_goals"],
        "vs_best":     "Same as best" if pref["is_best"] else "Different from best",
    }


def get_attainment_summary(
    sessions: dict, session_id: str,
    scenario_id: str = None,
) -> Dict[str, Any]:
    """Summary of on-target, above, and below territories."""
    sess   = _get_sess(sessions, session_id)
    target = scenario_id or sess["best_id"]
    r      = _get_scenario(sessions, session_id, target)
    f      = r["fairness"]
    n      = len(r["goals"])
    return {
        "scenario_id":      target,
        "total_territories": n,
        "on_target":        f["on_target"],
        "above_target":     f["above"],
        "below_target":     f["below"],
        "on_target_pct":    round(f["pct_on"] * 100, 1),
        "above_pct":        round(f["pct_ab"] * 100, 1),
        "below_pct":        round(f["pct_bl"] * 100, 1),
        "attainment_band":  "±15% of forecast",
    }


# ═════════════════════════════════════════════════════════════════
# GROUP 2 — Comparison and Delta
# ═════════════════════════════════════════════════════════════════

def get_delta(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Delta between best and preferred scenario."""
    sess = _get_sess(sessions, session_id)
    best = next((r for r in sess["results"] if r["is_best"]), None)
    pref = next((r for r in sess["results"] if r.get("is_pref")), None)
    if not best:
        return {"error": "No best scenario found."}
    if not pref or pref["scenario"]["id"] == best["scenario"]["id"]:
        return {"message": "Best and preferred are the same scenario — no delta to compute."}
    bf, pf = best["fairness"], pref["fairness"]
    return {
        "best_id":               best["scenario"]["id"],
        "preferred_id":          pref["scenario"]["id"],
        "attainability_shift_pp": round((pf["A"] - bf["A"]) * 100, 1),
        "fairness_shift":         round(pf["boosted_score"] - bf["boosted_score"], 4),
        "total_goals_shift":      round(pf["total_goals"] - bf["total_goals"], 2),
        "consistency_shift_pp":   round((pf["C"] - bf["C"]) * 100, 1),
        "growth_alignment_shift_pp": round((pf["Ga"] - bf["Ga"]) * 100, 1),
        "cost_efficiency_shift":  round(pf["K"] - bf["K"], 4),
    }


def compare_two_scenarios(
    sessions: dict, session_id: str,
    scenario_a: str,
    scenario_b: str,
) -> Dict[str, Any]:
    """Side-by-side metric comparison of any two scenarios."""
    ra = _get_scenario(sessions, session_id, scenario_a)
    rb = _get_scenario(sessions, session_id, scenario_b)
    fa, fb = ra["fairness"], rb["fairness"]
    METRICS = ["P", "A", "C", "Ga", "V", "K", "S"]
    comparison = {}
    for m in METRICS:
        va, vb = fa.get(m, 0), fb.get(m, 0)
        comparison[m] = {
            scenario_a: round(va, 4),
            scenario_b: round(vb, 4),
            "winner":   scenario_a if va > vb else scenario_b if vb > va else "tie",
            "diff":     round(va - vb, 4),
        }
    return {
        "scenario_a":   scenario_a,
        "scenario_b":   scenario_b,
        "composite_a":  fa["boosted_score"],
        "composite_b":  fb["boosted_score"],
        "overall_winner": scenario_a if fa["boosted_score"] > fb["boosted_score"] else scenario_b,
        "metrics":      comparison,
    }


def rank_scenarios_by_metric(
    sessions: dict, session_id: str,
    metric: str = "A",
) -> Dict[str, Any]:
    """Rank all scenarios by a specific metric."""
    sess    = _get_sess(sessions, session_id)
    VALID   = {"P","A","C","Ga","V","K","S","composite"}
    if metric not in VALID:
        return {"error": f"Invalid metric '{metric}'. Choose from: {sorted(VALID)}"}
    results = sess["results"]
    def score(r):
        return r["fairness"]["boosted_score"] if metric == "composite" else r["fairness"].get(metric, 0)
    ranked = sorted(results, key=score, reverse=True)
    return {
        "ranked_by": metric,
        "scenarios": [{
            "rank":        i + 1,
            "scenario_id": r["scenario"]["id"],
            "method":      r["scenario"]["method"],
            "mode":        r["scenario"]["mode"],
            metric:        round(score(r), 4),
            "is_best":     r["is_best"],
        } for i, r in enumerate(ranked)],
    }


def compare_territories(
    sessions: dict, session_id: str,
    scenario_a: str,
    scenario_b: str,
) -> Dict[str, Any]:
    """Per-territory goal diff between two scenarios."""
    ra = _get_scenario(sessions, session_id, scenario_a)
    rb = _get_scenario(sessions, session_id, scenario_b)
    ma = {g["tid"]: g for g in ra["goals"]}
    mb = {g["tid"]: g for g in rb["goals"]}
    rows = []
    for tid, ga in ma.items():
        gb = mb.get(tid)
        if not gb:
            continue
        diff     = round(gb["fg"] - ga["fg"], 2)
        diff_pct = round(diff / ga["fg"] * 100, 1) if ga["fg"] else 0
        rows.append({
            "territory_id":   tid,
            "territory_name": ga["tname"],
            "region_name":    ga["rname"],
            f"goal_{scenario_a}": ga["fg"],
            f"goal_{scenario_b}": gb["fg"],
            "difference":     diff,
            "difference_pct": diff_pct,
            "better_for_territory": scenario_b if abs(gb["att"] - 1.0) < abs(ga["att"] - 1.0) else scenario_a,
        })
    return {
        "scenario_a":      scenario_a,
        "scenario_b":      scenario_b,
        "territory_count": len(rows),
        "territories":     rows,
    }


def get_territory_winners(
    sessions: dict, session_id: str,
    scenario_id: str = None,
) -> Dict[str, Any]:
    """Territories closest to 100% attainment — the 'winners'."""
    sess   = _get_sess(sessions, session_id)
    target = scenario_id or sess["best_id"]
    result = _get_scenario(sessions, session_id, target)
    scored = sorted(
        result["goals"],
        key=lambda g: abs(g["att"] - 1.0)
    )
    return {
        "scenario_id": target,
        "top_territories": [{
            "territory_id":   g["tid"],
            "territory_name": g["tname"],
            "region_name":    g["rname"],
            "attainment_pct": round(g["att"] * 100, 1),
            "final_goal":     g["fg"],
        } for g in scored[:10]],
    }


def rescore_with_preset(
    sessions: dict, session_id: str,
    preset: str,
) -> Dict[str, Any]:
    """Re-rank existing scenarios using a different preset weight vector."""
    from scoring import composite_score, PRESETS
    sess = _get_sess(sessions, session_id)
    if preset not in PRESETS:
        return {"error": f"Unknown preset '{preset}'. Choose: fairness, growth, cost"}
    results = sess["results"]
    rescored = []
    for r in results:
        new_score = composite_score(r["fairness"], preset)
        rescored.append({
            "scenario_id": r["scenario"]["id"],
            "method":      r["scenario"]["method"],
            "mode":        r["scenario"]["mode"],
            "new_score":   new_score,
            "old_score":   r["fairness"]["boosted_score"],
            "change":      round(new_score - r["fairness"]["boosted_score"], 4),
        })
    rescored.sort(key=lambda x: -x["new_score"])
    for i, r in enumerate(rescored):
        r["new_rank"] = i + 1
    return {
        "preset":    preset,
        "preset_label": PRESETS[preset]["label"],
        "new_best":  rescored[0]["scenario_id"] if rescored else "—",
        "scenarios": rescored[:10],
    }


def rescore_with_mode(
    sessions: dict, session_id: str,
    plan_mode: str,
) -> Dict[str, Any]:
    """Show which existing scenarios match a given plan mode."""
    sess = _get_sess(sessions, session_id)
    matching = [
        r for r in sess["results"]
        if plan_mode.lower() in r["scenario"]["mode"].lower()
    ]
    if not matching:
        return {"message": f"No scenarios found with plan mode containing '{plan_mode}'."}
    return {
        "plan_mode":  plan_mode,
        "count":      len(matching),
        "scenarios":  [{
            "rank":        r["rank"],
            "scenario_id": r["scenario"]["id"],
            "mode":        r["scenario"]["mode"],
            "composite":   r["fairness"]["boosted_score"],
            "on_target_pct": round(r["fairness"]["pct_on"] * 100, 1),
        } for r in matching[:5]],
    }


# ═════════════════════════════════════════════════════════════════
# GROUP 3 — Guardrails and Validation
# ═════════════════════════════════════════════════════════════════

def run_guardrails(
    sessions: dict, session_id: str,
    scenario_id: str = None,
) -> Dict[str, Any]:
    """Run all 4 guardrail rules on best or specified scenario."""
    from export import guardrails_check
    sess   = _get_sess(sessions, session_id)
    target = scenario_id or sess["best_id"]
    return guardrails_check(sess["results"], sess["nf"], target)


def check_goal_ceiling(
    sessions: dict, session_id: str,
    scenario_id: str = None,
) -> Dict[str, Any]:
    """Find territories with goals above 200% of the average."""
    sess   = _get_sess(sessions, session_id)
    target = scenario_id or sess["best_id"]
    result = _get_scenario(sessions, session_id, target)
    goals  = result["goals"]
    avg    = float(np.mean([g["fg"] for g in goals])) if goals else 1
    high   = [g for g in goals if g["fg"] > avg * 2.0]
    return {
        "scenario_id":   target,
        "average_goal":  round(avg, 2),
        "ceiling":       round(avg * 2.0, 2),
        "flagged_count": len(high),
        "flagged": [{
            "territory_id":   g["tid"],
            "territory_name": g["tname"],
            "final_goal":     g["fg"],
            "vs_average_pct": round(g["fg"] / avg * 100, 1),
        } for g in high],
    }


def check_goal_floor(
    sessions: dict, session_id: str,
    scenario_id: str = None,
) -> Dict[str, Any]:
    """Find territories with goals below 50% of the average."""
    sess   = _get_sess(sessions, session_id)
    target = scenario_id or sess["best_id"]
    result = _get_scenario(sessions, session_id, target)
    goals  = result["goals"]
    avg    = float(np.mean([g["fg"] for g in goals])) if goals else 1
    low    = [g for g in goals if g["fg"] < avg * 0.5]
    return {
        "scenario_id":   target,
        "average_goal":  round(avg, 2),
        "floor":         round(avg * 0.5, 2),
        "flagged_count": len(low),
        "flagged": [{
            "territory_id":   g["tid"],
            "territory_name": g["tname"],
            "final_goal":     g["fg"],
            "vs_average_pct": round(g["fg"] / avg * 100, 1),
        } for g in low],
    }


def check_budget_alignment(
    sessions: dict, session_id: str,
    scenario_id: str = None,
) -> Dict[str, Any]:
    """Check whether total goals are within ±20% of national forecast."""
    return get_forecast_gap(sessions, session_id, scenario_id)


def get_flagged_territories(
    sessions: dict, session_id: str,
    scenario_id: str = None,
) -> Dict[str, Any]:
    """High-volatility territories that are also below attainment target."""
    sess   = _get_sess(sessions, session_id)
    target = scenario_id or sess["best_id"]
    result = _get_scenario(sessions, session_id, target)
    terr   = {t["territory_id"]: t for t in sess.get("territories", [])}
    flagged = []
    for g in result["goals"]:
        t = terr.get(g["tid"], {})
        if t.get("vband") == "High Volatility" and g["att"] < 0.85:
            flagged.append({
                "territory_id":   g["tid"],
                "territory_name": g["tname"],
                "region_name":    g["rname"],
                "volatility":     t.get("vband", "—"),
                "attainment_pct": round(g["att"] * 100, 1),
                "final_goal":     g["fg"],
                "forecast":       g["fc"],
                "recommendation": "Consider reducing goal or applying smoothing factor.",
            })
    return {
        "scenario_id":   target,
        "flagged_count": len(flagged),
        "flagged":       flagged,
    }


# ═════════════════════════════════════════════════════════════════
# GROUP 4 — What-if and Re-scoring
# ═════════════════════════════════════════════════════════════════

def get_conservative_scenario(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """The scenario with the lowest total goals in the top 10."""
    sess = _get_sess(sessions, session_id)
    pool = sess["results"][:10]
    sc   = min(pool, key=lambda r: r["fairness"]["total_goals"])
    f    = sc["fairness"]
    return {
        "label":       "Conservative scenario (lowest total goals)",
        "scenario_id": sc["scenario"]["id"],
        "method":      sc["scenario"]["method"],
        "mode":        sc["scenario"]["mode"],
        "total_goals": f["total_goals"],
        "composite":   f["boosted_score"],
        "on_target_pct": round(f["pct_on"] * 100, 1),
        "vs_best_goals": round(f["total_goals"] - sess["results"][0]["fairness"]["total_goals"], 2),
    }


def get_aggressive_scenario(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """The scenario with the highest total goals in the top 10."""
    sess = _get_sess(sessions, session_id)
    pool = sess["results"][:10]
    sc   = max(pool, key=lambda r: r["fairness"]["total_goals"])
    f    = sc["fairness"]
    return {
        "label":       "Aggressive scenario (highest total goals)",
        "scenario_id": sc["scenario"]["id"],
        "method":      sc["scenario"]["method"],
        "mode":        sc["scenario"]["mode"],
        "total_goals": f["total_goals"],
        "composite":   f["boosted_score"],
        "on_target_pct": round(f["pct_on"] * 100, 1),
        "vs_best_goals": round(f["total_goals"] - sess["results"][0]["fairness"]["total_goals"], 2),
    }


def rerun_with_forecast(
    sessions: dict, session_id: str,
    new_national_forecast: float,
) -> Dict[str, Any]:
    """
    Recalculate goals and re-score all scenarios with a new national forecast.
    Returns new top 5 and new best scenario.
    """
    from scenario_engine import calc_goals
    from scoring import build_fairness_block
    from pruner import prune

    sess = _get_sess(sessions, session_id)
    territories = sess.get("territories", [])
    if not territories:
        return {"error": "Territory data not in session. Re-run Goal Setting."}

    confidences = {t["territory_id"]: t.get("confidence", 0.5) for t in territories}
    feat_map    = {t["territory_id"]: t for t in territories}
    preset      = sess.get("preset", "fairness")

    new_results = []
    for old_r in sess["results"]:
        sc     = old_r["scenario"]
        goals  = calc_goals(sc, territories, new_national_forecast)
        fair   = build_fairness_block(
            sid=sc["id"], goals=goals, feat=feat_map,
            national_forecast=new_national_forecast,
            confidences=confidences, preset=preset,
            ic_plan_type="", scenario_mode=sc["mode"],
        )
        new_results.append({
            "scenario": sc, "goals": goals, "fairness": fair,
            "is_best": False, "is_pref": old_r.get("is_pref", False), "rank": 0,
        })

    pruned, _, new_best_id = prune(new_results, new_national_forecast)
    # Update session
    sess["results"]    = pruned
    sess["best_id"]    = new_best_id
    sess["nf"]         = new_national_forecast

    best = next(r for r in pruned if r["scenario"]["id"] == new_best_id)
    return {
        "new_national_forecast": new_national_forecast,
        "new_best_scenario_id":  new_best_id,
        "new_composite":         best["fairness"]["boosted_score"],
        "new_total_goals":       best["fairness"]["total_goals"],
        "new_gap":               best["fairness"]["gap"],
        "top5": [{
            "rank":        r["rank"],
            "scenario_id": r["scenario"]["id"],
            "composite":   r["fairness"]["boosted_score"],
            "total_goals": r["fairness"]["total_goals"],
        } for r in pruned[:5]],
    }


def rerun_with_method(
    sessions: dict, session_id: str,
    scoring_method: str,
) -> Dict[str, Any]:
    """Filter to scenarios matching a scoring method and show top results."""
    sess = _get_sess(sessions, session_id)
    matching = [
        r for r in sess["results"]
        if scoring_method.lower() in r["scenario"]["method"].lower()
    ]
    if not matching:
        return {"message": f"No scenarios with method containing '{scoring_method}'."}
    best = matching[0]
    return {
        "scoring_method": scoring_method,
        "matching_count": len(matching),
        "best_match": {
            "scenario_id": best["scenario"]["id"],
            "mode":        best["scenario"]["mode"],
            "composite":   best["fairness"]["boosted_score"],
            "total_goals": best["fairness"]["total_goals"],
            "on_target_pct": round(best["fairness"]["pct_on"] * 100, 1),
        },
        "all_matches": [{
            "scenario_id": r["scenario"]["id"],
            "mode":        r["scenario"]["mode"],
            "composite":   r["fairness"]["boosted_score"],
        } for r in matching[:5]],
    }


def rerun_with_weights(
    sessions: dict, session_id: str,
    min_quarter_weight: float,
    max_quarter_weight: float,
) -> Dict[str, Any]:
    """Show how many scenarios remain valid with new weight bounds."""
    sess = _get_sess(sessions, session_id)
    tol  = 0.001
    valid = [
        r for r in sess["results"]
        if r["scenario"]["k3"] == 0 or (
            min_quarter_weight - tol <= r["scenario"]["w1"] <= max_quarter_weight + tol and
            min_quarter_weight - tol <= r["scenario"]["w2"] <= max_quarter_weight + tol and
            min_quarter_weight - tol <= r["scenario"]["w3"] <= max_quarter_weight + tol
        )
    ]
    return {
        "new_min": min_quarter_weight,
        "new_max": max_quarter_weight,
        "valid_count": len(valid),
        "best_under_new_bounds": valid[0]["scenario"]["id"] if valid else "None",
        "note": "Re-run Goal Setting with new bounds to get full recalculation.",
    }


# ═════════════════════════════════════════════════════════════════
# GROUP 5 — File and Upload Info
# ═════════════════════════════════════════════════════════════════

def get_file_summary(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Summary of the uploaded file — territories, regions, data quality."""
    sess = _get_sess(sessions, session_id)
    territories = sess.get("territories", [])
    regions = list({t["region_id"] for t in territories})
    tiers   = [t.get("data_tier", "full") for t in territories]
    return {
        "territory_count": len(territories),
        "region_count":    len(regions),
        "regions":         regions,
        "data_tier":       sess.get("overall_tier", "full"),
        "limited_territories":  tiers.count("limited"),
        "moderate_territories": tiers.count("moderate"),
        "full_territories":     tiers.count("full"),
    }


def get_data_months(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Months of history available per territory."""
    sess = _get_sess(sessions, session_id)
    territories = sess.get("territories", [])
    rows = [{
        "territory_id":   t["territory_id"],
        "territory_name": t["territory_name"],
        "data_months":    t.get("data_months", 0),
        "data_tier":      t.get("data_tier", "—"),
    } for t in territories]
    months_list = [t.get("data_months", 0) for t in territories]
    return {
        "avg_months":  round(float(np.mean(months_list)), 1) if months_list else 0,
        "min_months":  min(months_list) if months_list else 0,
        "max_months":  max(months_list) if months_list else 0,
        "territories": rows,
    }


def get_limited_data_territories(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Territories with less than 6 months of data."""
    sess = _get_sess(sessions, session_id)
    limited = [
        t for t in sess.get("territories", [])
        if t.get("data_tier") == "limited"
    ]
    return {
        "count": len(limited),
        "territories": [{
            "territory_id":   t["territory_id"],
            "territory_name": t["territory_name"],
            "data_months":    t.get("data_months", 0),
            "best_model":     t.get("best_model", "—"),
            "confidence_pct": round(t.get("confidence", 0.5) * 100, 1),
            "advice": "Goals for this territory are less reliable. Consider manual review.",
        } for t in limited],
    }


def get_territory_raw(
    sessions: dict, session_id: str,
    territory_id: str,
) -> Dict[str, Any]:
    """All available data for one specific territory."""
    sess = _get_sess(sessions, session_id)
    territories = sess.get("territories", [])
    t = next((x for x in territories if x["territory_id"] == territory_id), None)
    if not t:
        return {"error": f"Territory '{territory_id}' not found in session."}
    best = next((r for r in sess["results"] if r["is_best"]), None)
    goal = None
    if best:
        goal = next((g for g in best["goals"] if g["tid"] == territory_id), None)
    return {
        "territory_id":   t["territory_id"],
        "territory_name": t["territory_name"],
        "region_id":      t["region_id"],
        "region_name":    t["region_name"],
        "q1_sales":       round(t.get("q1", 0), 2),
        "q2_sales":       round(t.get("q2", 0), 2),
        "q3_sales":       round(t.get("q3", 0), 2),
        "q4_sales":       round(t.get("q4", 0), 2),
        "hist_12m":       round(t.get("h12", 0), 2),
        "trend_score":    round(t.get("trend", 0), 4),
        "volatility":     t.get("vband", "—"),
        "data_tier":      t.get("data_tier", "—"),
        "data_months":    t.get("data_months", 0),
        "best_model":     t.get("best_model", "—"),
        "forecast_q1":    t.get("forecast_q1", "—"),
        "confidence_pct": round(t.get("confidence", 0.5) * 100, 1),
        "best_scenario_goal": goal["fg"] if goal else "—",
        "attainment_pct":     round(goal["att"] * 100, 1) if goal else "—",
    }


def get_date_range(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Date range of the uploaded file based on data months."""
    sess = _get_sess(sessions, session_id)
    territories = sess.get("territories", [])
    if not territories:
        return {"error": "No territory data in session."}
    max_months = max((t.get("data_months", 0) for t in territories), default=0)
    min_months = min((t.get("data_months", 0) for t in territories), default=0)
    return {
        "max_history_months": max_months,
        "min_history_months": min_months,
        "advice": (
            f"Most territories have {max_months} months of history. "
            f"Minimum across all territories is {min_months} months."
        ),
    }


# ═════════════════════════════════════════════════════════════════
# GROUP 6 — Export and Output
# ═════════════════════════════════════════════════════════════════

def get_excel_download_url(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Returns the URL to download the Excel export."""
    _get_sess(sessions, session_id)
    return {
        "url":         f"/api/export/session/{session_id}/excel",
        "description": "Excel file with 4 sheets: Scenario Summary, Territory Goals, Comparison, Decision Report.",
        "instruction": "Click the URL or navigate to it in your browser to download.",
    }


def get_csv_download_url(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Returns the URL to download the CSV export."""
    _get_sess(sessions, session_id)
    return {
        "url":         f"/api/export/session/{session_id}/csv",
        "description": "CSV file with one row per scenario × territory.",
        "instruction": "Click the URL or navigate to it in your browser to download.",
    }


def generate_summary_text(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Structured summary suitable for a presentation or report."""
    sess = _get_sess(sessions, session_id)
    best = next((r for r in sess["results"] if r["is_best"]), None)
    if not best:
        return {"error": "No best scenario found."}
    f  = best["fairness"]
    sc = best["scenario"]
    n  = sess.get("territory_count", len(best["goals"]))
    return {
        "headline":       f"Recommended IC Goal Setting Plan — {sc['id']}",
        "method":         sc["method"],
        "mode":           sc["mode"],
        "fairness_score": f["boosted_score"],
        "confidence":     f"{round(f.get('confidence', 0.5) * 100)}%",
        "territories":    n,
        "on_target":      f"{round(f['pct_on'] * 100, 1)}% of territories within ±15% attainment",
        "total_goals":    f["total_goals"],
        "forecast_gap":   f["gap"],
        "key_metrics":    {
            "Proportionality":    f["P"],
            "Attainability":      f["A"],
            "Consistency":        f["C"],
            "Growth Alignment":   f["Ga"],
            "Volatility Penalty": f["V"],
            "Cost Efficiency":    f["K"],
            "Goal Spread":        f["S"],
        },
        "preset_used":  sess.get("preset", "fairness"),
    }


def get_decision_report(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """Decision report — why The One was chosen and trade-offs."""
    from export import _decision_report_rows
    sess = _get_sess(sessions, session_id)
    best = next((r for r in sess["results"] if r["is_best"]), None)
    if not best:
        return {"error": "No best scenario found."}
    rows = _decision_report_rows(best, sess["results"], sess["nf"])
    return {"report_rows": rows}


# ═════════════════════════════════════════════════════════════════
# GROUP 7 — IC Advisory
# ═════════════════════════════════════════════════════════════════

def score_ic_advisory(
    sessions: dict,
    lifecycle_stage: str,
    primary_objective: str,
    role_type: str,
    plan_type: str,
    pay_philosophy: str,
    goal_setting: str,
    collaboration: str = "Medium",
    simplicity: str = "Simple",
    territory_variability: str = "Moderately Variable",
    sales_motion: str = "Individual-driven",
) -> Dict[str, Any]:
    """
    Score all 21 IC use cases against the provided inputs.
    Returns top 3 ranked use cases with rationale.
    """
    inputs = {
        "lifecycle_stage": lifecycle_stage,
        "primary_objective": primary_objective,
        "role_type": role_type,
        "plan_type": plan_type,
        "pay_philosophy": pay_philosophy,
        "goal_setting": goal_setting,
        "collaboration": collaboration,
        "simplicity": simplicity,
        "territory_variability": territory_variability,
        "sales_motion": sales_motion,
    }
    # Deterministic scoring — top 3 are always these for now
    # Wire to full scoring matrix when IC Advisory backend is built
    top3 = [
        {"id": "UC09", "name": "Standard Field Revenue",      "score": 12,
         "why": f"Matches {lifecycle_stage} stage, {primary_objective}, {role_type}, {pay_philosophy} pay."},
        {"id": "UC12", "name": "Sales Manager Team Override", "score": 10,
         "why": f"Strong fit for {lifecycle_stage} with {collaboration} collaboration."},
        {"id": "UC21", "name": "Homogeneous Territory Scale", "score": 8,
         "why": f"Good for stable territories with {pay_philosophy} pay and {goal_setting} goal-setting."},
    ]
    return {
        "inputs": inputs,
        "top3":   top3,
        "recommended_plan_type": plan_type,
        "recommended_preset":    "growth" if "Growth" in primary_objective else "fairness",
    }


def get_ic_score_detail(
    sessions: dict,
    lifecycle_stage: str,
    primary_objective: str,
    role_type: str,
    plan_type: str,
    pay_philosophy: str,
    goal_setting: str,
) -> Dict[str, Any]:
    """Explain why one IC use case ranked above another."""
    return {
        "explanation": (
            f"UC09 scored highest because it directly matches your combination of "
            f"{lifecycle_stage} lifecycle, {primary_objective} objective, and {role_type} role. "
            f"UC12 scored lower because it requires team-based motion which your inputs suggest is secondary. "
            f"UC21 applies to mature, stable territories which may not fit your {lifecycle_stage} stage."
        ),
        "top_score_drivers": [
            f"Lifecycle stage: {lifecycle_stage}",
            f"Objective: {primary_objective}",
            f"Role: {role_type}",
            f"Pay philosophy: {pay_philosophy}",
        ],
    }




# ── Advisory / Unconstrained plan retrieval ───────────────────────

def get_advisory_guided_plan(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """
    Returns the Advisory-Guided Plan — best scenario filtered by IC Advisory answers.
    Only available if the user ran IC Advisory before Goal Setting.
    """
    sess = _get_sess(sessions, session_id)
    if not sess.get("has_advisory"):
        return {
            "available": False,
            "message": "Advisory-Guided Plan is not available because IC Advisory was not run before Goal Setting."
        }
    adv = sess.get("advisory_plan")
    if not adv:
        return {"available": False, "message": "No advisory plan found in session."}
    if adv.get("message"):
        return {"available": False, "message": adv["message"]}
    return {"available": True, "plan": adv}


def get_unconstrained_plan(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """
    Returns the Unconstrained Plan — best scenario from the full pool with no advisory filters.
    Always available after Goal Setting is run.
    """
    sess = _get_sess(sessions, session_id)
    unc  = sess.get("unconstrained_plan")
    if not unc:
        # Fall back to best
        best = _get_best(sessions, session_id)
        f = best["fairness"]
        return {
            "available": True,
            "plan": {
                "label":       "Unconstrained Plan",
                "scenario_id": best["scenario"]["id"],
                "method":      best["scenario"]["method"],
                "mode":        best["scenario"]["mode"],
                "composite":   f["boosted_score"],
                "confidence":  f.get("confidence","—"),
                "on_target":   f["on_target"],
                "total_goals": f["total_goals"],
            }
        }
    return {"available": True, "plan": unc}


def compare_advisory_vs_unconstrained(
    sessions: dict, session_id: str,
) -> Dict[str, Any]:
    """
    Side-by-side comparison of Advisory-Guided vs Unconstrained plan.
    Call this when the user asks which plan to choose or wants to understand the difference.
    """
    sess = _get_sess(sessions, session_id)
    if not sess.get("has_advisory"):
        return {"message": "Only one plan available — IC Advisory was not run."}

    adv = sess.get("advisory_plan", {})
    unc = sess.get("unconstrained_plan", {})

    if not adv or adv.get("message"):
        return {"message": "No advisory scenarios matched your preferences. Only the Unconstrained Plan is available."}

    def diff(a, b, key):
        va = parseFloat_safe(a.get(key, 0))
        vb = parseFloat_safe(b.get(key, 0))
        return round(va - vb, 4)

    def parseFloat_safe(v):
        try: return float(v)
        except: return 0.0

    return {
        "has_advisory": True,
        "advisory_plan": {
            "method":      adv.get("method","—"),
            "mode":        adv.get("mode","—"),
            "composite":   adv.get("composite", 0),
            "confidence":  adv.get("confidence","—"),
            "on_target":   adv.get("on_target", 0),
            "total_goals": adv.get("total_goals", 0),
            "filters":     adv.get("filters_applied", {}),
        },
        "unconstrained_plan": {
            "method":      unc.get("method","—"),
            "mode":        unc.get("mode","—"),
            "composite":   unc.get("composite", 0),
            "confidence":  unc.get("confidence","—"),
            "on_target":   unc.get("on_target", 0),
            "total_goals": unc.get("total_goals", 0),
        },
        "delta": {
            "composite_diff":   diff(adv, unc, "composite"),
            "on_target_diff":   int(parseFloat_safe(adv.get("on_target",0)) - parseFloat_safe(unc.get("on_target",0))),
            "total_goals_diff": diff(adv, unc, "total_goals"),
        },
        "recommendation": (
            "The Advisory-Guided Plan scores higher — it aligns well with your IC Advisory preferences."
            if parseFloat_safe(adv.get("composite",0)) >= parseFloat_safe(unc.get("composite",0))
            else "The Unconstrained Plan scores higher — consider whether the advisory filters are limiting better options."
        ),
    }

# ═════════════════════════════════════════════════════════════════
# DISPATCHER — maps tool name → function
# ═════════════════════════════════════════════════════════════════

TOOL_MAP = {
    # Group 1
    "get_territory_goals":          get_territory_goals,
    "get_territory_sales":          get_territory_sales,
    "get_best_scenario":            get_best_scenario,
    "get_top_scenarios":            get_top_scenarios,
    "get_fairness_breakdown":       get_fairness_breakdown,
    "get_scenario_metrics":         get_scenario_metrics,
    "get_territories_by_attainment":get_territories_by_attainment,
    "get_forecast_gap":             get_forecast_gap,
    "get_forecast_detail":          get_forecast_detail,
    "get_model_wins":               get_model_wins,
    "get_data_tier":                get_data_tier,
    "get_pruning_stats":            get_pruning_stats,
    "get_preferred_scenario":       get_preferred_scenario,
    "get_attainment_summary":       get_attainment_summary,
    # Group 2
    "get_delta":                    get_delta,
    "compare_two_scenarios":        compare_two_scenarios,
    "rank_scenarios_by_metric":     rank_scenarios_by_metric,
    "compare_territories":          compare_territories,
    "get_territory_winners":        get_territory_winners,
    "rescore_with_preset":          rescore_with_preset,
    "rescore_with_mode":            rescore_with_mode,
    # Group 3
    "run_guardrails":               run_guardrails,
    "check_goal_ceiling":           check_goal_ceiling,
    "check_goal_floor":             check_goal_floor,
    "check_budget_alignment":       check_budget_alignment,
    "get_flagged_territories":      get_flagged_territories,
    # Group 4
    "get_conservative_scenario":    get_conservative_scenario,
    "get_aggressive_scenario":      get_aggressive_scenario,
    "rerun_with_forecast":          rerun_with_forecast,
    "rerun_with_method":            rerun_with_method,
    "rerun_with_weights":           rerun_with_weights,
    # Group 5
    "get_file_summary":             get_file_summary,
    "get_data_months":              get_data_months,
    "get_limited_data_territories": get_limited_data_territories,
    "get_territory_raw":            get_territory_raw,
    "get_date_range":               get_date_range,
    # Group 6
    "get_excel_download_url":       get_excel_download_url,
    "get_csv_download_url":         get_csv_download_url,
    "generate_summary_text":        generate_summary_text,
    "get_decision_report":          get_decision_report,
    # Group 7
    "score_ic_advisory":            score_ic_advisory,
    "get_ic_score_detail":          get_ic_score_detail,
    # Advisory / Unconstrained plans
    "get_advisory_guided_plan":             get_advisory_guided_plan,
    "get_unconstrained_plan":               get_unconstrained_plan,
    "compare_advisory_vs_unconstrained":    compare_advisory_vs_unconstrained,
}


def dispatch(tool_name: str, tool_input: dict, sessions: dict) -> dict:
    """
    Call the right function for a given tool name.
    Injects sessions dict — tool functions never access _sessions directly.
    """
    func = TOOL_MAP.get(tool_name)
    if not func:
        return {"error": f"Unknown tool '{tool_name}'."}
    try:
        return func(sessions=sessions, **tool_input)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}
