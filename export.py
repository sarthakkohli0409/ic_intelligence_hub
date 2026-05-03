"""
ProcDNA Intelligence — Export Engine v2
=========================================
Excel output — 4 sheets:
  1. Scenario Summary   — all pruned scenarios, 7 metrics, rank, confidence
  2. Territory Goals    — one row per (scenario × territory), final goals plain numbers
  3. Comparison         — best vs preferred side by side with delta
  4. Decision Report    — why chosen, trade-offs vs #2/#3, risk scenarios (rule-based)

CSV output — one row per (scenario × territory), flat.

Guardrails check — returns pass/fail per rule with explanation.
Delta insights   — attainment shift, goal difference, fairness change, rule-based narrative.
"""

import io
import csv
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

ATTAINMENT_LO = 0.85
ATTAINMENT_HI = 1.15


# ── Delta insights (Section X) ────────────────────────────────────────────────
def delta_insights(
    best: Dict,
    compare: Dict,
    national_forecast: float,
) -> Dict[str, Any]:
    """
    Structured delta between best scenario and a comparison scenario.
    Returns metric differences + 1-sentence rule-based narrative per key metric.
    """
    bf = best["fairness"]
    cf = compare["fairness"]

    att_shift   = round((cf["att"]  - bf["att"])  * 100, 1)
    fair_shift  = round(cf["score"] - bf["score"], 4)
    total_shift = round(cf["total_goals"] - bf["total_goals"], 2)
    cons_shift  = round((cf["cons"] - bf["cons"]) * 100, 1)
    ga_shift    = round((cf["Ga"]   - bf["Ga"])   * 100, 1)

    narratives = []

    if abs(att_shift) >= 1:
        direction = "improves" if att_shift > 0 else "reduces"
        narratives.append(
            f"Attainability {direction} by {abs(att_shift):.1f}pp — "
            f"{'more' if att_shift > 0 else 'fewer'} territories fall within the ±15% attainment band."
        )

    if abs(cons_shift) >= 2:
        direction = "more consistent" if cons_shift > 0 else "less consistent"
        narratives.append(
            f"Goals are {direction} across regions ({abs(cons_shift):.1f}pp shift in CV score)."
        )

    if abs(ga_shift) >= 2:
        direction = "better aligned to" if ga_shift > 0 else "less aligned to"
        narratives.append(
            f"Goals are {direction} territory growth trends ({abs(ga_shift):.1f}pp shift)."
        )

    if abs(total_shift) > 0:
        direction = "higher" if total_shift > 0 else "lower"
        pct = abs(total_shift / national_forecast * 100) if national_forecast > 0 else 0
        narratives.append(
            f"Total goals are {direction} by {abs(total_shift):,.0f} "
            f"({pct:.1f}% of national forecast)."
        )

    if abs(fair_shift) >= 0.01:
        direction = "better" if fair_shift > 0 else "lower"
        narratives.append(
            f"Overall fairness score is {direction} "
            f"({bf['score']:.3f} vs {cf['score']:.3f})."
        )

    return {
        "best_scenario_id":    best["scenario"]["id"],
        "compare_scenario_id": compare["scenario"]["id"],
        "attainability_shift_pp":    att_shift,
        "fairness_score_shift":      fair_shift,
        "total_goals_shift":         total_shift,
        "consistency_shift_pp":      cons_shift,
        "growth_alignment_shift_pp": ga_shift,
        "narrative": narratives if narratives else ["Scenarios are statistically similar across all metrics."],
    }


# ── Decision Report (Section XI) ─────────────────────────────────────────────
def _decision_report_rows(
    best: Dict,
    results: List[Dict],
    national_forecast: float,
) -> List[Dict]:
    """
    Build row data for Sheet 4 — Decision Report.
    Rule-based, no AI call.
    """
    bf    = best["fairness"]
    bsc   = best["scenario"]
    rows  = []

    # --- Section 1: Selected Plan ---
    rows.append({"Section": "SELECTED PLAN", "Item": "Scenario ID",       "Value": bsc["id"]})
    rows.append({"Section": "",               "Item": "Method",            "Value": bsc["method"]})
    rows.append({"Section": "",               "Item": "Plan Mode",         "Value": bsc["mode"]})
    rows.append({"Section": "",               "Item": "Preset Used",       "Value": bf.get("preset", "fairness")})
    rows.append({"Section": "",               "Item": "Composite Score",   "Value": bf["score"]})
    rows.append({"Section": "",               "Item": "Confidence",        "Value": bf.get("confidence", "—")})
    rows.append({"Section": "",               "Item": "On-Target %",       "Value": f"{bf['pct_on']*100:.1f}%"})
    rows.append({"Section": "",               "Item": "Total Goals",       "Value": bf["total_goals"]})
    rows.append({"Section": "",               "Item": "Forecast Gap",      "Value": bf["gap"]})

    # --- Section 2: Why it was chosen ---
    rows.append({"Section": "WHY CHOSEN", "Item": "", "Value": ""})
    metric_names = {"P": "Proportionality", "A": "Attainability", "C": "Consistency",
                    "Ga": "Growth Alignment", "V": "Volatility Penalty",
                    "K": "Cost Efficiency",  "S": "Goal Spread"}
    top3 = sorted(
        [(k, bf.get(k, 0)) for k in ["P","A","C","Ga","V","K","S"]],
        key=lambda x: -x[1]
    )[:3]
    for rank_i, (mk, mv) in enumerate(top3, 1):
        rows.append({
            "Section": "",
            "Item":    f"Top reason #{rank_i}: {metric_names.get(mk, mk)}",
            "Value":   f"{mv:.3f}",
        })

    # --- Section 3: Trade-offs vs #2 and #3 ---
    others = [r for r in results if not r["is_best"]][:2]
    rows.append({"Section": "TRADE-OFFS", "Item": "", "Value": ""})
    for alt in others:
        af  = alt["fairness"]
        aid = alt["scenario"]["id"]
        att_d = round((af["att"] - bf["att"]) * 100, 1)
        sc_d  = round(af["score"] - bf["score"], 4)
        tg_d  = round(af["total_goals"] - bf["total_goals"], 2)
        rows.append({
            "Section": f"vs {aid}",
            "Item":    "Attainability difference",
            "Value":   f"{'+' if att_d>=0 else ''}{att_d:.1f}pp",
        })
        rows.append({
            "Section": "",
            "Item":    "Fairness score difference",
            "Value":   f"{'+' if sc_d>=0 else ''}{sc_d:.4f}",
        })
        rows.append({
            "Section": "",
            "Item":    "Total goals difference",
            "Value":   f"{'+' if tg_d>=0 else ''}{tg_d:,.0f}",
        })

    # --- Section 4: Risk scenarios ---
    rows.append({"Section": "RISK SCENARIOS", "Item": "", "Value": ""})
    # Conservative: lowest total goals in top-10
    pool = results[:10]
    conservative = min(pool, key=lambda r: r["fairness"]["total_goals"])
    aggressive   = max(pool, key=lambda r: r["fairness"]["total_goals"])
    rows.append({
        "Section": "Conservative",
        "Item":    f"{conservative['scenario']['id']} — lowest total goals",
        "Value":   f"{conservative['fairness']['total_goals']:,.0f} "
                   f"(score {conservative['fairness']['score']:.3f})",
    })
    rows.append({
        "Section": "Aggressive",
        "Item":    f"{aggressive['scenario']['id']} — highest total goals",
        "Value":   f"{aggressive['fairness']['total_goals']:,.0f} "
                   f"(score {aggressive['fairness']['score']:.3f})",
    })

    return rows


# ── Guardrails (Section XII) ──────────────────────────────────────────────────
def guardrails_check(
    results: List[Dict],
    national_forecast: float,
    best_id: str,
) -> Dict[str, Any]:
    """
    Check the best scenario against 4 guardrail rules.
    Returns pass/fail per rule + explanation.
    """
    best = next((r for r in results if r["scenario"]["id"] == best_id), None)
    if not best:
        return {"error": "Best scenario not found"}

    goals = best["goals"]
    checks = []
    all_pass = True

    # Rule 1: Total goals within ±20% of national forecast
    if national_forecast > 0:
        total = sum(g["fg"] for g in goals)
        gap_pct = abs(total - national_forecast) / national_forecast * 100
        passed = gap_pct <= 20.0
        all_pass = all_pass and passed
        checks.append({
            "rule":        "Budget alignment",
            "description": "Total goals must be within ±20% of national forecast",
            "passed":      passed,
            "value":       f"{gap_pct:.1f}% deviation",
            "detail":      f"Total goals {total:,.0f} vs NF {national_forecast:,.0f}",
        })

    # Rule 2: No territory goal below 50% of average
    avg_goal = np.mean([g["fg"] for g in goals]) if goals else 1
    low_outliers = [g["tid"] for g in goals if g["fg"] < avg_goal * 0.5]
    passed = len(low_outliers) == 0
    all_pass = all_pass and passed
    checks.append({
        "rule":        "Minimum goal floor",
        "description": "No territory should have a goal below 50% of the average",
        "passed":      passed,
        "value":       f"{len(low_outliers)} territories below floor",
        "detail":      f"Outliers: {low_outliers[:5]}" if low_outliers else "None",
    })

    # Rule 3: No territory goal above 200% of average
    high_outliers = [g["tid"] for g in goals if g["fg"] > avg_goal * 2.0]
    passed = len(high_outliers) == 0
    all_pass = all_pass and passed
    checks.append({
        "rule":        "Maximum goal ceiling",
        "description": "No territory should have a goal above 200% of the average",
        "passed":      passed,
        "value":       f"{len(high_outliers)} territories above ceiling",
        "detail":      f"Outliers: {high_outliers[:5]}" if high_outliers else "None",
    })

    # Rule 4: Attainability — at least 70% of territories on target
    on_target_pct = sum(1 for g in goals if ATTAINMENT_LO <= g["att"] <= ATTAINMENT_HI) / max(len(goals), 1)
    passed = on_target_pct >= 0.70
    all_pass = all_pass and passed
    checks.append({
        "rule":        "Attainability floor",
        "description": "At least 70% of territories must have forecast within ±15% of goal",
        "passed":      passed,
        "value":       f"{on_target_pct*100:.1f}% on target",
        "detail":      f"{sum(1 for g in goals if ATTAINMENT_LO<=g['att']<=ATTAINMENT_HI)} / {len(goals)} territories",
    })

    return {
        "scenario_id": best_id,
        "all_pass":    all_pass,
        "checks":      checks,
        "summary":     "All guardrails passed." if all_pass else
                       f"{sum(1 for c in checks if not c['passed'])} guardrail(s) failed. Review before finalising.",
    }


# ── Excel export ──────────────────────────────────────────────────────────────
def to_excel(
    results: List[Dict],
    best_id: str,
    preferred_id: Optional[str],
    national_forecast: float,
) -> bytes:
    buf = io.BytesIO()

    # Sheet 1: Scenario Summary
    summary_rows = []
    for r in results:
        sc, fs = r["scenario"], r["fairness"]
        summary_rows.append({
            "Rank": r["rank"], "Scenario ID": sc["id"],
            "Method": sc["method"], "Window (months)": sc["window"],
            "Mode": sc["mode"], "Regional Share": sc["rs"],
            "Individual Share": sc["is_"], "National Share": sc["ns"],
            "W1": sc["w1"], "W2": sc["w2"], "W3": sc["w3"],
            "Composite Score": fs["boosted_score"],
            "Proportionality (P)": fs["P"],
            "Attainability (A)": fs["A"],
            "Consistency (C)": fs["C"],
            "Growth Alignment (Ga)": fs["Ga"],
            "Volatility Penalty (V)": fs["V"],
            "Cost Efficiency (K)": fs["K"],
            "Goal Spread (S)": fs["S"],
            "Confidence": fs.get("confidence", "—"),
            "% On Target": round(fs["pct_on"] * 100, 1),
            "Total Goals": fs["total_goals"],
            "Forecast Gap": fs["gap"],
            "Is Best": "YES" if r["is_best"] else "",
            "Is Preferred": "YES" if r.get("is_pref") else "",
            "IC Boost Applied": fs.get("ic_boost", 0),
        })

    # Sheet 2: Territory Goals
    goal_rows = []
    for r in results:
        sc = r["scenario"]
        for g in r["goals"]:
            goal_rows.append({
                "Scenario ID": g["sid"], "Rank": r["rank"],
                "Territory ID": g["tid"], "Territory": g["tname"],
                "Region ID": g["rid"], "Region": g["rname"],
                "Final Goal": g["fg"],
                "Regional Goal": g["rg"], "Individual Goal": g["ig"], "National Goal": g["ng"],
                "Regional Basis": g["rb"], "Individual Basis": g["ib"],
                "Forecast (F_i)": g["fc"],
                "Attainment (F/G)": round(g["att"] * 100, 1),
                "In ±15% Band": "YES" if ATTAINMENT_LO <= g["att"] <= ATTAINMENT_HI else "NO",
            })

    # Sheet 3: Comparison best vs preferred
    best_r = next((r for r in results if r["scenario"]["id"] == best_id), None)
    pref_r = next((r for r in results if r["scenario"]["id"] == preferred_id), None) if preferred_id else None
    comp_rows = []
    if best_r:
        bm = {g["tid"]: g for g in best_r["goals"]}
        pm = {g["tid"]: g for g in pref_r["goals"]} if pref_r else {}
        for tid, bg in bm.items():
            row = {
                "Territory ID": tid, "Territory": bg["tname"], "Region": bg["rname"],
                f"Best ({best_id}) Goal": bg["fg"],
                f"Best Attainment %": round(bg["att"] * 100, 1),
            }
            if tid in pm:
                pg = pm[tid]
                diff = round(pg["fg"] - bg["fg"], 2)
                diff_pct = round(diff / bg["fg"] * 100, 1) if bg["fg"] else 0
                row[f"Preferred ({preferred_id}) Goal"] = pg["fg"]
                row["Preferred Attainment %"] = round(pg["att"] * 100, 1)
                row["Goal Difference"] = diff
                row["Difference %"] = diff_pct
                row["Better For Territory"] = (
                    "Preferred" if abs(pg["att"] - 1.0) < abs(bg["att"] - 1.0) else "Best"
                )
            comp_rows.append(row)

    # Sheet 4: Decision Report
    decision_rows = []
    if best_r:
        decision_rows = _decision_report_rows(best_r, results, national_forecast)

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Scenario Summary", index=False)
        pd.DataFrame(goal_rows).to_excel(writer, sheet_name="Territory Goals", index=False)
        pd.DataFrame(comp_rows if comp_rows else [{"Note": "No comparison data"}]).to_excel(
            writer, sheet_name="Comparison", index=False
        )
        pd.DataFrame(decision_rows if decision_rows else [{"Note": "No decision data"}]).to_excel(
            writer, sheet_name="Decision Report", index=False
        )
        # Auto-width all sheets
        for sn in writer.sheets:
            ws = writer.sheets[sn]
            for col in ws.columns:
                ml = max((len(str(c.value or "")) for c in col), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(ml + 2, 50)

    return buf.getvalue()


# ── CSV export ────────────────────────────────────────────────────────────────
def to_csv(results: List[Dict]) -> bytes:
    out = io.StringIO()
    fields = [
        "scenario_id", "rank", "method", "mode",
        "rs", "is_", "ns", "w1", "w2", "w3",
        "composite", "confidence", "P", "A", "C", "Ga", "V", "K", "S",
        "territory_id", "territory", "region_id", "region",
        "final_goal", "forecast_fi", "attainment_pct",
        "is_best", "is_pref",
    ]
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for r in results:
        sc, fs = r["scenario"], r["fairness"]
        for g in r["goals"]:
            writer.writerow({
                "scenario_id": g["sid"], "rank": r["rank"],
                "method": sc["method"], "mode": sc["mode"],
                "rs": sc["rs"], "is_": sc["is_"], "ns": sc["ns"],
                "w1": sc["w1"], "w2": sc["w2"], "w3": sc["w3"],
                "composite": fs["boosted_score"],
                "confidence": fs.get("confidence", ""),
                "P": fs["P"], "A": fs["A"], "C": fs["C"],
                "Ga": fs["Ga"], "V": fs["V"], "K": fs["K"], "S": fs["S"],
                "territory_id": g["tid"], "territory": g["tname"],
                "region_id": g["rid"], "region": g["rname"],
                "final_goal": g["fg"],
                "forecast_fi": g["fc"],
                "attainment_pct": round(g["att"] * 100, 1),
                "is_best": "YES" if r["is_best"] else "",
                "is_pref": "YES" if r.get("is_pref") else "",
            })
    return out.getvalue().encode("utf-8")
