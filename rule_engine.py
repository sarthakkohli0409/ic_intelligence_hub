"""
ProcDNA Intelligence — Rule-Based Response Engine
Handles all chat queries without needing an Anthropic API key.
Uses the tool functions directly to query session data.
"""

import time
from typing import Any, Dict


def pf(n):
    try:
        return f"{float(n):.3f}"
    except Exception:
        return "—"


def rule_based_response(message: str, session_id: str, sessions: dict) -> Dict[str, Any]:
    """
    Fully rule-based chat responses — no API key needed.
    Handles all common IC planning questions using tool functions directly.
    """
    from tools import dispatch

    msg = message.lower().strip()
    tools_called = []
    tool_results = []
    start = time.time()

    def run(tool, **kwargs):
        if session_id:
            kwargs["session_id"] = session_id
        result = dispatch(tool, kwargs, sessions)
        tools_called.append(tool)
        tool_results.append({"tool": tool, "result": result})
        return result

    def reply(text):
        elapsed = max(600, int((time.time() - start) * 1000))
        return {
            "reply": text,
            "tools_called": tools_called,
            "tool_results": tool_results,
            "status_labels": ["Thinking...", "Analysing your data..."] if tools_called else ["Thinking..."],
            "elapsed_ms": elapsed,
        }

    has_session = bool(session_id and session_id in sessions)

    # ── Session-dependent queries ─────────────────────────────────

    if has_session:

        # Territory goals
        if any(x in msg for x in ["territory goal", "goals territory", "quarterly sales",
                                    "territory wise", "show goals", "goal breakdown"]):
            r = run("get_territory_goals")
            if "error" not in r:
                rows = r.get("territories", [])[:10]
                lines = ["| Territory | Region | Q1 Sales | Q2 Sales | Goal | Attainment |",
                         "|---|---|---|---|---|---|"]
                for t in rows:
                    lines.append(
                        "| " + t["territory_name"] + " | " + t["region_name"] +
                        " | " + str(int(t.get("q1_sales", 0))) +
                        " | " + str(int(t.get("q2_sales", 0))) +
                        " | " + str(int(t.get("final_goal", 0))) +
                        " | " + str(t.get("attainment_pct", 0)) + "% |"
                    )
                return reply(
                    "Here are the territory goals for scenario " +
                    r.get("scenario_id", "—") + ":\n\n" +
                    "\n".join(lines) +
                    "\n\nTotal territories: **" + str(r.get("territory_count", 0)) + "**"
                )

        # Below target
        if any(x in msg for x in ["below target", "at risk", "under target", "below attain", "miss"]):
            r = run("get_territories_by_attainment", filter="below")
            rows = r.get("territories", [])
            if rows:
                lines = ["| Territory | Region | Goal | Forecast | Attainment |",
                         "|---|---|---|---|---|"]
                for t in rows[:10]:
                    lines.append(
                        "| " + t["territory_name"] + " | " + t["region_name"] +
                        " | " + str(int(t.get("final_goal", 0))) +
                        " | " + str(int(t.get("forecast", 0))) +
                        " | " + str(t.get("attainment_pct", 0)) + "% |"
                    )
                return reply(
                    "**" + str(len(rows)) + " territories are below target** (attainment < 85%):\n\n" +
                    "\n".join(lines) +
                    "\n\nConsider reviewing goal levels or providing additional support for these territories."
                )
            return reply("All territories are on track — none are below the 85% attainment threshold.")

        # Above target
        if any(x in msg for x in ["above target", "over target", "exceeding", "overperform"]):
            r = run("get_territories_by_attainment", filter="above")
            rows = r.get("territories", [])
            if rows:
                lines = ["| Territory | Region | Goal | Forecast | Attainment |",
                         "|---|---|---|---|---|"]
                for t in rows[:10]:
                    lines.append(
                        "| " + t["territory_name"] + " | " + t["region_name"] +
                        " | " + str(int(t.get("final_goal", 0))) +
                        " | " + str(int(t.get("forecast", 0))) +
                        " | " + str(t.get("attainment_pct", 0)) + "% |"
                    )
                return reply(
                    "**" + str(len(rows)) + " territories are above target** (attainment > 115%):\n\n" +
                    "\n".join(lines)
                )
            return reply("No territories are significantly above target.")

        # Guardrails
        if any(x in msg for x in ["guardrail", "validate", "sanity check", "check plan"]):
            r = run("run_guardrails")
            checks = r.get("checks", [])
            lines = []
            for c in checks:
                icon = "✓" if c["passed"] else "✗"
                lines.append(icon + " **" + c["rule"] + "** — " + c["value"])
            summary = r.get("summary", "")
            return reply("**Guardrails Check**\n\n" + "\n".join(lines) + "\n\n" + summary)

        # Top 5
        if any(x in msg for x in ["top 5", "top five", "best scenario", "top scenario"]):
            r = run("get_top_scenarios", n=5)
            rows = r.get("scenarios", [])
            lines = ["| Rank | Scenario | Method | Mode | Score | On Target |",
                     "|---|---|---|---|---|---|"]
            for s in rows:
                lines.append(
                    "| #" + str(s["rank"]) +
                    " | " + s["scenario_id"] +
                    " | " + s["method"] +
                    " | " + s["mode"] +
                    " | " + pf(s.get("composite", 0)) +
                    " | " + str(s.get("on_target_pct", 0)) + "% |"
                )
            return reply("**Top 5 Scenarios by Fairness Score:**\n\n" + "\n".join(lines))

        # Fairness breakdown
        if any(x in msg for x in ["fairness", "metric", "breakdown", "score breakdown",
                                    "proportionality", "attainability", "consistency"]):
            r = run("get_fairness_breakdown")
            m = r.get("metrics", {})
            lines = ["| Metric | Score | What it means |", "|---|---|---|"]
            for k, v in m.items():
                lines.append("| " + k + " | " + pf(v["score"]) + " | " + v["explanation"] + " |")
            return reply(
                "**Fairness Score Breakdown — Scenario " + r.get("scenario_id", "—") + "**\n\n" +
                "Composite: **" + pf(r.get("composite", 0)) + "**\n\n" +
                "\n".join(lines)
            )

        # Delta / comparison
        if any(x in msg for x in ["delta", "compare best", "difference between", "vs preferred"]):
            r = run("get_delta")
            if "error" not in r and "message" not in r:
                text = (
                    "**Delta: " + str(r.get("best_id", "—")) + " vs " + str(r.get("preferred_id", "—")) + "**\n\n" +
                    "- Attainability shift: **" + ("+"+str(r.get("attainability_shift_pp",0)) if r.get("attainability_shift_pp",0)>=0 else str(r.get("attainability_shift_pp",0))) + "pp**\n" +
                    "- Fairness score shift: **" + ("+"+pf(r.get("fairness_shift",0)) if float(r.get("fairness_shift",0) or 0)>=0 else pf(r.get("fairness_shift",0))) + "**\n" +
                    "- Total goals shift: **" + str(int(r.get("total_goals_shift", 0))) + "**\n" +
                    "- Consistency shift: **" + ("+"+str(r.get("consistency_shift_pp",0)) if r.get("consistency_shift_pp",0)>=0 else str(r.get("consistency_shift_pp",0))) + "pp**"
                )
                return reply(text)
            return reply(r.get("message", "Best and preferred scenarios are the same — no delta to compute."))

        # Summary
        if any(x in msg for x in ["summary", "executive", "report", "present", "leadership"]):
            r = run("generate_summary_text")
            if "error" not in r:
                text = (
                    "**" + r.get("headline", "IC Plan Summary") + "**\n\n" +
                    "- Method: **" + str(r.get("method", "—")) + "** | Mode: **" + str(r.get("mode", "—")) + "**\n" +
                    "- Fairness Score: **" + str(r.get("fairness_score", "—")) + "** | Confidence: **" + str(r.get("confidence", "—")) + "**\n" +
                    "- Territories: **" + str(r.get("territories", "—")) + "** | On Target: **" + str(r.get("on_target", "—")) + "**\n" +
                    "- Total Goals: **" + str(int(r.get("total_goals", 0))) + "** | Forecast Gap: **" + str(int(r.get("forecast_gap", 0))) + "**\n\n" +
                    "Preset used: **" + str(r.get("preset_used", "fairness")) + "**"
                )
                return reply(text)

        # Export
        if any(x in msg for x in ["export", "download", "excel", "csv"]):
            r = run("get_excel_download_url")
            return reply(
                "Your Excel export (4 sheets: Scenario Summary, Territory Goals, Comparison, Decision Report) "
                "is ready. Click the **Download Excel** button on the left panel, or go to:\n\n"
                "`/api/export/session/" + str(session_id)[:8] + ".../excel`"
            )

        # Conservative
        if any(x in msg for x in ["conservative", "lowest goal", "safe plan", "low risk"]):
            r = run("get_conservative_scenario")
            return reply(
                "**Conservative Scenario:** " + str(r.get("scenario_id", "—")) + "\n" +
                "- Method: " + str(r.get("method", "—")) + " | Mode: " + str(r.get("mode", "—")) + "\n" +
                "- Total goals: **" + str(int(r.get("total_goals", 0))) + "** | Score: **" + pf(r.get("composite", 0)) + "**\n" +
                "- On target: **" + str(r.get("on_target_pct", 0)) + "%**"
            )

        # Aggressive
        if any(x in msg for x in ["aggressive", "highest goal", "stretch", "ambitious", "stretch goal"]):
            r = run("get_aggressive_scenario")
            return reply(
                "**Aggressive Scenario:** " + str(r.get("scenario_id", "—")) + "\n" +
                "- Method: " + str(r.get("method", "—")) + " | Mode: " + str(r.get("mode", "—")) + "\n" +
                "- Total goals: **" + str(int(r.get("total_goals", 0))) + "** | Score: **" + pf(r.get("composite", 0)) + "**\n" +
                "- On target: **" + str(r.get("on_target_pct", 0)) + "%**"
            )

        # Forecast models
        if any(x in msg for x in ["forecast model", "model win", "which model", "best model"]):
            r = run("get_model_wins")
            wins = r.get("model_wins", [])
            lines = ["| Model | Territories | Share |", "|---|---|---|"]
            for w in wins[:8]:
                lines.append("| " + w["model"] + " | " + str(w["territories"]) + " | " + str(w["share_pct"]) + "% |")
            return reply(
                "**Forecast Model Competition Results**\n\n" +
                "Average wMAPE: **" + str(r.get("avg_wmape", "—")) + "** | Average Confidence: **" + str(r.get("avg_confidence_pct", "—")) + "%**\n\n" +
                "Top model: **" + str(r.get("top_model", "—")) + "**\n\n" +
                "\n".join(lines)
            )

        # File summary
        if any(x in msg for x in ["file summary", "data summary", "how many territories", "how many regions"]):
            r = run("get_file_summary")
            return reply(
                "**Data Summary**\n\n" +
                "- Territories: **" + str(r.get("territory_count", 0)) + "**\n" +
                "- Regions: **" + str(r.get("region_count", 0)) + "**\n" +
                "- Data tier: **" + str(r.get("data_tier", "—")) + "**\n" +
                "- Full data territories: **" + str(r.get("full_territories", 0)) + "**\n" +
                "- Limited data territories: **" + str(r.get("limited_territories", 0)) + "**"
            )

    # ── General IC knowledge (no session needed) ──────────────────

    if "proportionality" in msg:
        return reply(
            "**Proportionality (P)** measures whether territory goals are proportional to territory potential. "
            "It uses Pearson correlation between final goals (G) and historical sales potential (T). "
            "A score of 1.0 means goals perfectly match potential — stronger territories get higher goals."
        )

    if "attainability" in msg or "attainment" in msg:
        return reply(
            "**Attainability (A)** measures the percentage of territories where the forecast (F_i) "
            "falls within ±15% of the goal (G_i). A score of 0.90 means 90% of territories are "
            "expected to hit their goal within that band. Higher is better."
        )

    if "holt-winters" in msg or "holt winters" in msg:
        return reply(
            "**Holt-Winters** is a triple exponential smoothing model that handles both trend and "
            "seasonality. It typically wins on pharma and medtech territories with 24+ months of data "
            "and clear seasonal patterns. It uses additive trend and additive seasonal components."
        )

    if any(x in msg for x in ["wmape", "mape", "forecast error"]):
        return reply(
            "**wMAPE** (Weighted Mean Absolute Percentage Error) measures forecast accuracy. Lower is better.\n\n"
            "- wMAPE < 0.10: Excellent\n"
            "- wMAPE 0.10–0.20: Good\n"
            "- wMAPE > 0.20: Moderate — treat with caution\n\n"
            "Confidence = 1 − wMAPE, capped by data tier (limited/moderate/full)."
        )

    if any(x in msg for x in ["hybrid", "60/40", "70/30", "50/50"]):
        return reply(
            "**Hybrid plan modes** combine Regional and Individual components:\n\n"
            "- **Hybrid 70/30**: 70% regional market share, 30% individual history\n"
            "- **Hybrid 60/40**: 60% regional, 40% individual (most common)\n"
            "- **Hybrid 50/50**: Equal weight\n\n"
            "Hybrid plans balance territory fairness (regional) with individual accountability (individual)."
        )

    if any(x in msg for x in ["preset", "fairness mode", "growth mode", "cost mode"]):
        return reply(
            "**Three scoring presets:**\n\n"
            "**Fairness Mode** — prioritises Proportionality (P), Consistency (C), Attainability (A). "
            "Default for most IC plans.\n\n"
            "**Growth Mode** — prioritises Attainability (A), Growth Alignment (Ga), Goal Spread (S). "
            "Use for aggressive growth cycles.\n\n"
            "**Cost Mode** — prioritises Cost Efficiency (K), Volatility Penalty (V), Consistency (C). "
            "Use when budget adherence is critical."
        )

    if any(x in msg for x in ["pareto", "the one", "best scenario selected"]):
        return reply(
            "**The One** is selected via a 3-step Pareto process:\n\n"
            "1. Build Pareto frontier — keep all non-dominated scenarios (no other scenario beats it on all 7 metrics)\n"
            "2. Rank within the frontier using preset weights\n"
            "3. Tie-break by: highest confidence → lowest cost variance → simplest structure\n\n"
            "This means The One isn't just the highest composite score — it's the best *balanced* option."
        )

    if any(x in msg for x in ["pay mix", "pay philosophy", "base salary", "accelerator"]):
        return reply(
            "**IC Pay Mix** is the ratio of fixed base salary to variable incentive:\n\n"
            "- **Aggressive** (e.g. 50/50): High upside, high risk — suited for competitive markets\n"
            "- **Balanced** (e.g. 60/40 or 70/30): Moderate upside — suited for most field sales roles\n"
            "- **Conservative** (e.g. 80/20): Stable income, small variable — suited for account management\n\n"
            "The accelerator (payout rate above 100% quota) should be monotonically increasing."
        )

    if any(x in msg for x in ["help", "what can you", "capabilities", "what do you do"]):
        return reply(
            "**I can help you with:**\n\n"
            "- **IC Advisory** — 10 questions to recommend the right IC structure\n"
            "- **Goal Setting** — upload sales data, run 15-model forecasting, score all scenarios\n"
            "- **Scenario Analysis** — territory goals, fairness breakdown, top 5 scenarios\n"
            "- **Guardrails** — validate the plan against budget, floor, ceiling, attainability rules\n"
            "- **Comparison** — compare best vs custom scenario with live sliders\n"
            "- **Export** — download Excel (4 sheets) or CSV\n\n"
            "Use the quick chips above or just ask me directly."
        )

    # No session + data question
    if not has_session and any(x in msg for x in ["goal", "territory", "scenario", "forecast", "attain", "score"]):
        return reply(
            "I don't have an active session yet. Please run **Goal Setting** first — "
            "upload your sales data file and I'll run the full pipeline. "
            "Once complete, I can answer detailed questions about your territories, scenarios, and forecasts."
        )

    # Default
    return reply(
        "I'm IC Intelligence, your IC planning assistant. I can help with territory goals, "
        "scenario analysis, guardrail checks, forecasting, and IC structure design. "
        "Ask me a specific question or use the quick chips above to get started."
    )
