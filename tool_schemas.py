"""
ProcDNA Intelligence — Tool Schemas
=====================================
Claude reads these descriptions to decide which tool to call.
The description field is the most important part — be specific.
"""

TOOL_SCHEMAS = [

    # ── Group 1: Data Retrieval ───────────────────────────────────

    {
        "name": "get_territory_goals",
        "description": (
            "Returns quarterly sales history and final goals for every territory. "
            "Call this when the user asks for territory goals, goal breakdown, "
            "quarterly sales per territory, or wants to see goals by region."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string", "description": "Active session ID"},
                "scenario_id": {"type": "string", "description": "Optional — defaults to best scenario"},
                "region_filter": {"type": "string", "description": "Optional — filter to one region ID"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_territory_sales",
        "description": (
            "Returns raw quarterly sales history from the uploaded file for all territories. "
            "Call this when the user asks about historical sales, raw data, or sales trends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_best_scenario",
        "description": (
            "Returns full details of the best (Pareto-selected) scenario including "
            "all 7 metric scores, confidence, and goal summary. "
            "Call this when the user asks what the best scenario is, what The One is, "
            "or wants a summary of the recommended plan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_top_scenarios",
        "description": (
            "Returns the top N scenarios ranked by composite score. "
            "Call this when the user asks to see the top scenarios, top 5, top 10, "
            "or wants to compare several scenarios side by side."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "n": {"type": "integer", "description": "Number of top scenarios to return. Default 5."},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_fairness_breakdown",
        "description": (
            "Returns all 7 metric scores (P, A, C, Ga, V, K, S) for a scenario "
            "with a plain-English explanation of each. "
            "Call this when the user asks about the fairness score, metric breakdown, "
            "why a scenario scored well, or what each metric means."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "scenario_id": {"type": "string", "description": "Optional — defaults to best"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_scenario_metrics",
        "description": (
            "Returns all 7 metric values for a specific scenario by ID. "
            "Use when the user names a specific scenario like SC_042."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "scenario_id": {"type": "string"},
            },
            "required": ["session_id", "scenario_id"],
        },
    },

    {
        "name": "get_territories_by_attainment",
        "description": (
            "Filter territories by attainment status: below target (<85%), "
            "above target (>115%), or on target (85-115%). "
            "Call this when the user asks which territories are at risk, "
            "below target, overperforming, or on track."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "filter":      {"type": "string", "enum": ["below", "above", "on_target"],
                                "description": "below = <85%, above = >115%, on_target = 85-115%"},
                "scenario_id": {"type": "string", "description": "Optional — defaults to best"},
            },
            "required": ["session_id", "filter"],
        },
    },

    {
        "name": "get_forecast_gap",
        "description": (
            "Returns total goals vs national forecast — the budget gap. "
            "Call this when the user asks if goals are within budget, "
            "what the total goals are vs forecast, or about budget alignment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "scenario_id": {"type": "string", "description": "Optional"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_forecast_detail",
        "description": (
            "Returns the forecast model, wMAPE, and confidence score per territory. "
            "Call this when the user asks about forecast accuracy, model confidence, "
            "which model was used per territory, or forecast reliability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_model_wins",
        "description": (
            "Returns which forecast model won for the most territories and the full breakdown. "
            "Call this when the user asks about model competition results, "
            "which algorithm performed best, or average forecast error."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_data_tier",
        "description": (
            "Returns the data tier classification (limited/moderate/full) for the uploaded file. "
            "Call this when the user asks about data quality, how much history they have, "
            "or whether the data is sufficient."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_pruning_stats",
        "description": (
            "Returns how many scenarios were pruned and why (structural, dominance, similarity). "
            "Call this when the user asks how scenarios were filtered, "
            "why only 25 scenarios remain, or about the pruning process."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_preferred_scenario",
        "description": (
            "Returns the scenario that best matches the user's stated preferences. "
            "Call this when the user asks about their preferred scenario, "
            "the scenario closest to their inputs, or how their preference compares to the best."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_attainment_summary",
        "description": (
            "Returns the count and percentage of territories on target, above, and below. "
            "Call this when the user asks how many territories are on track, "
            "what % are at risk, or for an attainment overview."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "scenario_id": {"type": "string", "description": "Optional"},
            },
            "required": ["session_id"],
        },
    },

    # ── Group 2: Comparison and Delta ────────────────────────────

    {
        "name": "get_delta",
        "description": (
            "Returns the delta between the best scenario and the preferred scenario. "
            "Call this when the user asks about trade-offs between scenarios, "
            "what changes if they use their preferred settings, or for delta insights."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "compare_two_scenarios",
        "description": (
            "Side-by-side metric comparison of any two scenarios by ID. "
            "Call this when the user names two specific scenarios and wants to compare them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "scenario_a": {"type": "string", "description": "First scenario ID e.g. SC_001"},
                "scenario_b": {"type": "string", "description": "Second scenario ID e.g. SC_002"},
            },
            "required": ["session_id", "scenario_a", "scenario_b"],
        },
    },

    {
        "name": "rank_scenarios_by_metric",
        "description": (
            "Rank all scenarios by a specific metric. "
            "Call this when the user asks which scenario is most attainable, "
            "most cost efficient, has the best growth alignment, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "metric": {
                    "type": "string",
                    "enum": ["P", "A", "C", "Ga", "V", "K", "S", "composite"],
                    "description": "P=Proportionality A=Attainability C=Consistency Ga=Growth V=Volatility K=Cost S=Spread",
                },
            },
            "required": ["session_id", "metric"],
        },
    },

    {
        "name": "compare_territories",
        "description": (
            "Per-territory goal difference between two scenarios. "
            "Call this when the user asks which territories benefit more from one scenario vs another."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "scenario_a": {"type": "string"},
                "scenario_b": {"type": "string"},
            },
            "required": ["session_id", "scenario_a", "scenario_b"],
        },
    },

    {
        "name": "get_territory_winners",
        "description": (
            "Returns territories closest to 100% attainment — those best served by the plan. "
            "Call this when the user asks which territories are best positioned, "
            "who wins under this plan, or for top performing territories."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "scenario_id": {"type": "string", "description": "Optional"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "rescore_with_preset",
        "description": (
            "Re-rank existing scenarios using a different scoring preset (fairness/growth/cost). "
            "Call this when the user wants to see how rankings change with a different mode, "
            "or asks 'what if I used growth mode instead'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "preset": {
                    "type": "string",
                    "enum": ["fairness", "growth", "cost"],
                },
            },
            "required": ["session_id", "preset"],
        },
    },

    {
        "name": "rescore_with_mode",
        "description": (
            "Show scenarios that match a specific plan mode. "
            "Call this when the user asks about a specific split like Hybrid 70/30 or Individual."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "plan_mode":  {"type": "string", "description": "e.g. Hybrid 70/30, Individual, Regional"},
            },
            "required": ["session_id", "plan_mode"],
        },
    },

    # ── Group 3: Guardrails ───────────────────────────────────────

    {
        "name": "run_guardrails",
        "description": (
            "Run all 4 guardrail rules: budget alignment, goal floor, goal ceiling, attainability floor. "
            "Call this when the user asks to validate the plan, run a sanity check, "
            "check if goals are reasonable, or run guardrails."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "scenario_id": {"type": "string", "description": "Optional — defaults to best"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "check_goal_ceiling",
        "description": (
            "Find territories with goals above 200% of average — outliers that may be unrealistic. "
            "Call this when the user asks if any goals are too high or unrealistic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "scenario_id": {"type": "string", "description": "Optional"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "check_goal_floor",
        "description": (
            "Find territories with goals below 50% of average — outliers that may be too easy. "
            "Call this when the user asks if any goals are too low or too easy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "scenario_id": {"type": "string", "description": "Optional"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "check_budget_alignment",
        "description": (
            "Check whether total goals are within ±20% of the national forecast budget. "
            "Call this when the user asks if the plan is within budget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "scenario_id": {"type": "string", "description": "Optional"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_flagged_territories",
        "description": (
            "Returns high-volatility territories that are also below attainment target. "
            "Call this when the user asks about risky territories, "
            "which territories need manual review, or where the plan is weakest."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "scenario_id": {"type": "string", "description": "Optional"},
            },
            "required": ["session_id"],
        },
    },

    # ── Group 4: What-if ──────────────────────────────────────────

    {
        "name": "get_conservative_scenario",
        "description": (
            "Returns the scenario with the lowest total goals — the most conservative option. "
            "Call this when the user asks for a safe plan, conservative option, or lowest goals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_aggressive_scenario",
        "description": (
            "Returns the scenario with the highest total goals — the most aggressive option. "
            "Call this when the user asks for an ambitious plan, stretch goals, or highest targets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "rerun_with_forecast",
        "description": (
            "Recalculate all goals and re-score using a new national forecast number. "
            "Call this when the user says 'what if my forecast is X' or wants to change the national forecast."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":            {"type": "string"},
                "new_national_forecast": {"type": "number", "description": "New national forecast value"},
            },
            "required": ["session_id", "new_national_forecast"],
        },
    },

    {
        "name": "rerun_with_method",
        "description": (
            "Show results for a specific scoring method (e.g. Quarter-wise, Historical 6 months). "
            "Call this when the user asks about a specific method."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":     {"type": "string"},
                "scoring_method": {"type": "string", "description": "e.g. Historical Total — 12 months"},
            },
            "required": ["session_id", "scoring_method"],
        },
    },

    {
        "name": "rerun_with_weights",
        "description": (
            "Show how many scenarios remain valid with new quarter weight bounds. "
            "Call this when the user asks about changing weight limits."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":          {"type": "string"},
                "min_quarter_weight":  {"type": "number"},
                "max_quarter_weight":  {"type": "number"},
            },
            "required": ["session_id", "min_quarter_weight", "max_quarter_weight"],
        },
    },

    # ── Group 5: File Info ────────────────────────────────────────

    {
        "name": "get_file_summary",
        "description": (
            "Summary of the uploaded file: territory count, region count, data quality. "
            "Call this when the user asks about their data, file contents, or coverage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_data_months",
        "description": (
            "Months of history available per territory. "
            "Call this when the user asks how much data they have or about history length."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_limited_data_territories",
        "description": (
            "Returns territories with less than 6 months of data — flagged for reliability concerns. "
            "Call this when the user asks about data gaps or unreliable territories."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_territory_raw",
        "description": (
            "Full data for one specific territory: sales history, trend, volatility, forecast, goal. "
            "Call this when the user asks about a specific territory by ID or name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id":   {"type": "string"},
                "territory_id": {"type": "string", "description": "Exact territory ID e.g. T_NE_01"},
            },
            "required": ["session_id", "territory_id"],
        },
    },

    {
        "name": "get_date_range",
        "description": (
            "Date range of the uploaded file based on months of history. "
            "Call this when the user asks about the time period covered by their data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    # ── Group 6: Export ───────────────────────────────────────────

    {
        "name": "get_excel_download_url",
        "description": (
            "Returns the URL to download the Excel export with 4 sheets. "
            "Call this when the user asks to download, export, or get the Excel file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_csv_download_url",
        "description": (
            "Returns the URL to download the CSV export. "
            "Call this when the user asks to download CSV or export flat data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "generate_summary_text",
        "description": (
            "Generates a structured summary of the recommended plan suitable for a presentation. "
            "Call this when the user asks for a summary, an executive overview, "
            "something to paste into a slide, or a report-ready output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_decision_report",
        "description": (
            "Returns the decision report: why The One was chosen, trade-offs, risk scenarios. "
            "Call this when the user asks why a scenario was selected or for a full justification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    # ── Group 7: IC Advisory ──────────────────────────────────────

    {
        "name": "score_ic_advisory",
        "description": (
            "Score all 21 IC use cases against the user's inputs and return top 3 ranked structures. "
            "Call this when the user asks which IC structure to use, "
            "wants a recommendation, or provides their IC planning context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lifecycle_stage":      {"type": "string"},
                "primary_objective":    {"type": "string"},
                "role_type":            {"type": "string"},
                "plan_type":            {"type": "string"},
                "pay_philosophy":       {"type": "string"},
                "goal_setting":         {"type": "string"},
                "collaboration":        {"type": "string"},
                "simplicity":           {"type": "string"},
                "territory_variability":{"type": "string"},
                "sales_motion":         {"type": "string"},
            },
            "required": ["lifecycle_stage", "primary_objective", "role_type",
                         "plan_type", "pay_philosophy", "goal_setting"],
        },
    },

    {
        "name": "get_ic_score_detail",
        "description": (
            "Explain why one IC use case ranked above another. "
            "Call this when the user asks why UC09 beat UC12, "
            "or wants to understand the scoring rationale."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lifecycle_stage":   {"type": "string"},
                "primary_objective": {"type": "string"},
                "role_type":         {"type": "string"},
                "plan_type":         {"type": "string"},
                "pay_philosophy":    {"type": "string"},
                "goal_setting":      {"type": "string"},
            },
            "required": ["lifecycle_stage", "primary_objective", "role_type",
                         "plan_type", "pay_philosophy", "goal_setting"],
        },
    },
    # ── Advisory / Unconstrained plans ───────────────────────────

    {
        "name": "get_advisory_guided_plan",
        "description": (
            "Returns the Advisory-Guided Plan — scenarios filtered by IC Advisory answers. "
            "Call this when the user asks about the advisory plan, their preference-based plan, "
            "or what the system recommends based on their IC Advisory inputs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "get_unconstrained_plan",
        "description": (
            "Returns the Unconstrained Plan — best scenario from all combinations with no filters. "
            "Call this when the user asks about the unconstrained plan or the full pool result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },

    {
        "name": "compare_advisory_vs_unconstrained",
        "description": (
            "Side-by-side comparison of the Advisory-Guided Plan vs the Unconstrained Plan. "
            "Call this when the user asks which plan to choose, wants to understand the difference "
            "between the two plans, or asks for a recommendation on which to proceed with."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },
]
