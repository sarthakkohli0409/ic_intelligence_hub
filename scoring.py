"""
ProcDNA Intelligence — Scoring Engine v2
=========================================
7 metrics (all normalised 0-1):
  P   Proportionality   goals ~ territory potential          (Pearson r)
  A   Attainability     % territories in F_i/G_i ∈ [0.85, 1.15]
  C   Consistency       1 - avg(CV per region)
  Ga  Growth Alignment  trending territories get higher goals (Pearson r)
  V   Volatility Penalty stable territories not penalised vs volatile
  K   Cost Efficiency   total goals vs national forecast budget
  S   Goal Spread       1 - normalised P90/P10 ratio

3 presets — each a weight vector across all 7 metrics:
  fairness  default — balanced proportionality, consistency, attainability
  growth    ambitious — prioritise attainability, growth alignment, spread
  cost      conservative — prioritise budget adherence, volatility control

Confidence weighting (Section V of improvement plan):
  confidence_i  = 1 - wmape_i  (per territory, from forecaster)
  adjusted A    = sum(on_target_i * confidence_i) / sum(confidence_i)
  scenario conf = mean(confidence_i)
"""

import numpy as np
from typing import List, Dict, Any

try:
    from scipy import stats as sp
    SCIPY = True
except ImportError:
    SCIPY = False

# ── Preset weight tables ──────────────────────────────────────────────────────
# Keys must match the 7 metric keys returned by compute_all_metrics()
PRESETS: Dict[str, Dict] = {
    "fairness": {
        "label": "Fairness Mode",
        "desc":  "Prioritises proportionality, consistency, and attainability. "
                 "Best for mature IC plans where equity across territories matters most.",
        "w": {"P": 0.30, "A": 0.25, "C": 0.20, "Ga": 0.10, "V": 0.10, "K": 0.03, "S": 0.02},
    },
    "growth": {
        "label": "Growth Mode",
        "desc":  "Prioritises attainability, growth alignment, and spread. "
                 "Best for aggressive growth cycles where rewarding trending territories drives performance.",
        "w": {"A": 0.30, "Ga": 0.25, "S": 0.15, "P": 0.15, "C": 0.10, "V": 0.03, "K": 0.02},
    },
    "cost": {
        "label": "Cost Mode",
        "desc":  "Prioritises budget adherence, volatility control, and consistency. "
                 "Best for cost-constrained cycles where payout predictability is critical.",
        "w": {"K": 0.30, "V": 0.25, "C": 0.20, "A": 0.15, "P": 0.07, "Ga": 0.02, "S": 0.01},
    },
}

ATTAINMENT_LO = 0.85
ATTAINMENT_HI = 1.15


# ── Pearson helper ────────────────────────────────────────────────────────────
def _pearson(x: List[float], y: List[float]) -> float:
    """Pearson r normalised to [0, 1]. Returns 0.5 if insufficient data."""
    if not SCIPY or len(x) < 3:
        return 0.5
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.5
    r, _ = sp.pearsonr(x, y)
    return round(float((r + 1) / 2), 4)


# ── Individual metric functions ───────────────────────────────────────────────

def metric_proportionality(goals: List[Dict], feat: Dict) -> float:
    """P = (1 + corr(G, T)) / 2   where T = territory_potential."""
    gv = [g["fg"] for g in goals if g["tid"] in feat and feat[g["tid"]]["pot"] > 0]
    pv = [feat[g["tid"]]["pot"] for g in goals if g["tid"] in feat and feat[g["tid"]]["pot"] > 0]
    return _pearson(gv, pv)


def metric_attainability(goals: List[Dict], confidences: Dict[str, float]) -> float:
    """
    A = confidence-weighted % of territories where F_i/G_i ∈ [0.85, 1.15].
    If no confidence data, falls back to simple count ratio.
    """
    if not goals:
        return 0.0
    if confidences:
        w_sum = sum(confidences.get(g["tid"], 1.0) for g in goals)
        if w_sum > 0:
            w_on = sum(
                confidences.get(g["tid"], 1.0)
                for g in goals
                if ATTAINMENT_LO <= g["att"] <= ATTAINMENT_HI
            )
            return round(w_on / w_sum, 4)
    on = sum(1 for g in goals if ATTAINMENT_LO <= g["att"] <= ATTAINMENT_HI)
    return round(on / len(goals), 4)


def metric_consistency(goals: List[Dict]) -> float:
    """C = 1 - (1/R) * sum(sigma_r / mu_r)  per region."""
    rg: Dict[str, List[float]] = {}
    for g in goals:
        rg.setdefault(g["rid"], []).append(g["fg"])
    cvs = []
    for vals in rg.values():
        mu = np.mean(vals)
        if mu > 0 and len(vals) > 1:
            cvs.append(np.std(vals) / mu)
    if not cvs:
        return 0.5
    return round(float(max(0.0, 1.0 - min(float(np.mean(cvs)), 1.0))), 4)


def metric_growth_alignment(goals: List[Dict], feat: Dict) -> float:
    """Ga = (1 + corr(trend_i, goal_growth_i)) / 2."""
    tv, ggv = [], []
    for g in goals:
        f = feat.get(g["tid"])
        if f and f["avg_q"] > 0:
            tv.append(f["trend"])
            ggv.append((g["fg"] - f["avg_q"]) / f["avg_q"])
    return _pearson(tv, ggv)


def metric_volatility_penalty(goals: List[Dict], feat: Dict) -> float:
    """
    V = 1 - mean(penalty_i)
    penalty_i = 1  if territory is high-vol AND goal_growth > avg
    penalty_i = 0  otherwise
    Normalised to [0, 1].
    """
    if not goals:
        return 0.5
    penalties = []
    avg_growth = np.mean([
        (g["fg"] - feat[g["tid"]]["avg_q"]) / feat[g["tid"]]["avg_q"]
        for g in goals
        if g["tid"] in feat and feat[g["tid"]]["avg_q"] > 0
    ]) if goals else 0.0
    for g in goals:
        f = feat.get(g["tid"])
        if not f or f["avg_q"] <= 0:
            continue
        goal_growth = (g["fg"] - f["avg_q"]) / f["avg_q"]
        is_high_vol = f["vband"] == "High Volatility"
        is_high_growth = goal_growth > avg_growth
        penalties.append(1.0 if (is_high_vol and is_high_growth) else 0.0)
    if not penalties:
        return 0.5
    return round(float(max(0.0, 1.0 - float(np.mean(penalties)))), 4)


def metric_cost_efficiency(goals: List[Dict], national_forecast: float) -> float:
    """
    K = 1 - |total_goals - NF| / NF
    Uses national forecast as the budget proxy.
    Perfect K=1 when total goals == NF exactly.
    """
    if national_forecast <= 0:
        return 0.5
    total = sum(g["fg"] for g in goals)
    k = 1.0 - abs(total - national_forecast) / national_forecast
    return round(float(max(0.0, min(1.0, k))), 4)


def metric_goal_spread(goals: List[Dict]) -> float:
    """
    S = max(0, 1 - (P90/P10 - 1)) normalised to [0, 1].
    S=1 means perfectly equal spread (P90 == P10).
    S drops as the ratio grows — punishes extreme inequality between territories.
    """
    vals = [g["fg"] for g in goals if g["fg"] > 0]
    if len(vals) < 4:
        return 0.5
    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))
    if p10 <= 0:
        return 0.0
    ratio = p90 / p10
    # ratio=1 → S=1 (ideal), ratio=10 → S=0 (terrible), capped at 0
    s = max(0.0, 1.0 - (ratio - 1.0) / 9.0)
    return round(s, 4)


# ── Master scorer ─────────────────────────────────────────────────────────────

def compute_all_metrics(
    sid: str,
    goals: List[Dict],
    feat: Dict,
    national_forecast: float,
    confidences: Dict[str, float],
) -> Dict[str, float]:
    """Compute all 7 metrics for one scenario. Returns raw metric dict."""
    return {
        "P":  metric_proportionality(goals, feat),
        "A":  metric_attainability(goals, confidences),
        "C":  metric_consistency(goals),
        "Ga": metric_growth_alignment(goals, feat),
        "V":  metric_volatility_penalty(goals, feat),
        "K":  metric_cost_efficiency(goals, national_forecast),
        "S":  metric_goal_spread(goals),
    }


def composite_score(metrics: Dict[str, float], preset: str = "fairness") -> float:
    """Weighted sum of 7 metrics using the chosen preset weight vector."""
    if preset not in PRESETS:
        preset = "fairness"
    w = PRESETS[preset]["w"]
    score = sum(w.get(k, 0.0) * v for k, v in metrics.items())
    return round(float(score), 4)


def build_fairness_block(
    sid: str,
    goals: List[Dict],
    feat: Dict,
    national_forecast: float,
    confidences: Dict[str, float],
    preset: str = "fairness",
    ic_plan_type: str = "",
    scenario_mode: str = "",
    ic_boost_lambda: float = 0.05,
) -> Dict[str, Any]:
    """
    Full fairness block for one scenario — metrics, composite, confidence,
    attainment stats, IC Advisory boost.
    """
    metrics = compute_all_metrics(sid, goals, feat, national_forecast, confidences)
    score   = composite_score(metrics, preset)

    # IX. IC Advisory consistency boost — add λ if scenario mode matches advisory recommendation
    ic_boost = 0.0
    if ic_plan_type and scenario_mode:
        mode_lower = scenario_mode.lower()
        plan_lower = ic_plan_type.lower()
        if ("hybrid" in plan_lower and "hybrid" in mode_lower) or \
           ("individual" in plan_lower and "individual" in mode_lower) or \
           ("regional" in plan_lower and "regional" in mode_lower):
            ic_boost = ic_boost_lambda
    boosted_score = round(min(1.0, score + ic_boost), 4)

    # Attainment stats
    n   = len(goals)
    ot  = [g for g in goals if ATTAINMENT_LO <= g["att"] <= ATTAINMENT_HI]
    ab  = [g for g in goals if g["att"] > ATTAINMENT_HI]
    bel = [g for g in goals if g["att"] < ATTAINMENT_LO]
    tot = sum(g["fg"] for g in goals)

    # Scenario confidence = mean(confidence_i)
    conf_vals = [confidences.get(g["tid"], 0.5) for g in goals]
    scen_conf = round(float(np.mean(conf_vals)), 4) if conf_vals else 0.5

    return {
        "sid":             sid,
        # 7 raw metrics
        "P": metrics["P"], "A": metrics["A"], "C": metrics["C"],
        "Ga": metrics["Ga"], "V": metrics["V"], "K": metrics["K"], "S": metrics["S"],
        # Legacy keys kept for backward compat with frontend
        "prop": metrics["P"], "att": metrics["A"], "cons": metrics["C"],
        "ga": metrics["Ga"], "vp": metrics["V"],
        # Scores
        "score":         score,
        "boosted_score": boosted_score,
        "ic_boost":      ic_boost,
        "preset":        preset,
        "confidence":    scen_conf,
        # Attainment
        "on_target":   len(ot),
        "above":       len(ab),
        "below":       len(bel),
        "pct_on":      round(len(ot) / n, 4) if n else 0,
        "pct_ab":      round(len(ab) / n, 4) if n else 0,
        "pct_bl":      round(len(bel) / n, 4) if n else 0,
        "total_goals": round(tot, 2),
        "gap":         round(tot - national_forecast, 2),
    }

# ═════════════════════════════════════════════════════════════════
# Advisory filter — hard post-generation filter for Advisory-Guided track
# ═════════════════════════════════════════════════════════════════

# Maps IC Advisory Goal Setting answer → allowed Key1 IDs
_GOAL_SETTING_TO_KEY1 = {
    "Historical (6/9/12 months)": [1, 2, 3],   # all Historical Total, no Quarter-wise
    "Historical":                  [1, 2, 3],
    "Absolute Growth-based":       [1, 2, 3, 4],
    "Percent Growth-based":        [1, 2, 3, 4],
    "Forecast-driven":             [1, 2, 3, 4],
    "Scenario-based":              [1, 2, 3, 4],
}

# Maps IC Advisory Plan Type → allowed Key2 IDs
_PLAN_TYPE_TO_KEY2 = {
    "Individual":      [2],              # Individual only 0/100
    "Regional / Team": [1],              # Regional only 100/0
    "Hybrid":          [3, 4, 5, 6, 7],  # All Hybrid splits
    "No Preference":   [1, 2, 3, 4, 5, 6, 7],  # no filter
}


def filter_by_advisory(
    scenarios: list,
    ic_plan_type: str = "",
    ic_goal_setting: str = "",
) -> list:
    """
    Filter scenario list to only those compatible with IC Advisory answers.
    Used to build the Advisory-Guided track.

    Rules:
      ic_plan_type     → filters by Key2 (plan mode / split)
      ic_goal_setting  → filters by Key1 (scoring method)

    If both are empty (advisory skipped) → returns all scenarios unchanged.
    If a value is set but not in the map → no filter applied for that dimension.
    """
    if not ic_plan_type and not ic_goal_setting:
        return scenarios  # advisory skipped — no filter

    allowed_k1 = None
    allowed_k2 = None

    if ic_goal_setting:
        # Match on first key that is a substring of the answer
        for key, k1_ids in _GOAL_SETTING_TO_KEY1.items():
            if key.lower() in ic_goal_setting.lower() or ic_goal_setting.lower() in key.lower():
                allowed_k1 = k1_ids
                break

    if ic_plan_type:
        for key, k2_ids in _PLAN_TYPE_TO_KEY2.items():
            if key.lower() in ic_plan_type.lower() or ic_plan_type.lower() in key.lower():
                allowed_k2 = k2_ids
                break

    filtered = []
    for sc in scenarios:
        if allowed_k1 is not None and sc.get("k1") not in allowed_k1:
            continue
        if allowed_k2 is not None and sc.get("k2") not in allowed_k2:
            continue
        filtered.append(sc)

    return filtered


def has_advisory_answers(ic_plan_type: str, ic_goal_setting: str) -> bool:
    """Returns True if the user completed IC Advisory before Goal Setting."""
    return bool(ic_plan_type.strip() or ic_goal_setting.strip())

