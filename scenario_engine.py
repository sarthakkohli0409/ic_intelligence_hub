"""
ProcDNA Intelligence — Scenario Engine
========================================
Two responsibilities:
  1. gen_scenarios  — generate all Key1 × Key2 × Key3 definitions with validity flags
  2. calc_goals     — compute Final Goal per territory for one scenario

Goal formula:
  Regional Basis  = NF × territory_market_share
  Individual Basis = territory_historical_sales (in chosen window)
  National Basis  = NF × territory_market_share  (only for 3-way split)

  For Historical Total (k3=0):
    Final Goal = rs × Regional Basis + is × Individual Basis + ns × National Basis

  For Quarter-wise (k3>0):
    Weighted Basis  = Q2×w1 + Q3×w2 + Q4×w3
    Weighted Share  = territory_weighted / national_weighted
    Regional Goal   = NF × weighted_share
    Individual Goal = territory_weighted_basis
    Final Goal      = rs × Regional Goal + is × Individual Goal + ns × National Goal

Attainment in goals uses forecast_q1 (real forecast) not last_q (actuals proxy).
"""

import numpy as np
from typing import List, Dict, Any, Tuple

KEY1 = [
    (1, "Historical Total — 6 months",          6,  False),
    (2, "Historical Total — 9 months",           9,  False),
    (3, "Historical Total — 12 months",          12, False),
    (4, "Quarter-wise Weighted — 12 months",     12, True),
    (5, "Absolute Growth",                       12, False),
    (6, "Percent Growth",                        12, False),
]
KEY2 = [
    (1, "Regional",   1.0, 0.0),
    (2, "Individual", 0.0, 1.0),
    (3, "Hybrid",     0.7, 0.3),
    (4, "Hybrid",     0.6, 0.4),
    (5, "Hybrid",     0.5, 0.5),
    (6, "Hybrid",     0.4, 0.6),
    (7, "Hybrid",     0.3, 0.7),
]


# ── Weight combos ─────────────────────────────────────────────────────────────
def weight_combos(mn: float, mx: float, step: float = 0.1) -> List[Tuple]:
    vals = [round(mn + i * step, 10) for i in range(round((mx - mn) / step) + 1)]
    out  = []
    for w1 in vals:
        for w2 in vals:
            w3 = round(1.0 - w1 - w2, 10)
            if mn <= w3 <= mx:
                out.append((round(w1, 3), round(w2, 3), round(w3, 3)))
    return out


def _k1_filter(scoring_method: str) -> List[int]:
    if scoring_method == "Run all methods":
        return [1, 2, 3, 4, 5, 6]
    if "6 month"      in scoring_method: return [1]
    if "9 month"      in scoring_method: return [2]
    if "Quarter"      in scoring_method: return [4]
    if "Absolute"     in scoring_method: return [5]
    if "Percent"      in scoring_method: return [6]
    return [3]  # default 12 month


def _k2_filter(plan_mode: str) -> List[int]:
    if plan_mode == "Run all splits": return list(range(1, 8))
    if "100/0"  in plan_mode: return [1]
    if "0/100"  in plan_mode: return [2]
    if "70/30"  in plan_mode: return [3]
    if "60/40"  in plan_mode: return [4]
    if "50/50"  in plan_mode: return [5]
    if "40/60"  in plan_mode: return [6]
    if "30/70"  in plan_mode: return [7]
    return list(range(1, 8))


# ── Scenario generation ───────────────────────────────────────────────────────
def gen_scenarios(
    scoring_method: str,
    plan_mode: str,
    include_nat: bool,
    nat_share: float,
    min_w: float,
    max_w: float,
) -> List[Dict]:
    k1_want = _k1_filter(scoring_method)
    k2_want = _k2_filter(plan_mode)
    # Guard: if filters return empty list (unexpected method string), default to all
    if not k1_want: k1_want = [1, 2, 3, 4]
    if not k2_want: k2_want = list(range(1, 8))
    wc      = weight_combos(min_w, max_w)
    ns      = nat_share if include_nat else 0.0
    pk1     = k1_want[-1]
    pk2     = k2_want[min(2, len(k2_want) - 1)]

    scenarios, ctr = [], 0
    for k1id, k1l, k1win, k1qw in KEY1:
        if k1id not in k1_want:
            continue
        for k2id, k2m, k2r, k2i in KEY2:
            if k2id not in k2_want:
                continue
            rs  = round(k2r * (1 - ns), 6)
            is2 = round(k2i * (1 - ns), 6)

            if k1qw:
                for k3i, (w1, w2, w3) in enumerate(wc):
                    tol = 0.001
                    ok  = (
                        min_w - tol <= w1 <= max_w + tol and
                        min_w - tol <= w2 <= max_w + tol and
                        min_w - tol <= w3 <= max_w + tol and
                        abs(w1 + w2 + w3 - 1.0) < tol
                    )
                    ctr += 1
                    scenarios.append({
                        "id": f"SC_{ctr:04d}", "k1": k1id, "k2": k2id, "k3": k3i + 1,
                        "method": k1l, "window": k1win, "mode": k2m,
                        "rs": rs, "is_": is2, "ns": ns,
                        "w1": w1, "w2": w2, "w3": w3,
                        "ok": ok,
                        "pref": (k1id == pk1 and k2id == pk2 and k3i == 0),
                    })
            else:
                ctr += 1
                scenarios.append({
                    "id": f"SC_{ctr:04d}", "k1": k1id, "k2": k2id, "k3": 0,
                    "method": k1l, "window": k1win, "mode": k2m,
                    "rs": rs, "is_": is2, "ns": ns,
                    "w1": 0.0, "w2": 0.0, "w3": 0.0,
                    "ok": True,
                    "pref": (k1id == pk1 and k2id == pk2),
                })
    return scenarios



# ── Goal calculation ──────────────────────────────────────────────────────────
def calc_goals(sc: Dict, territories: List[Dict], nf: float) -> List[Dict]:
    """
    Compute Final Goal for every territory under this scenario.

    METHOD 1 — Historical Aggregate (k1 in {1,2,3}, k3 == 0):
        MS  = territory_h / national_h  (market share)
        FG  = NF × MS  (same regardless of reg/ind split — both anchored to NF×MS)

    METHOD 2 — Historical Quarter-Wise (k1 == 4, k3 > 0):
        Q1/Q2/Q3 = most-recent / one-prior / two-prior quarter sales
        TW  = Q1×w1 + Q2×w2 + Q3×w3
        MS_terr = TW / SUM(all TW);  MS_reg = RegTW / SUM(unique RegTW)
        IG  = NF × MS_terr;  RG = NF × MS_reg
        FG  = is × IG + rs × (RG / n_in_region)

    METHOD 3 — Absolute Growth Quarter-Wise (k1 == 5, k3 > 0):
        Base        = Region Q4 (three-prior quarter) total sales
        growth_budget = NF − SUM(Method-2 FGs at same weights)
        GW_terr = (Q1−Q2)×w1 + (Q2−Q3)×w2 + (Q3−Q4)×w3
        GS_terr = GW_terr / SUM(all GW_terr)  [can be negative]
        GS_reg  = RegGW  / SUM(unique RegGW)
        FG  = Base + is × (GS_terr × budget) + rs × (GS_reg × budget / n_in_region)

    METHOD 4 — Percent Growth Quarter-Wise (k1 == 6, k3 > 0):
        Same as Method 3 but GW uses % deltas instead of absolute deltas.

    Attainment: forecast_q1 / FG  (both quarterly — NF is quarterly input)
    """
    rs  = sc["rs"]
    is_ = sc["is_"]
    w1, w2, w3 = sc["w1"], sc["w2"], sc["w3"]
    k1  = sc["k1"]
    k3  = sc["k3"]

    def _safe_div(a, b):
        return a / b if b != 0 else 0.0

    # ── METHOD 1 — Historical Aggregate ──────────────────────────────────────
    if k3 == 0 and k1 in (1, 2, 3):
        nat_ib = sum(
            t["h6"] if k1 == 1 else (t["h9"] if k1 == 2 else t["h12"])
            for t in territories
        )
        goals = []
        for t in territories:
            ib  = t["h6"] if k1 == 1 else (t["h9"] if k1 == 2 else t["h12"])
            ms  = _safe_div(ib, nat_ib)
            fg  = round(nf * ms, 2)

            fc   = t.get("forecast_q1", t.get("last_q", 0.0))
            att  = _safe_div(fc, fg)
            goals.append({
                "sid": sc["id"], "tid": t["territory_id"], "rid": t["region_id"],
                "tname": t["territory_name"], "rname": t["region_name"],
                "rb": fg, "ib": round(ib, 2), "nb": 0.0,
                "rg": fg, "ig": fg, "ng": 0.0,
                "fg": fg, "fc": round(fc, 2), "fc_q": round(fc, 2),
                "att": round(att, 4),
                "best_model": t.get("best_model", "—"),
            })
        return goals

    # ── METHOD 2 — Historical Quarter-Wise ───────────────────────────────────
    if k1 == 4 and k3 > 0:
        from collections import defaultdict

        tw_map = {
            t["territory_id"]: t["q1"] * w1 + t["q2"] * w2 + t["q3"] * w3
            for t in territories
        }
        nat_tw = sum(tw_map.values())

        rw_map: Dict[str, float] = defaultdict(float)
        rn_map: Dict[str, int]   = defaultdict(int)
        for t in territories:
            rw_map[t["region_id"]] += tw_map[t["territory_id"]]
            rn_map[t["region_id"]] += 1
        nat_rw = sum(rw_map.values())

        goals = []
        for t in territories:
            tid = t["territory_id"]
            rid = t["region_id"]
            tw  = tw_map[tid]
            rw  = rw_map[rid]
            n   = rn_map[rid]

            ig = nf * _safe_div(tw, nat_tw)
            rg = nf * _safe_div(rw, nat_rw)
            fg = round(is_ * ig + rs * _safe_div(rg, n), 2)

            fc   = t.get("forecast_q1", t.get("last_q", 0.0))
            att  = _safe_div(fc, fg)
            goals.append({
                "sid": sc["id"], "tid": tid, "rid": rid,
                "tname": t["territory_name"], "rname": t["region_name"],
                "rb": round(rg, 2), "ib": round(tw, 2), "nb": 0.0,
                "rg": round(rg, 2), "ig": round(ig, 2), "ng": 0.0,
                "fg": fg, "fc": round(fc, 2), "fc_q": round(fc, 2),
                "att": round(att, 4),
                "best_model": t.get("best_model", "—"),
            })
        return goals

    # ── METHODS 3 & 4 — Growth Quarter-Wise ──────────────────────────────────
    from collections import defaultdict

    # Step 1 — Method-2 base to compute growth_budget
    # Force k3=1 with equal weights if no quarter weights set (k3=0 would recurse infinitely)
    _w1, _w2, _w3 = (sc["w1"], sc["w2"], sc["w3"]) if sc["k3"] > 0 else (0.333, 0.333, 0.334)
    m2_sc = {**sc, "k1": 4, "k3": 1, "w1": _w1, "w2": _w2, "w3": _w3}
    m2_goals = calc_goals(m2_sc, territories, nf)
    growth_budget = nf - sum(g["fg"] for g in m2_goals)

    # Step 2 — Region Q4 total (three-prior quarter) as base per territory
    rq4_map: Dict[str, float] = defaultdict(float)
    rn_map2: Dict[str, int]   = defaultdict(int)
    for t in territories:
        rq4_map[t["region_id"]] += t.get("q4", 0.0)
        rn_map2[t["region_id"]] += 1

    # Step 3 — Growth weighted scores
    def _gw_abs(t):
        d1 = t.get("q1", 0) - t.get("q2", 0)
        d2 = t.get("q2", 0) - t.get("q3", 0)
        d3 = t.get("q3", 0) - t.get("q4", 0)
        return d1 * w1 + d2 * w2 + d3 * w3

    def _gw_pct(t):
        q1 = t.get("q1", 0); q2 = t.get("q2", 0)
        q3 = t.get("q3", 0); q4 = t.get("q4", 0)
        return (_safe_div(q1 - q2, q2) * w1 +
                _safe_div(q2 - q3, q3) * w2 +
                _safe_div(q3 - q4, q4) * w3)

    gw_fn  = _gw_abs if k1 == 5 else _gw_pct
    gw_t   = {t["territory_id"]: gw_fn(t) for t in territories}
    nat_gw = sum(gw_t.values())

    gw_r: Dict[str, float] = defaultdict(float)
    for t in territories:
        gw_r[t["region_id"]] += gw_t[t["territory_id"]]
    nat_rgw = sum(gw_r.values())

    goals = []
    for t in territories:
        tid  = t["territory_id"]
        rid  = t["region_id"]
        n    = rn_map2[rid]
        base = rq4_map[rid]

        gs_t = _safe_div(gw_t[tid],  nat_gw)
        gs_r = _safe_div(gw_r[rid],  nat_rgw)
        ig_g = gs_t * growth_budget
        rg_g = gs_r * growth_budget
        fg   = round(base + is_ * ig_g + rs * _safe_div(rg_g, n), 2)

        fc   = t.get("forecast_q1", t.get("last_q", 0.0))
        att  = _safe_div(fc, fg) if fg > 0 else 0.0
        goals.append({
            "sid": sc["id"], "tid": tid, "rid": rid,
            "tname": t["territory_name"], "rname": t["region_name"],
            "rb": round(base, 2), "ib": round(ig_g, 2), "nb": 0.0,
            "rg": round(rg_g, 2), "ig": round(ig_g, 2), "ng": 0.0,
            "fg": fg, "fc": round(fc, 2), "fc_q": round(fc, 2),
            "att": round(att, 4),
            "best_model": t.get("best_model", "—"),
        })
    return goals
