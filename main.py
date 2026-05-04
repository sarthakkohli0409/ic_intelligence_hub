"""
ProcDNA Intelligence — Goals Service v2
=========================================
Wires: parse_file → run_forecasts → gen_scenarios → calc_goals
       → build_fairness_block → prune → export

Routes:
  GET  /health
  POST /api/goals/generate          ← main pipeline
  GET  /api/goals/session/{sid}/all ← paginated scenario list
  GET  /api/compare/session/{sid}   ← delta insights best vs preferred
  GET  /api/guardrails/{sid}        ← guardrail check on best scenario
  GET  /api/export/session/{sid}/excel
  GET  /api/export/session/{sid}/csv
"""

import io
import pickle
import sys
import uuid
import warnings

sys.setrecursionlimit(5000)  # statsmodels optimisers can hit default 1000 on complex series
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, HTMLResponse
import pathlib
_HERE = pathlib.Path(__file__).parent

from scoring        import build_fairness_block, PRESETS, filter_by_advisory, has_advisory_answers
from forecaster     import run_forecasts
from scenario_engine import gen_scenarios, calc_goals
from pruner         import prune
from export         import to_excel, to_csv, guardrails_check, delta_insights

warnings.filterwarnings("ignore")

app = FastAPI(
    title="ProcDNA Intelligence — Goals Service",
    description="Scenario generation, fairness scoring, forecast integration, and export.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse
from fastapi import Request

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return JSON for all unhandled exceptions instead of plain text 500."""
    import traceback
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )

# ── Disk-based session store ──────────────────────────────────────────────────
# Mirrors the parse cache approach — survives Render free tier restarts.
_SESS_DIR = pathlib.Path("/tmp/procdna_sessions")
_SESS_DIR.mkdir(parents=True, exist_ok=True)

def _sess_path(sid: str) -> pathlib.Path:
    safe = "".join(c for c in sid if c.isalnum() or c == "-")
    return _SESS_DIR / f"{safe}.pkl"

def _sess_set(sid: str, data: dict) -> None:
    try:
        with open(_sess_path(sid), "wb") as f:
            pickle.dump(data, f, protocol=4)
    except Exception as e:
        print(f"[sess_set] WARNING: {e}")

def _sess_get(sid: str) -> dict | None:
    p = _sess_path(sid)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[sess_get] WARNING: {e}")
        return None

_sessions: dict = {}  # lightweight in-memory sentinel (sid → True)

REQUIRED_COLS = {"region_id", "region_name", "territory_id",
                 "territory_name", "transaction_date", "metric_value"}

# Automatic column alias map — Cabo file format → standard names
CABO_COL_MAP = {
    "wk_end_dt":  "transaction_date",
    "terr_id":    "territory_id",
    "terr_nm":    "territory_name",
    "regn_id":    "region_id",
    "regn_nm":    "region_name",
    "regn_name":  "region_name",
    "area_id":    "area_id_orig",   # keep but don't overwrite region_id
    "sales":      "metric_value",
}


# ─────────────────────────────────────────────────────────────────────────────
# File parser — returns territory feature dicts + raw df for forecaster
# ─────────────────────────────────────────────────────────────────────────────
def parse_file(file_bytes: bytes, filename: str, lookback_months: int):
    try:
        df = (
            pd.read_csv(io.BytesIO(file_bytes))
            if filename.lower().endswith(".csv")
            else pd.read_excel(io.BytesIO(file_bytes))
        )
    except Exception as e:
        raise HTTPException(422, f"Cannot read file: {e}")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Auto-remap Cabo/custom column aliases → standard names
    remap = {c: CABO_COL_MAP[c] for c in df.columns if c in CABO_COL_MAP}
    if remap:
        df = df.rename(columns=remap)

    # Fill missing optional columns with defaults
    if "territory_name" not in df.columns:
        df["territory_name"] = df.get("territory_id", df.iloc[:, 0]).astype(str)
    if "region_id" not in df.columns:
        df["region_id"] = "REG_01"
    if "region_name" not in df.columns:
        df["region_name"] = df.get("region_id", pd.Series(["Region"] * len(df))).astype(str)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise HTTPException(422, f"Missing columns: {', '.join(sorted(missing))}. File has: {list(df.columns)}")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce", dayfirst=False)
    df["metric_value"]     = pd.to_numeric(df["metric_value"], errors="coerce").fillna(0)
    df = df.dropna(subset=["transaction_date"]).sort_values("transaction_date")
    if df.empty:
        raise HTTPException(422, "No valid rows after date parsing.")

    # Aggregate weekly → monthly (Cabo file is weekly)
    rows_per_terr = len(df) / max(df["territory_id"].nunique(), 1)
    if rows_per_terr > 20:
        df["transaction_date"] = df["transaction_date"].dt.to_period("M").dt.to_timestamp()
        agg = {"metric_value": "sum"}
        for col in ["territory_name", "region_id", "region_name"]:
            if col in df.columns:
                agg[col] = "first"
        df = df.groupby(["territory_id", "transaction_date"]).agg(agg).reset_index()
        df = df.sort_values("transaction_date")

    ref = df["transaction_date"].max()

    def ql(d):
        m = (ref.year - d.year) * 12 + (ref.month - d.month)
        return "Q1" if m < 3 else "Q2" if m < 6 else "Q3" if m < 9 else "Q4" if m < 12 else "Q5+"

    df["ql"] = df["transaction_date"].apply(ql)

    def n_hist(months):
        cut = ref - pd.DateOffset(months=months)
        return df[df["transaction_date"] > cut]["metric_value"].sum()

    nat6, nat9, nat12 = n_hist(6), n_hist(9), n_hist(12)

    territories = []
    for (tid, rid), grp in df.groupby(["territory_id", "region_id"]):
        grp  = grp.sort_values("transaction_date")
        last = grp.iloc[-1]

        def qs(l):  return float(grp[grp["ql"] == l]["metric_value"].sum())
        def h(m):
            cut = ref - pd.DateOffset(months=m)
            return float(grp[grp["transaction_date"] > cut]["metric_value"].sum())

        q1, q2, q3, q4 = qs("Q1"), qs("Q2"), qs("Q3"), qs("Q4")
        h6, h9, h12    = h(6), h(9), h(12)
        pot            = h(lookback_months)

        def pg(n, o): return float((n - o) / o) if o != 0 else 0.0
        pg21, pg32, pg43 = pg(q1, q2), pg(q2, q3), pg(q3, q4)
        trend = 0.5 * pg21 + 0.3 * pg32 + 0.2 * pg43
        vol   = float(np.std([pg21, pg32, pg43]))

        territories.append({
            "territory_id":   str(tid),
            "region_id":      str(rid),
            "territory_name": str(last.get("territory_name", tid)),
            "region_name":    str(last.get("region_name",    rid)),
            "q1": q1, "q2": q2, "q3": q3, "q4": q4,
            "h6": h6, "h9": h9, "h12": h12,
            "pot": pot,
            "avg_q": float(grp.groupby("ql")["metric_value"].sum().mean()),
            "s6":  h6  / nat6  if nat6  > 0 else 0.0,
            "s9":  h9  / nat9  if nat9  > 0 else 0.0,
            "s12": h12 / nat12 if nat12 > 0 else 0.0,
            "trend":  round(trend, 6),
            "vol":    round(vol, 6),
            "vband":  (
                "Stable Territory" if vol < 0.05
                else "Moderate Volatility" if vol < 0.12
                else "High Volatility"
            ),
            "last_q": q1,
        })

    return territories, df


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    p = _HERE / "app.html"
    if not p.exists():
        return HTMLResponse("<h2>app.html not found — make sure it is deployed alongside main.py</h2>", 404)
    return FileResponse(str(p))

@app.get("/health")
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "ProcDNA Intelligence — Goals Service",
        "version": "2.0.0",
        "presets": list(PRESETS.keys()),
    }


@app.post("/api/goals/generate")
async def generate(
    file:                   UploadFile = File(...),
    national_forecast:      float      = Form(...),
    lookback_months:        int        = Form(12),
    territory_potential_basis: str     = Form("Historical sales value"),
    volatility_sensitivity: str        = Form("Apply a smoothing factor"),
    growth_trend_weight:    str        = Form("Medium"),
    quarter_reference:      str        = Form("All 4 quarters"),
    scoring_method:         str        = Form("Run all methods"),
    plan_mode:              str        = Form("Run all splits"),
    include_national:       bool       = Form(False),
    national_share:         float      = Form(0.1),
    min_quarter_weight:     float      = Form(0.1),
    max_quarter_weight:     float      = Form(0.7),
    # New v2 fields
    preset:                 str        = Form("fairness"),   # fairness / growth / cost
    ic_plan_type:           str        = Form(""),           # from IC Advisory Unit 1
    ic_goal_setting:        str        = Form(""),           # from IC Advisory Unit 1
    ic_boost_lambda:        float      = Form(0.05),         # λ for IC Advisory consistency boost
):
    """
    Full pipeline:
    1. Parse file → territory features
    2. Run 15-model forecast competition per territory
    3. Generate all Key1×Key2×Key3 scenario definitions
    4. Calculate goals for every valid scenario × territory
    5. Score each scenario across 7 metrics with preset weights
    6. Prune: structural → dominance → similarity → top-25
    7. Select "The One" via Pareto frontier
    8. Return results + session ID
    """
    # ── 1. Parse ───────────────────────────────────────────────────
    fb = await file.read()
    if not fb:
        raise HTTPException(400, "Uploaded file is empty.")

    # parse_file now handles all column remapping automatically via CABO_COL_MAP
    territories, raw_df = parse_file(fb, file.filename, lookback_months)
    # Remove unassigned / zero-sales territories
    territories = [
        t for t in territories
        if t.get("h12", 0) > 1
        and "unassign" not in str(t.get("territory_name", "")).lower()
        and "unassign" not in str(t.get("territory_id", "")).lower()
    ]
    if len(territories) < 2:
        raise HTTPException(422, f"Need at least 2 territories. Found: {len(territories)}")

    # ── 2. Forecast ────────────────────────────────────────────────
    territories = run_forecasts(raw_df, territories, fast_mode=True)

    # Build confidence map: territory_id → confidence
    confidences = {t["territory_id"]: t.get("confidence", 0.5) for t in territories}
    feat_map    = {t["territory_id"]: t for t in territories}

    # Data tier summary
    tiers = [t.get("data_tier", "full") for t in territories]
    tier_counts = {
        "limited":  tiers.count("limited"),
        "moderate": tiers.count("moderate"),
        "full":     tiers.count("full"),
    }
    overall_tier = (
        "limited"  if tier_counts["limited"]  > len(tiers) * 0.5 else
        "moderate" if tier_counts["full"]     < len(tiers) * 0.5 else
        "full"
    )

    # ── 3. Generate scenario definitions ──────────────────────────
    # Force national_share to 0 when not using national component
    if not include_national:
        national_share = 0.0

    all_defs  = gen_scenarios(scoring_method, plan_mode, include_national,
                              national_share, min_quarter_weight, max_quarter_weight)
    valid_defs = [s for s in all_defs if s["ok"]]
    if not valid_defs:
        raise HTTPException(422, "No valid scenarios. Try relaxing min/max quarter weight bounds.")

    # ── 4–5. Calculate goals + score each valid scenario ──────────
    if preset not in PRESETS:
        preset = "fairness"

    raw_results = []
    for sc in valid_defs:
        goals   = calc_goals(sc, territories, national_forecast)
        fairness = build_fairness_block(
            sid=sc["id"],
            goals=goals,
            feat=feat_map,
            national_forecast=national_forecast,
            confidences=confidences,
            preset=preset,
            ic_plan_type=ic_plan_type,
            scenario_mode=sc["mode"],
            ic_boost_lambda=ic_boost_lambda,
        )
        raw_results.append({
            "scenario":  sc,
            "goals":     goals,
            "fairness":  fairness,
            "is_best":   False,
            "is_pref":   sc["pref"],
            "rank":      0,
        })

    # ── 6–7. Prune + Pareto "The One" ─────────────────────────────
    # ── Unconstrained track — full pool, no filter ────────────────
    results, prune_stats, best_id = prune(raw_results, national_forecast)
    pref = next((r for r in results if r.get("is_pref")), None)

    # ── Advisory-Guided track — filter by IC Advisory answers ─────
    advisory_result = None
    if has_advisory_answers(ic_plan_type, ic_goal_setting):
        advisory_valid = filter_by_advisory(
            valid_defs,
            ic_plan_type=ic_plan_type,
            ic_goal_setting=ic_goal_setting,
        )
        if advisory_valid:
            advisory_raw = []
            for sc in advisory_valid:
                goals   = calc_goals(sc, territories, national_forecast)
                fairness = build_fairness_block(
                    sid=sc["id"], goals=goals, feat=feat_map,
                    national_forecast=national_forecast,
                    confidences=confidences, preset=preset,
                    ic_plan_type=ic_plan_type,
                    scenario_mode=sc["mode"],
                    ic_boost_lambda=ic_boost_lambda,
                )
                advisory_raw.append({
                    "scenario": sc, "goals": goals, "fairness": fairness,
                    "is_best": False, "is_pref": sc["pref"], "rank": 0,
                })
            adv_results, adv_prune_stats, adv_best_id = prune(advisory_raw, national_forecast)
            adv_best = next(r for r in adv_results if r["scenario"]["id"] == adv_best_id)
            advisory_result = {
                "best_scenario_id":  adv_best_id,
                "method":            adv_best["scenario"]["method"],
                "mode":              adv_best["scenario"]["mode"],
                "composite":         adv_best["fairness"]["boosted_score"],
                "confidence":        adv_best["fairness"].get("confidence", "—"),
                "on_target":         adv_best["fairness"]["on_target"],
                "total_goals":       adv_best["fairness"]["total_goals"],
                "scenarios_in_pool": len(advisory_valid),
                "scenarios_after_pruning": len(adv_results),
                "breakdown": {
                    "P": adv_best["fairness"]["P"],
                    "A": adv_best["fairness"]["A"],
                    "C": adv_best["fairness"]["C"],
                    "Ga": adv_best["fairness"]["Ga"],
                    "V": adv_best["fairness"]["V"],
                    "K": adv_best["fairness"]["K"],
                    "S": adv_best["fairness"]["S"],
                },
                "filters_applied": {
                    "plan_type":     ic_plan_type,
                    "goal_setting":  ic_goal_setting,
                },
            }
        else:
            advisory_result = {
                "message": "No scenarios matched your IC Advisory preferences. Showing unconstrained plan only.",
                "filters_applied": {"plan_type": ic_plan_type, "goal_setting": ic_goal_setting},
            }

    # ── Forecast summary metrics (must be before session store) ──────
    avg_wmape  = round(float(np.mean([t.get("wmape", 0.2) for t in territories])), 4)
    avg_conf   = round(float(np.mean([t.get("confidence", 0.5) for t in territories])), 4)
    model_wins = {}
    for t in territories:
        bm = t.get("best_model", "—")
        model_wins[bm] = model_wins.get(bm, 0) + 1
    top_model = max(model_wins, key=model_wins.get) if model_wins else "—"

    # ── 8. Store session ───────────────────────────────────────────
    sid = str(uuid.uuid4())
    _sess_data = {
        "results":          results,
        "best_id":          best_id,
        "preferred_id":     pref["scenario"]["id"] if pref else None,
        "nf":               national_forecast,
        "preset":           preset,
        "tier_counts":      tier_counts,
        "overall_tier":     overall_tier,
        "prune_stats":      prune_stats,
        "territories":      territories,
        "territory_count":  len(territories),
        "forecast_summary": {
            "avg_wmape":      avg_wmape,
            "avg_confidence": avg_conf,
            "top_model":      top_model,
            "model_wins":     model_wins,
        },
        "ic_plan_type":     ic_plan_type,
        "ic_goal_setting":  ic_goal_setting,
        "has_advisory":     has_advisory_answers(ic_plan_type, ic_goal_setting),
    }
    _sess_set(sid, _sess_data)
    _sessions[sid] = True

    # ── Build response ─────────────────────────────────────────────
    best = next(r for r in results if r["scenario"]["id"] == best_id)
    top5 = [
        {
            "rank":           r["rank"],
            "scenario_id":    r["scenario"]["id"],
            "composite":      r["fairness"]["boosted_score"],
            "method":         r["scenario"]["method"],
            "mode":           r["scenario"]["mode"],
            "on_target_pct":  round(r["fairness"]["pct_on"] * 100, 1),
            "confidence":     r["fairness"].get("confidence", "—"),
            "is_pref":        r.get("is_pref", False),
        }
        for r in results[:5]
    ]

    return {
        "session_id":       sid,
        "preset_used":      preset,
        "preset_label":     PRESETS[preset]["label"],
        "total_scenarios":  len(all_defs),
        "valid_scenarios":  len(valid_defs),
        "final_scenarios":  len(results),
        "territory_count":  len(territories),
        "pruning_summary":  prune_stats,
        "data_tier":        overall_tier,
        "tier_counts":      tier_counts,
        "has_advisory":     has_advisory_answers(ic_plan_type, ic_goal_setting),
        "forecast_summary": {
            "avg_wmape":    avg_wmape,
            "avg_confidence": avg_conf,
            "top_model":    top_model,
            "model_wins":   model_wins,
        },
        # ── Unconstrained Plan (always present) ───────────────────
        "unconstrained_plan": {
            "label":        "Unconstrained Plan",
            "description":  "Best scenario from all valid combinations. No advisory filters applied.",
            "scenario_id":  best["scenario"]["id"],
            "method":       best["scenario"]["method"],
            "mode":         best["scenario"]["mode"],
            "composite":    best["fairness"]["boosted_score"],
            "confidence":   best["fairness"].get("confidence", "—"),
            "on_target":    best["fairness"]["on_target"],
            "total_goals":  best["fairness"]["total_goals"],
            "breakdown": {
                "P": best["fairness"]["P"],   "A": best["fairness"]["A"],
                "C": best["fairness"]["C"],   "Ga": best["fairness"]["Ga"],
                "V": best["fairness"]["V"],   "K": best["fairness"]["K"],
                "S": best["fairness"]["S"],
            },
        },
        # ── Advisory-Guided Plan (only when advisory was run) ──────
        "advisory_plan": advisory_result,
        # Legacy keys kept for backward compat
        "best": {
            "scenario_id":  best["scenario"]["id"],
            "method":       best["scenario"]["method"],
            "mode":         best["scenario"]["mode"],
            "composite":    best["fairness"]["boosted_score"],
            "confidence":   best["fairness"].get("confidence", "—"),
            "on_target":    best["fairness"]["on_target"],
            "total_goals":  best["fairness"]["total_goals"],
            "breakdown": {
                "P": best["fairness"]["P"],   "A": best["fairness"]["A"],
                "C": best["fairness"]["C"],   "Ga": best["fairness"]["Ga"],
                "V": best["fairness"]["V"],   "K": best["fairness"]["K"],
                "S": best["fairness"]["S"],
            },
        },
        "preferred": {
            "scenario_id": pref["scenario"]["id"],
            "method":      pref["scenario"]["method"],
            "mode":        pref["scenario"]["mode"],
            "composite":   pref["fairness"]["boosted_score"],
        } if pref else None,
        "top5": top5,
    }


@app.get("/api/goals/session/{sid}/all")
async def get_all(sid: str, page: int = 1, size: int = 20):
    sess = _sess_get(sid)
    if not sess:
        raise HTTPException(404, "Session not found")
    rs = sess["results"]
    st = (page - 1) * size
    return {
        "total": len(rs),
        "page":  page,
        "size":  size,
        "scenarios": [
            {
                "rank":          r["rank"],
                "scenario_id":   r["scenario"]["id"],
                "method":        r["scenario"]["method"],
                "mode":          r["scenario"]["mode"],
                "composite":     r["fairness"]["boosted_score"],
                "confidence":    r["fairness"].get("confidence", "—"),
                "on_target_pct": round(r["fairness"]["pct_on"] * 100, 1),
                "P": r["fairness"]["P"], "A": r["fairness"]["A"],
                "C": r["fairness"]["C"], "Ga": r["fairness"]["Ga"],
                "V": r["fairness"]["V"], "K": r["fairness"]["K"],
                "S": r["fairness"]["S"],
                "is_best": r["is_best"],
                "is_pref": r.get("is_pref", False),
                # Include goals array for the best scenario (for comparison sliders)
                "goals": [
                    {"tid": g["tid"], "tname": g["tname"], "fg": g["fg"], "fc": g.get("fc", g["fg"])}
                    for g in r.get("goals", [])
                ] if r["is_best"] else [],
            }
            for r in rs[st: st + size]
        ],
    }


@app.get("/api/compare/session/{sid}")
async def compare_session(sid: str):
    """Delta insights: best vs preferred scenario."""
    sess = _sess_get(sid)
    if not sess:
        raise HTTPException(404, "Session not found")
    rs  = sess["results"]
    nf  = sess["nf"]
    best = next((r for r in rs if r["is_best"]), None)
    pref = next((r for r in rs if r.get("is_pref")), None)
    if not best:
        raise HTTPException(404, "No best scenario in session")
    if not pref or pref["scenario"]["id"] == best["scenario"]["id"]:
        return {"message": "Best and preferred are the same scenario — no delta to compute."}
    return delta_insights(best, pref, nf)


@app.get("/api/guardrails/{sid}")
async def run_guardrails(sid: str):
    """Guardrail check on the best scenario."""
    sess = _sess_get(sid)
    if not sess:
        raise HTTPException(404, "Session not found")
    return guardrails_check(sess["results"], sess["nf"], sess["best_id"])



@app.get("/api/session/{sid}/summary")
async def session_summary(sid: str):
    """Return best plan metrics + territory goals for comparison view."""
    sess = _sess_get(sid)
    if not sess:
        raise HTTPException(404, "Session not found or expired. Please re-run Goal Setting.")
    results  = sess["results"]
    best_id  = sess["best_id"]
    nf       = sess["nf"]
    best = next((r for r in results if r["scenario"]["id"] == best_id), None)
    if not best:
        raise HTTPException(404, "Best scenario not found in session.")
    fb = best["fairness"]
    goals = best.get("goals", [])
    return {
        "session_id":   sid,
        "scenario_id":  best_id,
        "method":       best["scenario"]["method"],
        "mode":         best["scenario"]["mode"],
        "composite":    round(float(fb["boosted_score"]), 4),
        "confidence":   round(float(fb.get("confidence", 0.5)), 4),
        "on_target":    int(fb["on_target"]),
        "total_goals":  round(float(fb["total_goals"]), 2),
        "breakdown":    {k: round(float(fb[k]), 4) for k in ["P","A","C","Ga","V","K","S"]},
        "goals": [
            {"tid": str(g["tid"]), "tname": str(g.get("tname", g["tid"])),
             "fg": round(float(g["fg"]), 2), "rg": round(float(g.get("rg", g["fg"])), 2),
             "fc": round(float(g.get("fc", g["fg"])), 2)}
            for g in goals
        ],
        "territory_count": len(goals),
        "national_forecast": round(nf, 2),
    }

@app.get("/api/export/session/{sid}/excel")
async def export_excel(sid: str):
    sess = _sess_get(sid)
    if not sess:
        raise HTTPException(404, "Session not found")
    xb = to_excel(
        sess["results"], sess["best_id"],
        sess.get("preferred_id"), sess["nf"],
        sess.get("territories", [])
    )
    return Response(
        content=xb,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=procdna_{sid[:8]}.xlsx"},
    )


@app.get("/api/export/session/{sid}/csv")
async def export_csv(sid: str):
    sess = _sess_get(sid)
    if not sess:
        raise HTTPException(404, "Session not found")
    return Response(
        content=to_csv(sess["results"]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=procdna_{sid[:8]}.csv"},
    )

@app.post("/forecast/upload")
async def forecast_upload(
    file: UploadFile = File(...),
    date_col: str = "wk_end_dt",
    id_col: str = "terr_id",
    value_col: str = "sales",
    forecast_horizon: int = 3,
):
    """Run 15-model forecast competition. Accepts same file as goal setting."""
    import io as _io
    fb = await file.read()
    if not fb:
        raise HTTPException(400, "Empty file")
    try:
        df = (pd.read_csv(_io.BytesIO(fb))
              if file.filename.lower().endswith(".csv")
              else pd.read_excel(_io.BytesIO(fb)))
    except Exception as e:
        raise HTTPException(422, f"Cannot read file: {e}")

    # Auto-remap Cabo column aliases → standard names
    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
    remap_fc = {c: CABO_COL_MAP[c] for c in df.columns if c in CABO_COL_MAP}
    if remap_fc:
        df = df.rename(columns=remap_fc)
        if date_col  in remap_fc: date_col  = remap_fc[date_col]
        if id_col    in remap_fc: id_col    = remap_fc[id_col]
        if value_col in remap_fc: value_col = remap_fc[value_col]

    for col in [date_col, id_col, value_col]:
        if col not in df.columns:
            raise HTTPException(422, f"Column '{col}' not found. File has: {list(df.columns)}")

    from forecaster import run_forecasts
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
    df = df.dropna(subset=[date_col])

    # Aggregate weekly → monthly for the forecaster
    rows_per_terr_fc = len(df) / max(df[id_col].nunique(), 1)
    if rows_per_terr_fc > 20:
        df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp()
        df = df.groupby([id_col, date_col])[value_col].sum().reset_index()

    stub_territories = [
        {"territory_id": str(tid), "region_id": "—",
         "territory_name": str(tid), "region_name": "—",
         "q1":0,"q2":0,"q3":0,"q4":0,"h6":0,"h9":0,"h12":0,"pot":0,
         "avg_q":0,"s6":0,"s9":0,"s12":0,"trend":0,"vol":0,
         "vband":"Stable Territory","last_q":0}
        for tid in df[id_col].unique()
    ]

    df_renamed = df.rename(columns={date_col:"transaction_date", id_col:"territory_id", value_col:"metric_value"})
    enriched = run_forecasts(df_renamed, stub_territories)

    summary = []
    for t in enriched:
        row = {
            "territory_id":       t["territory_id"],
            "best_model":         t.get("best_model","—"),
            "wmape_holdout":      t.get("wmape"),
            "holdout_months_used":t.get("data_months",0),
            "data_tier":          t.get("data_tier","—"),
            "confidence":         t.get("confidence"),
            "fc_m1":              round(t.get("forecast_q1",0)/forecast_horizon, 2),
            "fc_m2":              round(t.get("forecast_q1",0)/forecast_horizon, 2),
            "fc_m3":              round(t.get("forecast_q1",0)/forecast_horizon, 2),
            "fc_lo_m1":           round(t.get("fc_lo_q1",0)/forecast_horizon, 2),
            "fc_hi_m1":           round(t.get("fc_hi_q1",0)/forecast_horizon, 2),
        }
        summary.append(row)

    return {
        "territories_processed":  len(summary),
        "forecast_horizon_months": forecast_horizon,
        "holdout_months_used":     "adaptive",
        "summary":                 summary,
    }

# ─────────────────────────────────────────────────────────────────
# Chat endpoint — Claude + tool use loop
# ─────────────────────────────────────────────────────────────────
from pydantic import BaseModel as _BaseModel

class ChatRequest(_BaseModel):
    message:          str
    chat_session_id:  str
    goal_session_id:  str = ""

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Single endpoint for the conversational interface.
    Claude decides which tools to call based on the user message.
    """
    from chat import chat as _chat
    result = _chat(
        user_message    = req.message,
        chat_session_id = req.chat_session_id,
        goal_session_id = req.goal_session_id or None,
        sessions        = _sessions,
    )
    return result

@app.get("/api/chat/history/{chat_session_id}")
async def chat_history(chat_session_id: str):
    from chat import get_history
    return {"history": get_history(chat_session_id)}

@app.delete("/api/chat/history/{chat_session_id}")
async def chat_clear(chat_session_id: str):
    from chat import clear_history
    clear_history(chat_session_id)
    return {"status": "cleared"}

@app.get("/chat")
async def chat_page():
    p = _HERE / "chat.html"
    if not p.exists():
        return HTMLResponse("<h2>chat.html not found</h2>", 404)
    return FileResponse(str(p))

# ── IC Advisory natural language generation ──────────────────────
class AdvisoryRequest(_BaseModel):
    prompt:       str
    plan_type:    str = ""
    goal_setting: str = ""


def _advisory_fallback(plan_type: str, goal_setting: str) -> dict:
    """Rule-based IC Advisory recommendation — no API key needed."""
    text = (
        f"Based on your inputs, we recommend a {plan_type} quota structure. "
        f"This approach directly aligns individual accountability with territory potential, "
        f"ensuring that each rep owns their number and has a clear line of sight from daily activity to reward. "
        f"Using {goal_setting.lower()} as the goal-setting foundation gives the plan a credible, data-driven basis "
        f"that reps will trust and managers can defend.\n\n"
        f"For reps, this means a single clear quota target — no ambiguity about what they need to achieve. "
        f"Pay is directly proportional to individual performance, which maximises motivation for your highest performers.\n\n"
        f"Goals will be set by running historical sales data through the 12-model forecast competition, "
        f"selecting the best-fit model per territory, and calibrating final numbers against the national forecast budget. "
        f"Territories with high volatility will receive smoothed baselines to avoid luck-driven over- or under-attainment.\n\n"
        f"Key governance watch-out: ensure quota sign-off happens at least 30 days before the plan period starts. "
        f"Schedule a 90-day in-period review so you can catch systemic misses early and make mid-cycle corrections before they compound."
    )
    return {"text": text, "plan_type": plan_type, "goal_setting": goal_setting}

@app.post("/api/advisory/generate")
async def advisory_generate(req: AdvisoryRequest):
    """
    Calls Groq to generate a natural language IC Advisory recommendation.
    Uses a fixed 4-part structure, consultant tone, no bullet points or IDs.
    Falls back to rule-based text if no API key.
    """
    import httpx as _httpx
    import os as _os
    api_key = _os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        plan = req.plan_type or "Individual"
        gs   = req.goal_setting or "Historical"
        return {"text":
            "Based on your inputs, we recommend a " + plan + " quota structure with a balanced pay mix and " +
            gs.lower() + " goal setting. This approach gives your sales team clear individual accountability while "
            "ensuring goals are grounded in historical performance. Reps will benefit from transparent, data-driven "
            "targets that reflect their territory's actual potential rather than arbitrary top-down numbers. "
            "For governance, ensure quota documentation is signed off before launch and schedule a 90-day "
            "attainment review to course-correct early if needed.",
            "plan_type": req.plan_type,
            "goal_setting": req.goal_setting
        }
    try:
        resp = _httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "max_tokens": 500,
                "temperature": 0.4,
                "messages": [
                    {"role": "system", "content": "You are a senior IC consulting advisor. Write in flowing paragraphs, no bullet points, no IDs, no headers. Warm consultant tone speaking directly to the client."},
                    {"role": "user",   "content": req.prompt}
                ],
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"text": text, "plan_type": req.plan_type, "goal_setting": req.goal_setting}
    except Exception as e:
        # Fallback to rule-based on any API error
        return _advisory_fallback(req.plan_type or "Individual", req.goal_setting or "Historical")

# ══════════════════════════════════════════════════════════════════
# STEP-BY-STEP GOAL SETTING API
# Frontend calls these in sequence, showing progress after each step
# ══════════════════════════════════════════════════════════════════

# ── Disk-based parse cache ────────────────────────────────────────────────────
# Survives process restarts on Render free tier (spins down after 15min idle).
# Each parse_id gets its own file in /tmp/procdna_cache/.

_CACHE_DIR = pathlib.Path("/tmp/procdna_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(parse_id: str) -> pathlib.Path:
    # Sanitise — only allow uuid-like chars to prevent path traversal
    safe = "".join(c for c in parse_id if c.isalnum() or c == "-")
    return _CACHE_DIR / f"{safe}.pkl"

def _cache_set(parse_id: str, data: dict) -> None:
    try:
        with open(_cache_path(parse_id), "wb") as f:
            pickle.dump(data, f, protocol=4)
    except Exception as e:
        print(f"[cache_set] WARNING: {e}")

def _cache_get(parse_id: str) -> dict | None:
    p = _cache_path(parse_id)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[cache_get] WARNING: {e}")
        return None

def _cache_update(parse_id: str, key: str, value) -> None:
    """Add or update one key inside an existing cache entry."""
    data = _cache_get(parse_id) or {}
    data[key] = value
    _cache_set(parse_id, data)

# Keep a small in-memory index for O(1) existence checks (rebuilt on restart from disk)
_parse_cache: dict = {}  # parse_id → True  (lightweight sentinel only)

@app.post("/api/goals/step1-parse")
async def step1_parse(
    file:            UploadFile = File(...),
    lookback_months: int        = Form(12),
    date_col:        str        = Form("wk_end_dt"),
    id_col:          str        = Form("terr_id"),
    value_col:       str        = Form("sales"),
    name_col:        str        = Form("terr_nm"),
    region_id_col:   str        = Form("regn_id"),
    region_nm_col:   str        = Form("regn_nm"),
):
    """
    Step 1 — Parse the uploaded file and return territory summary.
    Fast (<1 second). Returns parse_id to use in step 2.
    """
    import uuid as _uuid
    fb = await file.read()
    if not fb:
        raise HTTPException(400, "Empty file")

    territories, raw_df = parse_file(fb, file.filename, lookback_months)
    # Remove unassigned / zero-sales territories
    territories = [
        t for t in territories
        if t.get("h12", 0) > 1
        and "unassign" not in str(t.get("territory_name", "")).lower()
        and "unassign" not in str(t.get("territory_id", "")).lower()
    ]
    if len(territories) < 2:
        raise HTTPException(422, f"Need at least 2 territories. Found: {len(territories)}")

    parse_id = str(_uuid.uuid4())
    _cache_set(parse_id, {"territories": territories, "raw_df": raw_df})
    _parse_cache[parse_id] = True  # in-memory sentinel

    total_h12 = sum(t.get("h12", 0) for t in territories)

    # ── Detect last full quarter and next planning quarter ─────────────
    import re as _re
    from collections import defaultdict as _dd
    _qm: dict = _dd(set)
    for _, row in raw_df.iterrows():
        _q = f"{row['transaction_date'].year}Q{(row['transaction_date'].month - 1) // 3 + 1}"
        _qm[_q].add(row['transaction_date'].month)

    _q_expected = {
        "Q1": {1,2,3}, "Q2": {4,5,6}, "Q3": {7,8,9}, "Q4": {10,11,12}
    }
    def _is_complete(q_str, months):
        suffix = q_str[-2:]  # "Q1" ... "Q4"
        return months >= _q_expected.get(suffix, set())

    _sorted_qs = sorted(_qm.keys())
    _last_full_q = None
    _all_quarters = []
    for _q in _sorted_qs:
        _year, _qn = int(_q[:4]), _q[5:]
        _complete = _is_complete(_q, _qm[_q])
        _all_quarters.append({"quarter": _q, "complete": _complete,
                               "months": sorted(_qm[_q])})
        if _complete:
            _last_full_q = _q

    # Compute next quarter
    _next_q = None
    if _last_full_q:
        _ly, _lqn = int(_last_full_q[:4]), int(_last_full_q[-1])
        if _lqn == 4:
            _next_q = f"{_ly+1}Q1"
        else:
            _next_q = f"{_ly}Q{_lqn+1}"

    # Quarter label helper
    _qlabels = {"Q1":"Jan-Mar","Q2":"Apr-Jun","Q3":"Jul-Sep","Q4":"Oct-Dec"}
    def _qlabel(q):
        yr, qn = q[:4], q[4:]
        return f"{q}  ({_qlabels.get(qn, qn)} {yr})"

    # Compute suggested quarterly NF from last full quarter + trend
    _last_q_total = sum(t.get("q1", 0) for t in territories)  # q1 = most recent quarter
    _qoq_growths  = []
    for t in territories:
        if t.get("q2", 0) > 0:
            _qoq_growths.append((t.get("q1", 0) - t.get("q2", 0)) / t.get("q2", 0))
    _avg_growth   = float(sum(_qoq_growths) / len(_qoq_growths)) if _qoq_growths else 0.0
    _suggested_nf = round(_last_q_total * (1 + _avg_growth), 0)

    return {
        "parse_id":        parse_id,
        "territory_count": len(territories),
        "total_h12":       round(total_h12, 0),
        "last_q_total":    round(_last_q_total, 0),
        "suggested_nf":    _suggested_nf,
        "avg_qoq_growth":  round(_avg_growth * 100, 2),
        "territories": [{
            "id":       t["territory_id"],
            "name":     t["territory_name"],
            "region":   t["region_name"],
            "q1":       round(t["q1"], 0),
            "q2":       round(t["q2"], 0),
            "q3":       round(t["q3"], 0),
            "q4":       round(t["q4"], 0),
            "volatility": t["vband"],
        } for t in territories],
        "last_full_quarter": _last_full_q,
        "next_quarter":      _next_q,
        "last_full_label":   _qlabel(_last_full_q) if _last_full_q else None,
        "next_quarter_label":_qlabel(_next_q) if _next_q else None,
        "message": f"Parsed {len(territories)} territories. Last full quarter: {_last_full_q}. Suggested goal quarter: {_next_q}."
    }


@app.post("/api/goals/step2-forecast")
async def step2_forecast(
    parse_id:     str   = Form(...),
    model_name:   str   = Form(...),
    precomputed:  str   = Form(""),
):
    """
    Step 2 — Run ONE forecast model across all territories.
    Call this once per model. Frontend calls it 12 times sequentially.
    Returns per-territory forecast results for this model.
    """
    cache = _cache_get(parse_id)
    if not cache:
        raise HTTPException(404, "parse_id not found. Run step1-parse first.")

    # Handle pre-computed forecast results from standalone forecasting
    if precomputed:
        import json as _json
        try:
            pre_results = _json.loads(precomputed)
            # Normalise shape to match step2 output
            results = [{"territory_id": r.get("territory_id","?"),
                        "model": "Precomputed",
                        "forecast_q1": float(r.get("fc_m1", 0)) * 3,  # monthly → quarterly
                        "wmape": float(r.get("wmape_holdout") or 0.15),
                        "status": "ok"} for r in pre_results]
            if "model_results" not in cache:
                cache["model_results"] = {}
            cache["model_results"]["Precomputed"] = results
            _cache_set(parse_id, cache)
            import numpy as _np2
            avg_w = round(float(_np2.mean([r["wmape"] for r in results])), 4)
            return {"parse_id": parse_id, "model_name": "Precomputed",
                    "territory_count": len(results), "ok_count": len(results),
                    "avg_wmape": avg_w, "results": results,
                    "message": f"Precomputed: {len(results)} territories loaded"}
        except Exception as _e:
            pass  # fall through to normal model run

    territories = cache["territories"]
    raw_df      = cache["raw_df"]

    from forecaster import FAST_MODELS, forecast_territory, winsorise
    import pandas as _pd

    from forecaster import ALL_MODELS as _ALL
    if model_name not in _ALL:
        raise HTTPException(400, f"Unknown model: {model_name}. Available: {list(_ALL.keys())}")

    from forecaster import ALL_MODELS
    model_fn = ALL_MODELS.get(model_name)
    if not model_fn:
        raise HTTPException(400, f"Model {model_name} not available")

    # Build monthly series per territory
    df = raw_df.copy()
    df["_month"] = _pd.to_datetime(df["transaction_date"]).dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby(["territory_id","_month"])["metric_value"].sum().reset_index()

    results = []
    from forecaster import wmape as _wmape, build_ci, adaptive_holdout, winsorise, FORECAST_HORIZON, MIN_TRAIN_POINTS
    import numpy as _np

    for t in territories:
        tid = t["territory_id"]
        grp = monthly[monthly["territory_id"]==tid].set_index("_month")["metric_value"]
        grp.index = _pd.DatetimeIndex(grp.index)
        grp = grp.sort_index()

        n = len(grp)
        avg_vol = float(grp.mean()) if len(grp) else 0
        holdout = adaptive_holdout(avg_vol)

        if n < MIN_TRAIN_POINTS + holdout:
            results.append({"territory_id": tid, "model": model_name,
                            "forecast_q1": round(avg_vol*3,2), "wmape": 0.20,
                            "status": "insufficient_data"})
            continue

        try:
            train   = winsorise(grp.iloc[:-holdout])
            actual  = grp.iloc[-holdout:].values
            preds   = model_fn(train, holdout)
            if _np.any(_np.isnan(preds)) or _np.any(_np.isinf(preds)):
                raise ValueError("NaN/Inf in predictions")
            err     = _wmape(actual, preds)
            full_fc = model_fn(winsorise(grp), FORECAST_HORIZON)
            full_fc = _np.maximum(full_fc, 0)
            lo, hi  = build_ci(full_fc, err if not _np.isnan(err) else 0.2)
            results.append({
                "territory_id": tid,
                "model":        model_name,
                "forecast_q1":  round(float(full_fc.sum()), 2),
                "wmape":        round(float(err), 4) if not _np.isnan(err) else 0.20,
                "fc_lo":        round(float(lo.sum()), 2),
                "fc_hi":        round(float(hi.sum()), 2),
                "status":       "ok",
            })
        except BaseException as ex:
            results.append({"territory_id": tid, "model": model_name,
                            "forecast_q1": round(avg_vol*3,2), "wmape": 0.25,
                            "status": "error", "note": str(ex)})

    # Store in cache — read-modify-write to disk so it survives restarts
    existing = _cache_get(parse_id) or cache
    if "model_results" not in existing:
        existing["model_results"] = {}
    existing["model_results"][model_name] = results
    _cache_set(parse_id, existing)

    ok_count  = sum(1 for r in results if r["status"]=="ok")
    avg_wmape = round(float(_np.mean([r["wmape"] for r in results])), 4)

    return {
        "parse_id":   parse_id,
        "model_name": model_name,
        "territory_count": len(results),
        "ok_count":   ok_count,
        "avg_wmape":  avg_wmape,
        "results":    results,
        "message":    f"{model_name}: ran on {ok_count}/{len(results)} territories, avg wMAPE={avg_wmape:.3f}"
    }


@app.post("/api/goals/step3-score")
async def step3_score(
    parse_id:          str   = Form(...),
    national_forecast: float = Form(...),
    scoring_method:    str   = Form("Run all methods"),
    plan_mode:         str   = Form("Run all splits"),
    preset:            str   = Form("fairness"),
    ic_plan_type:      str   = Form(""),
    ic_goal_setting:   str   = Form(""),
    plan_quarter:      str   = Form(""),
    hist_pct:          float = Form(1.0),
):
    """
    Step 3 — Select best model per territory, calculate goals,
    score scenarios, prune, return final result.
    Uses model results stored by step2-forecast calls.
    """
    print(f"[step3-score] national_forecast={national_forecast}, plan_quarter={plan_quarter}")
    import uuid as _uuid
    import numpy as _np

    cache = _cache_get(parse_id)
    if not cache:
        raise HTTPException(404, "parse_id not found. Run step1 and step2 first.")

    territories    = cache["territories"]
    model_results  = cache.get("model_results", {})

    if not model_results:
        raise HTTPException(422, "No forecast results found. Run step2-forecast for at least one model first.")

    # Select best model per territory (lowest wMAPE)
    best_per_territory = {}
    for model_name, results in model_results.items():
        for r in results:
            tid = r["territory_id"]
            if tid not in best_per_territory or r["wmape"] < best_per_territory[tid]["wmape"]:
                best_per_territory[tid] = r

    # Attach winning forecast to each territory
    for t in territories:
        tid = t["territory_id"]
        if tid in best_per_territory:
            best = best_per_territory[tid]
            t["forecast_q1"]  = best["forecast_q1"]
            t["wmape"]        = best["wmape"]
            t["confidence"]   = min(max(0.0, 1.0 - best["wmape"]), 0.95)
            t["best_model"]   = best["model"]
            t["data_tier"]    = "full"
            t["data_months"]  = 12
        else:
            t["forecast_q1"] = t.get("last_q", 0)
            t["wmape"]       = 0.20
            t["confidence"]  = 0.50
            t["best_model"]  = "Mean (fallback)"
            t["data_tier"]   = "limited"
            t["data_months"] = 0

    # Model competition summary
    model_wins = {}
    for t in territories:
        bm = t.get("best_model", "—")
        model_wins[bm] = model_wins.get(bm, 0) + 1
    top_model  = max(model_wins, key=model_wins.get) if model_wins else "—"
    avg_wmape  = round(float(_np.mean([t.get("wmape",0.2) for t in territories])), 4)
    avg_conf   = round(float(_np.mean([t.get("confidence",0.5) for t in territories])), 4)

    # Confidence + feature maps
    confidences = {t["territory_id"]: t.get("confidence", 0.5) for t in territories}
    feat_map    = {t["territory_id"]: t for t in territories}

    # Data tier summary
    tiers = [t.get("data_tier","full") for t in territories]
    tier_counts  = {"limited":tiers.count("limited"),"moderate":tiers.count("moderate"),"full":tiers.count("full")}
    overall_tier = "limited" if tier_counts["limited"]>len(tiers)*0.5 else "moderate" if tier_counts["full"]<len(tiers)*0.5 else "full"

    # Generate + score scenarios
    if preset not in PRESETS: preset = "fairness"
    # Always generate full pool — user's method/mode choice is used as preset preference
    # Running all combinations ensures maximum scenario diversity before pruning
    all_defs  = gen_scenarios("Run all methods", "Run all splits", False, 0.0, 0.1, 0.7)
    # Store user preference for advisory boost
    preferred_method = scoring_method
    preferred_mode   = plan_mode
    valid_defs = [s for s in all_defs if s["ok"]]
    if not valid_defs:
        raise HTTPException(422, "No valid scenarios generated.")

    raw_results = []
    for sc in valid_defs:
        goals    = calc_goals(sc, territories, national_forecast)
        fairness = build_fairness_block(
            sid=sc["id"], goals=goals, feat=feat_map,
            national_forecast=national_forecast, confidences=confidences,
            preset=preset, ic_plan_type=ic_plan_type,
            scenario_mode=sc["mode"], ic_boost_lambda=0.05,
        )
        raw_results.append({"scenario":sc,"goals":goals,"fairness":fairness,
                            "is_best":False,"is_pref":sc["pref"],"rank":0})

    results, prune_stats, best_id = prune(raw_results, national_forecast)
    pref = next((r for r in results if r.get("is_pref")), None)

    # Advisory track
    advisory_result = None
    if has_advisory_answers(ic_plan_type, ic_goal_setting):
        adv_defs = filter_by_advisory(valid_defs, ic_plan_type=ic_plan_type, ic_goal_setting=ic_goal_setting)
        if adv_defs:
            adv_raw = []
            for sc in adv_defs:
                goals    = calc_goals(sc, territories, national_forecast)
                fairness = build_fairness_block(
                    sid=sc["id"], goals=goals, feat=feat_map,
                    national_forecast=national_forecast, confidences=confidences,
                    preset=preset, ic_plan_type=ic_plan_type,
                    scenario_mode=sc["mode"], ic_boost_lambda=0.05,
                )
                adv_raw.append({"scenario":sc,"goals":goals,"fairness":fairness,
                                "is_best":False,"is_pref":sc["pref"],"rank":0})
            adv_results, _, adv_best_id = prune(adv_raw, national_forecast)
            adv_best = next(r for r in adv_results if r["scenario"]["id"]==adv_best_id)
            advisory_result = {
                "best_scenario_id": adv_best_id,
                "method": adv_best["scenario"]["method"],
                "mode":   adv_best["scenario"]["mode"],
                "composite": adv_best["fairness"]["boosted_score"],
                "confidence": adv_best["fairness"].get("confidence","—"),
                "on_target": adv_best["fairness"]["on_target"],
                "total_goals": adv_best["fairness"]["total_goals"],
                "scenarios_in_pool": len(adv_defs),
                "breakdown": {k: adv_best["fairness"][k] for k in ["P","A","C","Ga","V","K","S"]},
                "filters_applied": {"plan_type":ic_plan_type,"goal_setting":ic_goal_setting},
            }
        else:
            advisory_result = {"message":"No scenarios matched advisory preferences."}

    # Store session
    sid = str(_uuid.uuid4())
    _sess_data = {
        "results": results, "best_id": best_id,
        "preferred_id": pref["scenario"]["id"] if pref else None,
        "nf": national_forecast, "preset": preset,
        "tier_counts": tier_counts, "overall_tier": overall_tier,
        "prune_stats": prune_stats, "territories": territories,
        "territory_count": len(territories),
        "forecast_summary": {"avg_wmape":avg_wmape,"avg_confidence":avg_conf,
                             "top_model":top_model,"model_wins":model_wins},
        "ic_plan_type": ic_plan_type, "ic_goal_setting": ic_goal_setting,
        "has_advisory": has_advisory_answers(ic_plan_type, ic_goal_setting),
    }

    best = next(r for r in results if r["scenario"]["id"]==best_id)

    # NF is entered as a quarterly target — goals are already quarterly from calc_goals.
    # No ×0.25 scaling needed.

    # Historical vs Equal Allocation post-processing
    # final_goal = hist_pct × scenario_goal + (1 - hist_pct) × NF / N
    _hist = float(hist_pct) if 0.0 <= float(hist_pct) <= 1.0 else 1.0
    if _hist < 1.0:
        _n_terr = len(results[0].get("goals", [])) if results else 1
        _equal_share = national_forecast / max(_n_terr, 1)
        for _r in results:
            _new_total = 0.0
            for _g in _r.get("goals", []):
                _orig = float(_g.get("fg", 0))
                _g["fg"] = round(_hist * _orig + (1 - _hist) * _equal_share, 2)
                # Recompute attainment against the new blended fg
                _fc = float(_g.get("fc", 0))
                _g["att"] = round(_fc / _g["fg"], 4) if _g["fg"] > 0 else 0.0
                _new_total += _g["fg"]
            _r["fairness"]["total_goals"] = round(_new_total, 2)

    top5 = [{"rank":r["rank"],"scenario_id":r["scenario"]["id"],
              "composite":r["fairness"]["boosted_score"],"method":r["scenario"]["method"],
              "mode":r["scenario"]["mode"],"on_target_pct":round(r["fairness"]["pct_on"]*100,1),
              "confidence":r["fairness"].get("confidence","—"),"is_pref":r.get("is_pref",False)}
             for r in results[:5]]

    # Persist session to disk — after EA blending so goals are final
    _sess_data["results"]  = results
    _sess_data["best_id"]  = best_id
    _sess_set(sid, _sess_data)
    _sessions[sid] = True

    return {
        "session_id": sid,
        "preset_used": preset,
        "total_scenarios": len(all_defs),
        "valid_scenarios": len(valid_defs),
        "final_scenarios": len(results),
        "territory_count": len(territories),
        "pruning_summary": prune_stats,
        "data_tier": overall_tier,
        "has_advisory": has_advisory_answers(ic_plan_type, ic_goal_setting),
        "forecast_summary": {"avg_wmape":avg_wmape,"avg_confidence":avg_conf,
                             "top_model":top_model,"model_wins":model_wins},
        "unconstrained_plan": {
            "scenario_id": best["scenario"]["id"],
            "method": best["scenario"]["method"],
            "mode":   best["scenario"]["mode"],
            "w1":     best["scenario"].get("w1", 0),
            "w2":     best["scenario"].get("w2", 0),
            "w3":     best["scenario"].get("w3", 0),
            "rs":     best["scenario"].get("rs", 0),
            "is_":    best["scenario"].get("is_", 0),
            "composite": best["fairness"]["boosted_score"],
            "confidence": best["fairness"].get("confidence","—"),
            "on_target": best["fairness"]["on_target"],
            "total_goals": best["fairness"]["total_goals"],
            "breakdown": {k: best["fairness"][k] for k in ["P","A","C","Ga","V","K","S"]},
            "goals": [{"tid": str(g["tid"]), "tname": str(g.get("tname", g["tid"])),
                       "fg": float(g["fg"]), "rg": float(g.get("rg", g["fg"])),
                       "ig": float(g.get("ig", g["fg"])),
                       "fc": float(g.get("fc", g["fg"])),
                       "att": float(g.get("att", 0)),
                       "best_model": str(g.get("best_model","—"))}
                      for g in best.get("goals", [])],
        },
        "advisory_plan": advisory_result,
        "best": {
            "scenario_id": best["scenario"]["id"],
            "method": best["scenario"]["method"],
            "mode": best["scenario"]["mode"],
            "composite": best["fairness"]["boosted_score"],
            "confidence": best["fairness"].get("confidence","—"),
            "on_target": best["fairness"]["on_target"],
            "total_goals": best["fairness"]["total_goals"],
            "breakdown": {k: best["fairness"][k] for k in ["P","A","C","Ga","V","K","S"]},
        },
        "top5": top5,
    }
