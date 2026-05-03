"""
ProcDNA Intelligence — Forecaster v2
======================================
15 models in one model-competition pipeline per territory.
Best model selected by lowest wMAPE on adaptive holdout.

Improvements from TerraForecast:
  1. Outlier handling  — winsorise training series at 97th percentile
  2. Adaptive holdout  — 6m high-volume / 3m mid / 2m low-volume
  3. Confidence bands  — fc_lo / fc_hi (80% interval) per territory
  4. Data tier         — classify <6m / 6-12m / 12m+ → adjust confidence

All 15 models:
  Naive, Mean, Moving Average, Weighted MA, Linear Trend, Drift,
  Exp Trend, SES, Holt, Holt-Winters, ARIMA, Theta, Median,
  XGBoost, Poisson Trend

Output per territory (appended to territory dict from parse_file):
  forecast_q1   — next-quarter forecast (sum of 3 monthly forecasts)
  wmape         — best model's holdout error
  confidence    — 1 - wmape  (clamped 0-1)
  fc_lo / fc_hi — 80% confidence interval on forecast_q1
  best_model    — name of winning model
  data_tier     — "limited" / "moderate" / "full"
  data_months   — number of months of history available
"""

import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

FORECAST_HORIZON      = 3          # months ahead (= 1 quarter)
MIN_TRAIN_POINTS      = 6
LOW_VOL_THRESHOLD     = 4          # avg monthly TRx
HIGH_VOL_THRESHOLD    = 10
WINSOR_PCT            = 0.97
CI_COVERAGE           = 0.80
CI_SCALE              = 1.5        # empirical scale factor for 80% coverage


# ── Data tier classification ──────────────────────────────────────────────────
def data_tier(n_months: int) -> str:
    if n_months < 6:
        return "limited"
    if n_months < 12:
        return "moderate"
    return "full"


def tier_confidence_cap(tier: str) -> float:
    """Max confidence allowed per tier — limited data = lower cap."""
    return {"limited": 0.55, "moderate": 0.80, "full": 1.00}.get(tier, 1.0)


# ── Adaptive holdout ──────────────────────────────────────────────────────────
def adaptive_holdout(avg_vol: float) -> int:
    if avg_vol >= HIGH_VOL_THRESHOLD:
        return 6
    if avg_vol >= LOW_VOL_THRESHOLD:
        return 3
    return 2


# ── Outlier cap ───────────────────────────────────────────────────────────────
def winsorise(series: pd.Series, pct: float = WINSOR_PCT) -> pd.Series:
    cap = series.quantile(pct)
    return series.clip(upper=cap)


# ── Confidence band ───────────────────────────────────────────────────────────
def build_ci(forecast: np.ndarray, wmape_val: float) -> tuple:
    if np.isnan(wmape_val) or wmape_val == 0:
        half = forecast * 0.20
    else:
        half = forecast * wmape_val * CI_SCALE
    lo = np.maximum(forecast - half, 0)
    hi = forecast + half
    return lo, hi


# ── Error metrics ─────────────────────────────────────────────────────────────
def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual - predicted)) / denom)


# ── 15 model implementations ──────────────────────────────────────────────────

def model_naive(s, h):
    return np.full(h, s.iloc[-1])

def model_mean(s, h):
    return np.full(h, s.mean())

def model_moving_average(s, h, w=3):
    w = min(w, len(s))
    return np.full(h, s.iloc[-w:].mean())

def model_weighted_ma(s, h):
    w   = min(4, len(s))
    sl  = s.iloc[-w:].values
    wts = np.arange(1, w + 1, dtype=float)
    return np.maximum(np.full(h, np.dot(sl, wts) / wts.sum()), 0)

def model_linear_trend(s, h):
    from sklearn.linear_model import LinearRegression
    X = np.arange(len(s)).reshape(-1, 1)
    lr = LinearRegression().fit(X, s.values)
    return np.maximum(lr.predict(np.arange(len(s), len(s) + h).reshape(-1, 1)), 0)

def model_drift(s, h):
    if len(s) < 2:
        raise ValueError("Need 2+ points")
    drift = (s.iloc[-1] - s.iloc[0]) / (len(s) - 1)
    return np.maximum(s.iloc[-1] + drift * np.arange(1, h + 1), 0)

def model_exp_trend(s, h):
    pos   = np.maximum(s.values, 0.001)
    log_y = np.log(pos)
    x     = np.arange(len(s), dtype=float)
    m, b  = np.polyfit(x, log_y, 1)
    return np.maximum(np.exp(b + m * np.arange(len(s), len(s) + h, dtype=float)), 0)

def model_ses(s, h):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    fit = SimpleExpSmoothing(s.values, initialization_method="estimated").fit(optimized=True)
    return np.maximum(fit.forecast(h), 0)

def model_holt(s, h):
    from statsmodels.tsa.holtwinters import Holt
    if len(s) < 4:
        raise ValueError("Need 4+ points")
    fit = Holt(s.values, initialization_method="estimated").fit(optimized=True)
    return np.maximum(fit.forecast(h), 0)

def model_holt_winters(s, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    if len(s) < 24:
        raise ValueError("Need 24+ months for Holt-Winters")
    fit = ExponentialSmoothing(
        s.values, trend="add", seasonal="add",
        seasonal_periods=12, initialization_method="estimated"
    ).fit(optimized=True)
    return np.maximum(fit.forecast(h), 0)

def model_arima(s, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    for order in [(1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 1, 0)]:
        try:
            fit = SARIMAX(s.values, order=order, trend="n").fit(disp=False)
            return np.maximum(fit.forecast(h), 0)
        except Exception:
            continue
    raise ValueError("All ARIMA orders failed")

def model_theta(s, h):
    return np.maximum((model_ses(s, h) + model_linear_trend(s, h)) / 2, 0)

def model_median(s, h):
    med    = np.median(s.values)
    recent = s.iloc[-3:].values
    m      = np.polyfit(np.arange(len(recent), dtype=float), recent, 1)[0] if len(recent) > 1 else 0
    return np.maximum(med + m * np.arange(1, h + 1), 0)

def model_xgboost(s, h):
    import xgboost as xgb
    n_lags = min(6, len(s) - 1)
    if len(s) < n_lags + 2:
        raise ValueError("Not enough data for XGBoost")
    vals = list(s.values.astype(float))
    def make_X(v):
        return np.array([v[i - n_lags:i][::-1] for i in range(n_lags, len(v))])
    mdl = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
    mdl.fit(make_X(vals), np.array(vals[n_lags:]))
    preds, cur = [], vals[:]
    for _ in range(h):
        p = max(float(mdl.predict(np.array(cur[-n_lags:][::-1]).reshape(1, -1))[0]), 0)
        preds.append(p)
        cur.append(p)
    return np.array(preds)

def model_poisson_trend(s, h):
    import statsmodels.api as sm
    t   = np.arange(len(s), dtype=float)
    X   = sm.add_constant(t)
    y   = np.round(s.values).astype(int).clip(0)
    fit = sm.GLM(y, X, family=sm.families.Poisson()).fit(disp=False)
    return fit.predict(sm.add_constant(np.arange(len(s), len(s) + h, dtype=float)))


# ── Model registries ──────────────────────────────────────────────────────────
ALL_MODELS = {
    "Naive":          model_naive,
    "Mean":           model_mean,
    "Moving Average": model_moving_average,
    "Weighted MA":    model_weighted_ma,
    "Linear Trend":   model_linear_trend,
    "Drift":          model_drift,
    "Exp Trend":      model_exp_trend,
    "SES":            model_ses,
    "Holt":           model_holt,
    "Holt-Winters":   model_holt_winters,
    "ARIMA":          model_arima,
    "Theta":          model_theta,
    "Median":         model_median,
    "XGBoost":        model_xgboost,
    "Poisson Trend":  model_poisson_trend,
}

# Fast model set — skips XGBoost, ARIMA, Holt-Winters (slow on free tier)
# Runs in ~0.3s per territory vs ~5-10s for full set
FAST_MODELS = {
    k: v for k, v in ALL_MODELS.items()
    if k not in {"XGBoost", "ARIMA", "Holt-Winters"}
}

LOW_VOL_MODELS = {
    k: v for k, v in ALL_MODELS.items()
    if k in {"Naive","Mean","Moving Average","Weighted MA",
             "Linear Trend","SES","Holt","Theta","Median","Poisson Trend"}
}


# ── Per-territory evaluation ──────────────────────────────────────────────────
def forecast_territory(series: pd.Series, territory_id: str, fast_mode: bool = True) -> Dict[str, Any]:
    """
    Run model competition on one territory's monthly time series.
    Returns a dict of forecast outputs to be merged into the territory feature dict.
    """
    series    = series.sort_index().dropna()
    n_months  = len(series)
    avg_vol   = float(series.mean())
    holdout   = adaptive_holdout(avg_vol)
    tier      = data_tier(n_months)
    conf_cap  = tier_confidence_cap(tier)
    if fast_mode:
        model_set = {k: v for k, v in FAST_MODELS.items()
                     if avg_vol >= LOW_VOL_THRESHOLD or k in LOW_VOL_MODELS}
    else:
        model_set = LOW_VOL_MODELS if avg_vol < LOW_VOL_THRESHOLD else ALL_MODELS

    # Fallback: not enough data
    if n_months < MIN_TRAIN_POINTS + holdout:
        fc_vals      = np.full(FORECAST_HORIZON, max(avg_vol, 0))
        lo, hi       = build_ci(fc_vals, 0.20)
        fallback_wmape = 0.20
        return {
            "forecast_q1":  round(float(fc_vals.sum()), 2),
            "wmape":        fallback_wmape,
            "confidence":   round(min(1.0 - fallback_wmape, conf_cap), 4),
            "fc_lo_q1":     round(float(lo.sum()), 2),
            "fc_hi_q1":     round(float(hi.sum()), 2),
            "best_model":   "Mean (fallback - insufficient data)",
            "data_tier":    tier,
            "data_months":  n_months,
        }

    # Outlier cap on training only
    raw_train = series.iloc[:-holdout]
    train     = winsorise(raw_train)
    actual    = series.iloc[-holdout:].values

    best_name, best_wmape_val = "Naive", np.nan
    best_score = np.inf

    for name, func in model_set.items():
        try:
            preds = func(train, holdout)
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                continue
            w = wmape(actual, preds)
            if not np.isnan(w) and w < best_score:
                best_score     = w
                best_name      = name
                best_wmape_val = w
        except BaseException:
            continue

    # Re-fit winner on full winsorised series
    full_win = winsorise(series)
    try:
        fc_vals = ALL_MODELS[best_name](full_win, FORECAST_HORIZON)
    except BaseException:
        fc_vals = np.full(FORECAST_HORIZON, avg_vol)
    fc_vals = np.maximum(fc_vals, 0)

    if np.isnan(best_wmape_val):
        best_wmape_val = 0.20  # fallback error estimate

    lo, hi = build_ci(fc_vals, best_wmape_val)
    conf   = min(max(0.0, 1.0 - best_wmape_val), conf_cap)

    return {
        "forecast_q1":  round(float(fc_vals.sum()), 2),
        "wmape":        round(float(best_wmape_val), 4),
        "confidence":   round(conf, 4),
        "fc_lo_q1":     round(float(lo.sum()), 2),
        "fc_hi_q1":     round(float(hi.sum()), 2),
        "best_model":   best_name,
        "data_tier":    tier,
        "data_months":  n_months,
    }


# ── Pipeline: run all territories ─────────────────────────────────────────────
def run_forecasts(df: pd.DataFrame, territories: List[Dict], fast_mode: bool = True) -> List[Dict]:
    """
    Run forecast competition for every territory in `territories`.
    Merges forecast outputs back into each territory dict.
    df must have columns: territory_id, transaction_date, metric_value.
    """
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["_month"] = df["transaction_date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby(["territory_id", "_month"])["metric_value"]
        .sum()
        .reset_index()
        .rename(columns={"_month": "month"})
    )

    # Build lookup: territory_id -> Series
    series_map: Dict[str, pd.Series] = {}
    for tid, grp in monthly.groupby("territory_id"):
        s = grp.set_index("month")["metric_value"]
        s.index = pd.DatetimeIndex(s.index)
        series_map[str(tid)] = s

    enriched = []
    for t in territories:
        tid = t["territory_id"]
        if tid in series_map:
            fc_data = forecast_territory(series_map[tid], tid, fast_mode=fast_mode)
        else:
            # Territory in features but no monthly rows — use mean fallback
            fc_data = {
                "forecast_q1": round(t.get("last_q", 0), 2),
                "wmape": 0.20, "confidence": 0.50,
                "fc_lo_q1": 0.0, "fc_hi_q1": 0.0,
                "best_model": "Mean (no data)", "data_tier": "limited", "data_months": 0,
            }
        enriched.append({**t, **fc_data})

    return enriched
