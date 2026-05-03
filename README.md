# ProcDNA Intelligence
### Enterprise IC Planning Platform — v2.0

A full-stack Incentive Compensation planning tool. Upload territory sales data, answer 6–10 guided questions, and get a statistically optimal quarterly goal plan with forecasts, fairness scoring, and an Excel export.

---

## Folder Structure

```
procdna_structured/
│
├── README.md                   ← you are here
│
├── backend/                    ← all Python / FastAPI files
│   ├── main.py                 ← FastAPI app, all API endpoints
│   ├── scenario_engine.py      ← goal calculation (4 methods)
│   ├── scoring.py              ← 7 fairness metrics, composite score
│   ├── forecaster.py           ← 15 forecast models, wMAPE competition
│   ├── pruner.py               ← Pareto dominance + similarity pruning
│   ├── export.py               ← 4-sheet Excel, CSV, guardrails, delta
│   ├── chat.py                 ← IC Advisory conversational logic
│   ├── tools.py                ← Anthropic tool definitions
│   ├── tool_schemas.py         ← Tool JSON schemas
│   └── rule_engine.py          ← Advisory rule engine
│
├── frontend/
│   └── app.html                ← Single-page app (all 4 units)
│
└── deploy/
    ├── render.yaml             ← Render auto-deploy config
    ├── requirements.txt        ← Python dependencies
    ├── runtime.txt             ← Python 3.11
    └── DEPLOY.md               ← Step-by-step Render deploy guide
```

> **Render deployment note:** Render expects `main.py`, `app.html`, and all backend files at the **root level** of your repo. When pushing to GitHub, place all `backend/` files and `frontend/app.html` at root — use this folder structure for local navigation only.

---

## The Four Units

| Unit | Name | What it does |
|------|------|-------------|
| 01 | IC Advisory | 10 questions → AI-generated IC recommendation via Groq |
| 02 | Goal Setting | Upload → EDA dashboard → 6 questions → optimal quarterly goals |
| 03 | Forecasting | 15-model competition per territory → quarterly territory forecasts |
| 04 | Comparison | Best plan vs custom sliders → live territory bar chart |

---

## Goal Setting — How It Works

### Input
- Weekly or monthly sales file (`.xlsx`, `.xls`, `.csv`)
- Required columns: `terr_id`, `terr_nm`, `regn_id`, `regn_nm`, `wk_end_dt`, `sales`
- Auto-detected and auto-aggregated from weekly → monthly if needed

### The 3-Step Pipeline

**Step 1 — Parse** (`POST /api/goals/step1-parse`)
Reads file, aggregates to monthly, detects last full quarter and next quarter, computes Q1–Q4 and H6/H9/H12 per territory. Returns EDA data immediately.

**Step 2 — Forecast** (`POST /api/goals/step2-forecast`)
One API call per model per territory (15 models × N territories). Models compete on wMAPE holdout. Best model per territory wins. `forecast_q1` = 3-month ahead forecast.

**Step 3 — Score** (`POST /api/goals/step3-score`)
Generates up to 252 scenarios (method × split × weights), calculates goals for each, scores on 7 fairness metrics, prunes via Pareto dominance + similarity clustering, returns The One.

### Goal Calculation Methods

All goals are **quarterly** throughout. National Forecast (NF) is entered by the user as a **quarterly target**.

| Method | How it works |
|--------|-------------|
| **Historical Aggregate** | `Goal = NF × (territory H6/H9/H12 / national total)` |
| **Historical Quarter-Wise** | Weighted blend of Q1×w1 + Q2×w2 + Q3×w3 (Q1 = most recent). Market share applied to NF. |
| **Absolute Growth Quarter-Wise** | Builds on Historical Quarter-Wise base. Distributes growth budget via weighted QoQ absolute deltas. |
| **Percent Growth Quarter-Wise** | Same as Absolute but uses % deltas. |

> Q1 = most recent quarter, Q2 = one prior, Q3 = two prior, Q4 = three prior.

### Regional / Individual Blend

```
FG = regional_share × Regional_Goal + individual_share × Individual_Goal
```

Seven split options from 100/0 Regional to 0/100 Individual, plus five Hybrid variants (70/30 through 30/70).

### Equal Allocation (EA) Blending

```
Final_Goal = hist_pct × scenario_goal + (1 − hist_pct) × (NF / N_territories)
```

Applied as post-processing after goal calculation. `hist_pct = 1.0` means 100% history-driven. `hist_pct = 0.0` means every territory gets an equal share of NF.

### Attainment

```
Attainment = forecast_q1 / quarterly_goal
On Target  = Attainment ∈ [0.85, 1.15]
```

Both numerator and denominator are quarterly. No annualisation.

---

## Scenario Scoring — 7 Fairness Metrics

| Metric | Symbol | Formula |
|--------|--------|---------|
| Proportionality | P | Pearson corr(Goals, Territory size), normalised to [0,1] |
| Attainability | A | Confidence-weighted % of territories on target [0.85–1.15] |
| Consistency | C | 1 − mean(CV per region) |
| Growth Alignment | Ga | Pearson corr(Territory trend, Goal growth rate) |
| Volatility Penalty | V | 1 − mean(penalty for high-vol territories with above-avg goal growth) |
| Cost Efficiency | K | 1 − \|total_goals − NF\| / NF |
| Goal Spread | S | max(0, 1 − (P90/P10 − 1) / 9) |

**Composite Score** (Fairness preset):
```
P×0.30 + A×0.25 + C×0.20 + Ga×0.10 + V×0.10 + K×0.03 + S×0.02
```

---

## Guardrails (4 Rules)

| Rule | Threshold |
|------|-----------|
| Budget alignment | Total goals within ±20% of NF |
| Goal floor | No territory below 50% of average goal |
| Goal ceiling | No territory above 200% of average goal |
| Attainability floor | ≥70% of territories on target |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/goals/step1-parse` | Parse file, detect quarters, return EDA data |
| POST | `/api/goals/step2-forecast` | Run one forecast model for all territories |
| POST | `/api/goals/step3-score` | Score all scenarios, prune, return best plan |
| GET | `/api/session/{sid}/summary` | Best plan + territory goals (for Comparison reload) |
| GET | `/api/export/session/{sid}/excel` | 4-sheet Excel download |
| GET | `/api/export/session/{sid}/csv` | CSV download |
| GET | `/api/guardrails/{sid}` | 4-rule guardrail check on best scenario |
| GET | `/api/compare/{sid}` | Delta insights — best vs preferred scenario |
| GET | `/health` | Health check |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML / Forecasting | NumPy, SciPy, statsmodels, scikit-learn, XGBoost |
| Data | pandas, openpyxl |
| AI (IC Advisory) | Anthropic Claude via Groq (llama-3.1-8b-instant) |
| Frontend | Vanilla JS, single HTML file, Canvas API charts |
| Hosting | Render (free or Starter tier) |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | For IC Advisory narrative generation |
| `GROQ_API_KEY` | Yes | For Groq LLM calls in Unit 01 |
| `PORT` | Auto | Set by Render automatically |

---

## Quick Deploy

```bash
# 1. Place all backend/ files and frontend/app.html at repo root
# 2. Push to GitHub
git init && git add . && git commit -m "ProcDNA v2"
git remote add origin https://github.com/YOUR_USERNAME/procdna-goals.git
git push -u origin main

# 3. Connect repo on render.com → New Web Service
#    Add env vars: ANTHROPIC_API_KEY, GROQ_API_KEY
#    Render auto-detects render.yaml and deploys
```

Full step-by-step instructions in `deploy/DEPLOY.md`.

---

## Known Limitations / Next Up

- **Payout structure** — Regional and Individual goal tracking is computed but not yet surfaced as dual-tracked payout in the UI or export. Planned next.
- **EA cascade level** — Equal Allocation currently applies nationally (NF ÷ all territories). Regional-first cascade (region gets share, then equal within region) is designed but not yet built.
- **postMessage error** — File upload triggers a browser `FormData cannot be cloned` error in some environments. Under investigation.
- **Render free tier** — Cold starts after 15min idle take ~30s. Upgrade to Starter ($7/mo) for client demos.
- **Excel Sheet 2** — Audit trail sheet (showing step-by-step formula inputs per territory) designed but not yet built.

---

*ProcDNA Intelligence — built with FastAPI + Claude*
