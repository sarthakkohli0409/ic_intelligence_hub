# ProcDNA Intelligence — GitHub + Render Deploy Guide

Two steps: push to GitHub, connect to Render.
Total time: ~10 minutes the first time, ~60 seconds for every update after.

---

## Repo structure

```
procdna-goals/               ← this repo
├── goals-service/
│   ├── main.py              ← FastAPI app + all routes
│   ├── scoring.py           ← 7 metrics, 3 presets
│   ├── forecaster.py        ← 15-model forecast engine
│   ├── pruner.py            ← dominance, clustering, Pareto
│   ├── scenario_engine.py   ← scenario generation + goal calc
│   ├── export.py            ← Excel 4-sheet, CSV, guardrails
│   ├── requirements.txt
│   ├── render.yaml          ← Render auto-detects this
│   └── README.md
└── DEPLOY.md                ← this file
```

---

## Step 1 — Create GitHub repo

```bash
# Unzip the package
unzip procdna_goals_service.zip
cd goals-service

# Initialise git
git init
git add .
git commit -m "Initial commit — ProcDNA Goals Service v2"

# Create repo on GitHub (go to github.com → New repository)
# Name it: procdna-goals
# Keep it private
# Do NOT initialise with README

# Then push
git remote add origin https://github.com/YOUR_USERNAME/procdna-goals.git
git branch -M main
git push -u origin main
```

---

## Step 2 — Deploy on Render

1. Go to **render.com** → Sign in → **New +** → **Web Service**
2. Connect your GitHub account if not already connected
3. Select the `procdna-goals` repo
4. Render will auto-detect `render.yaml` — confirm the settings:
   - **Name:** `procdna-goals`
   - **Root Directory:** `goals-service`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2`
   - **Plan:** Starter ($7/month) — free tier spins down after 15min idle
5. Click **Advanced** → **Add Environment Variable**:
   - Key: `ANTHROPIC_API_KEY`
   - Value: `sk-ant-...` (your key)
6. Click **Create Web Service**

Render will build and deploy. Takes ~3 minutes the first time (installing scipy + xgboost).

Your live URL will be: `https://procdna-goals.onrender.com`

---

## Step 3 — Verify it's running

```bash
curl https://procdna-goals.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "ProcDNA Intelligence — Goals Service",
  "version": "2.0.0",
  "presets": ["fairness", "growth", "cost"]
}
```

Interactive API docs: `https://procdna-goals.onrender.com/docs`

---

## Step 4 — Point the frontend at your API

In `public/index.html`, find this line:

```javascript
var S = { ..., BACKEND: window.location.origin };
```

If you are deploying the frontend separately (e.g. Vercel/Netlify), change it to:

```javascript
var S = { ..., BACKEND: "https://procdna-goals.onrender.com" };
```

If both frontend and backend are on Render (same domain), keep `window.location.origin`.

---

## Deploying updates

Any `git push` to `main` triggers an automatic redeploy on Render:

```bash
# Edit a file
git add .
git commit -m "Update scoring weights"
git push
# Render redeploys in ~60 seconds
```

---

## New v2 API fields

The `/api/goals/generate` endpoint now accepts these additional form fields:

| Field | Default | Description |
|---|---|---|
| `preset` | `fairness` | Scoring preset: `fairness`, `growth`, or `cost` |
| `ic_plan_type` | `""` | Plan type from IC Advisory (e.g. `Hybrid`) — boosts matching scenarios |
| `ic_boost_lambda` | `0.05` | Boost strength for IC Advisory match (0–0.1) |

New endpoints:

| Endpoint | Description |
|---|---|
| `GET /api/compare/session/{sid}` | Delta insights — best vs preferred scenario |
| `GET /api/guardrails/{sid}` | 4-rule guardrail check on best scenario |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for AI narrative generation |
| `PORT` | Auto (Render sets it) | Server port |

---

## Estimated Render costs

| Plan | Cost | Notes |
|---|---|---|
| Free | $0 | Spins down after 15min idle — cold start ~30s |
| Starter | $7/month | Always on — recommended for sharing with clients |
| Standard | $25/month | More RAM — needed if running 40+ territories with XGBoost |

For a demo with <20 territories, Starter is sufficient.
For production with 40+ territories, use Standard.

