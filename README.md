<div align="center">

<!-- Hero Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=200&section=header&text=AI%20Football%20Intelligence%20Platform&fontSize=36&fontColor=ffffff&fontAlignY=38&desc=Production-Grade%20Football%20Analytics%20Powered%20by%20Machine%20Learning&descAlignY=58&descSize=16" width="100%"/>

<br/>

<!-- Badges Row 1 -->
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-FF6600?style=for-the-badge&logo=data:image/png;base64,&logoColor=white)](https://xgboost.readthedocs.io/)
[![Claude AI](https://img.shields.io/badge/Claude%20AI-Haiku%204.5-D97757?style=for-the-badge&logo=anthropic&logoColor=white)](https://docs.anthropic.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)

<!-- Badges Row 2 -->
[![Tests](https://img.shields.io/badge/Tests-472%20Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1.0-blue?style=for-the-badge)](pyproject.toml)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)](.)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-purple?style=for-the-badge)](https://github.com/astral-sh/ruff)

<!-- Badges Row 3 -->
[![ML](https://img.shields.io/badge/Machine%20Learning-VAEP%20%7C%20xG%20%7C%20xT-orange?style=flat-square)](.)
[![Football Analytics](https://img.shields.io/badge/Football%20Analytics-StatsBomb%20Format-green?style=flat-square)](.)
[![Architecture](https://img.shields.io/badge/Architecture-10%20Phase%20Pipeline-blueviolet?style=flat-square)](.)
[![Coverage](https://img.shields.io/badge/Test%20Coverage-Unit%20%7C%20Integration-brightgreen?style=flat-square)](tests/)

<br/>

> **"Transforming raw football event data into deep tactical intelligence through a production-grade AI pipeline."**

<br/>

[🚀 Quick Start](#-quick-start) · [📖 Documentation](#-project-overview) · [🏗️ Architecture](#️-system-architecture) · [🧠 ML Models](#-machine-learning-models) · [📊 Dashboard](#-dashboard--api) · [🐳 Docker](#-docker-deployment)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#️-system-architecture)
- [Core Concepts](#-core-concepts--formulas)
- [Machine Learning Models](#-machine-learning-models)
- [Tactical Intelligence Layer](#-tactical-intelligence-layer)
- [Project Structure](#-project-structure)
- [Technologies Used](#️-technologies-used)
- [Quick Start](#-quick-start)
- [Usage & CLI](#-usage--cli-reference)
- [Example Output](#-example-output)
- [Testing](#-testing)
- [Docker Deployment](#-docker-deployment)
- [Why This Project Stands Out](#-why-this-project-stands-out)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)

---

## 🎯 Project Overview

The **AI Football Intelligence Platform** is a full-stack, production-grade football analytics system that transforms raw StatsBomb event data into deep tactical insights, player ratings, match reports, and real-time intelligence — all powered by machine learning.

### The Problem

Modern football generates vast amounts of event data (passes, shots, dribbles, tackles) but the vast majority of clubs and analysts lack the engineering infrastructure to:
- Quantify the **true value of every individual action** on the pitch
- Detect **tactical weaknesses** and **formation patterns** automatically
- Deliver those insights through a **real-time API and interactive dashboard**
- Do all of this in a **reproducible, testable, production-ready pipeline**

### The Solution

This platform implements the **VAEP (Valuing Actions by Estimating Probabilities)** framework — the state-of-the-art method from KDD 2019 — along with **Expected Goals (xG)** and **Expected Threat (xT)** models, tactical intelligence modules, a FastAPI backend, and a Streamlit dashboard, all orchestrated through a 10-phase automated pipeline.

### Who Is It For?

| Audience | Use Case |
|---|---|
| 🏟️ Football Clubs | Player scouting, tactical analysis, opponent preparation |
| 📊 Data Scientists | Research platform for football ML models |
| 💻 Engineers | Reference architecture for sports analytics systems |
| 🎓 Students | Portfolio-quality demonstration of end-to-end ML engineering |

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🤖 Machine Learning Core
- **xG Model** — Shot quality estimation using logistic regression with isotonic calibration
- **xT Model** — Expected threat via 18×12 Markov-chain zone grid with value iteration
- **P(scores) Model** — XGBoost classifier estimating scoring probability per game state
- **P(concedes) Model** — XGBoost classifier estimating conceding probability per game state
- **105-dimensional feature vectors** — spatial, temporal, and contextual features per action

</td>
<td width="50%">

### ⚙️ Data Pipeline
- **StatsBomb CSV ingestion** — priority-based column resolution, deduplication
- **SPADL normalisation** — 22 action types, period/timestamp sorted
- **Possession detection** — time-gap + team-change segmentation
- **Game state labelling** — k-step look-ahead goal/concede labels
- **Parquet store** — snappy-compressed, match-partitioned storage

</td>
</tr>
<tr>
<td width="50%">

### 🧠 VAEP Engine
- **State value computation** — V(Sᵢ) = P_scores − P_concedes
- **Action value computation** — V(aᵢ) = V(Sᵢ) − V(Sᵢ₋₁)
- **Offensive/defensive decomposition** per action
- **Per-player VAEP aggregation** with VAEP/90 normalisation
- **Per-team VAEP summaries**

</td>
<td width="50%">

### 🎯 Tactical Intelligence
- **5-method player ranking** — overall, offensive, defensive, per-90, efficiency
- **Zone-level weakness detection** — 4 risk levels (critical/high/medium/low)
- **Formation analysis** — k-means clustering → "4-3-3" style string
- **Pressure maps** — intensity, opponent threat, action density
- **Match/Player/Team reports** — fully JSON-serialisable

</td>
</tr>
<tr>
<td width="50%">

### 🌐 API & Dashboard
- **FastAPI REST backend** — 40 endpoints across 9 tags, Pydantic v2 response models
- **Streamlit dashboard** — 12 interactive pages with Plotly visualisations
- **Custom `/docs` UI** — Swagger UI with dark/light theme toggle, live endpoint search, `persistAuthorization` (JWT survives page refresh)
- **WebSocket live hub** — real-time match events at `/api/v1/live/matches/{id}` (pro_analyst+)
- **Claude API scouting** — AI-generated scouting + tactical reports (`club_scout+` tier)

</td>
<td width="50%">

### 🏗️ Production Infrastructure
- **Unified CLI** — single entry point for all operations
- **Docker + Compose** — multi-service containerisation (postgres · redis · api · worker · dashboard)
- **472 tests** — unit + integration coverage, 0 warnings
- **Redis caching with football-aware TTLs** — 30 s live, 1 h stats, 24 h predictions
- **JWT + 4-tier RBAC** — free_user / pro_analyst / club_scout / admin
- **Alembic migrations · Prometheus `/metrics` · structured JSON logs**
- **Model registry** — joblib serialisation + metadata tracking

</td>
</tr>
</table>

---

## 🏗️ System Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AI FOOTBALL INTELLIGENCE PLATFORM                        │
│                          End-to-End Data Flow                                │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌───────────┐     ┌──────────────┐     ┌───────────────┐
  │  RAW CSV │────▶│ INGESTION │────▶│PREPROCESSING │────▶│   FEATURES    │
  │StatsBomb │     │  Phase 2  │     │   Phase 3    │     │   Phase 4     │
  └──────────┘     └───────────┘     └──────────────┘     └───────┬───────┘
                                                                   │
         ┌─────────────────────────────────────────────────────────▼──────┐
         │                      ML MODELS  (Phase 5)                       │
         │   ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌─────────────┐ │
         │   │  xG Model │  │ xT Model │  │  P_scores  │  │  P_concedes │ │
         │   │ LogReg+Cal│  │ Markov   │  │  XGBoost   │  │   XGBoost   │ │
         │   └──────────┘  └──────────┘  └────────────┘  └─────────────┘ │
         └──────────────────────────┬──────────────────────────────────────┘
                                    │
         ┌──────────────────────────▼──────────────────────────────────────┐
         │                     VAEP ENGINE  (Phase 6)                       │
         │      V(Sᵢ) = P_scores(Sᵢ) − P_concedes(Sᵢ)                   │
         │      V(aᵢ) = V(Sᵢ) − V(Sᵢ₋₁)    [within possession]          │
         └──────────────────────────┬──────────────────────────────────────┘
                                    │
         ┌──────────────────────────▼──────────────────────────────────────┐
         │              TACTICAL INTELLIGENCE  (Phase 7)                    │
         │   Player Rankings │ Weakness Zones │ Formations │ Pressure Maps  │
         └──────────┬─────────────────────────────────────┬────────────────┘
                    │                                     │
         ┌──────────▼──────────┐             ┌───────────▼──────────┐
         │   FastAPI Backend   │             │  Streamlit Dashboard  │
         │     Phase 8         │             │      Phase 9          │
         │  REST API  40 routes │             │  12 interactive pages │
         └─────────────────────┘             └──────────────────────┘
```

### Package Architecture

```
football_ai/
│
├── ingestion/          ← Phase 2: Raw data → validated events → Parquet
│   ├── adapters/       ← StatsBomb CSV adapter with priority column resolution
│   ├── validators/     ← Per-row schema validation with severity levels
│   └── storage/        ← Snappy-compressed, match-partitioned Parquet store
│
├── preprocessing/      ← Phase 3: SPADL normalisation + possession graph
│   ├── spadl/          ← 22 action type mapping, period/timestamp sort
│   ├── pitch/          ← 18×12 zone grid mapper
│   ├── possession/     ← Time-gap + team-change possession detector
│   └── game_state/     ← k-step look-ahead label generator
│
├── features/           ← Phase 4: 105-dimensional feature engineering
│   ├── spatial.py      ← 25 features: distance/angle to goal, zones, flags
│   ├── temporal.py     ← 37 features: time gaps, last-k action one-hots
│   └── contextual.py   ← 43 features: score diff, body part, result one-hots
│
├── ml/                 ← Phase 5: Machine learning models + registry
│   ├── models/         ← xg_model, xt_model, p_scores_model, p_concedes_model
│   ├── training/       ← Trainer, evaluator, MLTrainingPipeline
│   └── serving/        ← ModelRegistry with joblib + metadata.yaml
│
├── vaep/               ← Phase 6: VAEP engine
│   ├── state_value.py  ← V(Sᵢ) computation
│   ├── action_value.py ← V(aᵢ) with offensive/defensive decomposition
│   ├── aggregator.py   ← Per-player VAEP/90, per-team summaries
│   └── pipeline.py     ← End-to-end orchestration → 3 Parquet outputs
│
├── tactical/           ← Phase 7: Tactical intelligence layer
│   ├── player_ranker.py      ← 5 ranking methods + tier assignment
│   ├── weakness_detector.py  ← Zone-level opponent VAEP + risk levels
│   ├── formation_analyser.py ← k-means → "4-3-3" formation string
│   ├── pressure_map.py       ← 3 heatmaps + high-risk zone intersection
│   ├── report_builder.py     ← MatchReport / PlayerReport / TeamReport
│   └── pipeline.py           ← TacticalPipeline → 6 artefact files
│
├── api/                ← Phase 8+: FastAPI REST backend
│   ├── main.py         ← app factory + middleware + custom /docs route
│   ├── docs.py         ← custom Swagger UI (dark/light toggle, persistAuthorization)
│   ├── dependencies.py ← get_db, get_cache, get_current_user, require_role
│   ├── endpoints/health.py  ← /health · /health/ready · /metrics
│   ├── v1/router.py    ← aggregator mounted at /api/v1
│   ├── v1/endpoints/   ← 10 routers → 40 endpoints / 9 tags
│   │                     (auth · players · matches · teams · tactical ·
│   │                      predictions · analysis · live · admin · dashboard)
│   └── schemas/        ← Pydantic v2 response DTOs
│
└── dashboard/          ← Phase 9: Streamlit dashboard
    ├── data_loader.py  ← @st.cache_data wrappers over Parquet store
    └── pages/          ← 12 pages: overview, match_analysis, player_analysis,
                           player_comparison, rankings, tactical_heatmap,
                           tactical_report, passing_network, xt_heatmap,
                           formation_visualizer, action_timeline, demo_mode
```

---

## 🧠 Core Concepts & Formulas

### Expected Goals (xG)

> **xG** measures the quality of a shot — the probability it results in a goal, based purely on the circumstances of the shot itself (distance, angle, body part, pressure).

```
xG(shot) = P(goal | shot context)     ∈ [0, 1]
```

A tap-in from 3 metres has xG ≈ 0.85. A long-range effort from 35 metres has xG ≈ 0.02.

---

### Expected Threat (xT)

> **xT** quantifies how threatening each pitch zone is, by estimating the probability that possessing the ball in a given zone will lead to a goal within the next few actions.

```
xT(zone z) = P(shoot | z) × G(z)
           + P(move  | z) × Σ P(z → z') × xT(z')

where:
    G(z)         = empirical goal rate from zone z
    P(move | z)  = probability ball moves to another zone
    P(z → z')    = zone-to-zone transition probability (Markov chain)
```

Solved via **value iteration** over an 18×12 pitch zone grid (216 zones).

---

### VAEP — Valuing Actions by Estimating Probabilities

> **VAEP** assigns a numerical value to **every on-ball action** (not just shots), measuring how much it changes the team's probability of scoring or conceding.

#### Game State Value

```
V(Sᵢ) = P^k_scores(Sᵢ) − P^k_concedes(Sᵢ)     ∈ [−1, 1]

where:
    P^k_scores(Sᵢ)   = P(team scores within next k actions | state Sᵢ)
    P^k_concedes(Sᵢ) = P(team concedes within next k actions | state Sᵢ)
```

#### Action Value

```
V(aᵢ) = V(Sᵢ) − V(Sᵢ₋₁)     [within the same possession]
```

#### Offensive / Defensive Decomposition

```
vaep_offensive(aᵢ) = max( 0,  ΔP_scores(aᵢ) )
vaep_defensive(aᵢ) = max( 0, −ΔP_concedes(aᵢ) )
```

#### Player Rating

```
VAEP(player) = Σ V(aᵢ)   for all aᵢ by player
VAEP/90(p)   = VAEP(p) / minutes_played × 90
```

> *Reference: Decroos et al. (2019). "Actions Speak Louder Than Goals: Valuing Player Actions in Soccer." KDD 2019.*

---

## 🤖 Machine Learning Models

### Model Overview

| Model | Architecture | Training Target | Input | Output |
|---|---|---|---|---|
| **xG** | Logistic Regression + Isotonic Calibration | P(goal \| shot) | 36 spatial + contextual features | Probability ∈ [0,1] |
| **xT** | Markov Chain Grid (18×12) | Threat per zone | Zone transitions from SPADL | xT value per zone |
| **P_scores** | XGBoost Classifier | P(score in next k actions) | 105-dim feature vector | Probability ∈ [0,1] |
| **P_concedes** | XGBoost Classifier | P(concede in next k actions) | 105-dim feature vector | Probability ∈ [0,1] |
| **Match Predictor** | XGBoost Classifier | Home/draw/away outcome | 15 football features — **VAEP balance, xT diff**, xG rolling diff, form, home advantage, rest days, shot-quality ratio, … | Calibrated `{home, draw, away}` probabilities |
| **AI Scouting** | Anthropic Claude (`claude-haiku-4-5`) | Structured scouting + tactical narrative | Pre-computed VAEP / xT / formation payloads (grounded in numbers, not raw text) | JSON-shaped report — `club_scout+` tier, cached 24 h in Redis |

---

### xG Model — Expected Goals

**Purpose:** Estimate the probability that a shot results in a goal.

**Training data:** Shot and header events only (filtered from full action set).

**Key design decisions:**
- `CalibratedClassifierCV` with isotonic regression ensures predicted probabilities match true goal rates
- Graceful fallback to uncalibrated model when positive class count is too small for CV
- `class_weight='balanced'` handles severe class imbalance (goals are ~10% of shots)

```python
# Architecture
LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs')
→ CalibratedClassifierCV(method='isotonic', cv=safe_cv)
```

---

### xT Model — Expected Threat

**Purpose:** Assign a threat value to every pitch zone using Markov-chain dynamics.

**Key design decisions:**
- 18×12 grid = 216 zones (StatsBomb pitch: 120m × 80m)
- Laplace smoothing (`add-1`) prevents zero-probability zones
- Value iteration converges in ~26 iterations (tolerance 1e-6)
- No train/test split — uses all available match data for better zone estimates

```
Grid range on FUS vs FAR match: xT ∈ [0.0003, 0.0885]
```

---

### P_scores & P_concedes Models

**Purpose:** The probabilistic backbone of the VAEP engine — estimates scoring/conceding probability for every single game state.

**Key design decisions:**
- XGBoost chosen for its performance on tabular data with class imbalance
- `scale_pos_weight` computed automatically from class ratio at training time
- Early stopping on a held-out 15% internal validation set
- Separate models for scoring and conceding — different features matter for each task

```python
XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=auto,  # n_negative / n_positive
    early_stopping_rounds=20
)
```

**Results on FUS Rabat vs FAR Rabat:**

| Model | ROC-AUC | Brier Score | Log Loss |
|---|---|---|---|
| P_scores | **0.9787** | 0.0074 | 0.0330 |
| P_concedes | **1.0000** | 0.0005 | 0.0033 |

---

## 🎯 Tactical Intelligence Layer

### Player Ranking Engine

Five independent ranking methods, each producing a sortable, tier-labelled DataFrame:

| Method | Sort Metric | Description |
|---|---|---|
| `overall` | `vaep_total` | Total VAEP accumulated across all actions |
| `offensive` | `vaep_offensive` | Positive scoring contribution |
| `defensive` | `vaep_defensive` | Concede-risk reduction contribution |
| `per_90` | `vaep_per_90` | VAEP normalised to 90 minutes |
| `efficiency` | `vaep_per_action` | VAEP quality per touch |

Tiers assigned by percentile: **Elite** (top 10%) · **Strong** (60–90%) · **Average** (25–60%) · **Below Average** (bottom 25%)

---

### Team Weakness Detection

Identifies pitch zones where **opponents consistently create value** against a given team:

```
opponent_VAEP(zone z, team T) = mean V(aᵢ) for all opponent actions
                                 starting in zone z against team T
```

Risk classification by within-match percentile:
- 🔴 **Critical** — top 10% of dangerous zones
- 🟠 **High** — 75th–90th percentile
- 🟡 **Medium** — 50th–75th percentile
- 🟢 **Low** — below median

---

### Formation Detection

Infers team formation from the spatial distribution of player on-ball actions:

```
1. Collect all action start positions (x, y) for a team
2. Normalise coordinates to [0, 1]
3. Run k-means (k=10 outfield players) on StandardScaler-transformed positions
4. Classify each centroid as defence / midfield / attack by x-position
5. Count players per depth zone → "4-3-3", "4-4-2", "3-5-2" etc.
```

Confidence score reflects how balanced the distribution is across depth zones.

---

### Pressure Maps

Three complementary zone-level heatmaps per team:

| Map | Measures | High Value Means |
|---|---|---|
| `pressure_intensity` | Fraction of own actions under pressure | Team is consistently pressed in this zone |
| `opponent_threat` | Mean opponent VAEP per zone | Opponents create danger here against this team |
| `action_density` | Normalised action count per zone | Most of the game is played here |

**High-risk zones** = zones in top quartile of **all three** maps simultaneously.

---

## 🛠️ Technologies Used

<table>
<tr><th>Category</th><th>Technology</th><th>Version</th><th>Purpose</th></tr>
<tr><td>🐍 Runtime</td><td>Python</td><td>3.11+</td><td>Core language</td></tr>
<tr><td>📊 Data</td><td>Pandas</td><td>2.1+</td><td>DataFrame operations throughout pipeline</td></tr>
<tr><td>🔢 Numerics</td><td>NumPy</td><td>1.26+</td><td>Array computation, feature engineering</td></tr>
<tr><td>🤖 ML</td><td>Scikit-learn</td><td>1.4+</td><td>xG model, calibration, evaluation</td></tr>
<tr><td>🚀 ML</td><td>XGBoost</td><td>2.0+</td><td>P_scores and P_concedes models</td></tr>
<tr><td>⚡ API</td><td>FastAPI</td><td>0.109+</td><td>REST backend, OpenAPI docs</td></tr>
<tr><td>🎨 Dashboard</td><td>Streamlit</td><td>1.30+</td><td>Interactive analytics dashboard</td></tr>
<tr><td>📈 Viz</td><td>Plotly</td><td>5.18+</td><td>Interactive charts and pitch visualisations</td></tr>
<tr><td>💾 Storage</td><td>PyArrow</td><td>14.0+</td><td>Snappy-compressed Parquet files</td></tr>
<tr><td>⚙️ Config</td><td>PyYAML</td><td>6.0+</td><td>Settings and logging configuration</td></tr>
<tr><td>💿 Serialisation</td><td>Joblib</td><td>1.3+</td><td>Model persistence (registry)</td></tr>
<tr><td>✅ Testing</td><td>Pytest</td><td>9.0+</td><td>472 unit + integration tests</td></tr>
<tr><td>🐳 Containers</td><td>Docker</td><td>24+</td><td>Multi-stage build, production deployment</td></tr>
<tr><td>🔍 Data Validation</td><td>Pydantic</td><td>2.5+</td><td>API schemas, event validation</td></tr>
<tr><td>🗄️ Database</td><td>PostgreSQL / SQLAlchemy</td><td>16 / 2.0+</td><td>Async ORM + Alembic migrations</td></tr>
<tr><td>⚡ Cache / Queue</td><td>Redis + arq</td><td>7 / 0.25+</td><td>Football-aware TTLs (30 s live · 1 h stats · 24 h predictions) + background jobs</td></tr>
<tr><td>🧠 AI NLP</td><td>Anthropic Claude</td><td>haiku-4-5</td><td>Scouting reports + tactical commentary (`club_scout+` tier)</td></tr>
<tr><td>📡 Realtime</td><td>WebSocket (FastAPI)</td><td>—</td><td>Live match event hub (`pro_analyst+` tier)</td></tr>
<tr><td>📊 Observability</td><td>Prometheus + JSON logs</td><td>—</td><td>`/metrics` scrape, structured logging, `X-Request-ID`</td></tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- (Optional) Docker Desktop

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-football-intelligence-platform.git
cd ai-football-intelligence-platform
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e ".[dev]"
```

> Installs the full stack from `pyproject.toml` (FastAPI, XGBoost, SQLAlchemy, Redis, Anthropic, pytest, ruff, …) and registers `football_ai` as an editable package so no `PYTHONPATH` tweaking is needed.

### 4. Run the Services

```bash
# Start the FastAPI backend (→ http://localhost:8000/docs for the custom Swagger UI)
uvicorn football_ai.api.main:app --reload --port 8000

# Start the Streamlit dashboard (→ http://localhost:8501)
streamlit run football_ai/dashboard/app.py

# Or spin up the full stack with Docker (postgres · redis · api · worker · dashboard)
docker-compose up --build

# Run the test suite
python -m pytest tests/ -q
```

### 5. (Optional) Run the Analytics Pipeline on a CSV

```bash
python scripts/cli.py run --input data/raw/fus_FAR.csv
python scripts/cli.py status
```

> ✅ You should see all 4 models trained and all artefact types present.

---

## 💻 Usage & CLI Reference

The platform ships with a **unified CLI** (`scripts/cli.py`) that exposes all operations through a single entry point.

### Run the Full 7-Phase Pipeline

```bash
# Full pipeline (ingestion → preprocessing → features → train → VAEP → tactical)
python scripts/cli.py run --input data/raw/fus_FAR.csv

# Force reprocess all phases
python scripts/cli.py run --input data/raw/fus_FAR.csv --force

# Skip model training (use existing saved models)
python scripts/cli.py run --input data/raw/fus_FAR.csv --skip-training

# Skip VAEP and tactical (features only)
python scripts/cli.py run --input data/raw/fus_FAR.csv --skip-vaep --skip-tactical
```

### Train ML Models

```bash
# Train all 4 models
python scripts/cli.py train

# Force retrain even if models exist
python scripts/cli.py train --force

# Train specific models only
python scripts/cli.py train --models xg xt
python scripts/cli.py train --models p_scores p_concedes
```

### Start the REST API

```bash
# Start on default port 8000
python scripts/cli.py api

# Custom host and port
python scripts/cli.py api --host 0.0.0.0 --port 8080

# Interactive API docs → http://localhost:8000/docs
```

### Start the Dashboard

```bash
# Start on default port 8501
python scripts/cli.py dashboard

# Custom port
python scripts/cli.py dashboard --port 8502

# Dashboard → http://localhost:8501
```

### Platform Status

```bash
python scripts/cli.py status
```

```
=======================================================
  Football AI Platform  v0.1.0
=======================================================

📂 Data artefacts:
   SPADL processed : 1 match(es)
   Feature matrices: 1 match(es)
   VAEP scored     : 1 match(es)
   Tactical reports: 1 match(es)

🤖 Model registry:
   xg            : ✓
   xt            : ✓
   p_scores      : ✓
   p_concedes    : ✓

📊 Match details:
   3813041
=======================================================
```

### Run Tests

```bash
python scripts/cli.py test --suite unit
python scripts/cli.py test --suite integration
python scripts/cli.py test --suite all --verbose
```

### Clean Generated Artefacts

```bash
python scripts/cli.py clean --all       # everything
python scripts/cli.py clean --models    # model files only
python scripts/cli.py clean --data      # processed data only
python scripts/cli.py clean --logs      # log files only
```

### API Endpoints Quick Reference

```bash
# Health check
curl http://localhost:8000/health

# List all processed matches
curl http://localhost:8000/matches

# Full match analysis
curl http://localhost:8000/matches/3813041

# Player rankings (top 10 by VAEP)
curl "http://localhost:8000/players/rankings?metric=vaep_total&limit=10"

# Player profile
curl "http://localhost:8000/players/138962?match_id=3813041"

# Team profile with weaknesses
curl http://localhost:8000/teams/3813041/2761

# Full tactical intelligence report
curl http://localhost:8000/tactical/3813041
```

---

## 📊 Example Output

### Player Rankings (FUS Rabat vs FAR Rabat — match 3813041)

```
Rank │ Player                              │ Team        │ VAEP   │ Off.   │ Def.   │ VAEP/90
─────┼─────────────────────────────────────┼─────────────┼────────┼────────┼────────┼────────
  1  │ El Mehdi Moubarik                   │ FUS Rabat   │ +0.957 │ +1.015 │ +0.024 │  0.183
  2  │ Emmanuel Imanishimwe                │ FAR Rabat   │ +0.719 │ +0.572 │ +0.239 │  0.137
  3  │ Edilson Alberto Monteiro S. Borges  │ FAR Rabat   │ +0.614 │ +0.775 │ +0.023 │  0.117
  4  │ Ismail Khafi                        │ FUS Rabat   │ +0.505 │ +1.793 │ +0.058 │  0.096
  5  │ Mohamed Chibi                       │ FAR Rabat   │ +0.426 │ +0.489 │ +0.053 │  0.081
```

### Match Tactical Report Summary

```
╔══════════════════════════════════════════════════════════════╗
║              TACTICAL REPORT — Match 3813041                 ║
╠══════════════════════════════════════════════════════════════╣
║  FUS Rabat                    vs          FAR Rabat           ║
╠══════════════════════════════════╦═══════════════════════════╣
║  xG          1.877               ║  xG          1.139         ║
║  VAEP        2.089               ║  VAEP        2.040         ║
║  Formation   3-3-3               ║  Formation   2-4-3         ║
║  Actions     1,461               ║  Actions     1,444         ║
╠══════════════════════════════════╩═══════════════════════════╣
║  Weakness Analysis:                                           ║
║  FUS Rabat  → 183 zones analysed │ 19 critical │ 27 high-risk ║
║  FAR Rabat  → 159 zones analysed │ 16 critical │ 24 high-risk ║
╚══════════════════════════════════════════════════════════════╝
```

### Pipeline Execution Log

```
Phase 2  ─ Ingestion          3,023 events ingested · 0 errors · 0.65s
Phase 3  ─ Preprocessing      2,905 SPADL actions · 928 possessions · 0.36s
Phase 4  ─ Feature Eng.       105 feature columns · 2,905 rows · 0.18s
Phase 5  ─ Model Training     xG ✓  xT ✓  P_scores ✓  P_concedes ✓ · 2.3s
Phase 6  ─ VAEP Engine        2,905 actions scored · 29 players ranked · 0.26s
Phase 7  ─ Tactical Intel.    Weaknesses ✓ · Formations ✓ · Reports ✓ · 0.85s
──────────────────────────────────────────────────────────────────────────────
Total pipeline time: ~4.7 seconds for a complete 90-minute match
```

---

## 🧪 Testing

The platform ships with **472 tests** across unit and integration suites, enforcing zero `RuntimeWarning` tolerance.

### Run All Tests

```bash
# Full suite — unit + integration
python -m pytest tests/ -q
# Expected: 472 passed, 0 warnings

# Also acceptable — explicitly fail on RuntimeWarning
python -m pytest tests/ -W error::RuntimeWarning -q

# Unit tests only (fast — ~25 s)
python -m pytest tests/unit/ -v

# Integration tests only (end-to-end — ~12 s, in-memory SQLite + in-memory cache)
python -m pytest tests/integration/ -v
```

### Test Coverage by Module

The suite covers the analytics core (ingestion → VAEP → tactical), the full API surface (all 40 endpoints across 9 tags, including 401 / 403 / 404 / 429 branches), auth + rate-limiting, Redis caching, the API-Football client (respx-mocked), arq background jobs, the WebSocket live hub, and the Claude scouting layer.

| Area | Test files | Covers |
|---|---|---|
| Core analytics | `test_config`, `test_utils`, `test_ingestion`, `test_preprocessing`, `test_features` | Settings, geometry, CSV→SPADL→105-dim features |
| ML models | `test_models`, `test_vaep`, `test_tactical` | xG / xT / P_scores / P_concedes / ranker / formations / pressure |
| Predictions & AI | `test_predictions`, `test_analysis` | XGBoost match predictor, Claude scouting (respx-mocked) |
| API surface | `test_api_*` (per-tag) | All 40 endpoints across 9 tags, incl. 401/403/404/429 branches |
| Security | `test_auth`, `test_ratelimit`, `test_security` | JWT, `require_role`, rate-limit headers, security middleware |
| Infra | `test_cache`, `test_external`, `test_tasks`, `test_live_ws` | Redis TTLs, API-Football client, arq worker, WebSocket broadcast |
| Dashboard | `test_dashboard` | Data loaders, page imports, Plotly builders |
| End-to-end | `test_pipeline_integration` | Full ingest → VAEP → tactical pipeline |
| **Total** | — | **472 tests · 0 warnings** |

---

## 🐳 Docker Deployment

### Build and Start

```bash
# Copy environment file
cp .env.example .env

# Build and start the full stack (postgres · redis · api · worker · dashboard)
docker-compose up --build

# Just the API + dashboard (no worker)
docker compose up api dashboard

# One-shot analytics pipeline
docker compose --profile pipeline up

# One-shot model training
docker compose --profile training up trainer
```

### Service URLs

| Service | URL | Description |
|---|---|---|
| API | http://localhost:8000 | FastAPI REST backend |
| Custom Swagger UI | http://localhost:8000/docs | Dark/light theme toggle · live endpoint search · `persistAuthorization` (JWT survives refresh) |
| ReDoc | http://localhost:8000/redoc | Alternate API docs |
| Prometheus scrape | http://localhost:8000/metrics | Text-format exposition |
| Readiness probe | http://localhost:8000/health/ready | Pings DB + cache (use this for LB health checks) |
| Dashboard | http://localhost:8501 | Streamlit analytics UI (12 pages) |

### Environment Variables (`.env`)

```env
API_PORT=8000
DASHBOARD_PORT=8501
API_WORKERS=2
INPUT_CSV=fus_FAR.csv
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Docker Architecture

```
docker compose
│
├── postgres     → PostgreSQL 16 (primary store; named volume pgdata)
│                  healthcheck: pg_isready
│
├── redis        → Redis 7 with AOF (cache + arq queue; named volume redisdata)
│                  healthcheck: redis-cli ping
│
├── api          → uvicorn football_ai.api.main:app  (port 8000)
│                  runs: alembic upgrade head → uvicorn
│                  depends_on: postgres + redis (healthy)
│                  healthcheck: GET /health/ready (probes DB + cache)
│
├── worker       → arq football_ai.tasks.worker      (profile: worker)
│                  background jobs: API-Football sync, cache warm, prediction backfill
│
├── dashboard    → streamlit run app.py               (port 8501)
│                  depends_on: api (healthy)
│
├── pipeline     → python scripts/run_pipeline.py     (profile: pipeline)
│                  exits when complete
│
└── trainer      → python scripts/train_models.py     (profile: training)
                   exits when complete
```

---

## 🏆 Why This Project Stands Out

### 1. 📐 Production-Grade Architecture
Not a notebook. Not a script. A **fully modular, layered system** with clean separation between ingestion, preprocessing, features, models, VAEP, tactical intelligence, API, and dashboard — each independently testable and replaceable.

### 2. 🔬 State-of-the-Art Football Analytics
Implements **three complementary frameworks** (xG, xT, VAEP) that are used by professional clubs and published in top-tier research venues (KDD, IJCAI). The VAEP implementation faithfully follows the original paper's methodology.

### 3. 🧪 Exceptional Test Quality
**472 tests** including a full **integration test suite** that runs Phases 2–8 on a synthetic 200-row CSV — verifying that every phase produces correct artefacts and that downstream phases can consume upstream outputs. Zero `RuntimeWarning` tolerance enforced.

### 4. 🏗️ True End-to-End Pipeline
From raw CSV to interactive dashboard in a single command. The pipeline produces **14 distinct artefact types** (Parquet files, JSON reports, joblib models, metadata YAML) that are all versioned by match ID.

### 5. 🌐 Complete API + Dashboard
Not a bare ML model — a **full product** with a typed REST API (40 endpoints across 9 tags, Pydantic v2 response models, dependency injection, JWT + 4-tier RBAC, global exception handling, WebSocket live hub, Claude-powered scouting) and a 12-page interactive Streamlit dashboard with Plotly visualisations.

### 6. 🐳 Containerised & Deployable
Multi-stage Dockerfile (builder → runtime), non-root user, health checks, and a `docker-compose.yml` with profile-gated services for pipeline, training, API, and dashboard.

---

## 🔮 Future Roadmap

```
Q2 2025 ─ Multi-match training
           ├── Cross-season VAEP leaderboards
           ├── Transfer value estimation from VAEP curves
           └── Player similarity search (embedding-based)

Q3 2025 ─ Real-Time Inference
           ├── Live StatsBomb 360 / Opta data stream ingestion
           ├── WebSocket endpoint for live VAEP scoring
           └── In-match tactical alert system

Q4 2025 ─ Advanced Models
           ├── Expected Assists (xA) model
           ├── Goalkeeper-specific VAEP framework
           ├── Deep learning sequence model (Transformer over action chains)
           └── Graph Neural Network for team-level tactical embedding

2026 Q1 ─ Delivered (shipped in Phases 1–10)
           ├── ✅ PostgreSQL backend (async SQLAlchemy 2.0 + Alembic)
           ├── ✅ JWT auth + 4-tier RBAC (free_user / pro_analyst / club_scout / admin)
           ├── ✅ Redis caching with football-aware TTLs (30 s / 1 h / 24 h)
           ├── ✅ WebSocket live match hub (/api/v1/live/matches/{id})
           ├── ✅ XGBoost match outcome predictor (15 football features incl. VAEP + xT)
           ├── ✅ Claude API scouting reports (club_scout+ tier)
           ├── ✅ API-Football sync with quota tracking (arq worker)
           ├── ✅ Prometheus /metrics + structured JSON logging
           ├── ✅ CI/CD pipeline (GitHub Actions · ruff · pytest · coverage gate)
           └── ✅ Custom /docs UI with dark/light theme toggle

2026+   ─ Next
           ├── Multi-league dataset support (Premier League, La Liga, etc.)
           ├── Kubernetes deployment manifests
           └── Player similarity search (embedding-based)
```

---

## 📁 Generated Artefacts

After a full pipeline run the following files are created automatically:

```
data/
├── raw/
│   └── match_{id}_raw.parquet                    ← Validated raw events
├── processed/
│   ├── match_{id}_spadl.parquet                  ← SPADL actions
│   ├── match_{id}_vaep.parquet                   ← Per-action VAEP scores
│   ├── match_{id}_player_summary.parquet         ← Player VAEP aggregation
│   ├── match_{id}_team_summary.parquet           ← Team VAEP aggregation
│   ├── match_{id}_weaknesses_{team_id}.parquet   ← Zone weakness map
│   ├── match_{id}_pressure_{team_id}.parquet     ← Pressure heatmap
│   ├── match_{id}_player_rankings.parquet        ← Ranked player table
│   └── match_{id}_tactical_report.json          ← Full JSON report
└── features/
    └── match_{id}_features.parquet               ← 105-dim feature matrix

models/
├── xg_model.joblib
├── xt_model.joblib
├── p_scores_model.joblib
├── p_concedes_model.joblib
└── metadata.yaml                                 ← Training dates + metrics

logs/
└── football_ai.log                               ← Rotating file log
```

---

## 📜 References

- Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). [*Actions Speak Louder Than Goals: Valuing Player Actions in Soccer.*](https://dl.acm.org/doi/10.1145/3292500.3330758) KDD 2019.
- Singh, K. (2018). [*Introducing Expected Threat.*](https://karun.in/blog/expected-threat.html)
- StatsBomb. [*StatsBomb Open Data.*](https://github.com/statsbomb/open-data)
- Chen, T., & Guestrin, C. (2016). [*XGBoost: A Scalable Tree Boosting System.*](https://dl.acm.org/doi/10.1145/2939672.2939785) KDD 2016.

---

## 👨‍💻 Author

<div align="center">

### Khalid Morjane

**AI & Data Science Student**

*Passionate about applying machine learning to real-world problems*  
*Specialising in sports analytics, production ML systems, and full-stack AI applications*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/khalidmorjane)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/KHALIDMRJ)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:khalidmorjan37@email.com)

</div>

---

## 📄 License

```
MIT License

Copyright (c) 2025 Khalid Morjane

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f3460,50:16213e,100:1a1a2e&height=120&section=footer" width="100%"/>

**⭐ Star this repository if you found it useful!**

*Built with ❤️ by Khalid Morjane — AI & Data Science Student*

</div>
