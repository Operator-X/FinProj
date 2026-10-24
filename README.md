# Algorithmic Working Capital & Margin Optimization
### A Causal Double-ML Dynamic Pricing & Liquidity Engine on Brazilian E-Commerce Data

This engine turns transaction-level marketplace data into pricing decisions that maximise **Contribution Margin 2 (CM2) minus the cost of capital tied up in inventory**, not revenue.

**What it does:**
- forecasts demand with calibrated uncertainty;
- estimates *causal* price sensitivity;
- reads what customers say in their reviews;
- prices each product for this week, and plans prices across a whole selling season with a reinforcement-learning agent.

**How it stays safe:** every decision is bounded by hard financial guardrails (a margin floor, a volatility band and an aged-stock liquidation trigger).

**How it reports:** every model result is paired with its financial counterpart.

**Data:** the public Olist dataset (about 100k Brazilian orders, 2016–2018) and the Brazilian Central Bank's SELIC policy-rate series.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Architecture: seven phases](#2-architecture-seven-phases)
3. [Data sources](#3-data-sources)
4. [Repository layout](#4-repository-layout)
5. [Phase 0: financial invariants and engineering rules](#5-phase-0-financial-invariants-and-engineering-rules)
6. [Phase 1: data engineering and unit economics](#6-phase-1-data-engineering-and-unit-economics)
7. [Phase 2: demand forecasting with a financial scorecard](#7-phase-2-demand-forecasting-with-a-financial-scorecard)
8. [Phase 3: causal price elasticity (Double ML)](#8-phase-3-causal-price-elasticity-double-ml)
9. [Phase 4: constrained price optimisation](#9-phase-4-constrained-price-optimisation)
10. [Phase 5: AI decision layer](#10-phase-5-ai-decision-layer)
11. [Phase 6: Margem, the product](#11-phase-6-margem-the-product)
12. [Results](#12-results)
13. [Tests](#13-tests)
14. [Design decisions](#14-design-decisions)
15. [Limitations and honest caveats](#15-limitations-and-honest-caveats)
16. [Glossary](#16-glossary)

---

## 1. Quick start

Requirements: macOS or Linux, [`uv`](https://github.com/astral-sh/uv), Python 3.12 (uv installs it if missing), and the Claude Code CLI for Phase 5C.

```bash
cd finproj
uv sync                                   # create .venv with pinned dependencies

# 1) Data (once)
mkdir -p data/raw
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
uv run kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
curl -s 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados?formato=json&dataInicial=01/09/2016&dataFinal=31/10/2018' \
     -o data/raw/selic_432.json

# 2) Phase 5C: LLM review labels (once; cached in data/labels/, resumable)
uv run python -m engine.reviews           # ~35 batched `claude -p --model haiku` calls

# 3) Phases 1-5 batch pipeline (~4 min; the first run also trains the RL agent)
uv run python -m engine.pipeline          # -> artifacts/*.parquet, reports/results.md
uv run python -m engine.rl train eval     # optional: retrain the PPO agent (~4.5 min) and re-evaluate

# 4) Phase 6: the Margem web app (rebuilt automatically at the end of the pipeline)
uv run python -m engine.webexport         # -> web/margem.html, one self-contained file
web/publish.sh                            # publish to https://operator-x.github.io/margem/ (the poster QR code)
uv run python -m engine.api              # optional: same page from the Python reference server (http://127.0.0.1:8600)

# 5) Tests
uv run pytest                             # 40 tests
```

- **Model downloads:** the first pipeline run downloads two Chronos foundation models from Hugging Face (`amazon/chronos-bolt-small`, `amazon/chronos-2`).
- **Without review labels:** the Phase 5C stage is skipped and everything else still runs.
- **Without the SELIC file:** if `selic_432.json` is missing, the engine falls back to a hard-coded table of COPOM rate decisions (`engine/finance.py`).

---

## 2. Architecture: seven phases

```
                 Olist transactions + reviews          BACEN SELIC series
                              │                               │
Phase 0  Guardrails & rules ──┼───────────────────────────────┼──── apply to every phase below
                              ▼                               ▼
Phase 1  Weekly SKU panel, unit economics (CM2), SELIC-linked carrying cost
                              │
            ┌─────────────────┼──────────────────────────┐
            ▼                 ▼                          ▼
Phase 2  Point forecasts   Phase 3  Causal elasticity   Phase 5C  LLM customer voice
         (LightGBM vs      (fixed-effects Double ML)    (review themes → confounders,
          baselines)                │                    heterogeneity tests)
            │                       │                          │
            │   Phase 5B  Probabilistic forecasts               │
            │   (Chronos foundation models vs LightGBM          │
            │    quantiles → working-capital-at-risk)           │
            ▼                       ▼                          │
Phase 4  This week's price: constrained CM2 − carry optimiser ◄─┘
                              │
Phase 5A  Whole-season price path: RL agent (PPO) benchmarked against exact dynamic programming
                              │
Phase 6  Margem: the product (web app with Capi the capybara), full engine in the browser, public at operator-x.github.io/margem
```

| Phase | Question it answers | Module |
|---|---|---|
| 0 | What may the engine never do? | `config.py`, `finance.py` |
| 1 | What does each sale actually earn, and what does stock cost to hold? | `data.py` |
| 2 | How many units will sell next week? | `forecast.py` |
| 3 | How much does price *cause* demand to change? | `elasticity.py` |
| 4 | What is the best price this week? | `optimize.py` |
| 5A | What is the best price *path* to sell a lot of stock over a season? | `rl.py` |
| 5B | How much stock, and how much working capital, does a target service level need? | `foundation.py` |
| 5C | What do customers say, and does it change price sensitivity? | `reviews.py` |
| 6 | How does a seller, or a judge, use all of this in 30 seconds? | `web/`, `webexport.py`, `api.py` |

Phases 1–4 form the **causal pricing core**. Phase 5 adds the **AI decision layer**:
- 5A extends the one-week decision to a whole season;
- 5B turns point forecasts into calibrated distributions, and so into capital requirements;
- 5C adds unstructured text to the causal model.

Phase 6 puts every phase in front of the user as **Margem**, a pricing copilot with its own identity (Capi the capybara, notes in the margin), and a small server that runs the real engine behind it on demand.

---

## 3. Data sources

| Source | What it gives | Used for |
|---|---|---|
| **Olist Brazilian E-Commerce** (Kaggle `olistbr/brazilian-ecommerce`) | Orders, order items (price, freight), products (category), reviews (stars + Portuguese text), sellers | Demand, prices, freight, delivery lead times, review signals |
| **BACEN SGS series 432** (SELIC target rate, daily) | Policy rate, 14.25% (Sep 2016) → 6.50% (Mar 2018) | Cost of capital for inventory carrying cost and RL discounting |
| **Sector gross-margin benchmarks** (Damodaran LatAm retail; Mercado Libre filings) | Typical COGS share of price by sector | Synthesising COGS, which Olist does not record |
| **Chronos foundation models** (Amazon, Hugging Face) | Pretrained time-series transformers | Zero-shot probabilistic forecasts (Phase 5B) |
| **Claude Haiku** (via the Claude Code CLI) | Language understanding | Coding review text into themes (Phase 5C) |

SELIC changes picked up from the API (effective dates):

| Date | Rate | Date | Rate |
|---|---|---|---|
| 01 Sep 2016 | 14.25% | 27 Jul 2017 | 9.25% |
| 20 Oct 2016 | 14.00% | 08 Sep 2017 | 8.25% |
| 01 Dec 2016 | 13.75% | 26 Oct 2017 | 7.50% |
| 12 Jan 2017 | 13.00% | 07 Dec 2017 | 7.00% |
| 23 Feb 2017 | 12.25% | 08 Feb 2018 | 6.75% |
| 13 Apr 2017 | 11.25% | 22 Mar 2018 | 6.50% |
| 01 Jun 2017 | 10.25% | | |

`data/raw/` and `artifacts/` are git-ignored: the raw data is redistributable only through Kaggle, and the artifacts are regenerated by the pipeline.

---

## 4. Repository layout

```
finproj/
├── engine/                  # all modelling logic, importable and tested
│   ├── config.py            # Phase 0: seeds, window, categories, COGS shares, guardrails, artifact paths
│   ├── finance.py           # Phase 0/1: CM2, holding cost, DSI, safety stock, Lerner breakeven, SELIC loader
│   ├── data.py              # Phase 1: transactions -> weekly SKU panel
│   ├── forecast.py          # Phase 2: 3 forecasters, walk-forward CV, financial scorecard
│   ├── elasticity.py        # Phase 3: fixed-effects Double ML, SKU shrinkage, moderation tests
│   ├── optimize.py          # Phase 4: constrained price optimiser + uncertainty band
│   ├── rl.py                # Phase 5A: sell-through simulator, exact DP, PPO agent, Bayesian elasticity learning
│   ├── foundation.py        # Phase 5B: Chronos foundation models vs LightGBM quantiles, newsvendor backtest
│   ├── reviews.py           # Phase 5C: LLM review coding (Claude) -> leakage-safe customer-voice features
│   ├── webexport.py         # Phase 6: builds the app + website from the artifacts (incl. PPO weights, scenario draws, forecasts, reviews)
│   ├── api.py               # Phase 6: optional Python reference server (FastAPI): same page + JSON engine endpoints
│   └── pipeline.py          # batch orchestration of Phases 1-6 + reports/results.md writer
├── brand/capi/              # Capi artwork exported from the app code: 5 poses + logo, SVG (font embedded) + transparent PNG,
│                            #   in two inks: for-dark-backgrounds/ and for-light-backgrounds/ (posters, slides)
├── brand/qr/                # Poster QR codes for operator-x.github.io/margem (Capi in the centre, error correction H, decode-verified)
├── brand/make_qr.py         # regenerates the QR codes;  brand/render_capi.js re-exports the Capi artwork
├── web/                     # Phase 6: the Margem app
│   ├── template.html        #   design, UI logic and the in-browser ENGINE block (data slot: __DATA__)
│   ├── data.json            #   (generated) everything the page reads, ~0.5 MB
│   ├── margem.html          #   (generated) self-contained app
│   ├── site/                #   (generated) the public website: index.html, Capi favicon, .nojekyll
│   └── publish.sh           #   rebuilds and pushes web/site/ to GitHub Pages (repo Operator-X/margem)
├── tests/test_engine.py     # Phases 0-4 and 6: guardrails, leakage, Lerner test, server endpoints
├── tests/test_ai.py         # Phase 5-6: RL guardrails, DP vs myopic, write-down, review leakage, lab-solver parity, web build
├── tests/test_browser_engine.py  # Phase 6: in-browser optimiser and RL seasons vs the Python engine (run in Node)
├── data/labels/             # Phase 5C: cached LLM review labels (small, safe to commit)
├── data/raw/                # (downloaded) Olist CSVs + SELIC JSON
├── artifacts/               # (generated) parquet tables and the trained policy consumed by the app
├── reports/results.md       # (generated) every number in this README, regenerated each run
└── pyproject.toml / uv.lock # pinned environment (Python 3.12)
```

**Artifacts produced by `engine.pipeline`:**

| File | Phase | Grain | Content |
|---|---|---|---|
| `panel.parquet` | 1 | SKU × week | Modelling panel with features, unit economics and customer-voice signals |
| `forecasts.parquet`, `forecast_metrics.parquet` | 2 | SKU × week / model | Point forecasts per walk-forward origin; statistical and financial scorecard |
| `elasticity_category.parquet`, `elasticity_sku.parquet` | 3 | category / SKU | DML elasticities with CIs and naive comparisons; shrunk, de-attenuated SKU elasticities |
| `elasticity_robust.parquet` | 3 | category | Robustness check: DML on sale weeks only (outcome ln Q) |
| `sku_state.parquet`, `recommendations.parquet` | 4 | live SKU / SKU × scenario × mode | Pricing state per SKU; precomputed portfolio recommendations |
| `ppo_pricing.zip`, `rl_eval.parquet` | 5A | — / SKU × season × policy | Trained RL policy; out-of-sample policy comparison |
| `prob_forecasts.parquet`, `prob_metrics.parquet` | 5B | SKU × week × model / model | Quantile forecasts; probabilistic and newsvendor scorecard |
| `chronos_quantiles.parquet` | 5B | SKU × week × model | Raw Chronos output from its separate PyTorch process |
| `review_labels.parquet`, `elasticity_voice.parquet`, `moderation.parquet` | 5C | review / category / moderator | LLM labels; DML with review confounders; heterogeneity tests |

---

## 5. Phase 0: financial invariants and engineering rules

All parameters live in `engine/config.py` (`FinanceParams`), so the pipeline, the Margem app and the report can never disagree.

### Financial invariants (enforced in code and checked by tests)

| Rule | Implementation |
|---|---|
| Never optimise top-line revenue | Objective is CM2 − carrying cost in Phase 4 (`optimize.profit`), and CM2 − carry − write-down in Phase 5A |
| Margin floor: P ≥ 1.15 × COGS | `bounds()` in Phase 4 and `Lot.floor_k()` in Phase 5A; both tested |
| Liquidation floor: P ≥ 1.00 × COGS when weeks-of-supply > 8 | `aging_trigger_weeks = 8`, `floor_liquidation = 1.00` |
| Volatility band: 0.85 × P_last ≤ P ≤ 1.15 × P_last | `band = 0.15`; in Phase 5A, at most two lattice steps of √1.15 per week |
| Holding cost tied to macro: H_annual = SELIC_t + 6% | `holding_rate_annual()`; weekly SELIC merged into every row |
| Elasticity only from orthogonalised estimators | Double ML in Phase 3; naive OLS is shown only as a contrast |

**Guardrail priority.** If the 1.15 × COGS floor is above the +15% band ceiling (a SKU currently priced below its floor), the floor wins and the SKU is flagged `band_overridden`. Loss prevention ranks above price smoothness.

### Engineering rules

- **Type hints and docstrings** on every function, stating input/output shapes and finance definitions.
- **No leakage.**
  - All demand features are lagged (`shift(1)`), and review signals become visible only the week after they are written.
  - Validation is walk-forward only. There is no random shuffling across time anywhere; even the DML cross-fitting folds are contiguous time blocks.
- **Reproducibility.**
  - `SEED = 42` for numpy, LightGBM (`deterministic=True`), PyTorch/PPO, the optimiser's uncertainty draws and the RL evaluation scenarios.
  - LLM labels are cached, so results never depend on re-querying a model.
- **Latency.** Margem never refits a model. A live pricing request re-solves one SKU's 1-D price problem and its 200 uncertainty draws on the server in tens of milliseconds (asserted < 200 ms by a test).
- **Process isolation.** PyTorch (Chronos, PPO) and LightGBM ship different OpenMP runtimes, and on macOS combining them in one process segfaults or deadlocks. So:
  - the pipeline runs the PyTorch stages as separate processes;
  - the Margem server never imports LightGBM;
  - the app test runs in its own subprocess.

---

## 6. Phase 1: data engineering and unit economics

`engine/data.py → build_panel()`

### 6.1 Scope

| Choice | Value | Why |
|---|---|---|
| Order status | `delivered` only | Cancelled or unavailable orders are not realised demand |
| Window | weeks of 2017-01-02 → 2018-08-20 | 2016 is the platform's ramp-up (a few hundred orders); the last weeks of Aug 2018 are truncated |
| Week definition | Monday-start (`W-SUN` periods), keyed on purchase timestamp | Demand is dated when the customer decides, not when goods ship |
| Categories | `relogios_presentes` (watches_gifts), `cama_mesa_banho` (bed_bath_table), `telefonia` (telephony), `informatica_acessorios` (computers_accessories) | High-liquidity categories spanning three margin regimes |
| SKU eligibility | ≥ 10 selling weeks and ≥ 30 units | Elasticity needs repeated price observations within a SKU |

Result: **118 SKUs, 6,104 SKU-weeks, 8,315 units, with 52% of weeks having zero sales.**

### 6.2 Panel construction

1. **Item → (SKU, week) aggregation:** `units_sold` (item count), `avg_price`, `avg_freight`, `seller_count`, review sums, and mean `delivery_days` (purchase to customer delivery).
2. **Active listing span.** Each SKU runs from its first sale week to its last sale week + 4 weeks, capped at the window end. Genuine post-sale zero weeks are kept, but a delisted product is not padded with zeros forever.
3. **Zero-sale weeks:**
   - `units_sold = 0`.
   - Price, freight, seller count and delivery days are **forward-filled only**. The posted price persists, and back-filling would leak a future price into the past.
   - `avg_review_score` is the **cumulative mean of past reviews**, i.e. what a shopper could see at week t.
4. **Lagged demand features:** `lag_1`, `lag_2`, `lag_4`, and a 4-week rolling mean and std computed on `shift(1)`.
5. **Market context:**
   - `cat_velocity_lag1`: last week's total units for the category, using all SKUs, not only eligible ones.
   - `cat_med_price` / `log_cat_price`: weekly median price in the category, a competitor price index.
   - `price_ratio_cat`: own price ÷ category median.
6. **Calendar:**
   - Brazilian retail events: Black Friday 2017, the Christmas run-up, Mother's Day, Dia dos Namorados (12 Jun), Father's Day, Consumer Day (15 Mar).
   - A **May-2018 truckers' strike** flag, a national logistics shock that is not a demand effect.
   - Two harmonics of annual Fourier seasonality, plus a time index.
7. **Macro:** weekly mean SELIC merged by week.
8. **Customer voice:** the Phase 5C review signals are merged in by (SKU, week) when labels are available.

### 6.3 Unit economics (synthesised COGS)

Olist records the selling price P and freight F but not the seller's acquisition cost. COGS is therefore reconstructed from sector gross-margin benchmarks, applied to each SKU's **baseline price** (the median transacted price):

| Category | Benchmark gross margin | COGS share of baseline price |
|---|---|---|
| telephony, computers_accessories | ~18–22% | 0.80 |
| bed_bath_table | ~40–45% | 0.58 |
| watches_gifts | ~50–60% | 0.45 |

COGS is a **fixed per-SKU cost**, so it does not move when the engine changes price.

```
CM2_unit   = P · (1 − fee) − COGS − s · F            fee = 10%, s = freight share borne by seller (default 1.0)
H_annual   = SELIC_t + 6.0%                          storage / obsolescence spread
h_t        = H_annual / 52 · COGS                    weekly carrying cost of one unit
DSI (days) = inventory / weekly units · 7
```

**Finding (unit economics):**

| Category | Median price | Median freight | Median CM2 / unit | Share of SKU-weeks with CM2 < 0 |
|---|---|---|---|---|
| bed_bath_table | R$ 79.90 | R$ 16.15 | R$ 7.74 | 21% |
| computers_accessories | R$ 81.99 | R$ 16.02 | **−R$ 10.27** | 86% |
| telephony | R$ 29.99 | R$ 15.10 | **−R$ 12.01** | 92% |
| watches_gifts | R$ 119.00 | R$ 15.90 | R$ 34.42 | 7% |

Low-ticket electronics are **structurally CM2-negative** when the seller absorbs freight. A R$30 phone accessory carries about R$15 of freight against a 20% gross margin. Margem's "shipping paid by seller" slider (Scenario lab, and Simulate on any product) shows how this flips when the customer pays shipping.

---

## 7. Phase 2: demand forecasting with a financial scorecard

`engine/forecast.py`

### 7.1 Validation protocol (walk-forward, no leakage)

- **Hold-out:** 2018-03-05 → 2018-08-20.
- **Rolling origins:** every 4 weeks (7 origins). At each origin every model is **refit on weeks strictly before the origin**, then issues **one-step-ahead** forecasts for the next 4 weeks. One step ahead matches the weekly repricing cadence.
- **Scoring:** only SKUs already listed before the origin are scored (110 SKUs, 2,374 SKU-weeks). A SKU's first week always contains a sale by construction, and scoring it would reward knowing the launch date.
- **Shared protocol:** Phase 5B uses this exact protocol, so point and probabilistic forecasts are directly comparable.

### 7.2 A subtle leakage trap, and the fix

In a zero-sale week the price is unobserved and forward-filled. So if this week's price differs from last week's, a sale must have happened, and a model given the contemporaneous price would learn "price changed ⇒ sale". The forecaster therefore only sees:

- `posted_price`: the last observed price as of t−1.
- Lagged seller count, lagged delivery days, and the lagged category price.

This is enforced by `add_forecast_features()` and locked in by `test_forecast_features_do_not_leak_current_sales`.

### 7.3 Models

| Model | Specification |
|---|---|
| **Seasonal-Naive** | Q̂_t = Q_{t−52}; falls back to the trailing 4-week mean when the SKU was not listed a year earlier (most launched in 2017) |
| **SARIMAX** | Per SKU, ARIMA(1,0,1) + constant on ln(1+Q). Exogenous: posted log-price, holiday flag, truckers-strike flag, SELIC, annual Fourier terms (series are too short for a seasonal AR term). Parameters are frozen at the origin and filtered one step ahead. A plausibility guard rejects any forecast above 2× the SKU's historical peak and falls back to the 4-week mean |
| **LightGBM-Tweedie** (challenger) | One global model across SKUs, Tweedie loss (variance power 1.2) for zero-inflated, right-skewed counts. 25 features: lags, rolling stats, listing age, posted price, price ratio, category velocity, sellers, delivery days, reviews, SELIC, seasonality, events, category |

### 7.4 Financial evaluation layer

Each statistical metric is paired with a balance-sheet or P&L metric:

| Metric | Formula | Financial meaning |
|---|---|---|
| Forecast bias ratio | Σ(Q̂ − Q) / ΣQ | Systematic over- or under-ordering |
| Stockout capital loss | Σ max(0, Q − Q̂) · max(CM2, 0) | Margin forgone on unmet demand (a negative-CM2 unit is not a loss) |
| Excess holding burden | Σ max(0, Q̂ − Q) · h_t | Carrying cost of over-stocked units |
| Safety-stock capital | Σ_SKU 1.645 · σ(error) · COGS | Inventory BRL needed to absorb forecast error at a 95% cycle-service level, 1-week lead time |

---

## 8. Phase 3: causal price elasticity (Double ML)

`engine/elasticity.py`

### 8.1 Why a plain regression is wrong here

Prices are not set at random. Sellers move prices in response to demand: they promote into Black Friday and drift prices as the platform grows. Regressing demand on price therefore picks up timing, not causation.

### 8.2 Model

A partially linear model with SKU fixed effects, for SKU i in week t:

```
ln(Q_it + 1) = θ · ln(P_it) + g(X_it) + a_i + u_it
ln(P_it)     =                m(X_it) + b_i + v_it
```

**Confounders X:**
- Demand history: lags 1/2/4, rolling mean and std.
- Listing age and time trend.
- Category velocity and the category median price (competitor index).
- Lagged delivery lead time, cumulative review score, lagged seller count.
- SELIC, Fourier seasonality, the seven event dummies, and category.
- In the Phase 5C specification, also the LLM-coded customer-voice signals.

### 8.3 Estimator (Frisch–Waugh–Lovell with ML nuisances)

1. **Within-transform** Y and T by SKU. This removes a_i and b_i (fixed brand or quality level).
2. **Cross-fitted nuisances.** LightGBM models for E[Y|X] and E[T|X], using 5 folds made of **contiguous calendar-week blocks**. Each residual comes from a model that never saw that time block.
3. **Residual-on-residual slope:** θ = Σ T̃Ỹ / Σ T̃², with **SKU-clustered sandwich standard errors**.

Estimates are produced for the whole pool and per category (from the same cross-fitted residuals), alongside two naive estimators:
- **pooled OLS**, which mixes price levels across SKUs;
- **within-SKU OLS without confounder adjustment**.

### 8.4 From category to SKU elasticities

1. **Per-SKU slope** on that SKU's residuals, with its standard error.
2. **Empirical-Bayes shrinkage** toward the category estimate. The between-SKU variance τ² comes from **DerSimonian–Laird** random-effects meta-analysis:
   `θ_i^EB = (τ²·θ_i + se_i²·θ_cat) / (τ² + se_i²)`.
   SKUs with little price variation inherit the category value; well-identified SKUs keep their own.
3. **De-attenuation of log1p.** θ is d ln(Q+1)/d ln P. The demand elasticity is d ln Q/d ln P = θ·(Q+1)/Q, evaluated at the SKU's mean weekly units. At 1–2 units a week, θ alone understates sensitivity by roughly half.
4. **Uncertainty:** the SKU SE combines the posterior SE with the category SE, so a SKU is never more precise than the category it borrows from. This posterior is also the RL agent's prior belief in Phase 5A.
5. **Sign restriction.** An upward-sloping demand estimate (20 of 118 SKUs) is treated as noise, and the SKU falls back to its category estimate.
6. **Winsorising** at ε = −5 (4 SKUs), beyond what weekly price variation can identify.
7. **Regime label:** |ε| < 1 → *Inelastic: margin expansion*; |ε| > 1 → *Elastic: liquidity acceleration*. The split is 73 / 45 SKUs.

### 8.5 Robustness and heterogeneity

- **Intensive margin.** The same DML on **sale weeks only**, with outcome ln Q (where price is directly observed), gives estimates in the same direction and of similar magnitude to the de-attenuated main estimates (§12.2).
- **Moderation.** `moderation()` re-uses the cross-fitted residuals to test whether θ varies with a SKU-week characteristic z: Ỹ = θ₀T̃ + θ₁T̃·z_std, with SKU-clustered SEs. Phase 5C applies it to the customer-voice signals.

---

## 9. Phase 4: constrained price optimisation

`engine/optimize.py`

### 9.1 Objective

For one SKU and the coming week:

```
Q(P)  = Q_base · (P / P_last)^ε                         constant-elasticity demand through the status-quo point
Π(P)  = min(Q(P), I) · [P(1 − fee) − COGS − s·F]  −  HoldingCost(P)
```

- `Q_base` is a **true one-step-ahead LightGBM forecast** for the decision week (2018-08-20), at the status-quo price, from a model trained only on earlier weeks.
- Sales are capped by the inventory I on hand.

### 9.2 Two strategy modes

| Mode | Holding cost | Interpretation |
|---|---|---|
| **Margin Maximization (EBITDA)** | h · max(0, I − Q(P)) | One week of carry on leftover stock |
| **Aggressive Cash Recovery (Liquidity)** | h · L · min(L/Q(P), 52) / 2, with L = leftover units | Carry on leftover stock **until it is sold through** at the rate implied by the chosen price (linear depletion, capped at 52 weeks). This prices the remaining cash-conversion cycle, so slower sell-through costs more |

### 9.3 Constraints

```
P ≥ 1.15 · COGS                (1.00 · COGS if I / Q_base > 8 weeks: aged-stock liquidation)
0.85 · P_last ≤ P ≤ 1.15 · P_last
floor > band ceiling  ⇒  P = floor   (flagged)
```

### 9.4 Solver

- A 401-point grid over the feasible interval finds the global maximum. Π has a kink where the inventory cap binds, so a pure gradient method could stall there.
- A bounded Brent search (`scipy.optimize.minimize_scalar`) then refines around the best grid point.

### 9.5 Uncertainty band on P*

- Draw 200 elasticities ε ~ N(ε̂, se) (seeded, clipped to [−5, −0.05]) and re-solve for each draw.
- The 5th–95th percentile of the resulting P* values is reported.
- This answers "how far would the recommendation move if the elasticity were mis-estimated?"

### 9.6 Outputs per decision

- P* with its 90% band, and the feasible band.
- Q at the status-quo price and at P*.
- Total CM2, holding cost and net profit, each before and after.
- DSI before and after.
- Working capital freed: (Q* − Q_base) · COGS, i.e. inventory converted to cash this week.
- Flags: aged, floor binding, band overridden.

### 9.7 The Lerner discount test

With Π = Q(P)·(aP − c), where a = 1 − fee and c = COGS + s·F, and constant ε:

```
dΠ/dP < 0   ⇔   |ε| > aP / (aP − c) = 1 / m,     m = CM2 / net revenue
```

A price cut adds CM2 **only if demand is more elastic than 1/m**. If CM2 ≤ 0, no elasticity makes a cut profitable. `finance.breakeven_elasticity()` computes this threshold for every SKU; Margem shows it in every "why this price" note, and a test checks it against the numerical gradient.

### 9.8 Scope of Phase 4

Phase 4 is deliberately a **one-week** decision, because that is the weekly repricing cadence. It sees the future only through the carrying-cost term. Questions that span a season, such as "price high now and risk a write-down later, or clear stock early?", are the job of Phase 5A, which uses Phase 4 as one of its benchmarks.

---

## 10. Phase 5: AI decision layer

Phase 5 adds three AI components on top of the causal core. Each is judged with the same financial lens as Phases 2–4.

| Component | Builds on | Adds |
|---|---|---|
| **5A** Reinforcement-learning pricing agent | Phase 3 elasticity posterior, Phase 4 guardrails | Multi-week price paths for a lot of stock under elasticity uncertainty |
| **5B** GenAI time-series foundation models | Phase 2 protocol | Calibrated demand distributions → inventory capital and working-capital-at-risk |
| **5C** LLM-coded customer voice | Phase 1 panel, Phase 3 DML | Review text as causal confounders and moderators |

### 10.1 Phase 5A: reinforcement-learning sell-through pricing (`engine/rl.py`)

**Decision problem.** A seller holds a lot of stock and has a 26-week season to sell it. Each week they choose a price, trading margin now against the risk of a write-down later.

- **Season:** a fixed lot, a 26-week season and no replenishment.
- **Leftover stock:** anything unsold at the end goes to an outlet at 50% of cost. That is booked as a write-down of 50% of COGS, consistent with CM2, which already charges COGS on each unit sold.
- **Price moves:** each week the price moves on a lattice P0·√1.15^k, at most two steps (±15%) per week.
- **Price limits:** the Phase 0 margin floor always applies (relaxed to break-even once stock is aged). Prices are capped at +42%, the edge of the range in which elasticity was identified.
- **Demand:** negative binomial with mean q_base·(P/P0)^ε and dispersion 3, matching the lumpy weekly counts in the data.
- **Reward:** CM2 on units sold − SELIC-linked carrying cost − the end-of-season write-down, discounted at the weekly SELIC.
- **Learning while earning:** the true ε is drawn from the Phase 3 posterior and is *not* shown to the agent. The environment keeps an exact Bayesian posterior over ε on a grid, updated each week from observed sales, and correctly handles weeks where the agent sold out, because then demand was higher than the sales it saw. The agent sees the posterior mean and std, so trying a price to learn from the response has value it can discover.

**Agent.**
- **Algorithm and network:** one PPO policy (stable-baselines3, a 2×128 MLP) prices *every* SKU in the catalogue.
- **What it sees:** stock, time left, current price level, its elasticity belief, margin structure, cost of capital and recent sales.
- **Training:** 2M steps, about 4.5 minutes on a Mac CPU.
- **Training variety:** SELIC is drawn from 6.5–15% and stock from 4–52 weeks of supply. SKUs are sampled **evenly across elasticity regimes**: highly elastic SKUs are rare in the catalogue but are where pricing decisions matter most, so uniform sampling would under-train exactly the hardest cases.

**Benchmarks.** Every policy is run on identical demand scenarios (common random numbers): 72 SKUs × 30 seasons, SELIC 6.5%.

| Policy | What it is |
|---|---|
| status quo | hold today's price all season |
| myopic (Phase 4) | the Phase 4 optimiser, re-solved each week |
| CE-DP | exact dynamic programme that treats the estimated ε as true |
| **PPO agent** | the RL policy |
| oracle DP | exact DP that *knows* the true ε, an upper bound in expectation |

The exact DP is tractable here because the state is small (stock × week × price level). It certifies how close the learned policy gets to optimal. The PPO network is the scalable form of the same decision rule: one model for the whole catalogue, with room for richer state.

Results: §12.4.

### 10.2 Phase 5B: GenAI foundation models vs LightGBM, probabilistic (`engine/foundation.py`)

**Question.** Does a pretrained "GPT for time series" beat a model trained on our own data, and what does the answer mean for working capital?

**Models**, all producing one-week-ahead q10/q50/q90 quantiles on the Phase 2 walk-forward protocol:
- **Chronos-Bolt-small** (zero-shot, univariate);
- **Chronos-2** (zero-shot, with past/future covariates: posted log-price, holiday flags, truckers' strike, SELIC, category velocity);
- **LightGBM quantile regression** (pinball loss), refit at each origin.

A cross-learning variant of Chronos-2 (series batched by forecast week to stay leakage-safe) was also scored.

**Financial test (newsvendor backtest).** Each week, stock up to ⌈q90⌉:
- realised stockout margin loss;
- excess carrying cost;
- fill rate;
- average inventory capital;
- **Working-Capital-at-Risk (90%)** = Σ(q90 − q50)·COGS, the capital that buys the last 40 points of service level. This is only meaningful if q90 is calibrated, so coverage is reported too.

Results: §12.5.

### 10.3 Phase 5C: LLM-coded customer voice in the causal model (`engine/reviews.py`)

**Hypothesis.** Customer perception (quality, value, delivery) drives demand, and sellers price in response to it. Perception would then be a confounder in Phase 3, and possibly a moderator of price sensitivity. About 3.4k written Portuguese reviews of the panel's SKUs carry this signal.

**Coding.**
- **Model and taxonomy:** Claude (Haiku, via `claude -p`) codes each review into 9 themes plus a sentiment score in −1/0/1. The themes are value_good, price_high, quality_good, quality_bad, delivery_fast, delivery_late, wrong_or_missing, service_bad and recommend.
- **Batching and caching:** 3,351 unique reviews in 34 batched calls. Labels are cached in `data/labels/review_labels.jsonl`, so results are reproducible without re-querying.
- **Validation:** the LLM never sees the star ratings, so agreement with them is an independent accuracy check.

**Leakage-safe features.**
- At week t, a SKU "knows" only reviews created before t's Monday; a review written during week w is visible from w+1.
- Shares are smoothed toward the catalogue mean with 5 pseudo-reviews, so a SKU with two reviews does not get a 0% or 100% signal.
- Derived signal: `value_perception = value_good − price_high`.

**Uses.**
1. **Confounders:** the Phase 3 DML is re-estimated with the voice signals added to X.
2. **Moderators:** `moderation()` tests whether price sensitivity varies with value perception, quality complaints, late-delivery complaints and sentiment. Four tests, so the Bonferroni 5% threshold is |z| > 2.50.

Results: §12.6.

---

## 11. Phase 6: Margem, the product

Phase 6 turns the engine into something a seller would open every Monday, and that a judge can understand in 30 seconds. **Where it runs:** **https://operator-x.github.io/margem/** (the poster QR code), or `web/site/index.html` offline. Every feature works there with no server, because the interactive parts of the engine are ported to JavaScript and parity-tested against Python (§11.6). Optionally, `uv run python -m engine.api` serves the same page from a FastAPI server, and then the Scenario lab also confirms its numbers with the Python optimiser.

**Design rule: simple first, more on request.** Every screen opens with one sentence, one number and Capi. Detail appears only when asked for: cards expand into panels; a product page opens three tools (Simulate, Forecast, What buyers say); the Season planner offers "Race the AI on any product".

### 11.1 Identity

- **Name.** *Margem* is Portuguese for "margin", a double meaning that drives the whole design. Profit margin is what the engine protects, and the analysis is written as handwritten *notes in the margin*. Tagline: **"Notes in the margin. Money in the margin."**
- **Mascot: Capi the capybara**, Brazil's best-known animal, hand-drawn with rough.js and redrawn stroke by stroke each time a page opens. Capi has one pose per page:

  | Page | Pose | Says |
  |---|---|---|
  | Overview | holding a "+15%" price tag | *"psst… they won't notice"* |
  | Products | asleep on a box of unsold stock | *"napping on unsold stock"* |
  | Season planner | with a calendar | *"I plan the whole season"* |
  | Evidence | in a detective hat with a magnifying glass | *"Black Friday fooled them!"* |
  | Scenario lab | soaking in a hot tub of rates | *"rates up? I'll just soak here"* |

- **Look.** Three ingredients:
  - **Linear-style dark minimalism** as the base;
  - **Stripe-style live cards** that animate in and expand into full detail panels when clicked;
  - **Notion-style hand-drawn ink** (Caveat handwriting, sketched underlines, arrows and margin notes) on a dot-grid notebook background with a red margin rule.
- **Voice.** Plain English with a little Portuguese ("**Bom dia.**"). Every headline is one sentence with one hand-underlined number.
- **Sketches toggle.** A sidebar switch hides the doodles and notes for a clean, formal view.

### 11.2 Pages

Every page follows the same formula: a one-sentence headline with an underlined number, Capi, then animated cards that open into detail panels.

| Page | Phase | Headline (live data) | Content |
|---|---|---|---|
| **Overview** | 4 | *"You could earn R$618 more this week."* | Four 2-up live cards (weekly CM2 with a "next week" projection, cash in stock, forecast error, RL uplift); "3 things to do this week" with Approve flows |
| **Products** | 1, 4 | *"R$7,362 is stuck in products that lose money."* | Capital-traps and cash-machines cards; the capital map (margin × weekly sales × cash), annotated by hand |
| **Season planner** | 5A | *"Planning ahead turns a −R$204 season into R$62 profit."* | **Season replay** with a product picker: it opens on one featured lot, and any of the 72 products can be chosen and re-simulated in the browser in under 0.1 s. Each replay runs 26 weeks, comparing holding the price, the weekly optimiser and the AI planner on identical demand, with a live scoreboard and a closing verdict; out-of-sample policy chart and table |
| **Evidence** | 2, 3, 5B | *"A naive model thinks discounts work 2.9× better than they do."* | Naive vs Double-ML bars by category; forecast-error and GenAI-capital cards |
| **Scenario lab** | 4, 0 | *"At SELIC 15.00% with 20 weeks of stock, the engine cuts 13 prices…"* | Rate presets (Brazil 2016: 14.25%, 2018: 6.50%, 2025: 15.00%); sliders for stock, platform fee and seller-paid shipping; goal toggle (this week's profit or cash recovery). All 72 products re-price live in the browser, using the Phase 4 objective and the Phase 0 guardrails |

**Throughout:** ⌘K search jumps to any product; each product page shows the recommended price, 26 weeks of sales and a plain-English "why this price".

**Product names:** Olist has no product names, so they are generated from the category and weight, e.g. *"Bed & bath set · 850 g #704e"*.

### 11.3 Motion

- **Cards** count up and draw their charts when they first scroll into view; detail panels open from the card that was clicked.
- **Scroll-driven reveals:** sections rise in, and page headers drift back as you scroll.
- **Capi and the hand-drawn notes** sketch themselves in.

Motion follows `prefers-reduced-motion`, and every page is complete without it: content is never hidden waiting for an animation.

### 11.4 Build and sharing

- **Build:** `engine/webexport.py` reads the Phase 1–5 artifacts, generates the product names, and computes the season replay in a PyTorch child process (process-isolation rule). It writes `web/data.json` (about 0.5 MB) and injects it into `web/template.html`, producing **one self-contained file** that works offline on a laptop.
- **Representative replay:** the featured season replay uses the demand scenario whose AI-vs-hold gap is closest to the mean over 60 scenarios, so the demo is representative rather than lucky.
- **Hosting:** `web/site/` is the standalone website (full HTML document, Capi favicon, link-preview tags). `web/publish.sh` copies only those files into a separate public repo (`Operator-X/margem`) served by GitHub Pages at **https://operator-x.github.io/margem/**. No code or data beyond the page is published.
- **Poster QR codes:** `brand/qr/` holds light and dark versions pointing at that URL (Capi in the centre, error correction H, 4-module quiet zone), verified to decode at sizes from 15% to 100%. The URL never changes, so republishing never invalidates them.

### 11.5 The optional Margem server (`engine/api.py`)

Not needed to use Margem; it is the Python reference that serves the same page and exposes the engine as JSON (useful for demos on the laptop and as a second implementation):

| Endpoint | Phase | Used by | Typical time |
|### 11.6 The in-browser engine, and how we know it is faithful

The interactive features run in the visitor's browser (the `ENGINE` block of `web/template.html`), so the public site needs no server, works offline, and never cold-starts:

| Browser feature | Ported from | How it is made identical |
|---|---|---|
| Simulate, Scenario lab | `engine/optimize.py` (bounds, profit, 401-point grid + local refinement, uncertainty band) | The 200 elasticity draws are numpy's seeded draws, shipped with the page |
| Race the AI | `engine/rl.py` (lattice, guardrails, demand, Bayesian elasticity belief, write-down, exact DP) | Negative-binomial pmf/survival/quantile in closed form for dispersion 3; each scenario's uniform draws shipped with the page |
| The AI agent | The trained PPO network (12→128→128→5, tanh) | Weights exported from the trained model; observations rounded to float32 as in PyTorch |
| Forecast, What buyers say | `prob_forecasts.parquet`, `review_labels.parquet` | Exported per product into the page |

**Parity tests** (`tests/test_browser_engine.py`, `tests/test_ai.py`) run the browser code in Node and the Python engine on identical inputs:
- the optimiser agrees on price, profit, carrying cost, flags and the uncertainty band to within R$0.01–0.02;
- whole RL seasons (5 strategies × several products, stock levels and demand surprises, including the neural agent) produce **identical week-by-week stock paths** and profits within R$0.10;
- the Scenario lab matches across 4 scenarios × 72 products.

Page weight: about 0.5 MB of data (half of it the network's weights).

---|---|---|---|
| `POST /api/price` | 4 | Product → **Simulate**: stock, cost of capital, goal, fee and seller-paid shipping sliders; best price, profit curve with 90% band, P&L bridge, "why this price" | ~50 ms |
| `POST /api/season` | 5A | Season planner → **Race the AI on any product**: hold / weekly optimiser / exact planner / PPO agent / oracle on one shown season plus 40-scenario averages, replayed with the same animation | ~2 s |
| `GET /api/forecast/{sku}` | 5B | Product → **Forecast**: q10–q90 range vs actual units for LightGBM and both Chronos models | instant |
| `GET /api/reviews/{sku}` | 5C | Product → **What buyers say**: latest reviews with their LLM theme tags | instant |
| `POST /api/portfolio` | 4 | **Scenario lab**: re-prices all 72 products under the chosen SELIC, stock, fee, shipping and goal | ~70 ms |

Interactive API docs are at `/api/docs`. The server loads PyTorch for the PPO agent, so under the process-isolation rule it never imports LightGBM. Static extras that need no server are built into the page:
- **Evidence → Show the statistics:** the Phase 3 table (Double ML with 95% CIs, naive within-SKU and pooled OLS, the sale-weeks-only robustness check, observations and SKUs), plus CI whiskers on the chart.
- **Evidence → forecast card:** the full Phase 2 and 5B scorecard (RMSE, MAE, WAPE, bias, stockout loss, excess holding, safety-stock capital; quantile loss, 80% and 90% calibration, fill rate, stock capital, working capital at risk).
- **Evidence → Customer voice:** themes by category, the with/without-review elasticities per category with CIs, and the moderation table with its Bonferroni verdict.
- **Products → See all 72 products** (a sortable table) and **Batch scenarios** (the pipeline's 4 vs 12 weeks × goal totals).
- **Product page:** the elasticity regime, its 95% range, and a flag when the product's own estimate had the wrong sign and fell back to its category.
- **Race the AI** (live): a five-strategy results table and the value of knowing the true elasticity (oracle minus the best realistic strategy).

**Scenario lab engine.** The lab runs the in-browser port of `engine/optimize.py` (same bounds, profit, 401-point grid and local refinement) and labels itself that way on screen; when the page is served by the optional Python server it also confirms each scenario with `/api/portfolio`. A parity test runs that port in Node against Python on 4 scenarios × 72 products: every price agrees within R$0.01.

---|---|---|
| Pricing simulator | 4 | KPI cards, causal ε and Lerner breakeven, guardrail flags, profit and demand curves with 90% bands, P&L waterfall |
| RL agent | 5A | Expected NPV per policy over 40 shared scenarios, illustrative price and stock paths, shortfall-to-oracle chart |
| GenAI forecast | 5B | Scorecards, q10–q90 fan chart per SKU, the newsvendor table |
| Customer voice | 5C | Theme prevalence, DML with vs without review confounders, moderation, the latest tagged reviews |
| Portfolio | 4 | Recommendations for all 72 live SKUs under 2 scenarios × 2 modes |
| Model evidence | 2–3 | Naive vs Double-ML elasticities, the point-forecast scorecard |

---

## 12. Results

All figures are from `reports/results.md` and are regenerated on every pipeline run.

### 12.1 Phase 2: forecasting (walk-forward hold-out, Mar–Aug 2018)

| Model | RMSE | MAE | WAPE | Bias ratio | Stockout loss | Excess holding | Safety-stock capital |
|---|---|---|---|---|---|---|---|
| Seasonal-Naive | 3.15 | 1.59 | 104.8% | −3.5% | R$ 33,656 | R$ 274 | R$ 24,245 |
| SARIMAX | 3.18 | 1.62 | 106.8% | −3.5% | R$ 32,111 | R$ 283 | R$ 26,259 |
| **LightGBM-Tweedie** | **2.36** | **0.88** | **57.9%** | −8.8% | **R$ 20,385** | **R$ 140** | **R$ 17,543** |

- **Financial reading.** Replacing Seasonal-Naive with LightGBM cuts forgone margin on stockouts by **R$ 13,271 (−39%)** and releases **R$ 6,702 (−28%)** of safety-stock working capital.
- **Asymmetric costs.** Excess-holding cost is about 100× smaller than stockout loss. With a newsvendor critical ratio CM2/(CM2+h) close to 1, under-forecasting is the expensive error. LightGBM's −8.8% bias ratio points to a cheap next step: shade forecasts upward.
- **SARIMAX.** It does no better than the naive benchmark on short, sparse, zero-heavy series.

### 12.2 Phase 3: causal elasticity

Coefficient on ln P, outcome ln(Q+1):

| Category | **DML θ** | 95% CI | Naive within-SKU OLS | Naive pooled OLS |
|---|---|---|---|---|
| All | **−0.34** | [−0.46, −0.22] | −0.99 | +0.06 |
| bed_bath_table | **−0.57** | [−0.91, −0.22] | −0.69 | +0.09 |
| computers_accessories | **−0.18** | [−0.37, +0.02] | −1.02 | +0.17 |
| telephony | **−0.11** | [−0.30, +0.07] | −0.49 | +0.04 |
| watches_gifts | **−0.54** | [−0.74, −0.35] | −1.35 | 0.00 |

Robustness check on sale weeks only, outcome ln Q: All −0.77, bed_bath −1.38, computers −0.44, telephony −0.28, watches −1.22.

After de-attenuation and shrinkage, the median SKU elasticity is −1.61 (bed_bath), −0.32 (computers), −0.27 (telephony) and −0.95 (watches).

**Finding.** The confounding bias runs the opposite way to the textbook "sellers raise prices into peaks" story. On Olist, sellers **discount into demand peaks** (Black Friday, platform growth), so naive OLS **overstates** price sensitivity by about 3×. Pooled OLS even has the wrong sign. An engine that priced off the naive estimate would systematically over-discount and give away gross margin.

### 12.3 Phase 4: one-week optimisation, 72 live SKUs, decision week 2018-08-20 (model-implied, R$/week)

| Scenario | Mode | Profit before → after | CM2 before → after | Median DSI before → after |
|---|---|---|---|---|
| 4 wks supply | Margin | 570 → 1,187 | 600 → 1,218 | 35 → 37.5 days |
| 4 wks supply | Liquidity | 550 → 1,157 | 600 → 1,217 | 35 → 37.5 days |
| 12 wks supply (aged) | Margin | 495 → 1,110 | 600 → 1,215 | 105 → 111 days |
| 12 wks supply (aged) | Liquidity | −6 → 528 | 600 → 1,208 | 105 → 109 days |

**Lerner discount test:**

| Category | Median ε | Median breakeven ε* | SKUs where a cut adds CM2 |
|---|---|---|---|
| bed_bath_table | −1.35 | −8.96 | 0 / 22 |
| computers_accessories | −0.32 | never (CM2 ≤ 0) | 0 / 21 |
| telephony | −0.31 | never (CM2 ≤ 0) | 0 / 10 |
| watches_gifts | −1.10 | −3.66 | 0 / 19 |

**Headline.**
- Thin CM2 margins make discounting value-destroying. A price cut pays only if demand is 4–9× more elastic than the data shows, so the engine moves nearly every SKU to the **+15% volatility cap**. That cap, not the economics, is what binds.
- The trade-off is explicit: higher prices slow sell-through, so DSI rises a few days and working capital freed is negative (about −R$ 380/week).
- Price cuts appear only when carrying cost becomes material: high WACC, heavy overstock, liquidity mode. That is exactly the regime Phase 5A plans for.

### 12.4 Phase 5A: RL sell-through pricing (26-week season, SELIC 6.5%)

Mean season NPV per lot in R$, with the shortfall vs the oracle in brackets:

| Stock regime | Status quo | Myopic (Phase 4) | CE-DP | **PPO agent** | Oracle DP |
|---|---|---|---|---|---|
| lean (< 13 wks supply) | 75.8 (−250.9) | 322.0 (−4.7) | 325.9 (−0.8) | **326.2 (−0.6)** | 326.7 |
| full (13–26 wks) | 111.5 (−393.1) | 493.4 (−11.1) | 499.7 (−4.9) | **497.9 (−6.7)** | 504.6 |
| overstock (> 26 wks) | −203.8 (−287.2) | 58.6 (−24.8) | 61.8 (−21.6) | **61.8 (−21.7)** | 83.4 |

1. **One network nearly matches the exact optimum.** A single network for the whole catalogue matches the exact DP on lean and overstocked lots and is within R$2 per lot on full ones. It beats the one-week Phase 4 optimiser in every regime.
2. **Highly elastic SKUs.** On the 9 SKUs with ε < −2 it is within R$2.2 per lot of the exact DP. It falls short of holding today's price on 1 of 72 SKUs, and trails the myopic optimiser on 18 individual SKUs by small amounts while winning in aggregate.
3. **The price of not knowing ε.** With overstock, even the best realistic policy leaves about **R$22 per lot (≈ 26% of NPV)** below the oracle. That gap is the value of perfect elasticity information, i.e. an upper bound on what a price experiment is worth.
4. **Holding today's price is ruinous when overstocked.** It turns a +R$62 season into a −R$204 one.

### 12.5 Phase 5B: probabilistic forecasting and working capital

| Model | WQL ↓ | WAPE (median) | P(sales ≤ q90) | Fill rate | Avg inventory capital / wk | WC-at-Risk 90 / wk |
|---|---|---|---|---|---|---|
| Chronos-Bolt (zero-shot) | 0.573 | 85.6% | 89% | 88.9% | R$ 27,915 | R$ 17,667 |
| Chronos-2 + covariates (zero-shot) | 0.570 | 87.0% | 89% | 90.1% | R$ 27,845 | R$ 17,603 |
| **LightGBM quantile** | **0.377** | **56.2%** | 92% | 88.0% | **R$ 19,667** | **R$ 9,037** |

- On sparse, lumpy marketplace demand, the domain-trained model wins clearly.
- The foundation models buy 1–2 points more fill rate by holding **about 40% more inventory capital** (R$8.2k a week more), because their intervals are wider.
- Covariates and cross-learning (WQL 0.572) did not close the gap.
- So GenAI is not free: its generality shows up on the balance sheet.

### 12.6 Phase 5C: customer voice

**LLM validation.** Sentiment matches star polarity for **87.7%** of 1–2-star and 4–5-star reviews (95% of 1-star reviews coded negative, 93% of 5-star positive).

**Confounding.**

| Category | DML θ (Phase 3) | DML θ + review confounders |
|---|---|---|
| All | −0.34 | **−0.45** |
| bed_bath_table | −0.57 | −0.50 |
| computers_accessories | −0.18 | **−0.42** |
| telephony | −0.11 | −0.07 |
| watches_gifts | −0.54 | **−0.79** |

Controlling for perception makes demand about 30% more price-sensitive overall, and roughly 2× more for computer accessories. Customer perception was pulling the Phase 3 estimates toward zero.

**Moderation** (change in θ per 1 SD of each signal; positive = less price-sensitive):

| Signal | θ₁ per SD | z |
|---|---|---|
| value perception | +0.11 | 1.33 |
| quality complaints | +0.17 | 2.25 |
| late-delivery complaints | −0.04 | −0.49 |
| sentiment | −0.06 | −0.84 |

- **Value perception** points the expected way (good value → less price-sensitive) but is not significant: only about 3% of reviews discuss price at all.
- **No moderator** clears the Bonferroni threshold.
- **Pricing still uses the Phase 3 elasticities.** The review-adjusted estimates serve as a sensitivity analysis; adopting them would make the engine somewhat more cautious about price rises.

---

## 13. Tests

`uv run pytest`: 40 tests, about 9 s.

| Test | Phase | What it guarantees |
|---|---|---|
| `test_unit_economics` | 1 | CM2 and h_t formulas |
| `test_forecast_features_do_not_leak_current_sales` | 2 | A price change becomes visible to the forecaster only the following week |
| `test_guardrails_hold` (24 cases: ε × mode × inventory) | 4 | P* ≥ active floor; P* inside the band unless the band is overridden; P* never worse than the status quo; aged flag exactly when weeks-of-supply > 8 |
| `test_floor_overrides_band_for_underwater_sku` | 4 | Loss-prevention priority |
| `test_lerner_breakeven_matches_gradient` | 4 | Analytical breakeven ε* matches the sign of the numerical profit gradient |
| `test_env_respects_guardrails` (4 cases) | 5A | Under random actions the RL environment never breaks the floor, the +42% cap or the ±15% band (unless the floor forces a jump) |
| `test_dp_dominates_myopic_in_expectation` | 5A | Over 300 shared demand scenarios, the exact DP earns at least the myopic policy's NPV |
| `test_unsold_stock_books_a_writedown` | 5A | Unsold stock costs (1 − salvage)·COGS, never a phantom gain |
| `test_review_features_are_leakage_safe` | 5C | A review becomes visible only the week after it is written |
| `test_margem_server_endpoints` | 6 | The optional Python server serves the page; `/api/price` respects the floor and band and answers in under 200 ms; `/api/season` returns full 26-week paths; `/api/portfolio` re-prices all 72 products; unknown products return 404 (run in a subprocess) |
| `test_lab_solver_matches_python_optimiser` | 6 | The Scenario lab's in-browser solver matches `engine.optimize.optimise` to within R$0.01 on every product across 4 scenarios (run in Node) |
| `test_price_matches_python` | 6 | The browser optimiser (Simulate) equals `engine.optimize.optimise` on price, profits, flags and uncertainty band, across products, stock levels, rates, goals, fees and shipping |
| `test_season_matches_python` | 6 | Browser RL seasons equal `engine.rl` (hold, weekly, exact planner, PPO agent, oracle): identical stock paths, profits within R$0.10 (Python side in a subprocess) |
| `test_web_app_build_is_consistent` | 6 | The Margem template has exactly one data slot; every price shipped to the web app respects the margin floor and ±15% band; the season replay covers all 26 weeks |

---

## 14. Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Causal estimator | Hand-written FWL Double ML with LightGBM nuisances (rather than EconML) | Transparent; adds SKU fixed effects, blocked cross-fitting and clustered SEs explicitly; the same residuals power the moderation tests |
| Elasticity units | ln(Q+1) slope θ converted to ε = θ·(Q̄+1)/Q̄ | Plugging θ into Q = Q_base·(P/P_base)^ε would understate sensitivity by about half at these volumes |
| Seasonal baseline | Lag 52, falling back to the 4-week mean | Most SKUs did not exist 52 weeks earlier |
| SARIMAX seasonality | ARIMA(1,0,1) plus Fourier exogenous terms | Under two years of weekly data cannot identify a seasonal AR(52) term |
| Liquidity vs EBITDA mode | Liquidity mode charges carry until sell-through | Gives the toggle a precise corporate-finance meaning (remaining cash-conversion cycle) |
| Conflicting guardrails | Floor overrides band | The two constraints can be mutually infeasible; loss prevention takes priority |
| Explaining recommendations | Lerner breakeven elasticity per SKU | Shows *why* the optimiser recommends what it does |
| One-week vs season pricing | Phase 4 for the weekly decision, Phase 5A for the season | Keeps the weekly engine simple and fast while handling write-down risk where it matters |
| RL validation | Exact DP benchmarks (with estimated and true ε) | Certifies near-optimality instead of only comparing against heuristics |
| RL training distribution | SKUs balanced across elasticity regimes; SELIC 6.5–15%; 4–52 weeks of supply | Covers rare but decision-critical cases and macro regimes beyond the sample |
| Unsold stock accounting | Write-down of (1 − salvage)·COGS | Consistent with CM2, which books COGS only on units sold |
| Probabilistic evaluation | Newsvendor backtest at q90 + Working-Capital-at-Risk | Converts calibration into balance-sheet terms |
| LLM labelling | Batched Claude Haiku calls, cached labels, star-rating validation | Cheap, reproducible, independently checked |
| Process isolation | PyTorch stages in separate processes | PyTorch and LightGBM cannot share a macOS process |
| Product front end | One self-contained Margem page with the interactive engine ported to JavaScript and parity-tested against Python; an optional FastAPI server as the Python reference | Judges scan a QR code on conference Wi-Fi: nothing may sleep, cold-start or time out, yet every number must be provably the Python engine's |
| Visual identity | Capi the capybara, handwritten margin notes, dark notebook UI | Memorable for judges, rooted in the Brazilian data, and the "margin notes" carry the explanations |

---

## 15. Limitations and honest caveats

1. **COGS is synthesised** from sector benchmarks, not observed. Every CM2 figure inherits this assumption, which is why Margem exposes the fee and shipping assumptions as sliders.
2. **Freight treatment matters a lot.** The CM2 definition charges the full freight to the seller; on Olist the customer often pays it. With seller freight share = 0, electronics are no longer CM2-negative.
3. **Elasticities are short-run and weekly.** They do not capture competitor responses, search-ranking penalties, or customers switching to rival listings of the same product after a sustained price increase. The +15% recommendations are therefore **model-implied upper bounds** inside the volatility band, not forecasts of realised profit.
4. **Sparse demand.** The typical SKU averages about 1 unit a week over its listing. Half of the 72 live SKUs hit the forecast floor of 0.2 units a week, so their R$ figures are small and uncertain.
5. **Price variation is modest.** The median within-SKU std of log price is about 0.06, so SKU-level elasticities are noisy. Shrinkage, the sign restriction and winsorising are needed and are disclosed.
6. **Only 118 SKUs** pass the eligibility filter. That is the cost of requiring enough history for causal identification.
7. **Optimisation results are not a backtest.** Realised demand at the recommended prices cannot be observed. The portfolio figures are counterfactual model outputs.
8. **The elasticity is estimated on the full sample**, which is standard for causal estimation. The forecasting evaluation is strictly out-of-sample.
9. **The RL world is a simulator.** Demand follows the estimated constant-elasticity curve with negative-binomial noise. The agent is only as good as that model, and it has not been validated on live pricing.
10. **A single-lot season is a stylised setting** (no replenishment, a fixed 50% salvage value). Different salvage or season assumptions change the absolute NPVs.
11. **Review coverage is thin for price talk.** Only about 3% of reviews mention value or price, which limits the power of the moderation tests. LLM labels can also contain errors; the 87.7% agreement with star ratings is the check.
12. **Chronos is evaluated zero-shot only.** A version fine-tuned on Olist data might close part of the gap; that was not tested.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **CM2** | Contribution Margin 2: price net of platform fee, minus COGS, minus freight borne by the seller |
| **COGS** | Cost of goods sold (seller's acquisition cost per unit) |
| **SELIC** | Brazilian Central Bank policy rate; the base cost of capital |
| **h_t** | Weekly carrying cost of one unit: (SELIC + 6%)/52 × COGS |
| **DSI** | Days Sales of Inventory: days to sell current stock at the current rate |
| **CCC** | Cash Conversion Cycle: DSI + DSO − DPO |
| **WAPE** | Weighted absolute percentage error: Σ\|e\| / ΣQ |
| **Double ML / FWL** | Residualise outcome and treatment on confounders with ML, then regress residual on residual |
| **Cross-fitting** | Each residual comes from a model fitted on other folds, which avoids overfitting bias |
| **DerSimonian–Laird** | Random-effects estimator of between-unit variance, used here for shrinkage |
| **Lerner condition** | Margin/elasticity relationship: a price cut pays only if \|ε\| > 1 / margin |
| **Newsvendor critical ratio** | Underage cost / (underage + overage); sets the optimal service level |
| **PPO** | Proximal Policy Optimisation: a reinforcement-learning algorithm that improves a neural policy from simulated experience |
| **Dynamic programming (DP)** | Exact backward solution of a sequential decision problem; the gold-standard benchmark here |
| **CE-DP / oracle DP** | DP using the *estimated* ε as if true / DP knowing the *true* ε |
| **Foundation model (Chronos)** | Transformer pretrained on many time series, used without training on our data (zero-shot) |
| **WQL** | Weighted quantile loss: average pinball loss over the q10/q50/q90 quantiles, scaled by total demand |
| **Working-Capital-at-Risk (90%)** | Σ(q90 − q50)·COGS: capital needed to move from a 50% to a 90% in-stock rate |
| **Moderation** | Interaction showing how the price effect changes with another variable |
| **Margem** | Portuguese for "margin"; the product name of the Phase 6 web app |
