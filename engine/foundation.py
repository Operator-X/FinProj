"""Phase 5B: probabilistic forecasting with GenAI time-series foundation models vs LightGBM.

Chronos models are transformer language models pretrained on large corpora of
time series; here they forecast Olist demand *zero-shot* (no training on our
data). Both they and the LightGBM quantile models output demand quantiles, which
turn into balance-sheet quantities:

  * an order-up-to policy S = ceil(q90) -> realised stockout margin loss,
    excess carrying cost, and average inventory capital employed;
  * Working-Capital-at-Risk (90%) = sum (q90 - q50) * COGS: the extra capital
    needed to move from a coin-flip to a 90% in-stock rate. It is only meaningful
    if q90 is calibrated, so empirical coverage is reported next to it.

Same walk-forward protocol as Phase 2: every forecast for week t uses data up to
t-1 only, and only SKUs listed before the block origin are scored.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, PATHS, ROOT, SEED
from .forecast import HOLDOUT_START, LGB_FEATURES, STEP, _lgb_model, add_forecast_features

QUANTILES = [0.1, 0.5, 0.9]
CHRONOS_CMD = "import torch; from engine.foundation import _chronos_main; _chronos_main()"
COVARIATES_PAST = ["log_posted_price", "holiday", "ev_truckers_strike", "selic", "cat_velocity_lag1"]
COVARIATES_FUTURE = ["log_posted_price", "holiday", "ev_truckers_strike", "selic"]  # known at t-1


def _targets(df: pd.DataFrame, holdout_start: str, step: int) -> pd.DataFrame:
    """The (SKU, week) cells to forecast, tagged with their walk-forward origin."""
    weeks = np.sort(df["week"].unique())
    origins = [w for w in weeks if w >= np.datetime64(holdout_start)][::step]
    cells = []
    for o in origins:
        listed = df.loc[df["week"] < o, "product_id"].unique()
        block = df[(df["week"] >= o) & (df["week"] < o + np.timedelta64(7 * step, "D"))
                   & df["product_id"].isin(listed)]
        cells.append(block.assign(origin=o))
    return pd.concat(cells)


def _contexts(df: pd.DataFrame, cells: pd.DataFrame, with_cov: bool) -> list:
    """History up to t-1 for each target cell (and covariates if requested)."""
    import torch

    by_sku = {pid: g.reset_index(drop=True) for pid, g in df.groupby("product_id")}
    out = []
    for r in cells.itertuples():
        g = by_sku[r.product_id]
        hist = g[g["week"] < r.week]
        y = hist["units_sold"].to_numpy(dtype=np.float32)
        if not with_cov:
            out.append(torch.tensor(y))
            continue
        nxt = g[g["week"] == r.week]
        out.append({
            "target": y,
            "past_covariates": {c: hist[c].to_numpy(dtype=np.float32) for c in COVARIATES_PAST},
            "future_covariates": {c: nxt[c].to_numpy(dtype=np.float32) for c in COVARIATES_FUTURE},
        })
    return out


def chronos_forecasts(df: pd.DataFrame, cells: pd.DataFrame, model_id: str,
                      with_cov: bool = False) -> np.ndarray:
    """Zero-shot one-step-ahead quantiles, shape (n_cells, 3) for QUANTILES."""
    import torch
    from chronos import BaseChronosPipeline

    torch.manual_seed(SEED)
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
    ctx = _contexts(df, cells, with_cov)
    q, _ = pipe.predict_quantiles(ctx, prediction_length=1, quantile_levels=QUANTILES)
    arr = np.stack([np.asarray(t).reshape(-1, len(QUANTILES))[0] for t in q])
    return np.clip(np.sort(arr, axis=1), 0, None)  # enforce monotone, non-negative


def lgb_quantile_forecasts(df: pd.DataFrame, cells: pd.DataFrame) -> np.ndarray:
    """LightGBM quantile regressors (pinball loss), refit at each origin."""
    out = np.zeros((len(cells), len(QUANTILES)))
    pos = {idx: i for i, idx in enumerate(cells.index)}
    for o, block in cells.groupby("origin"):
        train = df[df["week"] < o]
        rows = [pos[i] for i in block.index]
        for j, a in enumerate(QUANTILES):
            m = _lgb_model().set_params(objective="quantile", alpha=a)
            out[rows, j] = m.fit(train[LGB_FEATURES], train["units_sold"]).predict(block[LGB_FEATURES])
    return np.clip(np.sort(out, axis=1), 0, None)


def run(panel: pd.DataFrame, holdout_start: str = HOLDOUT_START, step: int = STEP) -> pd.DataFrame:
    """All probabilistic forecasts in long format.

    Returns DataFrame[product_id, cat_en, week, origin, units_sold, cm2_unit,
    h_weekly, cogs, model, q10, q50, q90].
    """
    df = add_forecast_features(panel)
    cells = _targets(df, holdout_start, step)
    base = cells[["product_id", "cat_en", "week", "origin", "units_sold", "cm2_unit", "h_weekly", "cogs"]]
    # PyTorch and LightGBM ship different OpenMP runtimes; using both thread
    # pools in one macOS process segfaults or deadlocks. Chronos therefore runs
    # in a separate process, which must load torch *before* LightGBM.
    cache = ARTIFACT_DIR / "chronos_quantiles.parquet"
    subprocess.run([sys.executable, "-c", CHRONOS_CMD], check=True, cwd=ROOT)
    chron = pd.read_parquet(cache)
    results = {name: g[["q10", "q50", "q90"]].to_numpy() for name, g in chron.groupby("model", sort=False)}
    results["LightGBM quantile"] = lgb_quantile_forecasts(df, cells)
    return pd.concat([base.assign(model=name, q10=q[:, 0], q50=q[:, 1], q90=q[:, 2])
                      for name, q in results.items()], ignore_index=True)


def _chronos_main() -> None:
    """Entry point for the Chronos-only child process (never trains LightGBM)."""
    df = add_forecast_features(pd.read_parquet(PATHS.panel))
    cells = _targets(df, HOLDOUT_START, STEP)
    frames = []
    for name, model_id, cov in [("Chronos-Bolt (zero-shot)", "amazon/chronos-bolt-small", False),
                                ("Chronos-2 + covariates (zero-shot)", "amazon/chronos-2", True)]:
        q = chronos_forecasts(df, cells, model_id, cov)
        frames.append(pd.DataFrame({"model": name, "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2]}))
        print(f"  {name}: {len(q)} forecasts", flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(ARTIFACT_DIR / "chronos_quantiles.parquet")


def _pinball(y: np.ndarray, q: np.ndarray, a: float) -> np.ndarray:
    d = y - q
    return np.maximum(a * d, (a - 1) * d)


def probabilistic_metrics(pf: pd.DataFrame) -> pd.DataFrame:
    """Accuracy, calibration and the financial newsvendor backtest per model."""
    rows = []
    for name, g in pf.groupby("model", sort=False):
        y = g["units_sold"].to_numpy()
        qs = g[["q10", "q50", "q90"]].to_numpy()
        wql = 2 * sum(_pinball(y, qs[:, j], a).sum() for j, a in enumerate(QUANTILES)) / len(QUANTILES) / y.sum()
        s = np.ceil(qs[:, 2] - 1e-9)  # order-up-to the 90% quantile, whole units
        cm2 = np.clip(g["cm2_unit"].to_numpy(), 0, None)
        rows.append({
            "model": name,
            "WAPE_median": float(np.abs(qs[:, 1] - y).sum() / y.sum()),
            "WQL": float(wql),
            "coverage_80": float(((y >= qs[:, 0]) & (y <= qs[:, 2])).mean()),
            "coverage_q90": float((y <= qs[:, 2]).mean()),
            "stockout_loss_brl": float((np.clip(y - s, 0, None) * cm2).sum()),
            "excess_holding_brl": float((np.clip(s - y, 0, None) * g["h_weekly"]).sum()),
            "fill_rate": float(np.minimum(y, s).sum() / y.sum()),
            "avg_capital_brl_wk": float((s * g["cogs"]).groupby(g["week"]).sum().mean()),
            "wc_at_risk_90_brl_wk": float(((qs[:, 2] - qs[:, 1]) * g["cogs"]).groupby(g["week"]).sum().mean()),
        })
    return pd.DataFrame(rows)
