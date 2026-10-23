"""Phase 2: demand forecasting with walk-forward validation and a financial
evaluation layer.

Protocol (no leakage): rolling origins every `step` weeks over the hold-out
window. At each origin the model is fit on weeks < origin only, then issues
one-step-ahead forecasts for the next `step` weeks (the weekly repricing
cadence), using only information available at t-1. Only SKUs already listed
before the origin are scored.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import SEED
from .finance import safety_stock_units

HOLDOUT_START = "2018-03-05"
STEP = 4


def add_forecast_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe regressors for forecasting.

    In a zero-sale week the transaction price is unobserved (forward-filled),
    so a *changed* contemporaneous price would reveal that a sale happened.
    The forecaster therefore uses the posted price as of t-1, and lags every
    field that is only populated by realised sales.
    """
    df = panel.copy()
    g = df.groupby("product_id", sort=False)
    df["posted_price"] = g["avg_price"].shift(1).fillna(df["avg_price"])
    df["log_posted_price"] = np.log(df["posted_price"])
    df["seller_count_lag1"] = g["seller_count"].shift(1).bfill()
    df["delivery_days_lag1"] = g["delivery_days"].shift(1).bfill()
    df["cat_med_price_lag1"] = g["cat_med_price"].shift(1).fillna(df["cat_med_price"])
    df["price_ratio_lag1"] = df["posted_price"] / df["cat_med_price_lag1"]
    df["lag_52"] = g["units_sold"].shift(52)
    df["cat_code"] = df["category"].astype("category").cat.codes
    df["holiday"] = df[["ev_black_friday", "ev_christmas"]].max(axis=1)
    return df


LGB_FEATURES: list[str] = [
    "lag_1", "lag_2", "lag_4", "roll4_mean", "roll4_std", "weeks_listed",
    "log_posted_price", "price_ratio_lag1", "cat_velocity_lag1", "seller_count_lag1",
    "delivery_days_lag1", "avg_review_score", "selic", "sin1", "cos1", "sin2", "cos2",
    "ev_black_friday", "ev_christmas", "ev_mothers_day", "ev_valentines_br",
    "ev_fathers_day", "ev_consumer_day", "ev_truckers_strike", "cat_code",
]


def _seasonal_naive(test: pd.DataFrame) -> np.ndarray:
    """Q_hat_t = Q_{t-52}; falls back to the trailing 4-week mean when the SKU
    was not yet listed a year earlier (most of the catalogue launched in 2017)."""
    return test["lag_52"].fillna(test["roll4_mean"]).to_numpy()


def _sarimax_sku(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Per-SKU SARIMAX(1,0,1)+c on log1p(Q) with exogenous price, holiday,
    truckers-strike and SELIC regressors plus annual Fourier seasonality
    (the series are too short for a seasonal AR term). One-step-ahead
    forecasts over the test block use filtered states with frozen parameters.
    """
    cols = ["log_posted_price", "holiday", "ev_truckers_strike", "selic", "sin1", "cos1"]
    if len(train) < 12:
        return test["roll4_mean"].to_numpy()
    full = pd.concat([train, test])
    exog = full[cols]
    exog = exog.loc[:, train[cols].std() > 1e-9]  # drop regressors constant in-sample
    y = np.log1p(full["units_sold"].to_numpy())
    n = len(train)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ex_tr = exog.iloc[:n].to_numpy() if exog.shape[1] else None
            res = SARIMAX(y[:n], exog=ex_tr, order=(1, 0, 1), trend="c").fit(disp=False, maxiter=200)
            ex_full = exog.to_numpy() if exog.shape[1] else None
            res_full = res.apply(y, exog=ex_full)
            pred = res_full.get_prediction(start=n, end=len(full) - 1).predicted_mean
        out = np.expm1(np.asarray(pred))
        # Plausibility guard: exp() back-transform of an extrapolated exog can
        # explode; a forecast above 2x the SKU's historical peak is rejected.
        cap = 2.0 * train["units_sold"].max() + 1.0
        if not np.all(np.isfinite(out)) or np.any(out > cap):
            raise ValueError("implausible forecast")
        return np.clip(out, 0, None)
    except Exception:
        return test["roll4_mean"].to_numpy()


def _lgb_model() -> lgb.LGBMRegressor:
    """Tweedie LightGBM: a compound Poisson-gamma loss that fits the
    zero-inflated, right-skewed weekly unit distribution directly."""
    return lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.2, n_estimators=400,
        learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
        deterministic=True, force_row_wise=True, verbose=-1)


def walk_forward(panel: pd.DataFrame, holdout_start: str = HOLDOUT_START,
                 step: int = STEP) -> pd.DataFrame:
    """Run all three models over rolling origins.

    Returns:
        Long DataFrame [product_id, category, week, origin, units_sold,
        f_seasonal_naive, f_sarimax, f_lightgbm, cm2_unit, h_weekly, cogs].
    """
    df = add_forecast_features(panel)
    weeks = np.sort(df["week"].unique())
    origins = [w for w in weeks if w >= np.datetime64(holdout_start)][::step]
    out = []
    for origin in origins:
        block_end = origin + np.timedelta64(7 * step, "D")
        train = df[df["week"] < origin]
        listed = train["product_id"].unique()
        test = df[(df["week"] >= origin) & (df["week"] < block_end)
                  & df["product_id"].isin(listed)].copy()
        if test.empty:
            continue
        model = _lgb_model().fit(train[LGB_FEATURES], train["units_sold"])
        test["f_lightgbm"] = np.clip(model.predict(test[LGB_FEATURES]), 0, None)
        test["f_seasonal_naive"] = _seasonal_naive(test)
        test["f_sarimax"] = np.nan
        for pid, te in test.groupby("product_id"):
            tr = train[train["product_id"] == pid]
            test.loc[te.index, "f_sarimax"] = _sarimax_sku(tr, te)
        test["origin"] = origin
        out.append(test)
    keep = ["product_id", "category", "cat_en", "week", "origin", "units_sold",
            "f_seasonal_naive", "f_sarimax", "f_lightgbm", "cm2_unit", "h_weekly", "cogs"]
    return pd.concat(out, ignore_index=True)[keep]


MODELS = {"f_seasonal_naive": "Seasonal-Naive", "f_sarimax": "SARIMAX", "f_lightgbm": "LightGBM-Tweedie"}


def financial_metrics(fc: pd.DataFrame, z: float = 1.645) -> pd.DataFrame:
    """Statistical and financial scorecard per model.

    Financial dual-reporting:
      Stockout capital loss  = sum max(0, Q - Q_hat) * max(CM2, 0)
          (margin forgone on unmet demand; a negative-CM2 unit is not a loss).
      Excess holding burden  = sum max(0, Q_hat - Q) * h_t
          (one week of carrying cost on over-stocked units).
      Safety-stock capital   = sum_SKU z * sigma(e_SKU) * COGS_SKU
          (inventory BRL that must be held to absorb forecast error at 95% CSL).
    """
    rows = []
    q = fc["units_sold"].to_numpy()
    for col, name in MODELS.items():
        f = fc[col].to_numpy()
        e = f - q
        sigma = (fc.assign(e=e).groupby("product_id")
                 .agg(sd=("e", "std"), cogs=("cogs", "first")))
        ss_capital = float((safety_stock_units(sigma["sd"].fillna(0), z) * sigma["cogs"]).sum())
        stockout = float((np.clip(q - f, 0, None) * np.clip(fc["cm2_unit"], 0, None)).sum())
        excess = float((np.clip(f - q, 0, None) * fc["h_weekly"]).sum())
        rows.append({
            "model": name,
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAE": float(np.mean(np.abs(e))),
            "WAPE": float(np.abs(e).sum() / q.sum()),
            "bias_ratio": float(e.sum() / q.sum()),
            "stockout_loss_brl": stockout,
            "excess_holding_brl": excess,
            "safety_stock_capital_brl": ss_capital,
        })
    return pd.DataFrame(rows)
