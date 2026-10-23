"""Phase 3: causal price elasticity via fixed-effects Double Machine Learning.

Partially linear model for SKU i in week t:

    ln(Q_it + 1) = eps * ln(P_it) + g(X_it) + a_i + u_it
    ln(P_it)     =                  m(X_it) + b_i + v_it

Sellers raise prices into demand peaks, so E[v*u] != 0 under naive OLS and the
slope is biased toward zero (or even positive). The estimator:

  1. Within-transform Y and T by SKU (removes a_i, b_i: fixed quality/brand).
  2. Fit g, m with LightGBM using blocked temporal K-fold cross-fitting, so each
     residual comes from a model that never saw that time block.
  3. Frisch-Waugh-Lovell: eps = sum(T~ Y~) / sum(T~^2) on residuals, with
     SKU-clustered standard errors.

SKU-level elasticities are residual-on-residual slopes within each SKU, shrunk
toward their category estimate with an empirical-Bayes normal prior.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import SEED
from .forecast import add_forecast_features

CONFOUNDERS: list[str] = [
    "lag_1", "lag_2", "lag_4", "roll4_mean", "roll4_std", "weeks_listed", "t_index",
    "cat_velocity_lag1", "log_cat_price", "delivery_days_lag1", "avg_review_score",
    "seller_count_lag1", "selic", "sin1", "cos1", "sin2", "cos2",
    "ev_black_friday", "ev_christmas", "ev_mothers_day", "ev_valentines_br",
    "ev_fathers_day", "ev_consumer_day", "ev_truckers_strike", "cat_code",
]


def _nuisance() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
        deterministic=True, force_row_wise=True, verbose=-1)


def _ols_slope(t: np.ndarray, y: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """Slope through the origin and its cluster-robust (sandwich) SE."""
    sxx = float(t @ t)
    beta = float(t @ y) / sxx
    score = pd.Series(t * (y - beta * t)).groupby(cluster).sum().to_numpy()
    g = len(score)
    se = float(np.sqrt((score ** 2).sum() * g / max(g - 1, 1)) / sxx)
    return beta, se


def residualize(df: pd.DataFrame, y_col: str = "log_q", t_col: str = "log_p",
                n_folds: int = 5, confounders: list[str] | None = None) -> pd.DataFrame:
    """Steps 1-2: within-SKU demeaning + cross-fitted LightGBM residuals.

    Returns df with added columns y_res, t_res (orthogonalised outcome/treatment).
    Folds are contiguous blocks of calendar weeks, never random rows.
    """
    df = df.copy()
    for c in (y_col, t_col):
        df[f"{c}_w"] = df[c] - df.groupby("product_id")[c].transform("mean")
    weeks = np.sort(df["week"].unique())
    block = {w: i for i, chunk in enumerate(np.array_split(weeks, n_folds)) for w in chunk}
    fold = df["week"].map(block).to_numpy()
    y_hat = np.zeros(len(df))
    t_hat = np.zeros(len(df))
    X = df[confounders or CONFOUNDERS]
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        y_hat[te] = _nuisance().fit(X[tr], df.loc[tr, f"{y_col}_w"]).predict(X[te])
        t_hat[te] = _nuisance().fit(X[tr], df.loc[tr, f"{t_col}_w"]).predict(X[te])
    df["y_res"] = df[f"{y_col}_w"] - y_hat
    df["t_res"] = df[f"{t_col}_w"] - t_hat
    return df


def category_elasticities(res: pd.DataFrame, panel_feat: pd.DataFrame) -> pd.DataFrame:
    """Step 3 per category, alongside naive estimators that show the bias.

    Returns DataFrame[cat_en, eps_dml, se_dml, ci_lo, ci_hi, eps_ols_pooled,
    eps_ols_fe, n_obs, n_skus].
    """
    rows = []
    groups = [("ALL", res)] + list(res.groupby("cat_en"))
    for cat, g in groups:
        eps, se = _ols_slope(g["t_res"].to_numpy(), g["y_res"].to_numpy(), g["product_id"].to_numpy())
        raw = panel_feat.loc[g.index]
        # Naive 1: pooled OLS slope of ln(Q+1) on ln(P) (cross-SKU price levels).
        tp = raw["log_p"] - raw["log_p"].mean()
        ols_pooled = float(tp @ (raw["log_q"] - raw["log_q"].mean()) / (tp @ tp))
        # Naive 2: within-SKU OLS, no confounder adjustment (timing endogeneity).
        tw, yw = g["log_p_w"].to_numpy(), g["log_q_w"].to_numpy()
        ols_fe = float(tw @ yw / (tw @ tw))
        rows.append({"cat_en": cat, "eps_dml": eps, "se_dml": se,
                     "ci_lo": eps - 1.96 * se, "ci_hi": eps + 1.96 * se,
                     "eps_ols_pooled": ols_pooled, "eps_ols_fe": ols_fe,
                     "n_obs": len(g), "n_skus": g["product_id"].nunique()})
    return pd.DataFrame(rows)


def sku_elasticities(res: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of SKU slopes toward the category elasticity.

    eps_i^EB = (tau^2 * eps_i + se_i^2 * eps_c) / (tau^2 + se_i^2), with tau^2
    the method-of-moments between-SKU variance. SKUs with little price variation
    (huge se_i) inherit the category estimate; well-identified SKUs keep theirs.
    """
    cat_eps = cat.set_index("cat_en")["eps_dml"]
    rows = []
    for (pid, c), g in res.groupby(["product_id", "cat_en"]):
        t, y = g["t_res"].to_numpy(), g["y_res"].to_numpy()
        sxx = float(t @ t)
        if sxx < 1e-8:
            b, s = np.nan, np.inf
        else:
            b = float(t @ y) / sxx
            resid = y - b * t
            s = float(np.sqrt((resid @ resid) / max(len(t) - 1, 1) / sxx))
        rows.append({"product_id": pid, "cat_en": c, "eps_raw": b, "se_raw": s,
                     "price_cv": float(np.exp(g["log_p"]).std() / np.exp(g["log_p"]).mean())})
    sku = pd.DataFrame(rows)
    sku["eps_cat"] = sku["cat_en"].map(cat_eps)
    out = []
    for c, g in sku.groupby("cat_en"):
        # DerSimonian-Laird between-SKU variance (random-effects meta-analysis).
        ok = np.isfinite(g["se_raw"]) & g["eps_raw"].notna() & (g["se_raw"] > 0)
        b, v = g.loc[ok, "eps_raw"].to_numpy(), g.loc[ok, "se_raw"].to_numpy() ** 2
        w = 1.0 / v
        b_bar = (w * b).sum() / w.sum()
        q_stat = (w * (b - b_bar) ** 2).sum()
        tau2 = max((q_stat - (len(b) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()), 0.0)
        se2 = g["se_raw"].replace(np.inf, 1e6) ** 2
        wt = tau2 / (tau2 + se2)
        g = g.assign(theta=wt * g["eps_raw"].fillna(g["eps_cat"]) + (1 - wt) * g["eps_cat"],
                     theta_se=np.sqrt(tau2 * se2 / (tau2 + se2)) if tau2 > 0 else 0.0 * se2,
                     shrink_weight=wt, tau=np.sqrt(tau2))
        out.append(g)
    sku = pd.concat(out, ignore_index=True)
    # theta is d ln(Q+1)/d ln P. The demand elasticity d ln Q/d ln P equals
    # theta * (Q+1)/Q; evaluated at the SKU's mean weekly units this removes the
    # log1p attenuation that is severe at 1-2 units/week.
    qbar = sku["product_id"].map(res.groupby("product_id")["units_sold"].mean())
    scale = (qbar + 1.0) / qbar
    sku["mean_weekly_units"] = qbar
    sku["eps"] = sku["theta"] * scale
    # Category-level SE floor: SKU posteriors cannot be more precise than the
    # category estimate they borrow strength from.
    cat_se = sku["cat_en"].map(cat.set_index("cat_en")["se_dml"])
    sku["se"] = np.sqrt(sku["theta_se"] ** 2 + cat_se ** 2) * scale
    # Sign restriction: an upward-sloping demand curve is economically
    # implausible for these goods; such SKUs fall back to their category.
    bad = sku["eps"] >= 0
    sku["sign_restricted"] = bad
    sku.loc[bad, "eps"] = sku.loc[bad, "eps_cat"] * scale[bad]
    # Winsorise at -5: beyond that a 1% price move swings volume >5%, outside
    # the range the weekly price variation in this data can identify.
    sku["eps"] = sku["eps"].clip(lower=-5.0)
    sku["regime"] = np.where(sku["eps"].abs() < 1.0, "Inelastic: margin expansion",
                             "Elastic: liquidity acceleration")
    return sku


def run_dml(panel: pd.DataFrame, sales_weeks_only: bool = False,
            extra_confounders: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full Phase 3. `sales_weeks_only=True` gives the intensive-margin
    robustness check (ln Q on weeks with Q>0, where price is directly observed).

    Returns (category table, SKU table, residual frame).
    """
    feat = add_forecast_features(panel)
    if sales_weeks_only:
        feat = feat[feat["has_sale"] == 1].copy()
        feat["log_q"] = np.log(feat["units_sold"])
    res = residualize(feat, confounders=CONFOUNDERS + (extra_confounders or []))
    cat = category_elasticities(res, feat)
    sku = sku_elasticities(res, cat)
    return cat, sku, res


def moderation(res: pd.DataFrame, moderators: list[str]) -> pd.DataFrame:
    """Does price sensitivity vary with a SKU-week characteristic z?

    Fits y~ = theta0 * t~ + theta1 * t~ * (z - mean z) / sd(z) on the cross-fitted
    residuals, separately per moderator, with SKU-clustered sandwich SEs.
    theta1 is the change in the ln(Q+1) price coefficient per 1 SD of z;
    theta1 > 0 means *less* price-sensitive where z is high.
    """
    rows = []
    t, y, cl = res["t_res"].to_numpy(), res["y_res"].to_numpy(), res["product_id"].to_numpy()
    for z in moderators:
        zc = ((res[z] - res[z].mean()) / res[z].std()).to_numpy()
        X = np.column_stack([t, t * zc])
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        u = y - X @ beta
        scores = pd.DataFrame(X * u[:, None]).groupby(cl).sum().to_numpy()
        g = len(scores)
        V = XtX_inv @ (scores.T @ scores) @ XtX_inv * g / (g - 1)
        se = np.sqrt(np.diag(V))
        rows.append({"moderator": z, "theta0": beta[0], "theta0_se": se[0],
                     "theta1_per_sd": beta[1], "theta1_se": se[1], "z_stat": beta[1] / se[1]})
    return pd.DataFrame(rows)
