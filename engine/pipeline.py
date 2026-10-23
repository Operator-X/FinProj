"""End-to-end batch run: Phases 1-4 -> artifacts/*.parquet + reports/results.md.

Run:  uv run python -m engine.pipeline
The Streamlit app only reads these artifacts, so live interactions never refit.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, FIN, PATHS, REPORT_DIR, WINDOW_END, seed_everything
from .data import build_panel
from .elasticity import run_dml
from .finance import breakeven_elasticity, holding_rate_annual
from .forecast import LGB_FEATURES, _lgb_model, add_forecast_features, financial_metrics, walk_forward
from .optimize import SkuInputs, optimise

Q_FLOOR = 0.2  # units/week: minimum base demand for a live SKU
SCENARIOS = {"normal (4 wks supply)": 4.0, "aged (12 wks supply)": 12.0}


def sku_state(panel: pd.DataFrame, sku_eps: pd.DataFrame) -> pd.DataFrame:
    """Pricing state for every SKU live in the decision week (last panel week).

    q_base is a true one-step-ahead LightGBM forecast: the model is trained on
    weeks strictly before the decision week, at the status-quo posted price.
    """
    df = add_forecast_features(panel)
    decision = pd.Timestamp(WINDOW_END)
    model = _lgb_model().fit(df.loc[df["week"] < decision, LGB_FEATURES],
                             df.loc[df["week"] < decision, "units_sold"])
    live = df[df["week"] == decision].copy()
    live["q_base"] = np.maximum(model.predict(live[LGB_FEATURES]), Q_FLOOR)
    hist = df[df["week"] < decision].groupby("product_id").agg(
        units_hist=("units_sold", "sum"), weeks_hist=("week", "size"),
        price_min=("avg_price", "min"), price_max=("avg_price", "max"))
    live = live.merge(hist, on="product_id").merge(
        sku_eps[["product_id", "eps", "se", "regime", "eps_raw", "shrink_weight",
                 "sign_restricted", "mean_weekly_units"]], on="product_id")
    live["p_last"] = live["posted_price"]
    live["hold_rate_default"] = holding_rate_annual(live["selic"], FIN)
    live["eps_breakeven"] = breakeven_elasticity(live["p_last"], live["cogs"], live["avg_freight"])
    keep = ["product_id", "category", "cat_en", "week", "p_last", "base_price", "q_base",
            "eps", "se", "eps_breakeven", "eps_raw", "shrink_weight", "sign_restricted", "regime", "cogs",
            "avg_freight", "selic", "hold_rate_default", "units_hist", "weeks_hist",
            "price_min", "price_max", "mean_weekly_units", "avg_review_score"]
    return live[keep].reset_index(drop=True)


def to_inputs(row: pd.Series, inventory: float, hold_rate: float) -> SkuInputs:
    return SkuInputs(p_last=row["p_last"], q_base=row["q_base"], eps=row["eps"],
                     eps_se=row["se"], cogs=row["cogs"], freight=row["avg_freight"],
                     inventory=inventory, annual_hold_rate=hold_rate)


def recommend_all(state: pd.DataFrame) -> pd.DataFrame:
    """Portfolio-wide recommendations for each scenario x strategy mode."""
    rows = []
    for _, r in state.iterrows():
        for scen, wos in SCENARIOS.items():
            inv = float(np.ceil(wos * r["q_base"]))
            for mode in ("margin", "liquidity"):
                d = optimise(to_inputs(r, inv, r["hold_rate_default"]), mode).as_dict()
                d["p_ci_lo"], d["p_ci_hi"] = d.pop("p_star_ci")
                rows.append({"product_id": r["product_id"], "cat_en": r["cat_en"],
                             "scenario": scen, "mode": mode, "inventory": inv,
                             "p_last": r["p_last"], "cogs": r["cogs"], "eps": r["eps"], **d})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal markdown table writer (avoids a tabulate dependency)."""
    cols = list(df.columns)
    fmt = lambda v: format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(metrics: pd.DataFrame, cat: pd.DataFrame, cat_rob: pd.DataFrame,
                 sku: pd.DataFrame, rec: pd.DataFrame, panel: pd.DataFrame,
                 state: pd.DataFrame) -> None:
    """Poster-ready numbers with their financial counterparts."""
    REPORT_DIR.mkdir(exist_ok=True)
    lgb_row = metrics.set_index("model").loc["LightGBM-Tweedie"]
    naive_row = metrics.set_index("model").loc["Seasonal-Naive"]
    agg = (rec.groupby(["scenario", "mode"])
           .agg(skus=("product_id", "size"), profit_base=("profit_base", "sum"),
                profit_star=("profit_star", "sum"), cm2_base=("cm2_base", "sum"),
                cm2_star=("cm2_star", "sum"), hold_base=("hold_base", "sum"),
                hold_star=("hold_star", "sum"), wc_freed=("wc_freed", "sum"),
                dsi_base=("dsi_base", "median"), dsi_star=("dsi_star", "median"),
                price_up=("p_star", lambda s: int((s > rec.loc[s.index, "p_last"] * 1.001).sum())),
                floor_binding=("floor_binding", "sum"), band_overridden=("band_overridden", "sum"))
           .reset_index())
    agg["profit_uplift_pct"] = 100 * (agg["profit_star"] - agg["profit_base"]) / agg["profit_base"].abs()

    econ = (panel[panel["has_sale"] == 1].groupby("cat_en")
            .apply(lambda g: pd.Series({
                "median_price": g["avg_price"].median(),
                "cogs_share": (g["cogs"] / g["base_price"]).iloc[0],
                "median_freight": g["avg_freight"].median(),
                "median_cm2_unit": g["cm2_unit"].median(),
                "share_cm2_negative": (g["cm2_unit"] < 0).mean()}), include_groups=False)
            .reset_index())

    txt = f"""# Results: Algorithmic Working Capital & Margin Optimization

Generated {time.strftime('%Y-%m-%d %H:%M')}. Data window {panel['week'].min():%Y-%m-%d} to {WINDOW_END}.
Panel: {panel['product_id'].nunique()} SKUs, {len(panel):,} SKU-weeks, {panel['units_sold'].sum():,.0f} units,
{100 * (1 - panel['has_sale'].mean()):.0f}% zero-sale weeks.

## 1. Unit economics (CM2 = P - COGS - F - 10% fee)
{_md(econ, '.2f')}

Low-ticket electronics are **structurally CM2-negative** when the seller absorbs freight:
a BRL 30 phone accessory carries ~BRL 15 freight against a 20% gross margin.

## 2. Forecasting (walk-forward, {metrics.shape[0]} models, one-step-ahead, refit every 4 weeks)
{_md(metrics)}

Financial reading: moving from Seasonal-Naive to LightGBM-Tweedie cuts forgone CM2 on
stockouts by **BRL {naive_row.stockout_loss_brl - lgb_row.stockout_loss_brl:,.0f}**
({100 * (1 - lgb_row.stockout_loss_brl / naive_row.stockout_loss_brl):.0f}%) over the hold-out and releases
**BRL {naive_row.safety_stock_capital_brl - lgb_row.safety_stock_capital_brl:,.0f}** of safety-stock working capital
(95% cycle-service level, 1-week lead time). Excess-holding cost is two orders of magnitude
smaller than stockout loss: with critical ratio CM2/(CM2+h) near 1, the newsvendor-optimal
policy leans toward over-stocking, and the forecaster's negative bias ratio is the costly error.

## 3. Causal elasticity (FE Double-ML, blocked 5-fold cross-fitting, SKU-clustered SE)
Coefficient on ln P with outcome ln(Q+1):
{_md(cat)}

Robustness, intensive margin (weeks with Q>0, outcome ln Q):
{_md(cat_rob)}

Naive within-SKU OLS overstates price sensitivity (sellers cut prices into Black Friday and
platform-growth peaks, so price and demand co-move negatively for non-causal reasons).
Pooled cross-SKU OLS even has the wrong sign. Pricing off the naive estimate would
systematically over-discount and leak gross margin.

SKU elasticities (log1p-de-attenuated, DerSimonian-Laird shrinkage, sign-restricted):
{_md(sku.groupby('cat_en').agg(skus=('eps', 'size'), median_eps=('eps', 'median'),
                               share_elastic=('eps', lambda s: (s.abs() > 1).mean()),
                               sign_restricted=('sign_restricted', 'sum')).reset_index())}

## 4. Optimisation: portfolio impact for decision week {WINDOW_END} (model-implied, BRL/week)
{_md(agg, '.1f')}

**Lerner discount test.** A price cut raises CM2 only if |eps| > 1 / (CM2 / net revenue).
{_md(state.assign(accretive=state['eps'] < state['eps_breakeven'])
     .groupby('cat_en').agg(skus=('eps', 'size'), median_eps=('eps', 'median'),
                            median_breakeven=('eps_breakeven', 'median'),
                            discount_accretive=('accretive', 'sum')).reset_index(), '.2f')}

Thin CM2 margins make discounting value-destroying for almost every SKU: the
breakeven elasticity is far beyond anything identified in the data. Price cuts only
become rational once carrying cost bites (high cost of capital, heavy overstock), which is
what the liquidity mode and the aged-stock floor relaxation capture.

Margin mode is the spec objective (one week of carry). Liquidity mode charges carrying cost
until leftover stock is sold through, which pushes elastic SKUs toward the floor.
`band_overridden` counts SKUs whose 1.15x COGS floor exceeds +15% of the current price,
where the loss-prevention floor overrides the volatility band.
"""
    (REPORT_DIR / "results.md").write_text(txt)


def main() -> None:
    seed_everything()
    ARTIFACT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    panel = build_panel()
    panel.to_parquet(PATHS.panel)
    print(f"[1] panel {panel.shape} ({time.time() - t0:.0f}s)")

    fc = walk_forward(panel)
    fc.to_parquet(PATHS.forecasts)
    metrics = financial_metrics(fc)
    metrics.to_parquet(PATHS.forecast_metrics)
    print(f"[2] forecasts {fc.shape} ({time.time() - t0:.0f}s)\n{metrics.round(3).to_string()}")

    cat, sku, _ = run_dml(panel)
    cat_rob, _, _ = run_dml(panel, sales_weeks_only=True)
    cat.assign(spec="ln(Q+1), all weeks").to_parquet(PATHS.elasticity_cat)
    sku.to_parquet(PATHS.elasticity_sku)
    print(f"[3] elasticity ({time.time() - t0:.0f}s)\n{cat.round(3).to_string()}")

    state = sku_state(panel, sku)
    state.to_parquet(PATHS.sku_state)
    rec = recommend_all(state)
    rec.to_parquet(PATHS.recommendations)
    print(f"[4] {len(state)} live SKUs, {len(rec)} recommendations ({time.time() - t0:.0f}s)")

    write_report(metrics, cat, cat_rob, sku, rec, panel, state)
    print(f"report -> {REPORT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
