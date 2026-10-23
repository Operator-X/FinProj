"""Phase 5: live pricing & liquidity simulator.

Run: uv run streamlit run app.py   (after `uv run python -m engine.pipeline`)

Everything heavy (panel, forecasts, Double-ML) is precomputed; an interaction
only re-solves one SKU's 1-D price problem, so the response stays well under
200 ms.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from engine.config import FIN, PATHS, SEED
from engine.finance import breakeven_elasticity
from engine.optimize import GRID, SkuInputs, demand, optimise, profit

# Reference data-viz palette: categorical slots 1-2, status good/critical, muted ink.
BLUE, ORANGE, GOOD, BAD, MUTED = "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b", "#898781"
MODES = {"Margin Maximization (EBITDA)": "margin", "Aggressive Cash Recovery (Liquidity)": "liquidity"}

st.set_page_config(page_title="WC & Margin Engine", layout="wide")


@st.cache_data
def load() -> dict[str, pd.DataFrame]:
    """Read pipeline artifacts once per session."""
    return {k: pd.read_parquet(getattr(PATHS, k)) for k in
            ("sku_state", "elasticity_cat", "elasticity_sku", "forecast_metrics",
             "recommendations", "panel")}


def brl(v: float, signed: bool = False) -> str:
    return f"{'+' if signed and v >= 0 else ''}R$ {v:,.2f}"


def mbrl(v: float) -> str:
    """brl() for st.markdown, where a bare $ pair would render as LaTeX."""
    return brl(v).replace("$", "\\$")


try:
    data = load()
except FileNotFoundError:
    st.error("Artifacts missing. Run `uv run python -m engine.pipeline` first.")
    st.stop()
state = data["sku_state"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    cat = st.selectbox("Category", sorted(state["cat_en"].unique()))
    skus = state[state["cat_en"] == cat].sort_values("q_base", ascending=False)
    labels = {r.product_id: f"{r.product_id[:8]} · ε={r.eps:.2f} · {r.q_base:.1f} units/wk"
              for r in skus.itertuples()}
    pid = st.selectbox("SKU", list(labels), format_func=labels.get)
    row = state.set_index("product_id").loc[pid]
    inv = st.number_input("Current available inventory (units)", min_value=0, max_value=2000,
                          value=int(np.ceil(4 * row["q_base"])), step=1,
                          help="Default: 4 weeks of forecast supply. Above 8 weeks the aged-stock "
                               "trigger relaxes the price floor to 1.00x COGS.")
    wacc = st.slider("WACC / macro cost of capital (%)", 5.0, 20.0, float(round(row["selic"] * 100, 2)),
                     0.25, help=f"Default = SELIC in the decision week. Holding cost adds a "
                                f"{FIN.storage_spread:.0%} storage/obsolescence spread.")
    mode_label = st.radio("Pricing strategy", list(MODES))
    mode = MODES[mode_label]
    with st.expander("Unit-economics assumptions"):
        fee = st.slider("Platform fee rate", 0.0, 0.25, FIN.platform_fee, 0.01)
        fshare = st.slider("Freight borne by seller", 0.0, 1.0, FIN.freight_share, 0.05)
    fin = replace(FIN, platform_fee=fee, freight_share=fshare)

# ---------------------------------------------------------------- solve
t0 = time.perf_counter()
x = SkuInputs(p_last=row["p_last"], q_base=row["q_base"], eps=row["eps"], eps_se=row["se"],
              cogs=row["cogs"], freight=row["avg_freight"], inventory=float(inv),
              annual_hold_rate=wacc / 100 + fin.storage_spread)
d = optimise(x, mode, fin)

# Curves over a wider window than the feasible band, to show where guardrails cut.
p_grid = np.linspace(min(0.95 * x.cogs, 0.7 * x.p_last), 1.35 * x.p_last, GRID)
pi, cm2, hold = profit(p_grid, x, mode, fin)
q = demand(p_grid, x)
rng = np.random.default_rng(SEED)
draws = np.clip(rng.normal(x.eps, x.eps_se, 200), -5.0, -0.05)[:, None]
pi_d = profit(p_grid[None, :], x, mode, fin, eps=draws)[0]
q_d = demand(p_grid[None, :], x, eps=draws)
solve_ms = (time.perf_counter() - t0) * 1000

# ---------------------------------------------------------------- header + KPIs
st.title("Algorithmic Working Capital & Margin Optimization")
st.caption("Causal Double-ML elasticity → CM2-maximising price under margin-floor, volatility-band "
           "and SELIC-linked carrying-cost constraints. Olist marketplace data, decision week "
           f"{pd.Timestamp(row['week']):%d %b %Y}.")

tab_sim, tab_port, tab_ev = st.tabs(["Pricing simulator", "Portfolio", "Model evidence"])

with tab_sim:
    k1, k2, k3, k4 = st.columns(4)
    dp = d.p_star / x.p_last - 1
    k1.metric("Recommended price", brl(d.p_star), f"{dp:+.1%} vs status quo {brl(x.p_last)}")
    dcm = d.cm2_star - d.cm2_base
    k2.metric("Projected CM2 gain / week", brl(dcm, True),
              f"{dcm / abs(d.cm2_base):+.1%}" if abs(d.cm2_base) > 1e-6 else "n/a")
    ddsi = d.dsi_base - d.dsi_star
    k3.metric("DSI reduction", f"{ddsi:+.1f} days",
              f"{d.dsi_star - d.dsi_base:+.1f} days DSI ({d.dsi_base:.0f} → {d.dsi_star:.0f})",
              delta_color="inverse")
    k4.metric("Working capital freed / week", brl(d.wc_freed, True),
              "inventory → cash at COGS", delta_color="off")

    be = float(breakeven_elasticity(x.p_last, x.cogs, x.freight, fin))
    regime = "inelastic → margin expansion" if abs(x.eps) < 1 else "elastic → liquidity acceleration"
    flags = []
    if d.aged:
        flags.append(f"⚠ Aged stock ({d.weeks_of_supply:.1f} wks supply > {FIN.aging_trigger_weeks:.0f}): "
                     "floor relaxed to 1.00x COGS")
    if d.band_overridden:
        flags.append("⚠ Margin floor exceeds the +15% volatility band: floor overrides band")
    if row["sign_restricted"]:
        flags.append("ℹ SKU estimate had the wrong sign; using the category elasticity")
    st.markdown(
        f"**Causal elasticity** ε = {x.eps:.2f} (±{1.96 * x.eps_se:.2f}, 95%) · {regime} · "
        f"**Lerner breakeven** ε* = {'never (CM2 ≤ 0)' if not np.isfinite(be) else f'{be:.1f}'} "
        f"(price cuts add CM2 only if ε < ε*) · feasible band {mbrl(d.p_lo)}–{mbrl(d.p_hi)} · "
        f"P* 90% interval {mbrl(d.p_star_ci[0])}–{mbrl(d.p_star_ci[1])}")
    for f in flags:
        st.markdown(f)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("The causal sweet spot")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Weekly economic profit Π(P) = CM2 − carrying cost (R$)",
                                            "Expected weekly demand Q(P) (units)"))
        for r, (lo_band, hi_band, mid, color, name) in enumerate([
            (np.percentile(pi_d, 5, axis=0), np.percentile(pi_d, 95, axis=0), pi, BLUE, "Π(P)"),
            (np.percentile(q_d, 5, axis=0), np.percentile(q_d, 95, axis=0), q, ORANGE, "Q(P)")], start=1):
            fig.add_trace(go.Scatter(x=p_grid, y=hi_band, line=dict(width=0), hoverinfo="skip",
                                     showlegend=False), r, 1)
            fig.add_trace(go.Scatter(x=p_grid, y=lo_band, fill="tonexty", line=dict(width=0),
                                     fillcolor=f"rgba{tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)) + (0.15,)}",
                                     name="90% elasticity band", hoverinfo="skip", showlegend=r == 1), r, 1)
            fig.add_trace(go.Scatter(x=p_grid, y=mid, line=dict(color=color, width=2), name=name,
                                     hovertemplate="P=R$ %{x:.2f}<br>" + name + "=%{y:.2f}<extra></extra>"),
                          r, 1)
        for r in (1, 2):
            fig.add_vrect(x0=d.p_lo, x1=max(d.p_hi, d.p_lo + 1e-6), fillcolor=MUTED, opacity=0.08,
                          line_width=0, row=r, col=1)
            fig.add_vline(x=x.p_last, line=dict(color=MUTED, dash="dot", width=1.5), row=r, col=1)
            fig.add_vline(x=d.p_star, line=dict(color=GOOD, width=2), row=r, col=1)
        floor = (FIN.floor_liquidation if d.aged else FIN.floor_normal) * x.cogs
        fig.add_vline(x=floor, line=dict(color=BAD, dash="dash", width=1.5), row=1, col=1)
        fig.add_annotation(x=d.p_star, y=1.02, yref="paper", text=f"P* {brl(d.p_star)}",
                           showarrow=False, font=dict(color=GOOD))
        fig.add_annotation(x=x.p_last, y=0.97, yref="paper", text="status quo", showarrow=False,
                           font=dict(color=MUTED), xanchor="right")
        fig.add_annotation(x=floor, y=0.90, yref="paper", text=f"floor {floor / x.cogs:.2f}×COGS",
                           showarrow=False, font=dict(color=BAD), xanchor="left")
        fig.update_xaxes(title_text="Price (R$) · shaded = feasible band", row=2, col=1)
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                          legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("P&L bridge (R$ / week)")
        gain = d.cm2_star - d.cm2_base
        avoided = d.hold_base - d.hold_star
        wf = go.Figure(go.Waterfall(
            x=["Baseline profit", "Elasticity pricing gain", "Avoided holding cost", "Optimized net profit"],
            measure=["absolute", "relative", "relative", "total"],
            y=[d.profit_base, gain, avoided, d.profit_star],
            text=[brl(v, i in (1, 2)) for i, v in enumerate([d.profit_base, gain, avoided, d.profit_star])],
            textposition="outside", connector=dict(line=dict(color=MUTED, width=1)),
            increasing=dict(marker=dict(color=GOOD)), decreasing=dict(marker=dict(color=BAD)),
            totals=dict(marker=dict(color=BLUE))))
        wf.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                         yaxis_title="R$")
        st.plotly_chart(wf, use_container_width=True)
        st.dataframe(pd.DataFrame({
            "Status quo": [x.p_last, d.q_base, d.cm2_base, d.hold_base, d.profit_base, d.dsi_base],
            "Optimized": [d.p_star, d.q_star, d.cm2_star, d.hold_star, d.profit_star, d.dsi_star],
        }, index=["Price (R$)", "Units / week", "CM2 (R$)", "Carrying cost (R$)", "Net profit (R$)",
                  "DSI (days)"]).round(2), use_container_width=True)
    st.caption(f"Solve + curve evaluation: {solve_ms:.1f} ms · COGS {mbrl(x.cogs)} · freight {mbrl(x.freight)} · "
               f"h = {x.annual_hold_rate:.2%}/yr × COGS · forecast base demand {x.q_base:.2f} units/wk "
               "(LightGBM one-step-ahead).")

with tab_port:
    rec = data["recommendations"]
    st.subheader("Portfolio recommendations (precomputed, default SELIC carrying cost)")
    scen = st.radio("Inventory scenario", sorted(rec["scenario"].unique(), reverse=True), horizontal=True)
    md = st.radio("Strategy", ["margin", "liquidity"], horizontal=True, key="pmode")
    r = rec[(rec["scenario"] == scen) & (rec["mode"] == md)].copy()
    tot = r[["profit_base", "profit_star", "cm2_base", "cm2_star", "wc_freed"]].sum()
    a, b, c = st.columns(3)
    a.metric("Portfolio net profit / week", brl(tot["profit_star"]),
             f"{brl(tot['profit_star'] - tot['profit_base'], True)} vs status quo")
    b.metric("Portfolio CM2 / week", brl(tot["cm2_star"]), f"{brl(tot['cm2_star'] - tot['cm2_base'], True)}")
    c.metric("Working capital freed / week", brl(tot["wc_freed"], True), delta_color="off")
    r["price_change"] = r["p_star"] / r["p_last"] - 1
    agg = r.groupby("cat_en").agg(skus=("product_id", "size"), median_eps=("eps", "median"),
                                  median_price_change=("price_change", "median"),
                                  cm2_gain=("cm2_star", "sum"), profit_gain=("profit_star", "sum"))
    agg["cm2_gain"] -= r.groupby("cat_en")["cm2_base"].sum()
    agg["profit_gain"] -= r.groupby("cat_en")["profit_base"].sum()
    st.dataframe(agg.style.format({"median_eps": "{:.2f}", "median_price_change": "{:+.1%}",
                                   "cm2_gain": "R$ {:,.2f}", "profit_gain": "R$ {:,.2f}"}),
                 use_container_width=True)
    show = r[["product_id", "cat_en", "eps", "p_last", "p_star", "p_ci_lo", "p_ci_hi", "price_change",
              "cm2_base", "cm2_star", "dsi_base", "dsi_star", "aged", "band_overridden"]]
    st.dataframe(show.sort_values("price_change").style.format(
        {c: "{:.2f}" for c in ["eps", "p_last", "p_star", "p_ci_lo", "p_ci_hi", "cm2_base", "cm2_star",
                               "dsi_base", "dsi_star"]} | {"price_change": "{:+.1%}"}),
        use_container_width=True, height=380)

with tab_ev:
    st.subheader("Why Double-ML: naive regressions misprice demand")
    ec = data["elasticity_cat"]
    fig = go.Figure()
    for col, name, color in [("eps_ols_fe", "Naive within-SKU OLS", MUTED),
                             ("eps_dml", "Double-ML (causal)", BLUE)]:
        fig.add_trace(go.Bar(y=ec["cat_en"], x=ec[col], name=name, orientation="h", marker_color=color,
                             error_x=dict(type="data", array=1.96 * ec["se_dml"]) if col == "eps_dml" else None,
                             hovertemplate="%{y}: %{x:.2f}<extra>" + name + "</extra>"))
    fig.add_vline(x=-1, line=dict(color=MUTED, dash="dot"))
    fig.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Coefficient on ln P (outcome ln(Q+1)); dotted line = unit elasticity",
                      legend=dict(orientation="h", y=-0.25), bargap=0.3)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Naive OLS overstates price sensitivity because sellers discount into demand peaks "
               "(Black Friday, platform growth). Pricing off it would over-discount and leak margin.")
    st.dataframe(ec.drop(columns=["spec"]).style.format(precision=3), use_container_width=True)

    st.subheader("Forecast scorecard: statistics with their financial counterparts")
    fm = data["forecast_metrics"].set_index("model")
    st.dataframe(fm.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "WAPE": "{:.1%}", "bias_ratio": "{:+.1%}",
                                  "stockout_loss_brl": "R$ {:,.0f}", "excess_holding_brl": "R$ {:,.0f}",
                                  "safety_stock_capital_brl": "R$ {:,.0f}"}), use_container_width=True)
    st.caption("Walk-forward, refit every 4 weeks, one-step-ahead, Mar–Aug 2018. Stockout loss = unmet units × "
               "CM2; excess holding = over-forecast units × weekly carrying cost; safety-stock capital = "
               "1.645·σ(error)·COGS per SKU (95% cycle service level).")
