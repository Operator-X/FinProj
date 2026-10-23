"""Phase 6: interactive dashboard over Phases 1-5.

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

from engine.config import ARTIFACT_DIR, FIN, PATHS, SEED
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


@st.cache_data
def load_ai() -> dict[str, pd.DataFrame]:
    """Phase 5 artifacts (absent until the pipeline has produced them)."""
    names = ["prob_forecasts", "prob_metrics", "rl_eval", "review_labels", "moderation",
             "elasticity_voice"]
    return {n: pd.read_parquet(ARTIFACT_DIR / f"{n}.parquet") for n in names
            if (ARTIFACT_DIR / f"{n}.parquet").exists()}


@st.cache_resource
def load_ppo():
    """PPO policy (torch). The app never trains LightGBM, so torch is safe here."""
    from stable_baselines3 import PPO

    from engine.rl import MODEL_PATH
    return PPO.load(MODEL_PATH, device="cpu") if MODEL_PATH.exists() else None


def brl(v: float, signed: bool = False) -> str:
    """Currency with the sign first ("-R$ 3.00"), so st.metric colours deltas correctly."""
    sign = "-" if v < 0 else ("+" if signed else "")
    return f"{sign}R$ {abs(v):,.2f}"


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

tab_sim, tab_rl, tab_fc, tab_voice, tab_port, tab_ev = st.tabs(
    ["Pricing simulator", "RL agent", "GenAI forecast", "Customer voice", "Portfolio", "Model evidence"])
ai = load_ai()
# Fixed categorical slot per policy / model, never re-cycled when filters change.
POLICY_COLORS = {"status quo": MUTED, "myopic (Phase 4)": ORANGE, "CE-DP": "#1baf7a",
                 "PPO agent": BLUE, "oracle DP": "#e87ba4"}
MODEL_COLORS = {"LightGBM quantile": BLUE, "Chronos-Bolt (zero-shot)": ORANGE,
                "Chronos-2 + covariates (zero-shot)": "#1baf7a"}

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

with tab_rl:
    from engine.rl import H, SellThroughEnv, dp_policy, myopic_move, rollout, solve_dp

    N_SCEN = 40

    st.subheader("Selling a fixed lot over a 26-week season")
    st.caption("Each week the agent moves price by at most ±15% (floor and +42% cap enforced). Stock left "
               "after week 26 goes to an outlet at 50% of cost. The true elasticity is unknown: the "
               "agent sees a Bayesian posterior that updates from the sales it observes. "
               "Demo tip: bed_bath_table SKU 704ef63a with 40 weeks of supply, where planning ahead "
               "beats the one-week optimizer.")
    ppo = load_ppo()
    if ppo is None:
        st.info("No trained agent yet: run `uv run python -m engine.rl train eval`.")
    else:
        c1, c2, c3 = st.columns(3)
        wos = c1.slider("Stock on hand (weeks of supply)", 4, 52, 30)
        surprise = c2.select_slider("True elasticity vs estimate", ["much less elastic", "as estimated",
                                                                    "much more elastic"], "as estimated")
        seed = c3.number_input("Demand scenario (seed)", 0, 999, 7)
        env = SellThroughEnv(state)
        i = int(state.index[state["product_id"] == pid][0])
        lot = env.make_lot(i, wos, wacc / 100)
        shift = {"much less elastic": 1.0, "as estimated": 0.0, "much more elastic": -1.0}[surprise]
        eps_true = float(np.clip(lot.eps_hat + shift * 1.645 * lot.eps_se, -5, -0.05))
        if lot.inv0 > 400:
            st.warning("Lot too large for the exact DP benchmark at interactive speed; lower weeks of supply.")
        else:
            t0 = time.perf_counter()
            opts = {"sku": i, "wos": wos, "selic": wacc / 100, "eps_true": eps_true,
                    "u": np.random.default_rng(int(seed)).uniform(size=H)}
            pols = {"status quo": lambda e, o: 2, "myopic (Phase 4)": lambda e, o: myopic_move(e),
                    "CE-DP": dp_policy(solve_dp(lot, lot.eps_hat)),
                    "PPO agent": lambda e, o: int(ppo.predict(o, deterministic=True)[0]),
                    "oracle DP": dp_policy(solve_dp(lot, eps_true))}
            runs = {n: rollout(env, opts, f) for n, f in pols.items()}
            # Averages over many demand scenarios: one path is too noisy to rank policies.
            many = {n: [] for n in pols}
            for sd in range(N_SCEN):
                o = {**opts, "u": np.random.default_rng(1000 + sd).uniform(size=H)}
                for n, f in pols.items():
                    many[n].append(rollout(env, o, f))
            avg = {n: {k: float(np.mean([r[k] for r in rs])) for k in ("npv", "sell_through", "avg_price_ratio")}
                   for n, rs in many.items()}
            ms = (time.perf_counter() - t0) * 1000
            k = st.columns(4)
            k[0].metric("Lot", f"{lot.inv0} units", f"{brl(lot.inv0 * lot.cogs)} at cost", delta_color="off")
            k[1].metric("PPO expected season NPV", brl(avg["PPO agent"]["npv"]),
                        f"{brl(avg['PPO agent']['npv'] - avg['status quo']['npv'], True)} vs status quo")
            k[2].metric("PPO expected sell-through", f"{avg['PPO agent']['sell_through']:.0%}",
                        f"{avg['PPO agent']['sell_through'] - avg['status quo']['sell_through']:+.0%} vs status quo")
            k[3].metric("Elasticity", f"true {eps_true:.2f}", f"agent's prior {lot.eps_hat:.2f}", delta_color="off")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=("Price (R$)", "Units left in stock"))
            for n, r in runs.items():
                wk = np.arange(1, len(r["prices"]) + 1)
                dash = "dot" if n == "oracle DP" else None
                fig.add_trace(go.Scatter(x=wk, y=r["prices"], name=n, line=dict(color=POLICY_COLORS[n],
                                         width=2, dash=dash, shape="hv"), legendgroup=n), 1, 1)
                fig.add_trace(go.Scatter(x=wk, y=r["inv_path"], name=n, showlegend=False, legendgroup=n,
                                         line=dict(color=POLICY_COLORS[n], width=2, dash=dash, shape="hv")), 2, 1)
            fig.update_xaxes(title_text="Week of season", row=2, col=1)
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                              legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Charts: one illustrative demand path (pick another with the seed).")
            tbl = pd.DataFrame({n: {"Expected NPV (R$)": a["npv"], "Sell-through": a["sell_through"],
                                    "Avg price vs today": a["avg_price_ratio"],
                                    "This path NPV (R$)": runs[n]["npv"]} for n, a in avg.items()}).T
            st.dataframe(tbl.style.format({"Expected NPV (R$)": "R$ {:,.2f}", "Sell-through": "{:.0%}",
                                           "Avg price vs today": "{:.2f}x", "This path NPV (R$)": "R$ {:,.2f}"}),
                         use_container_width=True)
            st.caption(f"Expected values average {N_SCEN} demand scenarios shared by every policy (common "
                       f"random numbers). Simulated in {ms:.0f} ms. Oracle DP knows the true elasticity: an "
                       "upper bound in expectation, though a lucky single path can beat it.")
        if "rl_eval" in ai:
            from engine.rl import rl_summary
            summ = rl_summary(ai["rl_eval"])
            summ = summ[summ["policy"] != "oracle DP"]
            st.subheader("Out-of-sample evaluation: 72 SKUs × 30 seasons, SELIC 6.5%")
            fig = go.Figure()
            sq = summ[summ["policy"] == "status quo"].set_index("regime")["gap_to_oracle"]
            for n in ["myopic (Phase 4)", "CE-DP", "PPO agent"]:
                d = summ[summ["policy"] == n]
                fig.add_trace(go.Bar(x=d["regime"].astype(str), y=d["gap_to_oracle"], name=n,
                                     marker_color=POLICY_COLORS[n],
                                     hovertemplate="%{x}: R$ %{y:.1f}<extra>" + n + "</extra>"))
            fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="Mean NPV shortfall vs oracle (R$ per lot)",
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Closer to zero is better. One PPO network prices every SKU and lands within ~1% "
                       "of the exact dynamic programme; the remaining gap to the oracle is the value of "
                       "knowing the true elasticity, i.e. what a price experiment is worth. Holding today's "
                       "price all season (not shown) falls short by "
                       + ", ".join(f"R$ {-v:,.0f} ({k})" for k, v in sq.items()) + " per lot.")

with tab_fc:
    if not {"prob_forecasts", "prob_metrics"} <= ai.keys():
        st.info("Run the pipeline to produce probabilistic forecasts.")
    else:
        pf, pm = ai["prob_forecasts"], ai["prob_metrics"].set_index("model")
        st.subheader("Does a GenAI foundation model beat a trained LightGBM?")
        st.caption("Chronos models are transformers pretrained on millions of time series and used here "
                   "zero-shot. All three forecast demand quantiles one week ahead (walk-forward, Mar-Aug 2018). "
                   "Policy: stock up to the 90% quantile each week.")
        cols = st.columns(len(pm))
        for c, (name, r) in zip(cols, pm.iterrows()):
            c.markdown(f"**{name}**")
            c.metric("Weighted quantile loss", f"{r.WQL:.3f}")
            c.metric("Fill rate at q90 stock", f"{r.fill_rate:.1%}")
            c.metric("Avg inventory capital / week", brl(r.avg_capital_brl_wk))
            c.metric("Calibration: P(sales ≤ q90)", f"{r.coverage_q90:.0%}", "target 90%", delta_color="off")
        model = st.radio("Fan chart model", list(pm.index), horizontal=True)
        g = pf[(pf["product_id"] == pid) & (pf["model"] == model)].sort_values("week")
        if g.empty:
            st.info("This SKU was not in the forecast hold-out.")
        else:
            col = MODEL_COLORS[model]
            rgba = f"rgba{tuple(int(col[i:i + 2], 16) for i in (1, 3, 5)) + (0.18,)}"
            fig = go.Figure([
                go.Scatter(x=g["week"], y=g["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip"),
                go.Scatter(x=g["week"], y=g["q10"], fill="tonexty", fillcolor=rgba, line=dict(width=0),
                           name="80% interval (q10–q90)"),
                go.Scatter(x=g["week"], y=g["q50"], line=dict(color=col, width=2), name="median forecast"),
                go.Scatter(x=g["week"], y=g["units_sold"], mode="markers", name="actual units",
                           marker=dict(color=MUTED, size=8))])
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Units / week",
                              legend=dict(orientation="h", y=-0.2), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pm.style.format({"WAPE_median": "{:.1%}", "WQL": "{:.3f}", "coverage_80": "{:.0%}",
                                      "coverage_q90": "{:.0%}", "stockout_loss_brl": "R$ {:,.0f}",
                                      "excess_holding_brl": "R$ {:,.0f}", "fill_rate": "{:.1%}",
                                      "avg_capital_brl_wk": "R$ {:,.0f}", "wc_at_risk_90_brl_wk": "R$ {:,.0f}"}),
                     use_container_width=True)

with tab_voice:
    if "review_labels" not in ai:
        st.info("No review labels yet: run `uv run python -m engine.reviews`, then the pipeline.")
    else:
        from engine.reviews import THEMES

        lab = ai["review_labels"].merge(state[["product_id", "cat_en"]].drop_duplicates(), how="left",
                                        on="product_id")
        lab["cat_en"] = lab["cat_en"].fillna(lab["product_id"].map(
            data["panel"].drop_duplicates("product_id").set_index("product_id")["cat_en"]))
        st.subheader("What customers say, coded by an LLM")
        st.caption(f"{len(lab):,} Portuguese review comments coded by Claude into {len(THEMES)} themes. "
                   "Themes enter the causal model only as of the week after a review is written.")
        prev = lab.groupby("cat_en")[list(THEMES)].mean().T
        fig = go.Figure()
        cat_colors = dict(zip(sorted(prev.columns), [BLUE, ORANGE, "#1baf7a", "#eda100"]))
        for c in prev.columns:
            fig.add_trace(go.Bar(y=prev.index, x=prev[c], name=c, orientation="h", marker_color=cat_colors[c],
                                 hovertemplate="%{y}: %{x:.0%}<extra>" + c + "</extra>"))
        fig.update_layout(barmode="group", height=480, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_tickformat=".0%", xaxis_title="Share of reviews mentioning theme",
                          legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig, use_container_width=True)
        if "elasticity_voice" in ai:
            st.markdown("**Reviews are a hidden confounder.** Adding the LLM-coded review signals to the "
                        "Double-ML controls makes demand look more price-sensitive in most categories: "
                        "perception was pulling the estimates toward zero.")
            ec, ev = data["elasticity_cat"].set_index("cat_en"), ai["elasticity_voice"].set_index("cat_en")
            fig = go.Figure()
            for frame, name, color in [(ec, "Double-ML", MUTED), (ev, "Double-ML + review signals", BLUE)]:
                fig.add_trace(go.Bar(y=frame.index, x=frame["eps_dml"], name=name, orientation="h",
                                     marker_color=color, error_x=dict(type="data", array=1.96 * frame["se_dml"]),
                                     hovertemplate="%{y}: %{x:.2f}<extra>" + name + "</extra>"))
            fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Coefficient on ln P (95% CI)", legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)
        if "moderation" in ai:
            st.markdown("**Does customer perception change price sensitivity?** Change in the price "
                        "coefficient per 1 SD of each review signal (positive = less price-sensitive), "
                        "estimated on Double-ML residuals with SKU-clustered SEs.")
            st.dataframe(ai["moderation"].style.format(precision=3), use_container_width=True)
        mine = lab[lab["product_id"] == pid].sort_values("review_date", ascending=False)
        st.markdown(f"**Latest reviews for SKU {pid[:8]}** ({len(mine)} with text)")
        if len(mine):
            show = mine.head(8).assign(themes=lambda d: d[list(THEMES)].apply(
                lambda r: ", ".join(k for k in THEMES if r[k]), axis=1))
            show["review_date"] = show["review_date"].dt.date
            st.dataframe(show[["review_date", "review_score", "sentiment", "themes", "text"]],
                         use_container_width=True, hide_index=True)
