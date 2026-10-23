"""Phase 6: build the Margem web app (web/margem.html) from pipeline artifacts.

The page is a single self-contained HTML file: `web/template.html` holds the
design and logic, and this module injects the data it needs as JSON. Nothing in
the page refits a model; every number is read from Phases 1-5.

Run:  uv run python -m engine.webexport      (the pipeline also calls it last)

The season replay needs the PPO policy (PyTorch), which must not share a process
with LightGBM on macOS, so it is computed in a child process.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, PATHS, RAW_DIR, ROOT, SEED, WINDOW_END

WEB_DIR = ROOT / "web"
REPLAY_SKU, REPLAY_WOS, REPLAY_SEEDS = "704ef63a", 40, 60
NOUN = {"watches_gifts": "Watch", "bed_bath_table": "Bed & bath set",
        "telephony": "Phone accessory", "computers_accessories": "Computer accessory"}


def _product_name(cat: str, weight_g: float) -> str:
    """Olist has no product names: build a readable one from category and weight."""
    size = f"{weight_g / 1000:.1f} kg" if weight_g >= 1000 else f"{weight_g:.0f} g"
    return f"{NOUN[cat]} · {size}"


def season_replay() -> dict:
    """Policy paths for the showcase lot, in a separate (PyTorch) process.

    Picks the demand scenario whose agent-vs-hold gap is closest to the mean
    over REPLAY_SEEDS scenarios, so the replay is representative, not lucky.
    """
    code = f"""
import torch, json, numpy as np, pandas as pd
from stable_baselines3 import PPO
from engine.rl import SellThroughEnv, rollout, myopic_move, H, MODEL_PATH
skus = pd.read_parquet({str(PATHS.sku_state)!r}); env = SellThroughEnv(skus)
model = PPO.load(MODEL_PATH, device="cpu")
i = int(skus.index[skus.product_id.str.startswith({REPLAY_SKU!r})][0]); lot = env.make_lot(i, {REPLAY_WOS}, 0.065)
pols = {{"hold": lambda e, o: 2, "weekly": lambda e, o: myopic_move(e),
        "agent": lambda e, o: int(model.predict(o, deterministic=True)[0])}}
runs = []
for sd in range({REPLAY_SEEDS}):
    o = {{"sku": i, "wos": {REPLAY_WOS}, "selic": 0.065, "eps_true": lot.eps_hat,
         "u": np.random.default_rng(1000 + sd).uniform(size=H)}}
    r = {{k: rollout(env, o, f) for k, f in pols.items()}}
    runs.append((r["agent"]["npv"] - r["hold"]["npv"], r))
mean = float(np.mean([g for g, _ in runs])); r = min(runs, key=lambda x: abs(x[0] - mean))[1]
print(json.dumps({{"sku": {REPLAY_SKU!r}, "lot": lot.inv0, "p0": round(lot.p0, 2), "cogs": round(lot.cogs, 2),
    "eps": round(lot.eps_hat, 2), "mean_gap": round(mean, 1),
    "paths": {{k: {{"price": [round(x, 2) for x in v["prices"]], "inv": [int(x) for x in v["inv_path"]],
                  "npv": round(float(v["npv"]), 1), "sold": round(float(v["sell_through"]), 3)}} for k, v in r.items()}}}}))
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


SCENARIO_SEEDS = [7] + list(range(1000, 1040))   # shown season + 40 averaged scenarios (as engine/api.py)


def ppo_weights() -> dict:
    """The trained PPO policy network (12-128-128-5, tanh) for in-browser inference.

    Loaded in a child process: PyTorch must not share a process with LightGBM.
    """
    code = """
import torch, json
from stable_baselines3 import PPO
from engine.rl import MODEL_PATH
sd = PPO.load(MODEL_PATH, device="cpu").policy.state_dict()
keys = {"w1": "mlp_extractor.policy_net.0.weight", "b1": "mlp_extractor.policy_net.0.bias",
        "w2": "mlp_extractor.policy_net.2.weight", "b2": "mlp_extractor.policy_net.2.bias",
        "w3": "action_net.weight", "b3": "action_net.bias"}
print(json.dumps({k: [round(float(v), 7) for v in sd[n].flatten().tolist()] for k, n in keys.items()}))
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def scenario_uniforms() -> dict:
    """The exact uniform draws behind each demand scenario, so browser seasons equal Python's."""
    from .rl import H
    return {str(sd): [round(float(u), 12) for u in np.random.default_rng(sd).uniform(size=H)] for sd in SCENARIO_SEEDS}


def forecast_series(codes: set[str]) -> dict:
    """Walk-forward quantile forecasts per live SKU and model: [week, actual, q10, q50, q90]."""
    pf = pd.read_parquet(ARTIFACT_DIR / "prob_forecasts.parquet")
    pf = pf[pf["product_id"].str[:8].isin(codes)].sort_values("week")
    return {sku: {m: [[w.strftime("%Y-%m-%d"), int(y), round(a, 2), round(b, 2), round(c, 2)]
                      for w, y, a, b, c in zip(x.week, x.units_sold, x.q10, x.q50, x.q90)]
                  for m, x in g.groupby("model")}
            for sku, g in pf.groupby(pf["product_id"].str[:8])}


def latest_reviews(codes: set[str], n: int = 8) -> dict:
    """Latest LLM-tagged reviews per live SKU."""
    p = ARTIFACT_DIR / "review_labels.parquet"
    if not p.exists():
        return {}
    from .reviews import THEMES
    lab = pd.read_parquet(p)
    lab = lab[lab["product_id"].str[:8].isin(codes)].sort_values("review_date", ascending=False)
    return {sku: [{"date": r.review_date.strftime("%d %b %Y"), "stars": int(r.review_score), "sentiment": int(r.sentiment),
                   "themes": [k for k in THEMES if getattr(r, k)], "text": r.text} for r in g.head(n).itertuples()]
            for sku, g in lab.groupby(lab["product_id"].str[:8])}


def _dml_table() -> list[dict]:
    """Phase 3 estimates with CIs, both naive estimators and the sales-weeks robustness check."""
    cat = pd.read_parquet(PATHS.elasticity_cat)[["cat_en", "eps_dml", "se_dml", "ci_lo", "ci_hi", "eps_ols_fe",
                                                  "eps_ols_pooled", "n_obs", "n_skus"]]
    rob_p = ARTIFACT_DIR / "elasticity_robust.parquet"
    if rob_p.exists():
        rob = pd.read_parquet(rob_p)[["cat_en", "eps_dml", "se_dml"]].rename(columns={"eps_dml": "eps_rob", "se_dml": "se_rob"})
        cat = cat.merge(rob, on="cat_en", how="left")
    return cat.round(4).to_dict("records")


def _scenarios() -> list[dict]:
    """Batch Phase 4 portfolio totals: 2 stock scenarios x 2 goals (recommendations artifact)."""
    rec = pd.read_parquet(PATHS.recommendations)
    rec["up"] = rec["p_star"] > rec["p_last"] * 1.001
    rec["down"] = rec["p_star"] < rec["p_last"] * 0.999
    g = rec.groupby(["scenario", "mode"]).agg(
        skus=("product_id", "size"), up=("up", "sum"), down=("down", "sum"),
        cm2_base=("cm2_base", "sum"), cm2_star=("cm2_star", "sum"), profit_base=("profit_base", "sum"),
        profit_star=("profit_star", "sum"), dsi_base=("dsi_base", "median"), dsi_star=("dsi_star", "median"),
        wc_freed=("wc_freed", "sum")).reset_index()
    return g.round(2).to_dict("records")


def voice_summary(st: pd.DataFrame) -> dict | None:
    """Phase 5C headline numbers for the Evidence page (static, works offline)."""
    lab_p, ev_p = ARTIFACT_DIR / "review_labels.parquet", ARTIFACT_DIR / "elasticity_voice.parquet"
    if not (lab_p.exists() and ev_p.exists()):
        return None
    from .reviews import THEMES

    lab = pd.read_parquet(lab_p)
    cat = pd.read_parquet(PATHS.panel).drop_duplicates("product_id").set_index("product_id")["cat_en"]
    lab["cat_en"] = lab["product_id"].map(cat)
    stars = np.select([lab["review_score"] <= 2, lab["review_score"] >= 4], [-1, 1], 0)
    e0 = float(pd.read_parquet(PATHS.elasticity_cat).set_index("cat_en").loc["ALL", "eps_dml"])
    e1 = float(pd.read_parquet(ev_p).set_index("cat_en").loc["ALL", "eps_dml"])
    prev = lab.groupby("cat_en")[list(THEMES)].mean().round(4)
    c0 = pd.read_parquet(PATHS.elasticity_cat).set_index("cat_en")
    c1 = pd.read_parquet(ev_p).set_index("cat_en")
    bycat = [dict(cat=c, eps0=round(float(c0.loc[c, "eps_dml"]), 3), se0=round(float(c0.loc[c, "se_dml"]), 3),
                  eps1=round(float(c1.loc[c, "eps_dml"]), 3), se1=round(float(c1.loc[c, "se_dml"]), 3)) for c in c0.index]
    mod_p = ARTIFACT_DIR / "moderation.parquet"
    mod = pd.read_parquet(mod_p).round(4).to_dict("records") if mod_p.exists() else []
    return dict(n=int(len(lab)), agree=round(float((lab["sentiment"] == stars)[stars != 0].mean()), 4),
                eps0=round(e0, 3), eps1=round(e1, 3), shift=round(e1 / e0 - 1, 4),
                prev={c: prev.loc[c].to_dict() for c in prev.index}, bycat=bycat, moderation=mod)


def build_data() -> dict:
    """Everything the page reads, as plain JSON-serialisable values."""
    from .rl import rl_summary

    st = pd.read_parquet(PATHS.sku_state)
    rec = pd.read_parquet(PATHS.recommendations)
    pan = pd.read_parquet(PATHS.panel)
    prod = pd.read_csv(RAW_DIR / "olist_products_dataset.csv").set_index("product_id")
    r = rec[(rec["scenario"] == "normal (4 wks supply)") & (rec["mode"] == "margin")].set_index("product_id")

    skus = []
    for _, s in st.iterrows():
        x, hist = r.loc[s.product_id], pan[pan.product_id == s.product_id].sort_values("week").tail(26)
        skus.append(dict(
            id=s.product_id[:8], name=_product_name(s.cat_en, prod.loc[s.product_id, "product_weight_g"]),
            cat=s.cat_en, price=round(float(s.p_last), 2), pstar=round(float(x.p_star), 2),
            cm2=round(float(x.cm2_base), 2), cm2s=round(float(x.cm2_star), 2), eps=round(float(s.eps), 4), se=round(float(s.se), 4),
            be=None if not np.isfinite(s.eps_breakeven) else round(float(s.eps_breakeven), 1),
            q=round(float(s.q_base), 4), cogs=round(float(s.cogs), 4), fr=round(float(s.avg_freight), 4),
            regime=str(s.regime), signfix=bool(s.sign_restricted),
            inv=int(x.inventory), cap=round(float(x.inventory * s.cogs), 2),
            margin=round(float((s.p_last * 0.9 - s.cogs - s.avg_freight) / s.p_last), 3),
            review=round(float(s.avg_review_score), 2), hist=[int(v) for v in hist.units_sold],
            band=bool(x.band_overridden)))

    live = pan[pan.product_id.isin(set(st.product_id))]
    wk = (live.assign(c=live.units_sold * live.cm2_unit).groupby("week")
          .agg(cm2=("c", "sum"), u=("units_sold", "sum")).tail(52).reset_index())
    rl = rl_summary(pd.read_parquet(ARTIFACT_DIR / "rl_eval.parquet"))
    return dict(
        week=WINDOW_END, selic=float(st["selic"].iloc[0]), skus=skus,
        weekly=[dict(w=d.week.strftime("%Y-%m-%d"), cm2=round(float(d.cm2), 1), u=int(d.u)) for d in wk.itertuples()],
        prob=pd.read_parquet(ARTIFACT_DIR / "prob_metrics.parquet").round(4).to_dict("records"),
        point=pd.read_parquet(PATHS.forecast_metrics).round(4).to_dict("records"),
        rl=rl[["regime", "policy", "npv", "gap_to_oracle", "sell_through"]]
            .assign(regime=lambda d: d.regime.astype(str)).round(2).to_dict("records"),
        dml=_dml_table(),
        scenarios=_scenarios(),
        replay=season_replay(),
        ppo=ppo_weights(), uni=scenario_uniforms(),
        # The optimiser's 200 elasticity draws: eps + se * z with z from the same seeded stream as Python.
        z200=[round(float(v), 12) for v in np.random.default_rng(SEED).standard_normal(200)],
        fc=forecast_series({s["id"] for s in skus}), rv=latest_reviews({s["id"] for s in skus}),
        voice=voice_summary(st),
    )


SITE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Margem: a pricing copilot for Brazilian marketplace sellers. Causal pricing, AI season planning and working capital, with Capi the capybara.">
<meta property="og:title" content="Margem · notes in the margin">
<meta property="og:description" content="Causal Double-ML pricing, an RL season planner and working-capital analytics on real Olist data.">
<meta name="theme-color" content="#0b0c0e">
<link rel="icon" type="image/png" href="favicon.png">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
"""


def build_site(page: str) -> None:
    """Standalone website (web/site/) for public hosting: full document + Capi favicon."""
    import shutil

    site = WEB_DIR / "site"
    site.mkdir(exist_ok=True)
    split = page.index('<div class="shell">')
    (site / "index.html").write_text(SITE_HEAD + page[:split] + "</head>\n<body>\n" + page[split:] + "\n</body>\n</html>\n")
    icon = ROOT / "brand" / "capi" / "for-dark-backgrounds" / "capi-logo.png"
    if icon.exists():
        shutil.copy(icon, site / "favicon.png")
    (site / ".nojekyll").write_text("")


def build() -> None:
    """Write web/data.json and the self-contained web/margem.html."""
    data = build_data()
    payload = json.dumps(data, separators=(",", ":"))
    (WEB_DIR / "data.json").write_text(payload)
    template = (WEB_DIR / "template.html").read_text()
    assert template.count("__DATA__") == 1, "template must contain exactly one __DATA__ slot"
    page = template.replace("__DATA__", payload)
    (WEB_DIR / "margem.html").write_text(page)
    build_site(page)
    print(f"web app -> {WEB_DIR / 'margem.html'} ({len(payload) / 1024:.0f} KB data)")


if __name__ == "__main__":
    build()
