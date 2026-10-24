"""Phase 6: the Margem server, running the real engine behind the Margem web app.

    uv run python -m engine.api          -> http://127.0.0.1:8600

Serves web/margem.html with live mode switched on, plus JSON endpoints the page
calls when the user asks for more ("Simulate", Season replay "More options", ...):

  POST /api/price            Phase 4 optimiser for one SKU under user assumptions
  POST /api/season           Phase 5A: PPO agent vs baselines on any SKU (simulated season)
  GET  /api/forecast/{sku}   Phase 5B: walk-forward quantile forecasts for one SKU
  GET  /api/reviews/{sku}    Phase 5C: latest LLM-tagged reviews for one SKU

Process rule: this process loads PyTorch (PPO), so it never imports LightGBM.
"""
from __future__ import annotations

import threading
import webbrowser
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import ARTIFACT_DIR, FIN, PATHS, ROOT, SEED
from .finance import breakeven_elasticity
from .optimize import SkuInputs, demand, optimise, profit

PORT = 8600
app = FastAPI(title="Margem", docs_url="/api/docs")


@lru_cache(maxsize=1)
def _state() -> pd.DataFrame:
    s = pd.read_parquet(PATHS.sku_state)
    return s.assign(code=s["product_id"].str[:8]).set_index("code")


def _row(sku: str) -> pd.Series:
    st = _state()
    if sku not in st.index:
        raise HTTPException(404, f"Unknown product code {sku!r}")
    return st.loc[sku]


@lru_cache(maxsize=1)
def _ppo():
    from stable_baselines3 import PPO

    from .rl import MODEL_PATH
    return PPO.load(MODEL_PATH, device="cpu") if MODEL_PATH.exists() else None


def _py(o):
    """Recursively convert NumPy scalars/arrays into JSON-safe Python values."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    page = (ROOT / "web" / "margem.html").read_text()
    return page.replace("<script>\nconst D = ", "<script>window.MARGEM_LIVE = true;</script>\n<script>\nconst D = ", 1)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "skus": len(_state()), "rl": _ppo() is not None}


class PriceReq(BaseModel):
    sku: str
    inventory: float | None = Field(None, ge=0, le=5000, description="units on hand; default 4 weeks of supply")
    wacc: float = Field(0.065, ge=0.0, le=0.4, description="annual cost of capital (decimal)")
    goal: str = Field("margin", pattern="^(margin|liquidity)$")
    fee: float = Field(FIN.platform_fee, ge=0, le=0.5)
    freight_share: float = Field(FIN.freight_share, ge=0, le=1)


@app.post("/api/price")
def price(req: PriceReq) -> dict:
    """One SKU's constrained optimum, its uncertainty band and the curves to draw."""
    r = _row(req.sku)
    fin = replace(FIN, platform_fee=req.fee, freight_share=req.freight_share)
    inv = float(np.ceil(4 * r.q_base)) if req.inventory is None else req.inventory
    x = SkuInputs(p_last=r.p_last, q_base=r.q_base, eps=r.eps, eps_se=r.se, cogs=r.cogs,
                  freight=r.avg_freight, inventory=inv, annual_hold_rate=req.wacc + fin.storage_spread)
    d = optimise(x, req.goal, fin)
    grid = np.linspace(min(0.95 * x.cogs, 0.7 * x.p_last), 1.35 * x.p_last, 121)
    pi = profit(grid, x, req.goal, fin)[0]
    draws = np.clip(np.random.default_rng(SEED).normal(x.eps, x.eps_se, 200), -5.0, -0.05)[:, None]
    pi_d = profit(grid[None, :], x, req.goal, fin, eps=draws)[0]
    be = float(breakeven_elasticity(x.p_last, x.cogs, x.freight, fin))
    return _py({**d.as_dict(),
            "inventory": inv, "p_last": float(x.p_last), "cogs": float(x.cogs), "eps": float(x.eps),
            "eps_se": float(x.eps_se), "breakeven": be if np.isfinite(be) else None,
            "floor": float((FIN.floor_liquidation if d.aged else FIN.floor_normal) * x.cogs),
            "sign_restricted": bool(r.sign_restricted),
            "curve": {"p": grid.round(3).tolist(), "pi": pi.round(3).tolist(),
                      "lo": np.percentile(pi_d, 5, axis=0).round(3).tolist(),
                      "hi": np.percentile(pi_d, 95, axis=0).round(3).tolist(),
                      "q": demand(grid, x).round(4).tolist()}})


class PortfolioReq(BaseModel):
    wacc: float = Field(0.065, ge=0.0, le=0.4)
    wos: float = Field(4, ge=1, le=60, description="weeks of supply on hand for every product")
    fee: float = Field(FIN.platform_fee, ge=0, le=0.5)
    freight_share: float = Field(FIN.freight_share, ge=0, le=1)
    goal: str = Field("margin", pattern="^(margin|liquidity)$")


def portfolio_solve(req: PortfolioReq, st: pd.DataFrame | None = None) -> dict:
    """Phase 4 optimiser over the whole catalogue under one scenario (Scenario lab)."""
    st = _state() if st is None else st
    fin = replace(FIN, platform_fee=req.fee, freight_share=req.freight_share)
    out = dict(up=0, cut=0, hold=0, base=0.0, opt=0.0, baseCm2=0.0, optCm2=0.0, cap=0.0, carry=0.0, aged=0, prices={})
    for code, r in st.iterrows():
        inv = float(np.ceil(req.wos * r.q_base))
        x = SkuInputs(p_last=r.p_last, q_base=r.q_base, eps=r.eps, eps_se=r.se, cogs=r.cogs, freight=r.avg_freight,
                      inventory=inv, annual_hold_rate=req.wacc + fin.storage_spread)
        d = optimise(x, req.goal, fin, n_draws=5)
        out["prices"][code] = round(d.p_star, 4)
        out["up" if d.p_star > r.p_last * 1.001 else "cut" if d.p_star < r.p_last * 0.999 else "hold"] += 1
        out["base"] += d.profit_base; out["opt"] += d.profit_star
        out["baseCm2"] += d.cm2_base; out["optCm2"] += d.cm2_star
        out["cap"] += inv * r.cogs; out["carry"] += x.annual_hold_rate / 52 * r.cogs * inv
        out["aged"] += int(d.aged)
    return _py(out)


@app.post("/api/portfolio")
def portfolio(req: PortfolioReq) -> dict:
    return portfolio_solve(req)


class SeasonReq(BaseModel):
    sku: str
    wos: float = Field(30, ge=2, le=60, description="weeks of supply on hand")
    wacc: float = Field(0.065, ge=0.0, le=0.4)
    surprise: int = Field(0, ge=-1, le=1, description="-1 true demand much more elastic, +1 much less")
    seed: int = Field(7, ge=0, le=9999)
    scenarios: int = Field(40, ge=5, le=200)


def _pad(xs: list, n: int) -> list:
    """A lot that sells out early stops simulating; hold the last value to a full season."""
    return xs + [xs[-1]] * (n - len(xs)) if xs else xs


@app.post("/api/season")
def season(req: SeasonReq) -> dict:
    """Race the PPO agent against baselines on one lot: one shown path + scenario averages."""
    from .rl import H, SellThroughEnv, dp_policy, myopic_move, rollout, solve_dp

    model = _ppo()
    if model is None:
        raise HTTPException(503, "No trained agent: run `uv run python -m engine.rl train`.")
    st = _state().reset_index(drop=True)
    i = int(st.index[st["product_id"].str.startswith(req.sku)][0])
    env = SellThroughEnv(st)
    lot = env.make_lot(i, req.wos, req.wacc)
    if lot.inv0 > 400:
        raise HTTPException(422, f"Lot of {lot.inv0} units is too large for the exact benchmark; lower weeks of supply.")
    eps_true = float(np.clip(lot.eps_hat + req.surprise * 1.645 * lot.eps_se, -5, -0.05))
    pols = {"hold": lambda e, o: 2, "weekly": lambda e, o: myopic_move(e),
            "planner": dp_policy(solve_dp(lot, lot.eps_hat)),
            "agent": lambda e, o: int(model.predict(o, deterministic=True)[0]),
            "oracle": dp_policy(solve_dp(lot, eps_true))}
    base = {"sku": i, "wos": req.wos, "selic": req.wacc, "eps_true": eps_true}
    shown = {k: rollout(env, {**base, "u": np.random.default_rng(req.seed).uniform(size=H)}, f) for k, f in pols.items()}
    avg: dict[str, list] = {k: [] for k in pols}
    for sd in range(req.scenarios):
        o = {**base, "u": np.random.default_rng(1000 + sd).uniform(size=H)}
        for k, f in pols.items():
            avg[k].append(rollout(env, o, f))
    return {
        "sku": req.sku, "lot": lot.inv0, "p0": round(lot.p0, 2), "cogs": round(lot.cogs, 2),
        "eps": round(lot.eps_hat, 2), "eps_true": round(eps_true, 2),
        "paths": {k: {"price": _pad([round(p, 2) for p in v["prices"]], H), "inv": _pad([int(n) for n in v["inv_path"]], H),
                      "npv": round(float(v["npv"]), 1), "sold": round(float(v["sell_through"]), 3)}
                  for k, v in shown.items()},
        "expected": {k: {"npv": round(float(np.mean([r["npv"] for r in rs])), 1),
                         "sold": round(float(np.mean([r["sell_through"] for r in rs])), 3),
                         "price": round(float(np.mean([r["avg_price_ratio"] for r in rs])), 3)}
                     for k, rs in avg.items()},
        "scenarios": req.scenarios,
    }


@lru_cache(maxsize=1)
def _prob() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACT_DIR / "prob_forecasts.parquet")


@app.get("/api/forecast/{sku}")
def forecast(sku: str) -> dict:
    g = _prob()[_prob()["product_id"].str.startswith(sku)].sort_values("week")
    return {m: [{"w": w.strftime("%Y-%m-%d"), "y": int(y), "q10": round(a, 2), "q50": round(b, 2), "q90": round(c, 2)}
                for w, y, a, b, c in zip(x.week, x.units_sold, x.q10, x.q50, x.q90)]
            for m, x in g.groupby("model")}


@lru_cache(maxsize=1)
def _reviews() -> pd.DataFrame:
    p = ARTIFACT_DIR / "review_labels.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@app.get("/api/reviews/{sku}")
def reviews(sku: str) -> list[dict]:
    from .reviews import THEMES

    lab = _reviews()
    if lab.empty:
        return []
    mine = lab[lab["product_id"].str.startswith(sku)].sort_values("review_date", ascending=False).head(8)
    return [{"date": r.review_date.strftime("%d %b %Y"), "stars": int(r.review_score), "sentiment": int(r.sentiment),
             "themes": [k for k in THEMES if getattr(r, k)], "text": r.text} for r in mine.itertuples()]


def main() -> None:
    import uvicorn

    url = f"http://127.0.0.1:{PORT}"
    print(f"Margem live at {url}  (Ctrl+C to stop)")
    _state(); _ppo()                                  # warm caches before the first click
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
