"""Invariant tests: Phase 0 guardrails must hold for any input."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.config import FIN, PATHS
from engine.finance import breakeven_elasticity, unit_cm2, weekly_holding_cost
from engine.forecast import add_forecast_features
from engine.optimize import SkuInputs, optimise, profit

BASE = dict(p_last=100.0, q_base=5.0, eps=-1.5, eps_se=0.3, cogs=58.0, freight=15.0,
            inventory=20.0, annual_hold_rate=0.125)


def test_unit_economics():
    assert unit_cm2(100.0, 58.0, 15.0) == pytest.approx(100 * 0.9 - 58 - 15)
    assert weekly_holding_cost(52.0, 0.13) == pytest.approx(0.13)


@pytest.mark.parametrize("eps", [-0.3, -1.0, -2.5, -5.0])
@pytest.mark.parametrize("mode", ["margin", "liquidity"])
@pytest.mark.parametrize("inventory", [5.0, 20.0, 200.0])
def test_guardrails_hold(eps: float, mode: str, inventory: float):
    x = SkuInputs(**{**BASE, "eps": eps, "inventory": inventory})
    d = optimise(x, mode)
    floor = (FIN.floor_liquidation if d.aged else FIN.floor_normal) * x.cogs
    assert d.p_star >= floor - 1e-9
    if not d.band_overridden:
        assert (1 - FIN.band) * x.p_last - 1e-9 <= d.p_star <= (1 + FIN.band) * x.p_last + 1e-9
    assert d.profit_star >= d.profit_base - 1e-6  # status quo is always feasible here
    assert d.aged == (inventory / x.q_base > FIN.aging_trigger_weeks)


def test_floor_overrides_band_for_underwater_sku():
    # Price far below 1.15 x COGS: the loss-prevention floor must win.
    x = SkuInputs(**{**BASE, "p_last": 50.0, "cogs": 58.0})
    d = optimise(x, "margin")
    assert d.band_overridden and d.p_star == pytest.approx(1.15 * 58.0)


def test_lerner_breakeven_matches_gradient():
    x = SkuInputs(**{**BASE, "inventory": 1e6, "annual_hold_rate": 0.0})
    be = float(breakeven_elasticity(x.p_last, x.cogs, x.freight))
    for eps, sign in [(be + 0.5, 1), (be - 0.5, -1)]:
        xi = SkuInputs(**{**BASE, "eps": eps, "inventory": 1e6, "annual_hold_rate": 0.0})
        lo, hi = profit(np.array([99.9, 100.1]), xi, "margin")[0]
        assert np.sign(hi - lo) == sign


def test_forecast_features_do_not_leak_current_sales():
    panel = pd.DataFrame({
        "product_id": ["a"] * 4, "category": ["c"] * 4,
        "week": pd.date_range("2018-01-01", periods=4, freq="7D"),
        "units_sold": [3.0, 0.0, 5.0, 0.0], "avg_price": [10.0, 10.0, 12.0, 12.0],
        "seller_count": [1, 1, 2, 2], "delivery_days": [5.0, 5.0, 7.0, 7.0],
        "cat_med_price": [9.0, 9.0, 9.5, 9.5], "ev_black_friday": 0, "ev_christmas": 0})
    f = add_forecast_features(panel)
    # Week 3's price change is only visible to the forecaster from week 4 onward.
    assert f["posted_price"].tolist() == [10.0, 10.0, 10.0, 12.0]
    assert f["seller_count_lag1"].iloc[2] == 1


API_CHECK = """
import time
from fastapi.testclient import TestClient
from engine.api import app
c = TestClient(app)
assert c.get("/api/health").json()["ok"]
page = c.get("/").text
assert "window.MARGEM_LIVE = true" in page and page.count("const D = ") == 1
t0 = time.perf_counter()
r = c.post("/api/price", json={"sku": "704ef63a", "wacc": 0.12, "goal": "liquidity"}).json()
ms = (time.perf_counter() - t0) * 1000
assert r["p_lo"] - 1e-6 <= r["p_star"] <= r["p_hi"] + 1e-6 and r["p_star"] >= r["floor"] - 1e-6
assert len(r["curve"]["p"]) == len(r["curve"]["pi"]) == 121
s = c.post("/api/season", json={"sku": "704ef63a", "wos": 20, "scenarios": 5}).json()
assert set(s["paths"]) >= {"hold", "weekly", "agent"} and all(len(v["price"]) == len(v["inv"]) == 26 for v in s["paths"].values())
assert c.post("/api/price", json={"sku": "nope"}).status_code == 404
pf = c.post("/api/portfolio", json={"wacc": 0.15, "wos": 20, "goal": "liquidity"}).json()
assert pf["up"] + pf["cut"] + pf["hold"] == 72 and len(pf["prices"]) == 72
print(ms)
"""


@pytest.mark.skipif(not PATHS.sku_state.exists(), reason="run engine.pipeline first")
def test_margem_server_endpoints():
    # Separate process: the server loads torch (PPO), which must not share a
    # process with the LightGBM already imported by this test session.
    import subprocess
    import sys

    from engine.config import ROOT
    out = subprocess.run([sys.executable, "-c", API_CHECK], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert float(out.stdout.strip().splitlines()[-1]) < 200
