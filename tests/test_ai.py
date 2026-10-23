"""Invariants for the AI extensions: RL environment, DP, review features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.config import FIN
from engine.rl import H, MOVES, STEP, Lot, SellThroughEnv, myopic_move, rollout, solve_dp, dp_policy
from engine.reviews import THEMES, weekly_voice_features

SKUS = pd.DataFrame({"product_id": ["a", "b"], "cat_en": ["x", "y"], "p_last": [100.0, 50.0],
                     "q_base": [3.0, 0.5], "eps": [-1.2, -0.4], "se": [0.4, 0.2],
                     "cogs": [58.0, 40.0], "avg_freight": [15.0, 12.0]})


@pytest.mark.parametrize("sku", [0, 1])
@pytest.mark.parametrize("wos", [4.0, 30.0])
def test_env_respects_guardrails(sku: int, wos: float):
    env = SellThroughEnv(SKUS)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(options={"sku": sku, "wos": wos, "selic": 0.1})
    last_price, inv_before = None, env.inv
    while True:
        obs, _, done, _, info = env.step(int(rng.integers(len(MOVES))))
        p, lot = info["price"], env.lot
        aged = inv_before / lot.q_base > FIN.aging_trigger_weeks
        floor = (FIN.floor_liquidation if aged else FIN.floor_normal) * lot.cogs
        # Floor always holds up to lattice rounding upward; cap at P0 * STEP^5.
        assert p >= floor - 1e-9 or np.isclose(p, lot.p0 * STEP ** -6)
        assert p <= lot.p0 * STEP ** 5 + 1e-9
        if last_price is not None and p > floor * STEP:  # band binds unless floor forces a jump
            assert p / last_price <= 1.15 + 1e-9
        assert np.all(np.isfinite(obs))
        last_price, inv_before = p, info["inv"]
        if done:
            break


def test_dp_dominates_myopic_in_expectation():
    """The exact DP must earn at least the myopic policy's expected NPV."""
    env = SellThroughEnv(SKUS)
    lot = env.make_lot(0, 30.0, 0.1)
    table = solve_dp(lot, lot.eps_hat)
    rng = np.random.default_rng(1)
    dp, my = [], []
    for _ in range(300):
        opts = {"sku": 0, "wos": 30.0, "selic": 0.1, "eps_true": lot.eps_hat, "u": rng.uniform(size=H)}
        dp.append(rollout(env, opts, dp_policy(table))["npv"])
        my.append(rollout(env, opts, lambda e, o: myopic_move(e))["npv"])
    assert np.mean(dp) >= np.mean(my) - 2 * np.std(np.subtract(dp, my)) / np.sqrt(300)


def test_unsold_stock_books_a_writedown():
    lot = Lot(p0=100, q_base=0.01, eps_hat=-1, eps_se=0.1, cogs=60, freight=0, inv0=10, selic=0.0)
    table = solve_dp(lot, -5.0)  # demand ~0: nothing sells, the loss is the write-down
    env = SellThroughEnv(SKUS)
    env.make_lot = lambda *a, **k: lot  # noqa: E731
    r = rollout(env, {"sku": 0, "wos": 1, "selic": 0.0, "eps_true": -5.0, "u": np.zeros(H)},
                dp_policy(table))
    assert r["npv"] < 0 and r["npv"] >= -(10 * 60 * 0.5) - 1e-6 - 10 * FIN.storage_spread * 60 * H / 52


def test_review_features_are_leakage_safe():
    panel = pd.DataFrame({"product_id": "a", "week": pd.date_range("2018-01-01", periods=4, freq="7D")})
    labelled = pd.DataFrame({"product_id": ["a"], "review_date": [pd.Timestamp("2018-01-10")],
                             "sentiment": [-1], **{k: [int(k == "price_high")] for k in THEMES}})
    out = weekly_voice_features(panel, labelled, prior_n=0.0).set_index("week")
    # Written in the week of 8 Jan -> invisible that week, visible from 15 Jan.
    assert out.loc["2018-01-08", "voice_n"] == 0
    assert out.loc["2018-01-15", "voice_n"] == 1
    assert out.loc["2018-01-15", "voice_price_high"] == 1.0


def test_web_app_build_is_consistent():
    """Phase 6: the page has one data slot and every shipped price respects the guardrails."""
    import json

    from engine.config import FIN, ROOT
    web = ROOT / "web"
    tpl = (web / "template.html").read_text()
    assert tpl.count("__DATA__") == 1
    if not (web / "data.json").exists():
        pytest.skip("run engine.webexport first")
    d = json.loads((web / "data.json").read_text())
    assert {"skus", "weekly", "prob", "point", "rl", "dml", "replay"} <= d.keys()
    for s in d["skus"]:
        floor = FIN.floor_normal * s["cogs"]
        assert s["pstar"] >= min(floor, (1 + FIN.band) * s["price"]) - 0.02
        assert s["pstar"] <= max((1 + FIN.band) * s["price"], floor) + 0.02
    assert len(d["replay"]["paths"]["agent"]["price"]) == H


SCENARIOS = [dict(selic=6.5, wos=4, fee=10, fr=100, goal="profit"),
             dict(selic=15.0, wos=20, fee=10, fr=100, goal="cash"),
             dict(selic=14.25, wos=30, fee=5, fr=0, goal="cash"),
             dict(selic=2.0, wos=9, fee=20, fr=50, goal="profit")]


def test_lab_solver_matches_python_optimiser():
    """Phase 6: the Scenario lab's in-browser solver is an exact port of engine.optimize."""
    import json
    import math
    import shutil
    import subprocess
    from dataclasses import replace

    from engine.config import ROOT
    from engine.optimize import SkuInputs, optimise
    if shutil.which("node") is None or not (ROOT / "web" / "data.json").exists():
        pytest.skip("needs node and web/data.json")
    tpl = (ROOT / "web" / "template.html").read_text()
    js = tpl[tpl.index("/* LAB-SOLVER-START"):tpl.index("/* LAB-SOLVER-END */")]
    data = json.loads((ROOT / "web" / "data.json").read_text())
    script = f"const D = {json.dumps({'skus': data['skus']})};\n{js}\nconsole.log(JSON.stringify({json.dumps(SCENARIOS)}.map(solveLab)));"
    js_out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    for p, j in zip(SCENARIOS, js_out):
        fin = replace(FIN, platform_fee=p["fee"] / 100, freight_share=p["fr"] / 100)
        goal = "liquidity" if p["goal"] == "cash" else "margin"
        opt_total = 0.0
        for s in data["skus"]:
            x = SkuInputs(p_last=s["price"], q_base=s["q"], eps=s["eps"], eps_se=s["se"], cogs=s["cogs"],
                          freight=s["fr"], inventory=float(math.ceil(p["wos"] * s["q"])),
                          annual_hold_rate=p["selic"] / 100 + fin.storage_spread)
            d = optimise(x, goal, fin, n_draws=5)
            assert abs(d.p_star - j["prices"][s["id"]]) < 0.01, (p, s["id"], d.p_star, j["prices"][s["id"]])
            opt_total += d.profit_star
        assert abs(opt_total - j["opt"]) < 1.0
