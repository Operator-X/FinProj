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
