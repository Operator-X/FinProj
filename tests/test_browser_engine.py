"""Parity: the in-browser engine (web/template.html, ENGINE block) must reproduce the
Python engine (engine/optimize.py, engine/rl.py) on identical inputs.

The browser runs on the rounded inputs shipped in web/data.json, so the Python side
is fed exactly those inputs too.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import replace

import pytest

from engine.config import FIN, ROOT
from engine.optimize import SkuInputs, optimise

DATA = ROOT / "web" / "data.json"
pytestmark = pytest.mark.skipif(shutil.which("node") is None or not DATA.exists(),
                                reason="needs node and web/data.json (run engine.webexport)")


def _node(expr: str) -> object:
    """Evaluate `expr` in Node with the page's ENGINE block and data loaded."""
    tpl = (ROOT / "web" / "template.html").read_text()
    js = tpl[tpl.index("/* ENGINE-START"):tpl.index("/* ENGINE-END */")]
    script = f"const D = JSON.parse(require('fs').readFileSync({json.dumps(str(DATA))}, 'utf8'));\n{js}\n" \
             f"process.stdout.write(JSON.stringify({expr}));"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=600)
    return json.loads(out.stdout)


PRICE_CASES = [  # (sku, inventory, wacc, goal, fee, freight_share)
    ("704ef63a", 7, 0.065, "margin", 0.10, 1.0),
    ("704ef63a", 120, 0.15, "liquidity", 0.10, 1.0),
    ("703d7428", 15, 0.10, "margin", 0.05, 0.0),
    ("dc52f0f5", 60, 0.20, "liquidity", 0.20, 0.5),
]


def test_price_matches_python():
    d = json.loads(DATA.read_text())
    skus = {s["id"]: s for s in d["skus"]}
    cases = [c for c in PRICE_CASES if c[0] in skus]
    js = _node("[" + ",".join(
        f"ENG.price(D.skus.find((s) => s.id === {json.dumps(c[0])}), "
        f"{{inventory: {c[1]}, wacc: {c[2]}, goal: {json.dumps(c[3])}, fee: {c[4]}, freight_share: {c[5]}}})"
        for c in cases) + "]")
    for (sku, inv, wacc, goal, fee, share), j in zip(cases, js):
        s = skus[sku]
        fin = replace(FIN, platform_fee=fee, freight_share=share)
        x = SkuInputs(p_last=s["price"], q_base=s["q"], eps=s["eps"], eps_se=s["se"], cogs=s["cogs"],
                      freight=s["fr"], inventory=float(inv), annual_hold_rate=wacc + fin.storage_spread)
        p = optimise(x, goal, fin)
        assert abs(p.p_star - j["p_star"]) < 0.01, (sku, p.p_star, j["p_star"])
        for k in ("p_lo", "p_hi", "cm2_base", "hold_base", "profit_base", "wc_freed"):
            assert abs(getattr(p, k) - j[k]) < 0.01 + 1e-6 * abs(getattr(p, k)), (sku, k, getattr(p, k), j[k])
        for k in ("cm2_star", "hold_star", "profit_star"):
            assert abs(getattr(p, k) - j[k]) < 0.02, (sku, k, getattr(p, k), j[k])
        assert (p.aged, p.band_overridden) == (j["aged"], j["band_overridden"])
        # Same seeded draws on both sides, so the uncertainty band matches exactly too.
        for a, b in zip(p.p_star_ci, j["p_star_ci"]):
            assert abs(a - b) < 0.01, (sku, p.p_star_ci, j["p_star_ci"])


SEASON_CASES = [("704ef63a", 40, 0), ("704ef63a", 20, -1), ("dc52f0f5", 30, 1), ("703d7428", 12, 0)]

PY_SEASON = """
import torch, json, sys
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from engine.rl import H, MODEL_PATH, SellThroughEnv, dp_policy, myopic_move, rollout, solve_dp
d = json.load(open(sys.argv[1])); cases = json.loads(sys.argv[2]); n = int(sys.argv[3])
st = pd.DataFrame([{"product_id": s["id"], "cat_en": s["cat"], "p_last": s["price"], "q_base": s["q"], "eps": s["eps"],
                    "se": s["se"], "cogs": s["cogs"], "avg_freight": s["fr"]} for s in d["skus"]])
env, model, out = SellThroughEnv(st), PPO.load(MODEL_PATH, device="cpu"), []
for sku, wos, sur in cases:
    i = int(st.index[st.product_id == sku][0]); lot = env.make_lot(i, wos, 0.065)
    eps_true = float(np.clip(lot.eps_hat + sur * 1.645 * lot.eps_se, -5, -0.05))
    pols = {"hold": lambda e, o: 2, "weekly": lambda e, o: myopic_move(e), "planner": dp_policy(solve_dp(lot, lot.eps_hat)),
            "agent": lambda e, o: int(model.predict(o, deterministic=True)[0]), "oracle": dp_policy(solve_dp(lot, eps_true))}
    base = {"sku": i, "wos": wos, "selic": 0.065, "eps_true": eps_true}
    shown = {k: rollout(env, {**base, "u": np.array(d["uni"]["7"])}, f) for k, f in pols.items()}
    exp = {k: float(np.mean([rollout(env, {**base, "u": np.array(d["uni"][str(1000 + s)])}, f)["npv"] for s in range(n)]))
           for k, f in pols.items()}
    out.append({"lot": lot.inv0, "shown": {k: {"inv": [int(v) for v in r["inv_path"]], "npv": r["npv"]} for k, r in shown.items()}, "exp": exp})
print(json.dumps(out))
"""


def test_season_matches_python():
    n = 10
    js = _node("[" + ",".join(f"ENG.season(D.skus.find((s) => s.id === {json.dumps(sku)}), "
                              f"{{wos: {wos}, selic: 0.065, surprise: {sur}, scenarios: {n}}})"
                              for sku, wos, sur in SEASON_CASES) + "]")
    # Python reference in its own process: PyTorch (the PPO agent) must not share a process with LightGBM.
    py = subprocess.run([sys.executable, "-c", PY_SEASON, str(DATA), json.dumps(SEASON_CASES), str(n)], cwd=ROOT,
                        capture_output=True, text=True, timeout=900)
    assert py.returncode == 0, py.stderr[-2000:]
    ref = json.loads(py.stdout.strip().splitlines()[-1])
    for case, j, r in zip(SEASON_CASES, js, ref):
        assert j["lot"] == r["lot"], case
        for k in ("hold", "weekly", "planner", "oracle", "agent"):
            pinv = r["shown"][k]["inv"]
            assert j["paths"][k]["inv"][:len(pinv)] == pinv, (case, k)                  # identical season, week by week
            assert abs(j["paths"][k]["npv"] - r["shown"][k]["npv"]) < 0.1, (case, k)
            assert abs(j["expected"][k]["npv"] - r["exp"][k]) < 0.1, (case, k, j["expected"][k]["npv"], r["exp"][k])
