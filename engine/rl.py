"""Multi-period pricing: a reinforcement-learning agent for single-lot sell-through.

Problem (one SKU, one season of H weeks)
  * A lot of I0 units is on hand; no replenishment. Unsold units at the end
    are liquidated at `salvage` x COGS (outlet / clearance channel).
  * Each week the seller moves price on a lattice P_k = P0 * g^k, g = sqrt(1.15),
    by at most two steps (= the +/-15% volatility band), never below the
    margin floor (1.15 x COGS, relaxed to 1.00 x COGS once stock is aged, i.e.
    weeks-of-supply > 8) and never above P0 * g^5 (+42%: beyond the price range
    in which elasticity was identified).
  * Demand ~ NegBin(mean = q_base * (P/P0)^eps, dispersion 3): over-dispersed
    weekly counts as seen in the data.
  * Weekly reward  = sales * CM2_unit(P) - h * stock carried;  h = (SELIC+6%)/52 * COGS.
    Terminal reward = -(1 - salvage) * COGS * leftover: the write-down on stock
    cleared through the outlet channel (CM2 already books COGS on units sold,
    so unsold units must book their loss too).  Discount: weekly WACC (SELIC).

Learning while earning
  The true elasticity is unknown. Each episode draws eps_true ~ N(eps_hat, se)
  from the Double-ML posterior; the environment keeps an exact Bayesian grid
  posterior over eps, updated from observed (possibly stock-censored) sales,
  and shows the agent its posterior mean and std. Exploring (moving price to
  learn) therefore has value the agent can discover.

Policies compared on common random numbers
  status quo    hold P0 all season
  myopic        Phase-4 engine: each week maximise one-week CM2 - carry (eps_hat)
  CE-DP         exact dynamic programme using eps_hat as if it were true
  PPO agent     one neural policy for the whole catalogue (stable-baselines3)
  oracle DP     exact DP knowing eps_true: an upper bound no policy can beat
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from .config import ARTIFACT_DIR, FIN, PATHS, SEED

H = 26
STEP = np.sqrt(1.15)
K_LO, K_HI = -6, 5
KS = np.arange(K_LO, K_HI + 1)
MOVES = np.array([-2, -1, 0, 1, 2])
DISPERSION = 3.0
SALVAGE = 0.5
EPS_GRID = np.linspace(-6.0, 0.0, 61)
MODEL_PATH = ARTIFACT_DIR / "ppo_pricing.zip"


@dataclass(frozen=True)
class Lot:
    """One sell-through problem instance."""

    p0: float
    q_base: float
    eps_hat: float
    eps_se: float
    cogs: float
    freight: float
    inv0: int
    selic: float

    @property
    def h(self) -> float:
        return (self.selic + FIN.storage_spread) / 52.0 * self.cogs

    @property
    def gamma(self) -> float:
        return (1.0 + self.selic) ** (-1.0 / 52.0)

    def price(self, k: np.ndarray | int) -> np.ndarray:
        return self.p0 * STEP ** np.asarray(k, dtype=float)

    def unit_cm2(self, k: np.ndarray | int) -> np.ndarray:
        p = self.price(k)
        return p * (1 - FIN.platform_fee) - self.cogs - FIN.freight_share * self.freight

    def mu(self, k: np.ndarray | int, eps: np.ndarray | float) -> np.ndarray:
        return self.q_base * STEP ** (np.asarray(k, dtype=float) * eps)

    def floor_k(self, inv: np.ndarray | int) -> np.ndarray:
        """Lowest lattice index satisfying the (inventory-dependent) margin floor."""
        aged = np.asarray(inv) / max(self.q_base, 1e-9) > FIN.aging_trigger_weeks
        mult = np.where(aged, FIN.floor_liquidation, FIN.floor_normal)
        k = np.ceil(np.log(mult * self.cogs / self.p0) / np.log(STEP) - 1e-9)
        return np.clip(k, K_LO, K_HI).astype(int)

    def next_k(self, k: np.ndarray, move: np.ndarray, inv: np.ndarray) -> np.ndarray:
        """Apply a band-limited move, then enforce floor (floor overrides band) and cap."""
        return np.maximum(np.clip(k + move, K_LO, K_HI), self.floor_k(inv))


def _sales_dist(mu: float, inv_max: int) -> np.ndarray:
    """S[i, s] = P(sales = s | stock i): demand censored at stock."""
    n, p = DISPERSION, DISPERSION / (DISPERSION + mu)
    pmf = nbinom.pmf(np.arange(inv_max + 1), n, p)
    sf = nbinom.sf(np.arange(inv_max + 1) - 1, n, p)  # P(D >= i)
    S = np.zeros((inv_max + 1, inv_max + 1))
    for i in range(inv_max + 1):
        S[i, :i] = pmf[:i]
        S[i, i] = sf[i]
    return S


def solve_dp(lot: Lot, eps: float) -> np.ndarray:
    """Exact finite-horizon DP. Returns policy[t, inv, k_index] -> move index."""
    I = lot.inv0
    invs = np.arange(I + 1)
    nk = len(KS)
    S = np.stack([_sales_dist(float(lot.mu(k, eps)), I) for k in KS])  # (nk, I+1, I+1)
    exp_sales = (S * invs[None, None, :]).sum(-1)                       # (nk, I+1)
    cm2 = lot.unit_cm2(KS)
    V = np.zeros((I + 1, nk)) - ((1 - SALVAGE) * lot.cogs * invs)[:, None]  # terminal write-down
    policy = np.zeros((H, I + 1, nk), dtype=np.int8)
    # k' for every (inv, k, move); index into KS
    kp = np.stack([lot.next_k(KS[None, :], m, invs[:, None]) for m in MOVES], -1) - K_LO
    for t in reversed(range(H)):
        # Q-value of landing on price index j with stock i:
        #   E[cm2_j * s - h * (i - s) + gamma * V(i - s, j)]
        cont = np.zeros((I + 1, nk))
        for j in range(nk):
            Vj = V[:, j]
            nxt = np.array([S[j, i, :i + 1] @ Vj[i - np.arange(i + 1)] for i in invs])
            cont[:, j] = (cm2[j] * exp_sales[j] - lot.h * (invs - exp_sales[j]) + lot.gamma * nxt)
        cont[0, :] = 0.0  # sold out: season over
        q = cont[invs[:, None, None], kp]  # q[i, k, a] = cont[i, k'(i, k, a)]
        policy[t] = q.argmax(-1)
        V = q.max(-1)
    return policy


class SellThroughEnv(gym.Env):
    """Gymnasium environment over the live SKU catalogue (see module docstring)."""

    metadata = {"render_modes": []}

    def __init__(self, skus: pd.DataFrame, selic_range: tuple[float, float] = (0.065, 0.15),
                 wos_range: tuple[float, float] = (4.0, 52.0)):
        super().__init__()
        self.skus = skus.reset_index(drop=True)
        self.selic_range, self.wos_range = selic_range, wos_range
        # Balanced sampling across elasticity regimes: highly elastic SKUs are
        # rare in the catalogue but are where pricing decisions matter most.
        bucket = pd.cut(self.skus["eps"], [-np.inf, -2, -1, -0.5, 0]).astype(str)
        w = 1.0 / bucket.map(bucket.value_counts()).to_numpy(dtype=float)
        self.sku_p = w / w.sum()
        self.action_space = gym.spaces.Discrete(len(MOVES))
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(12,), dtype=np.float32)

    # -- episode construction -------------------------------------------------
    def make_lot(self, i: int, wos: float, selic: float) -> Lot:
        r = self.skus.iloc[i]
        return Lot(p0=r.p_last, q_base=r.q_base, eps_hat=r.eps, eps_se=r.se, cogs=r.cogs,
                   freight=r.avg_freight, inv0=int(max(np.ceil(wos * r.q_base), 2)), selic=selic)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        o = options or {}
        rng = self.np_random
        i = o.get("sku", int(rng.choice(len(self.skus), p=self.sku_p)))
        wos = o.get("wos", float(rng.uniform(*self.wos_range)))
        selic = o.get("selic", float(rng.uniform(*self.selic_range)))
        self.lot = self.make_lot(i, wos, selic)
        self.eps_true = o.get("eps_true", float(np.clip(rng.normal(self.lot.eps_hat, self.lot.eps_se), -5, -0.05)))
        self.u = o.get("u", rng.uniform(size=H))  # common random numbers for demand
        self.t, self.inv, self.k = 0, self.lot.inv0, int(self.lot.next_k(np.array(0), 0, self.lot.inv0))
        self.log_post = norm.logpdf(EPS_GRID, self.lot.eps_hat, max(self.lot.eps_se, 0.05))
        self.last_sales = self.lot.q_base
        return self._obs(), {}

    def posterior(self) -> tuple[float, float]:
        w = np.exp(self.log_post - self.log_post.max())
        w /= w.sum()
        m = float(w @ EPS_GRID)
        return m, float(np.sqrt(w @ (EPS_GRID - m) ** 2))

    def _obs(self) -> np.ndarray:
        lot = self.lot
        pm, ps = self.posterior()
        net0 = lot.p0 * (1 - FIN.platform_fee)
        remaining_demand = lot.q_base * (H - self.t)
        return np.array([
            self.inv / lot.inv0, 1 - self.t / H, self.k / K_HI, pm / 3, ps,
            (net0 - lot.cogs - FIN.freight_share * lot.freight) / net0, lot.cogs / lot.p0,
            np.log1p(remaining_demand / max(self.inv, 1)), lot.selic * 10,
            (self.k - lot.floor_k(self.inv)) / 5, np.log1p(self.last_sales / max(lot.q_base, 1e-6)),
            np.log(lot.q_base),
        ], dtype=np.float32)

    def step(self, action: int):
        lot = self.lot
        self.k = int(lot.next_k(np.array(self.k), MOVES[action], self.inv))
        mu = float(lot.mu(self.k, self.eps_true))
        n, p = DISPERSION, DISPERSION / (DISPERSION + mu)
        demand = max(int(nbinom.ppf(self.u[self.t], n, p)), 0)  # ppf(0) = -1
        sales = min(demand, self.inv)
        # Bayesian update over the elasticity grid (censored if we sold out).
        mus = lot.mu(self.k, EPS_GRID)
        ps = DISPERSION / (DISPERSION + mus)
        ll = (nbinom.logsf(self.inv - 1, n, ps) if sales == self.inv
              else nbinom.logpmf(sales, n, ps))
        if np.isfinite(ll).any():  # skip an update every grid point deems impossible
            self.log_post += np.where(np.isfinite(ll), ll, -1e9)
        self.inv -= sales
        self.last_sales = sales
        reward = sales * float(lot.unit_cm2(self.k)) - lot.h * self.inv
        self.t += 1
        done = self.t >= H or self.inv == 0
        if self.t >= H:
            reward -= (1 - SALVAGE) * lot.cogs * self.inv
        info = {"price": float(lot.price(self.k)), "sales": sales, "inv": self.inv, "reward": reward}
        # Train on return-on-inventory-investment so all SKUs share one scale.
        return self._obs(), reward / (lot.cogs * lot.inv0), done, False, info


# -- baseline policies ----------------------------------------------------------
def myopic_move(env: SellThroughEnv) -> int:
    """Phase-4 engine on the lattice: best one-week CM2 - carry under eps_hat."""
    lot, best, arg = env.lot, -np.inf, 2
    for a, m in enumerate(MOVES):
        k = int(lot.next_k(np.array(env.k), m, env.inv))
        q = min(float(lot.mu(k, lot.eps_hat)), env.inv)
        v = q * float(lot.unit_cm2(k)) - lot.h * (env.inv - q)
        if v > best + 1e-12:
            best, arg = v, a
    return arg


def rollout(env: SellThroughEnv, opts: dict, policy) -> dict:
    """Run one episode with `policy(env, obs) -> action`; discounted BRL outcome."""
    obs, _ = env.reset(options=opts)
    total, disc, prices, invs, sold = 0.0, 1.0, [], [], 0
    while True:
        obs, _, done, _, info = env.step(policy(env, obs))
        total += disc * info["reward"]
        disc *= env.lot.gamma
        prices.append(info["price"]); invs.append(info["inv"]); sold += info["sales"]
        if done:
            break
    return {"npv": total, "sell_through": sold / env.lot.inv0, "weeks": len(prices),
            "avg_price_ratio": float(np.mean(prices) / env.lot.p0), "prices": prices, "inv_path": invs}


def dp_policy(table: np.ndarray):
    return lambda env, obs: int(table[env.t, env.inv, env.k - K_LO])


def load_skus() -> pd.DataFrame:
    return pd.read_parquet(PATHS.sku_state)


def train(total_steps: int = 2_000_000) -> None:
    """Train one PPO policy across the whole live catalogue."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(lambda: SellThroughEnv(load_skus()), n_envs=8, seed=SEED)
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=512, learning_rate=3e-4, gamma=0.999,
                gae_lambda=0.95, ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]),
                seed=SEED, verbose=0, device="cpu")
    model.learn(total_timesteps=total_steps, progress_bar=False)
    model.save(MODEL_PATH)


def evaluate(n_episodes: int = 30, selic: float = 0.065, max_inv: int = 150) -> pd.DataFrame:
    """Every policy on identical (SKU, stock, eps_true, demand-noise) draws."""
    from stable_baselines3 import PPO

    skus = load_skus()
    env = SellThroughEnv(skus)
    model = PPO.load(MODEL_PATH, device="cpu")
    ppo = lambda e, o: int(model.predict(o, deterministic=True)[0])  # noqa: E731
    rng = np.random.default_rng(SEED + 1)
    rows, ce_cache = [], {}
    for i in range(len(skus)):
        for ep in range(n_episodes):
            wos = float(rng.uniform(4, 52))
            lot = env.make_lot(i, wos, selic)
            if lot.inv0 > max_inv:
                continue
            eps_true = float(np.clip(rng.normal(lot.eps_hat, lot.eps_se), -5, -0.05))
            opts = {"sku": i, "wos": wos, "selic": selic, "eps_true": eps_true, "u": rng.uniform(size=H)}
            key = (i, lot.inv0)
            if key not in ce_cache:
                ce_cache[key] = solve_dp(lot, lot.eps_hat)
            ce = ce_cache[key]
            policies = {"status quo": lambda e, o: 2, "myopic (Phase 4)": lambda e, o: myopic_move(e),
                        "CE-DP": dp_policy(ce), "PPO agent": ppo,
                        "oracle DP": dp_policy(solve_dp(lot, eps_true))}
            for name, pol in policies.items():
                r = rollout(env, opts, pol)
                rows.append({"product_id": skus.product_id[i], "cat_en": skus.cat_en[i], "episode": ep,
                             "wos": wos, "eps_true": eps_true, "policy": name,
                             "capital": lot.cogs * lot.inv0, **{k: r[k] for k in
                                                                ("npv", "sell_through", "weeks", "avg_price_ratio")}})
    return pd.DataFrame(rows)


def rl_summary(rl: pd.DataFrame) -> pd.DataFrame:
    """Mean outcomes per policy and stock regime; NPV also as % of the oracle."""
    rl = rl.assign(regime=pd.cut(rl["wos"], [0, 13, 26, 60],
                                 labels=["lean (<13 wks)", "full (13-26 wks)", "overstock (>26 wks)"]))
    oracle = rl[rl["policy"] == "oracle DP"].set_index(["product_id", "episode"])["npv"]
    rl["npv_vs_oracle"] = rl["npv"] - rl.set_index(["product_id", "episode"]).index.map(oracle)
    out = (rl.groupby(["regime", "policy"], observed=True)
           .agg(npv=("npv", "mean"), gap_to_oracle=("npv_vs_oracle", "mean"),
                sell_through=("sell_through", "mean"), avg_price_ratio=("avg_price_ratio", "mean"),
                roi=("npv", lambda s: s.sum() / rl.loc[s.index, "capital"].sum()))
           .reset_index())
    return out


if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    if "train" in sys.argv:
        nxt = sys.argv[sys.argv.index("train") + 1:]
        train(int(nxt[0]) if nxt and nxt[0].isdigit() else 2_000_000)
        print(f"trained in {time.time() - t0:.0f}s -> {MODEL_PATH}")
    if "eval" in sys.argv:
        res = evaluate()
        res.to_parquet(ARTIFACT_DIR / "rl_eval.parquet")
        print(f"evaluated in {time.time() - t0:.0f}s")
