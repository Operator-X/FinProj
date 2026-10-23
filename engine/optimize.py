"""Phase 4: constrained CM2-minus-carrying-cost price optimisation.

For one SKU and one week:

    Q(P)  = Q_base * (P / P_base) ** eps                    (constant-elasticity demand)
    Pi(P) = Q(P) * [P(1 - fee) - COGS - s*F] - HoldingCost(P)

Holding cost depends on the strategy mode:
  * "margin"    (EBITDA):   h * max(0, I - Q(P))            one week of carry on leftover stock
  * "liquidity" (cash):     h * L * min(L / Q(P), 52) / 2   full carry until the leftover L
                                                             is sold down at rate Q(P)
The second view prices the whole remaining cash-conversion cycle, so it rewards
faster sell-through more strongly.

Hard constraints (Phase 0):
  P >= 1.15 * COGS   (1.00 * COGS if weeks-of-supply > 8: aged-stock liquidation)
  0.85 * P_last <= P <= 1.15 * P_last
If the margin floor sits above the volatility band, the floor wins: loss
prevention outranks price smoothness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .config import FIN, SEED, FinanceParams

GRID = 401
MAX_CARRY_WEEKS = 52.0


@dataclass(frozen=True)
class SkuInputs:
    """State needed to price one SKU for the coming week (BRL, units/week)."""

    p_last: float          # last posted price = status-quo price
    q_base: float          # forecast weekly units at p_last
    eps: float             # demand elasticity (negative)
    eps_se: float          # its standard error
    cogs: float
    freight: float
    inventory: float       # units on hand
    annual_hold_rate: float  # WACC/SELIC + storage spread


@dataclass
class PriceDecision:
    p_star: float
    p_lo: float
    p_hi: float
    p_star_ci: tuple[float, float]
    q_base: float
    q_star: float
    cm2_base: float          # total CM2 at status quo (BRL/week)
    cm2_star: float
    hold_base: float         # carrying cost at status quo (BRL)
    hold_star: float
    profit_base: float
    profit_star: float
    dsi_base: float          # days
    dsi_star: float
    wc_freed: float          # BRL of inventory converted to cash this week vs status quo
    weeks_of_supply: float
    aged: bool
    floor_binding: bool
    band_overridden: bool

    def as_dict(self) -> dict:
        return asdict(self)


def demand(p: np.ndarray, x: SkuInputs, eps: np.ndarray | float | None = None) -> np.ndarray:
    """Constant-elasticity demand anchored at the status-quo point."""
    e = x.eps if eps is None else eps
    return x.q_base * (p / x.p_last) ** e


def holding_cost(q: np.ndarray, x: SkuInputs, mode: str) -> np.ndarray:
    """Carrying cost of stock left after this week's sales (BRL)."""
    h = x.annual_hold_rate / 52.0 * x.cogs
    left = np.clip(x.inventory - q, 0.0, None)
    if mode == "margin":
        return h * left
    weeks = np.minimum(np.divide(left, q, out=np.full_like(left, MAX_CARRY_WEEKS), where=q > 0),
                       MAX_CARRY_WEEKS)
    return h * left * weeks / 2.0


def profit(p: np.ndarray, x: SkuInputs, mode: str, fin: FinanceParams = FIN,
           eps: np.ndarray | float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (Pi, total CM2, holding cost) at price(s) p. Units sold are capped by stock."""
    q = np.minimum(demand(p, x, eps), x.inventory) if x.inventory > 0 else demand(p, x, eps)
    unit = p * (1.0 - fin.platform_fee) - x.cogs - fin.freight_share * x.freight
    cm2 = q * unit
    hold = holding_cost(q, x, mode)
    return cm2 - hold, cm2, hold


def bounds(x: SkuInputs, fin: FinanceParams = FIN) -> tuple[float, float, bool, bool]:
    """Feasible price interval. Returns (lo, hi, aged, band_overridden)."""
    wos = x.inventory / max(x.q_base, 1e-9)
    aged = wos > fin.aging_trigger_weeks
    floor = (fin.floor_liquidation if aged else fin.floor_normal) * x.cogs
    lo = max(floor, (1 - fin.band) * x.p_last)
    hi = (1 + fin.band) * x.p_last
    overridden = floor > hi
    if overridden:
        lo = hi = floor
    return lo, hi, aged, overridden


def _argmax(x: SkuInputs, mode: str, lo: float, hi: float, fin: FinanceParams) -> float:
    """Global grid search (Pi has a kink where stock binds) + bounded Brent polish."""
    if hi - lo < 1e-9:
        return lo
    grid = np.linspace(lo, hi, GRID)
    pi, _, _ = profit(grid, x, mode, fin)
    i = int(np.argmax(pi))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, GRID - 1)]
    res = minimize_scalar(lambda p: -profit(np.array([p]), x, mode, fin)[0][0],
                          bounds=(a, b), method="bounded", options={"xatol": 1e-4})
    return float(res.x) if -res.fun >= pi[i] else float(grid[i])


def optimise(x: SkuInputs, mode: str = "margin", fin: FinanceParams = FIN,
             n_draws: int = 200) -> PriceDecision:
    """Solve for P* and its 90% interval under elasticity uncertainty.

    The interval comes from re-solving on a grid for eps ~ N(eps_hat, se) draws
    (seeded), i.e. how much the recommendation moves if the elasticity estimate
    is off; it is a parameter-uncertainty band, not a demand forecast interval.
    """
    lo, hi, aged, overridden = bounds(x, fin)
    p_star = _argmax(x, mode, lo, hi, fin)

    rng = np.random.default_rng(SEED)
    draws = np.clip(rng.normal(x.eps, x.eps_se, n_draws), -5.0, -0.05)[:, None]
    grid = np.linspace(lo, hi, GRID)[None, :]
    pi_draws, _, _ = profit(grid, x, mode, fin, eps=draws)
    p_draws = grid[0, np.argmax(pi_draws, axis=1)]
    ci = (float(np.percentile(p_draws, 5)), float(np.percentile(p_draws, 95)))

    p0 = np.array([x.p_last])
    pi0, cm0, h0 = (v[0] for v in profit(p0, x, mode, fin))
    pis, cms, hs = (v[0] for v in profit(np.array([p_star]), x, mode, fin))
    q0 = float(min(demand(p0, x)[0], x.inventory) if x.inventory > 0 else demand(p0, x)[0])
    qs = float(min(demand(np.array([p_star]), x)[0], x.inventory) if x.inventory > 0
               else demand(np.array([p_star]), x)[0])
    dsi = lambda q: float(x.inventory / q * 7.0) if q > 0 else float("inf")  # noqa: E731
    return PriceDecision(
        p_star=p_star, p_lo=lo, p_hi=hi, p_star_ci=ci, q_base=q0, q_star=qs,
        cm2_base=float(cm0), cm2_star=float(cms), hold_base=float(h0), hold_star=float(hs),
        profit_base=float(pi0), profit_star=float(pis), dsi_base=dsi(q0), dsi_star=dsi(qs),
        wc_freed=float((qs - q0) * x.cogs), weeks_of_supply=x.inventory / max(x.q_base, 1e-9),
        aged=aged, floor_binding=bool(abs(p_star - lo) < 1e-6 and lo > (1 - fin.band) * x.p_last),
        band_overridden=overridden)
