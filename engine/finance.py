"""Unit economics, cost of capital and working-capital metrics.

Definitions used throughout (per unit, BRL):
    Net revenue       = P * (1 - fee)
    CM1 (gross margin)= P - COGS
    CM2               = P - COGS - freight_share * F - fee * P
    h_t (weekly hold) = (SELIC_t + spread) / 52 * COGS
    DSI (days)        = inventory / weekly units * 7
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FIN, RAW_DIR, FinanceParams

# BACEN COPOM target-rate decisions (series SGS 432), used only if the cached
# API download is missing. Effective dates are the day after the meeting.
_SELIC_FALLBACK: list[tuple[str, float]] = [
    ("2016-01-01", 14.25), ("2016-10-20", 14.00), ("2016-12-01", 13.75),
    ("2017-01-12", 13.00), ("2017-02-23", 12.25), ("2017-04-13", 11.25),
    ("2017-06-01", 10.25), ("2017-07-27", 9.25), ("2017-09-08", 8.25),
    ("2017-10-26", 7.50), ("2017-12-07", 7.00), ("2018-02-08", 6.75),
    ("2018-03-22", 6.50),
]


def load_selic_weekly(path: Path = RAW_DIR / "selic_432.json") -> pd.DataFrame:
    """Weekly average SELIC target rate.

    Returns:
        DataFrame[week: datetime64 (Monday), selic: float annual decimal, e.g. 0.065].
    """
    if path.exists():
        raw = pd.DataFrame(json.loads(path.read_text()))
        daily = pd.Series(raw["valor"].astype(float).values,
                          index=pd.to_datetime(raw["data"], format="%d/%m/%Y"))
    else:
        steps = pd.Series({pd.Timestamp(d): v for d, v in _SELIC_FALLBACK})
        idx = pd.date_range("2016-01-01", "2018-12-31", freq="D")
        daily = steps.reindex(idx).ffill()
    weekly = (daily / 100).resample("W-SUN").mean()
    weekly.index = weekly.index - pd.Timedelta(days=6)  # label by Monday
    return weekly.rename("selic").rename_axis("week").reset_index()


def holding_rate_annual(selic: np.ndarray | float, fin: FinanceParams = FIN) -> np.ndarray | float:
    """H_annual = SELIC_t + storage/depreciation spread (decimal per year)."""
    return selic + fin.storage_spread


def weekly_holding_cost(cogs: np.ndarray | float, annual_rate: np.ndarray | float) -> np.ndarray | float:
    """h_t = H_annual / 52 * COGS: carrying cost of one unit for one week (BRL)."""
    return annual_rate / 52.0 * cogs


def unit_cm2(price: np.ndarray | float, cogs: np.ndarray | float, freight: np.ndarray | float,
             fin: FinanceParams = FIN) -> np.ndarray | float:
    """Contribution Margin 2 per unit: P(1 - fee) - COGS - freight_share * F."""
    return price * (1.0 - fin.platform_fee) - cogs - fin.freight_share * freight


def dsi_days(inventory: np.ndarray | float, weekly_units: np.ndarray | float) -> np.ndarray | float:
    """Days Sales of Inventory = inventory / daily sell-through. Inf if no sales."""
    weekly_units = np.asarray(weekly_units, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(weekly_units > 0, np.asarray(inventory) / weekly_units * 7.0, np.inf)


def safety_stock_units(sigma_err: np.ndarray | float, z: float = 1.645,
                       lead_weeks: float = 1.0) -> np.ndarray | float:
    """Safety stock for a target cycle-service level: SS = z * sigma_e * sqrt(L).

    sigma_err is the std of weekly forecast errors (units); z=1.645 -> 95% CSL.
    The capital tied up in safety stock (SS * COGS) is the balance-sheet price of
    forecast uncertainty: better forecasts release this working capital.
    """
    return z * np.asarray(sigma_err) * np.sqrt(lead_weeks)


def breakeven_elasticity(price: np.ndarray | float, cogs: np.ndarray | float,
                         freight: np.ndarray | float, fin: FinanceParams = FIN) -> np.ndarray:
    """Elasticity beyond which a marginal price cut raises total CM2 (Lerner test).

    With Pi = Q(P) * (aP - c), a = 1 - fee, c = COGS + s*F and constant eps:
        dPi/dP < 0  <=>  |eps| > aP / (aP - c) = 1 / m,   m = CM2 / net revenue.
    Returns -1/m; -inf when CM2 <= 0 (no elasticity makes a cut accretive).
    """
    net = np.asarray(price) * (1.0 - fin.platform_fee)
    m = (net - np.asarray(cogs) - fin.freight_share * np.asarray(freight)) / net
    with np.errstate(divide="ignore"):
        return np.where(m > 0, -1.0 / m, -np.inf)
