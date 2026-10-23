"""Global configuration and financial invariants (Phase 0).

Every number that changes a P&L figure lives here so the poster, the pipeline
and the dashboard can never disagree about an assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEED: int = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

# Weeks are Monday-start (W-SUN periods). Olist volume before 2017 is a ramp-up
# artefact (a few hundred orders) and the final weeks of Aug-2018 are truncated,
# so the analysis window is restricted to complete, representative weeks.
WINDOW_START = "2017-01-02"
WINDOW_END = "2018-08-20"  # Monday of the last complete week

# Olist category (Portuguese) -> (English label, COGS as share of baseline price).
# COGS shares are sector gross-margin benchmarks (Damodaran LatAm retail /
# Mercado Libre filings): electronics GM ~20%, home textiles GM ~42%,
# watches & gifts GM ~55%.
CATEGORIES: dict[str, tuple[str, float]] = {
    "relogios_presentes": ("watches_gifts", 0.45),
    "cama_mesa_banho": ("bed_bath_table", 0.58),
    "telefonia": ("telephony", 0.80),
    "informatica_acessorios": ("computers_accessories", 0.80),
}

# SKU eligibility: enough selling weeks to carry within-SKU price variation.
MIN_SELLING_WEEKS = 10
MIN_UNITS = 30


@dataclass(frozen=True)
class FinanceParams:
    """Unit-economics and capital-cost assumptions.

    platform_fee: marketplace commission as a share of price (Olist/MeLi ~10%).
    freight_share: share of freight value absorbed by the seller. 1.0 follows the
        CM2 definition CM2 = P - COGS - F - fee*P (free-shipping subsidy model).
    storage_spread: annual warehousing + obsolescence spread over SELIC.
    floor_normal / floor_liquidation: price floors as multiples of COGS.
    band: max week-on-week relative price move (volatility guardrail).
    aging_trigger_weeks: weeks-of-supply above which stock is "aged" and the
        floor relaxes to breakeven to accelerate cash recovery.
    """

    platform_fee: float = 0.10
    freight_share: float = 1.0
    storage_spread: float = 0.06
    floor_normal: float = 1.15
    floor_liquidation: float = 1.00
    band: float = 0.15
    aging_trigger_weeks: float = 8.0


FIN = FinanceParams()


def seed_everything(seed: int = SEED) -> np.random.Generator:
    """Fix numpy's legacy and Generator RNGs; returns a seeded Generator."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


@dataclass
class Paths:
    """Artifact locations produced by the pipeline and consumed by the app."""

    panel: Path = field(default=ARTIFACT_DIR / "panel.parquet")
    forecasts: Path = field(default=ARTIFACT_DIR / "forecasts.parquet")
    forecast_metrics: Path = field(default=ARTIFACT_DIR / "forecast_metrics.parquet")
    elasticity_cat: Path = field(default=ARTIFACT_DIR / "elasticity_category.parquet")
    elasticity_sku: Path = field(default=ARTIFACT_DIR / "elasticity_sku.parquet")
    sku_state: Path = field(default=ARTIFACT_DIR / "sku_state.parquet")
    recommendations: Path = field(default=ARTIFACT_DIR / "recommendations.parquet")


PATHS = Paths()
