"""Phase 1: transaction spine -> weekly SKU panel with unit economics.

Output grain: one row per (product_id, week) over each SKU's active listing
span, zero-filled for weeks with no sales. All demand features are strictly
lagged so the panel can be used for walk-forward forecasting without leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (CATEGORIES, FIN, MIN_SELLING_WEEKS, MIN_UNITS, RAW_DIR,
                     WINDOW_END, WINDOW_START, FinanceParams)
from .finance import holding_rate_annual, load_selic_weekly, unit_cm2, weekly_holding_cost

# Brazilian retail events, keyed by the Monday of the affected week.
_EVENTS: dict[str, list[str]] = {
    "black_friday": ["2017-11-20"],
    "christmas": ["2017-12-04", "2017-12-11", "2017-12-18"],
    "mothers_day": ["2017-05-08", "2018-05-07"],
    "valentines_br": ["2017-06-05", "2018-06-04"],  # Dia dos Namorados, 12 Jun
    "fathers_day": ["2017-08-07", "2018-08-06"],
    "consumer_day": ["2017-03-13", "2018-03-12"],
    # National truckers' strike, 21-31 May 2018: logistics shock, not demand.
    "truckers_strike": ["2018-05-21", "2018-05-28"],
}


def _week(ts: pd.Series) -> pd.Series:
    """Map timestamps to the Monday that starts their ISO week."""
    return ts.dt.to_period("W-SUN").dt.start_time


def load_transactions() -> pd.DataFrame:
    """Delivered order-items in the target categories, one row per item sold.

    Returns columns: order_id, product_id, seller_id, category, price, freight,
    week, delivery_days, review_score.
    """
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv",
                         parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"])
    orders = orders[orders["order_status"] == "delivered"]
    items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv",
                           usecols=["product_id", "product_category_name"])
    reviews = (pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv",
                           usecols=["order_id", "review_score"])
               .groupby("order_id", as_index=False)["review_score"].mean())

    df = (items.merge(orders, on="order_id")
          .merge(products, on="product_id")
          .merge(reviews, on="order_id", how="left"))
    df = df[df["product_category_name"].isin(CATEGORIES)]
    df["week"] = _week(df["order_purchase_timestamp"])
    df = df[(df["week"] >= WINDOW_START) & (df["week"] <= WINDOW_END)]
    df["delivery_days"] = (df["order_delivered_customer_date"]
                           - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    return df.rename(columns={"product_category_name": "category", "freight_value": "freight"})[
        ["order_id", "product_id", "seller_id", "category", "price", "freight",
         "week", "delivery_days", "review_score"]]


def _weekly_sku(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate items to (product_id, week) demand and price observations."""
    return (tx.groupby(["product_id", "category", "week"], as_index=False)
            .agg(units_sold=("price", "size"), avg_price=("price", "mean"),
                 avg_freight=("freight", "mean"), seller_count=("seller_id", "nunique"),
                 review_sum=("review_score", "sum"), review_n=("review_score", "count"),
                 delivery_days=("delivery_days", "mean")))


def build_panel(fin: FinanceParams = FIN) -> pd.DataFrame:
    """Build the modelling panel (Phase 1).

    Returns:
        DataFrame with one row per (product_id, week) and columns
        [product_id, category, cat_en, week, units_sold, avg_price, avg_freight,
         seller_count, avg_review_score, delivery_days, has_sale, lag_1/2/4,
         roll4_mean, roll4_std, price_ratio_cat, log_cat_price, cat_velocity_lag1,
         event flags, selic, cogs, cm2_unit, h_weekly, ...].
    """
    tx = load_transactions()
    wk = _weekly_sku(tx)

    # Category-level context uses *all* SKUs in the category, not only eligible ones.
    cat_units = wk.groupby(["category", "week"])["units_sold"].sum().rename("cat_units")
    cat_price = wk.groupby(["category", "week"])["avg_price"].median().rename("cat_med_price")

    stats = wk.groupby("product_id").agg(weeks=("week", "nunique"), units=("units_sold", "sum"))
    keep = stats[(stats["weeks"] >= MIN_SELLING_WEEKS) & (stats["units"] >= MIN_UNITS)].index
    wk = wk[wk["product_id"].isin(keep)]

    # Active listing span: first sale -> last sale + 4 weeks (capped at window end).
    # The 4-week tail keeps genuine post-sale zero weeks without padding delisted SKUs.
    end_cap = pd.Timestamp(WINDOW_END)
    frames = []
    for (pid, cat), g in wk.groupby(["product_id", "category"]):
        span = pd.date_range(g["week"].min(), min(g["week"].max() + pd.Timedelta(weeks=4), end_cap),
                             freq="7D")
        f = g.set_index("week").reindex(span).rename_axis("week").reset_index()
        f["product_id"], f["category"] = pid, cat
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)

    panel["has_sale"] = panel["units_sold"].notna().astype(int)
    panel["units_sold"] = panel["units_sold"].fillna(0.0)
    panel = panel.sort_values(["product_id", "week"]).reset_index(drop=True)
    g = panel.groupby("product_id", sort=False)

    # Posted price/freight persist through zero-sale weeks (forward fill only:
    # a zero week reveals no new price, and back-filling would leak the future).
    for col in ["avg_price", "avg_freight", "seller_count", "delivery_days"]:
        panel[col] = g[col].ffill()
    # Review score as known to a shopper at week t: cumulative mean of past reviews.
    rs = g["review_sum"].cumsum() - panel["review_sum"].fillna(0)
    rn = g["review_n"].cumsum() - panel["review_n"].fillna(0)
    panel["avg_review_score"] = (rs / rn.replace(0, np.nan))
    panel["avg_review_score"] = panel["avg_review_score"].fillna(panel["avg_review_score"].median())
    panel = panel.drop(columns=["review_sum", "review_n"])

    # Lagged demand features (shift(1) => information set ends at t-1).
    g = panel.groupby("product_id", sort=False)["units_sold"]
    for k in (1, 2, 4):
        panel[f"lag_{k}"] = g.shift(k).fillna(0.0)
    past = g.shift(1)
    panel["roll4_mean"] = past.groupby(panel["product_id"]).transform(
        lambda s: s.rolling(4, min_periods=1).mean()).fillna(0.0)
    panel["roll4_std"] = past.groupby(panel["product_id"]).transform(
        lambda s: s.rolling(4, min_periods=2).std()).fillna(0.0)
    panel["weeks_listed"] = panel.groupby("product_id").cumcount()

    panel = panel.merge(cat_units.reset_index(), on=["category", "week"], how="left")
    panel = panel.merge(cat_price.reset_index(), on=["category", "week"], how="left")
    panel = panel.sort_values(["product_id", "week"]).reset_index(drop=True)
    panel["cat_units"] = panel["cat_units"].fillna(0.0)
    panel["cat_velocity_lag1"] = panel.groupby("product_id")["cat_units"].shift(1).fillna(0.0)
    panel["cat_med_price"] = panel.groupby("category")["cat_med_price"].ffill()
    panel["log_cat_price"] = np.log(panel["cat_med_price"])
    panel["price_ratio_cat"] = panel["avg_price"] / panel["cat_med_price"]
    panel = panel.drop(columns=["cat_units"])

    for name, weeks in _EVENTS.items():
        panel[f"ev_{name}"] = panel["week"].isin(pd.to_datetime(weeks)).astype(int)
    woy = panel["week"].dt.isocalendar().week.astype(float).values
    for k in (1, 2):
        panel[f"sin{k}"] = np.sin(2 * np.pi * k * woy / 52.18)
        panel[f"cos{k}"] = np.cos(2 * np.pi * k * woy / 52.18)
    panel["t_index"] = ((panel["week"] - pd.Timestamp(WINDOW_START)).dt.days // 7).astype(int)

    panel = panel.merge(load_selic_weekly(), on="week", how="left")
    panel["selic"] = panel["selic"].ffill()

    # Unit economics. Baseline price = SKU median transacted price; COGS is a
    # fixed accounting cost per SKU (it does not move with our price decision).
    panel["cat_en"] = panel["category"].map(lambda c: CATEGORIES[c][0])
    share = panel["category"].map(lambda c: CATEGORIES[c][1])
    base_price = panel[panel["has_sale"] == 1].groupby("product_id")["avg_price"].median()
    panel["base_price"] = panel["product_id"].map(base_price)
    panel["cogs"] = share * panel["base_price"]
    panel["cm2_unit"] = unit_cm2(panel["avg_price"], panel["cogs"], panel["avg_freight"], fin)
    panel["h_weekly"] = weekly_holding_cost(panel["cogs"], holding_rate_annual(panel["selic"], fin))

    panel["log_q"] = np.log1p(panel["units_sold"])
    panel["log_p"] = np.log(panel["avg_price"])
    return panel
