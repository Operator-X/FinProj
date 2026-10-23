"""Phase 5C: customer-voice signals from LLM-coded review themes -> causal features.

Olist reviews are free-text Portuguese. An LLM (Claude, via the `claude -p`
CLI) codes each comment into a fixed finance-relevant taxonomy; labels are
cached in data/labels/ so the pipeline is reproducible without re-querying.

Themes are then aggregated into *leakage-safe* SKU-week features: at week t a
SKU only "knows" reviews written before t, mirroring what a shopper sees.

Run labelling:  uv run python -m engine.reviews  (resumable, ~35 batched calls)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR, ROOT

LABEL_PATH = ROOT / "data" / "labels" / "review_labels.jsonl"
BATCH = 100
MODEL = "haiku"

THEMES: dict[str, str] = {
    "value_good": "good value for money / cheap for what you get / worth the price",
    "price_high": "too expensive / overpriced / not worth the price",
    "quality_good": "product quality praised (material, works well, beautiful, as described)",
    "quality_bad": "defective, poor quality, broke, fake, different from the listing",
    "delivery_fast": "arrived early / fast / on time praised",
    "delivery_late": "late, delayed, or not yet delivered",
    "wrong_or_missing": "wrong item, missing items, incomplete order",
    "service_bad": "seller/store service complaint (no response, refund problems, cancellation issues)",
    "recommend": "explicitly recommends the product or store",
}

PROMPT = """You are coding Brazilian e-commerce product reviews (Portuguese) for a finance study.
For EACH review, return which themes apply (zero or more) and an overall sentiment.

Themes:
{themes}

Sentiment: -1 negative, 0 neutral/mixed, 1 positive.

Return ONLY a JSON array, one object per review, same order, no prose:
[{{"id": "<id>", "themes": ["theme_key", ...], "sentiment": 1}}, ...]

Reviews:
{reviews}"""


def load_review_texts(product_ids: set[str]) -> pd.DataFrame:
    """Reviews with text for the given SKUs.

    Returns DataFrame[review_id, product_id, review_date, review_score, text].
    A review attaches to every eligible SKU in its order.
    """
    items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv", usecols=["order_id", "product_id"])
    rev = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv",
                      parse_dates=["review_creation_date"])
    m = rev.merge(items.drop_duplicates(), on="order_id")
    m = m[m["product_id"].isin(product_ids) & m["review_comment_message"].notna()]
    m["text"] = m["review_comment_message"].str.replace(r"\s+", " ", regex=True).str.strip()
    m = m[m["text"].str.len() > 1]
    return (m.rename(columns={"review_creation_date": "review_date"})
            [["review_id", "product_id", "review_date", "review_score", "text"]]
            .drop_duplicates(["review_id", "product_id"]).reset_index(drop=True))


def _call_llm(batch: pd.DataFrame) -> list[dict]:
    """One `claude -p` call for a batch of unique reviews; returns parsed labels."""
    themes = "\n".join(f"- {k}: {v}" for k, v in THEMES.items())
    reviews = "\n".join(json.dumps({"id": r.review_id, "text": r.text[:600]}, ensure_ascii=False)
                        for r in batch.itertuples())
    out = subprocess.run(["claude", "-p", "--model", MODEL],
                         input=PROMPT.format(themes=themes, reviews=reviews),
                         capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[:500])
    match = re.search(r"\[.*\]", out.stdout, flags=re.S)
    if not match:
        raise ValueError(f"no JSON array in output: {out.stdout[:300]}")
    rows = json.loads(match.group(0))
    valid = set(batch["review_id"])
    clean = []
    for r in rows:
        if r.get("id") in valid:
            clean.append({"review_id": r["id"],
                          "themes": [t for t in r.get("themes", []) if t in THEMES],
                          "sentiment": int(np.clip(int(r.get("sentiment", 0)), -1, 1))})
    return clean


def label_reviews(texts: pd.DataFrame) -> None:
    """Label every unseen review id, appending to LABEL_PATH (resumable)."""
    LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if LABEL_PATH.exists():
        done = {json.loads(l)["review_id"] for l in LABEL_PATH.read_text().splitlines() if l.strip()}
    todo = texts.drop_duplicates("review_id")
    todo = todo[~todo["review_id"].isin(done)]
    print(f"{len(done)} cached, {len(todo)} to label")
    for start in range(0, len(todo), BATCH):
        batch = todo.iloc[start:start + BATCH]
        for attempt in range(3):
            try:
                rows = _call_llm(batch)
                break
            except Exception as e:  # retry transient CLI / JSON failures
                print(f"  batch {start}: attempt {attempt + 1} failed: {e}", file=sys.stderr)
        else:
            continue
        with LABEL_PATH.open("a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  labelled {start + len(batch)}/{len(todo)} ({len(rows)} parsed)", flush=True)


def load_labels(texts: pd.DataFrame) -> pd.DataFrame:
    """Join cached labels onto review texts as 0/1 theme columns + sentiment."""
    lab = pd.DataFrame([json.loads(l) for l in LABEL_PATH.read_text().splitlines() if l.strip()])
    lab = lab.drop_duplicates("review_id", keep="last")
    df = texts.merge(lab, on="review_id", how="inner")
    for k in THEMES:
        df[k] = df["themes"].apply(lambda ts, k=k: int(k in ts))
    return df.drop(columns=["themes"])


def weekly_voice_features(panel: pd.DataFrame, labelled: pd.DataFrame,
                          prior_n: float = 5.0) -> pd.DataFrame:
    """Leakage-safe SKU-week customer-voice features.

    For SKU i at week t, uses only reviews created before the Monday of t.
    Shares are smoothed toward the catalogue-wide share with `prior_n` pseudo
    reviews, so a SKU with two reviews does not get a 0% or 100% signal.

    Adds columns voice_<theme>, voice_sentiment, voice_n, and
    value_perception = voice_value_good - voice_price_high.
    """
    keys = list(THEMES)
    lab = labelled.copy()
    # A review written during week w is visible from week w+1 onward.
    lab["visible_week"] = lab["review_date"].dt.to_period("W-SUN").dt.start_time + pd.Timedelta(weeks=1)
    agg = lab.groupby(["product_id", "visible_week"])[keys + ["sentiment"]].sum()
    agg["n"] = lab.groupby(["product_id", "visible_week"]).size()
    agg = agg.reset_index().rename(columns={"visible_week": "week"})

    base = labelled[keys + ["sentiment"]].mean()
    grid = panel[["product_id", "week"]].copy()
    merged = pd.merge_ordered(grid.assign(_g=1), agg, on=["product_id", "week"],
                              fill_method=None).sort_values(["product_id", "week"])
    cum = merged.groupby("product_id")[keys + ["sentiment", "n"]].transform(
        lambda s: s.fillna(0).cumsum())
    merged[keys + ["sentiment", "n"]] = cum
    merged = merged[merged["_g"] == 1]
    out = grid.merge(merged.drop(columns="_g"), on=["product_id", "week"], how="left")
    n = out["n"].fillna(0)
    for k in keys + ["sentiment"]:
        out[f"voice_{k}"] = (out[k].fillna(0) + prior_n * base[k]) / (n + prior_n)
    out["voice_n"] = n
    out["value_perception"] = out["voice_value_good"] - out["voice_price_high"]
    return out[["product_id", "week", "voice_n", "value_perception"]
               + [f"voice_{k}" for k in keys + ["sentiment"]]]


if __name__ == "__main__":
    panel = pd.read_parquet(ROOT / "artifacts" / "panel.parquet")
    label_reviews(load_review_texts(set(panel["product_id"])))
