# -*- coding: utf-8 -*-
r"""
01_build_ma_context_features_v1.py

Build higher-timeframe MA context features for SCE.

Fix v2:
Requires at least index >= 100 for each higher timeframe so MA100 slope is never null.

This script keeps the higher timeframe design unchanged:
- M30 / H1 / H4
- MA 20 / 50 / 100
- side-based rows: BUY and SELL per M5 anchor

Output collection by default:
sce_features_ma_context_m5_v1

No-Leak:
If candle datetime is OPEN time, a higher-TF candle is usable only when:
    tf_open + tf_minutes <= m5_open + 5min

Command:
python 02_features\ma_context\01_build_ma_context_features_v1.py --db market_data --drop-output
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import ASCENDING, MongoClient, UpdateOne


SIDE_BUY = "BUY"
SIDE_SELL = "SELL"
TF_MINUTES = {
    "m30": 30,
    "h1": 60,
    "h4": 240,
}


def parse_dt(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def sign_for_side(side: str) -> int:
    return 1 if side == SIDE_BUY else -1


def load_candles(db, collection: str, datetime_field: str, start, end, fields: list[str]) -> pd.DataFrame:
    query: dict[str, Any] = {}
    if start or end:
        query[datetime_field] = {}
        if start:
            query[datetime_field]["$gte"] = start
        if end:
            query[datetime_field]["$lt"] = end

    projection = {field: 1 for field in fields}
    projection["_id"] = 0

    rows = list(db[collection].find(query, projection).sort(datetime_field, ASCENDING))
    if not rows:
        raise RuntimeError(f"No rows found in collection={collection} query={query}")

    df = pd.DataFrame(rows)
    df[datetime_field] = pd.to_datetime(df[datetime_field])
    df = df.sort_values(datetime_field).reset_index(drop=True)

    for price_col in ["open", "high", "low", "close"]:
        if price_col in df.columns:
            df[price_col] = df[price_col].astype(float)

    return df


def add_ma_features(df: pd.DataFrame, close_col: str, periods: list[int]) -> pd.DataFrame:
    df = df.copy()
    close = df[close_col].astype(float)

    for period in periods:
        ma_col = f"ma{period}"
        slope_col = f"ma{period}_slope"
        df[ma_col] = close.rolling(period, min_periods=period).mean()
        df[slope_col] = df[ma_col] - df[ma_col].shift(1)

    return df


def build_tf_raw_features(tf_name: str, row: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    close = safe_float(row[args.close])
    ma20 = safe_float(row["ma20"])
    ma50 = safe_float(row["ma50"])
    ma100 = safe_float(row["ma100"])
    ma20_slope = safe_float(row["ma20_slope"])
    ma50_slope = safe_float(row["ma50_slope"])
    ma100_slope = safe_float(row["ma100_slope"])

    if ma20 is not None and ma50 is not None and ma100 is not None:
        if ma20 > ma50 > ma100:
            ma_stack_dir = 1
        elif ma20 < ma50 < ma100:
            ma_stack_dir = -1
        else:
            ma_stack_dir = 0
    else:
        ma_stack_dir = None

    return {
        f"{tf_name}_datetime": row[args.datetime_field].to_pydatetime(),
        f"{tf_name}_close": close,

        f"{tf_name}_ma20": ma20,
        f"{tf_name}_ma50": ma50,
        f"{tf_name}_ma100": ma100,

        f"{tf_name}_ma20_slope": ma20_slope,
        f"{tf_name}_ma50_slope": ma50_slope,
        f"{tf_name}_ma100_slope": ma100_slope,

        f"{tf_name}_price_vs_ma20": safe_float(close - ma20) if close is not None and ma20 is not None else None,
        f"{tf_name}_price_vs_ma50": safe_float(close - ma50) if close is not None and ma50 is not None else None,
        f"{tf_name}_price_vs_ma100": safe_float(close - ma100) if close is not None and ma100 is not None else None,

        f"{tf_name}_ma20_gt_ma50": None if ma20 is None or ma50 is None else int(ma20 > ma50),
        f"{tf_name}_ma50_gt_ma100": None if ma50 is None or ma100 is None else int(ma50 > ma100),
        f"{tf_name}_ma_stack_dir": ma_stack_dir,
    }


def add_side_context_features(base: dict[str, Any], side: str, tf_names: list[str]) -> dict[str, Any]:
    side_sign = sign_for_side(side)
    result: dict[str, Any] = {
        "side": side,
        "side_sign": side_sign,
    }

    aligned_count = 0
    available_count = 0
    slope_aligned_count = 0
    slope_available_count = 0

    for tf_name in tf_names:
        stack_dir = base.get(f"{tf_name}_ma_stack_dir")
        ma20_slope = base.get(f"{tf_name}_ma20_slope")
        price_vs_ma20 = base.get(f"{tf_name}_price_vs_ma20")
        price_vs_ma50 = base.get(f"{tf_name}_price_vs_ma50")
        price_vs_ma100 = base.get(f"{tf_name}_price_vs_ma100")

        if stack_dir is not None:
            available_count += 1
            if side_sign * stack_dir > 0:
                aligned_count += 1

        if ma20_slope is not None:
            slope_available_count += 1
            if side_sign * ma20_slope > 0:
                slope_aligned_count += 1

        result[f"side_{tf_name}_stack_alignment"] = (
            None if stack_dir is None else int(side_sign * stack_dir)
        )
        result[f"side_{tf_name}_ma20_slope"] = (
            None if ma20_slope is None else safe_float(side_sign * ma20_slope)
        )
        result[f"side_{tf_name}_price_vs_ma20"] = (
            None if price_vs_ma20 is None else safe_float(side_sign * price_vs_ma20)
        )
        result[f"side_{tf_name}_price_vs_ma50"] = (
            None if price_vs_ma50 is None else safe_float(side_sign * price_vs_ma50)
        )
        result[f"side_{tf_name}_price_vs_ma100"] = (
            None if price_vs_ma100 is None else safe_float(side_sign * price_vs_ma100)
        )

    result["side_context_stack_aligned_count"] = aligned_count
    result["side_context_stack_available_count"] = available_count
    result["side_context_slope_aligned_count"] = slope_aligned_count
    result["side_context_slope_available_count"] = slope_available_count

    result["side_context_stack_score"] = (
        None if available_count == 0 else safe_float(aligned_count / available_count)
    )
    result["side_context_slope_score"] = (
        None if slope_available_count == 0 else safe_float(slope_aligned_count / slope_available_count)
    )

    return result


def find_last_closed_index(tf_times: np.ndarray, anchor_time: pd.Timestamp, tf_minutes: int, datetime_is_open: bool) -> int:
    if datetime_is_open:
        anchor_close_time = anchor_time + pd.Timedelta(minutes=5)
        last_allowed_tf_time = anchor_close_time - pd.Timedelta(minutes=tf_minutes)
    else:
        last_allowed_tf_time = anchor_time

    return int(np.searchsorted(tf_times, np.datetime64(last_allowed_tf_time), side="right") - 1)


def write_report(report_path: Path, lines: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db", default="market_data")

    parser.add_argument("--m5", default="xauusd_m5")
    parser.add_argument("--m30", default="xauusd_m30")
    parser.add_argument("--h1", default="xauusd_h1")
    parser.add_argument("--h4", default="xauusd_h4")

    parser.add_argument("--out", default="sce_features_ma_context_m5_v1")
    parser.add_argument("--drop-output", action="store_true")

    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--datetime-field", default="datetime")
    parser.add_argument("--datetime-is-open", type=int, default=1)

    parser.add_argument("--open", default="open")
    parser.add_argument("--high", default="high")
    parser.add_argument("--low", default="low")
    parser.add_argument("--close", default="close")

    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--report-dir", default="02_features/reports")
    args = parser.parse_args()

    args.datetime_is_open = bool(args.datetime_is_open)

    started_at = dt.datetime.now()
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report_dir) / f"sce_build_ma_context_features_v1_{stamp}.txt"

    start = parse_dt(args.start)
    end = parse_dt(args.end)
    load_start = start - dt.timedelta(days=60) if start else None

    client = MongoClient(args.mongo_uri)
    db = client[args.db]

    fields = [args.datetime_field, args.open, args.high, args.low, args.close]

    print("Loading M5 anchor candles...", flush=True)
    m5 = load_candles(db, args.m5, args.datetime_field, load_start, end, fields)

    tf_collections = {
        "m30": args.m30,
        "h1": args.h1,
        "h4": args.h4,
    }

    tf_data: dict[str, pd.DataFrame] = {}
    tf_times: dict[str, np.ndarray] = {}

    for tf_name, collection_name in tf_collections.items():
        print(f"Loading {tf_name.upper()} candles...", flush=True)
        df = load_candles(db, collection_name, args.datetime_field, load_start, end, fields)
        print(f"Calculating MA20/50/100 for {tf_name.upper()}...", flush=True)
        df = add_ma_features(df, args.close, [20, 50, 100])
        tf_data[tf_name] = df
        tf_times[tf_name] = df[args.datetime_field].to_numpy(dtype="datetime64[ns]")

    if args.drop_output:
        print(f"Dropping output collection: {args.out}", flush=True)
        db.drop_collection(args.out)

    out = db[args.out]
    out.create_index([("datetime", ASCENDING), ("side", ASCENDING)], unique=True)
    out.create_index([("datetime", ASCENDING)])
    out.create_index([("side", ASCENDING)])
    out.create_index([("feature_version", ASCENDING)])

    ops: list[UpdateOne] = []
    processed_anchor_rows = 0
    skipped_no_history = 0
    written_or_updated = 0
    batches = 0

    print("Building MA context features...", flush=True)

    for i in range(len(m5)):
        anchor = m5.iloc[i]
        anchor_time = anchor[args.datetime_field]

        if start and anchor_time.to_pydatetime() < start:
            continue
        if end and anchor_time.to_pydatetime() >= end:
            continue

        processed_anchor_rows += 1

        indices: dict[str, int] = {}
        has_history = True

        for tf_name in ["m30", "h1", "h4"]:
            idx = find_last_closed_index(
                tf_times=tf_times[tf_name],
                anchor_time=anchor_time,
                tf_minutes=TF_MINUTES[tf_name],
                datetime_is_open=args.datetime_is_open,
            )
            indices[tf_name] = idx
            if idx < 100:
                has_history = False
                break

        if not has_history:
            skipped_no_history += 1
            continue

        base: dict[str, Any] = {
            "datetime": anchor_time.to_pydatetime(),
            "feature_version": "sce_ma_context_m5_v1",
            "anchor_tf": "M5",
            "context_tfs": ["M30", "H1", "H4"],
            "created_at": dt.datetime.now(),
        }

        for tf_name in ["m30", "h1", "h4"]:
            row = tf_data[tf_name].iloc[indices[tf_name]]
            base.update(build_tf_raw_features(tf_name, row, args))

        for side in [SIDE_BUY, SIDE_SELL]:
            doc = dict(base)
            doc.update(add_side_context_features(base, side, ["m30", "h1", "h4"]))
            ops.append(UpdateOne({"datetime": doc["datetime"], "side": doc["side"]}, {"$set": doc}, upsert=True))

        if len(ops) >= args.batch_size:
            result = out.bulk_write(ops, ordered=False)
            written_or_updated += result.upserted_count + result.modified_count
            batches += 1
            ops.clear()
            if batches % 20 == 0:
                print(
                    f"batches={batches:,} | processed_anchor_rows={processed_anchor_rows:,} | written_or_updated≈{written_or_updated:,}",
                    flush=True,
                )

    if ops:
        result = out.bulk_write(ops, ordered=False)
        written_or_updated += result.upserted_count + result.modified_count
        batches += 1
        ops.clear()

    finished_at = dt.datetime.now()
    total_docs = out.estimated_document_count()

    report_lines = [
        "SCE MA Context Feature Build Report",
        "=" * 80,
        f"started_at             : {started_at}",
        f"finished_at            : {finished_at}",
        f"duration_seconds       : {(finished_at - started_at).total_seconds():.2f}",
        "",
        "Mongo",
        "-" * 80,
        f"mongo_uri              : {args.mongo_uri}",
        f"db                     : {args.db}",
        f"m5_collection          : {args.m5}",
        f"m30_collection         : {args.m30}",
        f"h1_collection          : {args.h1}",
        f"h4_collection          : {args.h4}",
        f"output_collection      : {args.out}",
        f"drop_output            : {args.drop_output}",
        "",
        "Counts",
        "-" * 80,
        f"m5_rows_loaded         : {len(m5):,}",
        f"m30_rows_loaded        : {len(tf_data['m30']):,}",
        f"h1_rows_loaded         : {len(tf_data['h1']):,}",
        f"h4_rows_loaded         : {len(tf_data['h4']):,}",
        f"processed_anchor_rows  : {processed_anchor_rows:,}",
        f"skipped_no_history     : {skipped_no_history:,}",
        f"bulk_batches           : {batches:,}",
        f"written_or_updated≈    : {written_or_updated:,}",
        f"output_docs_total≈     : {total_docs:,}",
        "",
        "Feature version",
        "-" * 80,
        "sce_ma_context_m5_v1",
        "",
        "Design",
        "-" * 80,
        "Higher timeframe context unchanged: M30/H1/H4 with MA20/50/100.",
        "This collection is side-based and joins with lower entry features by datetime + side.",
    ]

    write_report(report_path, report_lines)
    print("\n".join(report_lines), flush=True)
    print(f"\nReport saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
