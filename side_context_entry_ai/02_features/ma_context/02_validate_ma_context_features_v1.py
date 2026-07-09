# -*- coding: utf-8 -*-
r"""
02_validate_ma_context_features_v1.py

Validate SCE MA Context features.

Collection:
sce_features_ma_context_m5_v1

Checks:
- BUY/SELL balance
- pair sample: each datetime has BUY and SELL
- null/missing required fields
- numeric consistency for side scores and stack values
- sample-based No-Leak for M30/H1/H4

Command:
python 02_features\ma_context\02_validate_ma_context_features_v1.py --db market_data --features sce_features_ma_context_m5_v1
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from pymongo import MongoClient


SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

TF_MINUTES = {
    "m30": 30,
    "h1": 60,
    "h4": 240,
}

REQUIRED_FIELDS = [
    "datetime", "side", "feature_version",
    "m30_datetime", "h1_datetime", "h4_datetime",
    "m30_close", "h1_close", "h4_close",
    "m30_ma20", "m30_ma50", "m30_ma100",
    "h1_ma20", "h1_ma50", "h1_ma100",
    "h4_ma20", "h4_ma50", "h4_ma100",
    "m30_ma20_slope", "m30_ma50_slope", "m30_ma100_slope",
    "h1_ma20_slope", "h1_ma50_slope", "h1_ma100_slope",
    "h4_ma20_slope", "h4_ma50_slope", "h4_ma100_slope",
    "m30_price_vs_ma20", "m30_price_vs_ma50", "m30_price_vs_ma100",
    "h1_price_vs_ma20", "h1_price_vs_ma50", "h1_price_vs_ma100",
    "h4_price_vs_ma20", "h4_price_vs_ma50", "h4_price_vs_ma100",
    "m30_ma_stack_dir", "h1_ma_stack_dir", "h4_ma_stack_dir",
    "side_m30_stack_alignment", "side_h1_stack_alignment", "side_h4_stack_alignment",
    "side_m30_price_vs_ma20", "side_h1_price_vs_ma20", "side_h4_price_vs_ma20",
    "side_context_stack_aligned_count", "side_context_stack_available_count",
    "side_context_slope_aligned_count", "side_context_slope_available_count",
    "side_context_stack_score", "side_context_slope_score",
]


def log(message: str) -> None:
    now = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def write_report(report_path: Path, lines: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def null_expr(field: str) -> dict[str, Any]:
    return {"$cond": [{"$eq": [{"$ifNull": [f"${field}", None]}, None]}, 1, 0]}


def has_field_expr(field: str) -> dict[str, Any]:
    return {"$ne": [{"$ifNull": [f"${field}", None]}, None]}


def run_one_pass_aggregate(collection) -> dict[str, Any]:
    group_stage: dict[str, Any] = {
        "_id": None,
        "total_docs": {"$sum": 1},
        "buy_count": {"$sum": {"$cond": [{"$eq": ["$side", SIDE_BUY]}, 1, 0]}},
        "sell_count": {"$sum": {"$cond": [{"$eq": ["$side", SIDE_SELL]}, 1, 0]}},
        "other_side_count": {
            "$sum": {
                "$cond": [
                    {"$and": [{"$ne": ["$side", SIDE_BUY]}, {"$ne": ["$side", SIDE_SELL]}]},
                    1,
                    0,
                ]
            }
        },
    }

    for field in REQUIRED_FIELDS:
        group_stage[f"null__{field}"] = {"$sum": null_expr(field)}

    for field in ["side_context_stack_score", "side_context_slope_score"]:
        group_stage[f"invalid_{field}"] = {
            "$sum": {
                "$cond": [
                    {
                        "$and": [
                            has_field_expr(field),
                            {"$or": [{"$lt": [f"${field}", 0]}, {"$gt": [f"${field}", 1]}]},
                        ]
                    },
                    1,
                    0,
                ]
            }
        }

    for field in ["m30_ma_stack_dir", "h1_ma_stack_dir", "h4_ma_stack_dir", "side_m30_stack_alignment", "side_h1_stack_alignment", "side_h4_stack_alignment"]:
        group_stage[f"invalid_{field}"] = {
            "$sum": {
                "$cond": [
                    {
                        "$and": [
                            has_field_expr(field),
                            {"$not": {"$in": [f"${field}", [-1, 0, 1]]}},
                        ]
                    },
                    1,
                    0,
                ]
            }
        }

    result = list(collection.aggregate([{"$group": group_stage}], allowDiskUse=True))
    if not result:
        return {"total_docs": 0, "buy_count": 0, "sell_count": 0, "other_side_count": 0}
    return result[0]


def sample_pair_check(collection, sample_size: int) -> tuple[int, list[str]]:
    bad_count = 0
    examples: list[str] = []

    rows = list(collection.aggregate([
        {"$sample": {"size": sample_size}},
        {"$project": {"_id": 0, "datetime": 1}},
    ], allowDiskUse=True))

    seen = set()
    for row in rows:
        current_dt = row.get("datetime")
        if current_dt is None or current_dt in seen:
            continue
        seen.add(current_dt)

        docs = list(collection.find({"datetime": current_dt}, {"_id": 0, "side": 1}))
        sides = sorted([d.get("side") for d in docs])
        if len(docs) != 2 or sides != [SIDE_BUY, SIDE_SELL]:
            bad_count += 1
            if len(examples) < 10:
                examples.append(f"datetime={current_dt} docs={len(docs)} sides={sides}")

    return bad_count, examples


def sample_no_leak_check(collection, sample_size: int, datetime_is_open: bool) -> tuple[int, list[str]]:
    bad_count = 0
    examples: list[str] = []

    rows = list(collection.aggregate([
        {"$sample": {"size": sample_size}},
        {"$project": {"_id": 0, "datetime": 1, "side": 1, "m30_datetime": 1, "h1_datetime": 1, "h4_datetime": 1}},
    ], allowDiskUse=True))

    for row in rows:
        anchor_time = row.get("datetime")
        if not anchor_time:
            bad_count += 1
            if len(examples) < 10:
                examples.append(f"missing anchor datetime: {row}")
            continue

        for tf, minutes in TF_MINUTES.items():
            tf_time = row.get(f"{tf}_datetime")
            if not tf_time:
                bad_count += 1
                if len(examples) < 10:
                    examples.append(f"missing {tf}_datetime: {row}")
                continue

            if datetime_is_open:
                anchor_close_time = anchor_time + dt.timedelta(minutes=5)
                tf_close_time = tf_time + dt.timedelta(minutes=minutes)
                valid = tf_close_time <= anchor_close_time
            else:
                valid = tf_time <= anchor_time

            if not valid:
                bad_count += 1
                if len(examples) < 10:
                    examples.append(
                        f"NO_LEAK_FAIL anchor={anchor_time} side={row.get('side')} tf={tf} tf_time={tf_time}"
                    )

    return bad_count, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db", default="market_data")
    parser.add_argument("--features", default="sce_features_ma_context_m5_v1")
    parser.add_argument("--datetime-is-open", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--report-dir", default="02_features/reports")
    args = parser.parse_args()

    datetime_is_open = bool(args.datetime_is_open)
    started_at = dt.datetime.now()
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report_dir) / f"sce_validate_ma_context_features_v1_{stamp}.txt"

    log("Connecting to MongoDB...")
    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    collection = db[args.features]

    log("Reading indexes...")
    indexes = list(collection.list_indexes())
    index_names = [idx.get("name", "") for idx in indexes]

    log("Running one-pass aggregate for counts/nulls/numeric rules...")
    agg = run_one_pass_aggregate(collection)

    total_docs = int(agg.get("total_docs", 0))
    buy_count = int(agg.get("buy_count", 0))
    sell_count = int(agg.get("sell_count", 0))
    other_side_count = int(agg.get("other_side_count", 0))

    null_counts = {field: int(agg.get(f"null__{field}", 0)) for field in REQUIRED_FIELDS}
    invalid_counts = {k: int(v) for k, v in agg.items() if str(k).startswith("invalid_")}

    log("Running sample pair check...")
    pair_bad_count, pair_examples = sample_pair_check(collection, args.sample_size)

    log("Running sample No-Leak check...")
    no_leak_bad_count, no_leak_examples = sample_no_leak_check(collection, args.sample_size, datetime_is_open)

    severe_errors: list[str] = []
    warnings: list[str] = []

    if total_docs <= 0:
        severe_errors.append("Collection is empty.")
    if buy_count != sell_count:
        severe_errors.append(f"BUY/SELL mismatch: BUY={buy_count}, SELL={sell_count}")
    if other_side_count > 0:
        severe_errors.append(f"Invalid side values exist: {other_side_count}")
    if total_docs != buy_count + sell_count + other_side_count:
        severe_errors.append("Total docs does not equal side count sum.")

    for field, count in null_counts.items():
        if count > 0:
            severe_errors.append(f"Null/missing required field: {field} count={count}")

    for field, count in invalid_counts.items():
        if count > 0:
            severe_errors.append(f"Invalid numeric field: {field} count={count}")

    if pair_bad_count > 0:
        severe_errors.append(f"Sample pair check failures: {pair_bad_count}")
    if no_leak_bad_count > 0:
        severe_errors.append(f"No-Leak sample failures: {no_leak_bad_count}")

    if "datetime_1_side_1" not in index_names:
        warnings.append("Expected unique index datetime_1_side_1 was not found.")

    status = "PASS"
    if severe_errors:
        status = "FAIL"
    elif warnings:
        status = "WARN"

    finished_at = dt.datetime.now()

    report_lines = [
        "SCE MA Context Feature Validation Report",
        "=" * 80,
        f"started_at             : {started_at}",
        f"finished_at            : {finished_at}",
        f"duration_seconds       : {(finished_at - started_at).total_seconds():.2f}",
        f"status                 : {status}",
        "",
        "Mongo",
        "-" * 80,
        f"mongo_uri              : {args.mongo_uri}",
        f"db                     : {args.db}",
        f"features_collection    : {args.features}",
        "",
        "Counts",
        "-" * 80,
        f"total_docs             : {fmt_int(total_docs)}",
        f"buy_count              : {fmt_int(buy_count)}",
        f"sell_count             : {fmt_int(sell_count)}",
        f"other_side_count       : {fmt_int(other_side_count)}",
        f"expected_buy_sell_sum  : {fmt_int(buy_count + sell_count)}",
        "",
        "Pair Check",
        "-" * 80,
        f"sample_size            : {fmt_int(args.sample_size)}",
        f"sample_pair_bad_count  : {fmt_int(pair_bad_count)}",
        "",
        "No-Leak Sample Check",
        "-" * 80,
        f"datetime_is_open       : {datetime_is_open}",
        f"no_leak_bad_count      : {fmt_int(no_leak_bad_count)}",
        "",
        "Null / Missing Required Fields",
        "-" * 80,
    ]

    for field, count in null_counts.items():
        report_lines.append(f"{field:<45} {fmt_int(count)}")

    report_lines.extend(["", "Invalid Numeric Checks", "-" * 80])
    for field, count in invalid_counts.items():
        report_lines.append(f"{field:<45} {fmt_int(count)}")

    report_lines.extend(["", "Sample Pair Examples", "-" * 80])
    report_lines.extend(pair_examples if pair_examples else ["No sample pair errors."])

    report_lines.extend(["", "No-Leak Examples", "-" * 80])
    report_lines.extend(no_leak_examples if no_leak_examples else ["No no-leak errors in sample."])

    report_lines.extend(["", "Indexes", "-" * 80])
    for idx in indexes:
        report_lines.append(str(idx))

    report_lines.extend(["", "Warnings", "-" * 80])
    report_lines.extend(warnings if warnings else ["No warnings."])

    report_lines.extend(["", "Errors", "-" * 80])
    report_lines.extend(severe_errors if severe_errors else ["No errors."])

    write_report(report_path, report_lines)
    log(f"Report saved: {report_path}")
    print("\n".join(report_lines), flush=True)

    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
