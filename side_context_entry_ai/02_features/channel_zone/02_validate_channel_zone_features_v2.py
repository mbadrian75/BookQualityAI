# -*- coding: utf-8 -*-
r"""
02_validate_channel_zone_features_v2.py

Fast + progress version - v2.

Update in v2:
price_position_in_channel outside [-1, 2] is treated as a warning/observation, not a fatal error.
Strong breakouts can naturally produce values outside this range.

Purpose:
Validate sce_features_channel_zone_entry_m5_v2 without long silent waits.

Default mode is QUICK QA:
- count documents
- count BUY / SELL / invalid side
- verify indexes
- one-pass aggregation for null fields and numeric rules
- sample-based pair check
- sample-based No-Leak check

Optional full pair check:
Add --full-pair-check if you want a complete datetime group validation.
This can be slower on millions of rows.

Command:
python 02_features\channel_zone\02_validate_channel_zone_features_v2.py --db market_data --features sce_features_channel_zone_entry_m5_v2
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from pymongo import ASCENDING, MongoClient


SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

REQUIRED_FIELDS = [
    "datetime", "side", "feature_version",
    "open", "high", "low", "close",
    "channel_high", "channel_low", "channel_mid", "channel_range",
    "price_position_in_channel",
    "high_zone_low", "high_zone_high", "high_zone_thickness", "high_zone_m5_datetime",
    "low_zone_low", "low_zone_high", "low_zone_thickness", "low_zone_m5_datetime",
    "last_closed_m15_datetime",
    "close_above_high_zone", "close_below_high_zone", "close_inside_high_zone",
    "close_above_low_zone", "close_below_low_zone", "close_inside_low_zone",
    "high_zone_break_up", "high_zone_break_down", "low_zone_break_up", "low_zone_break_down",
    "side_zone_primary_break", "side_adverse_zone_break", "side_distance_to_mid",
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


def has_all_expr(*fields: str) -> dict[str, Any]:
    return {
        "$and": [
            {"$ne": [{"$ifNull": [f"${field}", None]}, None]}
            for field in fields
        ]
    }


def invalid_numeric_group_expressions() -> dict[str, Any]:
    return {
        "invalid_channel_high_low": {
            "$sum": {
                "$cond": [
                    {
                        "$and": [
                            has_all_expr("channel_high", "channel_low"),
                            {"$lte": ["$channel_high", "$channel_low"]},
                        ]
                    },
                    1,
                    0,
                ]
            }
        },
        "invalid_channel_range": {
            "$sum": {
                "$cond": [
                    {
                        "$and": [
                            has_all_expr("channel_range"),
                            {"$lte": ["$channel_range", 0]},
                        ]
                    },
                    1,
                    0,
                ]
            }
        },
        "invalid_high_zone": {
            "$sum": {
                "$cond": [
                    {
                        "$and": [
                            has_all_expr("high_zone_high", "high_zone_low"),
                            {"$lt": ["$high_zone_high", "$high_zone_low"]},
                        ]
                    },
                    1,
                    0,
                ]
            }
        },
        "invalid_low_zone": {
            "$sum": {
                "$cond": [
                    {
                        "$and": [
                            has_all_expr("low_zone_high", "low_zone_low"),
                            {"$lt": ["$low_zone_high", "$low_zone_low"]},
                        ]
                    },
                    1,
                    0,
                ]
            }
        },
        "invalid_price_position_low": {
            "$sum": {
                "$cond": [
                    {
                        "$and": [
                            has_all_expr("price_position_in_channel"),
                            {"$lt": ["$price_position_in_channel", -1]},
                        ]
                    },
                    1,
                    0,
                ]
            }
        },
        "invalid_price_position_high": {
            "$sum": {
                "$cond": [
                    {
                        "$and": [
                            has_all_expr("price_position_in_channel"),
                            {"$gt": ["$price_position_in_channel", 2]},
                        ]
                    },
                    1,
                    0,
                ]
            }
        },
    }


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

    group_stage.update(invalid_numeric_group_expressions())

    result = list(collection.aggregate([{"$group": group_stage}], allowDiskUse=True))
    if not result:
        return {"total_docs": 0, "buy_count": 0, "sell_count": 0, "other_side_count": 0}

    return result[0]


def sample_pair_check(collection, sample_size: int) -> tuple[int, list[str]]:
    bad_count = 0
    examples: list[str] = []

    pipeline = [
        {"$sample": {"size": sample_size}},
        {"$project": {"_id": 0, "datetime": 1}},
    ]

    rows = list(collection.aggregate(pipeline, allowDiskUse=True))
    seen = set()

    for row in rows:
        current_dt = row.get("datetime")
        if current_dt is None or current_dt in seen:
            continue
        seen.add(current_dt)

        docs = list(
            collection.find(
                {"datetime": current_dt},
                {"_id": 0, "datetime": 1, "side": 1},
            )
        )
        sides = sorted([d.get("side") for d in docs])

        if len(docs) != 2 or sides != [SIDE_BUY, SIDE_SELL]:
            bad_count += 1
            if len(examples) < 10:
                examples.append(f"datetime={current_dt} docs={len(docs)} sides={sides}")

    return bad_count, examples


def sample_no_leak_check(collection, sample_size: int, datetime_is_open: bool) -> tuple[int, list[str]]:
    bad_count = 0
    examples: list[str] = []

    pipeline = [
        {"$sample": {"size": sample_size}},
        {
            "$project": {
                "_id": 0,
                "datetime": 1,
                "side": 1,
                "last_closed_m15_datetime": 1,
            }
        },
    ]

    rows = list(collection.aggregate(pipeline, allowDiskUse=True))

    for row in rows:
        anchor_time = row.get("datetime")
        last_m15_time = row.get("last_closed_m15_datetime")

        if not anchor_time or not last_m15_time:
            bad_count += 1
            if len(examples) < 10:
                examples.append(f"missing datetime fields: {row}")
            continue

        if datetime_is_open:
            anchor_close_time = anchor_time + dt.timedelta(minutes=5)
            m15_close_time = last_m15_time + dt.timedelta(minutes=15)
            valid = m15_close_time <= anchor_close_time
        else:
            valid = last_m15_time <= anchor_time

        if not valid:
            bad_count += 1
            if len(examples) < 10:
                examples.append(
                    f"datetime={anchor_time} side={row.get('side')} last_m15={last_m15_time}"
                )

    return bad_count, examples


def full_pair_check(collection, limit_examples: int = 10) -> tuple[int, list[str]]:
    pipeline = [
        {
            "$group": {
                "_id": "$datetime",
                "count": {"$sum": 1},
                "sides": {"$addToSet": "$side"},
            }
        },
        {
            "$match": {
                "$or": [
                    {"count": {"$ne": 2}},
                    {"sides": {"$not": {"$all": [SIDE_BUY, SIDE_SELL]}}},
                ]
            }
        },
        {"$limit": limit_examples},
    ]

    examples_raw = list(collection.aggregate(pipeline, allowDiskUse=True))
    examples = [str(x) for x in examples_raw]
    return len(examples_raw), examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db", default="market_data")
    parser.add_argument("--features", default="sce_features_channel_zone_entry_m5_v2")
    parser.add_argument("--datetime-is-open", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--full-pair-check", action="store_true")
    parser.add_argument("--report-dir", default="02_features/reports")
    args = parser.parse_args()

    datetime_is_open = bool(args.datetime_is_open)
    started_at = dt.datetime.now()
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report_dir) / f"sce_validate_channel_zone_features_v2_{stamp}.txt"

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
    invalid_numeric_rules = {
        key: int(agg.get(key, 0))
        for key in [
            "invalid_channel_high_low",
            "invalid_channel_range",
            "invalid_high_zone",
            "invalid_low_zone",
            "invalid_price_position_low",
            "invalid_price_position_high",
        ]
    }

    log("Running sample pair check...")
    pair_bad_count, pair_examples = sample_pair_check(collection, args.sample_size)

    log("Running sample No-Leak check...")
    no_leak_bad_count, no_leak_examples = sample_no_leak_check(
        collection, args.sample_size, datetime_is_open
    )

    full_pair_bad_count = 0
    full_pair_examples: list[str] = []
    if args.full_pair_check:
        log("Running FULL pair check. This can take longer...")
        full_pair_bad_count, full_pair_examples = full_pair_check(collection)

    severe_errors: list[str] = []
    warnings: list[str] = []

    if total_docs <= 0:
        severe_errors.append("Collection is empty.")
    if buy_count != sell_count:
        severe_errors.append(f"BUY/SELL mismatch: BUY={buy_count}, SELL={sell_count}")
    if other_side_count > 0:
        severe_errors.append(f"Invalid side values exist: {other_side_count}")
    if total_docs != buy_count + sell_count + other_side_count:
        severe_errors.append("Total docs does not match side count sum.")

    for field, count in null_counts.items():
        if count > 0:
            severe_errors.append(f"Null/missing required field: {field} count={count}")

    position_observation_rules = {
        "invalid_price_position_low",
        "invalid_price_position_high",
    }

    for rule_name, count in invalid_numeric_rules.items():
        if count <= 0:
            continue

        if rule_name in position_observation_rules:
            warnings.append(
                f"Price position outside observation range: {rule_name} count={count}. "
                "This can happen during strong channel breakouts and is not treated as data corruption."
            )
        else:
            severe_errors.append(f"Invalid numeric rule: {rule_name} count={count}")

    if pair_bad_count > 0:
        severe_errors.append(f"Sample pair check failures: {pair_bad_count}")
    if no_leak_bad_count > 0:
        severe_errors.append(f"No-Leak sample failures: {no_leak_bad_count}")
    if full_pair_bad_count > 0:
        severe_errors.append(f"Full pair check failures: {full_pair_bad_count}")

    if "datetime_1_side_1" not in index_names:
        warnings.append("Expected index name datetime_1_side_1 was not found.")

    status = "PASS"
    if severe_errors:
        status = "FAIL"
    elif warnings:
        status = "WARN"

    finished_at = dt.datetime.now()
    duration_seconds = (finished_at - started_at).total_seconds()

    report_lines = [
        "SCE Channel-Zone Feature Validation Report - FAST v2",
        "=" * 80,
        f"started_at             : {started_at}",
        f"finished_at            : {finished_at}",
        f"duration_seconds       : {duration_seconds:.2f}",
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
        f"full_pair_check        : {args.full_pair_check}",
        f"full_pair_bad_count    : {fmt_int(full_pair_bad_count)}",
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
        report_lines.append(f"{field:<42} {fmt_int(count)}")

    report_lines.extend(["", "Invalid Numeric Rules", "-" * 80])
    for rule_name, count in invalid_numeric_rules.items():
        report_lines.append(f"{rule_name:<42} {fmt_int(count)}")

    report_lines.extend(["", "Sample Pair Examples", "-" * 80])
    if pair_examples:
        report_lines.extend(pair_examples)
    else:
        report_lines.append("No sample pair errors.")

    report_lines.extend(["", "No-Leak Examples", "-" * 80])
    if no_leak_examples:
        report_lines.extend(no_leak_examples)
    else:
        report_lines.append("No no-leak errors in sample.")

    report_lines.extend(["", "Indexes", "-" * 80])
    for idx in indexes:
        report_lines.append(str(idx))

    report_lines.extend(["", "Warnings", "-" * 80])
    if warnings:
        report_lines.extend(warnings)
    else:
        report_lines.append("No warnings.")

    report_lines.extend(["", "Errors", "-" * 80])
    if severe_errors:
        report_lines.extend(severe_errors)
    else:
        report_lines.append("No errors.")

    write_report(report_path, report_lines)
    log(f"Report saved: {report_path}")
    print("\n".join(report_lines), flush=True)

    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
