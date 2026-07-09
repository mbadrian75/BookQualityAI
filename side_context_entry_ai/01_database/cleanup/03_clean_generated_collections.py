# -*- coding: utf-8 -*-
"""
03_clean_generated_collections.py

Safe MongoDB cleanup for SCE / BookQuality project.

Purpose:
- Keep raw timeframe collections untouched.
- Drop only generated / temporary / model-output collections from previous runs.
- Write a report file under 01_database/reports.

Default mode is DRY-RUN. Use --apply to actually drop collections.

Example:
python 01_database\cleanup\03_clean_generated_collections.py --db market_data --apply
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Iterable

from pymongo import MongoClient


DEFAULT_KEEP_EXACT = {
    # common raw collection names
    "m1", "m5", "m15", "m30", "h1", "h4", "d1",
    "M1", "M5", "M15", "M30", "H1", "H4", "D1",

    # alternate raw collection names
    "xauusd_m1", "xauusd_m5", "xauusd_m15", "xauusd_m30", "xauusd_h1", "xauusd_h4", "xauusd_d1",
    "XAUUSD_M1", "XAUUSD_M5", "XAUUSD_M15", "XAUUSD_M30", "XAUUSD_H1", "XAUUSD_H4", "XAUUSD_D1",

    # older possible names
    "market_data_m1", "market_data_m5", "market_data_m15", "market_data_m30",
    "market_data_h1", "market_data_h4", "market_data_d1",
}

DEFAULT_DROP_PREFIXES = (
    "sce_features_",
    "sce_labels_",
    "sce_dataset_",
    "sce_predictions_",
    "sce_api_",
    "sce_meta_",

    "bq_features_",
    "bq_labels_",
    "bq_dataset_",
    "bq_predictions_",
    "bq_api_",
    "bq_meta_",

    "tmp_",
    "temp_",
)

DEFAULT_DROP_REGEX = (
    r"^dataset_book_reader_.*",
    r"^features_.*",
    r"^labels_.*",
    r"^predictions_.*",
)


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def should_drop(
    name: str,
    keep_exact: set[str],
    drop_prefixes: tuple[str, ...],
    drop_regex: tuple[str, ...],
    extra_drop_exact: set[str],
) -> tuple[bool, str]:
    if name in keep_exact:
        return False, "kept_raw_or_whitelisted"

    if name in extra_drop_exact:
        return True, "extra_drop_exact"

    for prefix in drop_prefixes:
        if name.startswith(prefix):
            return True, f"drop_prefix:{prefix}"

    for pattern in drop_regex:
        if re.match(pattern, name):
            return True, f"drop_regex:{pattern}"

    return False, "not_matched"


def write_report(report_path: Path, lines: Iterable[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db", default="market_data")
    parser.add_argument("--apply", action="store_true", help="Actually drop collections. Default is dry-run.")
    parser.add_argument("--report-dir", default="01_database/reports")
    parser.add_argument("--keep-extra", default="", help="Comma-separated extra collection names to keep.")
    parser.add_argument("--drop-extra", default="", help="Comma-separated exact collection names to drop.")
    parser.add_argument("--drop-prefix-extra", default="", help="Comma-separated extra prefixes to drop.")
    args = parser.parse_args()

    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report_dir) / f"sce_clean_generated_collections_{now}.txt"

    client = MongoClient(args.mongo_uri)
    db = client[args.db]

    keep_exact = set(DEFAULT_KEEP_EXACT)
    keep_exact.update(split_csv(args.keep_extra))

    drop_prefixes = tuple(DEFAULT_DROP_PREFIXES + tuple(split_csv(args.drop_prefix_extra)))
    drop_regex = tuple(DEFAULT_DROP_REGEX)
    extra_drop_exact = set(split_csv(args.drop_extra))

    names = sorted(db.list_collection_names())

    to_drop: list[tuple[str, str, int]] = []
    kept: list[tuple[str, str, int]] = []

    for name in names:
        count = db[name].estimated_document_count()
        drop, reason = should_drop(name, keep_exact, drop_prefixes, drop_regex, extra_drop_exact)
        if drop:
            to_drop.append((name, reason, count))
        else:
            kept.append((name, reason, count))

    report_lines = [
        "SCE / BQ MongoDB Cleanup Report",
        "=" * 80,
        f"datetime           : {now}",
        f"mongo_uri          : {args.mongo_uri}",
        f"db                 : {args.db}",
        f"mode               : {'APPLY' if args.apply else 'DRY_RUN'}",
        f"collections_total  : {len(names)}",
        f"collections_to_drop: {len(to_drop)}",
        f"collections_kept   : {len(kept)}",
        "",
        "DROP LIST",
        "-" * 80,
    ]

    if not to_drop:
        report_lines.append("No generated collections matched the drop rules.")
    else:
        for name, reason, count in to_drop:
            report_lines.append(f"{name:<60} docs≈{count:<12} reason={reason}")

    report_lines.extend(["", "KEEP LIST", "-" * 80])
    for name, reason, count in kept:
        report_lines.append(f"{name:<60} docs≈{count:<12} reason={reason}")

    if args.apply:
        report_lines.extend(["", "DROP EXECUTION", "-" * 80])
        for name, reason, count in to_drop:
            db.drop_collection(name)
            report_lines.append(f"DROPPED: {name} | docs≈{count} | reason={reason}")
    else:
        report_lines.extend([
            "",
            "DRY-RUN NOTE",
            "-" * 80,
            "No collection was deleted.",
            "Run again with --apply to execute the cleanup.",
        ])

    write_report(report_path, report_lines)

    print("\n".join(report_lines[:20]))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
