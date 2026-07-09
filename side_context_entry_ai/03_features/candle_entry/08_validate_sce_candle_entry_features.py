# -*- coding: utf-8 -*-
"""
SCE Candle Entry Feature QA

Location:
03_features/candle_entry/

Report:
03_features/candle_entry/reports/

Purpose:
- Validate candle entry feature collection after build
- Does not modify database
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from pymongo import MongoClient
except ImportError as exc:
    print("ERROR: pymongo is not installed. Run: pip install pymongo")
    raise exc

from config.sce_project_config import (
    COLL_FEATURES_CANDLE_ENTRY_M5_V1,
    DB_NAME,
    FEATURE_VERSION_CANDLE_ENTRY_M5_V1,
    MONGO_URI,
    PROJECT_CODE,
    PROJECT_NAME,
    SIDES,
)

RUN_TYPE = "sce_validate_candle_entry_features"
RUN_ID = f"{RUN_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

CRITICAL_FIELDS = [
    "anchor_time", "entry_time", "side", "entry_price", "feature_version",
    "m5_open", "m5_high", "m5_low", "m5_close", "m5_range", "m5_body", "m5_body_ratio", "m5_close_position", "m5_direction",
    "m5_side_body_power", "m5_side_rejection_wick_ratio", "m5_side_close_position",
    "m15_open", "m15_high", "m15_low", "m15_close", "m15_range", "m15_body", "m15_body_ratio", "m15_close_position", "m15_direction",
    "m15_side_body_power", "m15_side_rejection_wick_ratio", "m15_side_close_position",
    "side_body_power_total", "side_rejection_wick_total", "candle_context_agreement", "candle_rejection_agreement", "candle_breakout_agreement", "candle_failed_breakout_support",
]

BOOL_FIELDS = [
    "m5_is_doji", "m15_is_doji", "m5_side_engulfing", "m15_side_engulfing", "m5_inside_bar", "m15_inside_bar",
    "m5_side_breakout", "m15_side_breakout", "m5_side_failed_breakout", "m15_side_failed_breakout",
    "m5_is_expansion", "m15_is_expansion", "m5_is_compression", "m15_is_compression",
    "candle_context_agreement", "candle_rejection_agreement", "candle_breakout_agreement", "candle_failed_breakout_support",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SCE candle entry features.")
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--database", default=DB_NAME)
    parser.add_argument("--collection", default=COLL_FEATURES_CANDLE_ENTRY_M5_V1)
    parser.add_argument("--feature-version", default=FEATURE_VERSION_CANDLE_ENTRY_M5_V1)
    parser.add_argument("--check-pairs", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def run_pipeline(coll, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(coll.aggregate(pipeline, allowDiskUse=True))


def kv_counts(coll, field: str, query: Dict[str, Any]) -> Dict[str, int]:
    rows = run_pipeline(coll, [{"$match": query}, {"$group": {"_id": f"${field}", "count": {"$sum": 1}}}, {"$sort": {"_id": 1}}])
    return {str(row["_id"]): int(row["count"]) for row in rows}


def bool_counts(coll, field: str, query: Dict[str, Any]) -> Dict[str, int]:
    rows = run_pipeline(coll, [{"$match": query}, {"$group": {"_id": f"${field}", "count": {"$sum": 1}}}, {"$sort": {"_id": 1}}])
    return {str(row["_id"]): int(row["count"]) for row in rows}


def pair_integrity(coll, query: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_pipeline(coll, [
        {"$match": query},
        {"$group": {"_id": "$anchor_time", "row_count": {"$sum": 1}, "sides": {"$addToSet": "$side"}}},
        {"$project": {"row_count": 1, "side_count": {"$size": "$sides"}}},
        {"$group": {"_id": {"row_count": "$row_count", "side_count": "$side_count"}, "anchor_count": {"$sum": 1}}},
        {"$sort": {"_id.row_count": 1, "_id.side_count": 1}},
    ])


def year_side_counts(coll, query: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_pipeline(coll, [
        {"$match": query},
        {"$project": {"year": {"$year": "$anchor_time"}, "side": 1}},
        {"$group": {"_id": {"year": "$year", "side": "$side"}, "count": {"$sum": 1}}},
        {"$sort": {"_id.year": 1, "_id.side": 1}},
    ])


def pct(part: int, total: int) -> str:
    return "0.0000%" if total == 0 else f"{(part / total) * 100:.4f}%"


def write_report(args: argparse.Namespace, status: str, stats: Dict[str, Any], error_text: str = "") -> Path:
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{RUN_ID}.txt"
    total = int(stats.get("total_docs", 0))
    lines: List[str] = []
    lines.append("Side Context Entry AI - Candle Entry Feature QA Report")
    lines.append("=" * 76)
    lines.append(f"run_id                   : {RUN_ID}")
    lines.append(f"status                   : {status}")
    lines.append(f"project_name             : {PROJECT_NAME}")
    lines.append(f"project_code             : {PROJECT_CODE}")
    lines.append(f"database                 : {args.database}")
    lines.append(f"collection               : {args.collection}")
    lines.append(f"feature_version           : {args.feature_version}")
    lines.append("database_modified         : False")
    lines.append("raw_data_modified         : False")
    lines.append("report_folder_policy      : local_section_reports")
    lines.append("")
    lines.append("Summary:")
    for key, value in stats.get("summary", {}).items():
        lines.append(f"{key:34}: {value}")
    lines.append("")
    lines.append("Side counts:")
    for key, value in stats.get("side_counts", {}).items():
        lines.append(f"{key:34}: {value} | {pct(value, total)}")
    lines.append("")
    lines.append("Direction distributions:")
    for field in ["m5_direction", "m15_direction"]:
        lines.append(f"{field}:")
        for key, value in stats.get(field, {}).items():
            lines.append(f"  {key:30}: {value} | {pct(value, total)}")
    lines.append("")
    lines.append("Boolean feature distributions:")
    for field in BOOL_FIELDS:
        lines.append(f"{field}:")
        for key, value in stats.get(field, {}).items():
            lines.append(f"  {key:30}: {value} | {pct(value, total)}")
    lines.append("")
    lines.append("Missing critical fields:")
    for key, value in stats.get("missing_fields", {}).items():
        lines.append(f"{key:34}: {value}")
    lines.append("")
    lines.append("Pair integrity:")
    for row in stats.get("pair_integrity", []):
        lines.append(f"row_count={row['_id'].get('row_count')} | side_count={row['_id'].get('side_count')} : {row.get('anchor_count')}")
    lines.append("")
    lines.append("Year x Side counts:")
    for row in stats.get("year_side_counts", []):
        rid = row["_id"]
        lines.append(f"year={rid.get('year')} | side={rid.get('side')} : {row.get('count')}")
    if error_text:
        lines.append("")
        lines.append("Error:")
        lines.append(error_text)
    lines.append("")
    lines.append("End of report.")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    client = None
    stats: Dict[str, Any] = {}
    try:
        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[args.database]
        coll = db[args.collection]
        query = {"feature_version": args.feature_version}
        total_docs = coll.count_documents(query)
        invalid_side_count = coll.count_documents({**query, "side": {"$nin": SIDES}})
        missing_fields = {field: coll.count_documents({**query, field: {"$exists": False}}) for field in CRITICAL_FIELDS}
        side_counts = kv_counts(coll, "side", query)
        pairs = pair_integrity(coll, query) if args.check_pairs == 1 else []
        unique_anchor_count = sum(int(row.get("anchor_count", 0)) for row in pairs) if pairs else None
        stats = {
            "total_docs": total_docs,
            "summary": {"total_docs": total_docs, "unique_anchor_count": unique_anchor_count, "invalid_side_count": invalid_side_count, "check_pairs": args.check_pairs},
            "side_counts": side_counts,
            "missing_fields": missing_fields,
            "pair_integrity": pairs,
            "year_side_counts": year_side_counts(coll, query),
            "m5_direction": kv_counts(coll, "m5_direction", query),
            "m15_direction": kv_counts(coll, "m15_direction", query),
        }
        for field in BOOL_FIELDS:
            stats[field] = bool_counts(coll, field, query)
        report_path = write_report(args, "success", stats)
        print("SCE candle entry feature QA completed.")
        print(f"Report: {report_path}")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        print("ERROR: SCE candle entry feature QA failed.")
        print(error_text)
        report_path = write_report(args, "failed", stats, error_text)
        print(f"Failure report: {report_path}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
