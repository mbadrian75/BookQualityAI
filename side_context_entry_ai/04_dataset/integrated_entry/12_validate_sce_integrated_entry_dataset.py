# -*- coding: utf-8 -*-
"""
SCE Integrated Entry Dataset QA

Location:
04_dataset/integrated_entry/

Report:
04_dataset/integrated_entry/reports/

Purpose:
- Validate integrated dataset after build
- Checks target columns, split, pair integrity, and critical feature fields
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
    COLL_DATASET_INTEGRATED_ENTRY_M5_V1,
    DATASET_VERSION_INTEGRATED_ENTRY_M5_V1,
    DB_NAME,
    ENTRY_LABELS,
    ENTRY_TP_BUCKETS,
    FEATURE_VERSION_INTEGRATED_ENTRY_M5_V1,
    LABEL_VERSION_ENTRY_SIDE_M5_V1,
    MONGO_URI,
    PROJECT_CODE,
    PROJECT_NAME,
    SIDES,
)

RUN_TYPE = "sce_validate_integrated_entry_dataset"
RUN_ID = f"{RUN_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
VALID_SPLITS = ["train", "valid", "test"]
CRITICAL_FIELDS = [
    "anchor_time", "entry_time", "side", "entry_price", "feature_version", "label_version", "dataset_version", "split",
    "entry_label", "entry_label_id", "is_good_entry", "entry_tp_bucket", "entry_tp_bucket_id", "max_entry_success_tp_atr",
    "m30_ma20", "m30_ma50", "m30_ma100", "h1_ma20", "h1_ma50", "h1_ma100", "h4_ma20", "h4_ma50", "h4_ma100",
    "ma_context_side_alignment_total", "ma_context_side_slope_total", "ma_context_tp_capacity_min", "ma_context_tp_capacity_avg",
    "m5_open", "m5_high", "m5_low", "m5_close", "m5_direction", "m5_side_body_power", "m5_side_rejection_wick_ratio",
    "m15_open", "m15_high", "m15_low", "m15_close", "m15_direction", "m15_side_body_power", "m15_side_rejection_wick_ratio",
    "side_body_power_total", "side_rejection_wick_total", "candle_context_agreement", "candle_rejection_agreement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SCE integrated entry dataset.")
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--database", default=DB_NAME)
    parser.add_argument("--collection", default=COLL_DATASET_INTEGRATED_ENTRY_M5_V1)
    parser.add_argument("--dataset-version", default=DATASET_VERSION_INTEGRATED_ENTRY_M5_V1)
    parser.add_argument("--check-pairs", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def run_pipeline(coll, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(coll.aggregate(pipeline, allowDiskUse=True))


def kv_counts(coll, field: str, query: Dict[str, Any]) -> Dict[str, int]:
    rows = run_pipeline(coll, [{"$match": query}, {"$group": {"_id": f"${field}", "count": {"$sum": 1}}}, {"$sort": {"_id": 1}}])
    return {str(row["_id"]): int(row["count"]) for row in rows}


def compound_counts(coll, fields: List[str], query: Dict[str, Any]) -> Dict[str, int]:
    group_id = {field: f"${field}" for field in fields}
    rows = run_pipeline(coll, [{"$match": query}, {"$group": {"_id": group_id, "count": {"$sum": 1}}}, {"$sort": {"count": -1}}])
    out: Dict[str, int] = {}
    for row in rows:
        key = " | ".join(f"{field}={row['_id'].get(field)}" for field in fields)
        out[key] = int(row["count"])
    return out


def pair_integrity(coll, query: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_pipeline(coll, [
        {"$match": query},
        {"$group": {"_id": "$anchor_time", "row_count": {"$sum": 1}, "sides": {"$addToSet": "$side"}}},
        {"$project": {"row_count": 1, "side_count": {"$size": "$sides"}}},
        {"$group": {"_id": {"row_count": "$row_count", "side_count": "$side_count"}, "anchor_count": {"$sum": 1}}},
        {"$sort": {"_id.row_count": 1, "_id.side_count": 1}},
    ])


def year_split_side_counts(coll, query: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_pipeline(coll, [
        {"$match": query},
        {"$project": {"year": {"$year": "$anchor_time"}, "split": 1, "side": 1, "entry_label": 1}},
        {"$group": {"_id": {"year": "$year", "split": "$split", "side": "$side", "entry_label": "$entry_label"}, "count": {"$sum": 1}}},
        {"$sort": {"_id.year": 1, "_id.split": 1, "_id.side": 1, "_id.entry_label": 1}},
    ])


def pct(part: int, total: int) -> str:
    return "0.0000%" if total == 0 else f"{(part / total) * 100:.4f}%"


def write_report(args: argparse.Namespace, status: str, stats: Dict[str, Any], error_text: str = "") -> Path:
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{RUN_ID}.txt"
    total = int(stats.get("total_docs", 0))
    lines: List[str] = []
    lines.append("Side Context Entry AI - Integrated Entry Dataset QA Report")
    lines.append("=" * 80)
    lines.append(f"run_id                   : {RUN_ID}")
    lines.append(f"status                   : {status}")
    lines.append(f"project_name             : {PROJECT_NAME}")
    lines.append(f"project_code             : {PROJECT_CODE}")
    lines.append(f"database                 : {args.database}")
    lines.append(f"collection               : {args.collection}")
    lines.append(f"dataset_version           : {args.dataset_version}")
    lines.append("database_modified         : False")
    lines.append("raw_data_modified         : False")
    lines.append("report_folder_policy      : local_section_reports")
    lines.append("")
    lines.append("Summary:")
    for key, value in stats.get("summary", {}).items():
        lines.append(f"{key:34}: {value}")
    lines.append("")
    for title, key in [("Side counts", "side_counts"), ("Entry label counts", "entry_label_counts"), ("TP bucket counts", "bucket_counts"), ("Split counts", "split_counts")]:
        lines.append(f"{title}:")
        for k, v in stats.get(key, {}).items():
            lines.append(f"  {k:30}: {v} | {pct(v, total)}")
        lines.append("")
    lines.append("Split x Entry label:")
    for k, v in stats.get("split_label_counts", {}).items():
        lines.append(f"{k:44}: {v} | {pct(v, total)}")
    lines.append("")
    lines.append("Split x Side:")
    for k, v in stats.get("split_side_counts", {}).items():
        lines.append(f"{k:44}: {v} | {pct(v, total)}")
    lines.append("")
    lines.append("Missing critical fields:")
    for k, v in stats.get("missing_fields", {}).items():
        lines.append(f"{k:34}: {v}")
    lines.append("")
    lines.append("Pair integrity:")
    for row in stats.get("pair_integrity", []):
        lines.append(f"row_count={row['_id'].get('row_count')} | side_count={row['_id'].get('side_count')} : {row.get('anchor_count')}")
    lines.append("")
    lines.append("Year x Split x Side x Entry label:")
    for row in stats.get("year_split_side_label", []):
        rid = row["_id"]
        lines.append(f"year={rid.get('year')} | split={rid.get('split')} | side={rid.get('side')} | label={rid.get('entry_label')} : {row.get('count')}")
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
        query = {"dataset_version": args.dataset_version}
        total_docs = coll.count_documents(query)
        invalid_side_count = coll.count_documents({**query, "side": {"$nin": SIDES}})
        invalid_label_count = coll.count_documents({**query, "entry_label": {"$nin": ENTRY_LABELS}})
        invalid_bucket_count = coll.count_documents({**query, "entry_tp_bucket": {"$nin": ENTRY_TP_BUCKETS}})
        invalid_split_count = coll.count_documents({**query, "split": {"$nin": VALID_SPLITS}})
        wrong_feature_version_count = coll.count_documents({**query, "feature_version": {"$ne": FEATURE_VERSION_INTEGRATED_ENTRY_M5_V1}})
        wrong_label_version_count = coll.count_documents({**query, "label_version": {"$ne": LABEL_VERSION_ENTRY_SIDE_M5_V1}})
        missing_fields = {field: coll.count_documents({**query, field: {"$exists": False}}) for field in CRITICAL_FIELDS}
        pairs = pair_integrity(coll, query) if args.check_pairs == 1 else []
        unique_anchor_count = sum(int(row.get("anchor_count", 0)) for row in pairs) if pairs else None
        stats = {
            "total_docs": total_docs,
            "summary": {"total_docs": total_docs, "unique_anchor_count": unique_anchor_count, "invalid_side_count": invalid_side_count, "invalid_label_count": invalid_label_count, "invalid_bucket_count": invalid_bucket_count, "invalid_split_count": invalid_split_count, "wrong_feature_version_count": wrong_feature_version_count, "wrong_label_version_count": wrong_label_version_count, "check_pairs": args.check_pairs},
            "side_counts": kv_counts(coll, "side", query),
            "entry_label_counts": kv_counts(coll, "entry_label", query),
            "bucket_counts": kv_counts(coll, "entry_tp_bucket", query),
            "split_counts": kv_counts(coll, "split", query),
            "split_label_counts": compound_counts(coll, ["split", "entry_label"], query),
            "split_side_counts": compound_counts(coll, ["split", "side"], query),
            "missing_fields": missing_fields,
            "pair_integrity": pairs,
            "year_split_side_label": year_split_side_counts(coll, query),
        }
        report_path = write_report(args, "success", stats)
        print("SCE integrated entry dataset QA completed.")
        print(f"Report: {report_path}")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        print("ERROR: SCE integrated entry dataset QA failed.")
        print(error_text)
        report_path = write_report(args, "failed", stats, error_text)
        print(f"Failure report: {report_path}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
