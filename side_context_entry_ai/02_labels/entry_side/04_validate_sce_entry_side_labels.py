# -*- coding: utf-8 -*-
"""
SCE Entry Side Label QA

Location:
02_labels/entry_side/

Report:
02_labels/entry_side/reports/

Purpose:
- Validate full label collection after build
- Does not modify database
- Produces side/label/bucket/year distributions
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
    COLL_LABELS_ENTRY_SIDE_M5_V1,
    DB_NAME,
    ENTRY_LABELS,
    ENTRY_TP_BUCKETS,
    LABEL_VERSION_ENTRY_SIDE_M5_V1,
    MONGO_URI,
    PROJECT_CODE,
    PROJECT_NAME,
    SIDES,
)

RUN_TYPE = "sce_validate_entry_side_labels"
RUN_ID = f"{RUN_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SCE entry-side labels.")
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--database", default=DB_NAME)
    parser.add_argument("--collection", default=COLL_LABELS_ENTRY_SIDE_M5_V1)
    parser.add_argument("--label-version", default=LABEL_VERSION_ENTRY_SIDE_M5_V1)
    parser.add_argument("--check-pairs", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def run_pipeline(coll, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(coll.aggregate(pipeline, allowDiskUse=True))


def kv_counts(coll, field: str, query: Dict[str, Any]) -> Dict[str, int]:
    rows = run_pipeline(coll, [{"$match": query}, {"$group": {"_id": f"${field}", "count": {"$sum": 1}}}, {"$sort": {"_id": 1}}])
    return {str(x["_id"]): int(x["count"]) for x in rows}


def compound_counts(coll, fields: List[str], query: Dict[str, Any]) -> Dict[str, int]:
    group_id = {field: f"${field}" for field in fields}
    rows = run_pipeline(coll, [{"$match": query}, {"$group": {"_id": group_id, "count": {"$sum": 1}}}, {"$sort": {"count": -1}}])
    out: Dict[str, int] = {}
    for row in rows:
        key = " | ".join(f"{field}={row['_id'].get(field)}" for field in fields)
        out[key] = int(row["count"])
    return out


def year_counts(coll, query: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_pipeline(coll, [
        {"$match": query},
        {"$project": {"year": {"$year": "$anchor_time"}, "side": 1, "entry_label": 1, "entry_tp_bucket": 1}},
        {"$group": {"_id": {"year": "$year", "side": "$side", "entry_label": "$entry_label"}, "count": {"$sum": 1}}},
        {"$sort": {"_id.year": 1, "_id.side": 1, "_id.entry_label": 1}},
    ])


def pair_integrity(coll, query: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_pipeline(coll, [
        {"$match": query},
        {"$group": {"_id": "$anchor_time", "row_count": {"$sum": 1}, "sides": {"$addToSet": "$side"}}},
        {"$project": {"row_count": 1, "side_count": {"$size": "$sides"}}},
        {"$group": {"_id": {"row_count": "$row_count", "side_count": "$side_count"}, "anchor_count": {"$sum": 1}}},
        {"$sort": {"_id.row_count": 1, "_id.side_count": 1}},
    ])


def pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0000%"
    return f"{(part / total) * 100:.4f}%"


def write_report(args: argparse.Namespace, status: str, stats: Dict[str, Any], error_text: str = "") -> Path:
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{RUN_ID}.txt"
    total = int(stats.get("total_docs", 0))
    lines: List[str] = []
    lines.append("Side Context Entry AI - Entry Side Label QA Report")
    lines.append("=" * 68)
    lines.append(f"run_id                   : {RUN_ID}")
    lines.append(f"status                   : {status}")
    lines.append(f"project_name             : {PROJECT_NAME}")
    lines.append(f"project_code             : {PROJECT_CODE}")
    lines.append(f"database                 : {args.database}")
    lines.append(f"collection               : {args.collection}")
    lines.append(f"label_version            : {args.label_version}")
    lines.append("database_modified         : False")
    lines.append("raw_data_modified         : False")
    lines.append("report_folder_policy      : local_section_reports")
    lines.append("")
    lines.append("Summary:")
    for key, value in stats.get("summary", {}).items():
        lines.append(f"{key:30}: {value}")
    lines.append("")
    lines.append("Side counts:")
    for key, value in stats.get("side_counts", {}).items():
        lines.append(f"{key:30}: {value} | {pct(value, total)}")
    lines.append("")
    lines.append("Entry label counts:")
    for key, value in stats.get("entry_label_counts", {}).items():
        lines.append(f"{key:30}: {value} | {pct(value, total)}")
    lines.append("")
    lines.append("TP bucket counts:")
    for key, value in stats.get("bucket_counts", {}).items():
        lines.append(f"{key:30}: {value} | {pct(value, total)}")
    lines.append("")
    lines.append("Side x Entry label:")
    for key, value in stats.get("side_label_counts", {}).items():
        lines.append(f"{key:30}: {value} | {pct(value, total)}")
    lines.append("")
    lines.append("Side x TP bucket:")
    for key, value in stats.get("side_bucket_counts", {}).items():
        lines.append(f"{key:30}: {value} | {pct(value, total)}")
    lines.append("")
    lines.append("Pair integrity:")
    for row in stats.get("pair_integrity", []):
        lines.append(f"row_count={row['_id'].get('row_count')} | side_count={row['_id'].get('side_count')} : {row.get('anchor_count')}")
    lines.append("")
    lines.append("Year x Side x Entry label:")
    for row in stats.get("year_side_label", []):
        rid = row["_id"]
        lines.append(f"year={rid.get('year')} | side={rid.get('side')} | label={rid.get('entry_label')} : {row.get('count')}")
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
        query = {"label_version": args.label_version}
        total_docs = coll.count_documents(query)
        invalid_side_count = coll.count_documents({**query, "side": {"$nin": SIDES}})
        invalid_label_count = coll.count_documents({**query, "entry_label": {"$nin": ENTRY_LABELS}})
        invalid_bucket_count = coll.count_documents({**query, "entry_tp_bucket": {"$nin": ENTRY_TP_BUCKETS}})
        side_counts = kv_counts(coll, "side", query)
        entry_label_counts = kv_counts(coll, "entry_label", query)
        bucket_counts = kv_counts(coll, "entry_tp_bucket", query)
        side_label_counts = compound_counts(coll, ["side", "entry_label"], query)
        side_bucket_counts = compound_counts(coll, ["side", "entry_tp_bucket"], query)
        pairs = pair_integrity(coll, query) if args.check_pairs == 1 else []
        yearly = year_counts(coll, query)
        unique_anchor_count = sum(int(row.get("anchor_count", 0)) for row in pairs) if pairs else None
        stats = {
            "total_docs": total_docs,
            "summary": {
                "total_docs": total_docs,
                "unique_anchor_count": unique_anchor_count,
                "invalid_side_count": invalid_side_count,
                "invalid_label_count": invalid_label_count,
                "invalid_bucket_count": invalid_bucket_count,
                "check_pairs": args.check_pairs,
            },
            "side_counts": side_counts,
            "entry_label_counts": entry_label_counts,
            "bucket_counts": bucket_counts,
            "side_label_counts": side_label_counts,
            "side_bucket_counts": side_bucket_counts,
            "pair_integrity": pairs,
            "year_side_label": yearly,
        }
        report_path = write_report(args, "success", stats)
        print("SCE entry-side label QA completed.")
        print(f"Report: {report_path}")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        print("ERROR: SCE entry-side label QA failed.")
        print(error_text)
        report_path = write_report(args, "failed", stats, error_text)
        print(f"Failure report: {report_path}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
