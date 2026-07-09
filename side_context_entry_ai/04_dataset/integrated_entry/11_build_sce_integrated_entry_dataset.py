# -*- coding: utf-8 -*-
"""
SCE Integrated Entry Dataset Builder

Location:
04_dataset/integrated_entry/

Report:
04_dataset/integrated_entry/reports/

Purpose:
- Join integrated features with labels on (anchor_time, side)
- Store train/valid/test split
- Writes to sce_dataset_integrated_entry_m5_v1
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from pymongo import ASCENDING, MongoClient, ReplaceOne
except ImportError as exc:
    print("ERROR: pymongo is not installed. Run: pip install pymongo")
    raise exc

from config.sce_project_config import (
    COLL_DATASET_INTEGRATED_ENTRY_M5_V1,
    COLL_FEATURES_INTEGRATED_ENTRY_M5_V1,
    COLL_LABELS_ENTRY_SIDE_M5_V1,
    DATASET_VERSION_INTEGRATED_ENTRY_M5_V1,
    DB_NAME,
    FEATURE_VERSION_INTEGRATED_ENTRY_M5_V1,
    LABEL_VERSION_ENTRY_SIDE_M5_V1,
    MONGO_URI,
    PROJECT_CODE,
    PROJECT_NAME,
    SIDES,
)

RUN_TYPE = "sce_build_integrated_entry_dataset"
RUN_ID = f"{RUN_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EXCLUDE_FEATURE_FIELDS = {"_id", "run_id", "created_at", "updated_at"}
ENTRY_LABEL_ID = {"BAD_ENTRY": 0, "GOOD_ENTRY": 1, "WAIT": 2}
TP_BUCKET_ID = {"NO_TRADE": 0, "SMALL": 1, "NORMAL": 2, "LARGE": 3}


def parse_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"]:
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            pass
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCE integrated entry dataset.")
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--database", default=DB_NAME)
    parser.add_argument("--feature-collection", default=COLL_FEATURES_INTEGRATED_ENTRY_M5_V1)
    parser.add_argument("--label-collection", default=COLL_LABELS_ENTRY_SIDE_M5_V1)
    parser.add_argument("--target-collection", default=COLL_DATASET_INTEGRATED_ENTRY_M5_V1)
    parser.add_argument("--feature-version", default=FEATURE_VERSION_INTEGRATED_ENTRY_M5_V1)
    parser.add_argument("--label-version", default=LABEL_VERSION_ENTRY_SIDE_M5_V1)
    parser.add_argument("--dataset-version", default=DATASET_VERSION_INTEGRATED_ENTRY_M5_V1)
    parser.add_argument("--limit", type=int, default=50000, help="Integrated feature row limit. Use 0 for full run.")
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--train-end", default="2022-12-31 23:59:59")
    parser.add_argument("--valid-end", default="2023-12-31 23:59:59")
    parser.add_argument("--read-batch-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--clear-existing", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def base_match_query(args: argparse.Namespace) -> Dict[str, Any]:
    query: Dict[str, Any] = {"feature_version": args.feature_version}
    start = parse_time(args.start_time)
    end = parse_time(args.end_time)
    if start or end:
        tf: Dict[str, Any] = {}
        if start:
            tf["$gte"] = start
        if end:
            tf["$lte"] = end
        query["anchor_time"] = tf
    return query


def paginated_query(base_query: Dict[str, Any], last_key: Optional[Tuple[datetime, str]]) -> Dict[str, Any]:
    if last_key is None:
        return dict(base_query)
    last_anchor_time, last_side = last_key
    return {"$and": [base_query, {"$or": [{"anchor_time": {"$gt": last_anchor_time}}, {"anchor_time": last_anchor_time, "side": {"$gt": last_side}}]}]}


def make_key(doc: Dict[str, Any]) -> Tuple[Any, Any]:
    return doc.get("anchor_time"), doc.get("side")


def load_label_docs(label_coll, anchor_times: List[datetime], args: argparse.Namespace) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    if not anchor_times:
        return {}
    query = {"label_version": args.label_version, "anchor_time": {"$in": anchor_times}}
    projection = {"anchor_time": 1, "side": 1, "entry_label": 1, "entry_tp_bucket": 1, "max_entry_success_tp_atr": 1, "success_tp_levels_atr": 1, "fail_tp_levels_atr": 1, "wait_tp_levels_atr": 1, "label_version": 1, "_id": 0}
    return {make_key(doc): doc for doc in label_coll.find(query, projection).batch_size(5000)}


def clean_feature_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in doc.items():
        if key in EXCLUDE_FEATURE_FIELDS:
            continue
        out[key] = value
    return out


def split_name(anchor_time: datetime, train_end: datetime, valid_end: datetime) -> str:
    if anchor_time <= train_end:
        return "train"
    if anchor_time <= valid_end:
        return "valid"
    return "test"


def write_report(args: argparse.Namespace, status: str, stats: Dict[str, Any], error_text: str = "") -> Path:
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{RUN_ID}.txt"
    lines: List[str] = []
    lines.append("Side Context Entry AI - Integrated Entry Dataset Builder Report")
    lines.append("=" * 80)
    lines.append(f"run_id                   : {RUN_ID}")
    lines.append(f"status                   : {status}")
    lines.append(f"project_name             : {PROJECT_NAME}")
    lines.append(f"project_code             : {PROJECT_CODE}")
    lines.append(f"database                 : {args.database}")
    lines.append(f"feature_collection        : {args.feature_collection}")
    lines.append(f"label_collection          : {args.label_collection}")
    lines.append(f"target_collection         : {args.target_collection}")
    lines.append(f"feature_version           : {args.feature_version}")
    lines.append(f"label_version             : {args.label_version}")
    lines.append(f"dataset_version           : {args.dataset_version}")
    lines.append("raw_data_modified         : False")
    lines.append("dataset_targets_included  : True")
    lines.append("report_folder_policy      : local_section_reports")
    lines.append("")
    lines.append("Runtime parameters:")
    for key in ["limit", "start_time", "end_time", "train_end", "valid_end", "read_batch_size", "batch_size", "clear_existing"]:
        lines.append(f"{key:26}: {getattr(args, key)}")
    lines.append("")
    lines.append("Counts:")
    for key, value in stats.get("counts", {}).items():
        lines.append(f"{key:26}: {value}")
    lines.append("")
    lines.append("Side distribution:")
    for key, value in sorted(stats.get("side_counts", {}).items()):
        lines.append(f"{key:26}: {value}")
    lines.append("")
    lines.append("Entry label distribution:")
    for key, value in sorted(stats.get("entry_label_counts", {}).items()):
        lines.append(f"{key:26}: {value}")
    lines.append("")
    lines.append("TP bucket distribution:")
    for key, value in sorted(stats.get("tp_bucket_counts", {}).items()):
        lines.append(f"{key:26}: {value}")
    lines.append("")
    lines.append("Split distribution:")
    for key, value in sorted(stats.get("split_counts", {}).items()):
        lines.append(f"{key:26}: {value}")
    lines.append("")
    lines.append("Skip reasons:")
    for key, value in sorted(stats.get("skip_reasons", {}).items()):
        lines.append(f"{key:26}: {value}")
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
    stats: Dict[str, Any] = {"counts": {}, "side_counts": {}, "entry_label_counts": {}, "tp_bucket_counts": {}, "split_counts": {}, "skip_reasons": {}}
    try:
        train_end = parse_time(args.train_end)
        valid_end = parse_time(args.valid_end)
        if train_end is None or valid_end is None:
            raise ValueError("train_end and valid_end are required")
        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[args.database]
        feature_coll = db[args.feature_collection]
        label_coll = db[args.label_collection]
        target = db[args.target_collection]
        if args.clear_existing == 1:
            clear_result = target.delete_many({"dataset_version": args.dataset_version})
            stats["counts"]["cleared_existing_docs"] = clear_result.deleted_count
        else:
            stats["counts"]["cleared_existing_docs"] = 0
        base_query = base_match_query(args)
        last_key: Optional[Tuple[datetime, str]] = None
        total_scanned = 0
        total_saved = 0
        total_missing_label = 0
        batches_processed = 0
        first_anchor_time = None
        last_anchor_time = None
        while True:
            remaining = None if args.limit == 0 else args.limit - total_scanned
            if remaining is not None and remaining <= 0:
                break
            current_limit = args.read_batch_size if remaining is None else min(args.read_batch_size, remaining)
            query = paginated_query(base_query, last_key)
            feature_docs = list(feature_coll.find(query).sort([("anchor_time", ASCENDING), ("side", ASCENDING)]).limit(current_limit).batch_size(current_limit))
            if not feature_docs:
                break
            batches_processed += 1
            total_scanned += len(feature_docs)
            anchor_times = sorted({doc.get("anchor_time") for doc in feature_docs if doc.get("anchor_time") is not None})
            label_map = load_label_docs(label_coll, anchor_times, args)
            operations: List[ReplaceOne] = []
            for feature_doc in feature_docs:
                side = feature_doc.get("side")
                if side not in SIDES:
                    stats["skip_reasons"]["invalid_side"] = stats["skip_reasons"].get("invalid_side", 0) + 1
                    continue
                label_doc = label_map.get(make_key(feature_doc))
                if label_doc is None:
                    total_missing_label += 1
                    continue
                anchor_time = feature_doc.get("anchor_time")
                if first_anchor_time is None:
                    first_anchor_time = anchor_time
                last_anchor_time = anchor_time
                entry_label = label_doc.get("entry_label")
                entry_tp_bucket = label_doc.get("entry_tp_bucket")
                split = split_name(anchor_time, train_end, valid_end)
                now = utc_now()
                doc = clean_feature_doc(feature_doc)
                doc.update({
                    "dataset_version": args.dataset_version,
                    "feature_version": args.feature_version,
                    "label_version": args.label_version,
                    "run_id": RUN_ID,
                    "entry_label": entry_label,
                    "entry_label_id": ENTRY_LABEL_ID.get(entry_label, -1),
                    "is_good_entry": 1 if entry_label == "GOOD_ENTRY" else 0,
                    "entry_tp_bucket": entry_tp_bucket,
                    "entry_tp_bucket_id": TP_BUCKET_ID.get(entry_tp_bucket, -1),
                    "max_entry_success_tp_atr": label_doc.get("max_entry_success_tp_atr"),
                    "success_tp_levels_atr": label_doc.get("success_tp_levels_atr"),
                    "fail_tp_levels_atr": label_doc.get("fail_tp_levels_atr"),
                    "wait_tp_levels_atr": label_doc.get("wait_tp_levels_atr"),
                    "split": split,
                    "created_at": now,
                    "updated_at": now,
                })
                operations.append(ReplaceOne({"anchor_time": doc["anchor_time"], "side": side}, doc, upsert=True))
                stats["side_counts"][side] = stats["side_counts"].get(side, 0) + 1
                stats["entry_label_counts"][entry_label] = stats["entry_label_counts"].get(entry_label, 0) + 1
                stats["tp_bucket_counts"][entry_tp_bucket] = stats["tp_bucket_counts"].get(entry_tp_bucket, 0) + 1
                stats["split_counts"][split] = stats["split_counts"].get(split, 0) + 1
            if operations:
                target.bulk_write(operations, ordered=False)
                total_saved += len(operations)
            last_doc = feature_docs[-1]
            last_key = (last_doc.get("anchor_time"), last_doc.get("side"))
            if total_saved % (args.batch_size * 10) < args.batch_size:
                print(f"Dataset rows saved: {total_saved:,} | scanned: {total_scanned:,}")
        stats["counts"].update({"integrated_feature_rows_scanned": total_scanned, "saved_dataset_docs": total_saved, "missing_label_docs": total_missing_label, "batches_processed": batches_processed, "first_anchor_time": first_anchor_time, "last_anchor_time": last_anchor_time})
        stats["skip_reasons"]["missing_label_docs"] = total_missing_label
        report_path = write_report(args, "success", stats)
        print("SCE integrated entry dataset built.")
        print(f"Saved dataset docs: {total_saved:,}")
        print(f"Report: {report_path}")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        print("ERROR: SCE integrated entry dataset build failed.")
        print(error_text)
        report_path = write_report(args, "failed", stats, error_text)
        print(f"Failure report: {report_path}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
