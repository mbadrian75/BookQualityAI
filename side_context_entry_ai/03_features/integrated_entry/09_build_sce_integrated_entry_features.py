# -*- coding: utf-8 -*-
"""
SCE Integrated Entry Feature Builder v2 - Chunked

Location:
03_features/integrated_entry/

Report:
03_features/integrated_entry/reports/

Fix in v2:
- Removed long MongoDB aggregation cursor with $lookup
- Uses paginated MA source reads + candle batch lookup by anchor_time
- Avoids CursorNotFound on full run
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
    COLL_FEATURES_CANDLE_ENTRY_M5_V1,
    COLL_FEATURES_INTEGRATED_ENTRY_M5_V1,
    COLL_FEATURES_MA_CONTEXT_M5_V1,
    DB_NAME,
    FEATURE_VERSION_CANDLE_ENTRY_M5_V1,
    FEATURE_VERSION_INTEGRATED_ENTRY_M5_V1,
    FEATURE_VERSION_MA_CONTEXT_M5_V1,
    MONGO_URI,
    PROJECT_CODE,
    PROJECT_NAME,
    RAW_M1_AS_FEATURE,
    SIDES,
)

RUN_TYPE = "sce_build_integrated_entry_features"
RUN_ID = f"{RUN_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EXCLUDE_FROM_SOURCE = {"_id", "feature_version", "run_id", "created_at", "updated_at"}


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
    parser = argparse.ArgumentParser(description="Build SCE integrated entry features in chunks.")
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--database", default=DB_NAME)
    parser.add_argument("--ma-collection", default=COLL_FEATURES_MA_CONTEXT_M5_V1)
    parser.add_argument("--candle-collection", default=COLL_FEATURES_CANDLE_ENTRY_M5_V1)
    parser.add_argument("--target-collection", default=COLL_FEATURES_INTEGRATED_ENTRY_M5_V1)
    parser.add_argument("--ma-feature-version", default=FEATURE_VERSION_MA_CONTEXT_M5_V1)
    parser.add_argument("--candle-feature-version", default=FEATURE_VERSION_CANDLE_ENTRY_M5_V1)
    parser.add_argument("--feature-version", default=FEATURE_VERSION_INTEGRATED_ENTRY_M5_V1)
    parser.add_argument("--limit", type=int, default=50000, help="Source MA feature row limit. Use 0 for full run.")
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--read-batch-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--clear-existing", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def base_match_query(args: argparse.Namespace) -> Dict[str, Any]:
    query: Dict[str, Any] = {"feature_version": args.ma_feature_version}
    start = parse_time(args.start_time)
    end = parse_time(args.end_time)
    if start or end:
        time_filter: Dict[str, Any] = {}
        if start:
            time_filter["$gte"] = start
        if end:
            time_filter["$lte"] = end
        query["anchor_time"] = time_filter
    return query


def paginated_query(base_query: Dict[str, Any], last_key: Optional[Tuple[datetime, str]]) -> Dict[str, Any]:
    if last_key is None:
        return dict(base_query)
    last_anchor_time, last_side = last_key
    return {"$and": [base_query, {"$or": [{"anchor_time": {"$gt": last_anchor_time}}, {"anchor_time": last_anchor_time, "side": {"$gt": last_side}}]}]}


def clean_source_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in doc.items():
        if key in EXCLUDE_FROM_SOURCE:
            continue
        if key in {"anchor_time", "entry_time", "side", "entry_price", "label_version", "source_tfs", "source_feature_collections", "label_targets_included"}:
            continue
        out[key] = value
    return out


def make_key(doc: Dict[str, Any]) -> Tuple[Any, Any]:
    return doc.get("anchor_time"), doc.get("side")


def load_candle_docs(candle_coll, anchor_times: List[datetime], args: argparse.Namespace) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    if not anchor_times:
        return {}
    query = {"feature_version": args.candle_feature_version, "anchor_time": {"$in": anchor_times}}
    docs = candle_coll.find(query).batch_size(5000)
    return {make_key(doc): doc for doc in docs}


def write_report(args: argparse.Namespace, status: str, stats: Dict[str, Any], error_text: str = "") -> Path:
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{RUN_ID}.txt"
    lines: List[str] = []
    lines.append("Side Context Entry AI - Integrated Entry Feature Builder Report")
    lines.append("=" * 80)
    lines.append(f"run_id                   : {RUN_ID}")
    lines.append(f"status                   : {status}")
    lines.append(f"project_name             : {PROJECT_NAME}")
    lines.append(f"project_code             : {PROJECT_CODE}")
    lines.append(f"database                 : {args.database}")
    lines.append(f"ma_collection             : {args.ma_collection}")
    lines.append(f"candle_collection         : {args.candle_collection}")
    lines.append(f"target_collection         : {args.target_collection}")
    lines.append(f"feature_version           : {args.feature_version}")
    lines.append(f"ma_feature_version        : {args.ma_feature_version}")
    lines.append(f"candle_feature_version    : {args.candle_feature_version}")
    lines.append("raw_data_modified         : False")
    lines.append(f"m1_as_feature             : {RAW_M1_AS_FEATURE}")
    lines.append("label_targets_included    : False")
    lines.append("join_mode                 : chunked_python_batch_lookup")
    lines.append("report_folder_policy      : local_section_reports")
    lines.append("")
    lines.append("Runtime parameters:")
    for key in ["limit", "start_time", "end_time", "read_batch_size", "batch_size", "clear_existing"]:
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
    stats: Dict[str, Any] = {"counts": {}, "side_counts": {}, "skip_reasons": {}}
    try:
        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[args.database]
        ma_coll = db[args.ma_collection]
        candle_coll = db[args.candle_collection]
        target = db[args.target_collection]
        if args.clear_existing == 1:
            clear_result = target.delete_many({"feature_version": args.feature_version})
            stats["counts"]["cleared_existing_docs"] = clear_result.deleted_count
        else:
            stats["counts"]["cleared_existing_docs"] = 0
        base_query = base_match_query(args)
        last_key: Optional[Tuple[datetime, str]] = None
        total_scanned = 0
        total_saved = 0
        total_missing_candle = 0
        batches_processed = 0
        first_anchor_time = None
        last_anchor_time = None
        side_counts: Dict[str, int] = {}
        while True:
            remaining = None if args.limit == 0 else args.limit - total_scanned
            if remaining is not None and remaining <= 0:
                break
            current_limit = args.read_batch_size if remaining is None else min(args.read_batch_size, remaining)
            query = paginated_query(base_query, last_key)
            ma_docs = list(ma_coll.find(query).sort([("anchor_time", ASCENDING), ("side", ASCENDING)]).limit(current_limit).batch_size(current_limit))
            if not ma_docs:
                break
            batches_processed += 1
            total_scanned += len(ma_docs)
            anchor_times = sorted({doc.get("anchor_time") for doc in ma_docs if doc.get("anchor_time") is not None})
            candle_map = load_candle_docs(candle_coll, anchor_times, args)
            operations: List[ReplaceOne] = []
            for ma_doc in ma_docs:
                side = ma_doc.get("side")
                if side not in SIDES:
                    stats["skip_reasons"]["invalid_side"] = stats["skip_reasons"].get("invalid_side", 0) + 1
                    continue
                key = make_key(ma_doc)
                candle_doc = candle_map.get(key)
                if candle_doc is None:
                    total_missing_candle += 1
                    continue
                anchor_time = ma_doc.get("anchor_time")
                entry_time = ma_doc.get("entry_time")
                entry_price = ma_doc.get("entry_price")
                if first_anchor_time is None:
                    first_anchor_time = anchor_time
                last_anchor_time = anchor_time
                now = utc_now()
                doc: Dict[str, Any] = {"anchor_time": anchor_time, "entry_time": entry_time, "side": side, "entry_price": entry_price, "feature_version": args.feature_version, "ma_feature_version": args.ma_feature_version, "candle_feature_version": args.candle_feature_version, "run_id": RUN_ID, "source_feature_collections": [args.ma_collection, args.candle_collection], "label_targets_included": False, "created_at": now, "updated_at": now}
                doc.update(clean_source_doc(ma_doc))
                doc.update(clean_source_doc(candle_doc))
                operations.append(ReplaceOne({"anchor_time": anchor_time, "side": side}, doc, upsert=True))
                side_counts[side] = side_counts.get(side, 0) + 1
            if operations:
                target.bulk_write(operations, ordered=False)
                total_saved += len(operations)
            last_doc = ma_docs[-1]
            last_key = (last_doc.get("anchor_time"), last_doc.get("side"))
            if total_saved % (args.batch_size * 10) < args.batch_size:
                print(f"Integrated features saved: {total_saved:,} | scanned: {total_scanned:,}")
        stats["counts"].update({"ma_source_rows_scanned": total_scanned, "saved_feature_docs": total_saved, "missing_candle_docs": total_missing_candle, "batches_processed": batches_processed, "first_anchor_time": first_anchor_time, "last_anchor_time": last_anchor_time})
        stats["side_counts"] = side_counts
        stats["skip_reasons"]["missing_candle_docs"] = total_missing_candle
        report_path = write_report(args, "success", stats)
        print("SCE integrated entry features built.")
        print(f"Saved feature docs: {total_saved:,}")
        print(f"Report: {report_path}")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        print("ERROR: SCE integrated entry feature build failed.")
        print(error_text)
        report_path = write_report(args, "failed", stats, error_text)
        print(f"Failure report: {report_path}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
