# -*- coding: utf-8 -*-
"""
SCE Database Bootstrap - Full Index Plan

Location:
01_database/bootstrap/

Report:
01_database/bootstrap/reports/
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from pymongo import ASCENDING, MongoClient
except ImportError as exc:
    print("ERROR: pymongo is not installed. Run: pip install pymongo")
    raise exc

from config.sce_project_config import *

RUN_TYPE = "sce_db_bootstrap"
RUN_ID = f"{RUN_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_collection(db, name: str) -> Tuple[str, str]:
    if name in set(db.list_collection_names()):
        return name, "already_exists"
    db.create_collection(name)
    return name, "created"


def index_exists(db, collection_name: str, index_name: str) -> bool:
    return any(idx.get("name") == index_name for idx in db[collection_name].list_indexes())


def ensure_index(db, collection_name: str, keys: List[Tuple[str, int]], name: str, unique: bool = False) -> Dict[str, Any]:
    if index_exists(db, collection_name, name):
        return {"collection": collection_name, "index_name": name, "status": "already_exists", "unique": unique, "keys": keys}
    db[collection_name].create_index(keys, name=name, unique=unique)
    return {"collection": collection_name, "index_name": name, "status": "created", "unique": unique, "keys": keys}


def create_indexes(db) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def add(coll: str, keys: List[Tuple[str, int]], name: str, unique: bool = False) -> None:
        out.append(ensure_index(db, coll, keys, name, unique))

    add(COLL_META_PROJECT, [("project_code", ASCENDING)], "ux_project_code", True)
    add(COLL_META_PROJECT, [("database", ASCENDING)], "idx_database")
    add(COLL_META_PROJECT, [("derived_prefix", ASCENDING)], "idx_derived_prefix")

    add(COLL_META_RUNS, [("run_id", ASCENDING)], "ux_run_id", True)
    add(COLL_META_RUNS, [("project_code", ASCENDING)], "idx_project_code")
    add(COLL_META_RUNS, [("run_type", ASCENDING)], "idx_run_type")
    add(COLL_META_RUNS, [("status", ASCENDING)], "idx_status")
    add(COLL_META_RUNS, [("created_at", ASCENDING)], "idx_created_at")

    add(COLL_META_FEATURE_VERSIONS, [("feature_version", ASCENDING)], "ux_feature_version", True)
    add(COLL_META_FEATURE_VERSIONS, [("collection_name", ASCENDING)], "idx_collection_name")
    add(COLL_META_FEATURE_VERSIONS, [("status", ASCENDING)], "idx_status")

    add(COLL_META_LABEL_VERSIONS, [("label_version", ASCENDING)], "ux_label_version", True)
    add(COLL_META_LABEL_VERSIONS, [("collection_name", ASCENDING)], "idx_collection_name")
    add(COLL_META_LABEL_VERSIONS, [("status", ASCENDING)], "idx_status")

    add(COLL_META_DATASET_VERSIONS, [("dataset_version", ASCENDING)], "ux_dataset_version", True)
    add(COLL_META_DATASET_VERSIONS, [("collection_name", ASCENDING)], "idx_collection_name")
    add(COLL_META_DATASET_VERSIONS, [("status", ASCENDING)], "idx_status")

    add(COLL_META_MODEL_VERSIONS, [("model_version", ASCENDING)], "ux_model_version", True)
    add(COLL_META_MODEL_VERSIONS, [("dataset_version", ASCENDING)], "idx_dataset_version")
    add(COLL_META_MODEL_VERSIONS, [("status", ASCENDING)], "idx_status")

    add(COLL_LABELS_ENTRY_SIDE_M5_V1, [("anchor_time", ASCENDING), ("side", ASCENDING)], "ux_anchor_time_side", True)
    add(COLL_LABELS_ENTRY_SIDE_M5_V1, [("anchor_time", ASCENDING)], "idx_anchor_time")
    add(COLL_LABELS_ENTRY_SIDE_M5_V1, [("side", ASCENDING)], "idx_side")
    add(COLL_LABELS_ENTRY_SIDE_M5_V1, [("entry_label", ASCENDING)], "idx_entry_label")
    add(COLL_LABELS_ENTRY_SIDE_M5_V1, [("entry_tp_bucket", ASCENDING)], "idx_entry_tp_bucket")
    add(COLL_LABELS_ENTRY_SIDE_M5_V1, [("max_entry_success_tp_atr", ASCENDING)], "idx_max_entry_success_tp_atr")
    add(COLL_LABELS_ENTRY_SIDE_M5_V1, [("created_at", ASCENDING)], "idx_created_at")

    for coll in [COLL_FEATURES_MA_CONTEXT_M5_V1, COLL_FEATURES_CANDLE_ENTRY_M5_V1, COLL_FEATURES_INTEGRATED_ENTRY_M5_V1]:
        add(coll, [("anchor_time", ASCENDING), ("side", ASCENDING)], "ux_anchor_time_side", True)
        add(coll, [("anchor_time", ASCENDING)], "idx_anchor_time")
        add(coll, [("side", ASCENDING)], "idx_side")
        add(coll, [("feature_version", ASCENDING)], "idx_feature_version")
        add(coll, [("created_at", ASCENDING)], "idx_created_at")

    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("anchor_time", ASCENDING), ("side", ASCENDING)], "ux_anchor_time_side", True)
    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("anchor_time", ASCENDING)], "idx_anchor_time")
    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("side", ASCENDING)], "idx_side")
    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("entry_label", ASCENDING)], "idx_entry_label")
    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("entry_tp_bucket", ASCENDING)], "idx_entry_tp_bucket")
    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("split", ASCENDING)], "idx_split")
    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("feature_version", ASCENDING)], "idx_feature_version")
    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("label_version", ASCENDING)], "idx_label_version")
    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("created_at", ASCENDING)], "idx_created_at")

    for coll in [COLL_PREDICTIONS_INTEGRATED_ENTRY_M5_V1, COLL_API_DECISIONS_M5_V1]:
        add(coll, [("anchor_time", ASCENDING), ("side", ASCENDING), ("model_version", ASCENDING)], "ux_anchor_time_side_model_version", True)
        add(coll, [("anchor_time", ASCENDING)], "idx_anchor_time")
        add(coll, [("side", ASCENDING)], "idx_side")
        add(coll, [("model_version", ASCENDING)], "idx_model_version")
        add(coll, [("created_at", ASCENDING)], "idx_created_at")

    add(COLL_BACKTEST_RESULTS_M5_V1, [("run_id", ASCENDING)], "idx_run_id")
    add(COLL_BACKTEST_RESULTS_M5_V1, [("model_version", ASCENDING)], "idx_model_version")
    add(COLL_BACKTEST_RESULTS_M5_V1, [("side_mode", ASCENDING)], "idx_side_mode")
    add(COLL_BACKTEST_RESULTS_M5_V1, [("from_time", ASCENDING), ("to_time", ASCENDING)], "idx_period")
    add(COLL_BACKTEST_RESULTS_M5_V1, [("created_at", ASCENDING)], "idx_created_at")

    return out


def upsert_metadata(db) -> Dict[str, int]:
    now = utc_now()
    architecture = {"anchor_tf": ANCHOR_TF, "candle_context_tf": CANDLE_CONTEXT_TF, "ma_context_tfs": MA_CONTEXT_TFS, "label_checker_tf": LABEL_CHECKER_TF, "training_mode": TRAINING_MODE, "side_based": SIDE_BASED, "sides": SIDES, "raw_m1_as_feature": RAW_M1_AS_FEATURE, "entry_labels": ENTRY_LABELS, "entry_tp_buckets": ENTRY_TP_BUCKETS, "tp_levels_atr": TP_LEVELS_ATR, "fail_ratio_of_tp": FAIL_RATIO_OF_TP}

    db[COLL_META_PROJECT].update_one({"project_code": PROJECT_CODE}, {"$set": {"project_code": PROJECT_CODE, "project_name": PROJECT_NAME, "database": DB_NAME, "derived_prefix": DERIVED_PREFIX, "architecture": architecture, "updated_at": now}, "$setOnInsert": {"created_at": now}}, upsert=True)

    feature_docs = [
        {"feature_version": FEATURE_VERSION_MA_CONTEXT_M5_V1, "collection_name": COLL_FEATURES_MA_CONTEXT_M5_V1, "description": "Side-based MA context features for anchor M5 using M30/H1/H4 MA20/MA50/MA100.", "anchor_tf": ANCHOR_TF, "source_tfs": MA_CONTEXT_TFS, "side_based": True, "status": "active"},
        {"feature_version": FEATURE_VERSION_CANDLE_ENTRY_M5_V1, "collection_name": COLL_FEATURES_CANDLE_ENTRY_M5_V1, "description": "Side-based candle entry features from M5 anchor and M15 candle context.", "anchor_tf": ANCHOR_TF, "source_tfs": [ANCHOR_TF, CANDLE_CONTEXT_TF], "side_based": True, "status": "active"},
        {"feature_version": FEATURE_VERSION_INTEGRATED_ENTRY_M5_V1, "collection_name": COLL_FEATURES_INTEGRATED_ENTRY_M5_V1, "description": "Integrated side/context-aware entry features combining MA, candle, and interaction features.", "anchor_tf": ANCHOR_TF, "source_tfs": [ANCHOR_TF, CANDLE_CONTEXT_TF] + MA_CONTEXT_TFS, "side_based": True, "status": "active"},
    ]
    label_docs = [{"label_version": LABEL_VERSION_ENTRY_SIDE_M5_V1, "collection_name": COLL_LABELS_ENTRY_SIDE_M5_V1, "description": "M1 future checker labels for BUY/SELL candidate rows at each M5 anchor time.", "anchor_tf": ANCHOR_TF, "checker_tf": LABEL_CHECKER_TF, "side_based": True, "entry_labels": ENTRY_LABELS, "entry_tp_buckets": ENTRY_TP_BUCKETS, "tp_levels_atr": TP_LEVELS_ATR, "fail_ratio_of_tp": FAIL_RATIO_OF_TP, "status": "active"}]
    dataset_docs = [{"dataset_version": DATASET_VERSION_INTEGRATED_ENTRY_M5_V1, "collection_name": COLL_DATASET_INTEGRATED_ENTRY_M5_V1, "description": "Integrated training dataset with BUY and SELL candidate rows for each M5 anchor_time.", "anchor_tf": ANCHOR_TF, "feature_version": FEATURE_VERSION_INTEGRATED_ENTRY_M5_V1, "label_version": LABEL_VERSION_ENTRY_SIDE_M5_V1, "side_based": True, "status": "active"}]
    model_docs = [{"model_version": MODEL_VERSION_INTEGRATED_ENTRY_M5_V1, "description": "Initial integrated Side Context Entry model version for XAUUSD.", "anchor_tf": ANCHOR_TF, "dataset_version": DATASET_VERSION_INTEGRATED_ENTRY_M5_V1, "training_mode": TRAINING_MODE, "side_based": True, "status": "planned"}]

    for doc in feature_docs:
        db[COLL_META_FEATURE_VERSIONS].update_one({"feature_version": doc["feature_version"]}, {"$set": {**doc, "project_code": PROJECT_CODE, "updated_at": now}, "$setOnInsert": {"created_at": now}}, upsert=True)
    for doc in label_docs:
        db[COLL_META_LABEL_VERSIONS].update_one({"label_version": doc["label_version"]}, {"$set": {**doc, "project_code": PROJECT_CODE, "updated_at": now}, "$setOnInsert": {"created_at": now}}, upsert=True)
    for doc in dataset_docs:
        db[COLL_META_DATASET_VERSIONS].update_one({"dataset_version": doc["dataset_version"]}, {"$set": {**doc, "project_code": PROJECT_CODE, "updated_at": now}, "$setOnInsert": {"created_at": now}}, upsert=True)
    for doc in model_docs:
        db[COLL_META_MODEL_VERSIONS].update_one({"model_version": doc["model_version"]}, {"$set": {**doc, "project_code": PROJECT_CODE, "updated_at": now}, "$setOnInsert": {"created_at": now}}, upsert=True)

    return {"project_upserts": 1, "feature_version_upserts": len(feature_docs), "label_version_upserts": len(label_docs), "dataset_version_upserts": len(dataset_docs), "model_version_upserts": len(model_docs)}


def write_report(status: str, collection_results: List[Dict[str, Any]], index_results: List[Dict[str, Any]], metadata_counts: Dict[str, int], error_text: str = "") -> Path:
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{RUN_ID}.txt"
    lines = ["Side Context Entry AI - Database Bootstrap Report", "=" * 60, f"run_id                   : {RUN_ID}", f"status                   : {status}", f"project_name             : {PROJECT_NAME}", f"project_code             : {PROJECT_CODE}", f"database                 : {DB_NAME}", f"mongo_uri                : {MONGO_URI}", f"derived_prefix           : {DERIVED_PREFIX}", "raw_data_modified         : False", "report_folder_policy      : local_section_reports", "", "Architecture:", f"anchor_tf                : {ANCHOR_TF}", f"candle_context_tf         : {CANDLE_CONTEXT_TF}", f"ma_context_tfs           : {', '.join(MA_CONTEXT_TFS)}", f"label_checker_tf         : {LABEL_CHECKER_TF}", f"training_mode            : {TRAINING_MODE}", f"side_based               : {SIDE_BASED}", f"raw_m1_as_feature        : {RAW_M1_AS_FEATURE}", "", "Collections:", f"created_count            : {sum(1 for x in collection_results if x['status'] == 'created')}", f"already_exists_count     : {sum(1 for x in collection_results if x['status'] == 'already_exists')}"]
    lines += [f"- {item['name']}: {item['status']}" for item in collection_results]
    lines += ["", "Indexes:", f"created_count            : {sum(1 for x in index_results if x['status'] == 'created')}", f"already_exists_count     : {sum(1 for x in index_results if x['status'] == 'already_exists')}"]
    lines += [f"- {item['collection']}.{item['index_name']}: {item['status']}{' unique' if item['unique'] else ''}" for item in index_results]
    lines += ["", "Metadata upserts:"] + [f"{k:26}: {v}" for k, v in metadata_counts.items()]
    lines += ["", "Core unique rules:", "- Every M5 anchor_time has two candidate rows: BUY and SELL.", "- Main training collections use unique index: (anchor_time, side).", "- Prediction/API collections use unique index: (anchor_time, side, model_version)."]
    if error_text:
        lines += ["", "Error:", error_text]
    lines += ["", "End of report."]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    client = None
    collection_results: List[Dict[str, Any]] = []
    index_results: List[Dict[str, Any]] = []
    metadata_counts: Dict[str, int] = {}
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[DB_NAME]
        for name in COLLECTIONS:
            collection_name, collection_status = ensure_collection(db, name)
            collection_results.append({"name": collection_name, "status": collection_status})
        index_results = create_indexes(db)
        metadata_counts = upsert_metadata(db)
        db[COLL_META_RUNS].update_one({"run_id": RUN_ID}, {"$set": {"run_id": RUN_ID, "run_type": RUN_TYPE, "status": "success", "project_code": PROJECT_CODE, "database": DB_NAME, "derived_prefix": DERIVED_PREFIX, "created_at": utc_now(), "raw_data_modified": False}}, upsert=True)
        report_path = write_report("success", collection_results, index_results, metadata_counts)
        print("SCE database bootstrap completed.")
        print(f"Report: {report_path}")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        print("ERROR: SCE database bootstrap failed.")
        print(error_text)
        report_path = write_report("failed", collection_results, index_results, metadata_counts, error_text)
        print(f"Failure report: {report_path}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
