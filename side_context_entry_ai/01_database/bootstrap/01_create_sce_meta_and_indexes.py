# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, traceback
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

def utc_now() -> datetime: return datetime.now(timezone.utc)
def ensure_collection(db, name: str) -> Tuple[str, str]:
    if name in set(db.list_collection_names()): return name, "already_exists"
    db.create_collection(name); return name, "created"
def index_exists(db, collection_name: str, index_name: str) -> bool: return any(idx.get("name") == index_name for idx in db[collection_name].list_indexes())
def ensure_index(db, collection_name: str, keys: List[Tuple[str, int]], name: str, unique: bool = False) -> Dict[str, Any]:
    if index_exists(db, collection_name, name): return {"collection": collection_name, "index_name": name, "status": "already_exists", "unique": unique}
    db[collection_name].create_index(keys, name=name, unique=unique); return {"collection": collection_name, "index_name": name, "status": "created", "unique": unique}
def create_indexes(db) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    def add(coll: str, keys: List[Tuple[str, int]], name: str, unique: bool = False) -> None: out.append(ensure_index(db, coll, keys, name, unique))
    add(COLL_META_PROJECT, [("project_code", ASCENDING)], "ux_project_code", True)
    add(COLL_META_RUNS, [("run_id", ASCENDING)], "ux_run_id", True)
    add(COLL_META_RUNS, [("run_type", ASCENDING)], "idx_run_type")
    add(COLL_META_RUNS, [("status", ASCENDING)], "idx_status")
    add(COLL_META_RUNS, [("created_at", ASCENDING)], "idx_created_at")
    add(COLL_META_FEATURE_VERSIONS, [("feature_version", ASCENDING)], "ux_feature_version", True)
    add(COLL_META_LABEL_VERSIONS, [("label_version", ASCENDING)], "ux_label_version", True)
    add(COLL_META_DATASET_VERSIONS, [("dataset_version", ASCENDING)], "ux_dataset_version", True)
    add(COLL_META_MODEL_VERSIONS, [("model_version", ASCENDING)], "ux_model_version", True)
    for coll in [COLL_LABELS_ENTRY_SIDE_M5_V1, COLL_FEATURES_MA_CONTEXT_M5_V1, COLL_FEATURES_CANDLE_ENTRY_M5_V1, COLL_FEATURES_INTEGRATED_ENTRY_M5_V1, COLL_DATASET_INTEGRATED_ENTRY_M5_V1]:
        add(coll, [("anchor_time", ASCENDING), ("side", ASCENDING)], "ux_anchor_time_side", True)
        add(coll, [("anchor_time", ASCENDING)], "idx_anchor_time")
        add(coll, [("side", ASCENDING)], "idx_side")
        add(coll, [("created_at", ASCENDING)], "idx_created_at")
    add(COLL_LABELS_ENTRY_SIDE_M5_V1, [("entry_label", ASCENDING)], "idx_entry_label")
    add(COLL_LABELS_ENTRY_SIDE_M5_V1, [("entry_tp_bucket", ASCENDING)], "idx_entry_tp_bucket")
    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("entry_label", ASCENDING)], "idx_entry_label")
    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("entry_tp_bucket", ASCENDING)], "idx_entry_tp_bucket")
    add(COLL_DATASET_INTEGRATED_ENTRY_M5_V1, [("split", ASCENDING)], "idx_split")
    for coll in [COLL_PREDICTIONS_INTEGRATED_ENTRY_M5_V1, COLL_API_DECISIONS_M5_V1]:
        add(coll, [("anchor_time", ASCENDING), ("side", ASCENDING), ("model_version", ASCENDING)], "ux_anchor_time_side_model_version", True)
        add(coll, [("anchor_time", ASCENDING)], "idx_anchor_time")
        add(coll, [("side", ASCENDING)], "idx_side")
        add(coll, [("model_version", ASCENDING)], "idx_model_version")
        add(coll, [("created_at", ASCENDING)], "idx_created_at")
    add(COLL_BACKTEST_RESULTS_M5_V1, [("run_id", ASCENDING)], "idx_run_id")
    add(COLL_BACKTEST_RESULTS_M5_V1, [("model_version", ASCENDING)], "idx_model_version")
    add(COLL_BACKTEST_RESULTS_M5_V1, [("created_at", ASCENDING)], "idx_created_at")
    return out
def upsert_metadata(db) -> Dict[str, int]:
    now = utc_now()
    architecture = {"anchor_tf": ANCHOR_TF, "candle_context_tf": CANDLE_CONTEXT_TF, "ma_context_tfs": MA_CONTEXT_TFS, "label_checker_tf": LABEL_CHECKER_TF, "training_mode": TRAINING_MODE, "side_based": SIDE_BASED, "sides": SIDES, "raw_m1_as_feature": RAW_M1_AS_FEATURE, "tp_levels_atr": TP_LEVELS_ATR, "fail_ratio_of_tp": FAIL_RATIO_OF_TP}
    db[COLL_META_PROJECT].update_one({"project_code": PROJECT_CODE}, {"$set": {"project_code": PROJECT_CODE, "project_name": PROJECT_NAME, "database": DB_NAME, "derived_prefix": DERIVED_PREFIX, "architecture": architecture, "updated_at": now}, "$setOnInsert": {"created_at": now}}, upsert=True)
    return {"project_upserts": 1}
def write_report(status: str, collection_results: List[Dict[str, Any]], index_results: List[Dict[str, Any]], metadata_counts: Dict[str, int], error_text: str = "") -> Path:
    report_dir = Path(__file__).resolve().parent / "reports"; report_dir.mkdir(parents=True, exist_ok=True); report_path = report_dir / f"{RUN_ID}.txt"
    lines = ["Side Context Entry AI - Database Bootstrap Report", "=" * 60, f"run_id                   : {RUN_ID}", f"status                   : {status}", f"project_name             : {PROJECT_NAME}", f"project_code             : {PROJECT_CODE}", f"database                 : {DB_NAME}", f"mongo_uri                : {MONGO_URI}", f"derived_prefix           : {DERIVED_PREFIX}", "raw_data_modified         : False", "report_folder_policy      : local_section_reports", "", "Architecture:", f"anchor_tf                : {ANCHOR_TF}", f"candle_context_tf         : {CANDLE_CONTEXT_TF}", f"ma_context_tfs           : {', '.join(MA_CONTEXT_TFS)}", f"label_checker_tf         : {LABEL_CHECKER_TF}", f"training_mode            : {TRAINING_MODE}", f"side_based               : {SIDE_BASED}", f"raw_m1_as_feature        : {RAW_M1_AS_FEATURE}", "", "Collections:", f"created_count            : {sum(1 for x in collection_results if x['status'] == 'created')}", f"already_exists_count     : {sum(1 for x in collection_results if x['status'] == 'already_exists')}"]
    lines += [f"- {x['name']}: {x['status']}" for x in collection_results]
    lines += ["", "Indexes:", f"created_count            : {sum(1 for x in index_results if x['status'] == 'created')}", f"already_exists_count     : {sum(1 for x in index_results if x['status'] == 'already_exists')}"]
    lines += [f"- {x['collection']}.{x['index_name']}: {x['status']}{' unique' if x['unique'] else ''}" for x in index_results]
    lines += ["", "Metadata upserts:"] + [f"{k:26}: {v}" for k, v in metadata_counts.items()]
    if error_text: lines += ["", "Error:", error_text]
    lines += ["", "End of report."]
    report_path.write_text("\n".join(lines), encoding="utf-8"); return report_path
def main() -> int:
    client = None; collection_results = []; index_results = []; metadata_counts = {}
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000); client.admin.command("ping"); db = client[DB_NAME]
        for name in COLLECTIONS:
            collection_name, status = ensure_collection(db, name); collection_results.append({"name": collection_name, "status": status})
        index_results = create_indexes(db); metadata_counts = upsert_metadata(db)
        db[COLL_META_RUNS].update_one({"run_id": RUN_ID}, {"$set": {"run_id": RUN_ID, "run_type": RUN_TYPE, "status": "success", "project_code": PROJECT_CODE, "created_at": utc_now(), "raw_data_modified": False}}, upsert=True)
        report_path = write_report("success", collection_results, index_results, metadata_counts); print("SCE database bootstrap completed."); print(f"Report: {report_path}"); return 0
    except Exception:
        error_text = traceback.format_exc(); print("ERROR: SCE database bootstrap failed."); print(error_text); report_path = write_report("failed", collection_results, index_results, metadata_counts, error_text); print(f"Failure report: {report_path}"); return 1
    finally:
        if client is not None: client.close()
if __name__ == "__main__": raise SystemExit(main())
