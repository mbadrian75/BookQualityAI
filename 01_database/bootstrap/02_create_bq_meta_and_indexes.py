# -*- coding: utf-8 -*-
"""
Book & Quality - MongoDB Bootstrap

Location:
    Book_Quality/01_database/bootstrap/02_create_bq_meta_and_indexes.py

Purpose:
    Creates BQ collections, indexes, and metadata documents inside the existing market_data database.

Important:
    - This script does NOT delete or change raw candle data.
    - This script creates only bq_* project collections and their indexes.
    - This script always creates JSON and TXT reports in:
        Book_Quality/01_database/reports/

Requirements:
    pip install pymongo

Run:
    cd C:\Project\Book_Quality\01_database\bootstrap
    python -u 02_create_bq_meta_and_indexes.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import CollectionInvalid, OperationFailure


SCRIPT_NAME = "02_create_bq_meta_and_indexes.py"

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "raw_collections": {
        "m1": "xauusd_m1",
        "m5": "xauusd_m5",
        "m15": "xauusd_m15",
        "m30": "xauusd_m30",
        "h1": "xauusd_h1",
        "h4": "xauusd_h4",
        "d1": "xauusd_d1"
    },
    "bq_collections": {
        "features_book_entry": "bq_features_book_entry_m15_v1",
        "features_trade_quality": "bq_features_trade_quality_m15_v1",
        "labels_book_entry": "bq_labels_book_entry_m15_v1",
        "labels_trade_quality": "bq_labels_trade_quality_m15_v1",
        "dataset_book_entry": "bq_dataset_book_entry_m15_v1",
        "dataset_trade_quality": "bq_dataset_trade_quality_m15_v1",
        "predictions_book_entry": "bq_predictions_book_entry_m15_v1",
        "predictions_trade_quality": "bq_predictions_trade_quality_m15_v1",
        "api_decisions": "bq_api_decisions_m15_v1"
    }
}

PROJECT_NAME = "Book & Quality"
PROJECT_CODE = "bq"
PROJECT_VERSION = "v1"
ANCHOR_TIMEFRAME = "M15"

META_COLLECTIONS: List[str] = [
    "bq_meta_project",
    "bq_meta_feature_versions",
    "bq_meta_label_versions",
    "bq_meta_dataset_versions",
    "bq_meta_model_versions",
    "bq_meta_runs",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def file_stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def get_project_root() -> Path:
    # script: Book_Quality/01_database/bootstrap/script.py
    return Path(__file__).resolve().parents[2]


def get_reports_dir() -> Path:
    reports_dir = get_project_root() / "01_database" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def load_config() -> Dict[str, Any]:
    config_path = get_project_root() / "00_config" / "mongo_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            merged = DEFAULT_CONFIG.copy()
            merged.update(config)
            return merged
        except Exception as exc:
            print(f"[WARN] Could not read config file, using defaults: {config_path} | {exc}", flush=True)
    return DEFAULT_CONFIG


def connect(config: Dict[str, Any]) -> Database:
    client = MongoClient(config["mongo_uri"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[config["database"]]


def create_collection_if_missing(db: Database, name: str, report: Dict[str, Any]) -> None:
    existing = db.list_collection_names()
    if name in existing:
        print(f"[OK] Collection exists: {name}", flush=True)
        report["collections_existing"].append(name)
        return

    try:
        db.create_collection(name)
        print(f"[CREATE] Collection created: {name}", flush=True)
        report["collections_created"].append(name)
    except CollectionInvalid:
        print(f"[OK] Collection already exists: {name}", flush=True)
        report["collections_existing"].append(name)


def check_raw_collections(db: Database, raw_collections: Dict[str, str], report: Dict[str, Any]) -> None:
    print("\n=== Checking raw collections ===", flush=True)

    existing = set(db.list_collection_names())

    for tf, coll_name in raw_collections.items():
        item = {
            "timeframe": tf,
            "collection": coll_name,
            "exists": coll_name in existing,
            "estimated_count": None,
            "sample_keys": [],
        }

        if coll_name not in existing:
            print(f"[MISSING] {tf.upper()} -> {coll_name}", flush=True)
            report["warnings"].append(f"Missing raw collection: {coll_name}")
            report["raw_collections"][tf] = item
            continue

        try:
            count = db[coll_name].estimated_document_count()
            sample = db[coll_name].find_one()
            item["estimated_count"] = count
            item["sample_keys"] = list(sample.keys()) if sample else []
            print(f"[OK] {tf.upper():>3} -> {coll_name:<12} count≈{count:,} keys={item['sample_keys']}", flush=True)
        except Exception as exc:
            report["warnings"].append(f"Could not inspect {coll_name}: {exc}")
            print(f"[WARN] Could not inspect {coll_name}: {exc}", flush=True)

        report["raw_collections"][tf] = item


def create_index_safe(coll: Collection, keys: List[tuple], report: Dict[str, Any], **kwargs: Any) -> None:
    try:
        index_name = coll.create_index(keys, **kwargs)
        print(f"[INDEX] {coll.name}: {index_name}", flush=True)
        report["indexes_created_or_existing"].append({
            "collection": coll.name,
            "index_name": index_name,
            "keys": str(keys),
            "unique": kwargs.get("unique", False),
        })
    except OperationFailure as exc:
        warning = f"Index failed on {coll.name}: {exc}"
        print(f"[WARN] {warning}", flush=True)
        report["warnings"].append(warning)


def create_all_bq_collections(db: Database, bq_collection_names: List[str], report: Dict[str, Any]) -> None:
    print("\n=== Creating BQ collections ===", flush=True)
    for name in bq_collection_names:
        create_collection_if_missing(db, name, report)


def create_indexes(db: Database, bq: Dict[str, str], report: Dict[str, Any]) -> None:
    print("\n=== Creating indexes ===", flush=True)

    create_index_safe(db[bq["features_book_entry"]],
        [("symbol", ASCENDING), ("anchor_time", ASCENDING)],
        report, unique=True, name="uq_symbol_anchor_time")
    create_index_safe(db[bq["features_book_entry"]],
        [("feature_version", ASCENDING), ("anchor_time", ASCENDING)],
        report, name="ix_feature_version_anchor_time")

    create_index_safe(db[bq["features_trade_quality"]],
        [("symbol", ASCENDING), ("anchor_time", ASCENDING)],
        report, unique=True, name="uq_symbol_anchor_time")
    create_index_safe(db[bq["features_trade_quality"]],
        [("feature_version", ASCENDING), ("anchor_time", ASCENDING)],
        report, name="ix_feature_version_anchor_time")

    create_index_safe(db[bq["labels_book_entry"]],
        [("symbol", ASCENDING), ("anchor_time", ASCENDING)],
        report, unique=True, name="uq_symbol_anchor_time")
    create_index_safe(db[bq["labels_book_entry"]],
        [("label_version", ASCENDING), ("label", ASCENDING)],
        report, name="ix_label_version_label")

    create_index_safe(db[bq["labels_trade_quality"]],
        [("symbol", ASCENDING), ("anchor_time", ASCENDING)],
        report, unique=True, name="uq_symbol_anchor_time")
    create_index_safe(db[bq["labels_trade_quality"]],
        [("label_version", ASCENDING), ("anchor_time", ASCENDING)],
        report, name="ix_label_version_anchor_time")

    create_index_safe(db[bq["dataset_book_entry"]],
        [("symbol", ASCENDING), ("anchor_time", ASCENDING)],
        report, unique=True, name="uq_symbol_anchor_time")
    create_index_safe(db[bq["dataset_book_entry"]],
        [("dataset_version", ASCENDING), ("y.label", ASCENDING)],
        report, name="ix_dataset_version_y_label")

    create_index_safe(db[bq["dataset_trade_quality"]],
        [("symbol", ASCENDING), ("anchor_time", ASCENDING)],
        report, unique=True, name="uq_symbol_anchor_time")
    create_index_safe(db[bq["dataset_trade_quality"]],
        [("dataset_version", ASCENDING), ("y.trend_phase", ASCENDING)],
        report, name="ix_dataset_version_trend_phase")

    create_index_safe(db[bq["predictions_book_entry"]],
        [("symbol", ASCENDING), ("anchor_time", ASCENDING), ("model_version", ASCENDING)],
        report, unique=True, name="uq_symbol_anchor_model_version")
    create_index_safe(db[bq["predictions_book_entry"]],
        [("model_name", ASCENDING), ("model_version", ASCENDING), ("anchor_time", ASCENDING)],
        report, name="ix_model_anchor_time")

    create_index_safe(db[bq["predictions_trade_quality"]],
        [("symbol", ASCENDING), ("anchor_time", ASCENDING), ("model_version", ASCENDING)],
        report, unique=True, name="uq_symbol_anchor_model_version")
    create_index_safe(db[bq["predictions_trade_quality"]],
        [("model_name", ASCENDING), ("model_version", ASCENDING), ("anchor_time", ASCENDING)],
        report, name="ix_model_anchor_time")

    create_index_safe(db[bq["api_decisions"]],
        [("symbol", ASCENDING), ("anchor_time", ASCENDING), ("decision_version", ASCENDING)],
        report, unique=True, name="uq_symbol_anchor_decision_version")
    create_index_safe(db[bq["api_decisions"]],
        [("decision.final_signal", ASCENDING), ("anchor_time", ASCENDING)],
        report, name="ix_final_signal_anchor_time")
    create_index_safe(db[bq["api_decisions"]],
        [("decision.risk_mode", ASCENDING), ("anchor_time", ASCENDING)],
        report, name="ix_risk_mode_anchor_time")

    create_index_safe(db["bq_meta_project"],
        [("project_code", ASCENDING), ("version", ASCENDING)],
        report, unique=True, name="uq_project_code_version")
    create_index_safe(db["bq_meta_feature_versions"],
        [("feature_version", ASCENDING)],
        report, unique=True, name="uq_feature_version")
    create_index_safe(db["bq_meta_label_versions"],
        [("label_version", ASCENDING)],
        report, unique=True, name="uq_label_version")
    create_index_safe(db["bq_meta_dataset_versions"],
        [("dataset_version", ASCENDING)],
        report, unique=True, name="uq_dataset_version")
    create_index_safe(db["bq_meta_model_versions"],
        [("model_name", ASCENDING), ("model_version", ASCENDING)],
        report, unique=True, name="uq_model_name_version")
    create_index_safe(db["bq_meta_runs"],
        [("run_id", ASCENDING)],
        report, unique=True, name="uq_run_id")


def upsert_meta_documents(db: Database, config: Dict[str, Any], report: Dict[str, Any], run_id: str) -> None:
    now = utc_now()
    raw = config["raw_collections"]
    bq = config["bq_collections"]
    symbol = config["symbol"]

    print("\n=== Upserting metadata ===", flush=True)

    project_doc = {
        "project_name": PROJECT_NAME,
        "project_code": PROJECT_CODE,
        "version": PROJECT_VERSION,
        "symbol": symbol,
        "anchor_timeframe": ANCHOR_TIMEFRAME,
        "raw_collections": raw,
        "bq_collections": bq,
        "architecture": {
            "book_entry_model": {
                "timeframes": ["M1", "M5", "M15"],
                "target": "Entry direction detection",
                "output": ["p_buy", "p_sell", "p_notrade"],
            },
            "trade_quality_model": {
                "timeframes": ["M30", "H1", "H4"],
                "target": "Trade quality and market phase detection",
                "output": ["buy_quality_score", "sell_quality_score", "fake_score", "range_score", "trend_phase"],
            },
            "api_decision_engine": {
                "target": "Combine Book and Quality outputs",
                "output": ["final_signal", "lot_multiplier", "trailing_mode", "breakeven_mode", "tp_mode", "risk_mode", "reason"],
            },
        },
        "rules": {
            "raw_collections_policy": "read_only",
            "bq_collections_policy": "project_outputs_only",
            "every_script_must_generate_report": True,
            "no_leak_rule": "For each anchor_time, only candles already closed at or before anchor_time may be used.",
        },
        "status": "active",
        "updated_at": now,
    }

    db["bq_meta_project"].update_one(
        {"project_code": PROJECT_CODE, "version": PROJECT_VERSION},
        {"$set": project_doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    report["metadata_upserts"].append("bq_meta_project")

    feature_docs = [
        {
            "feature_version": "book_entry_features_v1",
            "target_collection": bq["features_book_entry"],
            "model_type": "book_entry",
            "anchor_timeframe": "M15",
            "input_timeframes": ["M1", "M5", "M15"],
            "description": "Entry direction features using only M1, M5 and M15 closed candles.",
            "no_leak_rule": "Only candles with close time <= anchor_time are allowed.",
            "status": "draft",
        },
        {
            "feature_version": "trade_quality_features_v1",
            "target_collection": bq["features_trade_quality"],
            "model_type": "trade_quality",
            "anchor_timeframe": "M15",
            "input_timeframes": ["M30", "H1", "H4"],
            "description": "Trade quality and market phase features using only M30, H1 and H4 closed candles.",
            "no_leak_rule": "Only higher timeframe candles closed before or at anchor_time are allowed.",
            "status": "draft",
        },
    ]

    for doc in feature_docs:
        doc["updated_at"] = now
        db["bq_meta_feature_versions"].update_one(
            {"feature_version": doc["feature_version"]},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        report["metadata_upserts"].append(f"feature:{doc['feature_version']}")

    label_docs = [
        {
            "label_version": "book_entry_label_v1",
            "target_collection": bq["labels_book_entry"],
            "model_type": "book_entry",
            "anchor_timeframe": "M15",
            "entry_rule": "Entry at next M15 open after anchor_time.",
            "config": {"tp_usd": 2.0, "sl_usd": 2.0, "horizon_m15": 8, "ambiguous_policy": "NOTRADE"},
            "output_labels": ["BUY", "SELL", "NOTRADE"],
            "status": "draft",
        },
        {
            "label_version": "trade_quality_label_v1",
            "target_collection": bq["labels_trade_quality"],
            "model_type": "trade_quality",
            "anchor_timeframe": "M15",
            "entry_rule": "Quality is evaluated from next M15 open after anchor_time.",
            "config": {
                "base_r_usd": 2.0,
                "horizon_m15_short": 8,
                "horizon_m15_mid": 16,
                "horizon_m15_long": 32,
                "quality_score_min": 0,
                "quality_score_max": 100,
            },
            "output_labels": ["buy_quality_score", "sell_quality_score", "fake_score", "range_score", "trend_phase"],
            "trend_phase_classes": [
                "UP_TREND_EARLY", "UP_TREND_MATURE", "UP_TREND_EXHAUSTION",
                "DOWN_TREND_EARLY", "DOWN_TREND_MATURE", "DOWN_TREND_EXHAUSTION",
                "RANGE", "TRANSITION"
            ],
            "status": "draft",
        },
    ]

    for doc in label_docs:
        doc["updated_at"] = now
        db["bq_meta_label_versions"].update_one(
            {"label_version": doc["label_version"]},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        report["metadata_upserts"].append(f"label:{doc['label_version']}")

    dataset_docs = [
        {
            "dataset_version": "dataset_book_entry_m15_v1",
            "target_collection": bq["dataset_book_entry"],
            "model_type": "book_entry",
            "feature_version": "book_entry_features_v1",
            "label_version": "book_entry_label_v1",
            "symbol": symbol,
            "anchor_timeframe": "M15",
            "status": "draft",
        },
        {
            "dataset_version": "dataset_trade_quality_m15_v1",
            "target_collection": bq["dataset_trade_quality"],
            "model_type": "trade_quality",
            "feature_version": "trade_quality_features_v1",
            "label_version": "trade_quality_label_v1",
            "symbol": symbol,
            "anchor_timeframe": "M15",
            "status": "draft",
        },
    ]

    for doc in dataset_docs:
        doc["updated_at"] = now
        db["bq_meta_dataset_versions"].update_one(
            {"dataset_version": doc["dataset_version"]},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        report["metadata_upserts"].append(f"dataset:{doc['dataset_version']}")

    model_docs = [
        {
            "model_name": "book_entry_model",
            "model_version": "untrained_v1",
            "model_type": "book_entry",
            "dataset_version": "dataset_book_entry_m15_v1",
            "target_collection": bq["predictions_book_entry"],
            "output": ["p_buy", "p_sell", "p_notrade"],
            "status": "placeholder",
        },
        {
            "model_name": "trade_quality_model",
            "model_version": "untrained_v1",
            "model_type": "trade_quality",
            "dataset_version": "dataset_trade_quality_m15_v1",
            "target_collection": bq["predictions_trade_quality"],
            "output": ["buy_quality_score", "sell_quality_score", "fake_score", "range_score", "trend_phase"],
            "status": "placeholder",
        },
    ]

    for doc in model_docs:
        doc["updated_at"] = now
        db["bq_meta_model_versions"].update_one(
            {"model_name": doc["model_name"], "model_version": doc["model_version"]},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        report["metadata_upserts"].append(f"model:{doc['model_name']}/{doc['model_version']}")

    db["bq_meta_runs"].update_one(
        {"run_id": run_id},
        {"$set": {
            "run_id": run_id,
            "run_type": "bootstrap",
            "script": SCRIPT_NAME,
            "project_code": PROJECT_CODE,
            "project_version": PROJECT_VERSION,
            "status": "success",
            "created_at": now,
        }},
        upsert=True,
    )
    report["metadata_upserts"].append(f"run:{run_id}")


def print_summary(db: Database, collection_names: List[str], report: Dict[str, Any]) -> None:
    print("\n=== Summary ===", flush=True)
    for name in collection_names:
        count = db[name].estimated_document_count()
        report["final_collection_counts"][name] = count
        print(f"{name:<45} count≈{count:,}", flush=True)


def build_txt_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Book & Quality - MongoDB Bootstrap Report")
    lines.append("=" * 60)
    lines.append(f"Run ID       : {report['run_id']}")
    lines.append(f"Status       : {report['status']}")
    lines.append(f"Database     : {report['database']}")
    lines.append(f"Symbol       : {report['symbol']}")
    lines.append(f"Start Time   : {report['start_time']}")
    lines.append(f"End Time     : {report['end_time']}")
    lines.append(f"Duration Sec : {report['duration_seconds']}")
    lines.append("")
    lines.append(f"Collections Created : {len(report['collections_created'])}")
    lines.append(f"Collections Existing: {len(report['collections_existing'])}")
    lines.append(f"Indexes             : {len(report['indexes_created_or_existing'])}")
    lines.append(f"Metadata Upserts    : {len(report['metadata_upserts'])}")
    lines.append(f"Warnings            : {len(report['warnings'])}")
    lines.append(f"Errors              : {len(report['errors'])}")
    lines.append("")

    lines.append("Final Collection Counts")
    lines.append("-" * 60)
    for name, count in report["final_collection_counts"].items():
        lines.append(f"{name:<45} count≈{count}")

    if report["warnings"]:
        lines.append("")
        lines.append("Warnings")
        lines.append("-" * 60)
        for warning in report["warnings"]:
            lines.append(f"- {warning}")

    if report["errors"]:
        lines.append("")
        lines.append("Errors")
        lines.append("-" * 60)
        for error in report["errors"]:
            lines.append(f"- {error}")

    return "\n".join(lines)


def save_reports(report: Dict[str, Any], stamp: str) -> None:
    reports_dir = get_reports_dir()
    json_path = reports_dir / f"mongo_bootstrap_report_{stamp}.json"
    txt_path = reports_dir / f"mongo_bootstrap_report_{stamp}.txt"

    report["report_json_path"] = str(json_path)
    report["report_txt_path"] = str(txt_path)

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(build_txt_report(report), encoding="utf-8")


def main() -> None:
    start_dt = utc_now()
    stamp = file_stamp(start_dt)
    run_id = f"mongo_bootstrap_{stamp}"

    config = load_config()
    raw = config["raw_collections"]
    bq = config["bq_collections"]

    bq_collection_names = list(bq.values()) + META_COLLECTIONS

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": run_id,
        "project_root": str(get_project_root()),
        "database": config.get("database"),
        "symbol": config.get("symbol"),
        "mongo_uri": config.get("mongo_uri"),
        "start_time": start_dt.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "status": "running",
        "raw_collections": {},
        "collections_created": [],
        "collections_existing": [],
        "indexes_created_or_existing": [],
        "metadata_upserts": [],
        "final_collection_counts": {},
        "warnings": [],
        "errors": [],
    }

    print("=== Book & Quality MongoDB Bootstrap ===", flush=True)
    print(f"Project Root: {report['project_root']}", flush=True)
    print(f"Database    : {report['database']}", flush=True)
    print(f"Symbol      : {report['symbol']}", flush=True)

    try:
        db = connect(config)

        check_raw_collections(db, raw, report)
        create_all_bq_collections(db, bq_collection_names, report)
        create_indexes(db, bq, report)
        upsert_meta_documents(db, config, report, run_id)
        print_summary(db, bq_collection_names, report)

        report["status"] = "success" if not report["errors"] else "failed"

    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc))
        print(f"[ERROR] {exc}", flush=True)

    finally:
        end_dt = utc_now()
        report["end_time"] = end_dt.isoformat()
        report["duration_seconds"] = round((end_dt - start_dt).total_seconds(), 3)
        save_reports(report, stamp)

    print(f"\nJSON Report: {report.get('report_json_path')}", flush=True)
    print(f"TXT Report : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] == "success" else "[FAILED]", flush=True)

    if report["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
