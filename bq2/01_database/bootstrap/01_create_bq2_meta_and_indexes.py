# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Create Meta Collections and Indexes

Run:
    cd C:\Project\Book_Quality
    python -u 01_database\bootstrap\01_create_bq2_meta_and_indexes.py

Reset only bq2 collections:
    python -u 01_database\bootstrap\01_create_bq2_meta_and_indexes.py --reset-bq2
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from pymongo import MongoClient, ASCENDING

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "raw_collections": {
        "m1": "xauusd_m1", "m5": "xauusd_m5", "m15": "xauusd_m15",
        "m30": "xauusd_m30", "h1": "xauusd_h1", "h4": "xauusd_h4"
    },
    "bq2_collections": {
        "labels_ma_quality": "bq2_labels_ma_quality_m15_v1",
        "features_ma_quality": "bq2_features_ma_quality_m15_v1",
        "dataset_ma_quality": "bq2_dataset_ma_quality_m15_v1",
        "predictions_ma_quality": "bq2_predictions_ma_quality_m15_v1",
        "labels_candle_book": "bq2_labels_candle_book_m15_v1",
        "features_candle_book": "bq2_features_candle_book_m15_v1",
        "dataset_candle_book": "bq2_dataset_candle_book_m15_v1",
        "predictions_candle_book": "bq2_predictions_candle_book_m15_v1",
        "api_decisions": "bq2_api_decisions_m15_v1"
    }
}
META_COLLECTIONS = [
    "bq2_meta_project", "bq2_meta_label_versions", "bq2_meta_feature_versions",
    "bq2_meta_dataset_versions", "bq2_meta_model_versions", "bq2_meta_runs"
]

def now_utc(): return datetime.now(timezone.utc)
def stamp(dt): return dt.strftime("%Y%m%d_%H%M%S")
def root(): return Path(__file__).resolve().parents[2]
def reports_dir():
    p = root()/"01_database/reports"; p.mkdir(parents=True, exist_ok=True); return p
def read_json(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
def deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
def load_config():
    cfg = DEFAULT_CONFIG
    file_cfg = read_json(root()/"00_config/bq2_config.json")
    return deep_merge(cfg, file_cfg) if file_cfg else cfg

def ensure_coll(db, name, actions):
    if name not in db.list_collection_names():
        db.create_collection(name)
        actions.append({"collection": name, "action": "created"})
    else:
        actions.append({"collection": name, "action": "exists"})

def idx(db, coll, spec, name, actions, unique=False):
    db[coll].create_index(spec, name=name, unique=unique)
    actions.append({"collection": coll, "action": "index", "name": name})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-bq2", action="store_true")
    args = ap.parse_args()

    started = now_utc()
    cfg = load_config()
    client = MongoClient(cfg["mongo_uri"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[cfg["database"]]
    actions = []

    all_bq2 = list(cfg["bq2_collections"].values()) + META_COLLECTIONS
    if args.reset_bq2:
        for c in all_bq2:
            if c.startswith("bq2_"):
                db[c].drop()
                actions.append({"collection": c, "action": "dropped"})

    for key, coll in cfg["bq2_collections"].items():
        ensure_coll(db, coll, actions)
        if key != "api_decisions":
            idx(db, coll, [("symbol", ASCENDING), ("anchor_time", ASCENDING)], "uq_symbol_anchor_time", actions, unique=True)
        if key.startswith("labels_"):
            idx(db, coll, [("label_version", ASCENDING), ("anchor_time", ASCENDING)], "ix_label_version_anchor_time", actions)
        elif key.startswith("features_"):
            idx(db, coll, [("feature_version", ASCENDING), ("anchor_time", ASCENDING)], "ix_feature_version_anchor_time", actions)
            idx(db, coll, [("feature_version", ASCENDING), ("feature_count", ASCENDING)], "ix_feature_version_feature_count", actions)
        elif key.startswith("dataset_"):
            idx(db, coll, [("dataset_version", ASCENDING), ("anchor_time", ASCENDING)], "ix_dataset_version_anchor_time", actions)
            idx(db, coll, [("split", ASCENDING), ("anchor_time", ASCENDING)], "ix_split_anchor_time", actions)
        elif key.startswith("predictions_"):
            idx(db, coll, [("model_version", ASCENDING), ("anchor_time", ASCENDING)], "ix_model_version_anchor_time", actions)
        elif key == "api_decisions":
            idx(db, coll, [("symbol", ASCENDING), ("decision_time", ASCENDING)], "ix_symbol_decision_time", actions)
            idx(db, coll, [("final_signal", ASCENDING), ("decision_time", ASCENDING)], "ix_final_signal_decision_time", actions)

    for coll in META_COLLECTIONS:
        ensure_coll(db, coll, actions)
        idx(db, coll, [("name", ASCENDING)], "ix_name", actions)
        idx(db, coll, [("version", ASCENDING)], "ix_version", actions)

    now = now_utc()
    db["bq2_meta_project"].update_one(
        {"name": "Book & Quality v2", "version": "bq2_v1"},
        {"$set": {
            "name": "Book & Quality v2", "version": "bq2_v1",
            "symbol": cfg["symbol"], "database": cfg["database"],
            "raw_collections": cfg["raw_collections"],
            "bq2_collections": cfg["bq2_collections"],
            "design": {
                "quality_model": "MA Phase & Quality predicts scenario, probability, risk and management.",
                "book_model": "Candle Book confirms entry with M1/M5/M15.",
                "api_rule": "Quality predicts; Book confirms; low quality probability can trade small lot if confirmed."
            },
            "updated_at": now
        }, "$setOnInsert": {"created_at": now}},
        upsert=True
    )
    actions.append({"collection": "bq2_meta_project", "action": "upsert_project_meta"})

    ended = now_utc()
    report = {
        "run_id": f"create_bq2_meta_indexes_{stamp(started)}",
        "status": "success",
        "database": cfg["database"], "symbol": cfg["symbol"],
        "start_time": started.isoformat(), "end_time": ended.isoformat(),
        "duration_seconds": round((ended-started).total_seconds(), 3),
        "reset_bq2": args.reset_bq2,
        "raw_data_modified": False,
        "actions_count": len(actions), "actions": actions,
        "errors": []
    }
    jp = reports_dir()/f"create_bq2_meta_indexes_report_{stamp(started)}.json"
    tp = reports_dir()/f"create_bq2_meta_indexes_report_{stamp(started)}.txt"
    report["report_json_path"] = str(jp); report["report_txt_path"] = str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tp.write_text("\n".join([
        "Book & Quality v2 - Meta and Indexes Report",
        "="*74,
        f"Run ID            : {report['run_id']}",
        f"Status            : {report['status']}",
        f"Database          : {report['database']}",
        f"Symbol            : {report['symbol']}",
        f"Reset BQ2         : {report['reset_bq2']}",
        f"Raw Data Modified : {report['raw_data_modified']}",
        f"Duration Sec      : {report['duration_seconds']}",
        f"Actions Count     : {report['actions_count']}",
    ]), encoding="utf-8")

    print("=== Book & Quality v2 - Meta and Indexes ===")
    print("Status            : success")
    print(f"Database          : {cfg['database']}")
    print("Raw Data Modified : False")
    print(f"Reset BQ2         : {args.reset_bq2}")
    print(f"JSON Report       : {jp}")
    print(f"TXT Report        : {tp}")
    print("[DONE]")

if __name__ == "__main__":
    main()
