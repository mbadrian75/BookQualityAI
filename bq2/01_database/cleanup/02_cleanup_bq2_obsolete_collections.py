# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Cleanup Obsolete Collections

Default mode is DRY RUN. Use --execute to actually drop collections.

DRY RUN:
    cd C:/Project/Book_Quality
    python -u 01_database/cleanup/02_cleanup_bq2_obsolete_collections.py

EXECUTE:
    python -u 01_database/cleanup/02_cleanup_bq2_obsolete_collections.py --execute

Optional old bq_ v1 cleanup:
    python -u 01_database/cleanup/02_cleanup_bq2_obsolete_collections.py --include-bq1 --execute
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pymongo import MongoClient


DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
}

RAW_PREFIXES_NEVER_DROP = ("xauusd_",)

OBSOLETE_BQ2_COLLECTIONS = ['bq2_labels_ma_quality_m15_v1', 'bq2_features_ma_quality_m15_v1', 'bq2_dataset_ma_quality_m15_v1', 'bq2_predictions_ma_quality_m15_v1', 'bq2_dataset_ma_quality_m30_v1', 'bq2_predictions_ma_quality_m30_v1']

OPTIONAL_BQ1_COLLECTIONS = [
    "bq_labels_book_entry_m15_v1",
    "bq_features_book_entry_m15_v1",
    "bq_dataset_book_entry_m15_v1",
    "bq_predictions_book_entry_m15_v1",
    "bq_labels_trade_quality_m15_v1",
    "bq_features_trade_quality_m15_v1",
    "bq_dataset_trade_quality_m15_v1",
    "bq_predictions_trade_quality_m15_v1",
    "bq_api_decisions_m15_v1",
]

KEEP_BQ2_COLLECTIONS = [
    "bq2_labels_ma_quality_m30_v1",
    "bq2_features_ma_quality_m30_v1",
    "bq2_labels_candle_book_m15_v1",
    "bq2_features_candle_book_m15_v1",
    "bq2_dataset_candle_book_m15_v1",
    "bq2_predictions_candle_book_m15_v1",
    "bq2_api_decisions_m15_v1",
    "bq2_meta_project",
    "bq2_meta_label_versions",
    "bq2_meta_feature_versions",
    "bq2_meta_dataset_versions",
    "bq2_meta_model_versions",
    "bq2_meta_runs",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def project_root() -> Path:
    return Path.cwd().resolve()


def reports_dir(root: Path) -> Path:
    p = root / "01_database" / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_config(root: Path) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    file_cfg = read_json(root / "00_config" / "bq2_config.json")
    if file_cfg:
        cfg["mongo_uri"] = file_cfg.get("mongo_uri", cfg["mongo_uri"])
        cfg["database"] = file_cfg.get("database", cfg["database"])
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup obsolete Book & Quality collections.")
    parser.add_argument("--execute", action="store_true", help="Actually drop obsolete collections. Without this, DRY RUN only.")
    parser.add_argument("--include-bq1", action="store_true", help="Also drop old bq_ v1 experiment collections.")
    return parser.parse_args()


def safe_to_drop(name: str) -> bool:
    return not name.startswith(RAW_PREFIXES_NEVER_DROP)


def main() -> None:
    args = parse_args()
    started = utc_now()
    root = project_root()
    cfg = load_config(root)

    client = MongoClient(cfg["mongo_uri"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[cfg["database"]]

    existing = set(db.list_collection_names())

    requested = list(OBSOLETE_BQ2_COLLECTIONS)
    if args.include_bq1:
        requested.extend(OPTIONAL_BQ1_COLLECTIONS)

    actions: List[Dict[str, Any]] = []
    blocked: List[str] = []
    missing: List[str] = []

    for coll_name in requested:
        if coll_name not in existing:
            missing.append(coll_name)
            actions.append({"collection": coll_name, "exists": False, "action": "missing_skipped"})
            continue

        if not safe_to_drop(coll_name):
            blocked.append(coll_name)
            actions.append({"collection": coll_name, "exists": True, "action": "blocked_raw_protection"})
            continue

        count = db[coll_name].estimated_document_count()
        if args.execute:
            db[coll_name].drop()
            action = "dropped"
        else:
            action = "dry_run_would_drop"

        actions.append({
            "collection": coll_name,
            "exists": True,
            "estimated_count": count,
            "action": action,
        })

    ended = utc_now()
    report = {
        "run_id": f"cleanup_bq2_obsolete_collections_{stamp(started)}",
        "status": "success",
        "mode": "execute" if args.execute else "dry_run",
        "database": cfg["database"],
        "mongo_uri": cfg["mongo_uri"],
        "raw_data_modified": False,
        "include_bq1": args.include_bq1,
        "start_time": started.isoformat(),
        "end_time": ended.isoformat(),
        "duration_seconds": round((ended - started).total_seconds(), 3),
        "obsolete_bq2_collections": OBSOLETE_BQ2_COLLECTIONS,
        "optional_bq1_collections": OPTIONAL_BQ1_COLLECTIONS if args.include_bq1 else [],
        "keep_bq2_collections": KEEP_BQ2_COLLECTIONS,
        "missing": missing,
        "blocked": blocked,
        "actions": actions,
    }

    jp = reports_dir(root) / f"cleanup_bq2_obsolete_collections_report_{stamp(started)}.json"
    tp = reports_dir(root) / f"cleanup_bq2_obsolete_collections_report_{stamp(started)}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)

    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "Book & Quality v2 - Cleanup Obsolete Collections Report",
        "=" * 74,
        f"Run ID            : {report['run_id']}",
        f"Status            : {report['status']}",
        f"Mode              : {report['mode']}",
        f"Database          : {report['database']}",
        f"Raw Data Modified : {report['raw_data_modified']}",
        f"Include BQ1       : {report['include_bq1']}",
        f"Duration Sec      : {report['duration_seconds']}",
        "",
        "Actions",
        "-" * 74,
    ]

    for action in actions:
        count = action.get("estimated_count", "")
        lines.append(f"{action['collection']:<45} {action['action']:<25} {count}")

    lines.extend(["", "Kept BQ2 Collections", "-" * 74])
    lines.extend(KEEP_BQ2_COLLECTIONS)

    if blocked:
        lines.extend(["", "Blocked", "-" * 74])
        lines.extend(blocked)

    tp.write_text("\n".join(lines), encoding="utf-8")

    print("=== Book & Quality v2 - Cleanup Obsolete Collections ===")
    print(f"Mode              : {report['mode']}")
    print(f"Database          : {report['database']}")
    print("Raw Data Modified : False")
    print(f"Include BQ1       : {report['include_bq1']}")
    print(f"Actions           : {len(actions)}")
    print(f"Report TXT        : {tp}")
    print(f"Report JSON       : {jp}")
    print("[DONE]")


if __name__ == "__main__":
    main()
