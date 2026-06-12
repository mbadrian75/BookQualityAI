# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Build MA Quality Dataset M30 v1

Location:
    Book_Quality/04_dataset/ma_quality/01_build_ma_quality_dataset_m30_v1.py

Input:
    bq2_features_ma_quality_m30_v1

Output:
    bq2_dataset_ma_quality_m30_v1

TEST ONLY:
    cd C:/Project/Book_Quality/04_dataset/ma_quality
    python -u 01_build_ma_quality_dataset_m30_v1.py --limit 1000 --reset

FULL BUILD:
    python -u 01_build_ma_quality_dataset_m30_v1.py --reset
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError


SCRIPT_NAME = "01_build_ma_quality_dataset_m30_v1.py"
DATASET_VERSION = "ma_quality_dataset_m30_v1"
FEATURE_VERSION = "ma_quality_features_m30_v1"
LABEL_VERSION = "ma_quality_label_m30_v1"
BATCH_SIZE = 1000

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "bq2_collections": {
        "features_ma_quality": "bq2_features_ma_quality_m30_v1",
        "dataset_ma_quality": "bq2_dataset_ma_quality_m30_v1",
    },
    "ma_quality_dataset_m30_v1": {
        "train_end": "2023-01-01 00:00:00",
        "valid_end": "2024-01-01 00:00:00",
        "test_end": None,
        "require_no_leak_ok": True,
        "keep_feature_dict": False
    },
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reports_dir() -> Path:
    p = project_root() / "04_dataset" / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def exports_dir() -> Path:
    p = project_root() / "04_dataset" / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read config {path}: {exc}", flush=True)
        return None


def deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG
    p = project_root() / "00_config" / "bq2_config.json"
    file_cfg = read_json(p)
    if file_cfg:
        cfg = deep_merge(cfg, file_cfg)

    cfg.setdefault("bq2_collections", {})
    cfg["bq2_collections"]["features_ma_quality"] = "bq2_features_ma_quality_m30_v1"
    cfg["bq2_collections"]["dataset_ma_quality"] = "bq2_dataset_ma_quality_m30_v1"
    return cfg


def connect(cfg: Dict[str, Any]) -> Database:
    client = MongoClient(cfg["mongo_uri"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[cfg["database"]]


def parse_dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v.replace(tzinfo=None)
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=None)
    raise ValueError(f"Unsupported datetime: {v!r}")


def parse_dt_optional(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=None)


def sf(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def split_for(anchor_time: datetime, train_end: datetime, valid_end: datetime, test_end: Optional[datetime]) -> str:
    if anchor_time < train_end:
        return "train"
    if anchor_time < valid_end:
        return "valid"
    if test_end is None or anchor_time < test_end:
        return "test"
    return "ignored"


def make_update_op(symbol: str, doc: Dict[str, Any]) -> UpdateOne:
    created_at = doc.pop("created_at", utc_now())
    return UpdateOne(
        {"symbol": symbol, "anchor_time": doc["anchor_time"]},
        {"$set": doc, "$setOnInsert": {"created_at": created_at}},
        upsert=True,
    )


def execute_bulk(db: Database, coll_name: str, ops: List[UpdateOne], report: Dict[str, Any]) -> None:
    if not ops:
        return
    try:
        res = db[coll_name].bulk_write(ops, ordered=False)
        report["counts"]["upserted_or_modified"] += res.upserted_count + res.modified_count
        report["counts"]["inserted"] += res.upserted_count
        report["counts"]["modified"] += res.modified_count
        report["counts"]["matched"] += res.matched_count
    except BulkWriteError as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc.details)[:20000])
        raise


def compact_y(y: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "predicted_direction", "prediction_probability_label",
        "m30_future_direction", "h1_future_direction", "h4_future_direction",
        "m30_phase", "h1_phase", "h4_phase",
        "buy_quality_score", "sell_quality_score",
        "buy_risk_mode", "sell_risk_mode",
        "buy_lot_multiplier", "sell_lot_multiplier",
        "buy_tp_mode", "sell_tp_mode",
        "buy_sl_mode", "sell_sl_mode",
        "buy_trailing_mode", "sell_trailing_mode",
        "buy_exit_mode", "sell_exit_mode",
    ]
    out = {}
    for k in keys:
        v = y.get(k)
        if k.endswith("_score") or k.endswith("_multiplier") or k == "prediction_probability_label":
            out[k] = sf(v)
        else:
            out[k] = v
    return out


def save_reports(report: Dict[str, Any], rs: str) -> None:
    jp = reports_dir() / f"ma_quality_dataset_m30_v1_report_{rs}.json"
    tp = reports_dir() / f"ma_quality_dataset_m30_v1_report_{rs}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "Book & Quality v2 - MA Quality Dataset M30 v1 Report",
        "=" * 74,
        f"{'run_id':<34}: {report.get('run_id')}",
        f"{'status':<34}: {report.get('status')}",
        f"{'database':<34}: {report.get('database')}",
        f"{'symbol':<34}: {report.get('symbol')}",
        f"{'start_time':<34}: {report.get('start_time')}",
        f"{'end_time':<34}: {report.get('end_time')}",
        f"{'duration_seconds':<34}: {report.get('duration_seconds')}",
        "",
        "Collections",
        "-" * 74,
    ]
    for k, v in report["collections"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Config", "-" * 74]
    for k, v in report["dataset_config"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Counts", "-" * 74]
    for k, v in report["counts"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Feature Schema", "-" * 74]
    for k, v in report["feature_schema"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Split Distribution", "-" * 74]
    for k, v in report["split_distribution"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Target Distribution", "-" * 74]
    for target, dist in report["target_distribution"].items():
        lines.append(f"{target}:")
        for k, v in dist.items():
            lines.append(f"  {str(k):<31}: {v}")

    if report["schema_mismatch_examples"]:
        lines += ["", "Schema Mismatch Examples", "-" * 74]
        for item in report["schema_mismatch_examples"][:10]:
            lines.append(json.dumps(item, ensure_ascii=False))

    if report["warnings"]:
        lines += ["", "Warnings", "-" * 74]
        lines += [f"- {w}" for w in report["warnings"]]

    if report["errors"]:
        lines += ["", "Errors", "-" * 74]
        lines += [f"- {e}" for e in report["errors"][:50]]

    tp.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build bq2 MA Quality Dataset M30 v1.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    rs = stamp(started)

    cfg = load_config()
    db_name = cfg["database"]
    symbol = cfg["symbol"]
    bq2 = cfg["bq2_collections"]
    dcfg = cfg.get("ma_quality_dataset_m30_v1", DEFAULT_CONFIG["ma_quality_dataset_m30_v1"])

    source_coll = bq2["features_ma_quality"]
    target_coll = bq2["dataset_ma_quality"]

    train_end = parse_dt_optional(dcfg.get("train_end")) or datetime(2023, 1, 1)
    valid_end = parse_dt_optional(dcfg.get("valid_end")) or datetime(2024, 1, 1)
    test_end = parse_dt_optional(dcfg.get("test_end"))
    keep_feature_dict = bool(dcfg.get("keep_feature_dict", False))
    require_no_leak_ok = bool(dcfg.get("require_no_leak_ok", True))

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"ma_quality_dataset_m30_v1_{rs}",
        "status": "running",
        "database": db_name,
        "symbol": symbol,
        "start_time": started.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "collections": {"features_source": source_coll, "target": target_coll},
        "dataset_config": {
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "anchor_timeframe": "M30",
            "train_end": str(train_end),
            "valid_end": str(valid_end),
            "test_end": str(test_end) if test_end else None,
            "require_no_leak_ok": require_no_leak_ok,
            "keep_feature_dict": keep_feature_dict,
        },
        "counts": {
            "features_seen": 0,
            "rows_built": 0,
            "upserted_or_modified": 0,
            "inserted": 0,
            "matched": 0,
            "modified": 0,
            "skipped_existing": 0,
            "skipped_missing_features": 0,
            "skipped_missing_y": 0,
            "skipped_no_leak_not_ok": 0,
            "skipped_schema_mismatch": 0,
            "skipped_ignored_split": 0,
            "row_errors": 0,
        },
        "feature_schema": {},
        "split_distribution": {},
        "target_distribution": {
            "predicted_direction": {},
            "buy_risk_mode": {},
            "sell_risk_mode": {},
        },
        "schema_mismatch_examples": [],
        "warnings": [],
        "errors": [],
        "args": vars(args),
    }

    print("=== Book & Quality v2 - Build MA Quality Dataset M30 v1 ===", flush=True)
    print(f"Database : {db_name}", flush=True)
    print(f"Symbol   : {symbol}", flush=True)
    print(f"Source   : {source_coll}", flush=True)
    print(f"Target   : {target_coll}", flush=True)

    try:
        db = connect(cfg)

        if args.reset:
            deleted = db[target_coll].delete_many({})
            report["reset_deleted_count"] = deleted.deleted_count
            print(f"[RESET] Deleted {deleted.deleted_count:,} docs from {target_coll}", flush=True)

        db[target_coll].create_index([("symbol", ASCENDING), ("anchor_time", ASCENDING)], unique=True, name="uq_symbol_anchor_time")
        db[target_coll].create_index([("dataset_version", ASCENDING), ("anchor_time", ASCENDING)], name="ix_dataset_version_anchor_time")
        db[target_coll].create_index([("split", ASCENDING), ("anchor_time", ASCENDING)], name="ix_split_anchor_time")
        db[target_coll].create_index([("y.predicted_direction", ASCENDING), ("split", ASCENDING)], name="ix_y_direction_split")
        db[target_coll].create_index([("feature_count", ASCENDING)], name="ix_feature_count")

        q: Dict[str, Any] = {"feature_version": FEATURE_VERSION, "label_version": LABEL_VERSION}
        if args.start or args.end:
            tf: Dict[str, Any] = {}
            if args.start:
                tf["$gte"] = parse_dt_optional(args.start)
            if args.end:
                tf["$lte"] = parse_dt_optional(args.end)
            q["anchor_time"] = tf

        first = db[source_coll].find_one(q, projection={"features": 1, "feature_count": 1, "_id": 0}, sort=[("anchor_time", ASCENDING)])
        if not first or not isinstance(first.get("features"), dict):
            raise RuntimeError("No valid source feature document found.")

        feature_names = sorted(first["features"].keys())
        expected_keys = set(feature_names)
        feature_count = len(feature_names)

        export_path = exports_dir() / f"ma_quality_m30_v1_feature_names_{rs}.json"
        export_path.write_text(json.dumps({
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "feature_count": feature_count,
            "feature_names": feature_names,
            "created_at": utc_now().isoformat()
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        report["feature_schema"] = {
            "feature_count": feature_count,
            "first_doc_feature_count_field": first.get("feature_count"),
            "feature_names_export": str(export_path),
        }

        cur = (
            db[source_coll]
            .find(q, projection={
                "_id": 0,
                "symbol": 1, "anchor_time": 1, "decision_time": 1,
                "entry_time": 1, "entry_price": 1,
                "features": 1, "feature_count": 1,
                "feature_meta": 1, "y": 1,
            })
            .sort("anchor_time", ASCENDING)
            .batch_size(10000)
        )

        ops: List[UpdateOne] = []

        for src in cur:
            try:
                if args.limit is not None and report["counts"]["features_seen"] >= args.limit:
                    break

                report["counts"]["features_seen"] += 1
                src_symbol = src.get("symbol", symbol)
                anchor_time = parse_dt(src["anchor_time"])
                decision_time = parse_dt(src["decision_time"])
                split = split_for(anchor_time, train_end, valid_end, test_end)

                if split == "ignored":
                    report["counts"]["skipped_ignored_split"] += 1
                    continue

                if args.skip_existing:
                    exists = db[target_coll].find_one({"symbol": src_symbol, "anchor_time": anchor_time}, projection={"_id": 1})
                    if exists:
                        report["counts"]["skipped_existing"] += 1
                        continue

                features = src.get("features")
                y_full = src.get("y")
                meta = src.get("feature_meta", {})

                if not isinstance(features, dict) or not features:
                    report["counts"]["skipped_missing_features"] += 1
                    continue
                if not isinstance(y_full, dict) or not y_full:
                    report["counts"]["skipped_missing_y"] += 1
                    continue
                if require_no_leak_ok and meta.get("no_leak_ok") is not True:
                    report["counts"]["skipped_no_leak_not_ok"] += 1
                    continue

                keys = set(features.keys())
                if keys != expected_keys:
                    report["counts"]["skipped_schema_mismatch"] += 1
                    if len(report["schema_mismatch_examples"]) < 10:
                        report["schema_mismatch_examples"].append({
                            "anchor_time": str(anchor_time),
                            "feature_count": len(keys),
                            "missing": sorted(list(expected_keys - keys))[:20],
                            "extra": sorted(list(keys - expected_keys))[:20],
                        })
                    continue

                y = compact_y(y_full)
                x = [sf(features[name]) for name in feature_names]

                doc = {
                    "symbol": src_symbol,
                    "anchor_time": anchor_time,
                    "decision_time": decision_time,
                    "entry_time": parse_dt(src.get("entry_time", decision_time)),
                    "entry_price": sf(src.get("entry_price", 0.0)),
                    "dataset_version": DATASET_VERSION,
                    "feature_version": FEATURE_VERSION,
                    "label_version": LABEL_VERSION,
                    "model_type": "ma_phase_quality",
                    "anchor_timeframe": "M30",
                    "input_timeframes": ["M30", "H1", "H4"],
                    "split": split,
                    "x": x,
                    "feature_count": feature_count,
                    "y": y,
                    "y_full": y_full,
                    "feature_meta": {
                        "no_leak_ok": meta.get("no_leak_ok"),
                        "max_feature_time": meta.get("max_feature_time"),
                        "tf_last_closed_time": meta.get("tf_last_closed_time"),
                        "tf_current_direction": meta.get("tf_current_direction"),
                    },
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                }
                if keep_feature_dict:
                    doc["features"] = features

                ops.append(make_update_op(src_symbol, doc))
                report["counts"]["rows_built"] += 1
                report["split_distribution"][split] = report["split_distribution"].get(split, 0) + 1

                for target_name, target_key in [
                    ("predicted_direction", "predicted_direction"),
                    ("buy_risk_mode", "buy_risk_mode"),
                    ("sell_risk_mode", "sell_risk_mode"),
                ]:
                    val = y.get(target_key)
                    d = report["target_distribution"][target_name]
                    d[val] = d.get(val, 0) + 1

                if len(ops) >= BATCH_SIZE:
                    execute_bulk(db, target_coll, ops, report)
                    ops = []
                    print(
                        f"[PROGRESS] seen={report['counts']['features_seen']:,} "
                        f"built={report['counts']['rows_built']:,} "
                        f"written≈{report['counts']['upserted_or_modified']:,}",
                        flush=True,
                    )

            except Exception as row_exc:
                report["counts"]["row_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(f"anchor={src.get('anchor_time')} | {row_exc}")

        if ops:
            execute_bulk(db, target_coll, ops, report)

        report["final_target_count"] = db[target_coll].estimated_document_count()
        report["status"] = "success" if report["counts"]["row_errors"] == 0 and not report["errors"] else "success_with_row_errors"

    except Exception as exc:
        report["status"] = "failed"
        if not report["errors"]:
            report["errors"].append(str(exc))
        print(f"[ERROR] {exc}", flush=True)

    finally:
        ended = utc_now()
        report["end_time"] = ended.isoformat()
        report["duration_seconds"] = round((ended - started).total_seconds(), 3)
        save_reports(report, rs)

    print("\n=== Final Summary ===", flush=True)
    print(f"Status            : {report['status']}", flush=True)
    print(f"Features Seen     : {report['counts']['features_seen']:,}", flush=True)
    print(f"Rows Built        : {report['counts']['rows_built']:,}", flush=True)
    print(f"Upserted/Modified : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Row Errors        : {report['counts']['row_errors']:,}", flush=True)
    print(f"Feature Count     : {report['feature_schema'].get('feature_count')}", flush=True)
    print(f"Split Distribution: {report['split_distribution']}", flush=True)
    print(f"JSON Report       : {report.get('report_json_path')}", flush=True)
    print(f"TXT Report        : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
