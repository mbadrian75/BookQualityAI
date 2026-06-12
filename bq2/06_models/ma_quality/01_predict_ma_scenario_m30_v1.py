# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Predict MA Scenario Model M30 v1

Location:
    Book_Quality/06_models/ma_quality/01_predict_ma_scenario_m30_v1.py

Purpose:
    Generate scenario predictions from the trained MA Scenario Model.

Model role:
    Predicts market scenario and probabilities only.
    API Decision Engine will decide:
        lot size, TP/SL, trailing, breakeven, exit, and operational risk.

Input:
    Model:
        06_models/ma_quality/ma_scenario_model_m30_v1_latest.joblib

    Features:
        bq2_features_ma_quality_m30_v1

Output:
    bq2_predictions_ma_scenario_m30_v1

TEST:
    cd C:/Project/BookQuality/bq2/06_models/ma_quality
    python -u 01_predict_ma_scenario_m30_v1.py --limit 1000 --reset

FULL HISTORICAL PREDICTION:
    python -u 01_predict_ma_scenario_m30_v1.py --reset
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError


SCRIPT_NAME = "01_predict_ma_scenario_m30_v1.py"
MODEL_VERSION = "ma_scenario_model_m30_v1"
FEATURE_VERSION = "ma_quality_features_m30_v1"
LABEL_VERSION = "ma_quality_label_m30_v1"
PREDICTION_VERSION = "ma_scenario_prediction_m30_v1"
BATCH_SIZE = 1000

SCENARIO_TARGETS = [
    "predicted_direction",
    "m30_future_direction",
    "h1_future_direction",
    "h4_future_direction",
    "m30_phase",
    "h1_phase",
    "h4_phase",
]

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "bq2_collections": {
        "features_ma_quality": "bq2_features_ma_quality_m30_v1",
        "predictions_ma_scenario": "bq2_predictions_ma_scenario_m30_v1",
    },
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reports_dir() -> Path:
    p = project_root() / "06_models" / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    return project_root() / "06_models" / "ma_quality"


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read json {path}: {exc}", flush=True)
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
    file_cfg = read_json(project_root() / "00_config" / "bq2_config.json")
    if file_cfg:
        cfg = deep_merge(cfg, file_cfg)
    cfg.setdefault("bq2_collections", {})
    cfg["bq2_collections"]["features_ma_quality"] = "bq2_features_ma_quality_m30_v1"
    cfg["bq2_collections"]["predictions_ma_scenario"] = "bq2_predictions_ma_scenario_m30_v1"
    return cfg


def connect(cfg: Dict[str, Any]) -> Database:
    client = MongoClient(cfg["mongo_uri"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[cfg["database"]]


def check_dependencies():
    try:
        import joblib
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Missing dependencies. Install them: pip install joblib numpy scikit-learn") from exc
    return {"joblib": joblib, "np": np}


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


def load_model_bundle(args: argparse.Namespace, deps: Dict[str, Any]) -> Dict[str, Any]:
    if args.model_path:
        path = Path(args.model_path)
    else:
        path = models_dir() / f"{MODEL_VERSION}_latest.joblib"

    if not path.exists():
        raise RuntimeError(f"Model file not found: {path}")

    bundle = deps["joblib"].load(path)

    if bundle.get("model_version") != MODEL_VERSION:
        raise RuntimeError(f"Unexpected model_version: {bundle.get('model_version')}")

    return bundle


def build_x_from_features(features: Dict[str, Any], feature_names: List[str], np_mod: Any) -> Any:
    return np_mod.asarray([[sf(features.get(name, 0.0)) for name in feature_names]], dtype=np_mod.float32)


def proba_dict(model: Any, encoder: Any, x: Any) -> Tuple[str, float, Dict[str, float]]:
    pred_encoded = model.predict(x)[0]
    pred_label = str(encoder.inverse_transform([pred_encoded])[0])

    probs: Dict[str, float] = {}
    confidence = 0.0

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)[0]
        classes = list(encoder.classes_)
        for cls, val in zip(classes, p):
            probs[str(cls)] = float(val)
        confidence = float(max(p))
    else:
        probs[pred_label] = 1.0
        confidence = 1.0

    return pred_label, confidence, probs


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


def save_reports(report: Dict[str, Any], rs: str) -> None:
    jp = reports_dir() / f"ma_scenario_predict_m30_v1_report_{rs}.json"
    tp = reports_dir() / f"ma_scenario_predict_m30_v1_report_{rs}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)

    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "Book & Quality v2 - Predict MA Scenario Model M30 v1 Report",
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

    lines += ["", "Model", "-" * 74]
    for k, v in report["model"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Counts", "-" * 74]
    for k, v in report["counts"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Prediction Distribution", "-" * 74]
    for target, dist in report["prediction_distribution"].items():
        lines.append(f"{target}:")
        for k, v in dist.items():
            lines.append(f"  {str(k):<31}: {v}")

    lines += ["", "Confidence Summary", "-" * 74]
    for target, summary in report["confidence_summary"].items():
        lines.append(f"{target}:")
        for k, v in summary.items():
            lines.append(f"  {k:<31}: {v}")

    if report["errors"]:
        lines += ["", "Errors", "-" * 74]
        lines += [f"- {e}" for e in report["errors"][:50]]

    tp.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MA Scenario M30 v1.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    rs = stamp(started)

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"predict_ma_scenario_m30_v1_{rs}",
        "status": "running",
        "database": None,
        "symbol": None,
        "start_time": started.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "collections": {},
        "model": {},
        "counts": {
            "features_seen": 0,
            "predictions_built": 0,
            "upserted_or_modified": 0,
            "inserted": 0,
            "matched": 0,
            "modified": 0,
            "skipped_existing": 0,
            "skipped_missing_features": 0,
            "skipped_feature_schema": 0,
            "row_errors": 0,
        },
        "prediction_distribution": {target: {} for target in SCENARIO_TARGETS},
        "confidence_accumulator": {target: {"sum": 0.0, "count": 0, "p70": 0, "p80": 0} for target in SCENARIO_TARGETS},
        "confidence_summary": {},
        "errors": [],
        "args": vars(args),
    }

    print("=== Book & Quality v2 - Predict MA Scenario Model M30 v1 ===", flush=True)

    try:
        deps = check_dependencies()
        cfg = load_config()
        db = connect(cfg)
        bundle = load_model_bundle(args, deps)

        source_coll = cfg["bq2_collections"]["features_ma_quality"]
        target_coll = cfg["bq2_collections"]["predictions_ma_scenario"]

        report["database"] = cfg["database"]
        report["symbol"] = cfg["symbol"]
        report["collections"] = {
            "features_source": source_coll,
            "target": target_coll,
        }
        report["model"] = {
            "model_version": bundle.get("model_version"),
            "dataset_version": bundle.get("dataset_version"),
            "feature_version": bundle.get("feature_version"),
            "label_version": bundle.get("label_version"),
            "trained_at": bundle.get("trained_at"),
            "feature_count": bundle.get("feature_count"),
            "targets": bundle.get("targets"),
        }

        feature_names = bundle.get("feature_names") or []
        if not feature_names:
            raise RuntimeError("Model bundle does not contain feature_names. Re-train model after dataset export exists.")

        models = bundle.get("models", {})
        encoders = bundle.get("encoders", {})

        for target in SCENARIO_TARGETS:
            if target not in models or target not in encoders:
                raise RuntimeError(f"Missing model or encoder for target={target}")

        if args.reset:
            deleted = db[target_coll].delete_many({})
            report["reset_deleted_count"] = deleted.deleted_count
            print(f"[RESET] Deleted {deleted.deleted_count:,} docs from {target_coll}", flush=True)

        db[target_coll].create_index([("symbol", ASCENDING), ("anchor_time", ASCENDING)], unique=True, name="uq_symbol_anchor_time")
        db[target_coll].create_index([("prediction_version", ASCENDING), ("anchor_time", ASCENDING)], name="ix_prediction_version_anchor_time")
        db[target_coll].create_index([("predicted_direction", ASCENDING), ("direction_confidence", ASCENDING)], name="ix_direction_confidence")

        query: Dict[str, Any] = {"feature_version": FEATURE_VERSION, "label_version": LABEL_VERSION}
        if args.start or args.end:
            tf: Dict[str, Any] = {}
            if args.start:
                tf["$gte"] = parse_dt_optional(args.start)
            if args.end:
                tf["$lte"] = parse_dt_optional(args.end)
            query["anchor_time"] = tf

        cursor = (
            db[source_coll]
            .find(query, projection={
                "_id": 0,
                "symbol": 1,
                "anchor_time": 1,
                "decision_time": 1,
                "entry_time": 1,
                "entry_price": 1,
                "features": 1,
                "feature_count": 1,
                "feature_meta": 1,
            })
            .sort("anchor_time", ASCENDING)
            .batch_size(10000)
        )

        ops: List[UpdateOne] = []
        np_mod = deps["np"]

        for src in cursor:
            try:
                if args.limit is not None and report["counts"]["features_seen"] >= args.limit:
                    break

                report["counts"]["features_seen"] += 1

                symbol = src.get("symbol", cfg["symbol"])
                anchor_time = parse_dt(src["anchor_time"])
                decision_time = parse_dt(src["decision_time"])

                if args.skip_existing:
                    exists = db[target_coll].find_one({"symbol": symbol, "anchor_time": anchor_time}, projection={"_id": 1})
                    if exists:
                        report["counts"]["skipped_existing"] += 1
                        continue

                features = src.get("features")
                if not isinstance(features, dict) or not features:
                    report["counts"]["skipped_missing_features"] += 1
                    continue

                if any(name not in features for name in feature_names):
                    report["counts"]["skipped_feature_schema"] += 1
                    continue

                x = build_x_from_features(features, feature_names, np_mod)

                target_predictions: Dict[str, Any] = {}
                target_probabilities: Dict[str, Dict[str, float]] = {}

                for target in SCENARIO_TARGETS:
                    pred, conf, probs = proba_dict(models[target], encoders[target], x)
                    target_predictions[target] = pred
                    target_probabilities[target] = probs

                    dist = report["prediction_distribution"][target]
                    dist[pred] = dist.get(pred, 0) + 1

                    acc = report["confidence_accumulator"][target]
                    acc["sum"] += conf
                    acc["count"] += 1
                    if conf >= 0.70:
                        acc["p70"] += 1
                    if conf >= 0.80:
                        acc["p80"] += 1

                direction_probs = target_probabilities["predicted_direction"]
                p_buy = float(direction_probs.get("BUY", 0.0))
                p_sell = float(direction_probs.get("SELL", 0.0))
                p_range = float(direction_probs.get("RANGE", 0.0))

                doc = {
                    "symbol": symbol,
                    "anchor_time": anchor_time,
                    "decision_time": decision_time,
                    "entry_time": parse_dt(src.get("entry_time", decision_time)),
                    "entry_price": sf(src.get("entry_price", 0.0)),

                    "prediction_version": PREDICTION_VERSION,
                    "model_version": MODEL_VERSION,
                    "feature_version": FEATURE_VERSION,
                    "label_version": LABEL_VERSION,
                    "anchor_timeframe": "M30",
                    "input_timeframes": ["M30", "H1", "H4"],

                    "predicted_direction": target_predictions["predicted_direction"],
                    "direction_confidence": max(p_buy, p_sell, p_range),
                    "p_buy": p_buy,
                    "p_sell": p_sell,
                    "p_range": p_range,

                    "m30_future_direction": target_predictions["m30_future_direction"],
                    "h1_future_direction": target_predictions["h1_future_direction"],
                    "h4_future_direction": target_predictions["h4_future_direction"],
                    "m30_phase": target_predictions["m30_phase"],
                    "h1_phase": target_predictions["h1_phase"],
                    "h4_phase": target_predictions["h4_phase"],

                    "target_predictions": target_predictions,
                    "target_probabilities": target_probabilities,

                    "feature_meta": src.get("feature_meta", {}),
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                }

                ops.append(make_update_op(symbol, doc))
                report["counts"]["predictions_built"] += 1

                if len(ops) >= BATCH_SIZE:
                    execute_bulk(db, target_coll, ops, report)
                    ops = []
                    print(
                        f"[PROGRESS] seen={report['counts']['features_seen']:,} "
                        f"built={report['counts']['predictions_built']:,} "
                        f"written≈{report['counts']['upserted_or_modified']:,}",
                        flush=True,
                    )

            except Exception as row_exc:
                report["counts"]["row_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(f"anchor={src.get('anchor_time')} | {row_exc}")

        if ops:
            execute_bulk(db, target_coll, ops, report)

        for target, acc in report["confidence_accumulator"].items():
            count = acc["count"]
            report["confidence_summary"][target] = {
                "avg_confidence": round(acc["sum"] / count, 6) if count else None,
                "p70_rate": round(acc["p70"] / count, 6) if count else None,
                "p80_rate": round(acc["p80"] / count, 6) if count else None,
            }

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
    print(f"Status             : {report['status']}", flush=True)
    print(f"Features Seen      : {report['counts']['features_seen']:,}", flush=True)
    print(f"Predictions Built  : {report['counts']['predictions_built']:,}", flush=True)
    print(f"Upserted/Modified  : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Row Errors         : {report['counts']['row_errors']:,}", flush=True)
    print(f"TXT Report         : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
