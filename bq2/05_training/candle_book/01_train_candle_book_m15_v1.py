# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Train Candle Book Model M15 v1

Location:
    Book_Quality/05_training/candle_book/01_train_candle_book_m15_v1.py

Purpose:
    Train Candle Book Confirmation Model.

Input:
    bq2_dataset_candle_book_m15_v1

Output model:
    06_models/candle_book/candle_book_model_m15_v1_<timestamp>.joblib
    06_models/candle_book/candle_book_model_m15_v1_latest.joblib

Model role:
    Candle Book Model confirms M15 entry timing only.
    It predicts:
        book_direction = BUY / SELL / NOTRADE
    It does NOT decide lot, TP/SL, trailing, breakeven, exit, or risk.

TEST:
    cd C:/Project/BookQuality/bq2/05_training/candle_book
    python -u 01_train_candle_book_m15_v1.py --limit-per-split 5000 --max-iter 50

FULL:
    python -u 01_train_candle_book_m15_v1.py --max-iter 200 --l2-regularization 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING
from pymongo.database import Database


SCRIPT_NAME = "01_train_candle_book_m15_v1.py"
MODEL_VERSION = "candle_book_model_m15_v1"
DATASET_VERSION = "candle_book_dataset_m15_v1"
FEATURE_VERSION = "candle_book_features_m15_v1"
LABEL_VERSION = "candle_book_label_m15_v1"
TARGET = "book_direction"

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "bq2_collections": {
        "dataset_candle_book": "bq2_dataset_candle_book_m15_v1",
    },
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reports_dir() -> Path:
    p = project_root() / "05_training" / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = project_root() / "06_models" / "candle_book"
    p.mkdir(parents=True, exist_ok=True)
    return p


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
    cfg["bq2_collections"]["dataset_candle_book"] = "bq2_dataset_candle_book_m15_v1"
    return cfg


def connect(cfg: Dict[str, Any]) -> Database:
    client = MongoClient(cfg["mongo_uri"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[cfg["database"]]


def check_dependencies():
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
        from sklearn.preprocessing import LabelEncoder
    except Exception as exc:
        raise RuntimeError("Missing dependencies. Install them: pip install numpy scikit-learn joblib") from exc

    return {
        "joblib": joblib,
        "np": np,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "log_loss": log_loss,
        "LabelEncoder": LabelEncoder,
    }


def parse_dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v.replace(tzinfo=None)
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=None)
    raise ValueError(f"Unsupported datetime: {v!r}")


def sf(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def load_latest_feature_names() -> Optional[List[str]]:
    exports = sorted((project_root() / "04_dataset" / "exports").glob("candle_book_m15_v1_feature_names_*.json"))
    if not exports:
        return None
    data = read_json(exports[-1])
    if not data:
        return None
    names = data.get("feature_names")
    if isinstance(names, list) and names:
        return [str(x) for x in names]
    return None


def load_dataset(
    db: Database,
    collection: str,
    split_name: str,
    limit: Optional[int],
    np_mod: Any,
) -> Tuple[Any, List[str], List[datetime], List[float]]:
    query = {"dataset_version": DATASET_VERSION, "split": split_name}
    projection = {"_id": 0, "x": 1, "y.book_direction": 1, "anchor_time": 1, "entry_price": 1}

    cursor = (
        db[collection]
        .find(query, projection=projection)
        .sort("anchor_time", ASCENDING)
        .batch_size(10000)
    )

    X: List[List[float]] = []
    y: List[str] = []
    times: List[datetime] = []
    prices: List[float] = []

    for doc in cursor:
        if limit is not None and len(y) >= limit:
            break
        x = doc.get("x")
        yd = doc.get("y", {}).get("book_direction")
        if not isinstance(x, list) or yd is None:
            continue
        X.append([sf(v) for v in x])
        y.append(str(yd))
        times.append(parse_dt(doc["anchor_time"]))
        prices.append(sf(doc.get("entry_price", 0.0)))

    if X:
        X_arr = np_mod.asarray(X, dtype=np_mod.float32)
    else:
        X_arr = np_mod.empty((0, 0), dtype=np_mod.float32)
    return X_arr, y, times, prices


def balanced_sample_weight(y_encoded: Any, np_mod: Any) -> Any:
    values, counts = np_mod.unique(y_encoded, return_counts=True)
    total = len(y_encoded)
    n_classes = len(values)
    weights = {int(v): total / (n_classes * int(c)) for v, c in zip(values, counts)}
    return np_mod.asarray([weights[int(v)] for v in y_encoded], dtype=np_mod.float32)


def probability_summary(y_prob: Any, np_mod: Any) -> Dict[str, Any]:
    if y_prob is None or len(y_prob) == 0:
        return {"avg_confidence": None, "p60_rate": None, "p70_rate": None, "p80_rate": None}
    maxp = np_mod.max(y_prob, axis=1)
    return {
        "avg_confidence": round(float(np_mod.mean(maxp)), 6),
        "p60_rate": round(float(np_mod.mean(maxp >= 0.60)), 6),
        "p70_rate": round(float(np_mod.mean(maxp >= 0.70)), 6),
        "p80_rate": round(float(np_mod.mean(maxp >= 0.80)), 6),
    }


def evaluate_split(
    split_name: str,
    model: Any,
    encoder: Any,
    X: Any,
    y_text: List[str],
    deps: Dict[str, Any],
) -> Dict[str, Any]:
    np_mod = deps["np"]
    accuracy_score = deps["accuracy_score"]
    f1_score = deps["f1_score"]
    classification_report = deps["classification_report"]
    confusion_matrix = deps["confusion_matrix"]
    log_loss_fn = deps["log_loss"]

    if len(y_text) == 0:
        return {"rows": 0}

    y_true = encoder.transform(y_text)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    labels = list(range(len(encoder.classes_)))
    classes = [str(c) for c in encoder.classes_]

    metrics: Dict[str, Any] = {
        "rows": int(len(y_text)),
        "label_distribution": dict(Counter(y_text)),
        "prediction_distribution": dict(Counter([str(x) for x in encoder.inverse_transform(y_pred)])),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)), 6),
        "classes": classes,
        "confidence": probability_summary(y_prob, np_mod),
    }

    if y_prob is not None:
        try:
            metrics["log_loss"] = round(float(log_loss_fn(y_true, y_prob, labels=labels)), 6)
        except Exception as exc:
            metrics["log_loss_error"] = str(exc)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report_dict
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return metrics


def save_reports(report: Dict[str, Any], rs: str) -> None:
    jp = reports_dir() / f"candle_book_train_m15_v1_report_{rs}.json"
    tp = reports_dir() / f"candle_book_train_m15_v1_report_{rs}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)

    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "Book & Quality v2 - Train Candle Book Model M15 v1 Report",
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

    lines += ["", "Training Config", "-" * 74]
    for k, v in report["training_config"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Counts", "-" * 74]
    for k, v in report["counts"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Model Output", "-" * 74]
    for k, v in report["model_output"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Metrics", "-" * 74]
    for split, metrics in report.get("metrics", {}).items():
        lines.append(f"{split}:")
        for key in ["rows", "accuracy", "macro_f1", "weighted_f1", "log_loss"]:
            if key in metrics:
                lines.append(f"  {key:<31}: {metrics[key]}")
        if "confidence" in metrics:
            for ck, cv in metrics["confidence"].items():
                lines.append(f"  confidence.{ck:<20}: {cv}")
        if "label_distribution" in metrics:
            lines.append(f"  label_distribution          : {metrics['label_distribution']}")
        if "prediction_distribution" in metrics:
            lines.append(f"  prediction_distribution     : {metrics['prediction_distribution']}")

    if report["errors"]:
        lines += ["", "Errors", "-" * 74]
        lines += [f"- {e}" for e in report["errors"][:50]]

    tp.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Candle Book Model M15 v1.")
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-sample-weight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    rs = stamp(started)

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"candle_book_train_m15_v1_{rs}",
        "status": "running",
        "database": None,
        "symbol": None,
        "start_time": started.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "collections": {},
        "training_config": {
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "target": TARGET,
            "classifier": "HistGradientBoostingClassifier",
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization,
            "max_leaf_nodes": args.max_leaf_nodes,
            "limit_per_split": args.limit_per_split,
            "sample_weight": not args.no_sample_weight,
        },
        "counts": {},
        "model_output": {},
        "metrics": {},
        "errors": [],
        "args": vars(args),
    }

    print("=== Book & Quality v2 - Train Candle Book Model M15 v1 ===", flush=True)

    try:
        deps = check_dependencies()
        np_mod = deps["np"]
        LabelEncoder = deps["LabelEncoder"]
        HistGradientBoostingClassifier = deps["HistGradientBoostingClassifier"]

        cfg = load_config()
        db = connect(cfg)
        dataset_coll = cfg["bq2_collections"]["dataset_candle_book"]
        report["database"] = cfg["database"]
        report["symbol"] = cfg["symbol"]
        report["collections"] = {"dataset_source": dataset_coll}

        print("[LOAD] Loading train/valid/test dataset...", flush=True)
        X_train, y_train, t_train, p_train = load_dataset(db, dataset_coll, "train", args.limit_per_split, np_mod)
        X_valid, y_valid, t_valid, p_valid = load_dataset(db, dataset_coll, "valid", args.limit_per_split, np_mod)
        X_test, y_test, t_test, p_test = load_dataset(db, dataset_coll, "test", args.limit_per_split, np_mod)

        report["counts"] = {
            "train": len(y_train),
            "valid": len(y_valid),
            "test": len(y_test),
            "rows_loaded": len(y_train) + len(y_valid) + len(y_test),
            "feature_count": int(X_train.shape[1]) if len(y_train) else None,
            "train_label_distribution": dict(Counter(y_train)),
            "valid_label_distribution": dict(Counter(y_valid)),
            "test_label_distribution": dict(Counter(y_test)),
        }

        if len(y_train) == 0 or len(y_valid) == 0 or len(y_test) == 0:
            raise RuntimeError("One or more splits are empty. Cannot train safely.")
        if X_train.shape[1] != X_valid.shape[1] or X_train.shape[1] != X_test.shape[1]:
            raise RuntimeError("Feature count mismatch between splits.")

        feature_names = load_latest_feature_names()
        if not feature_names:
            feature_names = [f"f_{i:03d}" for i in range(int(X_train.shape[1]))]
        if len(feature_names) != int(X_train.shape[1]):
            raise RuntimeError(f"feature_names length mismatch: {len(feature_names)} vs {X_train.shape[1]}")

        encoder = LabelEncoder()
        encoder.fit(y_train + y_valid + y_test)
        y_train_enc = encoder.transform(y_train)

        sample_weight = None
        if not args.no_sample_weight:
            sample_weight = balanced_sample_weight(y_train_enc, np_mod)

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
            max_leaf_nodes=args.max_leaf_nodes,
            random_state=args.random_state,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=20,
        )

        print("[TRAIN] Fitting model...", flush=True)
        model.fit(X_train, y_train_enc, sample_weight=sample_weight)

        print("[EVAL] Evaluating splits...", flush=True)
        report["metrics"]["train"] = evaluate_split("train", model, encoder, X_train, y_train, deps)
        report["metrics"]["valid"] = evaluate_split("valid", model, encoder, X_valid, y_valid, deps)
        report["metrics"]["test"] = evaluate_split("test", model, encoder, X_test, y_test, deps)

        bundle = {
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "target": TARGET,
            "symbol": cfg["symbol"],
            "anchor_timeframe": "M15",
            "input_timeframes": ["M1", "M5", "M15"],
            "trained_at": utc_now().isoformat(),
            "model": model,
            "encoder": encoder,
            "classes": [str(c) for c in encoder.classes_],
            "feature_count": int(X_train.shape[1]),
            "feature_names": feature_names,
            "training_config": report["training_config"],
            "metrics": report["metrics"],
        }

        out_path = models_dir() / f"{MODEL_VERSION}_{rs}.joblib"
        latest_path = models_dir() / f"{MODEL_VERSION}_latest.joblib"
        deps["joblib"].dump(bundle, out_path, compress=3)
        shutil.copy2(out_path, latest_path)

        report["model_output"] = {
            "model_path": str(out_path),
            "latest_model_path": str(latest_path),
            "classes": bundle["classes"],
            "feature_count": bundle["feature_count"],
        }
        report["status"] = "success"

    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc))
        print(f"[ERROR] {exc}", flush=True)

    finally:
        ended = utc_now()
        report["end_time"] = ended.isoformat()
        report["duration_seconds"] = round((ended - started).total_seconds(), 3)
        save_reports(report, rs)

    print("\n=== Final Summary ===", flush=True)
    print(f"Status       : {report['status']}", flush=True)
    print(f"Rows Loaded  : {report.get('counts', {}).get('rows_loaded')}", flush=True)
    print(f"Train        : {report.get('counts', {}).get('train')}", flush=True)
    print(f"Valid        : {report.get('counts', {}).get('valid')}", flush=True)
    print(f"Test         : {report.get('counts', {}).get('test')}", flush=True)
    for split in ["valid", "test"]:
        m = report.get("metrics", {}).get(split, {})
        if m:
            print(f"{split} acc/f1  : {m.get('accuracy')} / {m.get('macro_f1')}", flush=True)
    print(f"Model Path   : {report.get('model_output', {}).get('model_path')}", flush=True)
    print(f"TXT Report   : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] == "success" else "[FAILED]", flush=True)

    if report["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
