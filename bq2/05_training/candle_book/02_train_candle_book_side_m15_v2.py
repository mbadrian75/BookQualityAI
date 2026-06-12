# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Train Candle Book Side Model M15 v2

Location:
    Book_Quality/05_training/candle_book/02_train_candle_book_side_m15_v2.py

Purpose:
    Train side-specific Candle Book confirmation models.

Input:
    bq2_dataset_candle_book_side_m15_v2

Outputs:
    06_models/candle_book/candle_book_side_model_m15_v2_<timestamp>.joblib
    06_models/candle_book/candle_book_side_model_m15_v2_latest.joblib

Targets:
    buy_outcome  = WIN / LOSS / FLAT / AMBIGUOUS
    sell_outcome = WIN / LOSS / FLAT / AMBIGUOUS

Model role:
    MA Scenario decides candidate direction.
    Candle Book Side model estimates whether that side has good immediate entry timing.
    API uses p_buy_win / p_sell_win and related probabilities.

TEST:
    cd C:/Project/BookQuality/bq2/05_training/candle_book
    python -u 02_train_candle_book_side_m15_v2.py --limit-per-split 5000 --max-iter 50

FULL:
    python -u 02_train_candle_book_side_m15_v2.py --max-iter 200 --l2-regularization 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING
from pymongo.database import Database


SCRIPT_NAME = "02_train_candle_book_side_m15_v2.py"
MODEL_VERSION = "candle_book_side_model_m15_v2"
DATASET_VERSION = "candle_book_side_dataset_m15_v2"
FEATURE_VERSION = "candle_book_side_features_m15_v2"
LABEL_VERSION = "candle_book_side_label_m15_v2"
TARGETS = ["buy_outcome", "sell_outcome"]

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "bq2_collections": {
        "dataset_candle_book_side": "bq2_dataset_candle_book_side_m15_v2",
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
    cfg["bq2_collections"]["dataset_candle_book_side"] = "bq2_dataset_candle_book_side_m15_v2"
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
        from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report, confusion_matrix
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies. Install them: pip install scikit-learn joblib numpy"
        ) from exc

    return {
        "joblib": joblib,
        "np": np,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "log_loss": log_loss,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "LabelEncoder": LabelEncoder,
        "compute_sample_weight": compute_sample_weight,
    }


def sf(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def load_latest_feature_names() -> List[str]:
    files = sorted(exports_dir().glob("candle_book_side_m15_v2_feature_names_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return []
    data = read_json(files[-1]) or {}
    names = data.get("feature_names") or []
    return [str(x) for x in names]


def load_split(db: Database, coll_name: str, split: str, limit: Optional[int], drop_ambiguous: bool) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {"dataset_version": DATASET_VERSION, "split": split}
    cursor = db[coll_name].find(
        query,
        projection={"_id": 0, "x": 1, "y": 1, "feature_count": 1, "anchor_time": 1},
    ).sort("anchor_time", ASCENDING)
    if limit:
        cursor = cursor.limit(limit)

    rows: List[Dict[str, Any]] = []
    for doc in cursor:
        y = doc.get("y", {})
        if drop_ambiguous and (y.get("buy_outcome") == "AMBIGUOUS" or y.get("sell_outcome") == "AMBIGUOUS"):
            continue
        rows.append(doc)
    return rows


def rows_to_arrays(rows: List[Dict[str, Any]], target: str, np_mod: Any) -> Tuple[Any, List[str]]:
    X = np_mod.asarray([[sf(v) for v in row["x"]] for row in rows], dtype=np_mod.float32)
    y = [str(row["y"][target]) for row in rows]
    return X, y


def confidence_summary(proba: Any, np_mod: Any) -> Dict[str, Any]:
    if proba is None or len(proba) == 0:
        return {"avg_confidence": None, "p60_rate": None, "p70_rate": None, "p80_rate": None}
    conf = np_mod.max(proba, axis=1)
    return {
        "avg_confidence": round(float(np_mod.mean(conf)), 6),
        "p60_rate": round(float(np_mod.mean(conf >= 0.60)), 6),
        "p70_rate": round(float(np_mod.mean(conf >= 0.70)), 6),
        "p80_rate": round(float(np_mod.mean(conf >= 0.80)), 6),
    }


def safe_log_loss(y_true_encoded: Any, proba: Any, labels: List[int], log_loss_fn) -> Optional[float]:
    try:
        return round(float(log_loss_fn(y_true_encoded, proba, labels=labels)), 6)
    except Exception:
        return None


def evaluate(model: Any, encoder: Any, X: Any, y_labels: List[str], deps: Dict[str, Any]) -> Dict[str, Any]:
    np_mod = deps["np"]
    y_true = encoder.transform(y_labels)
    y_pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    labels_int = list(range(len(encoder.classes_)))

    pred_labels = [str(x) for x in encoder.inverse_transform(y_pred)]
    metrics = {
        "rows": int(len(y_labels)),
        "accuracy": round(float(deps["accuracy_score"](y_true, y_pred)), 6),
        "macro_f1": round(float(deps["f1_score"](y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(deps["f1_score"](y_true, y_pred, average="weighted", zero_division=0)), 6),
        "log_loss": safe_log_loss(y_true, proba, labels_int, deps["log_loss"]),
        "confidence": confidence_summary(proba, np_mod),
        "label_distribution": dict(Counter(y_labels)),
        "prediction_distribution": dict(Counter(pred_labels)),
        "classes": [str(x) for x in encoder.classes_],
    }

    try:
        metrics["classification_report"] = deps["classification_report"](
            y_true, y_pred, target_names=[str(x) for x in encoder.classes_], zero_division=0, output_dict=True
        )
        metrics["confusion_matrix"] = deps["confusion_matrix"](y_true, y_pred, labels=labels_int).tolist()
    except Exception as exc:
        metrics["report_error"] = str(exc)

    if proba is not None:
        target_probs = {}
        for class_index, class_name in enumerate(encoder.classes_):
            target_probs[f"avg_p_{class_name}"] = round(float(np_mod.mean(proba[:, class_index])), 6)
        metrics["avg_class_probabilities"] = target_probs

    return metrics


def save_reports(report: Dict[str, Any], rs: str) -> None:
    jp = reports_dir() / f"candle_book_side_train_m15_v2_report_{rs}.json"
    tp = reports_dir() / f"candle_book_side_train_m15_v2_report_{rs}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "Book & Quality v2 - Train Candle Book Side Model M15 v2 Report",
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
    for target, split_metrics in report.get("metrics", {}).items():
        lines.append(f"Target: {target}")
        for split, m in split_metrics.items():
            lines.append(f"  {split}:")
            for key in ["rows", "accuracy", "macro_f1", "weighted_f1", "log_loss"]:
                lines.append(f"    {key:<29}: {m.get(key)}")
            conf = m.get("confidence", {})
            for ck, cv in conf.items():
                lines.append(f"    confidence.{ck:<18}: {cv}")
            lines.append(f"    label_distribution          : {m.get('label_distribution')}")
            lines.append(f"    prediction_distribution     : {m.get('prediction_distribution')}")
        lines.append("")

    if report.get("errors"):
        lines += ["", "Errors", "-" * 74]
        lines += [f"- {e}" for e in report["errors"][:50]]

    tp.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Candle Book Side Model M15 v2.")
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--no-sample-weight", action="store_true")
    parser.add_argument("--drop-ambiguous", action="store_true", help="Exclude rows where either side outcome is AMBIGUOUS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    rs = stamp(started)

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"candle_book_side_train_m15_v2_{rs}",
        "status": "running",
        "database": None,
        "symbol": None,
        "start_time": started.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "collections": {},
        "training_config": {},
        "counts": {},
        "model_output": {},
        "metrics": {},
        "errors": [],
        "args": vars(args),
    }

    print("=== Book & Quality v2 - Train Candle Book Side Model M15 v2 ===", flush=True)

    try:
        deps = check_dependencies()
        np_mod = deps["np"]

        cfg = load_config()
        db = connect(cfg)
        coll = cfg["bq2_collections"]["dataset_candle_book_side"]

        report["database"] = cfg["database"]
        report["symbol"] = cfg["symbol"]
        report["collections"] = {"dataset_source": coll}
        report["training_config"] = {
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "targets": TARGETS,
            "classifier": "HistGradientBoostingClassifier",
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization,
            "max_leaf_nodes": args.max_leaf_nodes,
            "limit_per_split": args.limit_per_split,
            "sample_weight": not args.no_sample_weight,
            "drop_ambiguous": args.drop_ambiguous,
        }

        rows_by_split = {
            "train": load_split(db, coll, "train", args.limit_per_split, args.drop_ambiguous),
            "valid": load_split(db, coll, "valid", args.limit_per_split, args.drop_ambiguous),
            "test": load_split(db, coll, "test", args.limit_per_split, args.drop_ambiguous),
        }
        if not rows_by_split["train"] or not rows_by_split["valid"] or not rows_by_split["test"]:
            raise RuntimeError("Missing one or more splits. Cannot train.")

        feature_count = int(rows_by_split["train"][0].get("feature_count", len(rows_by_split["train"][0]["x"])))
        feature_names = load_latest_feature_names()
        if not feature_names or len(feature_names) != feature_count:
            feature_names = [f"f_{i:03d}" for i in range(feature_count)]

        report["counts"] = {
            "train": len(rows_by_split["train"]),
            "valid": len(rows_by_split["valid"]),
            "test": len(rows_by_split["test"]),
            "rows_loaded": sum(len(v) for v in rows_by_split.values()),
            "feature_count": feature_count,
        }
        for split, rows in rows_by_split.items():
            for target in TARGETS:
                report["counts"][f"{split}_{target}_distribution"] = dict(Counter(str(r["y"][target]) for r in rows))

        models: Dict[str, Any] = {}
        encoders: Dict[str, Any] = {}
        metrics: Dict[str, Dict[str, Any]] = {}

        for target in TARGETS:
            print(f"[TRAIN] target={target}", flush=True)
            X_train, y_train_labels = rows_to_arrays(rows_by_split["train"], target, np_mod)
            X_valid, y_valid_labels = rows_to_arrays(rows_by_split["valid"], target, np_mod)
            X_test, y_test_labels = rows_to_arrays(rows_by_split["test"], target, np_mod)

            encoder = deps["LabelEncoder"]()
            y_train = encoder.fit_transform(y_train_labels)

            model = deps["HistGradientBoostingClassifier"](
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                l2_regularization=args.l2_regularization,
                max_leaf_nodes=args.max_leaf_nodes,
                random_state=42,
            )

            fit_kwargs = {}
            if not args.no_sample_weight:
                fit_kwargs["sample_weight"] = deps["compute_sample_weight"]("balanced", y_train)

            model.fit(X_train, y_train, **fit_kwargs)
            models[target] = model
            encoders[target] = encoder

            metrics[target] = {
                "train": evaluate(model, encoder, X_train, y_train_labels, deps),
                "valid": evaluate(model, encoder, X_valid, y_valid_labels, deps),
                "test": evaluate(model, encoder, X_test, y_test_labels, deps),
            }

        report["metrics"] = metrics

        model_path = models_dir() / f"{MODEL_VERSION}_{rs}.joblib"
        latest_path = models_dir() / f"{MODEL_VERSION}_latest.joblib"
        bundle = {
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "symbol": cfg["symbol"],
            "created_at": utc_now().isoformat(),
            "trained_at": utc_now().isoformat(),
            "targets": TARGETS,
            "feature_count": feature_count,
            "feature_names": feature_names,
            "models": models,
            "encoders": encoders,
            "training_config": report["training_config"],
            "metrics_summary": {
                target: {
                    split: {k: v for k, v in metrics[target][split].items() if k in ["rows", "accuracy", "macro_f1", "weighted_f1", "log_loss", "confidence", "label_distribution", "prediction_distribution"]}
                    for split in ["train", "valid", "test"]
                }
                for target in TARGETS
            },
        }
        deps["joblib"].dump(bundle, model_path)
        shutil.copyfile(model_path, latest_path)

        report["model_output"] = {
            "model_path": str(model_path),
            "latest_model_path": str(latest_path),
            "targets": TARGETS,
            "feature_count": feature_count,
            "classes": {target: [str(x) for x in encoders[target].classes_] for target in TARGETS},
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
    print(f"Status      : {report['status']}", flush=True)
    print(f"Rows Loaded : {report.get('counts', {}).get('rows_loaded')}", flush=True)
    print(f"Feature Cnt : {report.get('counts', {}).get('feature_count')}", flush=True)
    print(f"Model Path  : {report.get('model_output', {}).get('model_path')}", flush=True)
    print(f"TXT Report  : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
