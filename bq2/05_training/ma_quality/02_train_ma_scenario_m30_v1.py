# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Train MA Scenario Model M30 v1

Location:
    Book_Quality/05_training/ma_quality/02_train_ma_scenario_m30_v1.py

Purpose:
    Train the corrected MA Quality scenario model.

Architecture:
    MA Scenario Model predicts:
        - predicted_direction
        - m30_future_direction
        - h1_future_direction
        - h4_future_direction
        - m30_phase
        - h1_phase
        - h4_phase

    The model does NOT predict lot size, TP/SL, trailing, exit, or risk mode.
    API Decision Engine will decide operational trade management from:
        model probabilities + Candle Book confirmation + phase/rule matrix.

Input:
    bq2_dataset_ma_scenario_m30_v1

Output:
    06_models/ma_quality/ma_scenario_model_m30_v1_<timestamp>.joblib
    06_models/ma_quality/ma_scenario_model_m30_v1_latest.joblib
    05_training/reports/ma_scenario_train_m30_v1_report_<timestamp>.txt/json

TEST WITH SPLITS:
    cd C:/Project/BookQuality/bq2/05_training/ma_quality
    python -u 02_train_ma_scenario_m30_v1.py --limit-per-split 5000 --max-iter 50

FULL TRAIN:
    python -u 02_train_ma_scenario_m30_v1.py --max-iter 200
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymongo import MongoClient, ASCENDING
from pymongo.database import Database


SCRIPT_NAME = "02_train_ma_scenario_m30_v1.py"
MODEL_VERSION = "ma_scenario_model_m30_v1"
DATASET_VERSION = "ma_scenario_dataset_m30_v1"
FEATURE_VERSION = "ma_quality_features_m30_v1"
LABEL_VERSION = "ma_quality_label_m30_v1"

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "bq2_collections": {
        "dataset_ma_scenario": "bq2_dataset_ma_scenario_m30_v1",
    },
}

SCENARIO_TARGETS = [
    "predicted_direction",
    "m30_future_direction",
    "h1_future_direction",
    "h4_future_direction",
    "m30_phase",
    "h1_phase",
    "h4_phase",
]


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
    p = project_root() / "06_models" / "ma_quality"
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
    cfg["bq2_collections"]["dataset_ma_scenario"] = "bq2_dataset_ma_scenario_m30_v1"
    return cfg


def connect(cfg: Dict[str, Any]) -> Database:
    client = MongoClient(cfg["mongo_uri"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[cfg["database"]]


def sf(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def latest_feature_names_export() -> Optional[Dict[str, Any]]:
    export_dir = project_root() / "04_dataset" / "exports"
    if not export_dir.exists():
        return None
    files = sorted(export_dir.glob("ma_scenario_m30_v1_feature_names_*.json"))
    if not files:
        return None
    return read_json(files[-1])


def check_dependencies():
    try:
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
        from sklearn.preprocessing import LabelEncoder
    except Exception as exc:
        raise RuntimeError(
            "Missing Python ML dependencies. Install them in your venv: "
            "pip install scikit-learn joblib"
        ) from exc

    return {
        "joblib": joblib,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "LabelEncoder": LabelEncoder,
    }


def load_dataset(
    db: Database,
    coll_name: str,
    limit: Optional[int],
    limit_per_split: Optional[int],
    report: Dict[str, Any],
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
    query = {
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "label_version": LABEL_VERSION,
    }

    projection = {
        "_id": 0,
        "anchor_time": 1,
        "split": 1,
        "x": 1,
        "feature_count": 1,
        "y": 1,
    }

    x_rows: List[List[float]] = []
    y_rows: List[Dict[str, Any]] = []
    splits: List[str] = []
    per_split_counter = {"train": 0, "valid": 0, "test": 0}
    expected_feature_count: Optional[int] = None

    cursor = (
        db[coll_name]
        .find(query, projection=projection)
        .sort("anchor_time", ASCENDING)
        .batch_size(5000)
    )

    for doc in cursor:
        if limit is not None and len(x_rows) >= limit:
            break

        split = doc.get("split")
        if split not in {"train", "valid", "test"}:
            report["counts"]["skipped_bad_split"] += 1
            continue

        if limit_per_split is not None and per_split_counter[split] >= limit_per_split:
            continue

        x = doc.get("x")
        y = doc.get("y")

        if not isinstance(x, list) or not x:
            report["counts"]["skipped_bad_x"] += 1
            continue
        if not isinstance(y, dict) or not y:
            report["counts"]["skipped_bad_y"] += 1
            continue
        if any(y.get(target) is None for target in SCENARIO_TARGETS):
            report["counts"]["skipped_missing_target"] += 1
            continue

        if expected_feature_count is None:
            expected_feature_count = len(x)
        elif len(x) != expected_feature_count:
            report["counts"]["skipped_feature_count_mismatch"] += 1
            continue

        x_rows.append([sf(v) for v in x])
        y_rows.append(y)
        splits.append(split)
        per_split_counter[split] += 1
        report["split_distribution_loaded"][split] = report["split_distribution_loaded"].get(split, 0) + 1

    if not x_rows:
        raise RuntimeError("No dataset rows loaded.")

    X = np.asarray(x_rows, dtype=np.float32)
    report["counts"]["rows_loaded"] = int(X.shape[0])
    report["feature_count"] = int(X.shape[1])
    return X, y_rows, splits


def split_indices(splits: List[str]) -> Dict[str, np.ndarray]:
    arr = np.asarray(splits)
    return {
        "train": np.where(arr == "train")[0],
        "valid": np.where(arr == "valid")[0],
        "test": np.where(arr == "test")[0],
    }


def train_classifier(
    target: str,
    X: np.ndarray,
    y_rows: List[Dict[str, Any]],
    idx: Dict[str, np.ndarray],
    deps: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    LabelEncoder = deps["LabelEncoder"]
    Model = deps["HistGradientBoostingClassifier"]
    accuracy_score = deps["accuracy_score"]
    f1_score = deps["f1_score"]
    classification_report = deps["classification_report"]
    confusion_matrix = deps["confusion_matrix"]

    y_raw = [str(row.get(target)) for row in y_rows]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    model = Model(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        random_state=args.random_state,
    )

    train_idx = idx["train"]
    if len(train_idx) == 0:
        raise RuntimeError(f"No train rows for target={target}")

    model.fit(X[train_idx], y[train_idx])

    metrics: Dict[str, Any] = {
        "target": target,
        "type": "classification",
        "classes": list(encoder.classes_),
        "split_metrics": {},
    }

    for split_name in ["train", "valid", "test"]:
        sidx = idx[split_name]
        if len(sidx) == 0:
            continue

        pred = model.predict(X[sidx])
        metrics["split_metrics"][split_name] = {
            "rows": int(len(sidx)),
            "accuracy": float(accuracy_score(y[sidx], pred)),
            "macro_f1": float(f1_score(y[sidx], pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y[sidx], pred, average="weighted", zero_division=0)),
            "classification_report": classification_report(
                y[sidx],
                pred,
                target_names=list(encoder.classes_),
                zero_division=0,
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y[sidx], pred).tolist(),
        }

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X[sidx])
            max_proba = np.max(proba, axis=1)
            metrics["split_metrics"][split_name]["avg_max_probability"] = float(np.mean(max_proba))
            metrics["split_metrics"][split_name]["p70_rate"] = float(np.mean(max_proba >= 0.70))
            metrics["split_metrics"][split_name]["p80_rate"] = float(np.mean(max_proba >= 0.80))

    return {"model": model, "encoder": encoder, "metrics": metrics}


def build_text_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Book & Quality v2 - Train MA Scenario Model M30 v1 Report")
    lines.append("=" * 74)

    for key in ["run_id", "status", "database", "symbol", "start_time", "end_time", "duration_seconds"]:
        lines.append(f"{key:<34}: {report.get(key)}")

    lines += ["", "Training Config", "-" * 74]
    for k, v in report["training_config"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Counts", "-" * 74]
    for k, v in report["counts"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Split Distribution Loaded", "-" * 74]
    for k, v in report["split_distribution_loaded"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Metrics Summary", "-" * 74]
    for target, metric in report["metrics"].items():
        lines.append(f"{target} (classification):")
        for split, sm in metric.get("split_metrics", {}).items():
            lines.append(
                f"  {split:<8} rows={sm.get('rows')} "
                f"acc={sm.get('accuracy'):.4f} "
                f"macro_f1={sm.get('macro_f1'):.4f} "
                f"weighted_f1={sm.get('weighted_f1'):.4f} "
                f"avg_p={sm.get('avg_max_probability', 0):.4f} "
                f"p70_rate={sm.get('p70_rate', 0):.4f} "
                f"p80_rate={sm.get('p80_rate', 0):.4f}"
            )

    lines += ["", "Outputs", "-" * 74]
    for k, v in report["outputs"].items():
        lines.append(f"{k:<34}: {v}")

    if report["warnings"]:
        lines += ["", "Warnings", "-" * 74]
        lines += [f"- {w}" for w in report["warnings"]]

    if report["errors"]:
        lines += ["", "Errors", "-" * 74]
        lines += [f"- {e}" for e in report["errors"]]

    return "\n".join(lines)


def save_report(report: Dict[str, Any], rs: str) -> None:
    jp = reports_dir() / f"ma_scenario_train_m30_v1_report_{rs}.json"
    tp = reports_dir() / f"ma_scenario_train_m30_v1_report_{rs}.txt"
    report["outputs"]["report_json"] = str(jp)
    report["outputs"]["report_txt"] = str(tp)

    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tp.write_text(build_text_report(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MA Scenario Model M30 v1.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    rs = stamp(started)

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"train_ma_scenario_m30_v1_{rs}",
        "status": "running",
        "database": None,
        "symbol": None,
        "start_time": started.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "training_config": {
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "algorithm": "sklearn.HistGradientBoostingClassifier",
            "targets": SCENARIO_TARGETS,
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "l2_regularization": args.l2_regularization,
            "random_state": args.random_state,
            "limit": args.limit,
            "limit_per_split": args.limit_per_split,
        },
        "counts": {
            "rows_loaded": 0,
            "skipped_bad_x": 0,
            "skipped_bad_y": 0,
            "skipped_bad_split": 0,
            "skipped_missing_target": 0,
            "skipped_feature_count_mismatch": 0,
            "models_trained": 0,
        },
        "feature_count": None,
        "feature_names_source": None,
        "split_distribution_loaded": {},
        "targets_trained": [],
        "metrics": {},
        "outputs": {},
        "warnings": [],
        "errors": [],
    }

    print("=== Book & Quality v2 - Train MA Scenario Model M30 v1 ===", flush=True)

    try:
        deps = check_dependencies()
        cfg = load_config()
        db = connect(cfg)

        report["database"] = cfg["database"]
        report["symbol"] = cfg["symbol"]

        coll_name = cfg["bq2_collections"]["dataset_ma_scenario"]
        print(f"Database : {cfg['database']}", flush=True)
        print(f"Source   : {coll_name}", flush=True)
        print(f"Targets  : {SCENARIO_TARGETS}", flush=True)

        X, y_rows, splits = load_dataset(db, coll_name, args.limit, args.limit_per_split, report)
        idx = split_indices(splits)

        feature_names_export = latest_feature_names_export()
        if feature_names_export:
            report["feature_names_source"] = feature_names_export.get("created_at")
            feature_names = feature_names_export.get("feature_names", [])
        else:
            feature_names = []
            report["warnings"].append("Feature names export not found. Bundle will contain feature_count only.")

        if len(idx["valid"]) == 0 or len(idx["test"]) == 0:
            report["warnings"].append("Validation or test split is empty. Use --limit-per-split or full dataset for reliable evaluation.")

        bundle = {
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "trained_at": utc_now().isoformat(),
            "symbol": cfg["symbol"],
            "algorithm": "HistGradientBoostingClassifier",
            "feature_count": int(X.shape[1]),
            "feature_names": feature_names,
            "targets": SCENARIO_TARGETS,
            "models": {},
            "encoders": {},
            "metrics": {},
            "training_config": report["training_config"],
        }

        for target in SCENARIO_TARGETS:
            print(f"[TRAIN] classifier: {target}", flush=True)
            result = train_classifier(target, X, y_rows, idx, deps, args)
            bundle["models"][target] = result["model"]
            bundle["encoders"][target] = result["encoder"]
            bundle["metrics"][target] = result["metrics"]
            report["metrics"][target] = result["metrics"]
            report["targets_trained"].append(target)
            report["counts"]["models_trained"] += 1

        model_path = models_dir() / f"{MODEL_VERSION}_{rs}.joblib"
        latest_path = models_dir() / f"{MODEL_VERSION}_latest.joblib"

        deps["joblib"].dump(bundle, model_path)
        shutil.copyfile(model_path, latest_path)

        report["outputs"]["model_path"] = str(model_path)
        report["outputs"]["latest_model_path"] = str(latest_path)
        report["status"] = "success"

    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc))
        print(f"[ERROR] {exc}", flush=True)

    finally:
        ended = utc_now()
        report["end_time"] = ended.isoformat()
        report["duration_seconds"] = round((ended - started).total_seconds(), 3)
        save_report(report, rs)

    print("\n=== Final Summary ===", flush=True)
    print(f"Status        : {report['status']}", flush=True)
    print(f"Rows Loaded   : {report['counts']['rows_loaded']:,}", flush=True)
    print(f"Models Trained: {report['counts']['models_trained']}", flush=True)
    print(f"Model Path    : {report['outputs'].get('model_path')}", flush=True)
    print(f"Report TXT    : {report['outputs'].get('report_txt')}", flush=True)
    print("[DONE]" if report["status"] == "success" else "[FAILED]", flush=True)

    if report["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
