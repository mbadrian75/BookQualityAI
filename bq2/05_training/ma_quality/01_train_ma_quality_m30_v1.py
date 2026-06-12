# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Train MA Quality Model M30 v1

Location:
    Book_Quality/05_training/ma_quality/01_train_ma_quality_m30_v1.py

Input:
    bq2_dataset_ma_quality_m30_v1

Output:
    06_models/ma_quality/ma_quality_model_m30_v1_<target_set>_<timestamp>.joblib
    05_training/reports/ma_quality_train_m30_v1_report_<timestamp>.txt/json

TEST ONLY:
    cd C:/Project/Book_Quality/05_training/ma_quality
    python -u 01_train_ma_quality_m30_v1.py --limit 10000 --target-set core --max-iter 50

FULL CORE TRAIN:
    python -u 01_train_ma_quality_m30_v1.py --target-set core --max-iter 200

FULL MANAGEMENT TRAIN:
    python -u 01_train_ma_quality_m30_v1.py --target-set full --max-iter 200
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

SCRIPT_NAME = "01_train_ma_quality_m30_v1.py"
MODEL_VERSION = "ma_quality_model_m30_v1"
DATASET_VERSION = "ma_quality_dataset_m30_v1"
FEATURE_VERSION = "ma_quality_features_m30_v1"
LABEL_VERSION = "ma_quality_label_m30_v1"

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "bq2_collections": {"dataset_ma_quality": "bq2_dataset_ma_quality_m30_v1"},
}

CORE_CLASSIFICATION_TARGETS = [
    "predicted_direction",
    "buy_risk_mode",
    "sell_risk_mode",
    "buy_lot_multiplier",
    "sell_lot_multiplier",
]
CORE_REGRESSION_TARGETS = ["buy_quality_score", "sell_quality_score"]
MANAGEMENT_CLASSIFICATION_TARGETS = [
    "buy_tp_mode", "sell_tp_mode",
    "buy_sl_mode", "sell_sl_mode",
    "buy_trailing_mode", "sell_trailing_mode",
    "buy_exit_mode", "sell_exit_mode",
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
    return json.loads(path.read_text(encoding="utf-8"))


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
    if p.exists():
        cfg = deep_merge(cfg, read_json(p) or {})
    cfg.setdefault("bq2_collections", {})
    cfg["bq2_collections"]["dataset_ma_quality"] = "bq2_dataset_ma_quality_m30_v1"
    return cfg


def sf(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def check_dependencies() -> Dict[str, Any]:
    try:
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
        from sklearn.preprocessing import LabelEncoder
    except Exception as exc:
        raise RuntimeError("Missing dependencies. Run: pip install scikit-learn joblib") from exc

    return {
        "joblib": joblib,
        "HGBClassifier": HistGradientBoostingClassifier,
        "HGBRegressor": HistGradientBoostingRegressor,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "LabelEncoder": LabelEncoder,
    }


def connect(cfg: Dict[str, Any]):
    client = MongoClient(cfg["mongo_uri"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[cfg["database"]]


def load_dataset(db, coll_name: str, limit: Optional[int], report: Dict[str, Any]) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
    query = {"dataset_version": DATASET_VERSION, "feature_version": FEATURE_VERSION, "label_version": LABEL_VERSION}
    projection = {"_id": 0, "x": 1, "y": 1, "split": 1, "anchor_time": 1, "feature_count": 1}
    cur = db[coll_name].find(query, projection=projection).sort("anchor_time", ASCENDING).batch_size(5000)

    xs: List[List[float]] = []
    ys: List[Dict[str, Any]] = []
    splits: List[str] = []
    expected_count = None

    for doc in cur:
        if limit is not None and len(xs) >= limit:
            break
        x = doc.get("x")
        y = doc.get("y")
        split = doc.get("split")
        if not isinstance(x, list) or not x:
            report["counts"]["skipped_bad_x"] += 1
            continue
        if not isinstance(y, dict) or not y:
            report["counts"]["skipped_bad_y"] += 1
            continue
        if split not in {"train", "valid", "test"}:
            report["counts"]["skipped_bad_split"] += 1
            continue
        if expected_count is None:
            expected_count = len(x)
        elif len(x) != expected_count:
            report["counts"]["skipped_feature_count_mismatch"] += 1
            continue
        xs.append([sf(v) for v in x])
        ys.append(y)
        splits.append(split)
        report["split_distribution_loaded"][split] = report["split_distribution_loaded"].get(split, 0) + 1

    if not xs:
        raise RuntimeError("No dataset rows loaded.")
    X = np.asarray(xs, dtype=np.float32)
    report["counts"]["rows_loaded"] = int(X.shape[0])
    report["feature_count"] = int(X.shape[1])
    return X, ys, splits


def split_indices(splits: List[str]) -> Dict[str, np.ndarray]:
    arr = np.asarray(splits)
    return {s: np.where(arr == s)[0] for s in ["train", "valid", "test"]}


def train_classifier(target: str, X: np.ndarray, y_rows: List[Dict[str, Any]], idx: Dict[str, np.ndarray], deps: Dict[str, Any], args):
    enc = deps["LabelEncoder"]()
    y_raw = [str(row.get(target)) for row in y_rows]
    y = enc.fit_transform(y_raw)
    model = deps["HGBClassifier"](
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        random_state=args.random_state,
    )
    if len(idx["train"]) == 0:
        raise RuntimeError(f"No train rows for target={target}")
    model.fit(X[idx["train"]], y[idx["train"]])
    metrics = {"target": target, "type": "classification", "classes": list(enc.classes_), "split_metrics": {}}
    for split in ["train", "valid", "test"]:
        sidx = idx[split]
        if len(sidx) == 0:
            continue
        pred = model.predict(X[sidx])
        metrics["split_metrics"][split] = {
            "rows": int(len(sidx)),
            "accuracy": float(deps["accuracy_score"](y[sidx], pred)),
            "macro_f1": float(deps["f1_score"](y[sidx], pred, average="macro", zero_division=0)),
            "weighted_f1": float(deps["f1_score"](y[sidx], pred, average="weighted", zero_division=0)),
        }
    return model, enc, metrics


def train_regressor(target: str, X: np.ndarray, y_rows: List[Dict[str, Any]], idx: Dict[str, np.ndarray], deps: Dict[str, Any], args):
    y = np.asarray([sf(row.get(target)) for row in y_rows], dtype=np.float32)
    model = deps["HGBRegressor"](
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        random_state=args.random_state,
    )
    if len(idx["train"]) == 0:
        raise RuntimeError(f"No train rows for target={target}")
    model.fit(X[idx["train"]], y[idx["train"]])
    metrics = {"target": target, "type": "regression", "split_metrics": {}}
    for split in ["train", "valid", "test"]:
        sidx = idx[split]
        if len(sidx) == 0:
            continue
        pred = model.predict(X[sidx])
        mse = float(deps["mean_squared_error"](y[sidx], pred))
        metrics["split_metrics"][split] = {
            "rows": int(len(sidx)),
            "mae": float(deps["mean_absolute_error"](y[sidx], pred)),
            "rmse": float(math.sqrt(mse)),
            "r2": float(deps["r2_score"](y[sidx], pred)),
        }
    return model, metrics


def save_report(report: Dict[str, Any], rs: str) -> None:
    jp = reports_dir() / f"ma_quality_train_m30_v1_report_{rs}.json"
    tp = reports_dir() / f"ma_quality_train_m30_v1_report_{rs}.txt"
    report["outputs"]["report_json"] = str(jp)
    report["outputs"]["report_txt"] = str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "Book & Quality v2 - Train MA Quality Model M30 v1 Report",
        "=" * 74,
        f"{'run_id':<34}: {report.get('run_id')}",
        f"{'status':<34}: {report.get('status')}",
        f"{'database':<34}: {report.get('database')}",
        f"{'symbol':<34}: {report.get('symbol')}",
        f"{'duration_seconds':<34}: {report.get('duration_seconds')}",
        "",
        "Training Config",
        "-" * 74,
    ]
    for k, v in report["training_config"].items():
        lines.append(f"{k:<34}: {v}")
    lines += ["", "Counts", "-" * 74]
    for k, v in report["counts"].items():
        lines.append(f"{k:<34}: {v}")
    lines += ["", "Split Distribution Loaded", "-" * 74]
    for k, v in report["split_distribution_loaded"].items():
        lines.append(f"{k:<34}: {v}")
    lines += ["", "Metrics Summary", "-" * 74]
    for target, m in report["metrics"].items():
        lines.append(f"{target} ({m['type']}):")
        for split, sm in m.get("split_metrics", {}).items():
            if m["type"] == "classification":
                lines.append(f"  {split:<8} rows={sm['rows']} acc={sm['accuracy']:.4f} macro_f1={sm['macro_f1']:.4f} weighted_f1={sm['weighted_f1']:.4f}")
            else:
                lines.append(f"  {split:<8} rows={sm['rows']} mae={sm['mae']:.4f} rmse={sm['rmse']:.4f} r2={sm['r2']:.4f}")
    lines += ["", "Outputs", "-" * 74]
    for k, v in report["outputs"].items():
        lines.append(f"{k:<34}: {v}")
    if report["warnings"]:
        lines += ["", "Warnings", "-" * 74]
        lines += [f"- {w}" for w in report["warnings"]]
    if report["errors"]:
        lines += ["", "Errors", "-" * 74]
        lines += [f"- {e}" for e in report["errors"]]
    tp.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Train bq2 MA Quality Model M30 v1.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--target-set", choices=["core", "full"], default="core")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    rs = stamp(started)
    report = {
        "script_name": SCRIPT_NAME,
        "run_id": f"train_ma_quality_m30_v1_{rs}",
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
            "target_set": args.target_set,
            "algorithm": "sklearn.HistGradientBoosting",
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "random_state": args.random_state,
            "limit": args.limit,
        },
        "counts": {"rows_loaded": 0, "skipped_bad_x": 0, "skipped_bad_y": 0, "skipped_bad_split": 0, "skipped_feature_count_mismatch": 0, "models_trained": 0},
        "feature_count": None,
        "split_distribution_loaded": {},
        "targets_trained": [],
        "metrics": {},
        "outputs": {},
        "warnings": [],
        "errors": [],
    }

    print("=== Book & Quality v2 - Train MA Quality Model M30 v1 ===", flush=True)
    try:
        deps = check_dependencies()
        cfg = load_config()
        report["database"] = cfg["database"]
        report["symbol"] = cfg["symbol"]
        db = connect(cfg)
        coll = cfg["bq2_collections"]["dataset_ma_quality"]
        print(f"Database : {cfg['database']}", flush=True)
        print(f"Source   : {coll}", flush=True)
        print(f"Target set: {args.target_set}", flush=True)

        X, y_rows, splits = load_dataset(db, coll, args.limit, report)
        idx = split_indices(splits)
        if len(idx["valid"]) == 0 or len(idx["test"]) == 0:
            report["warnings"].append("Validation or test split is empty. Use full dataset for reliable evaluation.")

        cls_targets = list(CORE_CLASSIFICATION_TARGETS)
        reg_targets = list(CORE_REGRESSION_TARGETS)
        if args.target_set == "full":
            cls_targets += MANAGEMENT_CLASSIFICATION_TARGETS

        bundle = {
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "trained_at": utc_now().isoformat(),
            "symbol": cfg["symbol"],
            "algorithm": "HistGradientBoosting",
            "feature_count": int(X.shape[1]),
            "target_set": args.target_set,
            "models": {},
            "encoders": {},
            "metrics": {},
            "training_config": report["training_config"],
        }

        for target in cls_targets:
            print(f"[TRAIN] classifier: {target}", flush=True)
            model, enc, metrics = train_classifier(target, X, y_rows, idx, deps, args)
            bundle["models"][target] = model
            bundle["encoders"][target] = enc
            bundle["metrics"][target] = metrics
            report["metrics"][target] = metrics
            report["targets_trained"].append(target)
            report["counts"]["models_trained"] += 1

        for target in reg_targets:
            print(f"[TRAIN] regressor: {target}", flush=True)
            model, metrics = train_regressor(target, X, y_rows, idx, deps, args)
            bundle["models"][target] = model
            bundle["metrics"][target] = metrics
            report["metrics"][target] = metrics
            report["targets_trained"].append(target)
            report["counts"]["models_trained"] += 1

        model_path = models_dir() / f"{MODEL_VERSION}_{args.target_set}_{rs}.joblib"
        latest_path = models_dir() / f"{MODEL_VERSION}_{args.target_set}_latest.joblib"
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
