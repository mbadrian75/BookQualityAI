# -*- coding: utf-8 -*-
"""
Side Context Entry AI - Integrated Entry Model Training
v3: no-leak + chunked Mongo loader

Folder policy:
- Script lives in: 05_training/integrated_entry/
- Reports written to: 05_training/integrated_entry/reports/
- Models written to: 06_models/integrated_entry/

Runtime params are passed with --param value.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import AutoReconnect, CursorNotFound, NetworkTimeout, ServerSelectionTimeoutError
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover
    LGBMClassifier = None


PROJECT_NAME = "Side Context Entry AI"
PROJECT_CODE = "SCE"
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DATABASE = "market_data"
DEFAULT_DATASET_COLLECTION = "sce_dataset_integrated_entry_m5_v1"
DEFAULT_DATASET_VERSION = "sce_dataset_integrated_entry_m5_v1"
TARGET_COL = "is_good_entry"

# Never use these as model features.
DROP_ALWAYS = {
    "_id",
    "run_id",
    "created_at",
    "updated_at",
    "anchor_time",
    "entry_time",
    "split",
    "feature_version",
    "label_version",
    "dataset_version",
    "ma_feature_version",
    "candle_feature_version",
    "label_targets_included",
    "dataset_targets_included",
    "entry_label",
    "entry_label_id",
    "is_good_entry",
    "entry_tp_bucket",
    "entry_tp_bucket_id",
    "max_entry_success_tp_atr",
    "success_tp_levels_atr",
    "fail_tp_levels_atr",
    "wait_tp_levels_atr",
    "label_debug",
    "debug",
    "source",
}

FORBIDDEN_FEATURE_TOKENS = [
    "entry_label",
    "entry_tp_bucket",
    "is_good_entry",
    "max_entry_success",
    "success_tp",
    "fail_tp",
    "wait_tp",
    "split",
]

CATEGORICAL_CANDIDATES = [
    "side",
    "m30_alignment",
    "m30_trend_phase",
    "h1_alignment",
    "h1_trend_phase",
    "h4_alignment",
    "h4_trend_phase",
    "m5_direction",
    "m15_direction",
]

# Exclusion projection reduces network payload but keeps y/labels for reports.
EXCLUDE_FROM_MONGO_READ = {
    "_id": 0,
    "success_tp_levels_atr": 0,
    "fail_tp_levels_atr": 0,
    "wait_tp_levels_atr": 0,
    "label_debug": 0,
    "debug": 0,
    "source": 0,
}


def project_root_from_script() -> Path:
    # script: root/05_training/integrated_entry/13_train...
    return Path(__file__).resolve().parents[2]


def parse_dt(value: str) -> Optional[datetime]:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"Invalid datetime format: {value}")


def build_base_query(args: argparse.Namespace, split_name: str) -> Dict[str, Any]:
    q: Dict[str, Any] = {
        "dataset_version": args.dataset_version,
        "split": split_name,
    }
    start_dt = parse_dt(args.start_time)
    end_dt = parse_dt(args.end_time)
    if start_dt or end_dt:
        q["anchor_time"] = {}
        if start_dt:
            q["anchor_time"]["$gte"] = start_dt
        if end_dt:
            q["anchor_time"]["$lte"] = end_dt
    return q


def paginate_query(base_query: Dict[str, Any], last_anchor: Optional[datetime], last_side: Optional[str]) -> Dict[str, Any]:
    if last_anchor is None:
        return dict(base_query)
    q = dict(base_query)
    q["$or"] = [
        {"anchor_time": {"$gt": last_anchor}},
        {"anchor_time": last_anchor, "side": {"$gt": last_side or ""}},
    ]
    return q


def load_split_chunked(db, args: argparse.Namespace, split_name: str, limit: int) -> pd.DataFrame:
    coll = db[args.collection]
    base_query = build_base_query(args, split_name)
    rows: List[Dict[str, Any]] = []
    total_loaded = 0
    last_anchor: Optional[datetime] = None
    last_side: Optional[str] = None
    retries = 0
    page_no = 0

    while True:
        remaining = None if limit == 0 else max(0, limit - total_loaded)
        if remaining == 0:
            break
        page_size = args.read_batch_size if remaining is None else min(args.read_batch_size, remaining)
        q = paginate_query(base_query, last_anchor, last_side)

        for attempt in range(args.mongo_retries + 1):
            try:
                cursor = (
                    coll.find(q, EXCLUDE_FROM_MONGO_READ)
                    .sort([("anchor_time", ASCENDING), ("side", ASCENDING)])
                    .limit(page_size)
                    .batch_size(min(page_size, args.mongo_cursor_batch_size))
                )
                batch = list(cursor)
                break
            except (AutoReconnect, CursorNotFound, NetworkTimeout, ServerSelectionTimeoutError) as exc:
                retries += 1
                if attempt >= args.mongo_retries:
                    raise
                time.sleep(args.retry_sleep_seconds * (attempt + 1))
        else:  # pragma: no cover
            batch = []

        if not batch:
            break

        rows.extend(batch)
        total_loaded += len(batch)
        page_no += 1
        last_anchor = batch[-1].get("anchor_time")
        last_side = batch[-1].get("side")

        if args.progress_every_pages > 0 and page_no % args.progress_every_pages == 0:
            print(f"[{split_name}] pages={page_no} rows={total_loaded} last_anchor={last_anchor} last_side={last_side}", flush=True)

        if len(batch) < page_size:
            break

    df = pd.DataFrame(rows)
    df.attrs["mongo_retries"] = retries
    df.attrs["pages_loaded"] = page_no
    return df


def detect_scalar_columns(df: pd.DataFrame) -> List[str]:
    scalar_cols: List[str] = []
    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            scalar_cols.append(col)
            continue
        value = sample.iloc[0]
        if isinstance(value, (dict, list, tuple, set)):
            continue
        scalar_cols.append(col)
    return scalar_cols


def build_feature_matrix(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    for name, df in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        if TARGET_COL not in df.columns:
            raise RuntimeError(f"Missing target column {TARGET_COL} in {name} dataframe")

    base_cols = [c for c in train_df.columns if c not in DROP_ALWAYS]
    base_cols = [c for c in base_cols if c in detect_scalar_columns(train_df)]

    # Drop datetimes and any unexpected object fields not explicitly categorical.
    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in base_cols]
    numeric_cols: List[str] = []
    dropped_non_numeric: List[str] = []
    for col in base_cols:
        if col in categorical_cols:
            continue
        if pd.api.types.is_datetime64_any_dtype(train_df[col]):
            dropped_non_numeric.append(col)
            continue
        if pd.api.types.is_bool_dtype(train_df[col]) or pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_cols.append(col)
        else:
            # Try numeric conversion. If it fails, drop unless categorical whitelist.
            converted = pd.to_numeric(train_df[col], errors="coerce")
            non_null = train_df[col].notna().sum()
            converted_non_null = converted.notna().sum()
            if non_null == 0 or converted_non_null >= max(1, int(non_null * 0.98)):
                numeric_cols.append(col)
            else:
                dropped_non_numeric.append(col)

    feature_cols = numeric_cols + categorical_cols
    check_for_leakage(feature_cols, stage="raw_feature_cols")

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        x = df.reindex(columns=feature_cols).copy()
        for col in numeric_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
        for col in categorical_cols:
            x[col] = x[col].astype("string").fillna("__MISSING__")
        x = pd.get_dummies(x, columns=categorical_cols, dummy_na=False)
        # HistGradient works well with float32 and it saves memory.
        for col in x.columns:
            if x[col].dtype == bool:
                x[col] = x[col].astype(np.uint8)
            elif pd.api.types.is_integer_dtype(x[col]) or pd.api.types.is_float_dtype(x[col]):
                x[col] = x[col].astype(np.float32)
        return x

    x_train = transform(train_df)
    train_columns = list(x_train.columns)
    check_for_leakage(train_columns, stage="encoded_train_cols")

    x_valid = transform(valid_df).reindex(columns=train_columns, fill_value=0)
    x_test = transform(test_df).reindex(columns=train_columns, fill_value=0)
    return x_train, x_valid, x_test, feature_cols, train_columns, categorical_cols


def check_for_leakage(cols: Iterable[str], stage: str) -> None:
    bad = []
    for col in cols:
        low = col.lower()
        for token in FORBIDDEN_FEATURE_TOKENS:
            if token in low:
                bad.append(col)
                break
    if bad:
        preview = bad[:30]
        raise RuntimeError(f"Leakage guard failed at {stage}. Forbidden feature columns: {preview}")


def make_model(args: argparse.Namespace):
    model_type = args.model_type.lower()
    if model_type == "lightgbm" or (model_type == "auto" and LGBMClassifier is not None):
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM requested but not installed")
        return LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            objective="binary",
            class_weight=None,
        )
    if model_type not in ("auto", "histgb"):
        raise RuntimeError(f"Unknown model_type: {args.model_type}")
    return HistGradientBoostingClassifier(
        max_iter=args.n_estimators,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        random_state=args.random_state,
    )


def evaluate(name: str, y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    out: Dict[str, Any] = {
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "predicted_positive_rate": float(np.mean(pred)) if len(pred) else None,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix_0_1": confusion_matrix(y_true, pred).tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    except Exception:
        out["roc_auc"] = None
    try:
        out["pr_auc"] = float(average_precision_score(y_true, proba))
    except Exception:
        out["pr_auc"] = None
    return out


def write_report(path: Path, report: Dict[str, Any], error_text: Optional[str] = None) -> None:
    lines: List[str] = []
    lines.append("Side Context Entry AI - Integrated Entry Model Training Report")
    lines.append("=" * 80)
    header_keys = [
        "run_id", "status", "project_name", "project_code", "database", "collection",
        "dataset_version", "target", "leakage_guard", "loader_mode",
    ]
    for k in header_keys:
        lines.append(f"{k:<30}: {report.get(k, '')}")
    lines.append("")
    lines.append("Runtime parameters:")
    for k, v in report.get("runtime_parameters", {}).items():
        lines.append(f"{k:<30}: {v}")
    lines.append("")
    lines.append("Data:")
    for k, v in report.get("data", {}).items():
        lines.append(f"{k:<30}: {v}")
    lines.append("")
    lines.append("Feature info:")
    for k, v in report.get("feature_info", {}).items():
        lines.append(f"{k:<30}: {v}")
    lines.append("")
    lines.append("Metrics:")
    for split_name, metrics in report.get("metrics", {}).items():
        lines.append(f"[{split_name}]")
        for k, v in metrics.items():
            lines.append(f"{k:<30}: {v}")
        lines.append("")
    lines.append("Model:")
    for k, v in report.get("model", {}).items():
        lines.append(f"{k:<30}: {v}")
    if error_text:
        lines.append("")
        lines.append("Error:")
        lines.append(error_text)
    lines.append("")
    lines.append("End of report.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--collection", default=DEFAULT_DATASET_COLLECTION)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--train-limit", type=int, default=300000)
    parser.add_argument("--valid-limit", type=int, default=100000)
    parser.add_argument("--test-limit", type=int, default=100000)
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--model-type", default="auto", choices=["auto", "histgb", "lightgbm"])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--read-batch-size", type=int, default=5000)
    parser.add_argument("--mongo-cursor-batch-size", type=int, default=1000)
    parser.add_argument("--mongo-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--progress-every-pages", type=int, default=25)
    args = parser.parse_args()

    run_id = "sce_train_integrated_entry_model_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    root = project_root_from_script()
    report_dir = root / "05_training" / "integrated_entry" / "reports"
    model_dir = root / "06_models" / "integrated_entry"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_id}.txt"
    model_path = model_dir / f"sce_integrated_entry_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"

    report: Dict[str, Any] = {
        "run_id": run_id,
        "status": "failed",
        "project_name": PROJECT_NAME,
        "project_code": PROJECT_CODE,
        "database": args.database,
        "collection": args.collection,
        "dataset_version": args.dataset_version,
        "target": TARGET_COL,
        "leakage_guard": "enabled",
        "loader_mode": "chunked_anchor_time_side_pagination",
        "runtime_parameters": vars(args),
        "data": {},
        "feature_info": {
            "initial_feature_cols": None,
            "final_feature_cols": None,
            "categorical_cols": None,
            "leakage_guard_status": None,
        },
        "metrics": {},
        "model": {},
    }

    try:
        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=20000, connectTimeoutMS=20000)
        db = client[args.database]

        train_df = load_split_chunked(db, args, "train", args.train_limit)
        valid_df = load_split_chunked(db, args, "valid", args.valid_limit)
        test_df = load_split_chunked(db, args, "test", args.test_limit)

        if train_df.empty or valid_df.empty or test_df.empty:
            raise RuntimeError(f"Empty split detected: train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")

        y_train = train_df[TARGET_COL].astype(int).to_numpy()
        y_valid = valid_df[TARGET_COL].astype(int).to_numpy()
        y_test = test_df[TARGET_COL].astype(int).to_numpy()

        x_train, x_valid, x_test, raw_feature_cols, final_cols, categorical_cols = build_feature_matrix(train_df, valid_df, test_df)

        report["data"] = {
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "test_rows": int(len(test_df)),
            "train_positive_rate": float(np.mean(y_train)),
            "valid_positive_rate": float(np.mean(y_valid)),
            "test_positive_rate": float(np.mean(y_test)),
            "train_pages_loaded": train_df.attrs.get("pages_loaded"),
            "valid_pages_loaded": valid_df.attrs.get("pages_loaded"),
            "test_pages_loaded": test_df.attrs.get("pages_loaded"),
            "train_mongo_retries": train_df.attrs.get("mongo_retries"),
            "valid_mongo_retries": valid_df.attrs.get("mongo_retries"),
            "test_mongo_retries": test_df.attrs.get("mongo_retries"),
        }
        report["feature_info"] = {
            "initial_feature_cols": int(len(raw_feature_cols)),
            "final_feature_cols": int(len(final_cols)),
            "categorical_cols": categorical_cols,
            "leakage_guard_status": "passed",
        }

        model = make_model(args)
        model.fit(x_train, y_train)

        def prob(x: pd.DataFrame) -> np.ndarray:
            p = model.predict_proba(x)
            return p[:, 1]

        report["metrics"] = {
            "train": evaluate("train", y_train, prob(x_train), args.threshold),
            "valid": evaluate("valid", y_valid, prob(x_valid), args.threshold),
            "test": evaluate("test", y_test, prob(x_test), args.threshold),
        }

        artifact = {
            "model": model,
            "feature_columns": final_cols,
            "raw_feature_columns": raw_feature_cols,
            "categorical_columns": categorical_cols,
            "target": TARGET_COL,
            "dataset_version": args.dataset_version,
            "run_id": run_id,
            "leakage_guard": "enabled",
            "loader_mode": "chunked_anchor_time_side_pagination",
        }
        joblib.dump(artifact, model_path)
        report["model"] = {
            "model_class": model.__class__.__name__,
            "model_path": str(model_path),
            "artifact_saved": True,
        }
        report["status"] = "success"
        write_report(report_path, report)
        print(f"Report written: {report_path}")
        print(f"Model written: {model_path}")
        return 0
    except Exception:
        err = traceback.format_exc()
        report["model"] = {"artifact_saved": False}
        write_report(report_path, report, err)
        print(f"FAILED. Report written: {report_path}", file=sys.stderr)
        print(err, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
