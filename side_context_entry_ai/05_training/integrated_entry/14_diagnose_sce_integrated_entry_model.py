# -*- coding: utf-8 -*-
"""
Side Context Entry AI - Model Probability Diagnostics

Purpose:
- Evaluate a trained SCE integrated entry model without retraining.
- Check probability buckets, threshold grid, side/year breakdown.
- Reports are written locally in 05_training/integrated_entry/reports/

Runtime params use --param value.
"""
from __future__ import annotations

import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from pymongo import ASCENDING, MongoClient
from pymongo.errors import AutoReconnect, CursorNotFound, NetworkTimeout, ServerSelectionTimeoutError
import time

DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DATABASE = "market_data"
DEFAULT_COLLECTION = "sce_dataset_integrated_entry_m5_v1"
DEFAULT_DATASET_VERSION = "sce_dataset_integrated_entry_m5_v1"
TARGET_COL = "is_good_entry"

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
    return Path(__file__).resolve().parents[2]


def parse_dt(value: str):
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Invalid datetime: {value}")


def build_base_query(args, split_name: str) -> Dict[str, Any]:
    q: Dict[str, Any] = {"dataset_version": args.dataset_version}
    if split_name.lower() != "all":
        q["split"] = split_name
    start_dt = parse_dt(args.start_time)
    end_dt = parse_dt(args.end_time)
    if start_dt or end_dt:
        q["anchor_time"] = {}
        if start_dt:
            q["anchor_time"]["$gte"] = start_dt
        if end_dt:
            q["anchor_time"]["$lte"] = end_dt
    return q


def paginate_query(base_query: Dict[str, Any], last_anchor, last_side):
    if last_anchor is None:
        return dict(base_query)
    q = dict(base_query)
    q["$or"] = [
        {"anchor_time": {"$gt": last_anchor}},
        {"anchor_time": last_anchor, "side": {"$gt": last_side or ""}},
    ]
    return q


def load_rows_chunked(db, args, split_name: str) -> pd.DataFrame:
    coll = db[args.collection]
    base_query = build_base_query(args, split_name)
    rows: List[Dict[str, Any]] = []
    loaded = 0
    pages = 0
    retries = 0
    last_anchor = None
    last_side = None
    while True:
        remaining = None if args.limit == 0 else max(0, args.limit - loaded)
        if remaining == 0:
            break
        page_size = args.read_batch_size if remaining is None else min(args.read_batch_size, remaining)
        q = paginate_query(base_query, last_anchor, last_side)
        for attempt in range(args.mongo_retries + 1):
            try:
                batch = list(
                    coll.find(q, EXCLUDE_FROM_MONGO_READ)
                    .sort([("anchor_time", ASCENDING), ("side", ASCENDING)])
                    .limit(page_size)
                    .batch_size(min(page_size, args.mongo_cursor_batch_size))
                )
                break
            except (AutoReconnect, CursorNotFound, NetworkTimeout, ServerSelectionTimeoutError):
                retries += 1
                if attempt >= args.mongo_retries:
                    raise
                time.sleep(args.retry_sleep_seconds * (attempt + 1))
        if not batch:
            break
        rows.extend(batch)
        loaded += len(batch)
        pages += 1
        last_anchor = batch[-1].get("anchor_time")
        last_side = batch[-1].get("side")
        if args.progress_every_pages > 0 and pages % args.progress_every_pages == 0:
            print(f"[{split_name}] pages={pages} rows={loaded} last_anchor={last_anchor} side={last_side}", flush=True)
        if len(batch) < page_size:
            break
    df = pd.DataFrame(rows)
    df.attrs["pages_loaded"] = pages
    df.attrs["mongo_retries"] = retries
    return df


def transform_for_model(df: pd.DataFrame, artifact: Dict[str, Any]) -> pd.DataFrame:
    raw_cols = artifact["raw_feature_columns"]
    final_cols = artifact["feature_columns"]
    categorical_cols = artifact.get("categorical_columns", [])
    x = df.reindex(columns=raw_cols).copy()
    for col in raw_cols:
        if col in categorical_cols:
            x[col] = x[col].astype("string").fillna("__MISSING__")
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0).astype(np.float32)
    x = pd.get_dummies(x, columns=[c for c in categorical_cols if c in x.columns], dummy_na=False)
    x = x.reindex(columns=final_cols, fill_value=0)
    return x


def fmt_pct(x: float) -> str:
    return f"{x:.6f}"


def threshold_table(df: pd.DataFrame, thresholds: List[float]) -> List[Dict[str, Any]]:
    out = []
    base = float(df[TARGET_COL].mean()) if len(df) else 0.0
    for th in thresholds:
        selected = df[df["p_good"] >= th]
        if len(selected) == 0:
            out.append({"threshold": th, "selected": 0, "selected_rate": 0.0, "precision": None, "lift": None, "buy": 0, "sell": 0})
            continue
        precision = float(selected[TARGET_COL].mean())
        out.append({
            "threshold": th,
            "selected": int(len(selected)),
            "selected_rate": float(len(selected) / len(df)),
            "precision": precision,
            "lift": float(precision / base) if base else None,
            "buy": int((selected["side"] == "BUY").sum()) if "side" in selected else None,
            "sell": int((selected["side"] == "SELL").sum()) if "side" in selected else None,
        })
    return out


def decile_table(df: pd.DataFrame, n_bins: int) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    d = df.copy()
    try:
        d["prob_bin"] = pd.qcut(d["p_good"], q=n_bins, duplicates="drop")
    except ValueError:
        d["prob_bin"] = pd.cut(d["p_good"], bins=n_bins)
    base = float(d[TARGET_COL].mean())
    rows = []
    grouped = d.groupby("prob_bin", observed=True)
    for interval, g in grouped:
        rate = float(g[TARGET_COL].mean())
        rows.append({
            "bin": str(interval),
            "rows": int(len(g)),
            "p_min": float(g["p_good"].min()),
            "p_max": float(g["p_good"].max()),
            "positive_rate": rate,
            "lift": float(rate / base) if base else None,
            "buy": int((g["side"] == "BUY").sum()) if "side" in g else None,
            "sell": int((g["side"] == "SELL").sum()) if "side" in g else None,
        })
    return rows


def year_side_table(df: pd.DataFrame, top_quantile: float) -> List[Dict[str, Any]]:
    if df.empty or "anchor_time" not in df:
        return []
    cutoff = float(df["p_good"].quantile(top_quantile))
    selected = df[df["p_good"] >= cutoff].copy()
    selected["year"] = pd.to_datetime(selected["anchor_time"]).dt.year
    rows = []
    for (year, side), g in selected.groupby(["year", "side"], observed=True):
        rows.append({
            "year": int(year),
            "side": str(side),
            "rows": int(len(g)),
            "positive_rate": float(g[TARGET_COL].mean()),
            "p_min": float(g["p_good"].min()),
            "p_avg": float(g["p_good"].mean()),
        })
    return rows


def write_report(path: Path, report: Dict[str, Any], error_text: Optional[str] = None):
    lines = []
    lines.append("Side Context Entry AI - Model Probability Diagnostics Report")
    lines.append("=" * 84)
    for k in ["run_id", "status", "database", "collection", "dataset_version", "model_path", "split", "loader_mode"]:
        lines.append(f"{k:<34}: {report.get(k, '')}")
    lines.append("")
    lines.append("Runtime parameters:")
    for k, v in report.get("runtime_parameters", {}).items():
        lines.append(f"{k:<34}: {v}")
    lines.append("")
    lines.append("Data:")
    for k, v in report.get("data", {}).items():
        lines.append(f"{k:<34}: {v}")
    lines.append("")
    lines.append("Probability summary:")
    for k, v in report.get("probability_summary", {}).items():
        lines.append(f"{k:<34}: {v}")
    lines.append("")
    lines.append("Threshold table:")
    for row in report.get("threshold_table", []):
        lines.append(" | ".join(f"{k}={v}" for k, v in row.items()))
    lines.append("")
    lines.append("Probability bins:")
    for row in report.get("decile_table", []):
        lines.append(" | ".join(f"{k}={v}" for k, v in row.items()))
    lines.append("")
    lines.append("Top probability by year/side:")
    for row in report.get("year_side_table", []):
        lines.append(" | ".join(f"{k}={v}" for k, v in row.items()))
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
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--read-batch-size", type=int, default=5000)
    parser.add_argument("--mongo-cursor-batch-size", type=int, default=1000)
    parser.add_argument("--mongo-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--progress-every-pages", type=int, default=25)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--top-quantile", type=float, default=0.90)
    parser.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    args = parser.parse_args()

    run_id = "sce_diagnose_integrated_entry_model_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    root = project_root_from_script()
    report_dir = root / "05_training" / "integrated_entry" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_id}.txt"

    report: Dict[str, Any] = {
        "run_id": run_id,
        "status": "failed",
        "database": args.database,
        "collection": args.collection,
        "dataset_version": args.dataset_version,
        "model_path": args.model_path,
        "split": args.split,
        "loader_mode": "chunked_anchor_time_side_pagination",
        "runtime_parameters": vars(args),
    }

    try:
        artifact = joblib.load(args.model_path)
        model = artifact["model"]
        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=20000, connectTimeoutMS=20000)
        db = client[args.database]
        df = load_rows_chunked(db, args, args.split)
        if df.empty:
            raise RuntimeError("No rows loaded for diagnostics")
        x = transform_for_model(df, artifact)
        p = model.predict_proba(x)[:, 1]
        df["p_good"] = p
        df[TARGET_COL] = df[TARGET_COL].astype(int)
        thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

        report["data"] = {
            "rows": int(len(df)),
            "positive_rate": float(df[TARGET_COL].mean()),
            "pages_loaded": df.attrs.get("pages_loaded"),
            "mongo_retries": df.attrs.get("mongo_retries"),
        }
        report["probability_summary"] = {
            "p_min": float(df["p_good"].min()),
            "p_max": float(df["p_good"].max()),
            "p_mean": float(df["p_good"].mean()),
            "p_std": float(df["p_good"].std()),
            "p_q50": float(df["p_good"].quantile(0.50)),
            "p_q75": float(df["p_good"].quantile(0.75)),
            "p_q90": float(df["p_good"].quantile(0.90)),
            "p_q95": float(df["p_good"].quantile(0.95)),
            "p_q99": float(df["p_good"].quantile(0.99)),
        }
        report["threshold_table"] = threshold_table(df, thresholds)
        report["decile_table"] = decile_table(df, args.bins)
        report["year_side_table"] = year_side_table(df, args.top_quantile)
        report["status"] = "success"
        write_report(report_path, report)
        print(f"Report written: {report_path}")
        return 0
    except Exception:
        err = traceback.format_exc()
        write_report(report_path, report, err)
        print(f"FAILED. Report written: {report_path}")
        print(err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
