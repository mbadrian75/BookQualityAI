# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Train Candle Book Side Binary Model M15 v2

Location:
    Book_Quality/05_training/candle_book/03_train_candle_book_side_binary_m15_v2.py

Purpose:
    Train two binary confirmation models for API use:
        buy_win  : P(BUY side wins from current M15 entry point)
        sell_win : P(SELL side wins from current M15 entry point)

Why binary:
    The previous 4-class outcome model (WIN/LOSS/FLAT/AMBIGUOUS) produced low confidence.
    For API Decision Engine, we mainly need p_buy_win and p_sell_win.

Input:
    bq2_dataset_candle_book_side_m15_v2

Output model:
    06_models/candle_book/candle_book_side_binary_model_m15_v2_latest.joblib

Recommended TEST:
    cd C:/Project/BookQuality/bq2/05_training/candle_book
    python -u 03_train_candle_book_side_binary_m15_v2.py --limit-per-split 5000 --max-iter 50 --drop-ambiguous

Recommended FULL:
    python -u 03_train_candle_book_side_binary_m15_v2.py --max-iter 200 --l2-regularization 0.1 --drop-ambiguous
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING
from pymongo.database import Database


SCRIPT_NAME = "03_train_candle_book_side_binary_m15_v2.py"
MODEL_VERSION = "candle_book_side_binary_model_m15_v2"
DATASET_VERSION = "candle_book_side_dataset_m15_v2"
FEATURE_VERSION = "candle_book_side_features_m15_v2"
LABEL_VERSION = "candle_book_side_label_m15_v2"

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
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            roc_auc_score,
            log_loss,
            brier_score_loss,
            confusion_matrix,
        )
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies. Install them: pip install numpy scikit-learn joblib"
        ) from exc
    return {
        "joblib": joblib,
        "np": np,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "accuracy_score": accuracy_score,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "roc_auc_score": roc_auc_score,
        "log_loss": log_loss,
        "brier_score_loss": brier_score_loss,
        "confusion_matrix": confusion_matrix,
    }


def sf(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def outcome_to_binary(outcome: str) -> Optional[int]:
    if outcome == "WIN":
        return 1
    if outcome in ("LOSS", "FLAT", "AMBIGUOUS"):
        return 0
    return None


def load_split(
    db: Database,
    coll_name: str,
    split: str,
    limit: Optional[int],
    drop_ambiguous: bool,
    deps: Dict[str, Any],
) -> Dict[str, Any]:
    np = deps["np"]
    query = {"dataset_version": DATASET_VERSION, "split": split}
    cursor = (
        db[coll_name]
        .find(query, projection={"_id": 0, "x": 1, "y": 1, "feature_count": 1, "anchor_time": 1})
        .sort("anchor_time", ASCENDING)
    )
    if limit:
        cursor = cursor.limit(limit)

    X: List[List[float]] = []
    y_buy: List[int] = []
    y_sell: List[int] = []
    rows_seen = 0
    skipped_ambiguous = 0
    skipped_bad_y = 0
    feature_count = None
    dist = {
        "buy_outcome": {},
        "sell_outcome": {},
        "buy_win_binary": {"WIN": 0, "NOT_WIN": 0},
        "sell_win_binary": {"WIN": 0, "NOT_WIN": 0},
    }

    for doc in cursor:
        rows_seen += 1
        y = doc.get("y", {})
        buy_outcome = y.get("buy_outcome")
        sell_outcome = y.get("sell_outcome")
        if buy_outcome is None or sell_outcome is None:
            skipped_bad_y += 1
            continue
        if drop_ambiguous and (buy_outcome == "AMBIGUOUS" or sell_outcome == "AMBIGUOUS"):
            skipped_ambiguous += 1
            continue
        by = outcome_to_binary(buy_outcome)
        sy = outcome_to_binary(sell_outcome)
        if by is None or sy is None:
            skipped_bad_y += 1
            continue
        x = doc.get("x")
        if not isinstance(x, list):
            skipped_bad_y += 1
            continue
        feature_count = len(x) if feature_count is None else feature_count
        X.append([sf(v) for v in x])
        y_buy.append(by)
        y_sell.append(sy)
        dist["buy_outcome"][buy_outcome] = dist["buy_outcome"].get(buy_outcome, 0) + 1
        dist["sell_outcome"][sell_outcome] = dist["sell_outcome"].get(sell_outcome, 0) + 1
        dist["buy_win_binary"]["WIN" if by == 1 else "NOT_WIN"] += 1
        dist["sell_win_binary"]["WIN" if sy == 1 else "NOT_WIN"] += 1

    return {
        "X": np.asarray(X, dtype=np.float32),
        "buy_win": np.asarray(y_buy, dtype=np.int8),
        "sell_win": np.asarray(y_sell, dtype=np.int8),
        "rows_seen": rows_seen,
        "rows_used": len(X),
        "skipped_ambiguous": skipped_ambiguous,
        "skipped_bad_y": skipped_bad_y,
        "feature_count": feature_count,
        "distribution": dist,
    }


def balanced_sample_weight(y, np):
    total = len(y)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return np.ones(total, dtype=np.float32)
    w_pos = total / (2.0 * positives)
    w_neg = total / (2.0 * negatives)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def metric_block(y_true, proba_win, deps: Dict[str, Any]) -> Dict[str, Any]:
    np = deps["np"]
    y_pred = (proba_win >= 0.50).astype(np.int8)
    acc = deps["accuracy_score"](y_true, y_pred)
    precision, recall, f1, support = deps["precision_recall_fscore_support"](
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    try:
        auc = deps["roc_auc_score"](y_true, proba_win)
    except Exception:
        auc = None
    try:
        ll = deps["log_loss"](y_true, np.vstack([1 - proba_win, proba_win]).T, labels=[0, 1])
    except Exception:
        ll = None
    try:
        brier = deps["brier_score_loss"](y_true, proba_win)
    except Exception:
        brier = None

    thresholds = {}
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = proba_win >= th
        cov = float(np.mean(mask)) if len(mask) else 0.0
        n = int(np.sum(mask))
        if n > 0:
            win_rate = float(np.mean(y_true[mask] == 1))
        else:
            win_rate = None
        thresholds[f"p>={th:.2f}"] = {"coverage": round(cov, 6), "rows": n, "actual_win_rate": None if win_rate is None else round(win_rate, 6)}

    cm = deps["confusion_matrix"](y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "rows": int(len(y_true)),
        "true_win_rate": round(float(np.mean(y_true == 1)), 6) if len(y_true) else None,
        "accuracy_at_0_50": round(float(acc), 6),
        "not_win_precision": round(float(precision[0]), 6),
        "not_win_recall": round(float(recall[0]), 6),
        "not_win_f1": round(float(f1[0]), 6),
        "win_precision": round(float(precision[1]), 6),
        "win_recall": round(float(recall[1]), 6),
        "win_f1": round(float(f1[1]), 6),
        "macro_f1": round(float((f1[0] + f1[1]) / 2.0), 6),
        "roc_auc": None if auc is None else round(float(auc), 6),
        "log_loss": None if ll is None else round(float(ll), 6),
        "brier_score": None if brier is None else round(float(brier), 6),
        "avg_p_win": round(float(np.mean(proba_win)), 6) if len(proba_win) else None,
        "p60_rate": round(float(np.mean(proba_win >= 0.60)), 6) if len(proba_win) else None,
        "p70_rate": round(float(np.mean(proba_win >= 0.70)), 6) if len(proba_win) else None,
        "prediction_distribution_at_0_50": {
            "WIN": int(np.sum(y_pred == 1)),
            "NOT_WIN": int(np.sum(y_pred == 0)),
        },
        "confusion_matrix_labels_[NOT_WIN,WIN]": cm,
        "threshold_analysis": thresholds,
    }


def latest_feature_names_file() -> Optional[Path]:
    exports = project_root() / "04_dataset" / "exports"
    if not exports.exists():
        return None
    files = sorted(exports.glob("candle_book_side_m15_v2_feature_names_*.json"))
    return files[-1] if files else None


def load_feature_names(expected_count: int) -> List[str]:
    p = latest_feature_names_file()
    if p:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            names = data.get("feature_names", [])
            if len(names) == expected_count:
                return names
        except Exception:
            pass
    return [f"f_{i}" for i in range(expected_count)]


def save_reports(report: Dict[str, Any], rs: str) -> None:
    jp = reports_dir() / f"candle_book_side_binary_train_m15_v2_report_{rs}.json"
    tp = reports_dir() / f"candle_book_side_binary_train_m15_v2_report_{rs}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "Book & Quality v2 - Train Candle Book Side Binary Model M15 v2 Report",
        "=" * 74,
        f"{'run_id':<34}: {report.get('run_id')}",
        f"{'status':<34}: {report.get('status')}",
        f"{'database':<34}: {report.get('database')}",
        f"{'symbol':<34}: {report.get('symbol')}",
        f"{'start_time':<34}: {report.get('start_time')}",
        f"{'end_time':<34}: {report.get('end_time')}",
        f"{'duration_seconds':<34}: {report.get('duration_seconds')}",
        "", "Collections", "-" * 74,
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
    for target, target_metrics in report["metrics"].items():
        lines.append(f"Target: {target}")
        for split, metrics in target_metrics.items():
            lines.append(f"  {split}:")
            for k, v in metrics.items():
                if k == "threshold_analysis":
                    lines.append(f"    threshold_analysis:")
                    for th, tv in v.items():
                        lines.append(f"      {th}: {tv}")
                else:
                    lines.append(f"    {k:<29}: {v}")
        lines.append("")

    if report["errors"]:
        lines += ["", "Errors", "-" * 74]
        lines += [f"- {e}" for e in report["errors"][:50]]

    tp.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Candle Book Side Binary M15 v2.")
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--no-sample-weight", action="store_true")
    parser.add_argument("--drop-ambiguous", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    rs = stamp(started)

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"candle_book_side_binary_train_m15_v2_{rs}",
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
            "targets": ["buy_win", "sell_win"],
            "target_logic": "WIN=1; LOSS/FLAT/AMBIGUOUS=0; optional drop AMBIGUOUS rows",
            "classifier": "HistGradientBoostingClassifier",
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization,
            "max_leaf_nodes": args.max_leaf_nodes,
            "limit_per_split": args.limit_per_split,
            "sample_weight": not args.no_sample_weight,
            "drop_ambiguous": args.drop_ambiguous,
        },
        "counts": {},
        "model_output": {},
        "metrics": {"buy_win": {}, "sell_win": {}},
        "data_distributions": {},
        "errors": [],
    }

    print("=== Book & Quality v2 - Train Candle Book Side Binary Model M15 v2 ===", flush=True)

    try:
        deps = check_dependencies()
        np = deps["np"]
        HGB = deps["HistGradientBoostingClassifier"]
        cfg = load_config()
        db = connect(cfg)
        coll = cfg["bq2_collections"]["dataset_candle_book_side"]
        report["database"] = cfg["database"]
        report["symbol"] = cfg["symbol"]
        report["collections"] = {"dataset_source": coll}

        data = {}
        for split in ["train", "valid", "test"]:
            data[split] = load_split(db, coll, split, args.limit_per_split, args.drop_ambiguous, deps)
            print(f"[LOAD] {split}: seen={data[split]['rows_seen']:,} used={data[split]['rows_used']:,}", flush=True)

        feature_count = data["train"]["feature_count"]
        if not feature_count:
            raise RuntimeError("No train rows loaded.")
        feature_names = load_feature_names(feature_count)

        report["counts"] = {
            "train_seen": data["train"]["rows_seen"],
            "valid_seen": data["valid"]["rows_seen"],
            "test_seen": data["test"]["rows_seen"],
            "train_used": data["train"]["rows_used"],
            "valid_used": data["valid"]["rows_used"],
            "test_used": data["test"]["rows_used"],
            "rows_loaded": data["train"]["rows_used"] + data["valid"]["rows_used"] + data["test"]["rows_used"],
            "feature_count": feature_count,
            "skipped_ambiguous_train": data["train"]["skipped_ambiguous"],
            "skipped_ambiguous_valid": data["valid"]["skipped_ambiguous"],
            "skipped_ambiguous_test": data["test"]["skipped_ambiguous"],
            "skipped_bad_y_train": data["train"]["skipped_bad_y"],
            "skipped_bad_y_valid": data["valid"]["skipped_bad_y"],
            "skipped_bad_y_test": data["test"]["skipped_bad_y"],
        }
        report["data_distributions"] = {split: data[split]["distribution"] for split in ["train", "valid", "test"]}

        models = {}
        for target in ["buy_win", "sell_win"]:
            y_train = data["train"][target]
            X_train = data["train"]["X"]
            if len(np.unique(y_train)) < 2:
                raise RuntimeError(f"Target {target} has less than two classes in train.")
            model = HGB(
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                l2_regularization=args.l2_regularization,
                max_leaf_nodes=args.max_leaf_nodes,
                random_state=42,
            )
            sw = None if args.no_sample_weight else balanced_sample_weight(y_train, np)
            print(f"[TRAIN] {target} rows={len(y_train):,}", flush=True)
            model.fit(X_train, y_train, sample_weight=sw)
            models[target] = model

            for split in ["train", "valid", "test"]:
                X = data[split]["X"]
                y = data[split][target]
                if len(y) == 0:
                    continue
                proba = model.predict_proba(X)
                # class 1 probability. HistGradient returns columns ordered by model.classes_.
                classes = list(model.classes_)
                idx_1 = classes.index(1)
                p_win = proba[:, idx_1]
                report["metrics"][target][split] = metric_block(y, p_win, deps)

        model_path = models_dir() / f"{MODEL_VERSION}_{rs}.joblib"
        latest_path = models_dir() / f"{MODEL_VERSION}_latest.joblib"
        bundle = {
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "trained_at": utc_now().isoformat(),
            "targets": ["buy_win", "sell_win"],
            "target_logic": "WIN=1; LOSS/FLAT/AMBIGUOUS=0; optional drop AMBIGUOUS rows during training",
            "feature_count": feature_count,
            "feature_names": feature_names,
            "models": models,
            "args": vars(args),
            "metrics": report["metrics"],
        }
        deps["joblib"].dump(bundle, model_path)
        deps["joblib"].dump(bundle, latest_path)
        report["model_output"] = {
            "model_path": str(model_path),
            "latest_model_path": str(latest_path),
            "targets": ["buy_win", "sell_win"],
            "feature_count": feature_count,
            "classes": {k: [int(c) for c in v.classes_] for k, v in models.items()},
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
    print(f"Status          : {report['status']}", flush=True)
    print(f"Rows Loaded     : {report.get('counts', {}).get('rows_loaded')}", flush=True)
    print(f"Feature Count   : {report.get('counts', {}).get('feature_count')}", flush=True)
    for target in ["buy_win", "sell_win"]:
        tm = report.get("metrics", {}).get(target, {}).get("test", {})
        if tm:
            print(f"{target} test auc={tm.get('roc_auc')} win_precision={tm.get('win_precision')} win_recall={tm.get('win_recall')} p60={tm.get('p60_rate')}", flush=True)
    print(f"TXT Report      : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
