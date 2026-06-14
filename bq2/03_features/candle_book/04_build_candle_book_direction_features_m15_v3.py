# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Build Candle Book Direction Features M15 v3

Location:
    Book_Quality/03_features/candle_book/04_build_candle_book_direction_features_m15_v3.py

Purpose:
    Build no-leak features for the Candle Book Confirmation Model.

Model role:
    Candle Book confirms entry timing.
    It predicts book_direction = BUY / SELL / UNCLEAR.

Anchor:
    M15 closed candle.

Inputs:
    Labels:
        bq2_labels_candle_book_direction_m15_v3
    Raw market data:
        xauusd_m1
        xauusd_m5
        xauusd_m15

Output:
    bq2_features_candle_book_direction_m15_v3

No-Leak rule:
    label.decision_time = next M15 open after anchor candle.
    M1/M5 features use only candles with datetime < decision_time.
    M15 features use candles with datetime <= anchor_time.

TEST:
    cd C:/Project/BookQuality/bq2/03_features/candle_book
    python -u 04_build_candle_book_direction_features_m15_v3.py --limit 1000 --reset

FULL:
    python -u 04_build_candle_book_direction_features_m15_v3.py --reset
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError

try:
    import numpy as np
except Exception as exc:
    raise RuntimeError("numpy is required. Install it with: pip install numpy") from exc


SCRIPT_NAME = "04_build_candle_book_direction_features_m15_v3.py"
FEATURE_VERSION = "candle_book_direction_features_m15_v3"
LABEL_VERSION = "candle_book_direction_label_m15_v3"
BATCH_SIZE = 1000

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "raw_collections": {
        "m1": "xauusd_m1",
        "m5": "xauusd_m5",
        "m15": "xauusd_m15",
    },
    "bq2_collections": {
        "labels_candle_book": "bq2_labels_candle_book_direction_m15_v3",
        "features_candle_book": "bq2_features_candle_book_direction_m15_v3",
    },
    "candle_book_direction_features_m15_v3": {
        "m1_windows": [5, 15, 30, 60],
        "m5_windows": [3, 6, 12, 24, 48],
        "m15_windows": [3, 5, 10, 20, 50],
        "min_history": {"m1": 60, "m5": 48, "m15": 50},
    },
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reports_dir() -> Path:
    p = project_root() / "03_features" / "reports"
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

    cfg.setdefault("raw_collections", {})
    cfg.setdefault("bq2_collections", {})
    cfg["raw_collections"]["m1"] = cfg["raw_collections"].get("m1", "xauusd_m1")
    cfg["raw_collections"]["m5"] = cfg["raw_collections"].get("m5", "xauusd_m5")
    cfg["raw_collections"]["m15"] = cfg["raw_collections"].get("m15", "xauusd_m15")
    cfg["bq2_collections"]["labels_candle_book"] = "bq2_labels_candle_book_direction_m15_v3"
    cfg["bq2_collections"]["features_candle_book"] = "bq2_features_candle_book_direction_m15_v3"
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


class Series:
    def __init__(self, name: str, times: List[datetime], opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray):
        self.name = name
        self.times = times
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes

    def idx_lt(self, dt: datetime) -> int:
        return bisect.bisect_left(self.times, dt) - 1

    def idx_le(self, dt: datetime) -> int:
        return bisect.bisect_right(self.times, dt) - 1

    def count(self) -> int:
        return len(self.times)


def load_series(db: Database, coll_name: str, name: str) -> Series:
    print(f"[LOAD] {name} from {coll_name} ...", flush=True)
    times: List[datetime] = []
    opens: List[float] = []
    highs: List[float] = []
    lows: List[float] = []
    closes: List[float] = []

    cursor = (
        db[coll_name]
        .find({}, projection={"_id": 0, "datetime": 1, "open": 1, "high": 1, "low": 1, "close": 1})
        .sort("datetime", ASCENDING)
        .batch_size(50000)
    )
    for row in cursor:
        times.append(parse_dt(row["datetime"]))
        opens.append(sf(row["open"]))
        highs.append(sf(row["high"]))
        lows.append(sf(row["low"]))
        closes.append(sf(row["close"]))

    s = Series(
        name=name,
        times=times,
        opens=np.asarray(opens, dtype=np.float32),
        highs=np.asarray(highs, dtype=np.float32),
        lows=np.asarray(lows, dtype=np.float32),
        closes=np.asarray(closes, dtype=np.float32),
    )
    print(f"[LOAD] {name}: {s.count():,} rows", flush=True)
    return s


def add_last_candle_features(prefix: str, s: Series, idx: int, features: Dict[str, float]) -> None:
    o = float(s.opens[idx])
    h = float(s.highs[idx])
    l = float(s.lows[idx])
    c = float(s.closes[idx])
    rng = max(h - l, 1e-9)
    body = c - o

    features[f"{prefix}_last_range"] = h - l
    features[f"{prefix}_last_body"] = body
    features[f"{prefix}_last_abs_body"] = abs(body)
    features[f"{prefix}_last_body_to_range"] = abs(body) / rng
    features[f"{prefix}_last_upper_shadow_to_range"] = (h - max(o, c)) / rng
    features[f"{prefix}_last_lower_shadow_to_range"] = (min(o, c) - l) / rng
    features[f"{prefix}_last_close_pos"] = (c - l) / rng
    features[f"{prefix}_last_direction"] = 1.0 if c > o else (-1.0 if c < o else 0.0)


def add_window_features(prefix: str, s: Series, idx: int, window: int, features: Dict[str, float]) -> bool:
    start = idx - window + 1
    if start < 0:
        return False

    o = s.opens[start:idx + 1].astype(np.float64)
    h = s.highs[start:idx + 1].astype(np.float64)
    l = s.lows[start:idx + 1].astype(np.float64)
    c = s.closes[start:idx + 1].astype(np.float64)

    ranges = h - l
    bodies = c - o
    abs_bodies = np.abs(bodies)
    diffs = np.diff(c) if len(c) > 1 else np.asarray([0.0])

    first_open = float(o[0])
    last_close = float(c[-1])
    high_max = float(np.max(h))
    low_min = float(np.min(l))
    total_range = max(high_max - low_min, 1e-9)

    key = f"{prefix}_w{window}"
    features[f"{key}_return_usd"] = last_close - first_open
    features[f"{key}_return_to_range"] = (last_close - first_open) / total_range
    features[f"{key}_avg_range"] = float(np.mean(ranges))
    features[f"{key}_std_range"] = float(np.std(ranges))
    features[f"{key}_avg_abs_body"] = float(np.mean(abs_bodies))
    features[f"{key}_body_sum"] = float(np.sum(bodies))
    features[f"{key}_bull_ratio"] = float(np.mean(bodies > 0))
    features[f"{key}_bear_ratio"] = float(np.mean(bodies < 0))
    features[f"{key}_close_pos_in_swing"] = (last_close - low_min) / total_range
    features[f"{key}_distance_from_high"] = high_max - last_close
    features[f"{key}_distance_from_low"] = last_close - low_min
    features[f"{key}_volatility_close_diff"] = float(np.std(diffs)) if len(diffs) > 1 else 0.0
    features[f"{key}_momentum_per_bar"] = (last_close - first_open) / max(float(window), 1.0)

    if window >= 4:
        half = window // 2
        first_half = c[:half]
        second_half = c[-half:]
        features[f"{key}_first_half_return"] = float(first_half[-1] - first_half[0]) if len(first_half) > 1 else 0.0
        features[f"{key}_second_half_return"] = float(second_half[-1] - second_half[0]) if len(second_half) > 1 else 0.0
        features[f"{key}_acceleration"] = features[f"{key}_second_half_return"] - features[f"{key}_first_half_return"]

    return True


def build_features_for_label(label: Dict[str, Any], series: Dict[str, Series], fcfg: Dict[str, Any]) -> Tuple[Optional[Dict[str, float]], Dict[str, Any], Optional[str]]:
    anchor_time = parse_dt(label["anchor_time"])
    decision_time = parse_dt(label["decision_time"])

    idx_m1 = series["m1"].idx_lt(decision_time)
    idx_m5 = series["m5"].idx_lt(decision_time)
    idx_m15 = series["m15"].idx_le(anchor_time)

    min_history = fcfg.get("min_history", {"m1": 60, "m5": 48, "m15": 50})

    if idx_m1 < int(min_history.get("m1", 60)) - 1:
        return None, {}, "m1:not_enough_history"
    if idx_m5 < int(min_history.get("m5", 48)) - 1:
        return None, {}, "m5:not_enough_history"
    if idx_m15 < int(min_history.get("m15", 50)) - 1:
        return None, {}, "m15:not_enough_history"

    features: Dict[str, float] = {}

    for tf_name, idx, windows_key in [
        ("m1", idx_m1, "m1_windows"),
        ("m5", idx_m5, "m5_windows"),
        ("m15", idx_m15, "m15_windows"),
    ]:
        s = series[tf_name]
        add_last_candle_features(tf_name, s, idx, features)
        for w in fcfg.get(windows_key, DEFAULT_CONFIG["candle_book_direction_features_m15_v3"][windows_key]):
            ok = add_window_features(tf_name, s, idx, int(w), features)
            if not ok:
                return None, {}, f"{tf_name}:not_enough_history_w{w}"

    # Cross-timeframe features: short-term entry confirmation context.
    features["cross_m1_m5_last_dir_agree"] = 1.0 if features["m1_last_direction"] == features["m5_last_direction"] else 0.0
    features["cross_m5_m15_last_dir_agree"] = 1.0 if features["m5_last_direction"] == features["m15_last_direction"] else 0.0
    features["cross_all_last_dir_agree"] = 1.0 if (features["m1_last_direction"] == features["m5_last_direction"] == features["m15_last_direction"]) else 0.0

    features["cross_m1_15_ret_vs_m5_12_ret"] = features.get("m1_w15_return_usd", 0.0) - features.get("m5_w12_return_usd", 0.0)
    features["cross_m5_12_ret_vs_m15_5_ret"] = features.get("m5_w12_return_usd", 0.0) - features.get("m15_w5_return_usd", 0.0)
    features["cross_m1_60_ret_vs_m15_5_ret"] = features.get("m1_w60_return_usd", 0.0) - features.get("m15_w5_return_usd", 0.0)

    features["cross_m1_closepos_minus_m5_closepos"] = features.get("m1_w60_close_pos_in_swing", 0.0) - features.get("m5_w12_close_pos_in_swing", 0.0)
    features["cross_m5_closepos_minus_m15_closepos"] = features.get("m5_w48_close_pos_in_swing", 0.0) - features.get("m15_w20_close_pos_in_swing", 0.0)

    bullish_votes = 0
    bearish_votes = 0
    for k in ["m1_w15_return_usd", "m1_w60_return_usd", "m5_w12_return_usd", "m5_w48_return_usd", "m15_w5_return_usd", "m15_w20_return_usd"]:
        v = features.get(k, 0.0)
        bullish_votes += 1 if v > 0 else 0
        bearish_votes += 1 if v < 0 else 0
    features["cross_bullish_momentum_votes"] = float(bullish_votes)
    features["cross_bearish_momentum_votes"] = float(bearish_votes)
    features["cross_momentum_vote_balance"] = float(bullish_votes - bearish_votes)

    feature_meta = {
        "no_leak_ok": True,
        "anchor_time": anchor_time,
        "decision_time": decision_time,
        "max_feature_time": {
            "m1": series["m1"].times[idx_m1],
            "m5": series["m5"].times[idx_m5],
            "m15": series["m15"].times[idx_m15],
        },
        "indexes": {"m1": idx_m1, "m5": idx_m5, "m15": idx_m15},
        "rule": "m1/m5 datetime < decision_time; m15 datetime <= anchor_time",
    }

    return features, feature_meta, None


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
    jp = reports_dir() / f"candle_book_direction_features_m15_v3_report_{rs}.json"
    tp = reports_dir() / f"candle_book_direction_features_m15_v3_report_{rs}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)

    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "Book & Quality v2 - Candle Book Direction Features M15 v3 Report",
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
    for k, v in report["feature_config"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Counts", "-" * 74]
    for k, v in report["counts"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Feature Count", "-" * 74]
    for k, v in report["feature_count"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Label Distribution", "-" * 74]
    for k, v in report["label_distribution"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Skip Reasons", "-" * 74]
    for k, v in report["skip_reasons"].items():
        lines.append(f"{k:<34}: {v}")

    if report["errors"]:
        lines += ["", "Errors", "-" * 74]
        lines += [f"- {e}" for e in report["errors"][:50]]

    tp.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Candle Book Direction Features M15 v3.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    rs = stamp(started)

    cfg = load_config()
    db_name = cfg["database"]
    symbol = cfg["symbol"]
    raw = cfg["raw_collections"]
    bq2 = cfg["bq2_collections"]
    fcfg = cfg.get("candle_book_direction_features_m15_v3", DEFAULT_CONFIG["candle_book_direction_features_m15_v3"])

    label_coll = bq2.get("labels_candle_book_direction", "bq2_labels_candle_book_direction_m15_v3")
    target_coll = bq2.get("features_candle_book_direction", "bq2_features_candle_book_direction_m15_v3")

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"candle_book_direction_features_m15_v3_{rs}",
        "status": "running",
        "database": db_name,
        "symbol": symbol,
        "start_time": started.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "collections": {
            "labels_source": label_coll,
            "m1_source": raw["m1"],
            "m5_source": raw["m5"],
            "m15_source": raw["m15"],
            "target": target_coll,
        },
        "feature_config": {
            "feature_version": FEATURE_VERSION,
            "label_version_expected": LABEL_VERSION,
            "anchor_timeframe": "M15",
            "input_timeframes": ["M1", "M5", "M15"],
            "m1_windows": fcfg.get("m1_windows"),
            "m5_windows": fcfg.get("m5_windows"),
            "m15_windows": fcfg.get("m15_windows"),
            "min_history": fcfg.get("min_history"),
            "no_leak_rule": "m1/m5 datetime < decision_time; m15 datetime <= anchor_time",
        },
        "counts": {
            "labels_loaded": 0,
            "labels_processed": 0,
            "features_built": 0,
            "upserted_or_modified": 0,
            "inserted": 0,
            "matched": 0,
            "modified": 0,
            "skipped_existing": 0,
            "skipped_no_context": 0,
            "skipped_no_label_y": 0,
            "row_errors": 0,
        },
        "feature_count": {"min": None, "max": None, "last": None, "distinct": {}},
        "label_distribution": {},
        "skip_reasons": {},
        "errors": [],
        "args": vars(args),
    }

    print("=== Book & Quality v2 - Build Candle Book Direction Features M15 v3 ===", flush=True)
    print(f"Database : {db_name}", flush=True)
    print(f"Symbol   : {symbol}", flush=True)
    print(f"Labels   : {label_coll}", flush=True)
    print(f"Target   : {target_coll}", flush=True)

    try:
        db = connect(cfg)

        if args.reset:
            deleted = db[target_coll].delete_many({})
            report["reset_deleted_count"] = deleted.deleted_count
            print(f"[RESET] Deleted {deleted.deleted_count:,} docs from {target_coll}", flush=True)

        db[target_coll].create_index([("symbol", ASCENDING), ("anchor_time", ASCENDING)], unique=True, name="uq_symbol_anchor_time")
        db[target_coll].create_index([("feature_version", ASCENDING), ("anchor_time", ASCENDING)], name="ix_feature_version_anchor_time")
        db[target_coll].create_index([("label_version", ASCENDING), ("anchor_time", ASCENDING)], name="ix_label_version_anchor_time")
        db[target_coll].create_index([("y.book_direction", ASCENDING), ("anchor_time", ASCENDING)], name="ix_book_direction_anchor_time")
        db[target_coll].create_index([("y.gap_ratio", ASCENDING), ("anchor_time", ASCENDING)], name="ix_gap_ratio_anchor_time")

        series = {
            "m1": load_series(db, raw["m1"], "m1"),
            "m5": load_series(db, raw["m5"], "m5"),
            "m15": load_series(db, raw["m15"], "m15"),
        }

        label_query: Dict[str, Any] = {"label_version": LABEL_VERSION}
        if args.start or args.end:
            time_filter: Dict[str, Any] = {}
            if args.start:
                time_filter["$gte"] = parse_dt_optional(args.start)
            if args.end:
                time_filter["$lte"] = parse_dt_optional(args.end)
            label_query["anchor_time"] = time_filter

        total_labels = db[label_coll].count_documents(label_query)
        report["counts"]["labels_loaded"] = total_labels

        cursor = (
            db[label_coll]
            .find(label_query, projection={"_id": 0})
            .sort("anchor_time", ASCENDING)
            .batch_size(10000)
        )

        ops: List[UpdateOne] = []

        for label in cursor:
            try:
                if args.limit is not None and report["counts"]["labels_processed"] >= args.limit:
                    break

                report["counts"]["labels_processed"] += 1
                anchor_time = parse_dt(label["anchor_time"])

                if args.skip_existing:
                    exists = db[target_coll].find_one({"symbol": symbol, "anchor_time": anchor_time}, projection={"_id": 1})
                    if exists:
                        report["counts"]["skipped_existing"] += 1
                        continue

                y = label.get("y")
                if not isinstance(y, dict) or not y.get("book_direction"):
                    report["counts"]["skipped_no_label_y"] += 1
                    continue

                features, feature_meta, skip_reason = build_features_for_label(label, series, fcfg)
                if features is None:
                    report["counts"]["skipped_no_context"] += 1
                    report["skip_reasons"][skip_reason or "unknown"] = report["skip_reasons"].get(skip_reason or "unknown", 0) + 1
                    continue

                fcount = len(features)
                fc = report["feature_count"]
                fc["last"] = fcount
                fc["min"] = fcount if fc["min"] is None else min(fc["min"], fcount)
                fc["max"] = fcount if fc["max"] is None else max(fc["max"], fcount)
                fc["distinct"][str(fcount)] = fc["distinct"].get(str(fcount), 0) + 1

                direction = y["book_direction"]
                report["label_distribution"][direction] = report["label_distribution"].get(direction, 0) + 1

                now = utc_now()
                doc = {
                    "symbol": symbol,
                    "anchor_time": anchor_time,
                    "decision_time": parse_dt(label["decision_time"]),
                    "entry_time": parse_dt(label.get("entry_time", label["decision_time"])),
                    "entry_price": sf(label.get("entry_price", y.get("entry_price", 0.0))),

                    "feature_version": FEATURE_VERSION,
                    "label_version": LABEL_VERSION,
                    "anchor_timeframe": "M15",
                    "input_timeframes": ["M1", "M5", "M15"],
                    "features": features,
                    "feature_count": fcount,
                    "feature_meta": feature_meta,
                    "y": {
                        "book_direction": direction,
                        "label_method": y.get("label_method"),
                        "soft_buy_score": sf(y.get("soft_buy_score", 0.0)),
                        "soft_sell_score": sf(y.get("soft_sell_score", 0.0)),
                        "soft_unclear_score": sf(y.get("soft_unclear_score", 0.0)),
                        "buy_pressure": sf(y.get("buy_pressure", 0.0)),
                        "sell_pressure": sf(y.get("sell_pressure", 0.0)),
                        "pressure_gap": sf(y.get("pressure_gap", 0.0)),
                        "abs_pressure_gap": sf(y.get("abs_pressure_gap", 0.0)),
                        "gap_ratio": sf(y.get("gap_ratio", 0.0)),
                    },
                    "created_at": now,
                    "updated_at": now,
                }

                ops.append(make_update_op(symbol, doc))
                report["counts"]["features_built"] += 1

                if len(ops) >= BATCH_SIZE:
                    execute_bulk(db, target_coll, ops, report)
                    ops = []
                    print(
                        f"[PROGRESS] labels={report['counts']['labels_processed']:,} "
                        f"built={report['counts']['features_built']:,} "
                        f"written≈{report['counts']['upserted_or_modified']:,}",
                        flush=True,
                    )

            except Exception as row_exc:
                report["counts"]["row_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(f"anchor={label.get('anchor_time')} | {row_exc}")

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
    print(f"Status             : {report['status']}", flush=True)
    print(f"Labels Loaded      : {report['counts']['labels_loaded']:,}", flush=True)
    print(f"Labels Processed   : {report['counts']['labels_processed']:,}", flush=True)
    print(f"Features Built     : {report['counts']['features_built']:,}", flush=True)
    print(f"Upserted/Modified  : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Row Errors         : {report['counts']['row_errors']:,}", flush=True)
    print(f"Feature Count      : {report['feature_count']}", flush=True)
    print(f"TXT Report         : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
