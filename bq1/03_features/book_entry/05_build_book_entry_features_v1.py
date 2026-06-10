# -*- coding: utf-8 -*-
"""
Book & Quality - Build Book Entry Features v1

Location:
    Book_Quality/03_features/book_entry/05_build_book_entry_features_v1.py

Purpose:
    Build no-leak features for the Book Entry Model.

Model:
    Book Entry Model

Input timeframes:
    M1 / M5 / M15

Anchor:
    anchor_time comes from bq_labels_book_entry_m15_v1.
    decision_time = entry_time from label document.
    Features only use raw candles with datetime < decision_time.

Output collection:
    bq_features_book_entry_m15_v1

Reports:
    Book_Quality/03_features/reports/book_entry_features_v1_report_YYYYMMDD_HHMMSS.json
    Book_Quality/03_features/reports/book_entry_features_v1_report_YYYYMMDD_HHMMSS.txt

Requirements:
    pip install pymongo numpy

TEST ONLY:
    cd C:\Project\Book_Quality\03_features\book_entry
    python -u 05_build_book_entry_features_v1.py --limit 1000 --reset

FULL BUILD:
    python -u 05_build_book_entry_features_v1.py --reset
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from array import array
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError


SCRIPT_NAME = "05_build_book_entry_features_v1.py"
FEATURE_VERSION = "book_entry_features_v1"

TIME_FIELD = "datetime"
OPEN_FIELD = "open"
HIGH_FIELD = "high"
LOW_FIELD = "low"
CLOSE_FIELD = "close"
VOLUME_CANDIDATES = ["volume", "tick_volume", "Volume", "TickVolume", "vol"]

BATCH_SIZE = 1000

WINDOWS = {
    "m1": [15, 30, 60, 120],
    "m5": [3, 6, 12, 24],
    "m15": [1, 4, 8, 16, 32, 64],
}

TF_MINUTES = {
    "m1": 1,
    "m5": 5,
    "m15": 15,
}

MAX_LOOKBACK_MINUTES = max(
    max(WINDOWS["m1"]) * TF_MINUTES["m1"],
    max(WINDOWS["m5"]) * TF_MINUTES["m5"],
    max(WINDOWS["m15"]) * TF_MINUTES["m15"],
)

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "raw_collections": {
        "m1": "xauusd_m1",
        "m5": "xauusd_m5",
        "m15": "xauusd_m15",
    },
    "bq_collections": {
        "labels_book_entry": "bq_labels_book_entry_m15_v1",
        "features_book_entry": "bq_features_book_entry_m15_v1",
    },
}


# =========================
# Generic helpers
# =========================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_reports_dir() -> Path:
    reports_dir = get_project_root() / "03_features" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read JSON file: {path} | {exc}", flush=True)
        return None


def load_config() -> Dict[str, Any]:
    root = get_project_root()
    config = DEFAULT_CONFIG.copy()
    cfg = read_json(root / "00_config" / "mongo_config.json")
    if cfg:
        config.update(cfg)
    return config


def connect(config: Dict[str, Any]) -> Database:
    client = MongoClient(config["mongo_uri"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[config["database"]]


def parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    raise ValueError(f"Unsupported datetime value: {value!r}")


def parse_dt_arg(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def epoch_seconds(dt: datetime) -> int:
    # Naive datetimes are treated consistently as UTC-like sequence keys.
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def clean_number(value: Any) -> float:
    return float(round(safe_float(value), 10))


def clean_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


# =========================
# Data loading
# =========================

class TfArrays:
    def __init__(self, tf: str, times: np.ndarray, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray):
        self.tf = tf
        self.times = times
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.volumes = volumes

    @property
    def count(self) -> int:
        return int(len(self.times))

    def end_index_before(self, cutoff_epoch: int) -> int:
        # Uses candles with datetime < decision_time.
        return int(np.searchsorted(self.times, cutoff_epoch, side="left"))


def detect_volume_field(sample: Dict[str, Any]) -> Optional[str]:
    for key in VOLUME_CANDIDATES:
        if key in sample:
            return key
    return None


def load_tf_arrays(db: Database, collection_name: str, tf: str, start_dt: datetime, end_dt: datetime, report: Dict[str, Any]) -> TfArrays:
    print(f"[LOAD] {tf.upper()} from {collection_name}: {start_dt} -> {end_dt}", flush=True)

    sample = db[collection_name].find_one()
    if not sample:
        raise RuntimeError(f"Source collection is empty: {collection_name}")

    volume_field = detect_volume_field(sample)

    projection = {
        TIME_FIELD: 1,
        OPEN_FIELD: 1,
        HIGH_FIELD: 1,
        LOW_FIELD: 1,
        CLOSE_FIELD: 1,
        "_id": 0,
    }
    if volume_field:
        projection[volume_field] = 1

    q = {
        TIME_FIELD: {
            "$gte": start_dt,
            "$lt": end_dt,
        }
    }

    t_arr = array("q")
    o_arr = array("d")
    h_arr = array("d")
    l_arr = array("d")
    c_arr = array("d")
    v_arr = array("d")

    cursor = (
        db[collection_name]
        .find(q, projection=projection)
        .sort(TIME_FIELD, ASCENDING)
        .batch_size(10000)
    )

    count = 0
    for doc in cursor:
        dt = parse_dt(doc[TIME_FIELD])
        t_arr.append(epoch_seconds(dt))
        o_arr.append(safe_float(doc.get(OPEN_FIELD)))
        h_arr.append(safe_float(doc.get(HIGH_FIELD)))
        l_arr.append(safe_float(doc.get(LOW_FIELD)))
        c_arr.append(safe_float(doc.get(CLOSE_FIELD)))
        v_arr.append(safe_float(doc.get(volume_field)) if volume_field else 0.0)
        count += 1

    times = np.frombuffer(t_arr, dtype=np.int64).copy()
    opens = np.frombuffer(o_arr, dtype=np.float64).copy()
    highs = np.frombuffer(h_arr, dtype=np.float64).copy()
    lows = np.frombuffer(l_arr, dtype=np.float64).copy()
    closes = np.frombuffer(c_arr, dtype=np.float64).copy()
    volumes = np.frombuffer(v_arr, dtype=np.float64).copy()

    report["source_loads"][tf] = {
        "collection": collection_name,
        "start": start_dt.isoformat(sep=" "),
        "end": end_dt.isoformat(sep=" "),
        "count": int(count),
        "volume_field": volume_field,
    }

    print(f"[LOAD] {tf.upper()} loaded: {count:,}", flush=True)

    if count == 0:
        report["warnings"].append(f"{tf.upper()} source load returned 0 rows.")

    return TfArrays(tf, times, opens, highs, lows, closes, volumes)


def load_anchors(db: Database, labels_collection: str, symbol: str, start_arg: Optional[datetime], end_arg: Optional[datetime], limit: Optional[int]) -> List[Dict[str, Any]]:
    q: Dict[str, Any] = {"symbol": symbol}

    if start_arg or end_arg:
        q["anchor_time"] = {}
        if start_arg:
            q["anchor_time"]["$gte"] = start_arg
        if end_arg:
            q["anchor_time"]["$lte"] = end_arg

    cursor = (
        db[labels_collection]
        .find(q, projection={"anchor_time": 1, "entry_time": 1, "label": 1, "_id": 0})
        .sort("anchor_time", ASCENDING)
    )

    if limit is not None:
        cursor = cursor.limit(limit)

    anchors = list(cursor)

    for item in anchors:
        item["anchor_time"] = parse_dt(item["anchor_time"])
        item["entry_time"] = parse_dt(item["entry_time"])

    return anchors


# =========================
# Feature engineering
# =========================

def add_missing_window_features(features: Dict[str, float], prefix: str) -> None:
    keys = [
        "valid", "count", "close_change", "close_change_r", "high_low_range", "avg_range",
        "max_range", "avg_body", "body_abs_avg", "body_to_range_avg", "bull_ratio",
        "bear_ratio", "upper_wick_avg", "lower_wick_avg", "close_location_avg",
        "volatility_close_diff", "volume_sum", "volume_avg"
    ]
    for k in keys:
        features[f"{prefix}_{k}"] = 0.0


def add_window_features(features: Dict[str, float], prefix: str, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> None:
    n = len(closes)
    if n < 1:
        add_missing_window_features(features, prefix)
        return

    ranges = highs - lows
    bodies = closes - opens
    abs_bodies = np.abs(bodies)

    total_range = float(np.max(highs) - np.min(lows)) if n > 0 else 0.0
    close_change = float(closes[-1] - closes[0]) if n > 1 else float(closes[-1] - opens[-1])
    denom = total_range if total_range > 0 else 1.0

    upper_wicks = highs - np.maximum(opens, closes)
    lower_wicks = np.minimum(opens, closes) - lows
    safe_ranges = np.where(ranges == 0, 1.0, ranges)
    close_locations = (closes - lows) / safe_ranges

    if n > 1:
        diff_std = float(np.std(np.diff(closes)))
    else:
        diff_std = 0.0

    features[f"{prefix}_valid"] = 1.0
    features[f"{prefix}_count"] = float(n)
    features[f"{prefix}_close_change"] = clean_number(close_change)
    features[f"{prefix}_close_change_r"] = clean_number(close_change / denom)
    features[f"{prefix}_high_low_range"] = clean_number(total_range)
    features[f"{prefix}_avg_range"] = clean_number(np.mean(ranges))
    features[f"{prefix}_max_range"] = clean_number(np.max(ranges))
    features[f"{prefix}_avg_body"] = clean_number(np.mean(bodies))
    features[f"{prefix}_body_abs_avg"] = clean_number(np.mean(abs_bodies))
    features[f"{prefix}_body_to_range_avg"] = clean_number(np.mean(abs_bodies / safe_ranges))
    features[f"{prefix}_bull_ratio"] = clean_number(np.mean(closes > opens))
    features[f"{prefix}_bear_ratio"] = clean_number(np.mean(closes < opens))
    features[f"{prefix}_upper_wick_avg"] = clean_number(np.mean(upper_wicks))
    features[f"{prefix}_lower_wick_avg"] = clean_number(np.mean(lower_wicks))
    features[f"{prefix}_close_location_avg"] = clean_number(np.mean(close_locations))
    features[f"{prefix}_volatility_close_diff"] = clean_number(diff_std)
    features[f"{prefix}_volume_sum"] = clean_number(np.sum(volumes))
    features[f"{prefix}_volume_avg"] = clean_number(np.mean(volumes))


def add_last_candle_features(features: Dict[str, float], prefix: str, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> None:
    if len(closes) < 1:
        for k in ["open", "high", "low", "close", "range", "body", "body_abs", "direction", "upper_wick", "lower_wick", "close_location"]:
            features[f"{prefix}_last_{k}"] = 0.0
        return

    o = float(opens[-1])
    h = float(highs[-1])
    l = float(lows[-1])
    c = float(closes[-1])
    rng = h - l
    body = c - o

    features[f"{prefix}_last_open"] = clean_number(o)
    features[f"{prefix}_last_high"] = clean_number(h)
    features[f"{prefix}_last_low"] = clean_number(l)
    features[f"{prefix}_last_close"] = clean_number(c)
    features[f"{prefix}_last_range"] = clean_number(rng)
    features[f"{prefix}_last_body"] = clean_number(body)
    features[f"{prefix}_last_body_abs"] = clean_number(abs(body))
    features[f"{prefix}_last_direction"] = float(sign(body))
    features[f"{prefix}_last_upper_wick"] = clean_number(h - max(o, c))
    features[f"{prefix}_last_lower_wick"] = clean_number(min(o, c) - l)
    features[f"{prefix}_last_close_location"] = clean_number((c - l) / rng if rng > 0 else 0.5)


def add_ma_features(features: Dict[str, float], prefix: str, closes: np.ndarray) -> None:
    if len(closes) < 5:
        for k in [
            "sma5", "sma10", "sma20", "sma50",
            "close_minus_sma5", "close_minus_sma10", "close_minus_sma20", "close_minus_sma50",
            "above_sma5", "above_sma10", "above_sma20", "above_sma50",
            "sma5_gt_sma10", "sma10_gt_sma20", "sma20_gt_sma50",
            "ma_bull_alignment", "ma_bear_alignment",
            "sma5_slope", "sma10_slope", "sma20_slope",
        ]:
            features[f"{prefix}_{k}"] = 0.0
        return

    c = float(closes[-1])

    def sma(n: int) -> float:
        if len(closes) < n:
            return float(np.mean(closes))
        return float(np.mean(closes[-n:]))

    sma5 = sma(5)
    sma10 = sma(10)
    sma20 = sma(20)
    sma50 = sma(50)

    def slope(n: int) -> float:
        if len(closes) < n * 2:
            return 0.0
        return float(np.mean(closes[-n:]) - np.mean(closes[-2*n:-n]))

    features[f"{prefix}_sma5"] = clean_number(sma5)
    features[f"{prefix}_sma10"] = clean_number(sma10)
    features[f"{prefix}_sma20"] = clean_number(sma20)
    features[f"{prefix}_sma50"] = clean_number(sma50)

    features[f"{prefix}_close_minus_sma5"] = clean_number(c - sma5)
    features[f"{prefix}_close_minus_sma10"] = clean_number(c - sma10)
    features[f"{prefix}_close_minus_sma20"] = clean_number(c - sma20)
    features[f"{prefix}_close_minus_sma50"] = clean_number(c - sma50)

    features[f"{prefix}_above_sma5"] = 1.0 if c > sma5 else 0.0
    features[f"{prefix}_above_sma10"] = 1.0 if c > sma10 else 0.0
    features[f"{prefix}_above_sma20"] = 1.0 if c > sma20 else 0.0
    features[f"{prefix}_above_sma50"] = 1.0 if c > sma50 else 0.0

    features[f"{prefix}_sma5_gt_sma10"] = 1.0 if sma5 > sma10 else 0.0
    features[f"{prefix}_sma10_gt_sma20"] = 1.0 if sma10 > sma20 else 0.0
    features[f"{prefix}_sma20_gt_sma50"] = 1.0 if sma20 > sma50 else 0.0

    features[f"{prefix}_ma_bull_alignment"] = 1.0 if sma5 > sma10 > sma20 > sma50 else 0.0
    features[f"{prefix}_ma_bear_alignment"] = 1.0 if sma5 < sma10 < sma20 < sma50 else 0.0

    features[f"{prefix}_sma5_slope"] = clean_number(slope(5))
    features[f"{prefix}_sma10_slope"] = clean_number(slope(10))
    features[f"{prefix}_sma20_slope"] = clean_number(slope(20))


def add_round_price_features(features: Dict[str, float], close_price: float) -> None:
    if close_price <= 0:
        features["gold_dist_to_integer"] = 0.0
        features["gold_dist_to_half"] = 0.0
        features["gold_dist_to_10_level"] = 0.0
        features["gold_dist_to_5_level"] = 0.0
        features["gold_integer_zone_flag"] = 0.0
        return

    frac = close_price - math.floor(close_price)
    dist_integer = min(frac, 1.0 - frac)
    dist_half = abs(frac - 0.5)

    mod10 = close_price % 10.0
    dist10 = min(mod10, 10.0 - mod10)

    mod5 = close_price % 5.0
    dist5 = min(mod5, 5.0 - mod5)

    features["gold_dist_to_integer"] = clean_number(dist_integer)
    features["gold_dist_to_half"] = clean_number(dist_half)
    features["gold_dist_to_10_level"] = clean_number(dist10)
    features["gold_dist_to_5_level"] = clean_number(dist5)
    features["gold_integer_zone_flag"] = 1.0 if dist_integer <= 0.10 else 0.0


def compute_tf_features(tf_data: TfArrays, cutoff_epoch: int, windows: List[int]) -> Tuple[Dict[str, float], Dict[str, int]]:
    features: Dict[str, float] = {}

    end = tf_data.end_index_before(cutoff_epoch)
    available = int(end)

    if end <= 0:
        features[f"{tf_data.tf}_available_bars"] = 0.0
        for w in windows:
            add_missing_window_features(features, f"{tf_data.tf}_w{w}")
        add_last_candle_features(features, tf_data.tf, np.array([]), np.array([]), np.array([]), np.array([]))
        add_ma_features(features, tf_data.tf, np.array([]))
        return features, {"available": 0}

    max_w = max(windows + [50])
    start = max(0, end - max_w)

    o_all = tf_data.opens[start:end]
    h_all = tf_data.highs[start:end]
    l_all = tf_data.lows[start:end]
    c_all = tf_data.closes[start:end]
    v_all = tf_data.volumes[start:end]

    features[f"{tf_data.tf}_available_bars"] = float(available)

    add_last_candle_features(features, tf_data.tf, o_all, h_all, l_all, c_all)
    add_ma_features(features, tf_data.tf, c_all)

    for w in windows:
        local_start = max(0, len(c_all) - w)
        add_window_features(
            features,
            f"{tf_data.tf}_w{w}",
            o_all[local_start:],
            h_all[local_start:],
            l_all[local_start:],
            c_all[local_start:],
            v_all[local_start:],
        )

    return features, {"available": available}


def add_cross_features(features: Dict[str, float]) -> None:
    # Momentum alignment across M1/M5/M15.
    m1_mom = features.get("m1_w15_close_change", 0.0)
    m5_mom = features.get("m5_w3_close_change", 0.0)
    m15_body = features.get("m15_last_body", 0.0)

    m1_s = sign(m1_mom)
    m5_s = sign(m5_mom)
    m15_s = sign(m15_body)

    buy_votes = int(m1_s > 0) + int(m5_s > 0) + int(m15_s > 0)
    sell_votes = int(m1_s < 0) + int(m5_s < 0) + int(m15_s < 0)

    features["cross_buy_votes_m1_m5_m15"] = float(buy_votes)
    features["cross_sell_votes_m1_m5_m15"] = float(sell_votes)
    features["cross_direction_balance"] = float(buy_votes - sell_votes)
    features["cross_all_bullish"] = 1.0 if buy_votes == 3 else 0.0
    features["cross_all_bearish"] = 1.0 if sell_votes == 3 else 0.0

    # MA alignment cross-timeframe.
    bull_ma_votes = int(features.get("m1_ma_bull_alignment", 0.0) == 1.0) + int(features.get("m5_ma_bull_alignment", 0.0) == 1.0) + int(features.get("m15_ma_bull_alignment", 0.0) == 1.0)
    bear_ma_votes = int(features.get("m1_ma_bear_alignment", 0.0) == 1.0) + int(features.get("m5_ma_bear_alignment", 0.0) == 1.0) + int(features.get("m15_ma_bear_alignment", 0.0) == 1.0)

    features["cross_bull_ma_votes"] = float(bull_ma_votes)
    features["cross_bear_ma_votes"] = float(bear_ma_votes)
    features["cross_ma_balance"] = float(bull_ma_votes - bear_ma_votes)

    # Volatility relationship.
    m1_range = features.get("m1_w15_avg_range", 0.0)
    m5_range = features.get("m5_w3_avg_range", 0.0)
    m15_range = features.get("m15_last_range", 0.0)

    features["cross_m1_to_m5_range_ratio"] = clean_number(m1_range / m5_range if m5_range > 0 else 0.0)
    features["cross_m5_to_m15_range_ratio"] = clean_number(m5_range / m15_range if m15_range > 0 else 0.0)


def build_feature_doc(
    symbol: str,
    anchor: Dict[str, Any],
    data_by_tf: Dict[str, TfArrays],
) -> Dict[str, Any]:
    anchor_time = parse_dt(anchor["anchor_time"])
    decision_time = parse_dt(anchor["entry_time"])
    cutoff_epoch = epoch_seconds(decision_time)

    features: Dict[str, float] = {}
    source_counts: Dict[str, int] = {}

    for tf in ["m1", "m5", "m15"]:
        tf_features, tf_counts = compute_tf_features(data_by_tf[tf], cutoff_epoch, WINDOWS[tf])
        features.update(tf_features)
        source_counts[tf] = int(tf_counts.get("available", 0))

    # Gold round price features based on the latest closed M15 candle close.
    m15_close = features.get("m15_last_close", 0.0)
    add_round_price_features(features, m15_close)

    add_cross_features(features)

    # Final cleaning to pure Python scalars.
    clean_features: Dict[str, float] = {}
    for key, value in features.items():
        clean_features[key] = clean_number(value)

    now = utc_now()

    return {
        "symbol": symbol,
        "anchor_time": anchor_time,
        "decision_time": decision_time,
        "feature_version": FEATURE_VERSION,
        "model_type": "book_entry",
        "input_timeframes": ["M1", "M5", "M15"],
        "no_leak_cutoff_rule": "raw candle datetime < decision_time",
        "source_counts": source_counts,
        "features": clean_features,
        "feature_count": len(clean_features),
        "created_at": now,
        "updated_at": now,
    }


def make_update_op(symbol: str, doc: Dict[str, Any]) -> UpdateOne:
    set_doc = dict(doc)
    created_at = set_doc.pop("created_at", utc_now())

    return UpdateOne(
        {"symbol": symbol, "anchor_time": doc["anchor_time"]},
        {
            "$set": set_doc,
            "$setOnInsert": {"created_at": created_at},
        },
        upsert=True,
    )


def execute_bulk(db: Database, collection_name: str, ops: List[UpdateOne], report: Dict[str, Any]) -> None:
    if not ops:
        return
    try:
        result = db[collection_name].bulk_write(ops, ordered=False)
        report["counts"]["upserted_or_modified"] += result.upserted_count + result.modified_count
        report["counts"]["inserted"] += result.upserted_count
        report["counts"]["modified"] += result.modified_count
        report["counts"]["matched"] += result.matched_count
    except BulkWriteError as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc.details)[:20000])
        raise


# =========================
# Reports
# =========================

def build_txt_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Book & Quality - Book Entry Features v1 Report")
    lines.append("=" * 74)
    lines.append(f"Run ID              : {report['run_id']}")
    lines.append(f"Status              : {report['status']}")
    lines.append(f"Database            : {report['database']}")
    lines.append(f"Symbol              : {report['symbol']}")
    lines.append(f"Start Time          : {report['start_time']}")
    lines.append(f"End Time            : {report['end_time']}")
    lines.append(f"Duration Sec        : {report['duration_seconds']}")
    lines.append("")
    lines.append("Collections")
    lines.append("-" * 74)
    for k, v in report["collections"].items():
        lines.append(f"{k:<25}: {v}")
    lines.append("")
    lines.append("Args")
    lines.append("-" * 74)
    for k, v in report["args"].items():
        lines.append(f"{k:<25}: {v}")
    lines.append("")
    lines.append("Source Loads")
    lines.append("-" * 74)
    for tf, item in report["source_loads"].items():
        lines.append(f"{tf.upper():<5}: count={item.get('count')} | {item.get('start')} -> {item.get('end')}")
    lines.append("")
    lines.append("Counts")
    lines.append("-" * 74)
    for k, v in report["counts"].items():
        lines.append(f"{k:<25}: {v}")
    lines.append("")
    lines.append("Feature Info")
    lines.append("-" * 74)
    lines.append(f"feature_version          : {report.get('feature_version')}")
    lines.append(f"feature_count_min        : {report.get('feature_count_min')}")
    lines.append(f"feature_count_max        : {report.get('feature_count_max')}")
    lines.append(f"feature_count_last       : {report.get('feature_count_last')}")
    lines.append(f"max_lookback_minutes     : {report.get('max_lookback_minutes')}")

    if report["warnings"]:
        lines.append("")
        lines.append("Warnings")
        lines.append("-" * 74)
        for w in report["warnings"]:
            lines.append(f"- {w}")

    if report["errors"]:
        lines.append("")
        lines.append("Errors")
        lines.append("-" * 74)
        for e in report["errors"][:50]:
            lines.append(f"- {e}")

    return "\n".join(lines)


def save_reports(report: Dict[str, Any], rs: str) -> None:
    reports_dir = get_reports_dir()
    json_path = reports_dir / f"book_entry_features_v1_report_{rs}.json"
    txt_path = reports_dir / f"book_entry_features_v1_report_{rs}.txt"

    report["report_json_path"] = str(json_path)
    report["report_txt_path"] = str(txt_path)

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(build_txt_report(report), encoding="utf-8")


# =========================
# Main
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Book Entry Features v1.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of anchors to process.")
    parser.add_argument("--reset", action="store_true", help="Delete existing target docs before building.")
    parser.add_argument("--start", type=str, default=None, help="Optional anchor_time start, e.g. 2024-01-01 00:00:00")
    parser.add_argument("--end", type=str, default=None, help="Optional anchor_time end, e.g. 2024-12-31 23:59:59")
    parser.add_argument("--skip-existing", action="store_true", help="Skip anchors that already exist in target collection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_dt = utc_now()
    rs = stamp(start_dt)
    run_id = f"book_entry_features_v1_{rs}"

    config = load_config()
    db_name = config["database"]
    symbol = config["symbol"]
    raw = config["raw_collections"]
    bq = config["bq_collections"]

    labels_collection = bq.get("labels_book_entry", "bq_labels_book_entry_m15_v1")
    target_collection = bq.get("features_book_entry", "bq_features_book_entry_m15_v1")

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": run_id,
        "project_root": str(get_project_root()),
        "database": db_name,
        "symbol": symbol,
        "start_time": start_dt.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "status": "running",
        "feature_version": FEATURE_VERSION,
        "max_lookback_minutes": MAX_LOOKBACK_MINUTES,
        "collections": {
            "labels_source": labels_collection,
            "m1_source": raw.get("m1", "xauusd_m1"),
            "m5_source": raw.get("m5", "xauusd_m5"),
            "m15_source": raw.get("m15", "xauusd_m15"),
            "target": target_collection,
        },
        "args": {
            "limit": args.limit,
            "reset": args.reset,
            "start": args.start,
            "end": args.end,
            "skip_existing": args.skip_existing,
        },
        "source_loads": {},
        "counts": {
            "anchors_loaded": 0,
            "anchors_processed": 0,
            "upserted_or_modified": 0,
            "inserted": 0,
            "matched": 0,
            "modified": 0,
            "skipped_existing": 0,
            "row_errors": 0,
        },
        "feature_count_min": None,
        "feature_count_max": None,
        "feature_count_last": None,
        "warnings": [],
        "errors": [],
    }

    print("=== Book & Quality - Build Book Entry Features v1 ===", flush=True)
    print(f"Project Root : {report['project_root']}", flush=True)
    print(f"Database     : {db_name}", flush=True)
    print(f"Symbol       : {symbol}", flush=True)
    print(f"Target       : {target_collection}", flush=True)

    try:
        db = connect(config)

        if args.reset:
            deleted = db[target_collection].delete_many({})
            report["reset_deleted_count"] = deleted.deleted_count
            print(f"[RESET] Deleted {deleted.deleted_count:,} existing feature docs.", flush=True)

        db[target_collection].create_index(
            [("symbol", ASCENDING), ("anchor_time", ASCENDING)],
            unique=True,
            name="uq_symbol_anchor_time",
        )
        db[target_collection].create_index(
            [("feature_version", ASCENDING), ("anchor_time", ASCENDING)],
            name="ix_feature_version_anchor_time",
        )
        db[target_collection].create_index(
            [("feature_version", ASCENDING), ("feature_count", ASCENDING)],
            name="ix_feature_version_feature_count",
        )

        start_arg = parse_dt_arg(args.start)
        end_arg = parse_dt_arg(args.end)

        anchors = load_anchors(
            db=db,
            labels_collection=labels_collection,
            symbol=symbol,
            start_arg=start_arg,
            end_arg=end_arg,
            limit=args.limit,
        )

        report["counts"]["anchors_loaded"] = len(anchors)
        print(f"[LOAD] Anchors loaded from labels: {len(anchors):,}", flush=True)

        if not anchors:
            report["status"] = "failed"
            report["errors"].append("No anchors loaded. Build Book Entry Labels first.")
            return

        min_decision_time = min(x["entry_time"] for x in anchors)
        max_decision_time = max(x["entry_time"] for x in anchors)

        raw_start = min_decision_time - timedelta(minutes=MAX_LOOKBACK_MINUTES + 30)
        raw_end = max_decision_time + timedelta(minutes=1)

        data_by_tf = {
            "m1": load_tf_arrays(db, raw.get("m1", "xauusd_m1"), "m1", raw_start, raw_end, report),
            "m5": load_tf_arrays(db, raw.get("m5", "xauusd_m5"), "m5", raw_start, raw_end, report),
            "m15": load_tf_arrays(db, raw.get("m15", "xauusd_m15"), "m15", raw_start, raw_end, report),
        }

        ops: List[UpdateOne] = []

        for anchor in anchors:
            try:
                anchor_time = anchor["anchor_time"]

                if args.skip_existing:
                    exists = db[target_collection].find_one(
                        {"symbol": symbol, "anchor_time": anchor_time},
                        projection={"_id": 1},
                    )
                    if exists:
                        report["counts"]["skipped_existing"] += 1
                        continue

                doc = build_feature_doc(symbol, anchor, data_by_tf)

                fc = int(doc["feature_count"])
                report["feature_count_last"] = fc
                report["feature_count_min"] = fc if report["feature_count_min"] is None else min(report["feature_count_min"], fc)
                report["feature_count_max"] = fc if report["feature_count_max"] is None else max(report["feature_count_max"], fc)

                ops.append(make_update_op(symbol, doc))
                report["counts"]["anchors_processed"] += 1

                if len(ops) >= BATCH_SIZE:
                    execute_bulk(db, target_collection, ops, report)
                    ops = []
                    print(
                        f"[PROGRESS] processed={report['counts']['anchors_processed']:,} "
                        f"written≈{report['counts']['upserted_or_modified']:,} "
                        f"feature_count={report['feature_count_last']}",
                        flush=True,
                    )

            except Exception as row_exc:
                report["counts"]["row_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(f"anchor_time={anchor.get('anchor_time')} | {row_exc}")

        if ops:
            execute_bulk(db, target_collection, ops, report)

        report["final_target_count"] = db[target_collection].estimated_document_count()

        if report["counts"]["row_errors"] > 0:
            report["warnings"].append("Some row errors occurred. Check JSON report.")

        report["status"] = "success" if report["counts"]["row_errors"] == 0 and not report["errors"] else "success_with_row_errors"

    except Exception as exc:
        report["status"] = "failed"
        if not report["errors"]:
            report["errors"].append(str(exc))
        print(f"[ERROR] {exc}", flush=True)

    finally:
        end_dt = utc_now()
        report["end_time"] = end_dt.isoformat()
        report["duration_seconds"] = round((end_dt - start_dt).total_seconds(), 3)
        save_reports(report, rs)

    print("\n=== Final Summary ===", flush=True)
    print(f"Status              : {report['status']}", flush=True)
    print(f"Anchors Loaded      : {report['counts']['anchors_loaded']:,}", flush=True)
    print(f"Anchors Processed   : {report['counts']['anchors_processed']:,}", flush=True)
    print(f"Upserted/Modified   : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Inserted            : {report['counts']['inserted']:,}", flush=True)
    print(f"Modified            : {report['counts']['modified']:,}", flush=True)
    print(f"Skipped Existing    : {report['counts']['skipped_existing']:,}", flush=True)
    print(f"Row Errors          : {report['counts']['row_errors']:,}", flush=True)
    print(f"Feature Count Min   : {report.get('feature_count_min')}", flush=True)
    print(f"Feature Count Max   : {report.get('feature_count_max')}", flush=True)
    print(f"Feature Count Last  : {report.get('feature_count_last')}", flush=True)
    print(f"JSON Report         : {report.get('report_json_path')}", flush=True)
    print(f"TXT Report          : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
