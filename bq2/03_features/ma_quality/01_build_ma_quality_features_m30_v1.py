# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Build MA Phase & Quality Features M30 v1

Location:
    Book_Quality/03_features/ma_quality/01_build_ma_quality_features_m30_v1.py

Purpose:
    Build no-leak features for the MA Phase & Quality Model.

Anchor:
    M30 closed candle from bq2_labels_ma_quality_m30_v1.
    decision_time = next M30 open after anchor_time.

Inputs:
    bq2_labels_ma_quality_m30_v1
    xauusd_m30
    xauusd_h1
    xauusd_h4

Output:
    bq2_features_ma_quality_m30_v1

No-Leak Rule:
    For every label decision_time, each timeframe uses only candles fully closed
    before decision_time.
    - M30 uses the anchor M30 candle and earlier candles.
    - H1/H4 use only bars closed before decision_time.

Raw data:
    This script does NOT modify xauusd_* raw collections.

TEST ONLY:
    cd C:/Project/Book_Quality/03_features/ma_quality
    python -u 01_build_ma_quality_features_m30_v1.py --limit 1000 --reset

FULL BUILD:
    python -u 01_build_ma_quality_features_m30_v1.py --reset
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError


SCRIPT_NAME = "01_build_ma_quality_features_m30_v1.py"
FEATURE_VERSION = "ma_quality_features_m30_v1"
LABEL_VERSION_EXPECTED = "ma_quality_label_m30_v1"
BATCH_SIZE = 1000

TIME_FIELD = "datetime"
OPEN_FIELD = "open"
HIGH_FIELD = "high"
LOW_FIELD = "low"
CLOSE_FIELD = "close"

TF_MINUTES = {"m30": 30, "h1": 60, "h4": 240}
MA_PERIODS = [20, 50, 100]

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "raw_collections": {
        "m30": "xauusd_m30",
        "h1": "xauusd_h1",
        "h4": "xauusd_h4",
    },
    "bq2_collections": {
        "labels_ma_quality": "bq2_labels_ma_quality_m30_v1",
        "features_ma_quality": "bq2_features_ma_quality_m30_v1",
    },
    "ma_quality_features_m30_v1": {
        "min_history_bars": {"m30": 140, "h1": 140, "h4": 140},
        "ma_periods": [20, 50, 100],
        "candle_windows": [3, 5, 10, 20, 50],
        "slope_lookbacks": [1, 3, 5, 10],
        "range_period": 20,
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
        print(f"[WARN] Could not read {path}: {exc}", flush=True)
        return None


def deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG
    root = project_root()

    bq2_cfg = read_json(root / "00_config" / "bq2_config.json")
    if bq2_cfg:
        cfg = deep_merge(cfg, bq2_cfg)

    # Force M30 Quality collections for this script, even if an old config still exists.
    cfg.setdefault("bq2_collections", {})
    cfg["bq2_collections"]["labels_ma_quality"] = "bq2_labels_ma_quality_m30_v1"
    cfg["bq2_collections"]["features_ma_quality"] = "bq2_features_ma_quality_m30_v1"

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


def parse_dt_arg(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=None)


def epoch(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def sf(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def cn(v: Any, digits: int = 10) -> float:
    x = sf(v)
    return float(round(x, digits))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or math.isnan(b) or math.isinf(b):
        return default
    return a / b


class CandleArrays:
    def __init__(self, tf: str, minutes: int, times, opens, highs, lows, closes):
        self.tf = tf
        self.minutes = minutes
        self.seconds = minutes * 60
        self.times = times
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes

    @property
    def count(self) -> int:
        return int(len(self.times))

    def closed_end_index(self, decision_epoch: int) -> int:
        latest_open = decision_epoch - self.seconds
        return int(np.searchsorted(self.times, latest_open, side="right"))


def load_candles(db: Database, coll_name: str, tf: str, report: Dict[str, Any]) -> CandleArrays:
    print(f"[LOAD] {tf.upper()} from {coll_name}", flush=True)

    t = array("q")
    o = array("d")
    h = array("d")
    l = array("d")
    c = array("d")

    cur = (
        db[coll_name]
        .find({}, projection={TIME_FIELD: 1, OPEN_FIELD: 1, HIGH_FIELD: 1, LOW_FIELD: 1, CLOSE_FIELD: 1, "_id": 0})
        .sort(TIME_FIELD, ASCENDING)
        .batch_size(10000)
    )

    first = None
    last = None
    count = 0

    for d in cur:
        dt = parse_dt(d[TIME_FIELD])
        if first is None:
            first = dt
        last = dt

        t.append(epoch(dt))
        o.append(sf(d.get(OPEN_FIELD)))
        h.append(sf(d.get(HIGH_FIELD)))
        l.append(sf(d.get(LOW_FIELD)))
        c.append(sf(d.get(CLOSE_FIELD)))
        count += 1

    arr = CandleArrays(
        tf=tf,
        minutes=TF_MINUTES[tf],
        times=np.frombuffer(t, dtype=np.int64).copy(),
        opens=np.frombuffer(o, dtype=np.float64).copy(),
        highs=np.frombuffer(h, dtype=np.float64).copy(),
        lows=np.frombuffer(l, dtype=np.float64).copy(),
        closes=np.frombuffer(c, dtype=np.float64).copy(),
    )

    report["source_loads"][tf] = {
        "collection": coll_name,
        "count": count,
        "first_time": str(first),
        "last_time": str(last),
    }

    print(f"[LOAD] {tf.upper()} loaded: {count:,}", flush=True)
    return arr


def sma(closes: np.ndarray, end: int, period: int) -> Optional[float]:
    if end < period:
        return None
    return float(np.mean(closes[end - period:end]))


def avg_range(highs: np.ndarray, lows: np.ndarray, end: int, period: int = 20) -> float:
    if end <= 1:
        return 1.0
    start = max(0, end - period)
    r = highs[start:end] - lows[start:end]
    val = float(np.mean(r)) if len(r) else 1.0
    return val if val > 0 else 1.0


def add_feature(features: Dict[str, float], key: str, value: Any) -> None:
    features[key] = cn(value)


def add_binary(features: Dict[str, float], key: str, value: bool) -> None:
    features[key] = 1.0 if value else 0.0


def current_direction_from_features(ma20: float, ma50: float, ma100: float, slope20_r: float) -> str:
    if ma20 > ma50 > ma100 and slope20_r > 0.03:
        return "BUY"
    if ma20 < ma50 < ma100 and slope20_r < -0.03:
        return "SELL"
    if slope20_r > 0.12:
        return "BUY"
    if slope20_r < -0.12:
        return "SELL"
    return "RANGE"


def add_tf_features(
    features: Dict[str, float],
    data: CandleArrays,
    decision_epoch: int,
    windows: List[int],
    slope_lookbacks: List[int],
    range_period: int,
    min_history: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    end = data.closed_end_index(decision_epoch)
    max_need = max(max(windows), 100 + max(slope_lookbacks) + 1, min_history)

    if end < max_need:
        return None, f"{data.tf}:not_enough_history"
    if end > data.count:
        return None, f"{data.tf}:bad_end_index"

    prefix = data.tf
    o = data.opens
    h = data.highs
    l = data.lows
    c = data.closes

    close_now = float(c[end - 1])
    open_now = float(o[end - 1])
    high_now = float(h[end - 1])
    low_now = float(l[end - 1])
    range_now = max(high_now - low_now, 1e-9)
    body_now = close_now - open_now
    abs_body_now = abs(body_now)
    avg_r = avg_range(h, l, end, range_period)
    norm = avg_r if avg_r > 0 else 1.0

    add_feature(features, f"{prefix}_close", close_now)
    add_feature(features, f"{prefix}_range_now_r", range_now / norm)
    add_feature(features, f"{prefix}_body_now_r", body_now / norm)
    add_feature(features, f"{prefix}_abs_body_now_r", abs_body_now / norm)
    add_feature(features, f"{prefix}_upper_shadow_now_r", (high_now - max(open_now, close_now)) / norm)
    add_feature(features, f"{prefix}_lower_shadow_now_r", (min(open_now, close_now) - low_now) / norm)
    add_feature(features, f"{prefix}_close_pos_now", safe_div(close_now - low_now, range_now))
    add_binary(features, f"{prefix}_bull_candle_now", close_now > open_now)
    add_binary(features, f"{prefix}_bear_candle_now", close_now < open_now)

    ma_values: Dict[int, float] = {}
    slope_values: Dict[Tuple[int, int], float] = {}

    for period in MA_PERIODS:
        ma_now = sma(c, end, period)
        if ma_now is None:
            return None, f"{data.tf}:not_enough_ma_{period}"

        ma_values[period] = ma_now
        add_feature(features, f"{prefix}_ma{period}", ma_now)
        add_feature(features, f"{prefix}_price_dist_ma{period}_r", (close_now - ma_now) / norm)
        add_binary(features, f"{prefix}_price_above_ma{period}", close_now > ma_now)

        for lb in slope_lookbacks:
            ma_old = sma(c, end - lb, period)
            if ma_old is None:
                return None, f"{data.tf}:not_enough_slope_ma{period}_lb{lb}"
            slope = ma_now - ma_old
            slope_values[(period, lb)] = slope
            add_feature(features, f"{prefix}_ma{period}_slope_{lb}_r", slope / norm)

        # acceleration based on two equal slope segments
        lb = 5
        if end >= period + lb * 2:
            ma_mid = sma(c, end - lb, period)
            ma_old2 = sma(c, end - lb * 2, period)
            accel = (ma_now - ma_mid) - (ma_mid - ma_old2)
            add_feature(features, f"{prefix}_ma{period}_accel_5_r", accel / norm)
        else:
            add_feature(features, f"{prefix}_ma{period}_accel_5_r", 0.0)

    ma20 = ma_values[20]
    ma50 = ma_values[50]
    ma100 = ma_values[100]

    spread_20_50 = ma20 - ma50
    spread_50_100 = ma50 - ma100
    spread_20_100 = ma20 - ma100

    add_feature(features, f"{prefix}_ma20_50_spread_r", spread_20_50 / norm)
    add_feature(features, f"{prefix}_ma50_100_spread_r", spread_50_100 / norm)
    add_feature(features, f"{prefix}_ma20_100_spread_r", spread_20_100 / norm)
    add_feature(features, f"{prefix}_ma_compression_20_100", abs(spread_20_100) / norm)

    add_binary(features, f"{prefix}_ma20_above_ma50", ma20 > ma50)
    add_binary(features, f"{prefix}_ma50_above_ma100", ma50 > ma100)
    add_binary(features, f"{prefix}_ma_order_bull", ma20 > ma50 > ma100)
    add_binary(features, f"{prefix}_ma_order_bear", ma20 < ma50 < ma100)

    slope20_r = slope_values[(20, 5)] / norm if (20, 5) in slope_values else 0.0
    direction = current_direction_from_features(ma20, ma50, ma100, slope20_r)

    add_binary(features, f"{prefix}_current_dir_buy", direction == "BUY")
    add_binary(features, f"{prefix}_current_dir_sell", direction == "SELL")
    add_binary(features, f"{prefix}_current_dir_range", direction == "RANGE")

    # Candle aggregate features
    for w in windows:
        start = end - w
        opens_w = o[start:end]
        highs_w = h[start:end]
        lows_w = l[start:end]
        closes_w = c[start:end]

        ret = float(closes_w[-1] - opens_w[0])
        hh = float(np.max(highs_w))
        ll = float(np.min(lows_w))
        width = max(hh - ll, 1e-9)

        bodies = closes_w - opens_w
        ranges = highs_w - lows_w
        bullish = int(np.sum(closes_w > opens_w))
        bearish = int(np.sum(closes_w < opens_w))

        add_feature(features, f"{prefix}_ret_{w}_r", ret / norm)
        add_feature(features, f"{prefix}_hhll_width_{w}_r", width / norm)
        add_feature(features, f"{prefix}_close_pos_{w}", safe_div(close_now - ll, width))
        add_feature(features, f"{prefix}_bull_ratio_{w}", bullish / w)
        add_feature(features, f"{prefix}_bear_ratio_{w}", bearish / w)
        add_feature(features, f"{prefix}_avg_body_{w}_r", float(np.mean(bodies)) / norm)
        add_feature(features, f"{prefix}_avg_abs_body_{w}_r", float(np.mean(np.abs(bodies))) / norm)
        add_feature(features, f"{prefix}_avg_range_{w}_r", float(np.mean(ranges)) / norm)
        add_feature(features, f"{prefix}_volatility_{w}_r", float(np.std(closes_w)) / norm)

    # Swing position on recent windows
    for w in [20, 50]:
        start = end - w
        hh = float(np.max(h[start:end]))
        ll = float(np.min(l[start:end]))
        width = max(hh - ll, 1e-9)
        add_feature(features, f"{prefix}_swing_pos_{w}", safe_div(close_now - ll, width))
        add_feature(features, f"{prefix}_dist_swing_high_{w}_r", (hh - close_now) / norm)
        add_feature(features, f"{prefix}_dist_swing_low_{w}_r", (close_now - ll) / norm)

    meta = {
        "end_index": end,
        "last_closed_time_epoch": int(data.times[end - 1]),
        "last_closed_time": datetime.fromtimestamp(int(data.times[end - 1]), tz=timezone.utc).replace(tzinfo=None),
        "close_now": close_now,
        "avg_range": norm,
        "ma20": ma20,
        "ma50": ma50,
        "ma100": ma100,
        "slope20_5_r": slope20_r,
        "current_direction": direction,
    }
    return meta, None


def add_cross_tf_features(features: Dict[str, float], meta: Dict[str, Dict[str, Any]]) -> None:
    dirs = {tf: meta[tf]["current_direction"] for tf in ["m30", "h1", "h4"]}

    add_binary(features, "cross_m30_h1_same_buy", dirs["m30"] == "BUY" and dirs["h1"] == "BUY")
    add_binary(features, "cross_m30_h1_same_sell", dirs["m30"] == "SELL" and dirs["h1"] == "SELL")
    add_binary(features, "cross_m30_h4_same_buy", dirs["m30"] == "BUY" and dirs["h4"] == "BUY")
    add_binary(features, "cross_m30_h4_same_sell", dirs["m30"] == "SELL" and dirs["h4"] == "SELL")
    add_binary(features, "cross_all_buy", dirs["m30"] == dirs["h1"] == dirs["h4"] == "BUY")
    add_binary(features, "cross_all_sell", dirs["m30"] == dirs["h1"] == dirs["h4"] == "SELL")
    add_binary(features, "cross_any_range", any(d == "RANGE" for d in dirs.values()))

    buy_count = sum(1 for d in dirs.values() if d == "BUY")
    sell_count = sum(1 for d in dirs.values() if d == "SELL")
    range_count = sum(1 for d in dirs.values() if d == "RANGE")
    add_feature(features, "cross_buy_dir_count", buy_count)
    add_feature(features, "cross_sell_dir_count", sell_count)
    add_feature(features, "cross_range_dir_count", range_count)

    # Slope relationships
    s_m30 = meta["m30"]["slope20_5_r"]
    s_h1 = meta["h1"]["slope20_5_r"]
    s_h4 = meta["h4"]["slope20_5_r"]

    add_feature(features, "cross_slope_m30_minus_h1", s_m30 - s_h1)
    add_feature(features, "cross_slope_h1_minus_h4", s_h1 - s_h4)
    add_feature(features, "cross_slope_m30_minus_h4", s_m30 - s_h4)

    add_binary(features, "cross_m30_turning_up_vs_h1_down", s_m30 > 0 and s_h1 < 0)
    add_binary(features, "cross_m30_turning_down_vs_h1_up", s_m30 < 0 and s_h1 > 0)
    add_binary(features, "cross_m30_up_h1_h4_not_up", dirs["m30"] == "BUY" and not (dirs["h1"] == "BUY" and dirs["h4"] == "BUY"))
    add_binary(features, "cross_m30_down_h1_h4_not_down", dirs["m30"] == "SELL" and not (dirs["h1"] == "SELL" and dirs["h4"] == "SELL"))


def load_labels(db: Database, coll_name: str, args: argparse.Namespace, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {"label_version": LABEL_VERSION_EXPECTED}

    time_filter: Dict[str, Any] = {}
    start_dt = parse_dt_arg(args.start)
    end_dt = parse_dt_arg(args.end)
    if start_dt:
        time_filter["$gte"] = start_dt
    if end_dt:
        time_filter["$lte"] = end_dt
    if time_filter:
        query["anchor_time"] = time_filter

    projection = {
        "_id": 0,
        "symbol": 1,
        "anchor_time": 1,
        "decision_time": 1,
        "entry_time": 1,
        "entry_price": 1,
        "label_version": 1,
        "y": 1,
    }

    cur = (
        db[coll_name]
        .find(query, projection=projection)
        .sort("anchor_time", ASCENDING)
        .batch_size(10000)
    )

    labels: List[Dict[str, Any]] = []
    for doc in cur:
        labels.append(doc)
        if args.limit is not None and len(labels) >= args.limit:
            break

    report["counts"]["labels_loaded"] = len(labels)
    return labels


def make_update_op(symbol: str, doc: Dict[str, Any]) -> UpdateOne:
    set_doc = dict(doc)
    created_at = set_doc.pop("created_at", utc_now())
    return UpdateOne(
        {"symbol": symbol, "anchor_time": doc["anchor_time"]},
        {"$set": set_doc, "$setOnInsert": {"created_at": created_at}},
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


def build_txt_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("Book & Quality v2 - MA Quality Features M30 v1 Report")
    lines.append("=" * 74)
    for k in ["run_id", "status", "database", "symbol", "start_time", "end_time", "duration_seconds"]:
        lines.append(f"{k:<30}: {report.get(k)}")

    lines.append("")
    lines.append("Collections")
    lines.append("-" * 74)
    for k, v in report["collections"].items():
        lines.append(f"{k:<30}: {v}")

    lines.append("")
    lines.append("Config")
    lines.append("-" * 74)
    for k, v in report["feature_config"].items():
        lines.append(f"{k:<30}: {v}")

    lines.append("")
    lines.append("Counts")
    lines.append("-" * 74)
    for k, v in report["counts"].items():
        lines.append(f"{k:<30}: {v}")

    lines.append("")
    lines.append("Feature Count")
    lines.append("-" * 74)
    for k, v in report["feature_count"].items():
        lines.append(f"{k:<30}: {v}")

    if report.get("skip_reasons"):
        lines.append("")
        lines.append("Skip Reasons")
        lines.append("-" * 74)
        for k, v in report["skip_reasons"].items():
            lines.append(f"{k:<30}: {v}")

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
    rd = reports_dir()
    jp = rd / f"ma_quality_features_m30_v1_report_{rs}.json"
    tp = rd / f"ma_quality_features_m30_v1_report_{rs}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tp.write_text(build_txt_report(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build bq2 MA Quality Features M30 v1.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    rs = stamp(started)

    cfg = load_config()
    db_name = cfg["database"]
    symbol = cfg["symbol"]
    raw = cfg["raw_collections"]
    bq2 = cfg["bq2_collections"]
    fcfg = cfg.get("ma_quality_features_m30_v1", DEFAULT_CONFIG["ma_quality_features_m30_v1"])

    labels_coll = bq2.get("labels_ma_quality", "bq2_labels_ma_quality_m30_v1")
    target_coll = bq2.get("features_ma_quality", "bq2_features_ma_quality_m30_v1")

    windows = list(map(int, fcfg.get("candle_windows", [3, 5, 10, 20, 50])))
    slope_lookbacks = list(map(int, fcfg.get("slope_lookbacks", [1, 3, 5, 10])))
    range_period = int(fcfg.get("range_period", 20))
    min_history = fcfg.get("min_history_bars", {"m30": 140, "h1": 140, "h4": 140})

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"ma_quality_features_m30_v1_{rs}",
        "status": "running",
        "database": db_name,
        "symbol": symbol,
        "start_time": started.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "collections": {
            "labels_source": labels_coll,
            "m30_source": raw.get("m30", "xauusd_m30"),
            "h1_source": raw.get("h1", "xauusd_h1"),
            "h4_source": raw.get("h4", "xauusd_h4"),
            "target": target_coll,
        },
        "feature_config": {
            "feature_version": FEATURE_VERSION,
            "label_version_expected": LABEL_VERSION_EXPECTED,
            "anchor_timeframe": "M30",
            "input_timeframes": ["M30", "H1", "H4"],
            "ma_periods": MA_PERIODS,
            "candle_windows": windows,
            "slope_lookbacks": slope_lookbacks,
            "range_period": range_period,
            "min_history_bars": min_history,
        },
        "source_loads": {},
        "counts": {
            "labels_loaded": 0,
            "labels_processed": 0,
            "upserted_or_modified": 0,
            "inserted": 0,
            "matched": 0,
            "modified": 0,
            "skipped_existing": 0,
            "skipped_no_context": 0,
            "skipped_no_label_y": 0,
            "row_errors": 0,
        },
        "feature_count": {
            "min": None,
            "max": None,
            "last": None,
            "distinct": {},
        },
        "skip_reasons": {},
        "warnings": [],
        "errors": [],
        "args": {
            "limit": args.limit,
            "reset": args.reset,
            "start": args.start,
            "end": args.end,
            "skip_existing": args.skip_existing,
        },
    }

    print("=== Book & Quality v2 - Build MA Quality Features M30 v1 ===", flush=True)
    print(f"Database : {db_name}", flush=True)
    print(f"Symbol   : {symbol}", flush=True)
    print(f"Labels   : {labels_coll}", flush=True)
    print(f"Target   : {target_coll}", flush=True)

    try:
        db = connect(cfg)

        if args.reset:
            deleted = db[target_coll].delete_many({})
            print(f"[RESET] Deleted {deleted.deleted_count:,} docs from {target_coll}", flush=True)
            report["reset_deleted_count"] = deleted.deleted_count

        db[target_coll].create_index([("symbol", ASCENDING), ("anchor_time", ASCENDING)], unique=True, name="uq_symbol_anchor_time")
        db[target_coll].create_index([("feature_version", ASCENDING), ("anchor_time", ASCENDING)], name="ix_feature_version_anchor_time")
        db[target_coll].create_index([("label_version", ASCENDING), ("anchor_time", ASCENDING)], name="ix_label_version_anchor_time")
        db[target_coll].create_index([("feature_count", ASCENDING)], name="ix_feature_count")
        db[target_coll].create_index([("y.predicted_direction", ASCENDING), ("anchor_time", ASCENDING)], name="ix_y_direction_anchor_time")

        data = {
            "m30": load_candles(db, raw.get("m30", "xauusd_m30"), "m30", report),
            "h1": load_candles(db, raw.get("h1", "xauusd_h1"), "h1", report),
            "h4": load_candles(db, raw.get("h4", "xauusd_h4"), "h4", report),
        }

        labels = load_labels(db, labels_coll, args, report)
        print(f"[LOAD] Labels loaded: {len(labels):,}", flush=True)

        ops: List[UpdateOne] = []

        for idx, label in enumerate(labels):
            try:
                y = label.get("y")
                if not isinstance(y, dict) or not y:
                    report["counts"]["skipped_no_label_y"] += 1
                    continue

                anchor_time = parse_dt(label["anchor_time"])
                decision_time = parse_dt(label["decision_time"])
                decision_epoch = epoch(decision_time)

                if args.skip_existing:
                    exists = db[target_coll].find_one({"symbol": symbol, "anchor_time": anchor_time}, projection={"_id": 1})
                    if exists:
                        report["counts"]["skipped_existing"] += 1
                        continue

                features: Dict[str, float] = {}
                meta: Dict[str, Dict[str, Any]] = {}
                err = None

                for tf in ["m30", "h1", "h4"]:
                    tf_meta, tf_err = add_tf_features(
                        features=features,
                        data=data[tf],
                        decision_epoch=decision_epoch,
                        windows=windows,
                        slope_lookbacks=slope_lookbacks,
                        range_period=range_period,
                        min_history=int(min_history.get(tf, 140)),
                    )
                    if tf_err:
                        err = tf_err
                        break
                    meta[tf] = tf_meta

                if err:
                    report["counts"]["skipped_no_context"] += 1
                    report["skip_reasons"][err] = report["skip_reasons"].get(err, 0) + 1
                    continue

                add_cross_tf_features(features, meta)

                feature_count = len(features)
                fmin = report["feature_count"]["min"]
                fmax = report["feature_count"]["max"]
                report["feature_count"]["min"] = feature_count if fmin is None else min(fmin, feature_count)
                report["feature_count"]["max"] = feature_count if fmax is None else max(fmax, feature_count)
                report["feature_count"]["last"] = feature_count
                report["feature_count"]["distinct"][str(feature_count)] = report["feature_count"]["distinct"].get(str(feature_count), 0) + 1

                max_feature_time = max(meta[tf]["last_closed_time"] for tf in ["m30", "h1", "h4"])
                no_leak_ok = max_feature_time < decision_time

                if not no_leak_ok:
                    raise ValueError(f"No-leak violation: max_feature_time={max_feature_time}, decision_time={decision_time}")

                now = utc_now()
                doc = {
                    "symbol": symbol,
                    "anchor_time": anchor_time,
                    "decision_time": decision_time,
                    "entry_time": parse_dt(label.get("entry_time", decision_time)),
                    "entry_price": sf(label.get("entry_price", 0.0)),
                    "feature_version": FEATURE_VERSION,
                    "label_version": label.get("label_version", LABEL_VERSION_EXPECTED),
                    "model_type": "ma_phase_quality",
                    "anchor_timeframe": "M30",
                    "input_timeframes": ["M30", "H1", "H4"],
                    "features": features,
                    "feature_count": feature_count,
                    "feature_meta": {
                        "no_leak_ok": no_leak_ok,
                        "max_feature_time": max_feature_time,
                        "tf_last_closed_time": {
                            "m30": meta["m30"]["last_closed_time"],
                            "h1": meta["h1"]["last_closed_time"],
                            "h4": meta["h4"]["last_closed_time"],
                        },
                        "tf_current_direction": {
                            "m30": meta["m30"]["current_direction"],
                            "h1": meta["h1"]["current_direction"],
                            "h4": meta["h4"]["current_direction"],
                        },
                    },
                    "y": y,
                    "created_at": now,
                    "updated_at": now,
                }

                ops.append(make_update_op(symbol, doc))
                report["counts"]["labels_processed"] += 1

                if len(ops) >= BATCH_SIZE:
                    execute_bulk(db, target_coll, ops, report)
                    ops = []
                    print(
                        f"[PROGRESS] labels={report['counts']['labels_loaded']:,} "
                        f"processed={report['counts']['labels_processed']:,} "
                        f"written≈{report['counts']['upserted_or_modified']:,} "
                        f"feature_count_last={report['feature_count']['last']}",
                        flush=True,
                    )

            except Exception as row_exc:
                report["counts"]["row_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(f"idx={idx} anchor={label.get('anchor_time')} | {row_exc}")

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
    print(f"Status              : {report['status']}", flush=True)
    print(f"Labels Loaded       : {report['counts']['labels_loaded']:,}", flush=True)
    print(f"Labels Processed    : {report['counts']['labels_processed']:,}", flush=True)
    print(f"Upserted/Modified   : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Skipped No Context  : {report['counts']['skipped_no_context']:,}", flush=True)
    print(f"Row Errors          : {report['counts']['row_errors']:,}", flush=True)
    print(f"Feature Count       : {report['feature_count']}", flush=True)
    print(f"JSON Report         : {report.get('report_json_path')}", flush=True)
    print(f"TXT Report          : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
