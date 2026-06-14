# -*- coding: utf-8 -*-
"""
Book & Quality - Build Trade Quality Features v2 - MAX M30

Location:
    Book_Quality/03_features/trade_quality/06_build_trade_quality_features_v1.py

Purpose:
    Build no-leak features for Trade Quality Model without using timeframes above M30.

Inputs:
    bq_labels_trade_quality_m15_v1
    xauusd_m15
    xauusd_m30

Output:
    bq_features_trade_quality_m15_v1

Important change:
    H1 and H4 are NOT used.
    Maximum timeframe is M30.

No-leak rule:
    A candle is usable only if:
        candle_open_time + timeframe_duration <= decision_time
    where decision_time = label.entry_time.

Reports:
    Book_Quality/03_features/reports/trade_quality_features_v1_report_YYYYMMDD_HHMMSS.json
    Book_Quality/03_features/reports/trade_quality_features_v1_report_YYYYMMDD_HHMMSS.txt

Requirements:
    pip install pymongo numpy

TEST ONLY:
    cd C:\Project\Book_Quality\03_features\trade_quality
    python -u 06_build_trade_quality_features_v1.py --limit 1000 --reset

FULL BUILD:
    python -u 06_build_trade_quality_features_v1.py --reset
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


SCRIPT_NAME = "06_build_trade_quality_features_v1.py"
FEATURE_VERSION = "trade_quality_features_v1_max_m30"

TIME_FIELD = "datetime"
OPEN_FIELD = "open"
HIGH_FIELD = "high"
LOW_FIELD = "low"
CLOSE_FIELD = "close"
VOLUME_CANDIDATES = ["volume", "tick_volume", "Volume", "TickVolume", "vol"]

BATCH_SIZE = 1000

TF_MINUTES = {"m15": 15, "m30": 30}
WINDOWS = {
    "m15": [2, 4, 8, 16, 32, 64],
    "m30": [2, 4, 8, 16, 32, 64],
}
MAX_LOOKBACK_MINUTES = max(max(WINDOWS[tf]) * TF_MINUTES[tf] for tf in WINDOWS)

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "raw_collections": {
        "m15": "xauusd_m15",
        "m30": "xauusd_m30"
    },
    "bq_collections": {
        "labels_trade_quality": "bq_labels_trade_quality_m15_v1",
        "features_trade_quality": "bq_features_trade_quality_m15_v1"
    }
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
    file_cfg = read_json(project_root() / "00_config" / "mongo_config.json")
    if file_cfg:
        cfg = deep_merge(cfg, file_cfg)

    # Force the corrected max-M30 scope even if config still contains H1/H4.
    cfg["raw_collections"]["m15"] = cfg["raw_collections"].get("m15", "xauusd_m15")
    cfg["raw_collections"]["m30"] = cfg["raw_collections"].get("m30", "xauusd_m30")
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
    raise ValueError(f"Unsupported datetime value: {v!r}")


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


def cn(v: Any) -> float:
    return float(round(sf(v), 10))


def sg(v: float) -> int:
    return 1 if v > 0 else -1 if v < 0 else 0


class TF:
    def __init__(self, name: str, minutes: int, times, opens, highs, lows, closes, volumes):
        self.name = name
        self.minutes = minutes
        self.seconds = minutes * 60
        self.times = times
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.volumes = volumes

    def end_closed_by(self, decision_epoch: int) -> int:
        latest_allowed_open = decision_epoch - self.seconds
        return int(np.searchsorted(self.times, latest_allowed_open, side="right"))


def volume_field(sample: Dict[str, Any]) -> Optional[str]:
    for f in VOLUME_CANDIDATES:
        if f in sample:
            return f
    return None


def load_tf(db: Database, coll: str, tf: str, start: datetime, end: datetime, report: Dict[str, Any]) -> TF:
    print(f"[LOAD] {tf.upper()} {coll}: {start} -> {end}", flush=True)

    sample = db[coll].find_one()
    if not sample:
        raise RuntimeError(f"Empty collection: {coll}")

    vf = volume_field(sample)
    proj = {TIME_FIELD: 1, OPEN_FIELD: 1, HIGH_FIELD: 1, LOW_FIELD: 1, CLOSE_FIELD: 1, "_id": 0}
    if vf:
        proj[vf] = 1

    t = array("q")
    o = array("d")
    h = array("d")
    l = array("d")
    c = array("d")
    v = array("d")

    cur = db[coll].find(
        {TIME_FIELD: {"$gte": start, "$lt": end}},
        projection=proj
    ).sort(TIME_FIELD, ASCENDING).batch_size(10000)

    count = 0
    for d in cur:
        dt = parse_dt(d[TIME_FIELD])
        t.append(epoch(dt))
        o.append(sf(d.get(OPEN_FIELD)))
        h.append(sf(d.get(HIGH_FIELD)))
        l.append(sf(d.get(LOW_FIELD)))
        c.append(sf(d.get(CLOSE_FIELD)))
        v.append(sf(d.get(vf)) if vf else 0.0)
        count += 1

    report["source_loads"][tf] = {
        "collection": coll,
        "count": count,
        "start": start.isoformat(sep=" "),
        "end": end.isoformat(sep=" "),
        "volume_field": vf,
        "timeframe_minutes": TF_MINUTES[tf],
    }

    print(f"[LOAD] {tf.upper()} loaded: {count:,}", flush=True)

    return TF(
        tf, TF_MINUTES[tf],
        np.frombuffer(t, dtype=np.int64).copy(),
        np.frombuffer(o, dtype=np.float64).copy(),
        np.frombuffer(h, dtype=np.float64).copy(),
        np.frombuffer(l, dtype=np.float64).copy(),
        np.frombuffer(c, dtype=np.float64).copy(),
        np.frombuffer(v, dtype=np.float64).copy()
    )


def load_anchors(db: Database, coll: str, symbol: str, start: Optional[datetime], end: Optional[datetime], limit: Optional[int]) -> List[Dict[str, Any]]:
    q: Dict[str, Any] = {"symbol": symbol}
    if start or end:
        q["anchor_time"] = {}
        if start:
            q["anchor_time"]["$gte"] = start
        if end:
            q["anchor_time"]["$lte"] = end

    cur = db[coll].find(q, projection={"anchor_time": 1, "entry_time": 1, "_id": 0}).sort("anchor_time", ASCENDING)
    if limit is not None:
        cur = cur.limit(limit)

    rows = list(cur)
    for r in rows:
        r["anchor_time"] = parse_dt(r["anchor_time"])
        r["entry_time"] = parse_dt(r["entry_time"])
    return rows


def add_empty_win(feat: Dict[str, float], p: str) -> None:
    for k in [
        "valid", "count", "close_change", "close_change_r", "range_total",
        "avg_range", "max_range", "std_range", "avg_body", "avg_abs_body",
        "body_to_range_avg", "bull_ratio", "bear_ratio", "upper_wick_avg",
        "lower_wick_avg", "close_location_avg", "close_diff_std",
        "volume_sum", "volume_avg", "break_high_count", "break_low_count",
        "inside_bar_ratio", "outside_bar_ratio", "long_body_ratio"
    ]:
        feat[f"{p}_{k}"] = 0.0


def add_win(feat: Dict[str, float], p: str, o, h, l, c, v) -> None:
    n = len(c)
    if n < 1:
        add_empty_win(feat, p)
        return

    rng = h - l
    body = c - o
    abs_body = np.abs(body)
    safe_rng = np.where(rng == 0, 1.0, rng)

    total = float(np.max(h) - np.min(l))
    close_change = float(c[-1] - c[0]) if n > 1 else float(c[-1] - o[-1])
    denom = total if total > 0 else 1.0

    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    loc = (c - l) / safe_rng

    bh = bl = inside = outside = 0
    if n > 1:
        bh = int(np.sum(h[1:] > h[:-1]))
        bl = int(np.sum(l[1:] < l[:-1]))
        inside = int(np.sum((h[1:] <= h[:-1]) & (l[1:] >= l[:-1])))
        outside = int(np.sum((h[1:] >= h[:-1]) & (l[1:] <= l[:-1])))

    btr = abs_body / safe_rng

    feat[f"{p}_valid"] = 1.0
    feat[f"{p}_count"] = float(n)
    feat[f"{p}_close_change"] = cn(close_change)
    feat[f"{p}_close_change_r"] = cn(close_change / denom)
    feat[f"{p}_range_total"] = cn(total)
    feat[f"{p}_avg_range"] = cn(np.mean(rng))
    feat[f"{p}_max_range"] = cn(np.max(rng))
    feat[f"{p}_std_range"] = cn(np.std(rng))
    feat[f"{p}_avg_body"] = cn(np.mean(body))
    feat[f"{p}_avg_abs_body"] = cn(np.mean(abs_body))
    feat[f"{p}_body_to_range_avg"] = cn(np.mean(btr))
    feat[f"{p}_bull_ratio"] = cn(np.mean(c > o))
    feat[f"{p}_bear_ratio"] = cn(np.mean(c < o))
    feat[f"{p}_upper_wick_avg"] = cn(np.mean(upper))
    feat[f"{p}_lower_wick_avg"] = cn(np.mean(lower))
    feat[f"{p}_close_location_avg"] = cn(np.mean(loc))
    feat[f"{p}_close_diff_std"] = cn(np.std(np.diff(c)) if n > 1 else 0.0)
    feat[f"{p}_volume_sum"] = cn(np.sum(v))
    feat[f"{p}_volume_avg"] = cn(np.mean(v))
    feat[f"{p}_break_high_count"] = float(bh)
    feat[f"{p}_break_low_count"] = float(bl)
    feat[f"{p}_inside_bar_ratio"] = cn(inside / max(1, n - 1))
    feat[f"{p}_outside_bar_ratio"] = cn(outside / max(1, n - 1))
    feat[f"{p}_long_body_ratio"] = cn(np.mean(btr >= 0.65))


def add_last(feat: Dict[str, float], p: str, o, h, l, c) -> None:
    names = ["open", "high", "low", "close", "range", "body", "abs_body", "direction", "upper_wick", "lower_wick", "close_location", "full_body", "doji"]
    if len(c) < 1:
        for n in names:
            feat[f"{p}_last_{n}"] = 0.0
        return

    oo = float(o[-1]); hh = float(h[-1]); ll = float(l[-1]); cc = float(c[-1])
    rng = hh - ll
    body = cc - oo
    abs_body = abs(body)

    feat[f"{p}_last_open"] = cn(oo)
    feat[f"{p}_last_high"] = cn(hh)
    feat[f"{p}_last_low"] = cn(ll)
    feat[f"{p}_last_close"] = cn(cc)
    feat[f"{p}_last_range"] = cn(rng)
    feat[f"{p}_last_body"] = cn(body)
    feat[f"{p}_last_abs_body"] = cn(abs_body)
    feat[f"{p}_last_direction"] = float(sg(body))
    feat[f"{p}_last_upper_wick"] = cn(hh - max(oo, cc))
    feat[f"{p}_last_lower_wick"] = cn(min(oo, cc) - ll)
    feat[f"{p}_last_close_location"] = cn((cc - ll) / rng if rng > 0 else 0.5)
    feat[f"{p}_last_full_body"] = 1.0 if rng > 0 and abs_body >= 0.7 * rng else 0.0
    feat[f"{p}_last_doji"] = 1.0 if rng > 0 and abs_body <= 0.15 * rng else 0.0


def add_ma(feat: Dict[str, float], p: str, c) -> None:
    names = [
        "sma3", "sma5", "sma10", "sma20", "sma50",
        "close_minus_sma3", "close_minus_sma5", "close_minus_sma10", "close_minus_sma20", "close_minus_sma50",
        "above_sma3", "above_sma5", "above_sma10", "above_sma20", "above_sma50",
        "sma3_gt_sma5", "sma5_gt_sma10", "sma10_gt_sma20", "sma20_gt_sma50",
        "ma_bull_alignment", "ma_bear_alignment",
        "sma3_slope", "sma5_slope", "sma10_slope", "sma20_slope",
        "ema12", "ema26", "macd_like"
    ]
    if len(c) < 3:
        for n in names:
            feat[f"{p}_{n}"] = 0.0
        return

    last = float(c[-1])

    def sma(n: int) -> float:
        return float(np.mean(c[-n:])) if len(c) >= n else float(np.mean(c))

    def slope(n: int) -> float:
        if len(c) < 2 * n:
            return 0.0
        return float(np.mean(c[-n:]) - np.mean(c[-2*n:-n]))

    def ema(n: int) -> float:
        x = c[max(0, len(c) - 4*n):]
        alpha = 2.0 / (n + 1)
        e = float(x[0])
        for value in x[1:]:
            e = alpha * float(value) + (1.0 - alpha) * e
        return e

    sma3, sma5, sma10, sma20, sma50 = sma(3), sma(5), sma(10), sma(20), sma(50)
    ema12, ema26 = ema(12), ema(26)

    for n, val in [
        ("sma3", sma3), ("sma5", sma5), ("sma10", sma10), ("sma20", sma20), ("sma50", sma50),
        ("ema12", ema12), ("ema26", ema26), ("macd_like", ema12 - ema26)
    ]:
        feat[f"{p}_{n}"] = cn(val)

    for n, val in [
        ("close_minus_sma3", last-sma3), ("close_minus_sma5", last-sma5),
        ("close_minus_sma10", last-sma10), ("close_minus_sma20", last-sma20),
        ("close_minus_sma50", last-sma50)
    ]:
        feat[f"{p}_{n}"] = cn(val)

    feat[f"{p}_above_sma3"] = 1.0 if last > sma3 else 0.0
    feat[f"{p}_above_sma5"] = 1.0 if last > sma5 else 0.0
    feat[f"{p}_above_sma10"] = 1.0 if last > sma10 else 0.0
    feat[f"{p}_above_sma20"] = 1.0 if last > sma20 else 0.0
    feat[f"{p}_above_sma50"] = 1.0 if last > sma50 else 0.0

    feat[f"{p}_sma3_gt_sma5"] = 1.0 if sma3 > sma5 else 0.0
    feat[f"{p}_sma5_gt_sma10"] = 1.0 if sma5 > sma10 else 0.0
    feat[f"{p}_sma10_gt_sma20"] = 1.0 if sma10 > sma20 else 0.0
    feat[f"{p}_sma20_gt_sma50"] = 1.0 if sma20 > sma50 else 0.0
    feat[f"{p}_ma_bull_alignment"] = 1.0 if sma3 > sma5 > sma10 > sma20 > sma50 else 0.0
    feat[f"{p}_ma_bear_alignment"] = 1.0 if sma3 < sma5 < sma10 < sma20 < sma50 else 0.0

    feat[f"{p}_sma3_slope"] = cn(slope(3))
    feat[f"{p}_sma5_slope"] = cn(slope(5))
    feat[f"{p}_sma10_slope"] = cn(slope(10))
    feat[f"{p}_sma20_slope"] = cn(slope(20))


def add_structure(feat: Dict[str, float], p: str, h, l, c) -> None:
    names = [
        "hh_count_10", "ll_count_10", "dist_high_20", "dist_low_20",
        "channel_pos_20", "dist_high_50", "dist_low_50", "channel_pos_50",
        "swing_bias_20", "compression_ratio"
    ]
    if len(c) < 2:
        for n in names:
            feat[f"{p}_{n}"] = 0.0
        return

    n = len(c)
    last = float(c[-1])

    st10 = max(0, n - 10)
    hh = sum(1 for i in range(st10 + 1, n) if h[i] > h[i-1])
    ll = sum(1 for i in range(st10 + 1, n) if l[i] < l[i-1])

    def ch(w: int):
        s = max(0, n - w)
        hi = float(np.max(h[s:n]))
        lo = float(np.min(l[s:n]))
        width = hi - lo
        pos = (last - lo) / width if width > 0 else 0.5
        return hi, lo, width, pos

    hi20, lo20, w20, pos20 = ch(20)
    hi50, lo50, w50, pos50 = ch(50)
    recent = float(np.max(h[max(0, n-5):n]) - np.min(l[max(0, n-5):n]))

    feat[f"{p}_hh_count_10"] = float(hh)
    feat[f"{p}_ll_count_10"] = float(ll)
    feat[f"{p}_dist_high_20"] = cn(hi20 - last)
    feat[f"{p}_dist_low_20"] = cn(last - lo20)
    feat[f"{p}_channel_pos_20"] = cn(pos20)
    feat[f"{p}_dist_high_50"] = cn(hi50 - last)
    feat[f"{p}_dist_low_50"] = cn(last - lo50)
    feat[f"{p}_channel_pos_50"] = cn(pos50)
    feat[f"{p}_swing_bias_20"] = cn((pos20 - 0.5) * 2.0)
    feat[f"{p}_compression_ratio"] = cn(recent / w20 if w20 > 0 else 0.0)


def tf_features(tf: TF, decision_epoch: int) -> Tuple[Dict[str, float], int]:
    feat: Dict[str, float] = {}
    end = tf.end_closed_by(decision_epoch)
    available = int(end)
    feat[f"{tf.name}_available_bars"] = float(available)

    if end <= 0:
        for w in WINDOWS[tf.name]:
            add_empty_win(feat, f"{tf.name}_w{w}")
        add_last(feat, tf.name, np.array([]), np.array([]), np.array([]), np.array([]))
        add_ma(feat, tf.name, np.array([]))
        add_structure(feat, tf.name, np.array([]), np.array([]), np.array([]))
        return feat, available

    maxw = max(WINDOWS[tf.name] + [64])
    start = max(0, end - maxw)

    o = tf.opens[start:end]
    h = tf.highs[start:end]
    l = tf.lows[start:end]
    c = tf.closes[start:end]
    v = tf.volumes[start:end]

    add_last(feat, tf.name, o, h, l, c)
    add_ma(feat, tf.name, c)
    add_structure(feat, tf.name, h, l, c)

    for w in WINDOWS[tf.name]:
        s = max(0, len(c) - w)
        add_win(feat, f"{tf.name}_w{w}", o[s:], h[s:], l[s:], c[s:], v[s:])

    return feat, available


def add_cross(feat: Dict[str, float]) -> None:
    m15_mom = feat.get("m15_w8_close_change", 0.0)
    m30_mom = feat.get("m30_w8_close_change", 0.0)

    m15_s = sg(m15_mom)
    m30_s = sg(m30_mom)

    buy_votes = int(m15_s > 0) + int(m30_s > 0)
    sell_votes = int(m15_s < 0) + int(m30_s < 0)

    feat["cross_quality_buy_votes_m15_m30"] = float(buy_votes)
    feat["cross_quality_sell_votes_m15_m30"] = float(sell_votes)
    feat["cross_quality_direction_balance"] = float(buy_votes - sell_votes)
    feat["cross_quality_all_bullish"] = 1.0 if buy_votes == 2 else 0.0
    feat["cross_quality_all_bearish"] = 1.0 if sell_votes == 2 else 0.0
    feat["cross_quality_mixed_direction"] = 1.0 if buy_votes > 0 and sell_votes > 0 else 0.0

    bull_ma = int(feat.get("m15_ma_bull_alignment", 0.0) == 1.0) + int(feat.get("m30_ma_bull_alignment", 0.0) == 1.0)
    bear_ma = int(feat.get("m15_ma_bear_alignment", 0.0) == 1.0) + int(feat.get("m30_ma_bear_alignment", 0.0) == 1.0)

    feat["cross_quality_bull_ma_votes"] = float(bull_ma)
    feat["cross_quality_bear_ma_votes"] = float(bear_ma)
    feat["cross_quality_ma_balance"] = float(bull_ma - bear_ma)

    pos15 = feat.get("m15_channel_pos_20", 0.5)
    pos30 = feat.get("m30_channel_pos_20", 0.5)

    feat["cross_quality_avg_channel_pos_20"] = cn((pos15 + pos30) / 2.0)
    feat["cross_quality_channel_pos_dispersion"] = cn(abs(pos15 - pos30))
    feat["cross_quality_near_upper_all"] = 1.0 if pos15 >= 0.75 and pos30 >= 0.75 else 0.0
    feat["cross_quality_near_lower_all"] = 1.0 if pos15 <= 0.25 and pos30 <= 0.25 else 0.0

    m15r = feat.get("m15_w8_avg_range", 0.0)
    m30r = feat.get("m30_w4_avg_range", 0.0)
    feat["cross_quality_m15_to_m30_range_ratio"] = cn(m15r / m30r if m30r > 0 else 0.0)

    comp_votes = int(feat.get("m15_compression_ratio", 0.0) <= 0.35) + int(feat.get("m30_compression_ratio", 0.0) <= 0.35)
    feat["cross_quality_compression_votes"] = float(comp_votes)


def build_doc(symbol: str, anchor: Dict[str, Any], data: Dict[str, TF]) -> Dict[str, Any]:
    anchor_time = parse_dt(anchor["anchor_time"])
    decision_time = parse_dt(anchor["entry_time"])
    decision_epoch = epoch(decision_time)

    features: Dict[str, float] = {}
    source_counts: Dict[str, int] = {}

    for tf in ["m15", "m30"]:
        fpart, available = tf_features(data[tf], decision_epoch)
        features.update(fpart)
        source_counts[tf] = available

    add_cross(features)

    features = {k: cn(v) for k, v in features.items()}
    now = utc_now()

    return {
        "symbol": symbol,
        "anchor_time": anchor_time,
        "decision_time": decision_time,
        "feature_version": FEATURE_VERSION,
        "model_type": "trade_quality",
        "input_timeframes": ["M15", "M30"],
        "max_timeframe": "M30",
        "no_leak_cutoff_rule": "candle open time + timeframe duration <= decision_time",
        "source_counts": source_counts,
        "features": features,
        "feature_count": len(features),
        "created_at": now,
        "updated_at": now,
    }


def update_op(symbol: str, doc: Dict[str, Any]) -> UpdateOne:
    set_doc = dict(doc)
    created_at = set_doc.pop("created_at", utc_now())
    return UpdateOne(
        {"symbol": symbol, "anchor_time": doc["anchor_time"]},
        {"$set": set_doc, "$setOnInsert": {"created_at": created_at}},
        upsert=True,
    )


def bulk_write(db: Database, coll: str, ops: List[UpdateOne], report: Dict[str, Any]) -> None:
    if not ops:
        return
    try:
        res = db[coll].bulk_write(ops, ordered=False)
        report["counts"]["upserted_or_modified"] += res.upserted_count + res.modified_count
        report["counts"]["inserted"] += res.upserted_count
        report["counts"]["modified"] += res.modified_count
        report["counts"]["matched"] += res.matched_count
    except BulkWriteError as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc.details)[:20000])
        raise


def txt_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("Book & Quality - Trade Quality Features v1 MAX M30 Report")
    lines.append("=" * 74)
    for k in ["run_id", "status", "database", "symbol", "start_time", "end_time", "duration_seconds"]:
        lines.append(f"{k:<25}: {report.get(k)}")

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
    for k in ["feature_version", "max_timeframe", "feature_count_min", "feature_count_max", "feature_count_last", "max_lookback_minutes", "no_leak_rule"]:
        lines.append(f"{k:<25}: {report.get(k)}")

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
    jp = rd / f"trade_quality_features_v1_report_{rs}.json"
    tp = rd / f"trade_quality_features_v1_report_{rs}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tp.write_text(txt_report(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Trade Quality Features v1 MAX M30.")
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
    bq = cfg["bq_collections"]

    labels_coll = bq.get("labels_trade_quality", "bq_labels_trade_quality_m15_v1")
    target_coll = bq.get("features_trade_quality", "bq_features_trade_quality_m15_v1")

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"trade_quality_features_v1_max_m30_{rs}",
        "project_root": str(project_root()),
        "database": db_name,
        "symbol": symbol,
        "start_time": started.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "status": "running",
        "feature_version": FEATURE_VERSION,
        "max_timeframe": "M30",
        "max_lookback_minutes": MAX_LOOKBACK_MINUTES,
        "no_leak_rule": "candle open time + timeframe duration <= decision_time",
        "collections": {
            "labels_source": labels_coll,
            "m15_source": raw.get("m15", "xauusd_m15"),
            "m30_source": raw.get("m30", "xauusd_m30"),
            "target": target_coll,
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
        "warnings": [
            "This script intentionally excludes H1 and H4 because max allowed timeframe is M30."
        ],
        "errors": [],
    }

    print("=== Book & Quality - Build Trade Quality Features v1 MAX M30 ===", flush=True)
    print(f"Project Root : {report['project_root']}", flush=True)
    print(f"Database     : {db_name}", flush=True)
    print(f"Symbol       : {symbol}", flush=True)
    print(f"Target       : {target_coll}", flush=True)
    print("Scope        : M15 + M30 only; no H1/H4", flush=True)

    try:
        db = connect(cfg)

        if args.reset:
            deleted = db[target_coll].delete_many({})
            report["reset_deleted_count"] = deleted.deleted_count
            print(f"[RESET] Deleted {deleted.deleted_count:,} existing feature docs.", flush=True)

        db[target_coll].create_index([("symbol", ASCENDING), ("anchor_time", ASCENDING)], unique=True, name="uq_symbol_anchor_time")
        db[target_coll].create_index([("feature_version", ASCENDING), ("anchor_time", ASCENDING)], name="ix_feature_version_anchor_time")
        db[target_coll].create_index([("feature_version", ASCENDING), ("feature_count", ASCENDING)], name="ix_feature_version_feature_count")

        anchors = load_anchors(db, labels_coll, symbol, parse_dt_arg(args.start), parse_dt_arg(args.end), args.limit)
        report["counts"]["anchors_loaded"] = len(anchors)
        print(f"[LOAD] Anchors loaded from labels: {len(anchors):,}", flush=True)

        if not anchors:
            report["status"] = "failed"
            report["errors"].append("No anchors loaded. Build Trade Quality Labels first.")
            return

        min_dt = min(a["entry_time"] for a in anchors)
        max_dt = max(a["entry_time"] for a in anchors)
        raw_start = min_dt - timedelta(minutes=MAX_LOOKBACK_MINUTES + 60)
        raw_end = max_dt + timedelta(minutes=30)

        data = {
            "m15": load_tf(db, raw.get("m15", "xauusd_m15"), "m15", raw_start, raw_end, report),
            "m30": load_tf(db, raw.get("m30", "xauusd_m30"), "m30", raw_start, raw_end, report),
        }

        ops: List[UpdateOne] = []

        for a in anchors:
            try:
                if args.skip_existing:
                    exists = db[target_coll].find_one({"symbol": symbol, "anchor_time": a["anchor_time"]}, projection={"_id": 1})
                    if exists:
                        report["counts"]["skipped_existing"] += 1
                        continue

                doc = build_doc(symbol, a, data)

                fc = int(doc["feature_count"])
                report["feature_count_last"] = fc
                report["feature_count_min"] = fc if report["feature_count_min"] is None else min(report["feature_count_min"], fc)
                report["feature_count_max"] = fc if report["feature_count_max"] is None else max(report["feature_count_max"], fc)

                ops.append(update_op(symbol, doc))
                report["counts"]["anchors_processed"] += 1

                if len(ops) >= BATCH_SIZE:
                    bulk_write(db, target_coll, ops, report)
                    ops = []
                    print(
                        f"[PROGRESS] processed={report['counts']['anchors_processed']:,} "
                        f"written≈{report['counts']['upserted_or_modified']:,} "
                        f"feature_count={report['feature_count_last']}",
                        flush=True,
                    )

            except Exception as exc:
                report["counts"]["row_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(f"anchor_time={a.get('anchor_time')} | {exc}")

        if ops:
            bulk_write(db, target_coll, ops, report)

        report["final_target_count"] = db[target_coll].estimated_document_count()
        if report["counts"]["row_errors"]:
            report["warnings"].append("Some row errors occurred. Check JSON report.")

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
