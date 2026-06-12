# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Build MA Phase & Quality Labels v1

Location:
    Book_Quality/02_labels/ma_quality/01_build_ma_quality_labels_v1.py

Purpose:
    Build labels for the MA Phase & Quality Model.

Core idea:
    Quality predicts.
    Candle Book confirms later.

Inputs:
    xauusd_m15  -> anchors + future trade path
    xauusd_m30  -> MA phase / future direction label
    xauusd_h1   -> MA phase / future direction label
    xauusd_h4   -> MA phase / future direction label

Output:
    bq2_labels_ma_quality_m15_v1

Important:
    This script does NOT modify raw xauusd_* collections.
    It only writes to bq2_labels_ma_quality_m15_v1.

Run TEST ONLY:
    cd C:\Project\Book_Quality\02_labels\ma_quality
    python -u 01_build_ma_quality_labels_v1.py --limit 1000 --reset

Run FULL BUILD:
    python -u 01_build_ma_quality_labels_v1.py --reset
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


SCRIPT_NAME = "01_build_ma_quality_labels_v1.py"
LABEL_VERSION = "ma_quality_label_v1"
BATCH_SIZE = 1000

TIME_FIELD = "datetime"
OPEN_FIELD = "open"
HIGH_FIELD = "high"
LOW_FIELD = "low"
CLOSE_FIELD = "close"

TF_MINUTES = {
    "m15": 15,
    "m30": 30,
    "h1": 60,
    "h4": 240,
}

MA_PERIODS = [20, 50, 100]

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "raw_collections": {
        "m15": "xauusd_m15",
        "m30": "xauusd_m30",
        "h1": "xauusd_h1",
        "h4": "xauusd_h4",
    },
    "bq2_collections": {
        "labels_ma_quality": "bq2_labels_ma_quality_m15_v1",
    },
    "ma_quality_label_v1": {
        "base_r_usd": 2.0,
        "trade_horizon_m15": 64,
        "ma_periods": [20, 50, 100],
        "future_horizons": {
            "m30_bars": 8,
            "h1_bars": 6,
            "h4_bars": 4,
        },
        "direction_threshold_r": 0.35,
        "min_history_bars": {
            "m30": 120,
            "h1": 120,
            "h4": 120,
        },
    },
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reports_dir() -> Path:
    p = project_root() / "02_labels" / "reports"
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

    label_cfg = read_json(root / "00_config" / "bq2_label_config_v1.json")
    if label_cfg and "ma_quality_label_v1" in label_cfg:
        cfg["ma_quality_label_v1"] = deep_merge(
            cfg.get("ma_quality_label_v1", {}),
            label_cfg["ma_quality_label_v1"]
        )

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


def cn(v: Any, digits: int = 8) -> float:
    return float(round(sf(v), digits))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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

    def index_at_or_after(self, dt_epoch: int) -> int:
        return int(np.searchsorted(self.times, dt_epoch, side="left"))


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


def sma_at(closes: np.ndarray, end: int, period: int) -> Optional[float]:
    if end < period:
        return None
    return float(np.mean(closes[end - period:end]))


def ma_slope(closes: np.ndarray, end: int, period: int, lookback: int = 5) -> Optional[float]:
    if end < period + lookback:
        return None
    now = sma_at(closes, end, period)
    old = sma_at(closes, end - lookback, period)
    if now is None or old is None:
        return None
    return float(now - old)


def avg_range(highs: np.ndarray, lows: np.ndarray, end: int, period: int = 20) -> float:
    if end <= 1:
        return 1.0
    start = max(0, end - period)
    r = highs[start:end] - lows[start:end]
    val = float(np.mean(r)) if len(r) else 1.0
    return val if val > 0 else 1.0


def direction_from_slope_and_move(slope20: float, close_move: float, normalizer: float, threshold_r: float) -> str:
    move_r = close_move / normalizer if normalizer > 0 else 0.0
    slope_r = slope20 / normalizer if normalizer > 0 else 0.0

    if move_r >= threshold_r and slope_r >= 0.05:
        return "BUY"
    if move_r <= -threshold_r and slope_r <= -0.05:
        return "SELL"

    # Allow early MA turning even if close move is still moderate.
    if slope_r >= 0.18:
        return "BUY"
    if slope_r <= -0.18:
        return "SELL"

    return "RANGE"


def current_ma_direction(closes: np.ndarray, end: int, highs: np.ndarray, lows: np.ndarray) -> str:
    ma20 = sma_at(closes, end, 20)
    ma50 = sma_at(closes, end, 50)
    ma100 = sma_at(closes, end, 100)
    s20 = ma_slope(closes, end, 20, 5)
    norm = avg_range(highs, lows, end, 20)

    if ma20 is None or ma50 is None or ma100 is None or s20 is None:
        return "UNKNOWN"

    slope_r = s20 / norm if norm > 0 else 0.0

    if ma20 > ma50 > ma100 and slope_r > 0.03:
        return "BUY"
    if ma20 < ma50 < ma100 and slope_r < -0.03:
        return "SELL"
    if slope_r > 0.12:
        return "BUY"
    if slope_r < -0.12:
        return "SELL"
    return "RANGE"


def phase_label(
    current_direction: str,
    future_direction: str,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    end_now: int,
    end_future: int,
) -> str:
    ma20 = sma_at(closes, end_now, 20)
    ma50 = sma_at(closes, end_now, 50)
    ma100 = sma_at(closes, end_now, 100)
    s20 = ma_slope(closes, end_now, 20, 5)
    s20_old = ma_slope(closes, max(0, end_now - 5), 20, 5)
    norm = avg_range(highs, lows, end_now, 20)

    if ma20 is None or ma50 is None or ma100 is None or s20 is None:
        return "UNKNOWN"

    price = float(closes[end_now - 1])
    dist100_r = abs(price - ma100) / norm if norm > 0 else 0.0
    slope_r = s20 / norm if norm > 0 else 0.0
    decel = 0.0
    if s20_old is not None:
        decel = (s20 - s20_old) / norm if norm > 0 else 0.0

    # Reversal/transition states are first-class labels.
    if current_direction == "SELL" and future_direction == "BUY":
        if abs(slope_r) < 0.15 or decel > 0:
            return "REVERSAL_UP_START"
        return "DOWN_END"

    if current_direction == "BUY" and future_direction == "SELL":
        if abs(slope_r) < 0.15 or decel < 0:
            return "REVERSAL_DOWN_START"
        return "UP_END"

    if future_direction == "RANGE":
        if current_direction == "BUY":
            return "UP_EXHAUSTION" if dist100_r > 3.0 or decel < 0 else "UP_END"
        if current_direction == "SELL":
            return "DOWN_EXHAUSTION" if dist100_r > 3.0 or decel > 0 else "DOWN_END"
        return "RANGE"

    if future_direction == "BUY":
        if current_direction in {"SELL", "RANGE", "UNKNOWN"}:
            return "REVERSAL_UP_START"
        if dist100_r > 3.5 and decel < 0:
            return "UP_EXHAUSTION"
        if ma20 > ma50 and ma50 <= ma100:
            return "UP_START"
        return "UP_MIDDLE"

    if future_direction == "SELL":
        if current_direction in {"BUY", "RANGE", "UNKNOWN"}:
            return "REVERSAL_DOWN_START"
        if dist100_r > 3.5 and decel > 0:
            return "DOWN_EXHAUSTION"
        if ma20 < ma50 and ma50 >= ma100:
            return "DOWN_START"
        return "DOWN_MIDDLE"

    return "RANGE"


def analyze_tf(
    tf_data: CandleArrays,
    decision_epoch: int,
    horizon_bars: int,
    threshold_r: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    end_now = tf_data.closed_end_index(decision_epoch)
    end_future = end_now + horizon_bars

    if end_now < 120:
        return None, f"{tf_data.tf}:not_enough_history"
    if end_future > tf_data.count:
        return None, f"{tf_data.tf}:not_enough_future"

    close_now = float(tf_data.closes[end_now - 1])
    close_future = float(tf_data.closes[end_future - 1])
    norm = avg_range(tf_data.highs, tf_data.lows, end_now, 20)

    s20_future = ma_slope(tf_data.closes, end_future, 20, 5)
    if s20_future is None:
        return None, f"{tf_data.tf}:not_enough_future_ma"

    current_dir = current_ma_direction(tf_data.closes, end_now, tf_data.highs, tf_data.lows)
    future_dir = direction_from_slope_and_move(
        slope20=s20_future,
        close_move=close_future - close_now,
        normalizer=norm,
        threshold_r=threshold_r,
    )
    phase = phase_label(current_dir, future_dir, tf_data.closes, tf_data.highs, tf_data.lows, end_now, end_future)

    ma20_now = sma_at(tf_data.closes, end_now, 20)
    ma50_now = sma_at(tf_data.closes, end_now, 50)
    ma100_now = sma_at(tf_data.closes, end_now, 100)
    ma20_future = sma_at(tf_data.closes, end_future, 20)

    move_r = (close_future - close_now) / norm if norm > 0 else 0.0
    slope_future_r = s20_future / norm if norm > 0 else 0.0

    continuation_probability = 0.0
    reversal_probability = 0.0
    exhaustion_score = 0.0

    if current_dir == future_dir and future_dir in {"BUY", "SELL"}:
        continuation_probability = clamp(50.0 + abs(move_r) * 18.0 + abs(slope_future_r) * 30.0, 0, 100)
    elif current_dir in {"BUY", "SELL"} and future_dir in {"BUY", "SELL"} and current_dir != future_dir:
        reversal_probability = clamp(50.0 + abs(move_r) * 18.0 + abs(slope_future_r) * 30.0, 0, 100)
    elif future_dir == "RANGE":
        exhaustion_score = clamp(40.0 + abs(move_r) * 10.0, 0, 100)

    if "EXHAUSTION" in phase or phase.endswith("_END"):
        exhaustion_score = max(exhaustion_score, 65.0)
    if phase.startswith("REVERSAL"):
        reversal_probability = max(reversal_probability, 70.0)

    return {
        "tf": tf_data.tf,
        "current_ma_direction": current_dir,
        "future_direction": future_dir,
        "phase": phase,
        "close_now": cn(close_now),
        "close_future": cn(close_future),
        "future_move_usd": cn(close_future - close_now),
        "future_move_r": cn(move_r),
        "ma20_now": cn(ma20_now),
        "ma50_now": cn(ma50_now),
        "ma100_now": cn(ma100_now),
        "ma20_future": cn(ma20_future),
        "ma20_future_slope": cn(s20_future),
        "ma20_future_slope_r": cn(slope_future_r),
        "continuation_probability": cn(continuation_probability),
        "reversal_probability": cn(reversal_probability),
        "exhaustion_score": cn(exhaustion_score),
    }, None


def first_hit_index_buy(highs: np.ndarray, lows: np.ndarray, entry: float, target: float, stop: float) -> Tuple[Optional[int], Optional[int], bool]:
    tp_idx = None
    sl_idx = None
    ambiguous = False
    for i in range(len(highs)):
        tp = highs[i] >= target
        sl = lows[i] <= stop
        if tp and sl:
            ambiguous = True
            tp_idx = i
            sl_idx = i
            break
        if tp:
            tp_idx = i
            break
        if sl:
            sl_idx = i
            break
    return tp_idx, sl_idx, ambiguous


def first_hit_index_sell(highs: np.ndarray, lows: np.ndarray, entry: float, target: float, stop: float) -> Tuple[Optional[int], Optional[int], bool]:
    tp_idx = None
    sl_idx = None
    ambiguous = False
    for i in range(len(highs)):
        tp = lows[i] <= target
        sl = highs[i] >= stop
        if tp and sl:
            ambiguous = True
            tp_idx = i
            sl_idx = i
            break
        if tp:
            tp_idx = i
            break
        if sl:
            sl_idx = i
            break
    return tp_idx, sl_idx, ambiguous


def trade_quality(side: str, entry: float, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, base_r: float) -> Dict[str, Any]:
    if len(closes) == 0:
        return {
            "quality_score": 0, "risk_mode": "BLOCK", "lot_multiplier": 0.0,
            "tp_mode": "SHORT_TP", "sl_mode": "TIGHT_SL",
            "trailing_mode": "TIGHT_TRAIL", "exit_mode": "FAST_EXIT",
            "mfe": 0.0, "mae": 0.0
        }

    if side == "BUY":
        mfe = float(np.max(highs) - entry)
        mae = float(entry - np.min(lows))
        tp1, sl1, amb = first_hit_index_buy(highs, lows, entry, entry + base_r, entry - base_r)
        tp2, _, _ = first_hit_index_buy(highs, lows, entry, entry + 2 * base_r, entry - base_r)
        tp3, _, _ = first_hit_index_buy(highs, lows, entry, entry + 3 * base_r, entry - base_r)
        final_move = float(closes[-1] - entry)
    else:
        mfe = float(entry - np.min(lows))
        mae = float(np.max(highs) - entry)
        tp1, sl1, amb = first_hit_index_sell(highs, lows, entry, entry - base_r, entry + base_r)
        tp2, _, _ = first_hit_index_sell(highs, lows, entry, entry - 2 * base_r, entry + base_r)
        tp3, _, _ = first_hit_index_sell(highs, lows, entry, entry - 3 * base_r, entry + base_r)
        final_move = float(entry - closes[-1])

    mfe_r = mfe / base_r if base_r > 0 else 0.0
    mae_r = mae / base_r if base_r > 0 else 0.0
    final_r = final_move / base_r if base_r > 0 else 0.0

    score = 0.0
    if tp1 is not None and (sl1 is None or tp1 < sl1):
        score += 35
    if tp2 is not None and (sl1 is None or tp2 < sl1):
        score += 25
    if tp3 is not None and (sl1 is None or tp3 < sl1):
        score += 15

    score += clamp(mfe_r * 10.0, 0, 25)
    score += clamp(final_r * 8.0, -20, 20)
    score -= clamp(mae_r * 18.0, 0, 40)

    if sl1 is not None and (tp1 is None or sl1 < tp1):
        score -= 30
    if amb:
        score -= 20

    score = clamp(score, 0, 100)

    if score >= 85:
        lot = 2.0
        risk = "LOW_RISK"
    elif score >= 72:
        lot = 1.5
        risk = "LOW_RISK"
    elif score >= 58:
        lot = 1.0
        risk = "NORMAL_RISK"
    elif score >= 42:
        lot = 0.5
        risk = "NORMAL_RISK"
    elif score >= 25:
        lot = 0.25
        risk = "HIGH_RISK"
    else:
        lot = 0.0
        risk = "BLOCK"

    tp_mode = "SHORT_TP"
    if mfe_r >= 3:
        tp_mode = "HOLD_MORE"
    elif mfe_r >= 2:
        tp_mode = "WIDE_TP"
    elif mfe_r >= 1:
        tp_mode = "NORMAL_TP"

    sl_mode = "NORMAL_SL"
    if mae_r < 0.4:
        sl_mode = "TIGHT_SL"
    elif mae_r > 1.2:
        sl_mode = "WIDE_SL"

    # Giveback after good run-up => trailing needed.
    giveback_r = max(0.0, mfe_r - max(final_r, 0.0))
    if score < 35:
        trailing = "TIGHT_TRAIL"
        exit_mode = "FAST_EXIT"
    elif giveback_r >= 1.5:
        trailing = "FAST_TRAIL"
        exit_mode = "PROTECT_PROFIT"
    elif giveback_r >= 0.8:
        trailing = "NORMAL_TRAIL"
        exit_mode = "PROTECT_PROFIT"
    elif mfe_r >= 2.5 and final_r >= 1.0:
        trailing = "NO_TRAILING"
        exit_mode = "HOLD_MORE"
    else:
        trailing = "NORMAL_TRAIL"
        exit_mode = "NORMAL_EXIT"

    return {
        "quality_score": int(round(score)),
        "risk_mode": risk,
        "lot_multiplier": lot,
        "tp_mode": tp_mode,
        "sl_mode": sl_mode,
        "trailing_mode": trailing,
        "exit_mode": exit_mode,
        "mfe": cn(mfe),
        "mae": cn(mae),
        "mfe_r": cn(mfe_r),
        "mae_r": cn(mae_r),
        "final_r": cn(final_r),
        "tp1_index": tp1,
        "tp2_index": tp2,
        "tp3_index": tp3,
        "sl1_index": sl1,
        "ambiguous_intrabar": amb,
    }


def overall_direction(tf_labels: Dict[str, Dict[str, Any]], buy_q: int, sell_q: int) -> Tuple[str, float, str]:
    weights = {"m30": 0.40, "h1": 0.35, "h4": 0.25}
    buy_vote = 0.0
    sell_vote = 0.0
    range_vote = 0.0

    for tf, w in weights.items():
        d = tf_labels[tf]["future_direction"]
        phase = tf_labels[tf]["phase"]

        if d == "BUY":
            buy_vote += w
        elif d == "SELL":
            sell_vote += w
        else:
            range_vote += w

        # A phase ending opposite trend supports reversal.
        if phase in {"DOWN_END", "DOWN_EXHAUSTION", "REVERSAL_UP_START"}:
            buy_vote += w * 0.35
        if phase in {"UP_END", "UP_EXHAUSTION", "REVERSAL_DOWN_START"}:
            sell_vote += w * 0.35

    # Trade outcome label contributes too, but does not dominate MA scenario.
    if buy_q - sell_q >= 15:
        buy_vote += 0.25
    elif sell_q - buy_q >= 15:
        sell_vote += 0.25
    else:
        range_vote += 0.15

    total = buy_vote + sell_vote + range_vote
    if total <= 0:
        return "RANGE", 0.0, "no_vote"

    buy_p = buy_vote / total
    sell_p = sell_vote / total
    range_p = range_vote / total

    if buy_p >= sell_p and buy_p >= range_p and buy_p >= 0.42:
        return "BUY", cn(buy_p * 100, 2), f"buy_vote={cn(buy_vote,3)} sell_vote={cn(sell_vote,3)} range_vote={cn(range_vote,3)}"
    if sell_p >= buy_p and sell_p >= range_p and sell_p >= 0.42:
        return "SELL", cn(sell_p * 100, 2), f"buy_vote={cn(buy_vote,3)} sell_vote={cn(sell_vote,3)} range_vote={cn(range_vote,3)}"
    return "RANGE", cn(max(buy_p, sell_p, range_p) * 100, 2), f"buy_vote={cn(buy_vote,3)} sell_vote={cn(sell_vote,3)} range_vote={cn(range_vote,3)}"


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
    lines.append("Book & Quality v2 - MA Quality Labels v1 Report")
    lines.append("=" * 74)
    for k in ["run_id", "status", "database", "symbol", "start_time", "end_time", "duration_seconds"]:
        lines.append(f"{k:<28}: {report.get(k)}")

    lines.append("")
    lines.append("Collections")
    lines.append("-" * 74)
    for k, v in report["collections"].items():
        lines.append(f"{k:<28}: {v}")

    lines.append("")
    lines.append("Config")
    lines.append("-" * 74)
    for k, v in report["label_config"].items():
        lines.append(f"{k:<28}: {v}")

    lines.append("")
    lines.append("Counts")
    lines.append("-" * 74)
    for k, v in report["counts"].items():
        lines.append(f"{k:<28}: {v}")

    lines.append("")
    lines.append("Prediction Distribution")
    lines.append("-" * 74)
    for k, v in report["prediction_distribution"].items():
        lines.append(f"{k:<28}: {v}")

    lines.append("")
    lines.append("Phase Distribution")
    lines.append("-" * 74)
    for tf, d in report["phase_distribution"].items():
        lines.append(f"{tf.upper()}:")
        for k, v in d.items():
            lines.append(f"  {k:<25}: {v}")

    lines.append("")
    lines.append("Risk Distribution")
    lines.append("-" * 74)
    for side, d in report["risk_distribution"].items():
        lines.append(f"{side}:")
        for k, v in d.items():
            lines.append(f"  {k:<25}: {v}")

    if report.get("ma_context_skip_reasons"):
        lines.append("")
        lines.append("MA Context Skip Reasons")
        lines.append("-" * 74)
        for k, v in report["ma_context_skip_reasons"].items():
            lines.append(f"{k:<28}: {v}")

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
    jp = rd / f"ma_quality_labels_v1_report_{rs}.json"
    tp = rd / f"ma_quality_labels_v1_report_{rs}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tp.write_text(build_txt_report(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build bq2 MA Quality Labels v1.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--start", type=str, default=None, help="Anchor time start, e.g. 2020-01-01 00:00:00")
    p.add_argument("--end", type=str, default=None, help="Anchor time end, e.g. 2025-12-31 23:59:59")
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
    label_cfg = cfg["ma_quality_label_v1"]

    target_coll = bq2.get("labels_ma_quality", "bq2_labels_ma_quality_m15_v1")

    base_r = float(label_cfg.get("base_r_usd", 2.0))
    trade_horizon_m15 = int(label_cfg.get("trade_horizon_m15", 64))
    horizons = label_cfg.get("future_horizons", {})
    h_m30 = int(horizons.get("m30_bars", 8))
    h_h1 = int(horizons.get("h1_bars", 6))
    h_h4 = int(horizons.get("h4_bars", 4))
    threshold_r = float(label_cfg.get("direction_threshold_r", 0.35))

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"ma_quality_labels_v1_{rs}",
        "status": "running",
        "database": db_name,
        "symbol": symbol,
        "start_time": started.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "collections": {
            "m15_source": raw.get("m15", "xauusd_m15"),
            "m30_source": raw.get("m30", "xauusd_m30"),
            "h1_source": raw.get("h1", "xauusd_h1"),
            "h4_source": raw.get("h4", "xauusd_h4"),
            "target": target_coll,
        },
        "label_config": {
            "label_version": LABEL_VERSION,
            "base_r_usd": base_r,
            "trade_horizon_m15": trade_horizon_m15,
            "ma_periods": [20, 50, 100],
            "m30_horizon_bars": h_m30,
            "h1_horizon_bars": h_h1,
            "h4_horizon_bars": h_h4,
            "direction_threshold_r": threshold_r,
        },
        "source_loads": {},
        "counts": {
            "anchors_seen": 0,
            "anchors_processed": 0,
            "upserted_or_modified": 0,
            "inserted": 0,
            "matched": 0,
            "modified": 0,
            "skipped_existing": 0,
            "skipped_no_entry": 0,
            "skipped_no_future_trade_path": 0,
            "skipped_no_ma_context": 0,
            "row_errors": 0,
        },
        "prediction_distribution": {"BUY": 0, "SELL": 0, "RANGE": 0},
        "phase_distribution": {"m30": {}, "h1": {}, "h4": {}},
        "risk_distribution": {"buy": {}, "sell": {}},
        "ma_context_skip_reasons": {},
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

    print("=== Book & Quality v2 - Build MA Quality Labels v1 ===", flush=True)
    print(f"Database : {db_name}", flush=True)
    print(f"Symbol   : {symbol}", flush=True)
    print(f"Target   : {target_coll}", flush=True)

    try:
        db = connect(cfg)

        if args.reset:
            deleted = db[target_coll].delete_many({})
            print(f"[RESET] Deleted {deleted.deleted_count:,} docs from {target_coll}", flush=True)
            report["reset_deleted_count"] = deleted.deleted_count

        db[target_coll].create_index([("symbol", ASCENDING), ("anchor_time", ASCENDING)], unique=True, name="uq_symbol_anchor_time")
        db[target_coll].create_index([("label_version", ASCENDING), ("anchor_time", ASCENDING)], name="ix_label_version_anchor_time")
        db[target_coll].create_index([("y.predicted_direction", ASCENDING), ("anchor_time", ASCENDING)], name="ix_predicted_direction_anchor_time")

        data = {
            "m15": load_candles(db, raw.get("m15", "xauusd_m15"), "m15", report),
            "m30": load_candles(db, raw.get("m30", "xauusd_m30"), "m30", report),
            "h1": load_candles(db, raw.get("h1", "xauusd_h1"), "h1", report),
            "h4": load_candles(db, raw.get("h4", "xauusd_h4"), "h4", report),
        }

        start_arg = parse_dt_arg(args.start)
        end_arg = parse_dt_arg(args.end)
        start_epoch = epoch(start_arg) if start_arg else None
        end_epoch = epoch(end_arg) if end_arg else None

        ops: List[UpdateOne] = []
        max_i = data["m15"].count - 1
        # v2 fix:
        # --limit must be applied AFTER --start/--end filtering.
        # In v1, --limit 1000 with a later --start could still scan only the first
        # 1000 historical M15 candles and therefore process nothing.
        for i in range(max_i):
            try:
                anchor_epoch = int(data["m15"].times[i])
                entry_idx = i + 1

                if start_epoch and anchor_epoch < start_epoch:
                    continue
                if end_epoch and anchor_epoch > end_epoch:
                    continue

                if args.limit is not None and report["counts"]["anchors_seen"] >= args.limit:
                    break

                report["counts"]["anchors_seen"] += 1

                anchor_time = datetime.fromtimestamp(anchor_epoch, tz=timezone.utc).replace(tzinfo=None)
                decision_epoch = int(data["m15"].times[entry_idx])
                decision_time = datetime.fromtimestamp(decision_epoch, tz=timezone.utc).replace(tzinfo=None)
                entry_price = float(data["m15"].opens[entry_idx])

                if args.skip_existing:
                    exists = db[target_coll].find_one({"symbol": symbol, "anchor_time": anchor_time}, projection={"_id": 1})
                    if exists:
                        report["counts"]["skipped_existing"] += 1
                        continue

                trade_start = entry_idx
                trade_end = trade_start + trade_horizon_m15
                if trade_end > data["m15"].count:
                    report["counts"]["skipped_no_future_trade_path"] += 1
                    continue

                tf_labels: Dict[str, Dict[str, Any]] = {}
                ma_error = None
                for tf, horizon in [("m30", h_m30), ("h1", h_h1), ("h4", h_h4)]:
                    label, err = analyze_tf(data[tf], decision_epoch, horizon, threshold_r)
                    if err:
                        ma_error = err
                        break
                    tf_labels[tf] = label

                if ma_error:
                    report["counts"]["skipped_no_ma_context"] += 1
                    report["ma_context_skip_reasons"][ma_error] = report["ma_context_skip_reasons"].get(ma_error, 0) + 1
                    continue

                future_highs = data["m15"].highs[trade_start:trade_end]
                future_lows = data["m15"].lows[trade_start:trade_end]
                future_closes = data["m15"].closes[trade_start:trade_end]

                buy = trade_quality("BUY", entry_price, future_highs, future_lows, future_closes, base_r)
                sell = trade_quality("SELL", entry_price, future_highs, future_lows, future_closes, base_r)

                pred_dir, pred_prob, reason = overall_direction(tf_labels, buy["quality_score"], sell["quality_score"])

                now = utc_now()
                doc = {
                    "symbol": symbol,
                    "anchor_time": anchor_time,
                    "decision_time": decision_time,
                    "entry_time": decision_time,
                    "entry_price": cn(entry_price),
                    "label_version": LABEL_VERSION,
                    "model_type": "ma_phase_quality",
                    "input_timeframes": ["M30", "H1", "H4"],
                    "ma_periods": [20, 50, 100],
                    "config": report["label_config"],
                    "tf_labels": tf_labels,
                    "buy_target": buy,
                    "sell_target": sell,
                    "prediction": {
                        "predicted_direction": pred_dir,
                        "prediction_probability_label": pred_prob,
                        "reason": reason,
                    },
                    "y": {
                        "predicted_direction": pred_dir,
                        "prediction_probability_label": pred_prob,

                        "m30_future_direction": tf_labels["m30"]["future_direction"],
                        "h1_future_direction": tf_labels["h1"]["future_direction"],
                        "h4_future_direction": tf_labels["h4"]["future_direction"],

                        "m30_phase": tf_labels["m30"]["phase"],
                        "h1_phase": tf_labels["h1"]["phase"],
                        "h4_phase": tf_labels["h4"]["phase"],

                        "buy_quality_score": buy["quality_score"],
                        "sell_quality_score": sell["quality_score"],
                        "buy_risk_mode": buy["risk_mode"],
                        "sell_risk_mode": sell["risk_mode"],
                        "buy_lot_multiplier": buy["lot_multiplier"],
                        "sell_lot_multiplier": sell["lot_multiplier"],
                        "buy_tp_mode": buy["tp_mode"],
                        "sell_tp_mode": sell["tp_mode"],
                        "buy_sl_mode": buy["sl_mode"],
                        "sell_sl_mode": sell["sl_mode"],
                        "buy_trailing_mode": buy["trailing_mode"],
                        "sell_trailing_mode": sell["trailing_mode"],
                        "buy_exit_mode": buy["exit_mode"],
                        "sell_exit_mode": sell["exit_mode"],
                    },
                    "created_at": now,
                    "updated_at": now,
                }

                ops.append(make_update_op(symbol, doc))
                report["counts"]["anchors_processed"] += 1

                report["prediction_distribution"][pred_dir] = report["prediction_distribution"].get(pred_dir, 0) + 1
                for tf in ["m30", "h1", "h4"]:
                    ph = tf_labels[tf]["phase"]
                    report["phase_distribution"][tf][ph] = report["phase_distribution"][tf].get(ph, 0) + 1

                report["risk_distribution"]["buy"][buy["risk_mode"]] = report["risk_distribution"]["buy"].get(buy["risk_mode"], 0) + 1
                report["risk_distribution"]["sell"][sell["risk_mode"]] = report["risk_distribution"]["sell"].get(sell["risk_mode"], 0) + 1

                if len(ops) >= BATCH_SIZE:
                    execute_bulk(db, target_coll, ops, report)
                    ops = []
                    print(
                        f"[PROGRESS] seen={report['counts']['anchors_seen']:,} "
                        f"processed={report['counts']['anchors_processed']:,} "
                        f"written≈{report['counts']['upserted_or_modified']:,} "
                        f"BUY={report['prediction_distribution'].get('BUY',0):,} "
                        f"SELL={report['prediction_distribution'].get('SELL',0):,} "
                        f"RANGE={report['prediction_distribution'].get('RANGE',0):,}",
                        flush=True,
                    )

            except Exception as row_exc:
                report["counts"]["row_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(f"i={i} | {row_exc}")

        if ops:
            execute_bulk(db, target_coll, ops, report)

        report["final_target_count"] = db[target_coll].estimated_document_count()

        if report["counts"]["anchors_seen"] > 0 and report["counts"]["anchors_processed"] == 0 and report["counts"]["skipped_no_ma_context"] > 0:
            report["warnings"].append(
                "All selected anchors were skipped because MA context was not available. "
                "This usually happens when testing very early history before H4 has enough MA100 history. "
                "Run the test with a later --start date, e.g. --start \"2010-03-01 00:00:00\"."
            )

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
    print(f"Status                  : {report['status']}", flush=True)
    print(f"Anchors Seen            : {report['counts']['anchors_seen']:,}", flush=True)
    print(f"Anchors Processed       : {report['counts']['anchors_processed']:,}", flush=True)
    print(f"Upserted/Modified       : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Skipped No MA Context   : {report['counts']['skipped_no_ma_context']:,}", flush=True)
    print(f"Skipped No Future Path  : {report['counts']['skipped_no_future_trade_path']:,}", flush=True)
    print(f"Row Errors              : {report['counts']['row_errors']:,}", flush=True)
    print(f"Prediction Distribution : {report['prediction_distribution']}", flush=True)
    print(f"JSON Report             : {report.get('report_json_path')}", flush=True)
    print(f"TXT Report              : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
