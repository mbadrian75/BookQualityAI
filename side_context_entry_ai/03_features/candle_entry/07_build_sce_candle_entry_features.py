# -*- coding: utf-8 -*-
"""
SCE Candle Entry Feature Builder

Location:
03_features/candle_entry/

Report:
03_features/candle_entry/reports/

Purpose:
- Build side-based candle entry features for each labeled candidate row.
- Uses M5 anchor candle and M15 closed candle context only.
- Does not use M1 as feature.
- Writes to sce_features_candle_entry_m5_v1.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
import traceback
from array import array
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from pymongo import ASCENDING, MongoClient, ReplaceOne
except ImportError as exc:
    print("ERROR: pymongo is not installed. Run: pip install pymongo")
    raise exc

from config.sce_project_config import (
    COLL_FEATURES_CANDLE_ENTRY_M5_V1,
    COLL_LABELS_ENTRY_SIDE_M5_V1,
    DB_NAME,
    FEATURE_VERSION_CANDLE_ENTRY_M5_V1,
    LABEL_VERSION_ENTRY_SIDE_M5_V1,
    MONGO_URI,
    PROJECT_CODE,
    PROJECT_NAME,
    RAW_M1_AS_FEATURE,
    SIDES,
)

RUN_TYPE = "sce_build_candle_entry_features"
RUN_ID = f"{RUN_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SECONDS_M15 = 900


def parse_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"]:
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            pass
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCE side-based candle entry features.")
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--database", default=DB_NAME)
    parser.add_argument("--label-collection", default=COLL_LABELS_ENTRY_SIDE_M5_V1)
    parser.add_argument("--target-collection", default=COLL_FEATURES_CANDLE_ENTRY_M5_V1)
    parser.add_argument("--label-version", default=LABEL_VERSION_ENTRY_SIDE_M5_V1)
    parser.add_argument("--feature-version", default=FEATURE_VERSION_CANDLE_ENTRY_M5_V1)
    parser.add_argument("--limit", type=int, default=50000, help="Candidate row limit. Use 0 for full run.")
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--m5-lookback", type=int, default=20)
    parser.add_argument("--m15-lookback", type=int, default=20)
    parser.add_argument("--breakout-lookback", type=int, default=20)
    parser.add_argument("--compression-lookback", type=int, default=10)
    parser.add_argument("--doji-body-ratio", type=float, default=0.10)
    parser.add_argument("--expansion-ratio", type=float, default=1.50)
    parser.add_argument("--compression-ratio", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--clear-existing", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def dt_to_epoch_seconds(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.replace(tzinfo=None).timestamp())
    if isinstance(value, (int, float)):
        return int(value / 1000) if value > 10_000_000_000 else int(value)
    if isinstance(value, str):
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"]:
            try:
                return int(datetime.strptime(value[:19], fmt).timestamp())
            except Exception:
                pass
    return None


def epoch_to_dt(seconds: int) -> datetime:
    return datetime.fromtimestamp(int(seconds))


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def load_raw_config() -> Dict[str, Any]:
    config_path = ROOT / "config" / "sce_raw_collections_detected.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {"mapping": {"M5": {"collection": "xauusd_m5", "datetime_field": "datetime", "ohlc_fields": {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}}, "M15": {"collection": "xauusd_m15", "datetime_field": "datetime", "ohlc_fields": {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}}}}


def load_tf_arrays(db, tf: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    coll = db[cfg["collection"]]
    time_f = cfg["datetime_field"]
    ohlc = cfg["ohlc_fields"]
    times, opens, highs, lows, closes, volumes = array("q"), array("d"), array("d"), array("d"), array("d"), array("d")
    skipped = 0
    projection = {time_f: 1, ohlc["open"]: 1, ohlc["high"]: 1, ohlc["low"]: 1, ohlc["close"]: 1, ohlc.get("volume", "volume"): 1, "_id": 0}
    for doc in coll.find({}, projection).sort(time_f, ASCENDING).batch_size(10000):
        t = dt_to_epoch_seconds(doc.get(time_f))
        try:
            o = float(doc[ohlc["open"]]); h = float(doc[ohlc["high"]]); l = float(doc[ohlc["low"]]); c = float(doc[ohlc["close"]]); v = float(doc.get(ohlc.get("volume", "volume"), 0.0))
        except Exception:
            skipped += 1
            continue
        if t is None:
            skipped += 1
            continue
        times.append(t); opens.append(o); highs.append(h); lows.append(l); closes.append(c); volumes.append(v)
    return {"tf": tf, "times": times, "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes, "skipped": skipped}


def candle_direction(o: float, c: float, body_abs: float, rng: float, doji_ratio: float) -> str:
    if rng <= 0 or safe_div(body_abs, rng) <= doji_ratio:
        return "doji"
    return "bull" if c > o else "bear"


def basic_candle_features(prefix: str, data: Dict[str, Any], idx: int, side: str, doji_ratio: float) -> Dict[str, Any]:
    o, h, l, c = float(data["open"][idx]), float(data["high"][idx]), float(data["low"][idx]), float(data["close"][idx])
    rng = max(h - l, 0.0)
    body = c - o
    body_abs = abs(body)
    upper_wick = max(h - max(o, c), 0.0)
    lower_wick = max(min(o, c) - l, 0.0)
    close_pos = safe_div(c - l, rng)
    direction = candle_direction(o, c, body_abs, rng, doji_ratio)
    side_body = body if side == "BUY" else -body
    side_rejection_wick = lower_wick if side == "BUY" else upper_wick
    side_opposite_wick = upper_wick if side == "BUY" else lower_wick
    side_close_pos = close_pos if side == "BUY" else 1.0 - close_pos
    return {
        f"{prefix}_time": epoch_to_dt(int(data["times"][idx])),
        f"{prefix}_open": round(o, 5),
        f"{prefix}_high": round(h, 5),
        f"{prefix}_low": round(l, 5),
        f"{prefix}_close": round(c, 5),
        f"{prefix}_range": round(rng, 5),
        f"{prefix}_body": round(body, 5),
        f"{prefix}_body_abs": round(body_abs, 5),
        f"{prefix}_body_ratio": round(safe_div(body_abs, rng), 8),
        f"{prefix}_upper_wick": round(upper_wick, 5),
        f"{prefix}_lower_wick": round(lower_wick, 5),
        f"{prefix}_upper_wick_ratio": round(safe_div(upper_wick, rng), 8),
        f"{prefix}_lower_wick_ratio": round(safe_div(lower_wick, rng), 8),
        f"{prefix}_close_position": round(close_pos, 8),
        f"{prefix}_direction": direction,
        f"{prefix}_is_doji": direction == "doji",
        f"{prefix}_side_body": round(side_body, 5),
        f"{prefix}_side_body_power": round(safe_div(side_body, rng), 8),
        f"{prefix}_side_rejection_wick": round(side_rejection_wick, 5),
        f"{prefix}_side_rejection_wick_ratio": round(safe_div(side_rejection_wick, rng), 8),
        f"{prefix}_side_opposite_wick": round(side_opposite_wick, 5),
        f"{prefix}_side_opposite_wick_ratio": round(safe_div(side_opposite_wick, rng), 8),
        f"{prefix}_side_close_position": round(side_close_pos, 8),
    }


def engulfing_features(prefix: str, data: Dict[str, Any], idx: int, side: str) -> Dict[str, Any]:
    if idx <= 0:
        return {f"{prefix}_bullish_engulfing": False, f"{prefix}_bearish_engulfing": False, f"{prefix}_side_engulfing": False}
    o, c = float(data["open"][idx]), float(data["close"][idx])
    po, pc = float(data["open"][idx - 1]), float(data["close"][idx - 1])
    bullish = c > o and pc < po and o <= pc and c >= po
    bearish = c < o and pc > po and o >= pc and c <= po
    side_engulf = bullish if side == "BUY" else bearish
    return {f"{prefix}_bullish_engulfing": bullish, f"{prefix}_bearish_engulfing": bearish, f"{prefix}_side_engulfing": side_engulf}


def inside_bar_features(prefix: str, data: Dict[str, Any], idx: int) -> Dict[str, Any]:
    if idx <= 0:
        return {f"{prefix}_inside_bar": False}
    inside = float(data["high"][idx]) <= float(data["high"][idx - 1]) and float(data["low"][idx]) >= float(data["low"][idx - 1])
    return {f"{prefix}_inside_bar": inside}


def breakout_features(prefix: str, data: Dict[str, Any], idx: int, side: str, lookback: int) -> Dict[str, Any]:
    if idx - lookback < 0:
        return {f"{prefix}_side_breakout": False, f"{prefix}_opposite_breakout": False, f"{prefix}_side_failed_breakout": False, f"{prefix}_opposite_failed_breakout": False}
    prev_high = max(float(x) for x in data["high"][idx - lookback:idx])
    prev_low = min(float(x) for x in data["low"][idx - lookback:idx])
    h, l, c = float(data["high"][idx]), float(data["low"][idx]), float(data["close"][idx])
    buy_breakout = h > prev_high and c > prev_high
    sell_breakout = l < prev_low and c < prev_low
    failed_buy_breakout = h > prev_high and c <= prev_high
    failed_sell_breakout = l < prev_low and c >= prev_low
    side_breakout = buy_breakout if side == "BUY" else sell_breakout
    opposite_breakout = sell_breakout if side == "BUY" else buy_breakout
    side_failed = failed_sell_breakout if side == "BUY" else failed_buy_breakout
    opposite_failed = failed_buy_breakout if side == "BUY" else failed_sell_breakout
    return {
        f"{prefix}_prev_high_{lookback}": round(prev_high, 5),
        f"{prefix}_prev_low_{lookback}": round(prev_low, 5),
        f"{prefix}_buy_breakout": buy_breakout,
        f"{prefix}_sell_breakout": sell_breakout,
        f"{prefix}_failed_buy_breakout": failed_buy_breakout,
        f"{prefix}_failed_sell_breakout": failed_sell_breakout,
        f"{prefix}_side_breakout": side_breakout,
        f"{prefix}_opposite_breakout": opposite_breakout,
        f"{prefix}_side_failed_breakout": side_failed,
        f"{prefix}_opposite_failed_breakout": opposite_failed,
    }


def compression_expansion_features(prefix: str, data: Dict[str, Any], idx: int, lookback: int, expansion_ratio: float, compression_ratio: float) -> Dict[str, Any]:
    if idx - lookback < 0:
        return {f"{prefix}_avg_range_{lookback}": None, f"{prefix}_range_expansion_ratio": None, f"{prefix}_is_expansion": False, f"{prefix}_is_compression": False}
    current_range = float(data["high"][idx]) - float(data["low"][idx])
    ranges = [float(data["high"][j]) - float(data["low"][j]) for j in range(idx - lookback, idx)]
    avg_range = sum(ranges) / len(ranges) if ranges else 0.0
    ratio = safe_div(current_range, avg_range)
    return {f"{prefix}_avg_range_{lookback}": round(avg_range, 5), f"{prefix}_range_expansion_ratio": round(ratio, 8), f"{prefix}_is_expansion": ratio >= expansion_ratio, f"{prefix}_is_compression": ratio <= compression_ratio}


def pullback_features(prefix: str, features: Dict[str, Any]) -> Dict[str, Any]:
    body_power = float(features.get(f"{prefix}_side_body_power", 0.0))
    rejection = float(features.get(f"{prefix}_side_rejection_wick_ratio", 0.0))
    opposite_wick = float(features.get(f"{prefix}_side_opposite_wick_ratio", 0.0))
    pullback = body_power < 0 and rejection > 0.25
    rejection_entry = body_power >= -0.25 and rejection > opposite_wick
    return {f"{prefix}_side_pullback": pullback, f"{prefix}_side_rejection_entry": rejection_entry}


def build_tf_candle_features(prefix: str, data: Dict[str, Any], idx: int, side: str, args: argparse.Namespace) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    f.update(basic_candle_features(prefix, data, idx, side, args.doji_body_ratio))
    f.update(engulfing_features(prefix, data, idx, side))
    f.update(inside_bar_features(prefix, data, idx))
    f.update(breakout_features(prefix, data, idx, side, args.breakout_lookback))
    f.update(compression_expansion_features(prefix, data, idx, args.compression_lookback, args.expansion_ratio, args.compression_ratio))
    f.update(pullback_features(prefix, f))
    return f


def build_label_query(args: argparse.Namespace) -> Dict[str, Any]:
    query: Dict[str, Any] = {"label_version": args.label_version}
    start = parse_time(args.start_time)
    end = parse_time(args.end_time)
    if start or end:
        f: Dict[str, Any] = {}
        if start:
            f["$gte"] = start
        if end:
            f["$lte"] = end
        query["anchor_time"] = f
    return query


def write_report(args: argparse.Namespace, status: str, stats: Dict[str, Any], error_text: str = "") -> Path:
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{RUN_ID}.txt"
    lines: List[str] = []
    lines.append("Side Context Entry AI - Candle Entry Feature Builder Report")
    lines.append("=" * 76)
    lines.append(f"run_id                   : {RUN_ID}")
    lines.append(f"status                   : {status}")
    lines.append(f"project_name             : {PROJECT_NAME}")
    lines.append(f"project_code             : {PROJECT_CODE}")
    lines.append(f"database                 : {args.database}")
    lines.append(f"label_collection          : {args.label_collection}")
    lines.append(f"target_collection         : {args.target_collection}")
    lines.append(f"feature_version           : {args.feature_version}")
    lines.append("raw_data_modified         : False")
    lines.append(f"m1_as_feature             : {RAW_M1_AS_FEATURE}")
    lines.append("report_folder_policy      : local_section_reports")
    lines.append("")
    lines.append("Runtime parameters:")
    for key in ["limit", "start_time", "end_time", "m5_lookback", "m15_lookback", "breakout_lookback", "compression_lookback", "doji_body_ratio", "expansion_ratio", "compression_ratio", "batch_size", "clear_existing"]:
        lines.append(f"{key:26}: {getattr(args, key)}")
    lines.append("")
    lines.append("Counts:")
    for key, value in stats.get("counts", {}).items():
        lines.append(f"{key:26}: {value}")
    lines.append("")
    lines.append("Side distribution:")
    for key, value in sorted(stats.get("side_counts", {}).items()):
        lines.append(f"{key:26}: {value}")
    lines.append("")
    lines.append("Skip reasons:")
    for key, value in sorted(stats.get("skip_reasons", {}).items()):
        lines.append(f"{key:26}: {value}")
    if error_text:
        lines.append("")
        lines.append("Error:")
        lines.append(error_text)
    lines.append("")
    lines.append("End of report.")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    client = None
    stats: Dict[str, Any] = {"counts": {}, "side_counts": {}, "skip_reasons": {}}
    try:
        raw_config = load_raw_config()
        mapping = raw_config.get("mapping", {})
        if "M5" not in mapping or "M15" not in mapping:
            raise RuntimeError("Missing M5/M15 raw mapping. Run inspect_raw first.")
        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[args.database]
        print("Loading M5 arrays...")
        m5_data = load_tf_arrays(db, "M5", mapping["M5"])
        print("Loading M15 arrays...")
        m15_data = load_tf_arrays(db, "M15", mapping["M15"])
        target = db[args.target_collection]
        if args.clear_existing == 1:
            result = target.delete_many({"feature_version": args.feature_version})
            stats["counts"]["cleared_existing_docs"] = result.deleted_count
        else:
            stats["counts"]["cleared_existing_docs"] = 0
        label_query = build_label_query(args)
        label_coll = db[args.label_collection]
        projection = {"anchor_time": 1, "entry_time": 1, "side": 1, "entry_price": 1, "label_version": 1, "_id": 0}
        cursor = label_coll.find(label_query, projection).sort([("anchor_time", ASCENDING), ("side", ASCENDING)]).batch_size(10000)
        if args.limit > 0:
            cursor = cursor.limit(args.limit)
        operations: List[ReplaceOne] = []
        side_counter: Counter = Counter()
        skip_counter: Counter = Counter()
        scanned = 0
        saved = 0
        first_anchor_time = None
        last_anchor_time = None
        min_history = max(args.m5_lookback, args.breakout_lookback, args.compression_lookback)
        for label_doc in cursor:
            scanned += 1
            anchor_dt = label_doc.get("anchor_time")
            entry_dt = label_doc.get("entry_time")
            side = label_doc.get("side")
            entry_price = label_doc.get("entry_price")
            if side not in SIDES:
                skip_counter["invalid_side"] += 1
                continue
            anchor_time = dt_to_epoch_seconds(anchor_dt)
            entry_time = dt_to_epoch_seconds(entry_dt)
            if anchor_time is None or entry_time is None or entry_price is None:
                skip_counter["bad_label_time_or_price"] += 1
                continue
            if first_anchor_time is None:
                first_anchor_time = anchor_time
            last_anchor_time = anchor_time
            m5_idx = bisect.bisect_left(m5_data["times"], anchor_time)
            if m5_idx >= len(m5_data["times"]) or int(m5_data["times"][m5_idx]) != anchor_time:
                skip_counter["m5_anchor_not_found"] += 1
                continue
            if m5_idx < min_history:
                skip_counter["m5_history_not_available"] += 1
                continue
            m15_closed_limit = entry_time - SECONDS_M15
            m15_idx = bisect.bisect_right(m15_data["times"], m15_closed_limit) - 1
            if m15_idx < 0:
                skip_counter["m15_context_not_available"] += 1
                continue
            if m15_idx < max(args.m15_lookback, args.breakout_lookback, args.compression_lookback):
                skip_counter["m15_history_not_available"] += 1
                continue
            doc: Dict[str, Any] = {"anchor_time": epoch_to_dt(anchor_time), "entry_time": epoch_to_dt(entry_time), "side": side, "entry_price": round(float(entry_price), 5), "feature_version": args.feature_version, "label_version": args.label_version, "run_id": RUN_ID, "source_tfs": ["M5", "M15"], "created_at": utc_now(), "updated_at": utc_now()}
            m5_f = build_tf_candle_features("m5", m5_data, m5_idx, side, args)
            m15_f = build_tf_candle_features("m15", m15_data, m15_idx, side, args)
            doc.update(m5_f)
            doc.update(m15_f)
            m5_body_power = float(doc.get("m5_side_body_power", 0.0))
            m15_body_power = float(doc.get("m15_side_body_power", 0.0))
            m5_reject = float(doc.get("m5_side_rejection_wick_ratio", 0.0))
            m15_reject = float(doc.get("m15_side_rejection_wick_ratio", 0.0))
            doc["side_body_power_total"] = round(m5_body_power + m15_body_power, 8)
            doc["side_rejection_wick_total"] = round(m5_reject + m15_reject, 8)
            doc["candle_context_agreement"] = m5_body_power > 0 and m15_body_power > 0
            doc["candle_rejection_agreement"] = m5_reject > 0.25 and m15_reject > 0.20
            doc["candle_breakout_agreement"] = bool(doc.get("m5_side_breakout")) and bool(doc.get("m15_side_breakout"))
            doc["candle_failed_breakout_support"] = bool(doc.get("m5_side_failed_breakout")) or bool(doc.get("m15_side_failed_breakout"))
            operations.append(ReplaceOne({"anchor_time": doc["anchor_time"], "side": side}, doc, upsert=True))
            side_counter[side] += 1
            if len(operations) >= args.batch_size:
                target.bulk_write(operations, ordered=False)
                saved += len(operations)
                operations.clear()
                if saved % (args.batch_size * 10) == 0:
                    print(f"Saved candle entry features: {saved:,}")
        if operations:
            target.bulk_write(operations, ordered=False)
            saved += len(operations)
            operations.clear()
        stats["counts"].update({"candidate_rows_scanned": scanned, "saved_feature_docs": saved, "first_anchor_time": epoch_to_dt(first_anchor_time).isoformat(sep=" ") if first_anchor_time else None, "last_anchor_time": epoch_to_dt(last_anchor_time).isoformat(sep=" ") if last_anchor_time else None, "m5_raw_rows_loaded": len(m5_data["times"]), "m5_raw_rows_skipped": m5_data["skipped"], "m15_raw_rows_loaded": len(m15_data["times"]), "m15_raw_rows_skipped": m15_data["skipped"]})
        stats["side_counts"] = dict(side_counter)
        stats["skip_reasons"] = dict(skip_counter)
        report_path = write_report(args, "success", stats)
        print("SCE candle entry features built.")
        print(f"Saved feature docs: {saved:,}")
        print(f"Report: {report_path}")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        print("ERROR: SCE candle entry feature build failed.")
        print(error_text)
        report_path = write_report(args, "failed", stats, error_text)
        print(f"Failure report: {report_path}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
