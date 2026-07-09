# -*- coding: utf-8 -*-
"""
SCE MA Context Feature Builder

Location:
03_features/ma_context/

Report:
03_features/ma_context/reports/

Purpose:
- Build side-based MA context features for each labeled candidate row.
- Uses M30/H1/H4 raw data only.
- Does not use M1 as feature.
- Writes to sce_features_ma_context_m5_v1.
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
    COLL_FEATURES_MA_CONTEXT_M5_V1,
    COLL_LABELS_ENTRY_SIDE_M5_V1,
    DB_NAME,
    FEATURE_VERSION_MA_CONTEXT_M5_V1,
    LABEL_VERSION_ENTRY_SIDE_M5_V1,
    MA_CONTEXT_TFS,
    MONGO_URI,
    PROJECT_CODE,
    PROJECT_NAME,
    RAW_M1_AS_FEATURE,
    SIDES,
)

RUN_TYPE = "sce_build_ma_context_features"
RUN_ID = f"{RUN_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TF_SECONDS = {"M30": 1800, "H1": 3600, "H4": 14400}
MA_PERIODS = [20, 50, 100]


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
    parser = argparse.ArgumentParser(description="Build SCE MA context side-based features.")
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--database", default=DB_NAME)
    parser.add_argument("--label-collection", default=COLL_LABELS_ENTRY_SIDE_M5_V1)
    parser.add_argument("--target-collection", default=COLL_FEATURES_MA_CONTEXT_M5_V1)
    parser.add_argument("--label-version", default=LABEL_VERSION_ENTRY_SIDE_M5_V1)
    parser.add_argument("--feature-version", default=FEATURE_VERSION_MA_CONTEXT_M5_V1)
    parser.add_argument("--limit", type=int, default=50000, help="Candidate row limit. Use 0 for full run.")
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--slope-lookback", type=int, default=3)
    parser.add_argument("--capacity-lookback", type=int, default=20)
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


def load_raw_config() -> Dict[str, Any]:
    config_path = ROOT / "config" / "sce_raw_collections_detected.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {"mapping": {"M30": {"collection": "xauusd_m30", "datetime_field": "datetime", "ohlc_fields": {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}}, "H1": {"collection": "xauusd_h1", "datetime_field": "datetime", "ohlc_fields": {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}}, "H4": {"collection": "xauusd_h4", "datetime_field": "datetime", "ohlc_fields": {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}}}}


def load_tf_arrays(db, tf: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    coll = db[cfg["collection"]]
    time_f = cfg["datetime_field"]
    ohlc = cfg["ohlc_fields"]
    times, opens, highs, lows, closes = array("q"), array("d"), array("d"), array("d"), array("d")
    skipped = 0
    projection = {time_f: 1, ohlc["open"]: 1, ohlc["high"]: 1, ohlc["low"]: 1, ohlc["close"]: 1, "_id": 0}
    for doc in coll.find({}, projection).sort(time_f, ASCENDING).batch_size(10000):
        t = dt_to_epoch_seconds(doc.get(time_f))
        try:
            o, h, l, c = float(doc[ohlc["open"]]), float(doc[ohlc["high"]]), float(doc[ohlc["low"]]), float(doc[ohlc["close"]])
        except Exception:
            skipped += 1
            continue
        if t is None:
            skipped += 1
            continue
        times.append(t); opens.append(o); highs.append(h); lows.append(l); closes.append(c)
    return {"tf": tf, "times": times, "open": opens, "high": highs, "low": lows, "close": closes, "skipped": skipped}


def rolling_sma(values: array, period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    total = 0.0
    win = deque()
    for i, val in enumerate(values):
        win.append(float(val)); total += float(val)
        if len(win) > period:
            total -= win.popleft()
        if len(win) == period:
            out[i] = total / period
    return out


def enrich_tf_data(data: Dict[str, Any]) -> None:
    closes = data["close"]
    for period in MA_PERIODS:
        data[f"ma{period}"] = rolling_sma(closes, period)


def signed(value: float, side: str) -> float:
    return value if side == "BUY" else -value


def alignment(ma20: float, ma50: float, ma100: float) -> str:
    if ma20 > ma50 > ma100:
        return "bull"
    if ma20 < ma50 < ma100:
        return "bear"
    return "mixed"


def side_alignment_score(align: str, side: str) -> int:
    if align == "bull" and side == "BUY":
        return 1
    if align == "bear" and side == "SELL":
        return 1
    if align == "bull" and side == "SELL":
        return -1
    if align == "bear" and side == "BUY":
        return -1
    return 0


def classify_phase(align_score: int, side_slope_score: float, distance_to_ma20: float, close_price: float) -> str:
    dist_pct = abs(distance_to_ma20) / close_price if close_price else 0.0
    if align_score == 0:
        return "mixed_or_range"
    if side_slope_score <= 0:
        return "weak_or_reversal_risk"
    if dist_pct < 0.0015:
        return "pullback_or_start"
    if dist_pct > 0.012:
        return "late_or_extended"
    return "continuation"


def build_tf_features(tf: str, data: Dict[str, Any], entry_time: int, entry_price: float, side: str, slope_lookback: int, capacity_lookback: int) -> Optional[Dict[str, Any]]:
    tf_sec = TF_SECONDS[tf]
    closed_time_limit = entry_time - tf_sec
    idx = bisect.bisect_right(data["times"], closed_time_limit) - 1
    if idx < 0:
        return None
    if idx < max(MA_PERIODS) - 1:
        return None
    if idx - slope_lookback < 0:
        return None
    ma20 = data["ma20"][idx]
    ma50 = data["ma50"][idx]
    ma100 = data["ma100"][idx]
    if ma20 is None or ma50 is None or ma100 is None:
        return None
    prev_ma20 = data["ma20"][idx - slope_lookback]
    prev_ma50 = data["ma50"][idx - slope_lookback]
    prev_ma100 = data["ma100"][idx - slope_lookback]
    if prev_ma20 is None or prev_ma50 is None or prev_ma100 is None:
        return None
    close_price = float(data["close"][idx])
    align = alignment(ma20, ma50, ma100)
    align_score = side_alignment_score(align, side)
    slope20 = (ma20 - prev_ma20) / slope_lookback
    slope50 = (ma50 - prev_ma50) / slope_lookback
    slope100 = (ma100 - prev_ma100) / slope_lookback
    side_slope20 = signed(slope20, side)
    side_slope50 = signed(slope50, side)
    side_slope100 = signed(slope100, side)
    side_slope_score = (1 if side_slope20 > 0 else 0) + (1 if side_slope50 > 0 else 0) + (1 if side_slope100 > 0 else 0)
    distance_to_ma20 = entry_price - ma20
    distance_to_ma50 = entry_price - ma50
    distance_to_ma100 = entry_price - ma100
    side_distance_to_ma20 = signed(distance_to_ma20, side)
    side_distance_to_ma50 = signed(distance_to_ma50, side)
    side_distance_to_ma100 = signed(distance_to_ma100, side)
    start = max(0, idx - capacity_lookback + 1)
    recent_high = max(float(x) for x in data["high"][start:idx + 1])
    recent_low = min(float(x) for x in data["low"][start:idx + 1])
    capacity = recent_high - entry_price if side == "BUY" else entry_price - recent_low
    capacity_pct = capacity / entry_price if entry_price else 0.0
    trend_strength = abs(ma20 - ma100) / close_price if close_price else 0.0
    phase = classify_phase(align_score, side_slope_score, distance_to_ma20, close_price)
    prefix = tf.lower()
    return {
        f"{prefix}_context_time": epoch_to_dt(int(data["times"][idx])),
        f"{prefix}_close": round(close_price, 5),
        f"{prefix}_ma20": round(ma20, 5),
        f"{prefix}_ma50": round(ma50, 5),
        f"{prefix}_ma100": round(ma100, 5),
        f"{prefix}_ma20_slope": round(slope20, 8),
        f"{prefix}_ma50_slope": round(slope50, 8),
        f"{prefix}_ma100_slope": round(slope100, 8),
        f"{prefix}_alignment": align,
        f"{prefix}_side_alignment_score": align_score,
        f"{prefix}_side_slope20": round(side_slope20, 8),
        f"{prefix}_side_slope50": round(side_slope50, 8),
        f"{prefix}_side_slope100": round(side_slope100, 8),
        f"{prefix}_side_slope_score": side_slope_score,
        f"{prefix}_distance_to_ma20": round(distance_to_ma20, 5),
        f"{prefix}_distance_to_ma50": round(distance_to_ma50, 5),
        f"{prefix}_distance_to_ma100": round(distance_to_ma100, 5),
        f"{prefix}_side_distance_to_ma20": round(side_distance_to_ma20, 5),
        f"{prefix}_side_distance_to_ma50": round(side_distance_to_ma50, 5),
        f"{prefix}_side_distance_to_ma100": round(side_distance_to_ma100, 5),
        f"{prefix}_trend_strength": round(trend_strength, 8),
        f"{prefix}_trend_phase": phase,
        f"{prefix}_context_tp_capacity": round(capacity, 5),
        f"{prefix}_context_tp_capacity_pct": round(capacity_pct, 8),
    }


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
    lines.append("Side Context Entry AI - MA Context Feature Builder Report")
    lines.append("=" * 72)
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
    for key in ["limit", "start_time", "end_time", "slope_lookback", "capacity_lookback", "batch_size", "clear_existing"]:
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
        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[args.database]
        tf_data: Dict[str, Any] = {}
        for tf in MA_CONTEXT_TFS:
            if tf not in mapping:
                raise RuntimeError(f"Missing raw mapping for {tf}. Run inspect_raw first.")
            print(f"Loading {tf} arrays...")
            tf_data[tf] = load_tf_arrays(db, tf, mapping[tf])
            enrich_tf_data(tf_data[tf])
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
            doc: Dict[str, Any] = {"anchor_time": epoch_to_dt(anchor_time), "entry_time": epoch_to_dt(entry_time), "side": side, "entry_price": round(float(entry_price), 5), "feature_version": args.feature_version, "label_version": args.label_version, "run_id": RUN_ID, "source_tfs": MA_CONTEXT_TFS, "created_at": utc_now(), "updated_at": utc_now()}
            side_alignment_total = 0
            side_slope_total = 0
            capacity_values: List[float] = []
            missing = False
            for tf in MA_CONTEXT_TFS:
                f = build_tf_features(tf, tf_data[tf], entry_time, float(entry_price), side, args.slope_lookback, args.capacity_lookback)
                if f is None:
                    skip_counter[f"{tf.lower()}_context_not_available"] += 1
                    missing = True
                    break
                doc.update(f)
                side_alignment_total += int(f[f"{tf.lower()}_side_alignment_score"])
                side_slope_total += int(f[f"{tf.lower()}_side_slope_score"])
                capacity_values.append(float(f[f"{tf.lower()}_context_tp_capacity"]))
            if missing:
                continue
            doc["ma_context_side_alignment_total"] = side_alignment_total
            doc["ma_context_side_slope_total"] = side_slope_total
            doc["ma_context_tp_capacity_min"] = round(min(capacity_values), 5) if capacity_values else None
            doc["ma_context_tp_capacity_avg"] = round(sum(capacity_values) / len(capacity_values), 5) if capacity_values else None
            doc["ma_context_all_aligned"] = side_alignment_total == len(MA_CONTEXT_TFS)
            operations.append(ReplaceOne({"anchor_time": doc["anchor_time"], "side": side}, doc, upsert=True))
            side_counter[side] += 1
            if len(operations) >= args.batch_size:
                target.bulk_write(operations, ordered=False)
                saved += len(operations)
                operations.clear()
                if saved % (args.batch_size * 10) == 0:
                    print(f"Saved MA context features: {saved:,}")
        if operations:
            target.bulk_write(operations, ordered=False)
            saved += len(operations)
            operations.clear()
        stats["counts"].update({"candidate_rows_scanned": scanned, "saved_feature_docs": saved, "first_anchor_time": epoch_to_dt(first_anchor_time).isoformat(sep=" ") if first_anchor_time else None, "last_anchor_time": epoch_to_dt(last_anchor_time).isoformat(sep=" ") if last_anchor_time else None})
        for tf in MA_CONTEXT_TFS:
            stats["counts"][f"{tf.lower()}_raw_rows_loaded"] = len(tf_data[tf]["times"])
            stats["counts"][f"{tf.lower()}_raw_rows_skipped"] = tf_data[tf]["skipped"]
        stats["side_counts"] = dict(side_counter)
        stats["skip_reasons"] = dict(skip_counter)
        report_path = write_report(args, "success", stats)
        print("SCE MA context features built.")
        print(f"Saved feature docs: {saved:,}")
        print(f"Report: {report_path}")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        print("ERROR: SCE MA context feature build failed.")
        print(error_text)
        report_path = write_report(args, "failed", stats, error_text)
        print(f"Failure report: {report_path}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
