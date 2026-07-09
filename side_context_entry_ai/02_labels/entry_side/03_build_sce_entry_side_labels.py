# -*- coding: utf-8 -*-
"""
SCE Entry Side Label Builder

Location:
02_labels/entry_side/

Report:
02_labels/entry_side/reports/

Runtime parameters use argparse:
python -u .\02_labels\entry_side\03_build_sce_entry_side_labels.py --limit 50000 --atr-period 14 --lookahead-m1-bars 240 --batch-size 1000
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
import traceback
from array import array
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from pymongo import ASCENDING, MongoClient, ReplaceOne
except ImportError as exc:
    print("ERROR: pymongo is not installed. Run: pip install pymongo")
    raise exc

from config.sce_project_config import (
    COLL_LABELS_ENTRY_SIDE_M5_V1,
    DB_NAME,
    ENTRY_LABELS,
    ENTRY_TP_BUCKETS,
    FAIL_RATIO_OF_TP,
    LABEL_VERSION_ENTRY_SIDE_M5_V1,
    MONGO_URI,
    PROJECT_CODE,
    PROJECT_NAME,
    RAW_M1_AS_FEATURE,
    SIDES,
    TP_LEVELS_ATR,
)

RUN_TYPE = "sce_build_entry_side_labels"
RUN_ID = f"{RUN_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SECONDS_M5 = 300


def parse_optional_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"]:
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            pass
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCE side-based entry labels.")
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--database", default=DB_NAME)
    parser.add_argument("--target-collection", default=COLL_LABELS_ENTRY_SIDE_M5_V1)
    parser.add_argument("--limit", type=int, default=50000, help="M5 anchor limit. Use 0 for full run.")
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--lookahead-m1-bars", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=1000)
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
    return {
        "mapping": {
            "M1": {"collection": "xauusd_m1", "datetime_field": "datetime", "ohlc_fields": {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}},
            "M5": {"collection": "xauusd_m5", "datetime_field": "datetime", "ohlc_fields": {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}},
        }
    }


def get_mapping(raw_config: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    mapping = raw_config.get("mapping", {})
    m1_cfg = mapping.get("M1")
    m5_cfg = mapping.get("M5")
    if not m1_cfg or not m5_cfg:
        raise RuntimeError("M1/M5 mapping not found. Run 01_database/inspect_raw first.")
    return m1_cfg, m5_cfg


def build_time_query(time_field: str, start_time: Optional[datetime], end_time: Optional[datetime]) -> Dict[str, Any]:
    query: Dict[str, Any] = {}
    if start_time or end_time:
        f: Dict[str, Any] = {}
        if start_time:
            f["$gte"] = start_time
        if end_time:
            f["$lte"] = end_time
        query[time_field] = f
    return query


def load_m1_arrays(db, m1_cfg: Dict[str, Any]) -> tuple[array, array, array, int]:
    coll = db[m1_cfg["collection"]]
    tf = m1_cfg["datetime_field"]
    hf = m1_cfg["ohlc_fields"]["high"]
    lf = m1_cfg["ohlc_fields"]["low"]
    times, highs, lows = array("q"), array("d"), array("d")
    skipped = 0
    cur = coll.find({}, {tf: 1, hf: 1, lf: 1, "_id": 0}).sort(tf, ASCENDING).batch_size(10000)
    for doc in cur:
        t = dt_to_epoch_seconds(doc.get(tf))
        h = doc.get(hf)
        l = doc.get(lf)
        if t is None or h is None or l is None:
            skipped += 1
            continue
        times.append(t)
        highs.append(float(h))
        lows.append(float(l))
    if len(times) == 0:
        raise RuntimeError("No valid M1 rows loaded.")
    return times, highs, lows, skipped


def load_m5_rows(db, m5_cfg: Dict[str, Any], args: argparse.Namespace) -> tuple[List[Dict[str, Any]], int]:
    coll = db[m5_cfg["collection"]]
    tf = m5_cfg["datetime_field"]
    ohlc = m5_cfg["ohlc_fields"]
    start_time = parse_optional_time(args.start_time)
    end_time = parse_optional_time(args.end_time)
    query = build_time_query(tf, start_time, end_time)
    projection = {tf: 1, ohlc["open"]: 1, ohlc["high"]: 1, ohlc["low"]: 1, ohlc["close"]: 1, "_id": 0}
    rows: List[Dict[str, Any]] = []
    skipped = 0
    cur = coll.find(query, projection).sort(tf, ASCENDING).batch_size(10000)
    for doc in cur:
        t = dt_to_epoch_seconds(doc.get(tf))
        try:
            o, h, l, c = float(doc[ohlc["open"]]), float(doc[ohlc["high"]]), float(doc[ohlc["low"]]), float(doc[ohlc["close"]])
        except Exception:
            skipped += 1
            continue
        if t is None:
            skipped += 1
            continue
        rows.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        if args.limit > 0 and len(rows) >= args.limit:
            break
    return rows, skipped


def compute_atr(rows: List[Dict[str, Any]], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(rows)
    window: Deque[float] = deque()
    total = 0.0
    prev_close: Optional[float] = None
    for i, row in enumerate(rows):
        h, l, c = row["high"], row["low"], row["close"]
        tr = h - l if prev_close is None else max(h - l, abs(h - prev_close), abs(l - prev_close))
        window.append(tr)
        total += tr
        if len(window) > period:
            total -= window.popleft()
        if len(window) == period:
            out[i] = total / period
        prev_close = c
    return out


def evaluate_side(side: str, entry_price: float, atr: float, start_idx: int, m1_times: array, highs: array, lows: array, lookahead: int) -> Dict[str, Any]:
    max_idx = min(len(m1_times), start_idx + lookahead)
    tp_results: List[Dict[str, Any]] = []
    success_levels: List[float] = []
    fail_levels: List[float] = []
    wait_levels: List[float] = []
    ambiguous_count = 0
    for tp_atr in TP_LEVELS_ATR:
        tp_dist = tp_atr * atr
        fail_dist = FAIL_RATIO_OF_TP * tp_dist
        target = entry_price + tp_dist if side == "BUY" else entry_price - tp_dist
        fail = entry_price - fail_dist if side == "BUY" else entry_price + fail_dist
        outcome, touch_time, bars_to_touch, reason = "WAIT", None, None, "no_touch"
        for j in range(start_idx, max_idx):
            high, low = highs[j], lows[j]
            target_hit = high >= target if side == "BUY" else low <= target
            fail_hit = low <= fail if side == "BUY" else high >= fail
            if target_hit and fail_hit:
                outcome, touch_time, bars_to_touch, reason = "FAIL", int(m1_times[j]), j - start_idx, "target_and_fail_same_m1_bar_conservative_fail"
                ambiguous_count += 1
                break
            if fail_hit:
                outcome, touch_time, bars_to_touch, reason = "FAIL", int(m1_times[j]), j - start_idx, "fail_first"
                break
            if target_hit:
                outcome, touch_time, bars_to_touch, reason = "SUCCESS", int(m1_times[j]), j - start_idx, "target_first"
                break
        if outcome == "SUCCESS": success_levels.append(tp_atr)
        elif outcome == "FAIL": fail_levels.append(tp_atr)
        else: wait_levels.append(tp_atr)
        tp_results.append({"tp_atr": tp_atr, "target_price": round(target, 5), "fail_price": round(fail, 5), "outcome": outcome, "touch_time": epoch_to_dt(touch_time) if touch_time else None, "bars_to_touch": bars_to_touch, "touch_reason": reason})
    max_success = max(success_levels) if success_levels else None
    if max_success is None:
        entry_label = "BAD_ENTRY" if fail_levels else "WAIT"
        entry_tp_bucket = "NO_TRADE"
    else:
        entry_label = "GOOD_ENTRY"
        entry_tp_bucket = "SMALL" if max_success < 0.8 else "NORMAL" if max_success < 1.6 else "LARGE"
    return {"entry_label": entry_label, "entry_tp_bucket": entry_tp_bucket, "max_entry_success_tp_atr": max_success, "success_tp_levels_atr": success_levels, "fail_tp_levels_atr": fail_levels, "wait_tp_levels_atr": wait_levels, "tp_results": tp_results, "ambiguous_same_bar_count": ambiguous_count}


def write_report(args: argparse.Namespace, status: str, stats: Dict[str, Any], error_text: str = "") -> Path:
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{RUN_ID}.txt"
    lines = []
    lines.append("Side Context Entry AI - Entry Side Label Builder Report")
    lines.append("=" * 68)
    lines.append(f"run_id                   : {RUN_ID}")
    lines.append(f"status                   : {status}")
    lines.append(f"project_name             : {PROJECT_NAME}")
    lines.append(f"project_code             : {PROJECT_CODE}")
    lines.append(f"database                 : {args.database}")
    lines.append(f"target_collection        : {args.target_collection}")
    lines.append(f"label_version            : {LABEL_VERSION_ENTRY_SIDE_M5_V1}")
    lines.append("raw_data_modified         : False")
    lines.append(f"m1_as_feature             : {RAW_M1_AS_FEATURE}")
    lines.append("report_folder_policy      : local_section_reports")
    lines.append("")
    lines.append("Runtime parameters:")
    for key in ["limit", "start_time", "end_time", "atr_period", "lookahead_m1_bars", "batch_size", "clear_existing"]:
        lines.append(f"{key:26}: {getattr(args, key)}")
    lines.append("")
    lines.append("Locked label parameters:")
    lines.append(f"sides                    : {SIDES}")
    lines.append(f"entry_labels             : {ENTRY_LABELS}")
    lines.append(f"entry_tp_buckets          : {ENTRY_TP_BUCKETS}")
    lines.append(f"tp_levels_atr            : {TP_LEVELS_ATR}")
    lines.append(f"fail_ratio_of_tp          : {FAIL_RATIO_OF_TP}")
    lines.append("")
    lines.append("Counts:")
    for key, value in stats.get("counts", {}).items():
        lines.append(f"{key:26}: {value}")
    lines.append("")
    lines.append("Entry label distribution:")
    for key, value in sorted(stats.get("entry_label_counts", {}).items()):
        lines.append(f"{key:26}: {value}")
    lines.append("")
    lines.append("TP bucket distribution:")
    for key, value in sorted(stats.get("tp_bucket_counts", {}).items()):
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
    stats: Dict[str, Any] = {"counts": {}, "entry_label_counts": {}, "tp_bucket_counts": {}, "side_counts": {}, "skip_reasons": {}}
    try:
        raw_config = load_raw_config()
        m1_cfg, m5_cfg = get_mapping(raw_config)
        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[args.database]
        target = db[args.target_collection]
        if args.clear_existing == 1:
            clear_result = target.delete_many({})
            stats["counts"]["cleared_existing_docs"] = clear_result.deleted_count
        else:
            stats["counts"]["cleared_existing_docs"] = 0
        print("Loading M1 arrays...")
        m1_times, m1_highs, m1_lows, m1_skipped = load_m1_arrays(db, m1_cfg)
        print(f"M1 loaded: {len(m1_times):,}")
        print("Loading M5 rows...")
        m5_rows, m5_skipped = load_m5_rows(db, m5_cfg, args)
        print(f"M5 loaded: {len(m5_rows):,}")
        atrs = compute_atr(m5_rows, args.atr_period)
        entry_label_counter: Counter = Counter()
        tp_bucket_counter: Counter = Counter()
        side_counter: Counter = Counter()
        skip_counter: Counter = Counter()
        operations: List[ReplaceOne] = []
        saved_docs = 0
        ambiguous_total = 0
        first_anchor_time = None
        last_anchor_time = None
        for i, row in enumerate(m5_rows):
            anchor_time = int(row["time"])
            entry_time = anchor_time + SECONDS_M5
            atr = atrs[i]
            if first_anchor_time is None:
                first_anchor_time = anchor_time
            last_anchor_time = anchor_time
            if atr is None or atr <= 0:
                skip_counter["atr_not_available"] += 1
                continue
            start_idx = bisect.bisect_left(m1_times, entry_time)
            if start_idx >= len(m1_times):
                skip_counter["future_m1_not_available"] += 1
                continue
            if start_idx + args.lookahead_m1_bars > len(m1_times):
                skip_counter["not_enough_future_m1_bars"] += 1
                continue
            for side in SIDES:
                result = evaluate_side(side, row["close"], atr, start_idx, m1_times, m1_highs, m1_lows, args.lookahead_m1_bars)
                ambiguous_total += result["ambiguous_same_bar_count"]
                now = utc_now()
                doc = {"anchor_time": epoch_to_dt(anchor_time), "entry_time": epoch_to_dt(entry_time), "side": side, "entry_price": round(float(row["close"]), 5), "atr_tf": "M5", "atr_period": args.atr_period, "atr_value": round(float(atr), 5), "label_version": LABEL_VERSION_ENTRY_SIDE_M5_V1, "run_id": RUN_ID, "entry_label": result["entry_label"], "entry_tp_bucket": result["entry_tp_bucket"], "max_entry_success_tp_atr": result["max_entry_success_tp_atr"], "success_tp_levels_atr": result["success_tp_levels_atr"], "fail_tp_levels_atr": result["fail_tp_levels_atr"], "wait_tp_levels_atr": result["wait_tp_levels_atr"], "tp_results": result["tp_results"], "lookahead_m1_bars": args.lookahead_m1_bars, "fail_ratio_of_tp": FAIL_RATIO_OF_TP, "m1_as_feature": RAW_M1_AS_FEATURE, "created_at": now, "updated_at": now}
                operations.append(ReplaceOne({"anchor_time": doc["anchor_time"], "side": side}, doc, upsert=True))
                entry_label_counter[doc["entry_label"]] += 1
                tp_bucket_counter[doc["entry_tp_bucket"]] += 1
                side_counter[side] += 1
                if len(operations) >= args.batch_size:
                    target.bulk_write(operations, ordered=False)
                    saved_docs += len(operations)
                    operations.clear()
                    if saved_docs % (args.batch_size * 10) == 0:
                        print(f"Saved labels: {saved_docs:,}")
        if operations:
            target.bulk_write(operations, ordered=False)
            saved_docs += len(operations)
            operations.clear()
        stats["counts"].update({"m1_loaded": len(m1_times), "m1_skipped": m1_skipped, "m5_loaded": len(m5_rows), "m5_skipped": m5_skipped, "saved_candidate_labels": saved_docs, "ambiguous_same_bar_total": ambiguous_total, "first_anchor_time": epoch_to_dt(first_anchor_time).isoformat(sep=" ") if first_anchor_time else None, "last_anchor_time": epoch_to_dt(last_anchor_time).isoformat(sep=" ") if last_anchor_time else None})
        stats["entry_label_counts"] = dict(entry_label_counter)
        stats["tp_bucket_counts"] = dict(tp_bucket_counter)
        stats["side_counts"] = dict(side_counter)
        stats["skip_reasons"] = dict(skip_counter)
        report_path = write_report(args, "success", stats)
        print("SCE entry-side labels built.")
        print(f"Saved labels: {saved_docs:,}")
        print(f"Report: {report_path}")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        print("ERROR: SCE entry-side label build failed.")
        print(error_text)
        report_path = write_report(args, "failed", stats, error_text)
        print(f"Failure report: {report_path}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
