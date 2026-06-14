# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Build Candle Book Direction Labels M15 v3

Location:
    Book_Quality/02_labels/candle_book/03_build_candle_book_direction_labels_m15_v3.py

Purpose:
    Build labels for the Candle Book Direction Model v3.

Concept:
    M1  = letters
    M5  = words
    M15 = sentence

    At each closed M15 candle, we read previous candles later in the feature step.
    For the label only, we look ahead into the next 30 minutes and ask:
        - Is upward pressure stronger?
        - Is downward pressure stronger?
        - Or is the difference unclear?

Model output later:
    p_buy
    p_sell
    p_unclear
    book_direction = BUY / SELL / UNCLEAR
    book_gap       = abs(p_buy - p_sell)

Anchor:
    M15 closed candle.

Input raw collection:
    xauusd_m15

Output:
    bq2_labels_candle_book_direction_m15_v3

Default horizon:
    2 M15 candles = 30 minutes.

Label logic:
    entry_time  = next M15 candle open
    entry_price = next M15 candle open

    future_high_move = max(high over horizon) - entry_price
    future_low_move  = entry_price - min(low over horizon)
    future_close_move = last close in horizon - entry_price

    buy_pressure  = wick_weight * max(future_high_move, 0) + close_weight * max(future_close_move, 0)
    sell_pressure = wick_weight * max(future_low_move, 0)  + close_weight * max(-future_close_move, 0)

    If total pressure is too small -> UNCLEAR
    If buy/sell gap is too small -> UNCLEAR
    Else larger pressure determines BUY or SELL.

No-Leak:
    Future fields are label-only fields. The feature builder must not read them.

TEST:
    cd C:/Project/BookQuality/bq2/02_labels/candle_book
    python -u 03_build_candle_book_direction_labels_m15_v3.py --limit 1000 --reset

FULL:
    python -u 03_build_candle_book_direction_labels_m15_v3.py --reset
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError


SCRIPT_NAME = "03_build_candle_book_direction_labels_m15_v3.py"
LABEL_VERSION = "candle_book_direction_label_m15_v3"
TARGET_COLLECTION = "bq2_labels_candle_book_direction_m15_v3"
BATCH_SIZE = 1000

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "raw_collections": {
        "m15": "xauusd_m15",
    },
    "bq2_collections": {
        "labels_candle_book_direction": TARGET_COLLECTION,
    },
    "candle_book_direction_label_m15_v3": {
        "horizon_m15": 2,
        "wick_weight": 0.65,
        "close_weight": 0.35,
        "min_total_pressure_usd": 0.80,
        "min_gap_usd": 0.25,
        "min_gap_ratio": 0.08,
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
    cfg["raw_collections"]["m15"] = cfg["raw_collections"].get("m15", "xauusd_m15")
    cfg["bq2_collections"]["labels_candle_book_direction"] = TARGET_COLLECTION
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


def row_dt(row: Dict[str, Any]) -> datetime:
    return parse_dt(row.get("datetime") or row.get("time") or row.get("date"))


def build_label_for_anchor(
    anchor: Dict[str, Any],
    next_bar: Dict[str, Any],
    future_bars: List[Dict[str, Any]],
    cfg: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    if not future_bars:
        return None

    horizon_m15 = int(cfg["horizon_m15"])
    wick_weight = float(cfg["wick_weight"])
    close_weight = float(cfg["close_weight"])
    min_total_pressure_usd = float(cfg["min_total_pressure_usd"])
    min_gap_usd = float(cfg["min_gap_usd"])
    min_gap_ratio = float(cfg["min_gap_ratio"])

    anchor_time = row_dt(anchor)
    entry_time = row_dt(next_bar)
    entry_price = sf(next_bar["open"])

    max_high = max(sf(b["high"]) for b in future_bars)
    min_low = min(sf(b["low"]) for b in future_bars)
    last_close = sf(future_bars[-1]["close"])

    future_high_move_usd = max_high - entry_price
    future_low_move_usd = entry_price - min_low
    future_close_move_usd = last_close - entry_price

    buy_pressure = (wick_weight * max(future_high_move_usd, 0.0)) + (close_weight * max(future_close_move_usd, 0.0))
    sell_pressure = (wick_weight * max(future_low_move_usd, 0.0)) + (close_weight * max(-future_close_move_usd, 0.0))

    total_pressure = buy_pressure + sell_pressure
    pressure_gap = buy_pressure - sell_pressure
    abs_gap = abs(pressure_gap)
    gap_ratio = abs_gap / total_pressure if total_pressure > 1e-12 else 0.0

    if total_pressure < min_total_pressure_usd:
        book_direction = "UNCLEAR"
        label_method = "low_total_pressure"
    elif abs_gap < min_gap_usd:
        book_direction = "UNCLEAR"
        label_method = "small_gap_usd"
    elif gap_ratio < min_gap_ratio:
        book_direction = "UNCLEAR"
        label_method = "small_gap_ratio"
    elif pressure_gap > 0:
        book_direction = "BUY"
        label_method = "buy_pressure_dominant"
    else:
        book_direction = "SELL"
        label_method = "sell_pressure_dominant"

    soft_buy_score = buy_pressure / total_pressure if total_pressure > 1e-12 else 0.0
    soft_sell_score = sell_pressure / total_pressure if total_pressure > 1e-12 else 0.0
    soft_unclear_score = 1.0 - min(1.0, gap_ratio) if total_pressure >= min_total_pressure_usd else 1.0

    return {
        "anchor_time": anchor_time,
        "decision_time": entry_time,
        "entry_time": entry_time,
        "entry_price": entry_price,

        "label_version": LABEL_VERSION,
        "anchor_timeframe": "M15",
        "future_horizon_m15": len(future_bars),
        "future_horizon_minutes": 15 * len(future_bars),

        "label_config": {
            "horizon_m15": horizon_m15,
            "wick_weight": wick_weight,
            "close_weight": close_weight,
            "min_total_pressure_usd": min_total_pressure_usd,
            "min_gap_usd": min_gap_usd,
            "min_gap_ratio": min_gap_ratio,
        },

        "y": {
            "book_direction": book_direction,
            "label_method": label_method,
            "future_high_move_usd": float(future_high_move_usd),
            "future_low_move_usd": float(future_low_move_usd),
            "future_close_move_usd": float(future_close_move_usd),
            "future_max_high": float(max_high),
            "future_min_low": float(min_low),
            "future_last_close": float(last_close),
            "buy_pressure": float(buy_pressure),
            "sell_pressure": float(sell_pressure),
            "pressure_gap": float(pressure_gap),
            "abs_pressure_gap": float(abs_gap),
            "gap_ratio": float(gap_ratio),
            "soft_buy_score": float(soft_buy_score),
            "soft_sell_score": float(soft_sell_score),
            "soft_unclear_score": float(soft_unclear_score),
        },
        "no_leak_note": "Future fields are labels only. Feature builder must not read y/future fields.",
    }


def make_update_op(symbol: str, label_doc: Dict[str, Any]) -> UpdateOne:
    now = utc_now()
    doc = {
        "symbol": symbol,
        **label_doc,
        "created_at": now,
        "updated_at": now,
    }
    created_at = doc.pop("created_at")
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
    jp = reports_dir() / f"candle_book_direction_labels_m15_v3_report_{rs}.json"
    tp = reports_dir() / f"candle_book_direction_labels_m15_v3_report_{rs}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)

    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "Book & Quality v2 - Candle Book Direction Labels M15 v3 Report",
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
    for k, v in report["label_config"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Counts", "-" * 74]
    for k, v in report["counts"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Direction Distribution", "-" * 74]
    for k, v in report["direction_distribution"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Label Method Distribution", "-" * 74]
    for k, v in report["label_method_distribution"].items():
        lines.append(f"{k:<34}: {v}")

    lines += ["", "Pressure Summary", "-" * 74]
    for k, v in report["pressure_summary"].items():
        lines.append(f"{k:<34}: {v}")

    if report["errors"]:
        lines += ["", "Errors", "-" * 74]
        lines += [f"- {e}" for e in report["errors"][:50]]

    tp.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Candle Book Direction Labels M15 v3.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--horizon-m15", type=int, default=None)
    parser.add_argument("--min-total-pressure-usd", type=float, default=None)
    parser.add_argument("--min-gap-usd", type=float, default=None)
    parser.add_argument("--min-gap-ratio", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    rs = stamp(started)

    cfg = load_config()
    db_name = cfg["database"]
    symbol = cfg["symbol"]
    raw_coll = cfg["raw_collections"]["m15"]
    target_coll = cfg["bq2_collections"]["labels_candle_book_direction"]

    lcfg = dict(cfg.get("candle_book_direction_label_m15_v3", DEFAULT_CONFIG["candle_book_direction_label_m15_v3"]))
    if args.horizon_m15 is not None:
        lcfg["horizon_m15"] = args.horizon_m15
    if args.min_total_pressure_usd is not None:
        lcfg["min_total_pressure_usd"] = args.min_total_pressure_usd
    if args.min_gap_usd is not None:
        lcfg["min_gap_usd"] = args.min_gap_usd
    if args.min_gap_ratio is not None:
        lcfg["min_gap_ratio"] = args.min_gap_ratio

    horizon_m15 = int(lcfg["horizon_m15"])

    pressure_acc = {
        "buy_pressure_sum": 0.0,
        "sell_pressure_sum": 0.0,
        "abs_gap_sum": 0.0,
        "gap_ratio_sum": 0.0,
        "count": 0,
    }

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"candle_book_direction_labels_m15_v3_{rs}",
        "status": "running",
        "database": db_name,
        "symbol": symbol,
        "start_time": started.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "collections": {
            "m15_source": raw_coll,
            "target": target_coll,
        },
        "label_config": {
            "label_version": LABEL_VERSION,
            "anchor_timeframe": "M15",
            "entry_rule": "entry_time = next M15 open after anchor_time",
            "purpose": "predict directional candle pressure in the next 30 minutes",
            **lcfg,
        },
        "counts": {
            "m15_loaded": 0,
            "anchors_seen": 0,
            "labels_built": 0,
            "upserted_or_modified": 0,
            "inserted": 0,
            "matched": 0,
            "modified": 0,
            "skipped_existing": 0,
            "skipped_no_future_path": 0,
            "row_errors": 0,
        },
        "direction_distribution": {},
        "label_method_distribution": {},
        "pressure_summary": {},
        "errors": [],
        "args": vars(args),
    }

    print("=== Book & Quality v2 - Build Candle Book Direction Labels M15 v3 ===", flush=True)
    print(f"Database : {db_name}", flush=True)
    print(f"Symbol   : {symbol}", flush=True)
    print(f"Source   : {raw_coll}", flush=True)
    print(f"Target   : {target_coll}", flush=True)

    try:
        db = connect(cfg)

        if args.reset:
            deleted = db[target_coll].delete_many({})
            report["reset_deleted_count"] = deleted.deleted_count
            print(f"[RESET] Deleted {deleted.deleted_count:,} docs from {target_coll}", flush=True)

        db[target_coll].create_index([("symbol", ASCENDING), ("anchor_time", ASCENDING)], unique=True, name="uq_symbol_anchor_time")
        db[target_coll].create_index([("label_version", ASCENDING), ("anchor_time", ASCENDING)], name="ix_label_version_anchor_time")
        db[target_coll].create_index([("y.book_direction", ASCENDING), ("anchor_time", ASCENDING)], name="ix_book_direction_anchor_time")
        db[target_coll].create_index([("y.gap_ratio", ASCENDING), ("anchor_time", ASCENDING)], name="ix_gap_ratio_anchor_time")

        query: Dict[str, Any] = {}
        if args.start or args.end:
            time_filter: Dict[str, Any] = {}
            if args.start:
                time_filter["$gte"] = parse_dt_optional(args.start)
            if args.end:
                time_filter["$lte"] = parse_dt_optional(args.end)
            query["datetime"] = time_filter

        rows = list(
            db[raw_coll]
            .find(query, projection={"_id": 0, "datetime": 1, "open": 1, "high": 1, "low": 1, "close": 1})
            .sort("datetime", ASCENDING)
        )
        report["counts"]["m15_loaded"] = len(rows)

        if len(rows) <= horizon_m15 + 1:
            raise RuntimeError("Not enough M15 rows to build labels.")

        ops: List[UpdateOne] = []
        max_anchor_index = len(rows) - horizon_m15 - 1

        for i in range(max_anchor_index):
            try:
                if args.limit is not None and report["counts"]["anchors_seen"] >= args.limit:
                    break

                anchor = rows[i]
                next_bar = rows[i + 1]
                future_bars = rows[i + 1:i + 1 + horizon_m15]
                anchor_time = row_dt(anchor)
                report["counts"]["anchors_seen"] += 1

                if args.skip_existing:
                    exists = db[target_coll].find_one({"symbol": symbol, "anchor_time": anchor_time}, projection={"_id": 1})
                    if exists:
                        report["counts"]["skipped_existing"] += 1
                        continue

                label_doc = build_label_for_anchor(anchor, next_bar, future_bars, lcfg)
                if not label_doc:
                    report["counts"]["skipped_no_future_path"] += 1
                    continue

                y = label_doc["y"]
                direction = y["book_direction"]
                method = y["label_method"]
                report["direction_distribution"][direction] = report["direction_distribution"].get(direction, 0) + 1
                report["label_method_distribution"][method] = report["label_method_distribution"].get(method, 0) + 1

                pressure_acc["buy_pressure_sum"] += float(y["buy_pressure"])
                pressure_acc["sell_pressure_sum"] += float(y["sell_pressure"])
                pressure_acc["abs_gap_sum"] += float(y["abs_pressure_gap"])
                pressure_acc["gap_ratio_sum"] += float(y["gap_ratio"])
                pressure_acc["count"] += 1

                ops.append(make_update_op(symbol, label_doc))
                report["counts"]["labels_built"] += 1

                if len(ops) >= BATCH_SIZE:
                    execute_bulk(db, target_coll, ops, report)
                    ops = []
                    print(
                        f"[PROGRESS] anchors={report['counts']['anchors_seen']:,} "
                        f"built={report['counts']['labels_built']:,} "
                        f"written≈{report['counts']['upserted_or_modified']:,}",
                        flush=True,
                    )

            except Exception as row_exc:
                report["counts"]["row_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(f"i={i} anchor={anchor.get('datetime')} | {row_exc}")

        if ops:
            execute_bulk(db, target_coll, ops, report)

        cnt = pressure_acc["count"]
        if cnt:
            report["pressure_summary"] = {
                "avg_buy_pressure": round(pressure_acc["buy_pressure_sum"] / cnt, 6),
                "avg_sell_pressure": round(pressure_acc["sell_pressure_sum"] / cnt, 6),
                "avg_abs_pressure_gap": round(pressure_acc["abs_gap_sum"] / cnt, 6),
                "avg_gap_ratio": round(pressure_acc["gap_ratio_sum"] / cnt, 6),
            }

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
    print(f"Status            : {report['status']}", flush=True)
    print(f"M15 Loaded        : {report['counts']['m15_loaded']:,}", flush=True)
    print(f"Anchors Seen      : {report['counts']['anchors_seen']:,}", flush=True)
    print(f"Labels Built      : {report['counts']['labels_built']:,}", flush=True)
    print(f"Upserted/Modified : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Row Errors        : {report['counts']['row_errors']:,}", flush=True)
    print(f"Direction Dist    : {report['direction_distribution']}", flush=True)
    print(f"Pressure Summary  : {report['pressure_summary']}", flush=True)
    print(f"TXT Report        : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
