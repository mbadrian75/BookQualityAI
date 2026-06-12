# -*- coding: utf-8 -*-
"""
Book & Quality - Build Book Entry Labels v1

Path:
    Book_Quality/02_labels/book_entry/03_build_book_entry_labels_v1.py

Purpose:
    Build labels for the Book Entry Model.

Inputs:
    Raw MongoDB collections:
        xauusd_m15  -> anchor candles and next M15 entry candle
        xauusd_m1   -> future path evaluation

Output:
    bq_labels_book_entry_m15_v1

Label v1:
    anchor_time = closed M15 candle datetime
    entry_time  = next M15 candle datetime
    entry_price = next M15 open

    BUY TP  = entry_price + tp_usd
    BUY SL  = entry_price - sl_usd
    SELL TP = entry_price - tp_usd
    SELL SL = entry_price + sl_usd

    Default:
        tp_usd = 2.0
        sl_usd = 2.0
        horizon_m15 = 8

Reports:
    Always creates JSON and TXT reports:
        Book_Quality/02_labels/reports/book_entry_labels_v1_report_YYYYMMDD_HHMMSS.json
        Book_Quality/02_labels/reports/book_entry_labels_v1_report_YYYYMMDD_HHMMSS.txt

Run:
    cd C:\Project\Book_Quality\02_labels\book_entry
    python -u 03_build_book_entry_labels_v1.py

Quick test:
    python -u 03_build_book_entry_labels_v1.py --limit 1000 --reset

Full rebuild:
    python -u 03_build_book_entry_labels_v1.py --reset
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.database import Database


SCRIPT_NAME = "03_build_book_entry_labels_v1.py"

DEFAULT_MONGO_CONFIG: Dict[str, Any] = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "raw_collections": {
        "m1": "xauusd_m1",
        "m5": "xauusd_m5",
        "m15": "xauusd_m15",
        "m30": "xauusd_m30",
        "h1": "xauusd_h1",
        "h4": "xauusd_h4",
        "d1": "xauusd_d1"
    },
    "bq_collections": {
        "labels_book_entry": "bq_labels_book_entry_m15_v1"
    }
}

DEFAULT_LABEL_CONFIG: Dict[str, Any] = {
    "label_version": "book_entry_label_v1",
    "entry_rule": "Entry at next M15 open after anchor_time.",
    "tp_usd": 2.0,
    "sl_usd": 2.0,
    "horizon_m15": 8,
    "ambiguous_policy": "NOTRADE"
}

TIME_FIELD = "datetime"
OPEN_FIELD = "open"
HIGH_FIELD = "high"
LOW_FIELD = "low"
CLOSE_FIELD = "close"
BATCH_SIZE = 1000


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def get_project_root() -> Path:
    # Book_Quality/02_labels/book_entry/this_script.py
    return Path(__file__).resolve().parents[2]


def reports_dir() -> Path:
    path = get_project_root() / "02_labels" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}", flush=True)
        return None


def load_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    root = get_project_root()

    mongo_config = dict(DEFAULT_MONGO_CONFIG)
    mongo_file = read_json(root / "00_config" / "mongo_config.json")
    if mongo_file:
        mongo_config.update(mongo_file)
        mongo_config["raw_collections"] = {
            **DEFAULT_MONGO_CONFIG["raw_collections"],
            **mongo_file.get("raw_collections", {})
        }
        mongo_config["bq_collections"] = {
            **DEFAULT_MONGO_CONFIG["bq_collections"],
            **mongo_file.get("bq_collections", {})
        }

    label_config = dict(DEFAULT_LABEL_CONFIG)
    label_file = read_json(root / "00_config" / "label_config_v1.json")
    if label_file and "book_entry_label_v1" in label_file:
        label_config.update(label_file["book_entry_label_v1"])

    label_config["label_version"] = str(label_config.get("label_version", "book_entry_label_v1"))
    label_config["tp_usd"] = float(label_config.get("tp_usd", 2.0))
    label_config["sl_usd"] = float(label_config.get("sl_usd", 2.0))
    label_config["horizon_m15"] = int(label_config.get("horizon_m15", 8))
    label_config["ambiguous_policy"] = str(label_config.get("ambiguous_policy", "NOTRADE"))

    return mongo_config, label_config


def connect(config: Dict[str, Any]) -> Database:
    client = MongoClient(config["mongo_uri"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[config["database"]]


def parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def as_float(value: Any) -> float:
    return float(value)


def load_future_m1(db: Database, m1_coll: str, entry_time: datetime, horizon_m15: int) -> List[Dict[str, Any]]:
    end_time = entry_time + timedelta(minutes=15 * horizon_m15)
    return list(
        db[m1_coll]
        .find(
            {TIME_FIELD: {"$gte": entry_time, "$lt": end_time}},
            projection={
                TIME_FIELD: 1,
                OPEN_FIELD: 1,
                HIGH_FIELD: 1,
                LOW_FIELD: 1,
                CLOSE_FIELD: 1,
                "_id": 0,
            },
        )
        .sort(TIME_FIELD, ASCENDING)
    )


def evaluate_side(side: str, entry_price: float, m1_docs: List[Dict[str, Any]], tp_usd: float, sl_usd: float) -> Dict[str, Any]:
    if side == "BUY":
        tp_price = entry_price + tp_usd
        sl_price = entry_price - sl_usd
    elif side == "SELL":
        tp_price = entry_price - tp_usd
        sl_price = entry_price + sl_usd
    else:
        raise ValueError(f"Invalid side: {side}")

    tp_hit = False
    sl_hit = False
    ambiguous = False
    bars_to_tp = None
    bars_to_sl = None
    tp_time = None
    sl_time = None
    mfe = 0.0
    mae = 0.0

    for i, doc in enumerate(m1_docs, start=1):
        high = as_float(doc[HIGH_FIELD])
        low = as_float(doc[LOW_FIELD])
        t = parse_dt(doc[TIME_FIELD])

        if side == "BUY":
            favorable = high - entry_price
            adverse = entry_price - low
            tp_in_bar = high >= tp_price
            sl_in_bar = low <= sl_price
        else:
            favorable = entry_price - low
            adverse = high - entry_price
            tp_in_bar = low <= tp_price
            sl_in_bar = high >= sl_price

        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        if tp_in_bar and sl_in_bar:
            tp_hit = True
            sl_hit = True
            ambiguous = True
            bars_to_tp = i
            bars_to_sl = i
            tp_time = t
            sl_time = t
            break

        if tp_in_bar:
            tp_hit = True
            bars_to_tp = i
            tp_time = t
            break

        if sl_in_bar:
            sl_hit = True
            bars_to_sl = i
            sl_time = t
            break

    return {
        "side": side,
        "tp_price": round(float(tp_price), 5),
        "sl_price": round(float(sl_price), 5),
        "tp_hit": tp_hit,
        "sl_hit": sl_hit,
        "ambiguous": ambiguous,
        "bars_to_tp_m1": bars_to_tp,
        "bars_to_sl_m1": bars_to_sl,
        "tp_time": tp_time,
        "sl_time": sl_time,
        "mfe": round(float(mfe), 5),
        "mae": round(float(mae), 5),
    }


def choose_label(buy_path: Dict[str, Any], sell_path: Dict[str, Any]) -> Tuple[str, bool, str]:
    buy_valid = bool(buy_path["tp_hit"] and not buy_path["sl_hit"] and not buy_path["ambiguous"])
    sell_valid = bool(sell_path["tp_hit"] and not sell_path["sl_hit"] and not sell_path["ambiguous"])

    if buy_valid and not sell_valid:
        return "BUY", False, "buy_tp_before_sl"
    if sell_valid and not buy_valid:
        return "SELL", False, "sell_tp_before_sl"
    if buy_valid and sell_valid:
        return "NOTRADE", True, "both_sides_valid"
    if buy_path["ambiguous"] or sell_path["ambiguous"]:
        return "NOTRADE", True, "ambiguous_intrabar"
    return "NOTRADE", False, "no_clear_edge"


def build_doc(symbol: str, anchor_doc: Dict[str, Any], entry_doc: Dict[str, Any], m1_docs: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    anchor_time = parse_dt(anchor_doc[TIME_FIELD])
    entry_time = parse_dt(entry_doc[TIME_FIELD])
    entry_price = as_float(entry_doc[OPEN_FIELD])

    buy_path = evaluate_side("BUY", entry_price, m1_docs, cfg["tp_usd"], cfg["sl_usd"])
    sell_path = evaluate_side("SELL", entry_price, m1_docs, cfg["tp_usd"], cfg["sl_usd"])
    label, is_ambiguous, reason = choose_label(buy_path, sell_path)
    now = utc_now()

    return {
        "symbol": symbol,
        "anchor_time": anchor_time,
        "label_version": cfg["label_version"],
        "entry_rule": "next_m15_open",
        "entry_time": entry_time,
        "entry_price": round(float(entry_price), 5),
        "config": {
            "tp_usd": cfg["tp_usd"],
            "sl_usd": cfg["sl_usd"],
            "horizon_m15": cfg["horizon_m15"],
            "ambiguous_policy": cfg["ambiguous_policy"],
        },
        "buy_path": buy_path,
        "sell_path": sell_path,
        "label": label,
        "y": {
            "buy": 1 if label == "BUY" else 0,
            "sell": 1 if label == "SELL" else 0,
            "notrade": 1 if label == "NOTRADE" else 0,
        },
        "is_ambiguous": is_ambiguous,
        "reason": reason,
        "created_at": now,
        "updated_at": now,
    }


def parse_optional_dt(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)


def make_text_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Book & Quality - Book Entry Labels v1 Report")
    lines.append("=" * 70)
    lines.append(f"Run ID              : {report['run_id']}")
    lines.append(f"Status              : {report['status']}")
    lines.append(f"Database            : {report['database']}")
    lines.append(f"Symbol              : {report['symbol']}")
    lines.append(f"Start Time          : {report['start_time']}")
    lines.append(f"End Time            : {report['end_time']}")
    lines.append(f"Duration Sec        : {report['duration_seconds']}")
    lines.append("")
    lines.append("Config")
    lines.append("-" * 70)
    for k, v in report["label_config"].items():
        lines.append(f"{k:<24}: {v}")
    lines.append("")
    lines.append("Counts")
    lines.append("-" * 70)
    for k, v in report["counts"].items():
        lines.append(f"{k:<24}: {v}")
    lines.append("")
    lines.append("Label Distribution")
    lines.append("-" * 70)
    for k, v in report["label_distribution"].items():
        lines.append(f"{k:<24}: {v}")
    lines.append("")
    lines.append("Reason Distribution")
    lines.append("-" * 70)
    for k, v in report["reason_distribution"].items():
        lines.append(f"{k:<24}: {v}")

    if report["warnings"]:
        lines.append("")
        lines.append("Warnings")
        lines.append("-" * 70)
        for item in report["warnings"]:
            lines.append(f"- {item}")

    if report["errors"]:
        lines.append("")
        lines.append("Errors")
        lines.append("-" * 70)
        for item in report["errors"][:100]:
            lines.append(f"- {item}")

    return "\n".join(lines)


def save_reports(report: Dict[str, Any], run_stamp: str) -> None:
    rdir = reports_dir()
    json_path = rdir / f"book_entry_labels_v1_report_{run_stamp}.json"
    txt_path = rdir / f"book_entry_labels_v1_report_{run_stamp}.txt"

    report["report_json_path"] = str(json_path)
    report["report_txt_path"] = str(txt_path)

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(make_text_report(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Book Entry labels v1.")
    p.add_argument("--limit", type=int, default=None, help="Maximum number of M15 anchors to process.")
    p.add_argument("--reset", action="store_true", help="Delete existing Book Entry labels before processing.")
    p.add_argument("--skip-existing", action="store_true", help="Skip already labeled anchor_time values.")
    p.add_argument("--start", type=str, default=None, help="Start datetime, e.g. 2024-01-01 00:00:00")
    p.add_argument("--end", type=str, default=None, help="End datetime, e.g. 2024-12-31 23:59:59")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start_time = utc_now()
    run_stamp = stamp(start_time)
    run_id = f"book_entry_labels_v1_{run_stamp}"

    mongo_cfg, label_cfg = load_configs()
    db_name = mongo_cfg["database"]
    symbol = mongo_cfg["symbol"]
    m15_coll = mongo_cfg["raw_collections"].get("m15", "xauusd_m15")
    m1_coll = mongo_cfg["raw_collections"].get("m1", "xauusd_m1")
    label_coll = mongo_cfg["bq_collections"].get("labels_book_entry", "bq_labels_book_entry_m15_v1")

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": run_id,
        "project_root": str(get_project_root()),
        "database": db_name,
        "symbol": symbol,
        "start_time": start_time.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "status": "running",
        "collections": {
            "source_m15": m15_coll,
            "source_m1": m1_coll,
            "target_labels": label_coll,
        },
        "args": {
            "limit": args.limit,
            "reset": args.reset,
            "skip_existing": args.skip_existing,
            "start": args.start,
            "end": args.end,
        },
        "label_config": label_cfg,
        "counts": {
            "anchors_read": 0,
            "upserted_or_modified": 0,
            "skipped_existing": 0,
            "skipped_no_entry": 0,
            "skipped_no_future_m1": 0,
            "row_errors": 0,
        },
        "label_distribution": {"BUY": 0, "SELL": 0, "NOTRADE": 0},
        "reason_distribution": {},
        "warnings": [],
        "errors": [],
    }

    print("=== Book & Quality - Build Book Entry Labels v1 ===", flush=True)
    print(f"Project Root : {report['project_root']}", flush=True)
    print(f"Database     : {db_name}", flush=True)
    print(f"Symbol       : {symbol}", flush=True)
    print(f"M15 Source   : {m15_coll}", flush=True)
    print(f"M1 Source    : {m1_coll}", flush=True)
    print(f"Target       : {label_coll}", flush=True)
    print(f"Config       : {label_cfg}", flush=True)

    try:
        db = connect(mongo_cfg)
        out = db[label_coll]

        out.create_index([("symbol", ASCENDING), ("anchor_time", ASCENDING)], unique=True, name="uq_symbol_anchor_time")
        out.create_index([("label_version", ASCENDING), ("label", ASCENDING)], name="ix_label_version_label")

        if args.reset:
            res = out.delete_many({})
            report["reset_deleted_count"] = res.deleted_count
            print(f"[RESET] Deleted {res.deleted_count:,} existing labels.", flush=True)

        query: Dict[str, Any] = {}
        qdt: Dict[str, Any] = {}
        start_arg = parse_optional_dt(args.start)
        end_arg = parse_optional_dt(args.end)
        if start_arg:
            qdt["$gte"] = start_arg
        if end_arg:
            qdt["$lte"] = end_arg
        if qdt:
            query[TIME_FIELD] = qdt

        cursor = (
            db[m15_coll]
            .find(
                query,
                projection={
                    TIME_FIELD: 1,
                    OPEN_FIELD: 1,
                    HIGH_FIELD: 1,
                    LOW_FIELD: 1,
                    CLOSE_FIELD: 1,
                    "_id": 0,
                },
            )
            .sort(TIME_FIELD, ASCENDING)
            .batch_size(2000)
        )

        prev_anchor: Optional[Dict[str, Any]] = None
        bulk: List[UpdateOne] = []

        for current_m15 in cursor:
            # Pair: prev_anchor is anchor, current_m15 is next M15 entry candle.
            if prev_anchor is None:
                prev_anchor = current_m15
                continue

            if args.limit is not None and report["counts"]["anchors_read"] >= args.limit:
                break

            anchor_doc = prev_anchor
            entry_doc = current_m15
            prev_anchor = current_m15

            report["counts"]["anchors_read"] += 1
            anchor_time = parse_dt(anchor_doc[TIME_FIELD])
            entry_time = parse_dt(entry_doc[TIME_FIELD])

            try:
                if args.skip_existing:
                    exists = out.find_one({"symbol": symbol, "anchor_time": anchor_time}, projection={"_id": 1})
                    if exists:
                        report["counts"]["skipped_existing"] += 1
                        continue

                future_m1 = load_future_m1(db, m1_coll, entry_time, label_cfg["horizon_m15"])
                if not future_m1:
                    report["counts"]["skipped_no_future_m1"] += 1
                    continue

                doc = build_doc(symbol, anchor_doc, entry_doc, future_m1, label_cfg)
                label = doc["label"]
                reason = doc["reason"]

                report["label_distribution"][label] += 1
                report["reason_distribution"][reason] = report["reason_distribution"].get(reason, 0) + 1

                bulk.append(
                    UpdateOne(
                        {"symbol": symbol, "anchor_time": doc["anchor_time"]},
                        {"$set": doc},
                        upsert=True,
                    )
                )

                if len(bulk) >= BATCH_SIZE:
                    result = out.bulk_write(bulk, ordered=False)
                    report["counts"]["upserted_or_modified"] += result.upserted_count + result.modified_count
                    bulk = []
                    print(
                        f"[PROGRESS] anchors={report['counts']['anchors_read']:,} "
                        f"written≈{report['counts']['upserted_or_modified']:,} "
                        f"BUY={report['label_distribution']['BUY']:,} "
                        f"SELL={report['label_distribution']['SELL']:,} "
                        f"NOTRADE={report['label_distribution']['NOTRADE']:,}",
                        flush=True,
                    )

            except Exception as row_exc:
                report["counts"]["row_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(f"anchor_time={anchor_time} | {row_exc}")

        if bulk:
            result = out.bulk_write(bulk, ordered=False)
            report["counts"]["upserted_or_modified"] += result.upserted_count + result.modified_count

        report["final_target_count"] = out.estimated_document_count()

        if report["counts"]["row_errors"] > 0:
            report["warnings"].append("Some rows failed. Check JSON report for row-level errors.")
            report["status"] = "success_with_row_errors"
        else:
            report["status"] = "success"

    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc))
        print(f"[ERROR] {exc}", flush=True)

    finally:
        end_time = utc_now()
        report["end_time"] = end_time.isoformat()
        report["duration_seconds"] = round((end_time - start_time).total_seconds(), 3)
        save_reports(report, run_stamp)

    print("\n=== Final Summary ===", flush=True)
    print(f"Status              : {report['status']}", flush=True)
    print(f"Anchors Read        : {report['counts']['anchors_read']:,}", flush=True)
    print(f"Upserted/Modified   : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Skipped Existing    : {report['counts']['skipped_existing']:,}", flush=True)
    print(f"Skipped No Future M1: {report['counts']['skipped_no_future_m1']:,}", flush=True)
    print(f"Row Errors          : {report['counts']['row_errors']:,}", flush=True)
    print(f"BUY                 : {report['label_distribution']['BUY']:,}", flush=True)
    print(f"SELL                : {report['label_distribution']['SELL']:,}", flush=True)
    print(f"NOTRADE             : {report['label_distribution']['NOTRADE']:,}", flush=True)
    print(f"JSON Report         : {report.get('report_json_path')}", flush=True)
    print(f"TXT Report          : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
