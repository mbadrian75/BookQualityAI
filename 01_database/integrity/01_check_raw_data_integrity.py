# -*- coding: utf-8 -*-
"""
Book & Quality - Raw Data Integrity Checker

Location:
    Book_Quality/01_database/integrity/01_check_raw_data_integrity.py

Purpose:
    Checks raw XAUUSD candle collections before label/feature/dataset building.

Important:
    - This script does NOT modify raw collections.
    - This script always creates JSON and TXT reports in:
        Book_Quality/01_database/reports/

Requirements:
    pip install pymongo

Run:
    cd C:\Project\Book_Quality\01_database\integrity
    python -u 01_check_raw_data_integrity.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from pymongo import MongoClient, ASCENDING
from pymongo.database import Database


SCRIPT_NAME = "01_check_raw_data_integrity.py"

DEFAULT_CONFIG = {
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
    }
}

EXPECTED_DELTA: Dict[str, timedelta] = {
    "m1": timedelta(minutes=1),
    "m5": timedelta(minutes=5),
    "m15": timedelta(minutes=15),
    "m30": timedelta(minutes=30),
    "h1": timedelta(hours=1),
    "h4": timedelta(hours=4),
    "d1": timedelta(days=1),
}

TIME_FIELD_CANDIDATES = [
    "datetime", "time", "time_open", "date", "timestamp", "DateTime", "Time"
]

OHLC_CANDIDATES = {
    "open": ["open", "Open", "o"],
    "high": ["high", "High", "h"],
    "low": ["low", "Low", "l"],
    "close": ["close", "Close", "c"],
    "volume": ["volume", "tick_volume", "Volume", "TickVolume", "vol"],
}

# Full gap scan is accurate but may take time on M1.
# Set to an integer such as 500000 for faster first run.
MAX_GAP_SCAN_DOCS: Optional[int] = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def file_stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def get_project_root() -> Path:
    # script: Book_Quality/01_database/integrity/script.py
    return Path(__file__).resolve().parents[2]


def get_reports_dir() -> Path:
    reports_dir = get_project_root() / "01_database" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def load_config() -> Dict[str, Any]:
    config_path = get_project_root() / "00_config" / "mongo_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Could not read config file, using defaults: {config_path} | {exc}")
    return DEFAULT_CONFIG


def connect(config: Dict[str, Any]) -> Database:
    client = MongoClient(config["mongo_uri"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[config["database"]]


def parse_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.replace(tzinfo=None)

    if isinstance(value, (int, float)):
        try:
            if value > 10_000_000_000:
                return datetime.fromtimestamp(value / 1000).replace(tzinfo=None)
            return datetime.fromtimestamp(value).replace(tzinfo=None)
        except Exception:
            return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        text = text.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text).replace(tzinfo=None)
        except Exception:
            pass

        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y.%m.%d %H:%M:%S",
            "%Y.%m.%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d",
            "%Y.%m.%d",
            "%Y/%m/%d",
        ]:
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue

    return None


def detect_time_field(sample: Dict[str, Any]) -> Optional[str]:
    for field in TIME_FIELD_CANDIDATES:
        if field in sample and parse_time(sample.get(field)) is not None:
            return field

    for key, value in sample.items():
        if key == "_id":
            continue
        if parse_time(value) is not None:
            return key

    return None


def detect_ohlc_fields(sample: Dict[str, Any]) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {}
    for logical_name, candidates in OHLC_CANDIDATES.items():
        result[logical_name] = None
        for field in candidates:
            if field in sample:
                result[logical_name] = field
                break
    return result


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def get_first_last(db: Database, coll_name: str, time_field: str) -> Tuple[Optional[Any], Optional[Any]]:
    coll = db[coll_name]
    first = coll.find_one({}, sort=[(time_field, ASCENDING)], projection={time_field: 1, "_id": 0})
    last = coll.find_one({}, sort=[(time_field, -1)], projection={time_field: 1, "_id": 0})
    return (
        first.get(time_field) if first else None,
        last.get(time_field) if last else None
    )


def check_duplicate_examples(db: Database, coll_name: str, time_field: str) -> Dict[str, Any]:
    pipeline = [
        {"$group": {"_id": f"${time_field}", "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 20},
    ]

    examples = list(db[coll_name].aggregate(pipeline, allowDiskUse=True))
    return {
        "has_duplicate_examples": len(examples) > 0,
        "duplicate_examples": [
            {"time": str(item["_id"]), "count": item["count"]}
            for item in examples
        ],
    }


def scan_gaps(db: Database, tf: str, coll_name: str, time_field: str) -> Dict[str, Any]:
    expected = EXPECTED_DELTA[tf]
    cursor = (
        db[coll_name]
        .find({}, projection={time_field: 1, "_id": 0})
        .sort(time_field, ASCENDING)
        .batch_size(5000)
    )

    previous: Optional[datetime] = None
    scanned = 0
    gap_count = 0
    duplicate_or_reverse_count = 0
    examples: List[Dict[str, str]] = []

    for doc in cursor:
        current = parse_time(doc.get(time_field))
        if current is None:
            continue

        scanned += 1

        if previous is not None:
            delta = current - previous

            if delta <= timedelta(0):
                duplicate_or_reverse_count += 1
                if len(examples) < 20:
                    examples.append({
                        "type": "duplicate_or_reverse",
                        "previous": previous.isoformat(sep=" "),
                        "current": current.isoformat(sep=" "),
                        "delta": str(delta),
                    })

            elif delta > expected:
                gap_count += 1
                if len(examples) < 20:
                    examples.append({
                        "type": "gap",
                        "previous": previous.isoformat(sep=" "),
                        "current": current.isoformat(sep=" "),
                        "expected_delta": str(expected),
                        "actual_delta": str(delta),
                    })

        previous = current

        if MAX_GAP_SCAN_DOCS is not None and scanned >= MAX_GAP_SCAN_DOCS:
            break

    return {
        "checked": True,
        "expected_delta": str(expected),
        "scanned": scanned,
        "gap_count": gap_count,
        "duplicate_or_reverse_count": duplicate_or_reverse_count,
        "examples": examples,
        "max_gap_scan_docs": MAX_GAP_SCAN_DOCS,
    }


def check_ohlc_consistency(db: Database, coll_name: str, ohlc_fields: Dict[str, Optional[str]]) -> Dict[str, Any]:
    required = ["open", "high", "low", "close"]
    if any(ohlc_fields.get(k) is None for k in required):
        return {
            "checked": False,
            "reason": "OHLC fields are not fully detected.",
            "scanned": 0,
            "bad_count": None,
            "bad_examples": [],
        }

    open_f = ohlc_fields["open"]
    high_f = ohlc_fields["high"]
    low_f = ohlc_fields["low"]
    close_f = ohlc_fields["close"]

    cursor = db[coll_name].find(
        {},
        projection={open_f: 1, high_f: 1, low_f: 1, close_f: 1, "_id": 1},
        no_cursor_timeout=True,
    )

    scanned = 0
    bad_count = 0
    bad_examples: List[Dict[str, Any]] = []

    try:
        for doc in cursor:
            scanned += 1

            o = to_float(doc.get(open_f))
            h = to_float(doc.get(high_f))
            l = to_float(doc.get(low_f))
            c = to_float(doc.get(close_f))

            is_bad = (
                o is None or h is None or l is None or c is None
                or h < max(o, c)
                or l > min(o, c)
                or h < l
            )

            if is_bad:
                bad_count += 1
                if len(bad_examples) < 20:
                    bad_examples.append({
                        "_id": str(doc.get("_id")),
                        "open": doc.get(open_f),
                        "high": doc.get(high_f),
                        "low": doc.get(low_f),
                        "close": doc.get(close_f),
                    })
    finally:
        cursor.close()

    return {
        "checked": True,
        "scanned": scanned,
        "bad_count": bad_count,
        "bad_examples": bad_examples,
    }


def inspect_collection(db: Database, tf: str, coll_name: str) -> Dict[str, Any]:
    print(f"\n--- {tf.upper()} / {coll_name} ---", flush=True)

    result: Dict[str, Any] = {
        "timeframe": tf,
        "collection": coll_name,
        "exists": False,
        "status": "started",
    }

    if coll_name not in db.list_collection_names():
        print("[MISSING] Collection not found.", flush=True)
        result["status"] = "missing"
        result["error"] = "Collection not found."
        return result

    result["exists"] = True
    coll = db[coll_name]

    count = coll.estimated_document_count()
    sample = coll.find_one()

    result["estimated_count"] = count
    result["sample_keys"] = list(sample.keys()) if sample else []

    print(f"count≈{count:,}", flush=True)

    if not sample:
        result["status"] = "empty"
        result["error"] = "Empty collection."
        print("[WARN] Empty collection.", flush=True)
        return result

    time_field = detect_time_field(sample)
    ohlc_fields = detect_ohlc_fields(sample)

    result["detected_time_field"] = time_field
    result["detected_ohlc_fields"] = ohlc_fields

    print(f"time_field={time_field}", flush=True)
    print(f"ohlc_fields={ohlc_fields}", flush=True)

    if not time_field:
        result["status"] = "failed"
        result["error"] = "Could not detect datetime field."
        return result

    first_raw, last_raw = get_first_last(db, coll_name, time_field)
    first_dt = parse_time(first_raw)
    last_dt = parse_time(last_raw)

    result["first_time_raw"] = str(first_raw)
    result["last_time_raw"] = str(last_raw)
    result["first_time"] = first_dt.isoformat(sep=" ") if first_dt else None
    result["last_time"] = last_dt.isoformat(sep=" ") if last_dt else None

    print(f"first={result['first_time']}", flush=True)
    print(f"last ={result['last_time']}", flush=True)

    print("checking duplicate examples...", flush=True)
    result["duplicates"] = check_duplicate_examples(db, coll_name, time_field)
    print(f"duplicate_examples_found={result['duplicates']['has_duplicate_examples']}", flush=True)

    print("scanning gaps...", flush=True)
    result["gaps"] = scan_gaps(db, tf, coll_name, time_field)
    print(
        f"gaps={result['gaps']['gap_count']:,}, "
        f"duplicate/reverse={result['gaps']['duplicate_or_reverse_count']:,}, "
        f"scanned={result['gaps']['scanned']:,}",
        flush=True
    )

    print("checking OHLC consistency...", flush=True)
    result["ohlc_consistency"] = check_ohlc_consistency(db, coll_name, ohlc_fields)
    if result["ohlc_consistency"]["checked"]:
        print(
            f"ohlc_bad={result['ohlc_consistency']['bad_count']:,}, "
            f"scanned={result['ohlc_consistency']['scanned']:,}",
            flush=True
        )
    else:
        print(f"[WARN] {result['ohlc_consistency']['reason']}", flush=True)

    result["status"] = "success"
    return result


def build_txt_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Book & Quality - Raw Data Integrity Report")
    lines.append("=" * 60)
    lines.append(f"Run ID       : {report['run_id']}")
    lines.append(f"Status       : {report['status']}")
    lines.append(f"Database     : {report['database']}")
    lines.append(f"Symbol       : {report['symbol']}")
    lines.append(f"Start Time   : {report['start_time']}")
    lines.append(f"End Time     : {report['end_time']}")
    lines.append(f"Duration Sec : {report['duration_seconds']}")
    lines.append("")

    lines.append("Summary")
    lines.append("-" * 60)
    for tf, item in report["collections"].items():
        gaps = item.get("gaps", {}).get("gap_count")
        ohlc_bad = item.get("ohlc_consistency", {}).get("bad_count")
        lines.append(
            f"{tf.upper():>3} | exists={item.get('exists')} | "
            f"count≈{item.get('estimated_count')} | "
            f"time_field={item.get('detected_time_field')} | "
            f"gaps={gaps} | ohlc_bad={ohlc_bad} | status={item.get('status')}"
        )

    if report["warnings"]:
        lines.append("")
        lines.append("Warnings")
        lines.append("-" * 60)
        for warning in report["warnings"]:
            lines.append(f"- {warning}")

    if report["errors"]:
        lines.append("")
        lines.append("Errors")
        lines.append("-" * 60)
        for error in report["errors"]:
            lines.append(f"- {error}")

    return "\n".join(lines)


def save_reports(report: Dict[str, Any], stamp: str) -> None:
    reports_dir = get_reports_dir()
    json_path = reports_dir / f"raw_data_integrity_report_{stamp}.json"
    txt_path = reports_dir / f"raw_data_integrity_report_{stamp}.txt"

    report["report_json_path"] = str(json_path)
    report["report_txt_path"] = str(txt_path)

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(build_txt_report(report), encoding="utf-8")


def main() -> None:
    start_dt = utc_now()
    stamp = file_stamp(start_dt)
    run_id = f"raw_integrity_{stamp}"

    config = load_config()
    raw_collections = config.get("raw_collections", DEFAULT_CONFIG["raw_collections"])

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": run_id,
        "project_root": str(get_project_root()),
        "database": config.get("database"),
        "symbol": config.get("symbol"),
        "mongo_uri": config.get("mongo_uri"),
        "start_time": start_dt.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "status": "running",
        "collections": {},
        "warnings": [],
        "errors": [],
    }

    print("=== Book & Quality - Raw Data Integrity Check ===", flush=True)
    print(f"Project Root: {report['project_root']}", flush=True)
    print(f"Database    : {report['database']}", flush=True)
    print(f"Symbol      : {report['symbol']}", flush=True)

    try:
        db = connect(config)

        for tf, coll_name in raw_collections.items():
            report["collections"][tf] = inspect_collection(db, tf, coll_name)

        for tf, item in report["collections"].items():
            if item.get("status") != "success":
                report["warnings"].append(f"{tf.upper()} check status is {item.get('status')}")
            if item.get("duplicates", {}).get("has_duplicate_examples"):
                report["warnings"].append(f"{tf.upper()} has duplicate time examples.")
            if item.get("ohlc_consistency", {}).get("bad_count", 0) not in (0, None):
                report["warnings"].append(f"{tf.upper()} has OHLC consistency problems.")
            if item.get("gaps", {}).get("gap_count", 0) not in (0, None):
                report["warnings"].append(f"{tf.upper()} has time gaps. Some gaps may be normal around market closures.")

        report["status"] = "success" if not report["errors"] else "failed"

    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc))
        print(f"[ERROR] {exc}", flush=True)

    finally:
        end_dt = utc_now()
        report["end_time"] = end_dt.isoformat()
        report["duration_seconds"] = round((end_dt - start_dt).total_seconds(), 3)
        save_reports(report, stamp)

    print("\n=== Final Summary ===", flush=True)
    for tf, item in report["collections"].items():
        gaps = item.get("gaps", {}).get("gap_count")
        ohlc_bad = item.get("ohlc_consistency", {}).get("bad_count")
        print(
            f"{tf.upper():>3} | exists={item.get('exists')} | "
            f"count≈{item.get('estimated_count')} | "
            f"time_field={item.get('detected_time_field')} | "
            f"gaps={gaps} | ohlc_bad={ohlc_bad} | status={item.get('status')}",
            flush=True
        )

    print(f"\nJSON Report: {report.get('report_json_path')}", flush=True)
    print(f"TXT Report : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] == "success" else "[FAILED]", flush=True)

    if report["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
