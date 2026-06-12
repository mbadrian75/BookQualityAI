# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Build Candle Book Side Confirmation Labels M15 v2

Location:
    Book_Quality/02_labels/candle_book/02_build_candle_book_side_labels_m15_v2.py

Why v2:
    v1 trained Candle Book as a standalone 3-class direction model:
        BUY / SELL / NOTRADE
    Full training showed weak generalization and very low confidence.

    In final architecture, Candle Book should NOT independently choose market direction.
    MA Scenario gives the proposed direction.
    Candle Book should answer:
        Is BUY entry timing acceptable now?
        Is SELL entry timing acceptable now?

Purpose:
    Build side-specific labels for confirmation:
        buy_outcome  = WIN / LOSS / FLAT / AMBIGUOUS
        sell_outcome = WIN / LOSS / FLAT / AMBIGUOUS
        book_direction_v2 = BUY / SELL / NOTRADE  (derived, for analysis only)

Anchor:
    M15 closed candle.

Entry:
    next M15 open after anchor_time.

Input raw collection:
    xauusd_m15

Output:
    bq2_labels_candle_book_side_m15_v2

No-Leak:
    Future candles are used only for labels.
    Feature builders must not read y/future fields.

TEST:
    cd C:/Project/BookQuality/bq2/02_labels/candle_book
    python -u 02_build_candle_book_side_labels_m15_v2.py --limit 1000 --reset

FULL:
    python -u 02_build_candle_book_side_labels_m15_v2.py --reset
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

SCRIPT_NAME = "02_build_candle_book_side_labels_m15_v2.py"
LABEL_VERSION = "candle_book_side_label_m15_v2"
BATCH_SIZE = 1000

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "raw_collections": {"m15": "xauusd_m15"},
    "bq2_collections": {
        "labels_candle_book_side": "bq2_labels_candle_book_side_m15_v2"
    },
    "candle_book_side_label_m15_v2": {
        "base_r_usd": 2.0,
        "horizon_m15": 8,
        "direction_threshold_r": 0.35
    }
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
    cfg["bq2_collections"]["labels_candle_book_side"] = "bq2_labels_candle_book_side_m15_v2"
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


def candle_ohlc(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return sf(row["open"]), sf(row["high"]), sf(row["low"]), sf(row["close"])


def side_outcome(
    side: str,
    entry_price: float,
    future_bars: List[Dict[str, Any]],
    base_r_usd: float,
    direction_threshold_r: float,
) -> Dict[str, Any]:
    if side == "BUY":
        target_level = entry_price + base_r_usd
        stop_level = entry_price - base_r_usd
    elif side == "SELL":
        target_level = entry_price - base_r_usd
        stop_level = entry_price + base_r_usd
    else:
        raise ValueError("side must be BUY or SELL")

    first_event = None
    first_event_time = None
    first_event_bar_index = None
    ambiguous_bar = False
    max_favorable = -10**9
    max_adverse = -10**9

    for i, bar in enumerate(future_bars):
        t = row_dt(bar)
        _, h, l, c = candle_ohlc(bar)

        if side == "BUY":
            favorable = h - entry_price
            adverse = entry_price - l
            target_hit = h >= target_level
            stop_hit = l <= stop_level
        else:
            favorable = entry_price - l
            adverse = h - entry_price
            target_hit = l <= target_level
            stop_hit = h >= stop_level

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        if target_hit and stop_hit:
            first_event = "AMBIGUOUS"
            first_event_time = t
            first_event_bar_index = i
            ambiguous_bar = True
            break
        if target_hit:
            first_event = "TARGET"
            first_event_time = t
            first_event_bar_index = i
            break
        if stop_hit:
            first_event = "STOP"
            first_event_time = t
            first_event_bar_index = i
            break

    last_close = sf(future_bars[-1]["close"])
    if side == "BUY":
        final_move = last_close - entry_price
    else:
        final_move = entry_price - last_close

    threshold_usd = direction_threshold_r * base_r_usd

    if first_event == "TARGET":
        outcome = "WIN"
        method = f"{side.lower()}_target_first"
        confidence_label = min(1.0, max(0.50, max_favorable / max(base_r_usd, 1e-9)))
    elif first_event == "STOP":
        outcome = "LOSS"
        method = f"{side.lower()}_stop_first"
        confidence_label = min(1.0, max(0.50, max_adverse / max(base_r_usd, 1e-9)))
    elif first_event == "AMBIGUOUS":
        outcome = "AMBIGUOUS"
        method = f"{side.lower()}_ambiguous_same_bar"
        confidence_label = 0.0
    else:
        if final_move >= threshold_usd:
            outcome = "WIN"
            method = f"{side.lower()}_horizon_final_win"
            confidence_label = min(1.0, abs(final_move) / max(base_r_usd, 1e-9))
        elif final_move <= -threshold_usd:
            outcome = "LOSS"
            method = f"{side.lower()}_horizon_final_loss"
            confidence_label = min(1.0, abs(final_move) / max(base_r_usd, 1e-9))
        else:
            outcome = "FLAT"
            method = f"{side.lower()}_horizon_flat"
            confidence_label = 0.0

    return {
        "outcome": outcome,
        "method": method,
        "confidence_label": float(confidence_label),
        "target_level": float(target_level),
        "stop_level": float(stop_level),
        "first_event": first_event,
        "first_event_time": first_event_time,
        "first_event_bar_index": first_event_bar_index,
        "ambiguous_bar": ambiguous_bar,
        "max_favorable_usd": float(max_favorable),
        "max_adverse_usd": float(max_adverse),
        "final_move_usd": float(final_move),
    }


def derive_direction_v2(buy: Dict[str, Any], sell: Dict[str, Any]) -> str:
    bo = buy["outcome"]
    so = sell["outcome"]
    if bo == "WIN" and so in {"LOSS", "FLAT"}:
        return "BUY"
    if so == "WIN" and bo in {"LOSS", "FLAT"}:
        return "SELL"
    return "NOTRADE"


def build_label_for_anchor(anchor: Dict[str, Any], next_bar: Dict[str, Any], future_bars: List[Dict[str, Any]], base_r_usd: float, direction_threshold_r: float) -> Optional[Dict[str, Any]]:
    if not future_bars:
        return None

    anchor_time = row_dt(anchor)
    entry_time = row_dt(next_bar)
    entry_price = sf(next_bar["open"])

    buy = side_outcome("BUY", entry_price, future_bars, base_r_usd, direction_threshold_r)
    sell = side_outcome("SELL", entry_price, future_bars, base_r_usd, direction_threshold_r)
    direction_v2 = derive_direction_v2(buy, sell)

    return {
        "anchor_time": anchor_time,
        "decision_time": entry_time,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "label_version": LABEL_VERSION,
        "anchor_timeframe": "M15",
        "future_horizon_m15": len(future_bars),
        "base_r_usd": base_r_usd,
        "direction_threshold_r": direction_threshold_r,
        "y": {
            "book_direction_v2": direction_v2,
            "buy_outcome": buy["outcome"],
            "sell_outcome": sell["outcome"],
            "buy_method": buy["method"],
            "sell_method": sell["method"],
            "buy_confidence_label": buy["confidence_label"],
            "sell_confidence_label": sell["confidence_label"],
            "buy": buy,
            "sell": sell,
        },
        "no_leak_note": "Future fields are labels only. Feature builder must not read y/future fields.",
    }


def make_update_op(symbol: str, label_doc: Dict[str, Any]) -> UpdateOne:
    now = utc_now()
    doc = {"symbol": symbol, **label_doc, "created_at": now, "updated_at": now}
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


def inc(d: Dict[str, int], k: str) -> None:
    d[k] = d.get(k, 0) + 1


def save_reports(report: Dict[str, Any], rs: str) -> None:
    jp = reports_dir() / f"candle_book_side_labels_m15_v2_report_{rs}.json"
    tp = reports_dir() / f"candle_book_side_labels_m15_v2_report_{rs}.txt"
    report["report_json_path"] = str(jp)
    report["report_txt_path"] = str(tp)

    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "Book & Quality v2 - Candle Book Side Labels M15 v2 Report",
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

    for title, key in [
        ("Book Direction v2 Distribution", "book_direction_v2_distribution"),
        ("Buy Outcome Distribution", "buy_outcome_distribution"),
        ("Sell Outcome Distribution", "sell_outcome_distribution"),
        ("Buy Method Distribution", "buy_method_distribution"),
        ("Sell Method Distribution", "sell_method_distribution"),
    ]:
        lines += ["", title, "-" * 74]
        for k, v in report[key].items():
            lines.append(f"{k:<34}: {v}")

    if report["errors"]:
        lines += ["", "Errors", "-" * 74]
        lines += [f"- {e}" for e in report["errors"][:50]]

    tp.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Candle Book Side Confirmation Labels M15 v2.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    rs = stamp(started)

    cfg = load_config()
    db_name = cfg["database"]
    symbol = cfg["symbol"]
    raw_coll = cfg["raw_collections"]["m15"]
    target_coll = cfg["bq2_collections"]["labels_candle_book_side"]
    lcfg = cfg.get("candle_book_side_label_m15_v2", DEFAULT_CONFIG["candle_book_side_label_m15_v2"])
    base_r_usd = float(lcfg.get("base_r_usd", 2.0))
    horizon_m15 = int(lcfg.get("horizon_m15", 8))
    direction_threshold_r = float(lcfg.get("direction_threshold_r", 0.35))

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": f"candle_book_side_labels_m15_v2_{rs}",
        "status": "running",
        "database": db_name,
        "symbol": symbol,
        "start_time": started.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "collections": {"m15_source": raw_coll, "target": target_coll},
        "label_config": {
            "label_version": LABEL_VERSION,
            "anchor_timeframe": "M15",
            "entry_rule": "entry_time = next M15 open after anchor_time",
            "base_r_usd": base_r_usd,
            "horizon_m15": horizon_m15,
            "direction_threshold_r": direction_threshold_r,
            "purpose": "side-specific confirmation labels for API direction",
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
        "book_direction_v2_distribution": {},
        "buy_outcome_distribution": {},
        "sell_outcome_distribution": {},
        "buy_method_distribution": {},
        "sell_method_distribution": {},
        "errors": [],
        "args": vars(args),
    }

    print("=== Book & Quality v2 - Build Candle Book Side Labels M15 v2 ===", flush=True)
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
        db[target_coll].create_index([("y.book_direction_v2", ASCENDING), ("anchor_time", ASCENDING)], name="ix_book_direction_v2_anchor_time")
        db[target_coll].create_index([("y.buy_outcome", ASCENDING), ("y.sell_outcome", ASCENDING)], name="ix_side_outcomes")

        query: Dict[str, Any] = {}
        if args.start or args.end:
            tf: Dict[str, Any] = {}
            if args.start:
                tf["$gte"] = parse_dt_optional(args.start)
            if args.end:
                tf["$lte"] = parse_dt_optional(args.end)
            query["datetime"] = tf

        rows = list(db[raw_coll].find(query, projection={"_id": 0, "datetime": 1, "open": 1, "high": 1, "low": 1, "close": 1}).sort("datetime", ASCENDING))
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

                label_doc = build_label_for_anchor(anchor, next_bar, future_bars, base_r_usd, direction_threshold_r)
                if not label_doc:
                    report["counts"]["skipped_no_future_path"] += 1
                    continue

                y = label_doc["y"]
                inc(report["book_direction_v2_distribution"], y["book_direction_v2"])
                inc(report["buy_outcome_distribution"], y["buy_outcome"])
                inc(report["sell_outcome_distribution"], y["sell_outcome"])
                inc(report["buy_method_distribution"], y["buy_method"])
                inc(report["sell_method_distribution"], y["sell_method"])

                ops.append(make_update_op(symbol, label_doc))
                report["counts"]["labels_built"] += 1

                if len(ops) >= BATCH_SIZE:
                    execute_bulk(db, target_coll, ops, report)
                    ops = []
                    print(f"[PROGRESS] anchors={report['counts']['anchors_seen']:,} built={report['counts']['labels_built']:,} written≈{report['counts']['upserted_or_modified']:,}", flush=True)

            except Exception as row_exc:
                report["counts"]["row_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(f"i={i} anchor={anchor.get('datetime')} | {row_exc}")

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
    print(f"Status                 : {report['status']}", flush=True)
    print(f"M15 Loaded             : {report['counts']['m15_loaded']:,}", flush=True)
    print(f"Anchors Seen           : {report['counts']['anchors_seen']:,}", flush=True)
    print(f"Labels Built           : {report['counts']['labels_built']:,}", flush=True)
    print(f"Upserted/Modified      : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Row Errors             : {report['counts']['row_errors']:,}", flush=True)
    print(f"Direction v2 Dist      : {report['book_direction_v2_distribution']}", flush=True)
    print(f"Buy Outcome Dist       : {report['buy_outcome_distribution']}", flush=True)
    print(f"Sell Outcome Dist      : {report['sell_outcome_distribution']}", flush=True)
    print(f"TXT Report             : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
