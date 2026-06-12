# -*- coding: utf-8 -*-
"""
Book & Quality - Build Trade Quality Labels v1 - FIXED

Location:
    Book_Quality/02_labels/trade_quality/04_build_trade_quality_labels_v1.py

Fix in this version:
    MongoDB conflict fixed:
        "Updating the path 'created_at' would create a conflict at 'created_at'"

Cause:
    created_at was included in both:
        $set.created_at
        $setOnInsert.created_at

Solution:
    created_at is removed from $set and kept only in $setOnInsert.
    updated_at remains in $set.

Purpose:
    Build labels for Trade Quality Model.

Output collection:
    bq_labels_trade_quality_m15_v1

Reports:
    Book_Quality/02_labels/reports/trade_quality_labels_v1_report_YYYYMMDD_HHMMSS.json
    Book_Quality/02_labels/reports/trade_quality_labels_v1_report_YYYYMMDD_HHMMSS.txt

Quick test:
    cd C:\Project\Book_Quality\02_labels\trade_quality
    python -u 04_build_trade_quality_labels_v1.py --limit 1000 --reset

Full build:
    python -u 04_build_trade_quality_labels_v1.py --reset
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError


SCRIPT_NAME = "04_build_trade_quality_labels_v1.py"

TIME_FIELD = "datetime"
OPEN_FIELD = "open"
HIGH_FIELD = "high"
LOW_FIELD = "low"
CLOSE_FIELD = "close"

BATCH_SIZE = 1000

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "database": "market_data",
    "symbol": "XAUUSD",
    "raw_collections": {
        "m1": "xauusd_m1",
        "m15": "xauusd_m15",
        "m30": "xauusd_m30",
        "h1": "xauusd_h1",
        "h4": "xauusd_h4"
    },
    "bq_collections": {
        "labels_trade_quality": "bq_labels_trade_quality_m15_v1"
    }
}

DEFAULT_LABEL_CONFIG = {
    "label_version": "trade_quality_label_v1",
    "entry_rule": "Quality is evaluated from next M15 open after anchor_time.",
    "base_r_usd": 2.0,
    "horizon_m15_short": 8,
    "horizon_m15_mid": 16,
    "horizon_m15_long": 32,
    "quality_score_min": 0,
    "quality_score_max": 100,
    "trend_phase_classes": [
        "UP_TREND_EARLY",
        "UP_TREND_MATURE",
        "UP_TREND_EXHAUSTION",
        "DOWN_TREND_EARLY",
        "DOWN_TREND_MATURE",
        "DOWN_TREND_EXHAUSTION",
        "RANGE",
        "TRANSITION"
    ]
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run_stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_reports_dir() -> Path:
    reports_dir = get_project_root() / "02_labels" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read JSON file: {path} | {exc}", flush=True)
        return None


def load_config() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    root = get_project_root()

    mongo_config = DEFAULT_CONFIG.copy()
    cfg = read_json(root / "00_config" / "mongo_config.json")
    if cfg:
        mongo_config.update(cfg)

    label_config = DEFAULT_LABEL_CONFIG.copy()
    label_all = read_json(root / "00_config" / "label_config_v1.json")
    if label_all and "trade_quality_label_v1" in label_all:
        label_config.update(label_all["trade_quality_label_v1"])

    label_config["label_version"] = str(label_config.get("label_version", "trade_quality_label_v1"))
    label_config["base_r_usd"] = float(label_config.get("base_r_usd", 2.0))
    label_config["horizon_m15_short"] = int(label_config.get("horizon_m15_short", 8))
    label_config["horizon_m15_mid"] = int(label_config.get("horizon_m15_mid", 16))
    label_config["horizon_m15_long"] = int(label_config.get("horizon_m15_long", 32))
    label_config["quality_score_min"] = int(label_config.get("quality_score_min", 0))
    label_config["quality_score_max"] = int(label_config.get("quality_score_max", 100))

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
    raise ValueError(f"Unsupported datetime value: {value!r}")


def f(value: Any) -> float:
    return float(value)


def clamp(value: float, low: float = 0, high: float = 100) -> int:
    return int(round(max(low, min(high, value))))


def score_class(score: int) -> str:
    if score >= 90:
        return "EXCELLENT"
    if score >= 70:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    if score >= 30:
        return "LOW"
    return "BAD"


def score_bucket(score: int) -> str:
    return score_class(score)


def add_count(d: Dict[str, int], key: str, n: int = 1) -> None:
    d[key] = d.get(key, 0) + n


def first_bar_index(path: List[Dict[str, Any]], predicate) -> Optional[int]:
    for i, doc in enumerate(path, start=1):
        if predicate(doc):
            return i
    return None


def side_stats(side: str, entry_price: float, path: List[Dict[str, Any]], base_r: float, short_n: int, mid_n: int) -> Dict[str, Any]:
    if not path:
        return {
            "side": side,
            "score": 0,
            "class": "BAD",
            "mfe_short": 0.0,
            "mae_short": 0.0,
            "mfe_mid": 0.0,
            "mae_mid": 0.0,
            "mfe_long": 0.0,
            "mae_long": 0.0,
            "max_profit_r": 0.0,
            "max_loss_r": 0.0,
            "net_move": 0.0,
            "net_move_r": 0.0,
            "tp1_hit": False,
            "tp2_hit": False,
            "sl_hit": False,
            "bars_to_tp1": None,
            "bars_to_tp2": None,
            "bars_to_sl": None,
            "retracement_ratio": 0.0,
            "fake_score": 0,
        }

    def calc_mfe_mae(sub_path: List[Dict[str, Any]]) -> Tuple[float, float]:
        highs = [f(x[HIGH_FIELD]) for x in sub_path]
        lows = [f(x[LOW_FIELD]) for x in sub_path]
        if side == "BUY":
            mfe = max(highs) - entry_price
            mae = entry_price - min(lows)
        else:
            mfe = entry_price - min(lows)
            mae = max(highs) - entry_price
        return max(0.0, mfe), max(0.0, mae)

    short_path = path[:short_n]
    mid_path = path[:mid_n]
    long_path = path

    mfe_short, mae_short = calc_mfe_mae(short_path)
    mfe_mid, mae_mid = calc_mfe_mae(mid_path)
    mfe_long, mae_long = calc_mfe_mae(long_path)

    last_close = f(path[-1][CLOSE_FIELD])
    if side == "BUY":
        net_move = last_close - entry_price
        tp1 = entry_price + base_r
        tp2 = entry_price + 2 * base_r
        sl = entry_price - base_r
        bars_to_tp1 = first_bar_index(path, lambda d: f(d[HIGH_FIELD]) >= tp1)
        bars_to_tp2 = first_bar_index(path, lambda d: f(d[HIGH_FIELD]) >= tp2)
        bars_to_sl = first_bar_index(path, lambda d: f(d[LOW_FIELD]) <= sl)
    else:
        net_move = entry_price - last_close
        tp1 = entry_price - base_r
        tp2 = entry_price - 2 * base_r
        sl = entry_price + base_r
        bars_to_tp1 = first_bar_index(path, lambda d: f(d[LOW_FIELD]) <= tp1)
        bars_to_tp2 = first_bar_index(path, lambda d: f(d[LOW_FIELD]) <= tp2)
        bars_to_sl = first_bar_index(path, lambda d: f(d[HIGH_FIELD]) >= sl)

    tp1_hit = bars_to_tp1 is not None
    tp2_hit = bars_to_tp2 is not None
    sl_hit = bars_to_sl is not None

    tp1_before_sl = tp1_hit and (not sl_hit or bars_to_tp1 < bars_to_sl)
    tp2_before_sl = tp2_hit and (not sl_hit or bars_to_tp2 < bars_to_sl)
    sl_before_tp1 = sl_hit and (not tp1_hit or bars_to_sl <= bars_to_tp1)

    max_profit_r = mfe_long / base_r if base_r else 0.0
    max_loss_r = mae_long / base_r if base_r else 0.0
    net_move_r = net_move / base_r if base_r else 0.0

    if tp2_before_sl:
        score = 82 + min(18, max_profit_r * 3) - min(14, max_loss_r * 5)
    elif tp1_before_sl:
        score = 58 + min(20, max_profit_r * 5) - min(18, max_loss_r * 7)
    elif sl_before_tp1:
        score = 12 + min(22, max_profit_r * 7) - min(12, max_loss_r * 3)
    else:
        score = 30 + min(30, max_profit_r * 10) - min(25, max_loss_r * 9)

    # Penalize very late profit hits
    if tp1_hit and bars_to_tp1 and bars_to_tp1 > mid_n:
        score -= 12
    if tp2_hit and bars_to_tp2 and bars_to_tp2 > mid_n:
        score -= 8

    # Reward clean low-drawdown side
    if max_profit_r >= 2.0 and max_loss_r <= 0.5:
        score += 8

    # Retracement
    if mfe_long > 0:
        final_favorable = max(0.0, net_move)
        retracement_ratio = max(0.0, min(1.0, (mfe_long - final_favorable) / mfe_long))
    else:
        retracement_ratio = 0.0

    fake_score = 0
    if max_profit_r >= 1.0 and max_loss_r >= 1.0:
        fake_score = max(fake_score, 55)
    if tp1_hit and sl_hit and bars_to_tp1 is not None and bars_to_sl is not None and bars_to_tp1 < bars_to_sl:
        fake_score = max(fake_score, 75)
    if max_profit_r >= 1.5 and retracement_ratio >= 0.7:
        fake_score = max(fake_score, 85)
    if max_profit_r >= 2.0 and retracement_ratio >= 0.85:
        fake_score = max(fake_score, 100)

    score = clamp(score)

    return {
        "side": side,
        "score": score,
        "class": score_class(score),
        "mfe_short": round(mfe_short, 5),
        "mae_short": round(mae_short, 5),
        "mfe_mid": round(mfe_mid, 5),
        "mae_mid": round(mae_mid, 5),
        "mfe_long": round(mfe_long, 5),
        "mae_long": round(mae_long, 5),
        "max_profit_r": round(max_profit_r, 5),
        "max_loss_r": round(max_loss_r, 5),
        "net_move": round(net_move, 5),
        "net_move_r": round(net_move_r, 5),
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
        "sl_hit": sl_hit,
        "bars_to_tp1": bars_to_tp1,
        "bars_to_tp2": bars_to_tp2,
        "bars_to_sl": bars_to_sl,
        "retracement_ratio": round(retracement_ratio, 5),
        "fake_score": clamp(fake_score),
    }


def range_stats(entry_price: float, path: List[Dict[str, Any]], base_r: float, buy_q: Dict[str, Any], sell_q: Dict[str, Any]) -> Dict[str, Any]:
    if not path:
        return {
            "score": 0,
            "is_range": False,
            "entry_cross_count": 0,
            "net_abs": 0.0,
            "net_abs_r": 0.0,
            "realized_range": 0.0,
            "realized_range_r": 0.0,
        }

    highs = [f(d[HIGH_FIELD]) for d in path]
    lows = [f(d[LOW_FIELD]) for d in path]
    closes = [f(d[CLOSE_FIELD]) for d in path]

    realized_range = max(highs) - min(lows)
    net_abs = abs(closes[-1] - entry_price)

    cross_count = 0
    prev_side = None
    for d in path:
        h = f(d[HIGH_FIELD])
        l = f(d[LOW_FIELD])
        c = f(d[CLOSE_FIELD])

        if h >= entry_price and l <= entry_price:
            cross_count += 1

        side = 1 if c > entry_price else -1 if c < entry_price else 0
        if side != 0 and prev_side is not None and prev_side != 0 and side != prev_side:
            cross_count += 1
        if side != 0:
            prev_side = side

    realized_range_r = realized_range / base_r if base_r else 0.0
    net_abs_r = net_abs / base_r if base_r else 0.0

    score = 0
    if net_abs_r <= 0.5:
        score += 30
    elif net_abs_r <= 1.0:
        score += 20
    elif net_abs_r <= 1.5:
        score += 10

    if cross_count >= 3:
        score += min(35, cross_count * 5)

    if buy_q["score"] < 50 and sell_q["score"] < 50:
        score += 15

    if realized_range_r >= 1.0 and realized_range_r <= 4.0:
        score += 15
    elif realized_range_r > 6.0:
        score -= 15

    if max(buy_q["score"], sell_q["score"]) >= 70:
        score -= 20

    score = clamp(score)

    return {
        "score": score,
        "is_range": score >= 70,
        "entry_cross_count": int(cross_count),
        "net_abs": round(net_abs, 5),
        "net_abs_r": round(net_abs_r, 5),
        "realized_range": round(realized_range, 5),
        "realized_range_r": round(realized_range_r, 5),
    }


def trend_phase(buy_q: Dict[str, Any], sell_q: Dict[str, Any], fake: Dict[str, Any], rng: Dict[str, Any]) -> Dict[str, Any]:
    if rng["score"] >= 70:
        return {"label": "RANGE", "direction": "NONE", "strength": rng["score"]}

    diff = buy_q["score"] - sell_q["score"]

    if abs(diff) < 15:
        return {"label": "TRANSITION", "direction": "MIXED", "strength": max(buy_q["score"], sell_q["score"], rng["score"])}

    if diff > 0:
        if fake["fake_buy_score"] >= 80 or (buy_q["retracement_ratio"] >= 0.65 and buy_q["max_profit_r"] >= 1.5):
            label = "UP_TREND_EXHAUSTION"
        elif buy_q["max_profit_r"] >= 2.0 and buy_q["max_loss_r"] <= 1.0:
            label = "UP_TREND_MATURE"
        else:
            label = "UP_TREND_EARLY"
        return {"label": label, "direction": "UP", "strength": buy_q["score"]}

    if fake["fake_sell_score"] >= 80 or (sell_q["retracement_ratio"] >= 0.65 and sell_q["max_profit_r"] >= 1.5):
        label = "DOWN_TREND_EXHAUSTION"
    elif sell_q["max_profit_r"] >= 2.0 and sell_q["max_loss_r"] <= 1.0:
        label = "DOWN_TREND_MATURE"
    else:
        label = "DOWN_TREND_EARLY"
    return {"label": label, "direction": "DOWN", "strength": sell_q["score"]}


def build_label_doc(
    symbol: str,
    anchor_doc: Dict[str, Any],
    entry_doc: Dict[str, Any],
    path: List[Dict[str, Any]],
    label_config: Dict[str, Any],
) -> Dict[str, Any]:
    anchor_time = parse_dt(anchor_doc[TIME_FIELD])
    entry_time = parse_dt(entry_doc[TIME_FIELD])
    entry_price = f(entry_doc[OPEN_FIELD])

    base_r = label_config["base_r_usd"]
    short_n = label_config["horizon_m15_short"]
    mid_n = label_config["horizon_m15_mid"]

    buy_q = side_stats("BUY", entry_price, path, base_r, short_n, mid_n)
    sell_q = side_stats("SELL", entry_price, path, base_r, short_n, mid_n)

    fake = {
        "fake_score": max(buy_q["fake_score"], sell_q["fake_score"]),
        "fake_buy_score": buy_q["fake_score"],
        "fake_sell_score": sell_q["fake_score"],
        "fake_buy": buy_q["fake_score"] >= 70,
        "fake_sell": sell_q["fake_score"] >= 70,
    }

    rng = range_stats(entry_price, path, base_r, buy_q, sell_q)
    phase = trend_phase(buy_q, sell_q, fake, rng)

    now = utc_now()

    return {
        "symbol": symbol,
        "anchor_time": anchor_time,
        "label_version": label_config["label_version"],
        "entry_rule": "next_m15_open",
        "entry_time": entry_time,
        "entry_price": round(entry_price, 5),
        "path_timeframe": "M15",
        "path_bars_used": len(path),
        "config": {
            "base_r_usd": label_config["base_r_usd"],
            "horizon_m15_short": label_config["horizon_m15_short"],
            "horizon_m15_mid": label_config["horizon_m15_mid"],
            "horizon_m15_long": label_config["horizon_m15_long"],
            "quality_score_min": label_config["quality_score_min"],
            "quality_score_max": label_config["quality_score_max"],
        },
        "buy_quality": buy_q,
        "sell_quality": sell_q,
        "fake": fake,
        "range": rng,
        "trend_phase": phase,
        "y": {
            "buy_quality_score": buy_q["score"],
            "sell_quality_score": sell_q["score"],
            "fake_score": fake["fake_score"],
            "range_score": rng["score"],
            "trend_phase": phase["label"],
        },
        "created_at": now,
        "updated_at": now,
    }


def make_update_op(symbol: str, label_doc: Dict[str, Any]) -> UpdateOne:
    """
    Critical fix:
    created_at must NOT be present in both $set and $setOnInsert.
    """
    set_doc = dict(label_doc)
    created_at = set_doc.pop("created_at", utc_now())

    return UpdateOne(
        {"symbol": symbol, "anchor_time": label_doc["anchor_time"]},
        {
            "$set": set_doc,
            "$setOnInsert": {"created_at": created_at},
        },
        upsert=True,
    )


def execute_bulk(db: Database, collection_name: str, ops: List[UpdateOne], report: Dict[str, Any]) -> None:
    if not ops:
        return
    try:
        result = db[collection_name].bulk_write(ops, ordered=False)
        report["counts"]["upserted_or_modified"] += result.upserted_count + result.modified_count
        report["counts"]["matched"] += result.matched_count
        report["counts"]["inserted"] += result.upserted_count
        report["counts"]["modified"] += result.modified_count
    except BulkWriteError as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc.details)[:20000])
        raise


def parse_dt_arg(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def build_txt_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Book & Quality - Trade Quality Labels v1 Report")
    lines.append("=" * 74)
    lines.append(f"Run ID              : {report['run_id']}")
    lines.append(f"Status              : {report['status']}")
    lines.append(f"Database            : {report['database']}")
    lines.append(f"Symbol              : {report['symbol']}")
    lines.append(f"Start Time          : {report['start_time']}")
    lines.append(f"End Time            : {report['end_time']}")
    lines.append(f"Duration Sec        : {report['duration_seconds']}")
    lines.append("")
    lines.append("Config")
    lines.append("-" * 74)
    for k, v in report["label_config"].items():
        lines.append(f"{k:<25}: {v}")
    lines.append("")
    lines.append("Counts")
    lines.append("-" * 74)
    for k, v in report["counts"].items():
        lines.append(f"{k:<25}: {v}")
    lines.append("")
    lines.append("Buy Quality Buckets")
    lines.append("-" * 74)
    for k, v in report["buy_quality_buckets"].items():
        lines.append(f"{k:<15}: {v}")
    lines.append("")
    lines.append("Sell Quality Buckets")
    lines.append("-" * 74)
    for k, v in report["sell_quality_buckets"].items():
        lines.append(f"{k:<15}: {v}")
    lines.append("")
    lines.append("Trend Phase Distribution")
    lines.append("-" * 74)
    for k, v in report["trend_phase_distribution"].items():
        lines.append(f"{k:<30}: {v}")
    lines.append("")
    lines.append("Range Bucket Distribution")
    lines.append("-" * 74)
    for k, v in report["range_bucket_distribution"].items():
        lines.append(f"{k:<15}: {v}")
    lines.append("")
    lines.append("Boolean Counts")
    lines.append("-" * 74)
    for k, v in report["boolean_counts"].items():
        lines.append(f"{k:<25}: {v}")

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
        for e in report["errors"][:20]:
            lines.append(f"- {e}")

    return "\n".join(lines)


def save_reports(report: Dict[str, Any], rs: str) -> None:
    reports_dir = get_reports_dir()
    json_path = reports_dir / f"trade_quality_labels_v1_report_{rs}.json"
    txt_path = reports_dir / f"trade_quality_labels_v1_report_{rs}.txt"

    report["report_json_path"] = str(json_path)
    report["report_txt_path"] = str(txt_path)

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(build_txt_report(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Trade Quality Labels v1.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of M15 anchors to process.")
    parser.add_argument("--reset", action="store_true", help="Delete existing target docs before building.")
    parser.add_argument("--start", type=str, default=None, help="Optional start datetime.")
    parser.add_argument("--end", type=str, default=None, help="Optional end datetime.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing anchors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_dt = utc_now()
    rs = run_stamp(start_dt)
    run_id = f"trade_quality_labels_v1_{rs}"

    mongo_config, label_config = load_config()

    db_name = mongo_config["database"]
    symbol = mongo_config["symbol"]
    raw = mongo_config["raw_collections"]
    bq = mongo_config["bq_collections"]

    m15_collection = raw.get("m15", "xauusd_m15")
    target_collection = bq.get("labels_trade_quality", "bq_labels_trade_quality_m15_v1")

    report: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "run_id": run_id,
        "project_root": str(get_project_root()),
        "database": db_name,
        "symbol": symbol,
        "start_time": start_dt.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "status": "running",
        "collections": {
            "m15_source": m15_collection,
            "target": target_collection,
        },
        "args": {
            "limit": args.limit,
            "reset": args.reset,
            "start": args.start,
            "end": args.end,
            "skip_existing": args.skip_existing,
        },
        "label_config": label_config,
        "counts": {
            "anchors_read": 0,
            "upserted_or_modified": 0,
            "inserted": 0,
            "matched": 0,
            "modified": 0,
            "skipped_existing": 0,
            "skipped_no_entry": 0,
            "skipped_no_future_path": 0,
            "row_errors": 0,
        },
        "buy_quality_buckets": {"BAD": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "EXCELLENT": 0},
        "sell_quality_buckets": {"BAD": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "EXCELLENT": 0},
        "range_bucket_distribution": {"BAD": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "EXCELLENT": 0},
        "trend_phase_distribution": {},
        "boolean_counts": {
            "fake_buy": 0,
            "fake_sell": 0,
            "is_range": 0,
            "buy_quality_ge_70": 0,
            "sell_quality_ge_70": 0,
        },
        "warnings": [],
        "errors": [],
    }

    print("=== Book & Quality - Build Trade Quality Labels v1 - FIXED ===", flush=True)
    print(f"Project Root : {report['project_root']}", flush=True)
    print(f"Database     : {db_name}", flush=True)
    print(f"Symbol       : {symbol}", flush=True)
    print(f"Target       : {target_collection}", flush=True)

    try:
        db = connect(mongo_config)

        if args.reset:
            deleted = db[target_collection].delete_many({})
            report["reset_deleted_count"] = deleted.deleted_count
            print(f"[RESET] Deleted {deleted.deleted_count:,} existing target docs.", flush=True)

        db[target_collection].create_index(
            [("symbol", ASCENDING), ("anchor_time", ASCENDING)],
            unique=True,
            name="uq_symbol_anchor_time",
        )
        db[target_collection].create_index(
            [("label_version", ASCENDING), ("trend_phase.label", ASCENDING)],
            name="ix_label_version_trend_phase",
        )
        db[target_collection].create_index(
            [("label_version", ASCENDING), ("y.buy_quality_score", ASCENDING)],
            name="ix_label_version_buy_quality",
        )
        db[target_collection].create_index(
            [("label_version", ASCENDING), ("y.sell_quality_score", ASCENDING)],
            name="ix_label_version_sell_quality",
        )

        start_arg = parse_dt_arg(args.start)
        end_arg = parse_dt_arg(args.end)
        horizon = label_config["horizon_m15_long"]

        all_m15 = list(
            db[m15_collection]
            .find(
                {},
                projection={TIME_FIELD: 1, OPEN_FIELD: 1, HIGH_FIELD: 1, LOW_FIELD: 1, CLOSE_FIELD: 1, "_id": 0},
            )
            .sort(TIME_FIELD, ASCENDING)
        )

        print(f"[LOAD] Loaded M15 candles: {len(all_m15):,}", flush=True)

        ops: List[UpdateOne] = []
        max_i = len(all_m15) - 1

        for i, anchor_doc in enumerate(all_m15):
            if i >= max_i:
                break

            anchor_time = parse_dt(anchor_doc[TIME_FIELD])

            if start_arg and anchor_time < start_arg:
                continue
            if end_arg and anchor_time > end_arg:
                continue
            if args.limit is not None and report["counts"]["anchors_read"] >= args.limit:
                break

            report["counts"]["anchors_read"] += 1

            try:
                if args.skip_existing:
                    exists = db[target_collection].find_one(
                        {"symbol": symbol, "anchor_time": anchor_time},
                        projection={"_id": 1},
                    )
                    if exists:
                        report["counts"]["skipped_existing"] += 1
                        continue

                entry_i = i + 1
                if entry_i >= len(all_m15):
                    report["counts"]["skipped_no_entry"] += 1
                    continue

                entry_doc = all_m15[entry_i]
                path = all_m15[entry_i: entry_i + horizon]

                if len(path) == 0:
                    report["counts"]["skipped_no_future_path"] += 1
                    continue

                label_doc = build_label_doc(symbol, anchor_doc, entry_doc, path, label_config)

                add_count(report["buy_quality_buckets"], score_bucket(label_doc["buy_quality"]["score"]))
                add_count(report["sell_quality_buckets"], score_bucket(label_doc["sell_quality"]["score"]))
                add_count(report["range_bucket_distribution"], score_bucket(label_doc["range"]["score"]))
                add_count(report["trend_phase_distribution"], label_doc["trend_phase"]["label"])

                if label_doc["fake"]["fake_buy"]:
                    report["boolean_counts"]["fake_buy"] += 1
                if label_doc["fake"]["fake_sell"]:
                    report["boolean_counts"]["fake_sell"] += 1
                if label_doc["range"]["is_range"]:
                    report["boolean_counts"]["is_range"] += 1
                if label_doc["buy_quality"]["score"] >= 70:
                    report["boolean_counts"]["buy_quality_ge_70"] += 1
                if label_doc["sell_quality"]["score"] >= 70:
                    report["boolean_counts"]["sell_quality_ge_70"] += 1

                ops.append(make_update_op(symbol, label_doc))

                if len(ops) >= BATCH_SIZE:
                    execute_bulk(db, target_collection, ops, report)
                    ops = []
                    print(
                        f"[PROGRESS] anchors={report['counts']['anchors_read']:,} "
                        f"written≈{report['counts']['upserted_or_modified']:,} "
                        f"buy>=70={report['boolean_counts']['buy_quality_ge_70']:,} "
                        f"sell>=70={report['boolean_counts']['sell_quality_ge_70']:,} "
                        f"range={report['boolean_counts']['is_range']:,}",
                        flush=True,
                    )

            except Exception as row_exc:
                report["counts"]["row_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(f"anchor_time={anchor_time} | {row_exc}")

        if ops:
            execute_bulk(db, target_collection, ops, report)

        report["final_target_count"] = db[target_collection].estimated_document_count()

        if report["counts"]["row_errors"] > 0:
            report["warnings"].append("Some row errors occurred. Check JSON report.")

        report["status"] = "success" if report["counts"]["row_errors"] == 0 and not report["errors"] else "success_with_row_errors"

    except Exception as exc:
        report["status"] = "failed"
        if not report["errors"]:
            report["errors"].append(str(exc))
        print(f"[ERROR] {exc}", flush=True)

    finally:
        end_dt = utc_now()
        report["end_time"] = end_dt.isoformat()
        report["duration_seconds"] = round((end_dt - start_dt).total_seconds(), 3)
        save_reports(report, rs)

    print("\n=== Final Summary ===", flush=True)
    print(f"Status              : {report['status']}", flush=True)
    print(f"Anchors Read        : {report['counts']['anchors_read']:,}", flush=True)
    print(f"Upserted/Modified   : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Inserted            : {report['counts']['inserted']:,}", flush=True)
    print(f"Modified            : {report['counts']['modified']:,}", flush=True)
    print(f"Skipped Existing    : {report['counts']['skipped_existing']:,}", flush=True)
    print(f"Skipped No Entry    : {report['counts']['skipped_no_entry']:,}", flush=True)
    print(f"Skipped No Future   : {report['counts']['skipped_no_future_path']:,}", flush=True)
    print(f"Row Errors          : {report['counts']['row_errors']:,}", flush=True)
    print(f"Buy >= 70           : {report['boolean_counts']['buy_quality_ge_70']:,}", flush=True)
    print(f"Sell >= 70          : {report['boolean_counts']['sell_quality_ge_70']:,}", flush=True)
    print(f"Range Count         : {report['boolean_counts']['is_range']:,}", flush=True)
    print(f"JSON Report         : {report.get('report_json_path')}", flush=True)
    print(f"TXT Report          : {report.get('report_txt_path')}", flush=True)
    print("[DONE]" if report["status"] != "failed" else "[FAILED]", flush=True)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
