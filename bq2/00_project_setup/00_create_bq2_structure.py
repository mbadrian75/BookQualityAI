# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Create Project Structure

Run:
    cd C:\Project\Book_Quality
    python -u 00_project_setup\00_create_bq2_structure.py

Optional:
    python -u 00_project_setup\00_create_bq2_structure.py --root C:\Project\Book_Quality --overwrite-config
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

def now_utc(): return datetime.now(timezone.utc)
def stamp(dt): return dt.strftime("%Y%m%d_%H%M%S")

def write_json(path: Path, data, overwrite: bool, actions):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        actions.append({"path": str(path), "action": "skipped_exists"})
        return
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    actions.append({"path": str(path), "action": "written"})

def write_text(path: Path, text: str, overwrite: bool, actions):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        actions.append({"path": str(path), "action": "skipped_exists"})
        return
    path.write_text(text, encoding="utf-8")
    actions.append({"path": str(path), "action": "written"})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--overwrite-config", action="store_true")
    args = ap.parse_args()

    started = now_utc()
    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()

    folders = [
        "00_project_setup/reports", "00_config",
        "01_database/bootstrap", "01_database/reports",
        "02_labels/ma_quality", "02_labels/candle_book", "02_labels/reports",
        "03_features/ma_quality", "03_features/candle_book", "03_features/reports",
        "04_dataset/ma_quality", "04_dataset/candle_book", "04_dataset/reports", "04_dataset/exports",
        "05_training/ma_quality", "05_training/candle_book", "05_training/reports",
        "06_models/ma_quality", "06_models/candle_book", "06_models/archived",
        "07_backtest/bq2", "07_backtest/reports",
        "08_api/app", "08_api/decision_engine_bq2", "08_api/schemas", "08_api/reports",
        "09_mt4_mt5/experts", "09_mt4_mt5/indicators", "09_mt4_mt5/scripts",
        "10_docs", "11_logs", "12_exports/csv", "12_exports/json", "12_exports/charts",
    ]

    actions = []
    for f in folders:
        p = root / f
        p.mkdir(parents=True, exist_ok=True)
        actions.append({"path": str(p), "action": "folder_ok"})

    bq2_config = {
        "project_name": "Book & Quality v2",
        "version": "bq2_v1",
        "mongo_uri": "mongodb://localhost:27017",
        "database": "market_data",
        "symbol": "XAUUSD",
        "raw_collections": {
            "m1": "xauusd_m1", "m5": "xauusd_m5", "m15": "xauusd_m15",
            "m30": "xauusd_m30", "h1": "xauusd_h1", "h4": "xauusd_h4"
        },
        "bq2_collections": {
            "labels_ma_quality": "bq2_labels_ma_quality_m15_v1",
            "features_ma_quality": "bq2_features_ma_quality_m15_v1",
            "dataset_ma_quality": "bq2_dataset_ma_quality_m15_v1",
            "predictions_ma_quality": "bq2_predictions_ma_quality_m15_v1",
            "labels_candle_book": "bq2_labels_candle_book_m15_v1",
            "features_candle_book": "bq2_features_candle_book_m15_v1",
            "dataset_candle_book": "bq2_dataset_candle_book_m15_v1",
            "predictions_candle_book": "bq2_predictions_candle_book_m15_v1",
            "api_decisions": "bq2_api_decisions_m15_v1"
        },
        "model_design": {
            "ma_quality_model": {
                "role": "primary_prediction_and_trade_management",
                "input_timeframes": ["M30", "H1", "H4"],
                "ma_periods": [20, 50, 100],
                "important_rule": "Current MA alignment is not mandatory; future direction and phase matter."
            },
            "candle_book_model": {
                "role": "entry_confirmation",
                "input_timeframes": ["M1", "M5", "M15"]
            },
            "api_rule": "Quality predicts; Candle Book confirms. Low Quality probability + Book confirmation means small lot unless BLOCK/HIGH_RISK."
        }
    }

    label_config = {
        "ma_quality_label_v1": {
            "anchor_timeframe": "M15",
            "entry_rule": "decision_time = next M15 open after anchor_time",
            "quality_timeframes": ["M30", "H1", "H4"],
            "ma_periods": [20, 50, 100],
            "future_horizons": {"m30_bars": 8, "h1_bars": 6, "h4_bars": 4},
            "direction_classes": ["BUY", "SELL", "RANGE"],
            "phase_classes": [
                "UP_START", "UP_MIDDLE", "UP_END", "UP_EXHAUSTION",
                "DOWN_START", "DOWN_MIDDLE", "DOWN_END", "DOWN_EXHAUSTION",
                "REVERSAL_UP_START", "REVERSAL_DOWN_START", "RANGE"
            ],
            "lot_multiplier_classes": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
            "risk_modes": ["LOW_RISK", "NORMAL_RISK", "HIGH_RISK", "BLOCK"],
            "tp_modes": ["SHORT_TP", "NORMAL_TP", "WIDE_TP", "HOLD_MORE"],
            "sl_modes": ["TIGHT_SL", "NORMAL_SL", "WIDE_SL"],
            "trailing_modes": ["NO_TRAILING", "NORMAL_TRAIL", "FAST_TRAIL", "TIGHT_TRAIL"],
            "exit_modes": ["FAST_EXIT", "NORMAL_EXIT", "PROTECT_PROFIT", "HOLD_MORE"]
        },
        "candle_book_label_v1": {
            "anchor_timeframe": "M15",
            "entry_rule": "decision_time = next M15 open after anchor_time",
            "input_timeframes": ["M1", "M5", "M15"],
            "outputs": ["BUY", "SELL", "NOTRADE"],
            "tp_usd": 2.0, "sl_usd": 2.0, "horizon_m15": 8,
            "ambiguous_policy": "NOTRADE"
        }
    }

    spec = """# Book & Quality v2 - Locked Spec

## Core Rule
Quality predicts. Candle Book confirms.

## MA Phase & Quality Model
Inputs: M30/H1/H4 + MA 20/50/100.
Role: main scenario prediction, probability, phase, quality, risk, lot, TP, SL, trailing and exit.
Important: current alignment is not mandatory. A TF can be currently down but at DOWN_END and predicted to reverse up.

## Candle Book Confirmation Model
Inputs: M1/M5/M15 candle behavior.
Role: confirm entry timing and direction.

## API
Quality BUY + Book BUY => BUY.
Quality SELL + Book SELL => SELL.
Quality low probability + Book confirms => small lot.
Quality BLOCK/HIGH_RISK => NOTRADE.
"""

    write_json(root/"00_config/bq2_config.json", bq2_config, args.overwrite_config, actions)
    write_json(root/"00_config/bq2_label_config_v1.json", label_config, args.overwrite_config, actions)
    write_text(root/"10_docs/bq2_locked_spec.md", spec, args.overwrite_config, actions)

    ended = now_utc()
    rep = {
        "run_id": f"create_bq2_structure_{stamp(started)}",
        "status": "success",
        "project_root": str(root),
        "start_time": started.isoformat(),
        "end_time": ended.isoformat(),
        "duration_seconds": round((ended-started).total_seconds(), 3),
        "raw_data_modified": False,
        "actions_count": len(actions),
        "actions": actions
    }
    rdir = root/"00_project_setup/reports"; rdir.mkdir(parents=True, exist_ok=True)
    jp = rdir/f"create_bq2_structure_report_{stamp(started)}.json"
    tp = rdir/f"create_bq2_structure_report_{stamp(started)}.txt"
    jp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    tp.write_text("\n".join([
        "Book & Quality v2 - Create Structure Report",
        "="*74,
        f"Run ID            : {rep['run_id']}",
        f"Status            : {rep['status']}",
        f"Project Root      : {rep['project_root']}",
        f"Raw Data Modified : {rep['raw_data_modified']}",
        f"Duration Sec      : {rep['duration_seconds']}",
        f"Actions Count     : {rep['actions_count']}",
    ]), encoding="utf-8")

    print("=== Book & Quality v2 Structure ===")
    print("Status            : success")
    print(f"Project Root      : {root}")
    print("Raw Data Modified : False")
    print(f"JSON Report       : {jp}")
    print(f"TXT Report        : {tp}")
    print("[DONE]")

if __name__ == "__main__":
    main()
