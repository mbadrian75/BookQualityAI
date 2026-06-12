# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Update Quality Model Collections to M30

Location:
    Book_Quality/00_project_setup/01_update_bq2_quality_to_m30.py

Purpose:
    Update bq2 config so MA Quality Model uses M30 anchor collections.

Run:
    cd C:\Project\Book_Quality
    python -u 00_project_setup\01_update_bq2_quality_to_m30.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc)


def stamp(dt):
    return dt.strftime("%Y%m%d_%H%M%S")


def main():
    started = utc_now()
    root = Path.cwd().resolve()
    cfg_path = root / "00_config" / "bq2_config.json"
    label_cfg_path = root / "00_config" / "bq2_label_config_v1.json"
    report_dir = root / "00_project_setup" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    cfg.setdefault("bq2_collections", {})
    cfg["bq2_collections"]["labels_ma_quality"] = "bq2_labels_ma_quality_m30_v1"
    cfg["bq2_collections"]["features_ma_quality"] = "bq2_features_ma_quality_m30_v1"
    cfg["bq2_collections"]["dataset_ma_quality"] = "bq2_dataset_ma_quality_m30_v1"
    cfg["bq2_collections"]["predictions_ma_quality"] = "bq2_predictions_ma_quality_m30_v1"

    cfg.setdefault("model_design", {})
    cfg["model_design"].setdefault("ma_quality_model", {})
    cfg["model_design"]["ma_quality_model"]["anchor_timeframe"] = "M30"
    cfg["model_design"]["ma_quality_model"]["input_timeframes"] = ["M30", "H1", "H4"]
    cfg["model_design"]["ma_quality_model"]["ma_periods"] = [20, 50, 100]

    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    label_cfg = {}
    if label_cfg_path.exists():
        label_cfg = json.loads(label_cfg_path.read_text(encoding="utf-8"))

    label_cfg.setdefault("ma_quality_label_v1", {})
    label_cfg["ma_quality_label_v1"].update({
        "anchor_timeframe": "M30",
        "entry_rule": "decision_time = next M30 open after anchor_time",
        "quality_timeframes": ["M30", "H1", "H4"],
        "ma_periods": [20, 50, 100],
        "future_horizons": {
            "m30_bars": 8,
            "h1_bars": 6,
            "h4_bars": 4
        },
        "trade_horizon_m30": 16,
        "base_r_usd": 2.0
    })

    label_cfg_path.write_text(json.dumps(label_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "run_id": f"update_bq2_quality_to_m30_{stamp(started)}",
        "status": "success",
        "project_root": str(root),
        "config_updated": str(cfg_path),
        "label_config_updated": str(label_cfg_path),
        "ma_quality_collections": {
            "labels_ma_quality": "bq2_labels_ma_quality_m30_v1",
            "features_ma_quality": "bq2_features_ma_quality_m30_v1",
            "dataset_ma_quality": "bq2_dataset_ma_quality_m30_v1",
            "predictions_ma_quality": "bq2_predictions_ma_quality_m30_v1"
        },
        "raw_data_modified": False,
        "started_at": started.isoformat(),
        "ended_at": utc_now().isoformat()
    }

    jp = report_dir / f"update_bq2_quality_to_m30_report_{stamp(started)}.json"
    tp = report_dir / f"update_bq2_quality_to_m30_report_{stamp(started)}.txt"
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tp.write_text(
        "\n".join([
            "Book & Quality v2 - Update Quality Model to M30 Report",
            "=" * 74,
            f"Run ID            : {report['run_id']}",
            f"Status            : {report['status']}",
            f"Raw Data Modified : {report['raw_data_modified']}",
            "",
            "MA Quality collections now use M30 anchor.",
            "Config updated successfully."
        ]),
        encoding="utf-8"
    )

    print("=== Update BQ2 Quality Model to M30 ===")
    print("Status            : success")
    print("Raw Data Modified : False")
    print(f"Config            : {cfg_path}")
    print(f"Report JSON       : {jp}")
    print(f"Report TXT        : {tp}")
    print("[DONE]")


if __name__ == "__main__":
    main()
