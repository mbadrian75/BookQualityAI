# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime
from pathlib import Path

PROJECT_NAME = "Side Context Entry AI"
PROJECT_CODE = "SCE"
DB_NAME = "market_data"
DERIVED_PREFIX = "sce_"
FOLDERS = ["00_project_setup/reports", "config", "docs", "01_database/bootstrap/reports", "01_database/inspect_raw/reports", "02_labels/entry_side/reports", "03_features/ma_context/reports", "03_features/candle_entry/reports", "03_features/integrated_entry/reports", "04_dataset/integrated_entry/reports", "05_training/integrated_entry/reports", "06_models/integrated_entry/reports", "07_api/decision_engine/reports", "08_backtest/integrated_entry/reports", "09_logs"]

def main() -> int:
    run_id = f"create_sce_clean_project_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = Path(__file__).resolve().parents[1]
    created, already_exists = [], []
    for folder in FOLDERS:
        path = root / folder
        if path.exists(): already_exists.append(folder)
        else:
            path.mkdir(parents=True, exist_ok=True)
            created.append(folder)
    report_dir = root / "00_project_setup" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_id}.txt"
    lines = ["Side Context Entry AI - Clean Project Structure Report", "=" * 64, f"run_id                   : {run_id}", "status                   : success", f"project_name             : {PROJECT_NAME}", f"project_code             : {PROJECT_CODE}", f"root                     : {root}", f"database                 : {DB_NAME}", f"derived_prefix           : {DERIVED_PREFIX}", "raw_data_modified         : False", "previous_folders_modified : False", "", "Folder policy:", "- each script is located inside its own section folder", "- each section writes reports inside its own local reports folder", "- shared project parameters are in config/sce_project_config.py", "", f"created_count            : {len(created)}"]
    lines += [f"- created                : {x}" for x in created]
    lines += ["", f"already_exists_count     : {len(already_exists)}"]
    lines += [f"- already_exists         : {x}" for x in already_exists]
    lines += ["", "End of report."]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("SCE clean project structure created.")
    print(f"Report: {report_path}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
