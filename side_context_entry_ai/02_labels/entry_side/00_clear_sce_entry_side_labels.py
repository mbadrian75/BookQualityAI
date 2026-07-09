# -*- coding: utf-8 -*-
"""
SCE Entry Side Label Cleanup

Location:
02_labels/entry_side/

Report:
02_labels/entry_side/reports/

Safety:
- Deletes documents only from sce_labels_entry_side_m5_v1
- Does not drop collection
- Does not drop indexes
- Requires --confirm-delete 1
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from pymongo import MongoClient
except ImportError as exc:
    print("ERROR: pymongo is not installed. Run: pip install pymongo")
    raise exc

from config.sce_project_config import COLL_LABELS_ENTRY_SIDE_M5_V1, DB_NAME, MONGO_URI, PROJECT_CODE, PROJECT_NAME

RUN_TYPE = "sce_clear_entry_side_labels"
RUN_ID = f"{RUN_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear SCE entry-side labels safely.")
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--database", default=DB_NAME)
    parser.add_argument("--collection", default=COLL_LABELS_ENTRY_SIDE_M5_V1)
    parser.add_argument("--confirm-delete", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def write_report(args: argparse.Namespace, status: str, stats: Dict[str, Any], error_text: str = "") -> Path:
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{RUN_ID}.txt"
    lines = []
    lines.append("Side Context Entry AI - Entry Side Label Cleanup Report")
    lines.append("=" * 68)
    lines.append(f"run_id                   : {RUN_ID}")
    lines.append(f"status                   : {status}")
    lines.append(f"project_name             : {PROJECT_NAME}")
    lines.append(f"project_code             : {PROJECT_CODE}")
    lines.append(f"database                 : {args.database}")
    lines.append(f"mongo_uri                : {args.mongo_uri}")
    lines.append(f"target_collection        : {args.collection}")
    lines.append(f"confirm_delete           : {args.confirm_delete}")
    lines.append("collection_dropped        : False")
    lines.append("indexes_dropped           : False")
    lines.append("raw_data_modified         : False")
    lines.append("report_folder_policy      : local_section_reports")
    lines.append("")
    lines.append("Counts:")
    for key, value in stats.items():
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
    stats: Dict[str, Any] = {"existing_docs_before": 0, "deleted_docs": 0, "existing_docs_after": 0}
    client = None
    try:
        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[args.database]
        coll = db[args.collection]
        stats["existing_docs_before"] = coll.estimated_document_count()
        if args.confirm_delete != 1:
            report_path = write_report(args, "not_deleted_confirmation_required", stats)
            print("No documents deleted. Use --confirm-delete 1 to delete old labels.")
            print(f"Report: {report_path}")
            return 2
        result = coll.delete_many({})
        stats["deleted_docs"] = result.deleted_count
        stats["existing_docs_after"] = coll.estimated_document_count()
        report_path = write_report(args, "success", stats)
        print("SCE entry-side labels cleared.")
        print(f"Deleted docs: {result.deleted_count}")
        print(f"Report: {report_path}")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        print("ERROR: SCE entry-side label cleanup failed.")
        print(error_text)
        report_path = write_report(args, "failed", stats, error_text)
        print(f"Failure report: {report_path}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
