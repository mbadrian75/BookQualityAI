# -*- coding: utf-8 -*-
from __future__ import annotations
import json, statistics, sys, traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
except ImportError as exc:
    print("ERROR: pymongo is not installed. Run: pip install pymongo")
    raise exc
from config.sce_project_config import DB_NAME, DERIVED_PREFIX, LABEL_CHECKER_TF, MONGO_URI, PROJECT_CODE, PROJECT_NAME
RUN_TYPE = "sce_raw_sources_inspection"
RUN_ID = f"{RUN_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
REQUIRED_TFS = ["M1", "M5", "M15", "M30", "H1", "H4"]
DATETIME_FIELDS = ["datetime", "time", "timestamp", "date", "DateTime", "Date", "Time"]
OHLC_FIELDS = {"open": ["open", "Open"], "high": ["high", "High"], "low": ["low", "Low"], "close": ["close", "Close"], "volume": ["volume", "Volume"]}

def normalize_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime): return value
    if isinstance(value, str):
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"]:
            try: return datetime.strptime(value[:19], fmt)
            except Exception: pass
    return None
def detect_datetime_field(doc: Dict[str, Any]) -> Optional[str]:
    for field in DATETIME_FIELDS:
        if field in doc and normalize_datetime(doc.get(field)) is not None: return field
    return None
def detect_ohlc_fields(doc: Dict[str, Any]) -> Dict[str, Optional[str]]:
    out = {}
    for key, candidates in OHLC_FIELDS.items():
        out[key] = None
        for field in candidates:
            if field in doc: out[key] = field; break
    return out
def tf_from_name(name: str) -> Optional[str]:
    lower = name.lower()
    for tf, token in [("M30", "m30"), ("M15", "m15"), ("M5", "m5"), ("M1", "m1"), ("H4", "h4"), ("H1", "h1"), ("D1", "d1")]:
        if token in lower: return tf
    return None
def tf_from_seconds(seconds: float) -> Optional[str]:
    for tf, expected in [("M1", 60), ("M5", 300), ("M15", 900), ("M30", 1800), ("H1", 3600), ("H4", 14400), ("D1", 86400)]:
        if expected * 0.8 <= seconds <= expected * 1.2: return tf
    return None
def infer_tf_by_delta(coll, dt_field: str) -> Tuple[Optional[str], Optional[float]]:
    docs = list(coll.find({dt_field: {"$exists": True}}, {dt_field: 1, "_id": 0}).sort(dt_field, ASCENDING).limit(300))
    times = [normalize_datetime(x.get(dt_field)) for x in docs]
    times = [x for x in times if x is not None]
    if len(times) < 3: return None, None
    deltas = [(times[i] - times[i - 1]).total_seconds() for i in range(1, len(times))]
    deltas = [x for x in deltas if 0 < x <= 86400]
    if not deltas: return None, None
    median_delta = float(statistics.median(deltas))
    return tf_from_seconds(median_delta), median_delta
def inspect_collection(db, name: str) -> Dict[str, Any]:
    coll = db[name]; sample = coll.find_one()
    result = {"collection": name, "is_sce": name.startswith(DERIVED_PREFIX), "estimated_count": None, "datetime_field": None, "ohlc_fields": {}, "tf_from_name": tf_from_name(name), "tf_from_delta": None, "median_delta_seconds": None, "inferred_tf": None, "first_time": None, "last_time": None, "status": "unknown"}
    try: result["estimated_count"] = coll.estimated_document_count()
    except Exception: pass
    if not sample: result["status"] = "empty"; return result
    dt_field = detect_datetime_field(sample); result["datetime_field"] = dt_field; result["ohlc_fields"] = detect_ohlc_fields(sample)
    if dt_field:
        delta_tf, median_delta = infer_tf_by_delta(coll, dt_field); result["tf_from_delta"] = delta_tf; result["median_delta_seconds"] = median_delta
        first_doc = coll.find_one({dt_field: {"$exists": True}}, {dt_field: 1}, sort=[(dt_field, ASCENDING)])
        last_doc = coll.find_one({dt_field: {"$exists": True}}, {dt_field: 1}, sort=[(dt_field, DESCENDING)])
        first_time = normalize_datetime(first_doc.get(dt_field)) if first_doc else None; last_time = normalize_datetime(last_doc.get(dt_field)) if last_doc else None
        result["first_time"] = first_time.isoformat(sep=" ") if first_time else None; result["last_time"] = last_time.isoformat(sep=" ") if last_time else None
    result["inferred_tf"] = result["tf_from_delta"] or result["tf_from_name"]
    has_ohlc = all(result["ohlc_fields"].get(x) for x in ["open", "high", "low", "close"])
    result["status"] = "valid_raw_candidate" if dt_field and has_ohlc and result["inferred_tf"] else "not_valid"
    return result
def choose_mapping(items: List[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
    mapping = {tf: None for tf in REQUIRED_TFS}
    for tf in REQUIRED_TFS:
        candidates = [x for x in items if not x["is_sce"] and x["inferred_tf"] == tf and x["status"] == "valid_raw_candidate"]
        if candidates:
            candidates.sort(key=lambda x: int(x.get("estimated_count") or 0), reverse=True); best = candidates[0]
            mapping[tf] = {"collection": best["collection"], "datetime_field": best["datetime_field"], "ohlc_fields": best["ohlc_fields"], "estimated_count": best["estimated_count"], "first_time": best["first_time"], "last_time": best["last_time"], "status": best["status"]}
    return mapping
def write_config(mapping: Dict[str, Optional[Dict[str, Any]]]) -> Path:
    config_path = ROOT / "config" / "sce_raw_collections_detected.json"
    config_path.write_text(json.dumps({"project_code": PROJECT_CODE, "database": DB_NAME, "label_checker_tf": LABEL_CHECKER_TF, "mapping": mapping}, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path
def write_report(status: str, inspections: List[Dict[str, Any]], mapping: Dict[str, Optional[Dict[str, Any]]], config_path: Optional[Path], error_text: str = "") -> Path:
    report_dir = Path(__file__).resolve().parent / "reports"; report_dir.mkdir(parents=True, exist_ok=True); report_path = report_dir / f"{RUN_ID}.txt"; missing = [tf for tf, value in mapping.items() if value is None]
    lines = ["Side Context Entry AI - Raw Sources Inspection Report", "=" * 64, f"run_id                   : {RUN_ID}", f"status                   : {status}", f"project_name             : {PROJECT_NAME}", f"project_code             : {PROJECT_CODE}", f"database                 : {DB_NAME}", f"mongo_uri                : {MONGO_URI}", "raw_data_modified         : False", "database_modified         : False", "report_folder_policy      : local_section_reports", "", "Summary:", f"total_collections         : {len(inspections)}", f"sce_collections_skipped   : {sum(1 for x in inspections if x['is_sce'])}", f"raw_collections_checked   : {sum(1 for x in inspections if not x['is_sce'])}", f"missing_required_tfs      : {', '.join(missing) if missing else 'None'}"]
    if config_path: lines.append(f"detected_config_path      : {config_path}")
    lines += ["", "Detected timeframe mapping:"]
    for tf in REQUIRED_TFS:
        item = mapping.get(tf); lines.append(f"- {tf}: NOT_FOUND" if item is None else f"- {tf}: {item['collection']} | time={item['datetime_field']} | count={item['estimated_count']} | from={item['first_time']} | to={item['last_time']} | status={item['status']}")
    lines += ["", "Collection details:"]
    for item in inspections:
        if item["is_sce"]: continue
        lines += [f"- collection             : {item['collection']}", f"  status                 : {item['status']}", f"  estimated_count        : {item['estimated_count']}", f"  inferred_tf            : {item['inferred_tf']}", f"  tf_from_name           : {item['tf_from_name']}", f"  tf_from_delta          : {item['tf_from_delta']}", f"  median_delta_seconds   : {item['median_delta_seconds']}", f"  datetime_field         : {item['datetime_field']}", f"  ohlc_fields            : {item['ohlc_fields']}", f"  first_time             : {item['first_time']}", f"  last_time              : {item['last_time']}", ""]
    if error_text: lines += ["Error:", error_text, ""]
    lines.append("End of report."); report_path.write_text("\n".join(lines), encoding="utf-8"); return report_path
def main() -> int:
    client = None; inspections = []; mapping = {tf: None for tf in REQUIRED_TFS}; config_path = None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000); client.admin.command("ping"); db = client[DB_NAME]
        inspections = [inspect_collection(db, name) for name in sorted(db.list_collection_names())]
        mapping = choose_mapping(inspections); config_path = write_config(mapping); missing = [tf for tf, value in mapping.items() if value is None]; status = "success" if not missing else "partial_success"
        report_path = write_report(status, inspections, mapping, config_path); print("SCE raw sources inspection completed."); print(f"Status: {status}"); print(f"Report: {report_path}"); return 0 if not missing else 2
    except Exception:
        error_text = traceback.format_exc(); print("ERROR: SCE raw sources inspection failed."); print(error_text); report_path = write_report("failed", inspections, mapping, config_path, error_text); print(f"Failure report: {report_path}"); return 1
    finally:
        if client is not None: client.close()
if __name__ == "__main__": raise SystemExit(main())
