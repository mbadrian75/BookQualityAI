# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Build Candle Book Labels M15 v1

Location:
    Book_Quality/02_labels/candle_book/01_build_candle_book_labels_m15_v1.py

Purpose:
    Build labels for Candle Book Confirmation Model.
    This model confirms entry timing using M1/M5/M15 features later.

Anchor:
    M15 closed candle.

Source:
    xauusd_m15

Target:
    bq2_labels_candle_book_m15_v1

Label:
    book_direction = BUY / SELL / NOTRADE

Logic:
    entry_time  = next M15 candle open
    entry_price = next M15 candle open

    In the next horizon_m15 bars:
      - if entry + base_r_usd is hit before entry - base_r_usd => BUY
      - if entry - base_r_usd is hit before entry + base_r_usd => SELL
      - if both are hit in same M15 candle => NOTRADE
      - if neither is hit, final horizon move decides only if strong enough

TEST:
    cd C:/Project/BookQuality/bq2/02_labels/candle_book
    python -u 01_build_candle_book_labels_m15_v1.py --limit 1000 --reset

FULL:
    python -u 01_build_candle_book_labels_m15_v1.py --reset
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

SCRIPT_NAME = '01_build_candle_book_labels_m15_v1.py'
LABEL_VERSION = 'candle_book_label_m15_v1'
BATCH_SIZE = 1000

DEFAULT_CONFIG = {
    'mongo_uri': 'mongodb://localhost:27017',
    'database': 'market_data',
    'symbol': 'XAUUSD',
    'raw_collections': {'m15': 'xauusd_m15'},
    'bq2_collections': {'labels_candle_book': 'bq2_labels_candle_book_m15_v1'},
    'candle_book_label_m15_v1': {
        'base_r_usd': 2.0,
        'horizon_m15': 8,
        'direction_threshold_r': 0.35,
    },
}

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def stamp(dt: datetime) -> str:
    return dt.strftime('%Y%m%d_%H%M%S')

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def reports_dir() -> Path:
    p = project_root() / '02_labels' / 'reports'
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f'[WARN] Could not read config {path}: {exc}', flush=True)
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
    file_cfg = read_json(project_root() / '00_config' / 'bq2_config.json')
    if file_cfg:
        cfg = deep_merge(cfg, file_cfg)
    cfg.setdefault('raw_collections', {})
    cfg.setdefault('bq2_collections', {})
    cfg['raw_collections']['m15'] = cfg['raw_collections'].get('m15', 'xauusd_m15')
    cfg['bq2_collections']['labels_candle_book'] = 'bq2_labels_candle_book_m15_v1'
    return cfg

def connect(cfg: Dict[str, Any]) -> Database:
    client = MongoClient(cfg['mongo_uri'], serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    return client[cfg['database']]

def parse_dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v.replace(tzinfo=None)
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace('Z', '+00:00')).replace(tzinfo=None)
    raise ValueError(f'Unsupported datetime: {v!r}')

def parse_dt_optional(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    return datetime.fromisoformat(v.replace('Z', '+00:00')).replace(tzinfo=None)

def sf(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default

def row_dt(row: Dict[str, Any]) -> datetime:
    return parse_dt(row.get('datetime') or row.get('time') or row.get('date'))

def candle_ohlc(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return sf(row['open']), sf(row['high']), sf(row['low']), sf(row['close'])

def build_label_for_anchor(anchor, next_bar, future_bars, base_r_usd, direction_threshold_r):
    if not future_bars:
        return None
    entry_time = row_dt(next_bar)
    entry_price = sf(next_bar['open'])
    buy_level = entry_price + base_r_usd
    sell_level = entry_price - base_r_usd
    first_hit = None
    first_hit_time = None
    first_hit_bar_index = None
    ambiguous_bar = False
    max_up = -10**9
    max_down = 10**9
    for i, bar in enumerate(future_bars):
        t = row_dt(bar)
        _, h, l, _ = candle_ohlc(bar)
        max_up = max(max_up, h - entry_price)
        max_down = min(max_down, l - entry_price)
        buy_hit = h >= buy_level
        sell_hit = l <= sell_level
        if buy_hit and sell_hit:
            first_hit = 'AMBIGUOUS'
            first_hit_time = t
            first_hit_bar_index = i
            ambiguous_bar = True
            break
        if buy_hit:
            first_hit = 'BUY'
            first_hit_time = t
            first_hit_bar_index = i
            break
        if sell_hit:
            first_hit = 'SELL'
            first_hit_time = t
            first_hit_bar_index = i
            break
    last_close = sf(future_bars[-1]['close'])
    final_move = last_close - entry_price
    threshold_usd = direction_threshold_r * base_r_usd
    if first_hit == 'BUY':
        direction = 'BUY'; method = 'tp_buy_first'; strength = min(1.0, max(0.50, abs(max_up) / max(base_r_usd, 1e-9)))
    elif first_hit == 'SELL':
        direction = 'SELL'; method = 'tp_sell_first'; strength = min(1.0, max(0.50, abs(max_down) / max(base_r_usd, 1e-9)))
    elif first_hit == 'AMBIGUOUS':
        direction = 'NOTRADE'; method = 'ambiguous_same_bar'; strength = 0.0
    else:
        if final_move >= threshold_usd:
            direction = 'BUY'; method = 'horizon_final_move_buy'; strength = min(1.0, abs(final_move) / max(base_r_usd, 1e-9))
        elif final_move <= -threshold_usd:
            direction = 'SELL'; method = 'horizon_final_move_sell'; strength = min(1.0, abs(final_move) / max(base_r_usd, 1e-9))
        else:
            direction = 'NOTRADE'; method = 'horizon_flat'; strength = 0.0
    return {
        'anchor_time': row_dt(anchor),
        'decision_time': entry_time,
        'entry_time': entry_time,
        'entry_price': entry_price,
        'label_version': LABEL_VERSION,
        'anchor_timeframe': 'M15',
        'future_horizon_m15': len(future_bars),
        'base_r_usd': base_r_usd,
        'direction_threshold_r': direction_threshold_r,
        'y': {
            'book_direction': direction,
            'label_method': method,
            'label_strength': float(strength),
            'entry_price': entry_price,
            'buy_level': buy_level,
            'sell_level': sell_level,
            'first_hit': first_hit,
            'first_hit_time': first_hit_time,
            'first_hit_bar_index': first_hit_bar_index,
            'ambiguous_bar': ambiguous_bar,
            'future_max_up_usd': float(max_up),
            'future_max_down_usd': float(max_down),
            'future_final_move_usd': float(final_move),
            'future_last_close': float(last_close),
        },
        'no_leak_note': 'Future fields are labels only. Feature builder must not read y/future fields.',
    }

def make_update_op(symbol: str, label_doc: Dict[str, Any]) -> UpdateOne:
    now = utc_now()
    doc = {'symbol': symbol, **label_doc, 'created_at': now, 'updated_at': now}
    created_at = doc.pop('created_at')
    return UpdateOne(
        {'symbol': symbol, 'anchor_time': doc['anchor_time']},
        {'$set': doc, '$setOnInsert': {'created_at': created_at}},
        upsert=True,
    )

def execute_bulk(db: Database, coll_name: str, ops: List[UpdateOne], report: Dict[str, Any]) -> None:
    if not ops:
        return
    try:
        res = db[coll_name].bulk_write(ops, ordered=False)
        report['counts']['upserted_or_modified'] += res.upserted_count + res.modified_count
        report['counts']['inserted'] += res.upserted_count
        report['counts']['modified'] += res.modified_count
        report['counts']['matched'] += res.matched_count
    except BulkWriteError as exc:
        report['status'] = 'failed'
        report['errors'].append(str(exc.details)[:20000])
        raise

def save_reports(report: Dict[str, Any], rs: str) -> None:
    jp = reports_dir() / f'candle_book_labels_m15_v1_report_{rs}.json'
    tp = reports_dir() / f'candle_book_labels_m15_v1_report_{rs}.txt'
    report['report_json_path'] = str(jp)
    report['report_txt_path'] = str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    lines = [
        'Book & Quality v2 - Candle Book Labels M15 v1 Report',
        '=' * 74,
        f"{'run_id':<34}: {report.get('run_id')}",
        f"{'status':<34}: {report.get('status')}",
        f"{'database':<34}: {report.get('database')}",
        f"{'symbol':<34}: {report.get('symbol')}",
        f"{'start_time':<34}: {report.get('start_time')}",
        f"{'end_time':<34}: {report.get('end_time')}",
        f"{'duration_seconds':<34}: {report.get('duration_seconds')}",
        '', 'Collections', '-' * 74,
    ]
    for k, v in report['collections'].items():
        lines.append(f'{k:<34}: {v}')
    lines += ['', 'Config', '-' * 74]
    for k, v in report['label_config'].items():
        lines.append(f'{k:<34}: {v}')
    lines += ['', 'Counts', '-' * 74]
    for k, v in report['counts'].items():
        lines.append(f'{k:<34}: {v}')
    lines += ['', 'Label Distribution', '-' * 74]
    for k, v in report['label_distribution'].items():
        lines.append(f'{k:<34}: {v}')
    lines += ['', 'Label Method Distribution', '-' * 74]
    for k, v in report['label_method_distribution'].items():
        lines.append(f'{k:<34}: {v}')
    if report['errors']:
        lines += ['', 'Errors', '-' * 74]
        lines += [f'- {e}' for e in report['errors'][:50]]
    tp.write_text('\n'.join(lines), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build Candle Book Labels M15 v1.')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--skip-existing', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    started = utc_now(); rs = stamp(started)
    cfg = load_config(); db_name = cfg['database']; symbol = cfg['symbol']
    raw_coll = cfg['raw_collections']['m15']
    target_coll = cfg['bq2_collections']['labels_candle_book']
    lcfg = cfg.get('candle_book_label_m15_v1', DEFAULT_CONFIG['candle_book_label_m15_v1'])
    base_r_usd = float(lcfg.get('base_r_usd', 2.0))
    horizon_m15 = int(lcfg.get('horizon_m15', 8))
    direction_threshold_r = float(lcfg.get('direction_threshold_r', 0.35))
    report: Dict[str, Any] = {
        'script_name': SCRIPT_NAME,
        'run_id': f'candle_book_labels_m15_v1_{rs}',
        'status': 'running',
        'database': db_name,
        'symbol': symbol,
        'start_time': started.isoformat(),
        'end_time': None,
        'duration_seconds': None,
        'collections': {'m15_source': raw_coll, 'target': target_coll},
        'label_config': {
            'label_version': LABEL_VERSION,
            'anchor_timeframe': 'M15',
            'entry_rule': 'entry_time = next M15 open after anchor_time',
            'base_r_usd': base_r_usd,
            'horizon_m15': horizon_m15,
            'direction_threshold_r': direction_threshold_r,
        },
        'counts': {
            'm15_loaded': 0, 'anchors_seen': 0, 'labels_built': 0,
            'upserted_or_modified': 0, 'inserted': 0, 'matched': 0, 'modified': 0,
            'skipped_existing': 0, 'skipped_no_future_path': 0, 'row_errors': 0,
        },
        'label_distribution': {}, 'label_method_distribution': {}, 'errors': [], 'args': vars(args),
    }
    print('=== Book & Quality v2 - Build Candle Book Labels M15 v1 ===', flush=True)
    print(f'Database : {db_name}', flush=True); print(f'Symbol   : {symbol}', flush=True)
    print(f'Source   : {raw_coll}', flush=True); print(f'Target   : {target_coll}', flush=True)
    try:
        db = connect(cfg)
        if args.reset:
            deleted = db[target_coll].delete_many({})
            report['reset_deleted_count'] = deleted.deleted_count
            print(f'[RESET] Deleted {deleted.deleted_count:,} docs from {target_coll}', flush=True)
        db[target_coll].create_index([('symbol', ASCENDING), ('anchor_time', ASCENDING)], unique=True, name='uq_symbol_anchor_time')
        db[target_coll].create_index([('label_version', ASCENDING), ('anchor_time', ASCENDING)], name='ix_label_version_anchor_time')
        db[target_coll].create_index([('y.book_direction', ASCENDING), ('anchor_time', ASCENDING)], name='ix_book_direction_anchor_time')
        query: Dict[str, Any] = {}
        time_filter: Dict[str, Any] = {}
        if args.start: time_filter['$gte'] = parse_dt_optional(args.start)
        if args.end: time_filter['$lte'] = parse_dt_optional(args.end)
        if time_filter: query['datetime'] = time_filter
        rows = list(db[raw_coll].find(query, projection={'_id':0,'datetime':1,'open':1,'high':1,'low':1,'close':1}).sort('datetime', ASCENDING))
        report['counts']['m15_loaded'] = len(rows)
        if len(rows) <= horizon_m15 + 1: raise RuntimeError('Not enough M15 rows to build labels.')
        ops: List[UpdateOne] = []
        max_anchor_index = len(rows) - horizon_m15 - 1
        for i in range(max_anchor_index):
            try:
                if args.limit is not None and report['counts']['anchors_seen'] >= args.limit: break
                anchor = rows[i]; next_bar = rows[i+1]; future_bars = rows[i+1:i+1+horizon_m15]
                anchor_time = row_dt(anchor)
                report['counts']['anchors_seen'] += 1
                if args.skip_existing:
                    exists = db[target_coll].find_one({'symbol': symbol, 'anchor_time': anchor_time}, projection={'_id': 1})
                    if exists:
                        report['counts']['skipped_existing'] += 1; continue
                label_doc = build_label_for_anchor(anchor, next_bar, future_bars, base_r_usd, direction_threshold_r)
                if not label_doc:
                    report['counts']['skipped_no_future_path'] += 1; continue
                y = label_doc['y']; direction = y['book_direction']; method = y['label_method']
                report['label_distribution'][direction] = report['label_distribution'].get(direction, 0) + 1
                report['label_method_distribution'][method] = report['label_method_distribution'].get(method, 0) + 1
                ops.append(make_update_op(symbol, label_doc)); report['counts']['labels_built'] += 1
                if len(ops) >= BATCH_SIZE:
                    execute_bulk(db, target_coll, ops, report); ops = []
                    print(f"[PROGRESS] anchors={report['counts']['anchors_seen']:,} built={report['counts']['labels_built']:,} written≈{report['counts']['upserted_or_modified']:,}", flush=True)
            except Exception as row_exc:
                report['counts']['row_errors'] += 1
                if len(report['errors']) < 100:
                    report['errors'].append(f"i={i} anchor={anchor.get('datetime')} | {row_exc}")
        if ops: execute_bulk(db, target_coll, ops, report)
        report['final_target_count'] = db[target_coll].estimated_document_count()
        report['status'] = 'success' if report['counts']['row_errors'] == 0 and not report['errors'] else 'success_with_row_errors'
    except Exception as exc:
        report['status'] = 'failed'
        if not report['errors']: report['errors'].append(str(exc))
        print(f'[ERROR] {exc}', flush=True)
    finally:
        ended = utc_now(); report['end_time'] = ended.isoformat(); report['duration_seconds'] = round((ended-started).total_seconds(), 3); save_reports(report, rs)
    print('\n=== Final Summary ===', flush=True)
    print(f"Status            : {report['status']}", flush=True)
    print(f"M15 Loaded        : {report['counts']['m15_loaded']:,}", flush=True)
    print(f"Anchors Seen      : {report['counts']['anchors_seen']:,}", flush=True)
    print(f"Labels Built      : {report['counts']['labels_built']:,}", flush=True)
    print(f"Upserted/Modified : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Row Errors        : {report['counts']['row_errors']:,}", flush=True)
    print(f"Label Distribution: {report['label_distribution']}", flush=True)
    print(f"TXT Report        : {report.get('report_txt_path')}", flush=True)
    print('[DONE]' if report['status'] != 'failed' else '[FAILED]', flush=True)
    if report['status'] == 'failed': sys.exit(1)

if __name__ == '__main__':
    main()
