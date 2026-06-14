# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Cleanup Obsolete BQ2 MongoDB Collections v1

SAFETY FIRST
------------
Default mode is DRY RUN. It only reports what would be deleted.
To actually drop collections, you must pass:
    --execute --confirm DELETE_OBSOLETE_BQ2_COLLECTIONS

This script never drops raw market data collections.
It only considers collections starting with bq2_ and not in the active keep-list.

DRY RUN:
    cd C:/Project/BookQuality/bq2/01_database/cleanup
    python -u 01_cleanup_obsolete_bq2_collections_v1.py

EXECUTE after checking the dry-run report:
    python -u 01_cleanup_obsolete_bq2_collections_v1.py --execute --confirm DELETE_OBSOLETE_BQ2_COLLECTIONS
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from pymongo import MongoClient

DEFAULT_CONFIG={'mongo_uri':'mongodb://localhost:27017','database':'market_data','symbol':'XAUUSD'}
CONFIRM_TEXT='DELETE_OBSOLETE_BQ2_COLLECTIONS'

RAW_KEEP={
    'xauusd_m1','xauusd_m5','xauusd_m15','xauusd_m30','xauusd_h1','xauusd_h4','xauusd_d1',
    'market_data_m1','market_data_m5','market_data_m15','market_data_m30','market_data_h1','market_data_h4','market_data_d1',
}

# Active BQ2 collections for the current corrected architecture.
ACTIVE_KEEP_EXACT={
    # MA Scenario / Quality active pipeline
    'bq2_labels_ma_quality_m30_v1',
    'bq2_features_ma_quality_m30_v1',
    'bq2_dataset_ma_scenario_m30_v1',
    'bq2_predictions_ma_scenario_m30_v1',

    # Candle Book Direction v3 active as source for entry context
    'bq2_labels_candle_book_direction_m15_v3',
    'bq2_features_candle_book_direction_m15_v3',
    'bq2_dataset_candle_book_direction_m15_v3',

    # Entry Context active pipeline
    'bq2_dataset_entry_context_m15_v1',
    'bq2_labels_entry_context_m15_v1',
    'bq2_dataset_entry_context_trade_m15_v1',
    'bq2_predictions_entry_context_m15_v1',
}

ACTIVE_KEEP_PREFIXES={
    'bq2_meta_',
}

def utc_now(): return datetime.now(timezone.utc)
def stamp(dt): return dt.strftime('%Y%m%d_%H%M%S')
def project_root(): return Path(__file__).resolve().parents[2]
def reports_dir():
    p=project_root()/'01_database'/'reports'; p.mkdir(parents=True, exist_ok=True); return p

def read_json(path:Path):
    if not path.exists(): return None
    try: return json.loads(path.read_text(encoding='utf-8'))
    except Exception: return None

def deep_merge(a,b):
    out=dict(a)
    for k,v in b.items(): out[k]=deep_merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out

def load_config():
    cfg=DEFAULT_CONFIG
    fc=read_json(project_root()/'00_config'/'bq2_config.json')
    if fc: cfg=deep_merge(cfg,fc)
    return cfg

def connect(cfg):
    client=MongoClient(cfg['mongo_uri'], serverSelectionTimeoutMS=5000); client.admin.command('ping'); return client[cfg['database']]

def is_kept(name:str, extra_keep:set)->bool:
    if name in RAW_KEEP: return True
    if name in ACTIVE_KEEP_EXACT: return True
    if name in extra_keep: return True
    for p in ACTIVE_KEEP_PREFIXES:
        if name.startswith(p): return True
    if name.startswith('system.'): return True
    return False

def classify(name:str, extra_keep:set)->str:
    if name in RAW_KEEP: return 'keep_raw'
    if name in ACTIVE_KEEP_EXACT: return 'keep_active'
    if name in extra_keep: return 'keep_extra'
    for p in ACTIVE_KEEP_PREFIXES:
        if name.startswith(p): return 'keep_meta'
    if not name.startswith('bq2_'): return 'keep_non_bq2'
    return 'drop_obsolete_bq2'

def save_report(report, rs):
    jp=reports_dir()/f'cleanup_obsolete_bq2_collections_v1_report_{rs}.json'
    tp=reports_dir()/f'cleanup_obsolete_bq2_collections_v1_report_{rs}.txt'
    report['report_json_path']=str(jp); report['report_txt_path']=str(tp)
    jp.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    lines=['Book & Quality v2 - Cleanup Obsolete BQ2 Collections v1 Report','='*74]
    for k in ['run_id','status','mode','database','symbol','start_time','end_time','duration_seconds']:
        lines.append(f'{k:<34}: {report.get(k)}')
    lines+=['','Summary','-'*74]
    for k,v in report.get('summary',{}).items(): lines.append(f'{k:<34}: {v}')
    lines+=['','Kept Collections','-'*74]
    for item in report.get('kept_collections',[]): lines.append(f"{item['name']:<45} {item['reason']:<18} count={item.get('count')}")
    lines+=['','Drop Candidates / Dropped','-'*74]
    for item in report.get('drop_collections',[]): lines.append(f"{item['name']:<45} {item['action']:<12} count={item.get('count')}")
    if report.get('errors'):
        lines+=['','Errors','-'*74]+[f'- {e}' for e in report['errors']]
    tp.write_text('\n'.join(lines),encoding='utf-8')

def parse_args():
    p=argparse.ArgumentParser(description='Cleanup obsolete bq2_* collections safely.')
    p.add_argument('--execute', action='store_true', help='Actually drop obsolete bq2 collections. Default is dry-run.')
    p.add_argument('--confirm', type=str, default='', help=f'Required exact value for execute: {CONFIRM_TEXT}')
    p.add_argument('--extra-keep', type=str, default='', help='Comma-separated additional collection names to keep.')
    p.add_argument('--no-count', action='store_true', help='Skip count_documents for speed.')
    return p.parse_args()

def main():
    args=parse_args(); started=utc_now(); rs=stamp(started)
    mode='execute' if args.execute else 'dry_run'
    report={'run_id':f'cleanup_obsolete_bq2_collections_v1_{rs}','status':'running','mode':mode,'database':None,'symbol':None,'start_time':started.isoformat(),'end_time':None,'duration_seconds':None,'summary':{},'kept_collections':[],'drop_collections':[],'errors':[],'config':{'active_keep_exact':sorted(ACTIVE_KEEP_EXACT),'active_keep_prefixes':sorted(ACTIVE_KEEP_PREFIXES),'raw_keep':sorted(RAW_KEEP),'extra_keep':args.extra_keep}}
    print('=== Book & Quality v2 - Cleanup Obsolete BQ2 Collections v1 ===', flush=True)
    try:
        if args.execute and args.confirm != CONFIRM_TEXT:
            raise RuntimeError(f'Execute mode requires --confirm {CONFIRM_TEXT}')
        cfg=load_config(); db=connect(cfg); report['database']=cfg['database']; report['symbol']=cfg['symbol']
        extra_keep={x.strip() for x in args.extra_keep.split(',') if x.strip()}
        names=sorted(db.list_collection_names())
        kept=[]; drop=[]
        for name in names:
            reason=classify(name, extra_keep)
            cnt=None
            if not args.no_count:
                try: cnt=db[name].estimated_document_count()
                except Exception: cnt=None
            if reason=='drop_obsolete_bq2':
                action='would_drop'
                if args.execute:
                    db.drop_collection(name); action='dropped'
                drop.append({'name':name,'reason':reason,'action':action,'count':cnt})
            else:
                kept.append({'name':name,'reason':reason,'count':cnt})
        report['kept_collections']=kept; report['drop_collections']=drop
        report['summary']={'collections_total':len(names),'kept_total':len(kept),'obsolete_bq2_total':len(drop),'dropped_total':sum(1 for x in drop if x['action']=='dropped'),'dry_run':not args.execute}
        report['status']='success'
    except Exception as exc:
        report['status']='failed'; report['errors'].append(str(exc)); print(f'[ERROR] {exc}', flush=True)
    finally:
        ended=utc_now(); report['end_time']=ended.isoformat(); report['duration_seconds']=round((ended-started).total_seconds(),3); save_report(report,rs)
    print('\n=== Final Summary ===')
    print('Status       :', report['status'])
    print('Mode         :', report['mode'])
    print('Kept         :', report.get('summary',{}).get('kept_total'))
    print('Obsolete BQ2 :', report.get('summary',{}).get('obsolete_bq2_total'))
    print('Dropped      :', report.get('summary',{}).get('dropped_total'))
    print('TXT Report   :', report.get('report_txt_path'))
    print('[DONE]' if report['status']!='failed' else '[FAILED]')
    if report['status']=='failed': sys.exit(1)
if __name__=='__main__': main()
