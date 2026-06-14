# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Reset MongoDB to Raw Market Data Only v1

SAFETY FIRST
------------
Default mode is DRY RUN. It only reports what would be deleted.
To actually drop derived collections, you must pass:
    --execute --confirm RESET_TO_RAW_ONLY

This script keeps ONLY raw XAUUSD timeframe collections and system collections.
It drops all derived collections (bq_*, bq2_*, meta, datasets, labels, features, predictions, decisions, etc.)
when execute mode is confirmed.

DRY RUN:
    cd C:/Project/BookQuality/bq2/01_database/cleanup
    python -u 02_reset_to_raw_collections_only_v1.py

EXECUTE after checking the dry-run report:
    python -u 02_reset_to_raw_collections_only_v1.py --execute --confirm RESET_TO_RAW_ONLY
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
from pymongo import MongoClient

DEFAULT_CONFIG={'mongo_uri':'mongodb://localhost:27017','database':'market_data','symbol':'XAUUSD'}
CONFIRM_TEXT='RESET_TO_RAW_ONLY'
RAW_KEEP={
    'xauusd_m1','xauusd_m5','xauusd_m15','xauusd_m30','xauusd_h1','xauusd_h4','xauusd_d1'
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

def classify(name):
    if name in RAW_KEEP: return 'keep_raw'
    if name.startswith('system.'): return 'keep_system'
    return 'drop_derived'

def save_report(report,rs):
    jp=reports_dir()/f'reset_to_raw_only_v1_report_{rs}.json'
    tp=reports_dir()/f'reset_to_raw_only_v1_report_{rs}.txt'
    report['report_json_path']=str(jp); report['report_txt_path']=str(tp)
    jp.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    lines=['Book & Quality v2 - Reset to Raw Collections Only v1 Report','='*74]
    for k in ['run_id','status','mode','database','symbol','start_time','end_time','duration_seconds','raw_data_modified']:
        lines.append(f'{k:<34}: {report.get(k)}')
    lines+=['','Summary','-'*74]
    for k,v in report.get('summary',{}).items(): lines.append(f'{k:<34}: {v}')
    lines+=['','Kept Raw/System Collections','-'*74]
    for item in report.get('kept_collections',[]): lines.append(f"{item['name']:<45} {item['reason']:<14} count={item.get('count')}")
    lines+=['','Drop Candidates / Dropped','-'*74]
    for item in report.get('drop_collections',[]): lines.append(f"{item['name']:<45} {item['action']:<12} count={item.get('count')}")
    if report.get('errors'):
        lines+=['','Errors','-'*74]+[f'- {e}' for e in report['errors']]
    tp.write_text('\n'.join(lines),encoding='utf-8')

def parse_args():
    p=argparse.ArgumentParser(description='Reset database to raw XAUUSD collections only.')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--confirm', type=str, default='')
    p.add_argument('--no-count', action='store_true')
    return p.parse_args()

def main():
    args=parse_args(); started=utc_now(); rs=stamp(started)
    mode='execute' if args.execute else 'dry_run'
    report={'run_id':f'reset_to_raw_only_v1_{rs}','status':'running','mode':mode,'database':None,'symbol':None,'start_time':started.isoformat(),'end_time':None,'duration_seconds':None,'raw_data_modified':False,'summary':{},'kept_collections':[],'drop_collections':[],'errors':[],'config':{'raw_keep':sorted(RAW_KEEP),'confirm_text':CONFIRM_TEXT}}
    print('=== Book & Quality v2 - Reset to Raw Collections Only v1 ===', flush=True)
    try:
        if args.execute and args.confirm != CONFIRM_TEXT:
            raise RuntimeError(f'Execute mode requires --confirm {CONFIRM_TEXT}')
        cfg=load_config(); db=connect(cfg); report['database']=cfg['database']; report['symbol']=cfg['symbol']
        names=sorted(db.list_collection_names())
        kept=[]; drop=[]
        for name in names:
            reason=classify(name)
            cnt=None
            if not args.no_count:
                try: cnt=db[name].estimated_document_count()
                except Exception: cnt=None
            if reason.startswith('keep'):
                kept.append({'name':name,'reason':reason,'count':cnt})
            else:
                action='would_drop'
                if args.execute:
                    db.drop_collection(name); action='dropped'
                drop.append({'name':name,'reason':reason,'action':action,'count':cnt})
        report['kept_collections']=kept; report['drop_collections']=drop
        report['summary']={'collections_total':len(names),'kept_total':len(kept),'drop_candidates_total':len(drop),'dropped_total':sum(1 for x in drop if x['action']=='dropped'),'dry_run':not args.execute}
        report['status']='success'
    except Exception as exc:
        report['status']='failed'; report['errors'].append(str(exc)); print(f'[ERROR] {exc}', flush=True)
    finally:
        ended=utc_now(); report['end_time']=ended.isoformat(); report['duration_seconds']=round((ended-started).total_seconds(),3); save_report(report,rs)
    print('\n=== Final Summary ===')
    print('Status          :', report['status'])
    print('Mode            :', report['mode'])
    print('Kept            :', report.get('summary',{}).get('kept_total'))
    print('Drop candidates :', report.get('summary',{}).get('drop_candidates_total'))
    print('Dropped         :', report.get('summary',{}).get('dropped_total'))
    print('Raw Modified    :', report.get('raw_data_modified'))
    print('TXT Report      :', report.get('report_txt_path'))
    print('[DONE]' if report['status']!='failed' else '[FAILED]')
    if report['status']=='failed': sys.exit(1)
if __name__=='__main__': main()
