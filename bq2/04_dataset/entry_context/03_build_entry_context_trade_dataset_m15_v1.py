# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Build Entry Context Trade Dataset M15 v1

Purpose
-------
Join combined Entry Context features with trade-oriented Entry Context labels.

Source features dataset:
    bq2_dataset_entry_context_m15_v1
Source labels:
    bq2_labels_entry_context_m15_v1
Output dataset:
    bq2_dataset_entry_context_trade_m15_v1

Target:
    entry_label = BUY / SELL / NOTRADE

TEST:
    cd C:/Project/BookQuality/bq2/04_dataset/entry_context
    python -u 03_build_entry_context_trade_dataset_m15_v1.py --limit 1000 --reset

FULL:
    python -u 03_build_entry_context_trade_dataset_m15_v1.py --reset
"""
from __future__ import annotations
import argparse, json, math, sys, shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pymongo import MongoClient, ASCENDING

DATASET_VERSION='entry_context_trade_dataset_m15_v1'
SOURCE_DATASET_VERSION='entry_context_dataset_m15_v1'
LABEL_VERSION='entry_context_label_m15_v1'
SOURCE_DATASET_COLLECTION='bq2_dataset_entry_context_m15_v1'
LABEL_COLLECTION='bq2_labels_entry_context_m15_v1'
TARGET_COLLECTION='bq2_dataset_entry_context_trade_m15_v1'
TARGET='entry_label'

DEFAULT_CONFIG={'mongo_uri':'mongodb://localhost:27017','database':'market_data','symbol':'XAUUSD'}

def utc_now(): return datetime.now(timezone.utc)
def stamp(dt): return dt.strftime('%Y%m%d_%H%M%S')
def project_root(): return Path(__file__).resolve().parents[2]
def reports_dir():
    p=project_root()/'04_dataset'/'reports'; p.mkdir(parents=True, exist_ok=True); return p
def exports_dir():
    p=project_root()/'04_dataset'/'exports'; p.mkdir(parents=True, exist_ok=True); return p

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

def parse_dt(v):
    if v is None: return None
    if isinstance(v, datetime): return v.replace(tzinfo=None)
    if isinstance(v, str):
        try: return datetime.fromisoformat(v.replace('Z','+00:00')).replace(tzinfo=None)
        except Exception: return None
    return None

def sf(v, default=0.0):
    try:
        x=float(v)
        return default if math.isnan(x) or math.isinf(x) else x
    except Exception: return default

def latest_feature_names():
    files=sorted(exports_dir().glob('entry_context_m15_v1_feature_names_*.json'))
    if not files: return [], None
    p=files[-1]
    data=read_json(p) or {}
    return list(data.get('feature_names',[])), str(p)

def save_feature_names(names, source_path, rs):
    out=exports_dir()/f'entry_context_trade_m15_v1_feature_names_{rs}.json'
    out.write_text(json.dumps({'dataset_version':DATASET_VERSION,'source_dataset_version':SOURCE_DATASET_VERSION,'label_version':LABEL_VERSION,'feature_names':names,'feature_count':len(names),'source_feature_names_path':source_path},ensure_ascii=False,indent=2),encoding='utf-8')
    return str(out)

def save_report(report, rs):
    jp=reports_dir()/f'entry_context_trade_dataset_m15_v1_report_{rs}.json'
    tp=reports_dir()/f'entry_context_trade_dataset_m15_v1_report_{rs}.txt'
    report['report_json_path']=str(jp); report['report_txt_path']=str(tp)
    jp.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    lines=['Book & Quality v2 - Entry Context Trade Dataset M15 v1 Report','='*74]
    for k in ['run_id','status','database','symbol','start_time','end_time','duration_seconds']:
        lines.append(f'{k:<34}: {report.get(k)}')
    for title,key in [('Collections','collections'),('Config','config'),('Counts','counts'),('Feature Schema','feature_schema'),('Split Distribution','split_distribution'),('Label Distribution','label_distribution'),('Label Method Distribution','label_method_distribution')]:
        lines+=['',title,'-'*74]
        for kk,vv in report.get(key,{}).items(): lines.append(f'{kk:<34}: {vv}')
    if report.get('errors'):
        lines+=['','Errors','-'*74]+[f'- {e}' for e in report['errors'][:50]]
    tp.write_text('\n'.join(lines),encoding='utf-8')

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--reset', action='store_true')
    return p.parse_args()

def main():
    args=parse_args(); started=utc_now(); rs=stamp(started)
    report={'run_id':f'entry_context_trade_dataset_m15_v1_{rs}','status':'running','database':None,'symbol':None,'start_time':started.isoformat(),'end_time':None,'duration_seconds':None,'collections':{},'config':{},'counts':{},'feature_schema':{},'split_distribution':{},'label_distribution':{},'label_method_distribution':{},'errors':[],'args':vars(args)}
    print('=== Book & Quality v2 - Build Entry Context Trade Dataset M15 v1 ===', flush=True)
    try:
        cfg=load_config(); db=connect(cfg); report['database']=cfg['database']; report['symbol']=cfg['symbol']
        report['collections']={'source_dataset':SOURCE_DATASET_COLLECTION,'labels_source':LABEL_COLLECTION,'target':TARGET_COLLECTION}
        report['config']={'dataset_version':DATASET_VERSION,'source_dataset_version':SOURCE_DATASET_VERSION,'label_version':LABEL_VERSION,'target':TARGET}
        target=db[TARGET_COLLECTION]
        if args.reset:
            print(f'[RESET] Dropping target collection: {TARGET_COLLECTION}', flush=True); target.drop()
        target.create_index([('anchor_time',ASCENDING)], unique=True)
        target.create_index([('split',ASCENDING)])
        target.create_index([('dataset_version',ASCENDING),('split',ASCENDING)])
        feature_names, source_feature_path=latest_feature_names()
        feature_export=save_feature_names(feature_names, source_feature_path, rs)
        counts=Counter(); split_counts=Counter(); label_counts=Counter(); method_counts=Counter(); errs=[]
        cursor=db[SOURCE_DATASET_COLLECTION].find({'dataset_version':SOURCE_DATASET_VERSION},{'_id':0}).sort('anchor_time',ASCENDING)
        if args.limit: cursor=cursor.limit(args.limit)
        for doc in cursor:
            counts['source_rows_seen']+=1
            try:
                at=parse_dt(doc.get('anchor_time'))
                if at is None:
                    counts['skipped_missing_anchor_time']+=1; continue
                lab=db[LABEL_COLLECTION].find_one({'label_version':LABEL_VERSION,'anchor_time':at},{'_id':0})
                if not lab:
                    counts['skipped_missing_label']+=1; continue
                x=doc.get('x')
                if not isinstance(x,list) or not x:
                    counts['skipped_missing_x']+=1; continue
                entry_label=lab.get('entry_label')
                if entry_label not in ['BUY','SELL','NOTRADE']:
                    counts['skipped_bad_label']+=1; continue
                y={k:lab.get(k) for k in ['entry_label','label_method','buy_quality_score','sell_quality_score','quality_gap','abs_quality_gap','buy_runup_usd','buy_drawdown_usd','sell_runup_usd','sell_drawdown_usd','soft_buy_score','soft_sell_score','soft_notrade_score'] if k in lab}
                out={'dataset_version':DATASET_VERSION,'source_dataset_version':SOURCE_DATASET_VERSION,'label_version':LABEL_VERSION,'symbol':cfg['symbol'],'anchor_time':at,'decision_time':doc.get('decision_time'),'split':doc.get('split'),'target':TARGET,'x':[sf(v) for v in x],'y':y,'feature_meta':{**(doc.get('feature_meta') or {}),'feature_count':len(x),'feature_names_export':feature_export,'source_feature_names_path':source_feature_path}}
                target.replace_one({'anchor_time':at}, out, upsert=True)
                counts['rows_built']+=1; split_counts[out['split']]+=1; label_counts[entry_label]+=1; method_counts[y.get('label_method')]+=1
                if counts['rows_built']%50000==0: print(f"[PROGRESS] rows_built={counts['rows_built']}", flush=True)
            except Exception as exc:
                counts['row_errors']+=1
                if len(errs)<20: errs.append(str(exc))
        report['counts']=dict(counts); report['split_distribution']=dict(split_counts); report['label_distribution']=dict(label_counts); report['label_method_distribution']=dict(method_counts)
        report['feature_schema']={'feature_count':len(feature_names) if feature_names else None,'feature_names_export':feature_export,'source_feature_names_path':source_feature_path}
        if errs: report['errors']=errs
        report['status']='success' if counts.get('row_errors',0)==0 and counts.get('rows_built',0)>0 else 'success_with_warnings'
    except Exception as exc:
        report['status']='failed'; report['errors'].append(str(exc)); print(f'[ERROR] {exc}', flush=True)
    finally:
        ended=utc_now(); report['end_time']=ended.isoformat(); report['duration_seconds']=round((ended-started).total_seconds(),3); save_report(report,rs)
    print('\n=== Final Summary ===')
    print('Status     :', report['status'])
    print('Rows Built :', report.get('counts',{}).get('rows_built'))
    print('TXT Report :', report.get('report_txt_path'))
    print('[DONE]' if report['status']!='failed' else '[FAILED]')
    if report['status']=='failed': sys.exit(1)
if __name__=='__main__': main()
