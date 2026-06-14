# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Diagnose Entry Context Dataset M15 v1

Purpose:
    Verify that MA context features inside bq2_dataset_entry_context_m15_v1 are populated and not constant/zero.

Run:
    cd C:/Project/BookQuality/bq2/04_dataset/entry_context
    python -u 02_diagnose_entry_context_m15_v1.py --limit 50000
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from pymongo import MongoClient, ASCENDING

COLL='bq2_dataset_entry_context_m15_v1'
DATASET_VERSION='entry_context_dataset_m15_v1'
DEFAULT_CONFIG={'mongo_uri':'mongodb://localhost:27017','database':'market_data','symbol':'XAUUSD'}
MA_KEYS=['predicted_direction','m30_future_direction','h1_future_direction','h4_future_direction','m30_phase','h1_phase','h4_phase']

def utc_now(): return datetime.now(timezone.utc)
def stamp(dt): return dt.strftime('%Y%m%d_%H%M%S')
def project_root(): return Path(__file__).resolve().parents[2]
def reports_dir():
    p=project_root()/'04_dataset'/'reports'; p.mkdir(parents=True, exist_ok=True); return p

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

def sf(v, default=0.0):
    try:
        x=float(v)
        return default if math.isnan(x) or math.isinf(x) else x
    except Exception: return default

def get_nested(d, path, default=None):
    cur=d
    for p in path.split('.'):
        if not isinstance(cur,dict) or p not in cur: return default
        cur=cur[p]
    return cur

def save_report(report, rs):
    jp=reports_dir()/f'entry_context_diagnostic_m15_v1_report_{rs}.json'
    tp=reports_dir()/f'entry_context_diagnostic_m15_v1_report_{rs}.txt'
    report['report_json_path']=str(jp); report['report_txt_path']=str(tp)
    jp.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    lines=['Book & Quality v2 - Entry Context Dataset M15 v1 Diagnostic Report','='*74]
    for k in ['run_id','status','database','symbol','start_time','end_time','duration_seconds']:
        lines.append(f'{k:<34}: {report.get(k)}')
    for title,key in [('Counts','counts'),('Feature Schema','feature_schema'),('MA Context Distributions','ma_context_distributions'),('MA Feature Summary','ma_feature_summary'),('Target by MA Predicted Direction','target_by_ma_predicted_direction'),('Warnings','warnings')]:
        lines+=['',title,'-'*74]
        data=report.get(key,{})
        if isinstance(data, list):
            for item in data: lines.append(f'- {item}')
        elif isinstance(data, dict):
            for kk,vv in data.items(): lines.append(f'{kk:<34}: {vv}')
        else:
            lines.append(str(data))
    tp.write_text('\n'.join(lines),encoding='utf-8')

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=50000)
    return p.parse_args()

def main():
    args=parse_args(); started=utc_now(); rs=stamp(started)
    report={'run_id':f'entry_context_diagnostic_m15_v1_{rs}','status':'running','database':None,'symbol':None,'start_time':started.isoformat(),'end_time':None,'duration_seconds':None,'counts':{},'feature_schema':{},'ma_context_distributions':{},'ma_feature_summary':{},'target_by_ma_predicted_direction':{},'warnings':[],'examples':[]}
    print('=== Diagnose Entry Context Dataset M15 v1 ===', flush=True)
    try:
        cfg=load_config(); db=connect(cfg); report['database']=cfg['database']; report['symbol']=cfg['symbol']
        cursor=db[COLL].find({'dataset_version':DATASET_VERSION},{'_id':0,'x':1,'y.book_direction':1,'split':1,'feature_meta':1,'ma_context':1,'anchor_time':1}).sort('anchor_time',ASCENDING)
        if args.limit: cursor=cursor.limit(args.limit)
        rows=0; split=Counter(); target=Counter(); fc=Counter(); book_count=None; ma_count=None
        ma_ctx={k:Counter() for k in MA_KEYS}
        cross=defaultdict(Counter)
        ma_nonzero=None; ma_sum=None; ma_sum2=None; ma_min=None; ma_max=None
        for doc in cursor:
            rows+=1; split[doc.get('split')]+=1
            y=get_nested(doc,'y.book_direction'); target[y]+=1
            x=doc.get('x') or []
            fm=doc.get('feature_meta') or {}
            bc=int(fm.get('book_feature_count') or 253); mc=int(fm.get('ma_feature_count') or max(len(x)-bc,0))
            book_count=bc; ma_count=mc; fc[len(x)]+=1
            mctx=doc.get('ma_context') or {}
            for k in MA_KEYS:
                v=mctx.get(k)
                ma_ctx[k][str(v)]+=1
            cross[str(mctx.get('predicted_direction'))][str(y)]+=1
            if len(x)>=bc+mc and mc>0:
                vals=[sf(v) for v in x[bc:bc+mc]]
                if ma_nonzero is None:
                    ma_nonzero=[0]*mc; ma_sum=[0.0]*mc; ma_sum2=[0.0]*mc; ma_min=[float('inf')]*mc; ma_max=[float('-inf')]*mc
                for i,v in enumerate(vals):
                    if abs(v)>1e-12: ma_nonzero[i]+=1
                    ma_sum[i]+=v; ma_sum2[i]+=v*v; ma_min[i]=min(ma_min[i],v); ma_max[i]=max(ma_max[i],v)
            if len(report['examples'])<5:
                report['examples'].append({'anchor_time':doc.get('anchor_time'),'split':doc.get('split'),'target':y,'ma_context':mctx})
        report['counts']={'rows_scanned':rows,'split_distribution':dict(split),'target_distribution':dict(target)}
        report['feature_schema']={'feature_count_distribution':dict(fc),'book_feature_count':book_count,'ma_feature_count':ma_count}
        report['ma_context_distributions']={k:dict(v) for k,v in ma_ctx.items()}
        if ma_nonzero is not None and rows>0:
            nz_cols=sum(1 for c in ma_nonzero if c>0); constant_cols=sum(1 for i in range(len(ma_nonzero)) if ma_min[i]==ma_max[i])
            zero_cols=[i for i,c in enumerate(ma_nonzero) if c==0]
            report['ma_feature_summary']={'ma_columns':len(ma_nonzero),'nonzero_columns':nz_cols,'all_zero_columns':len(zero_cols),'constant_columns':constant_cols,'zero_column_indexes_first_30':zero_cols[:30],'avg_nonzero_rate':round(sum(c/rows for c in ma_nonzero)/len(ma_nonzero),6)}
        report['target_by_ma_predicted_direction']={k:dict(v) for k,v in cross.items()}
        # warnings
        if rows==0: report['warnings'].append('No rows scanned.')
        if ma_nonzero is not None and sum(ma_nonzero)==0: report['warnings'].append('All MA feature values are zero. MA context extraction is broken or MA docs do not contain expected fields.')
        for k,c in ma_ctx.items():
            if len(c)<=1: report['warnings'].append(f'MA context key has only one value: {k} -> {dict(c)}')
            if 'None' in c and c['None']==rows: report['warnings'].append(f'MA context key is None for all rows: {k}')
        report['status']='success' if not report['warnings'] else 'success_with_warnings'
    except Exception as exc:
        report['status']='failed'; report['warnings'].append(str(exc)); print(f'[ERROR] {exc}', flush=True)
    finally:
        ended=utc_now(); report['end_time']=ended.isoformat(); report['duration_seconds']=round((ended-started).total_seconds(),3); save_report(report,rs)
    print('\n=== Final Summary ===')
    print('Status     :', report['status'])
    print('Rows       :', report.get('counts',{}).get('rows_scanned'))
    print('TXT Report :', report.get('report_txt_path'))
    print('[DONE]' if report['status']!='failed' else '[FAILED]')
    if report['status']=='failed': sys.exit(1)
if __name__=='__main__': main()
