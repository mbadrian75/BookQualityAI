# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Build Entry Context Dataset M15 v1

Purpose
-------
Combine Candle Book Direction M15 dataset rows (x vector + y) with the latest available MA Scenario M30 prediction.

Input collections:
    bq2_dataset_candle_book_direction_m15_v3
    bq2_predictions_ma_scenario_m30_v1

Output collection:
    bq2_dataset_entry_context_m15_v1

No raw collections are modified.

TEST:
    cd C:/Project/BookQuality/bq2/04_dataset/entry_context
    python -u 01_build_entry_context_dataset_m15_v1.py --limit 1000 --reset

FULL:
    python -u 01_build_entry_context_dataset_m15_v1.py --reset
"""
from __future__ import annotations

import argparse, bisect, json, math, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING
from pymongo.database import Database

DATASET_VERSION='entry_context_dataset_m15_v1'
BOOK_FEATURE_VERSION='candle_book_direction_features_m15_v3'
BOOK_LABEL_VERSION='candle_book_direction_label_m15_v3'
MA_MODEL_VERSION='ma_scenario_model_m30_v1'
MA_PREDICTION_COLLECTION='bq2_predictions_ma_scenario_m30_v1'
BOOK_SOURCE_COLLECTION='bq2_dataset_candle_book_direction_m15_v3'
TARGET_COLLECTION='bq2_dataset_entry_context_m15_v1'
TARGET='book_direction'

TRAIN_END=datetime(2023,1,1)
VALID_END=datetime(2024,1,1)

DIR_CATS=['BUY','SELL','RANGE']
PHASE_CATS=['UP_START','UP_MIDDLE','UP_END','UP_EXHAUSTION','DOWN_START','DOWN_MIDDLE','DOWN_END','DOWN_EXHAUSTION','REVERSAL_UP_START','REVERSAL_DOWN_START','RANGE']
DIR_TARGETS=['predicted_direction','m30_future_direction','h1_future_direction','h4_future_direction']
PHASE_TARGETS=['m30_phase','h1_phase','h4_phase']

DEFAULT_CONFIG={
    'mongo_uri':'mongodb://localhost:27017',
    'database':'market_data',
    'symbol':'XAUUSD',
}

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
    except Exception as exc:
        print(f'[WARN] Could not read json {path}: {exc}', flush=True); return None

def deep_merge(base:Dict[str,Any], incoming:Dict[str,Any]):
    out=dict(base)
    for k,v in incoming.items():
        out[k]=deep_merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out

def load_config():
    cfg=DEFAULT_CONFIG
    file_cfg=read_json(project_root()/'00_config'/'bq2_config.json')
    if file_cfg: cfg=deep_merge(cfg,file_cfg)
    return cfg

def connect(cfg):
    client=MongoClient(cfg['mongo_uri'], serverSelectionTimeoutMS=5000); client.admin.command('ping'); return client[cfg['database']]

def sf(v, default=0.0):
    try:
        x=float(v)
        if math.isnan(x) or math.isinf(x): return default
        return x
    except Exception: return default

def parse_dt(v):
    if v is None: return None
    if isinstance(v, datetime): return v.replace(tzinfo=None)
    if isinstance(v, str):
        s=v.replace('Z','+00:00')
        try: return datetime.fromisoformat(s).replace(tzinfo=None)
        except Exception: return None
    return None

def split_for(dt:datetime)->str:
    d=dt.replace(tzinfo=None)
    if d<TRAIN_END: return 'train'
    if d<VALID_END: return 'valid'
    return 'test'

def get_nested(doc:Dict[str,Any], dotted:str, default=None):
    cur=doc
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur: return default
        cur=cur[part]
    return cur

def latest_feature_names(prefix:str)->Tuple[List[str], Optional[str]]:
    files=sorted(exports_dir().glob(prefix))
    if not files: return [], None
    path=files[-1]
    data=read_json(path) or {}
    return list(data.get('feature_names', [])), str(path)

def extract_pred_value(doc:Dict[str,Any], target:str):
    candidates=[
        target,
        f'prediction.{target}',
        f'predictions.{target}',
        f'y_pred.{target}',
        f'outputs.{target}',
        f'result.{target}',
    ]
    for key in candidates:
        val=get_nested(doc,key) if '.' in key else doc.get(key)
        if isinstance(val, dict):
            for sub in ['label','pred','prediction','value','class']:
                if sub in val: return val[sub]
        if val is not None: return val
    return None

def normalize_label(v):
    if v is None: return None
    s=str(v)
    # handle numpy repr-like values if they were stringified in some reports/documents
    if "np.str_('" in s: s=s.split("np.str_('",1)[1].split("')",1)[0]
    return s.strip().upper()

def extract_prob_dict(doc:Dict[str,Any], target:str)->Dict[str,float]:
    # Robustly handle multiple possible schemas produced by prediction scripts.
    candidates=[
        f'{target}_probabilities', f'{target}_proba', f'{target}_probs',
        f'probabilities.{target}', f'proba.{target}', f'probs.{target}',
        f'prediction_probabilities.{target}', f'prediction_proba.{target}',
        f'predictions.{target}.probabilities', f'predictions.{target}.proba',
        f'outputs.{target}.probabilities', f'outputs.{target}.proba',
    ]
    for key in candidates:
        val=get_nested(doc,key) if '.' in key else doc.get(key)
        if isinstance(val, dict):
            out={}
            for k,v in val.items(): out[normalize_label(k) or str(k).upper()]=sf(v)
            if out: return out
    # Some scripts store flat p_buy/p_sell/p_range only for predicted_direction
    if target=='predicted_direction':
        flat={}
        for cat in DIR_CATS:
            for key in [f'p_{cat.lower()}', f'prob_{cat.lower()}', f'predicted_direction_p_{cat.lower()}']:
                if key in doc: flat[cat]=sf(doc.get(key))
        if flat: return flat
    return {}

def one_hot_probs(pred_label, cats:List[str], prob_dict:Dict[str,float])->List[float]:
    pred=normalize_label(pred_label)
    vals=[]
    has_prob=bool(prob_dict)
    for cat in cats:
        if has_prob:
            vals.append(sf(prob_dict.get(cat, 0.0)))
        else:
            vals.append(1.0 if pred==cat else 0.0)
    return vals

def confidence(pred_label, cats:List[str], prob_dict:Dict[str,float])->float:
    pred=normalize_label(pred_label)
    if prob_dict and pred in prob_dict: return sf(prob_dict[pred])
    vals=one_hot_probs(pred, cats, prob_dict)
    return max(vals) if vals else 0.0

def build_ma_feature_names()->List[str]:
    names=[]
    for t in DIR_TARGETS:
        for c in DIR_CATS: names.append(f'ma_{t}_p_{c.lower()}')
        names.append(f'ma_{t}_confidence')
    for t in PHASE_TARGETS:
        for c in PHASE_CATS: names.append(f'ma_{t}_is_{c.lower()}')
        names.append(f'ma_{t}_confidence')
    names.append('ma_prediction_lag_minutes')
    return names

def build_ma_vector(doc:Dict[str,Any], m15_time:datetime)->Tuple[List[float], Dict[str,Any]]:
    x=[]; meta={}
    for t in DIR_TARGETS:
        pred=extract_pred_value(doc,t); prob=extract_prob_dict(doc,t)
        pred_norm=normalize_label(pred)
        meta[t]=pred_norm
        probs=one_hot_probs(pred_norm, DIR_CATS, prob)
        x.extend(probs); x.append(confidence(pred_norm, DIR_CATS, prob))
    for t in PHASE_TARGETS:
        pred=extract_pred_value(doc,t); prob=extract_prob_dict(doc,t)
        pred_norm=normalize_label(pred)
        meta[t]=pred_norm
        probs=one_hot_probs(pred_norm, PHASE_CATS, prob)
        x.extend(probs); x.append(confidence(pred_norm, PHASE_CATS, prob))
    ma_time=parse_dt(doc.get('decision_time')) or parse_dt(doc.get('anchor_time')) or parse_dt(doc.get('datetime')) or parse_dt(doc.get('time'))
    lag=0.0
    if ma_time:
        lag=(m15_time-ma_time).total_seconds()/60.0
    x.append(sf(lag)); meta['ma_prediction_time']=ma_time; meta['ma_prediction_lag_minutes']=lag
    return x, meta

def load_ma_predictions(db:Database, coll:str):
    projection=None
    docs=[]; times=[]; skipped=0
    cursor=db[coll].find({}, projection=projection).sort([('decision_time', ASCENDING), ('anchor_time', ASCENDING)])
    for doc in cursor:
        t=parse_dt(doc.get('decision_time')) or parse_dt(doc.get('anchor_time')) or parse_dt(doc.get('datetime')) or parse_dt(doc.get('time'))
        if t is None:
            skipped+=1; continue
        docs.append(doc); times.append(t)
    pairs=sorted(zip(times, docs), key=lambda z:z[0])
    if pairs:
        times=[p[0] for p in pairs]; docs=[p[1] for p in pairs]
    return times, docs, skipped

def find_latest_ma(times:List[datetime], docs:List[Dict[str,Any]], decision_time:datetime):
    idx=bisect.bisect_right(times, decision_time)-1
    if idx<0: return None, None
    return times[idx], docs[idx]

def book_y(doc:Dict[str,Any])->Dict[str,Any]:
    y=doc.get('y') if isinstance(doc.get('y'),dict) else {}
    out={}
    keys=['book_direction','label_method','soft_buy_score','soft_sell_score','soft_unclear_score','buy_pressure','sell_pressure','pressure_gap','abs_pressure_gap','gap_ratio']
    for k in keys:
        val=doc.get(k, y.get(k))
        if val is not None: out[k]=val
    return out

def book_anchor_time(doc):
    return parse_dt(doc.get('anchor_time')) or parse_dt(doc.get('decision_time')) or parse_dt(doc.get('datetime')) or parse_dt(doc.get('time'))

def book_decision_time(doc):
    return parse_dt(doc.get('decision_time')) or book_anchor_time(doc)

def save_report(report, rs):
    jp=reports_dir()/f'entry_context_dataset_m15_v1_report_{rs}.json'
    tp=reports_dir()/f'entry_context_dataset_m15_v1_report_{rs}.txt'
    report['report_json_path']=str(jp); report['report_txt_path']=str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    lines=['Book & Quality v2 - Entry Context Dataset M15 v1 Report','='*74]
    for k in ['run_id','status','database','symbol','start_time','end_time','duration_seconds']:
        lines.append(f'{k:<34}: {report.get(k)}')
    for title,key in [('Collections','collections'),('Config','config'),('Counts','counts'),('Feature Schema','feature_schema'),('Split Distribution','split_distribution'),('Direction Distribution','direction_distribution'),('MA Match Distribution','ma_match_distribution')]:
        lines+=['',title,'-'*74]
        data=report.get(key,{})
        for kk,vv in data.items(): lines.append(f'{kk:<34}: {vv}')
    if report.get('errors'):
        lines+=['','Errors','-'*74]+[f'- {e}' for e in report['errors'][:100]]
    tp.write_text('\n'.join(lines), encoding='utf-8')

def parse_args():
    p=argparse.ArgumentParser(description='Build Entry Context M15 v1 dataset from Book features + MA predictions.')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--reset', action='store_true')
    p.add_argument('--max-ma-lag-minutes', type=float, default=60.0, help='Skip if latest MA prediction is older than this. Default 60 minutes.')
    p.add_argument('--start-time', type=str, default=None, help='Optional ISO datetime lower bound for Book rows, e.g. 2010-03-01 00:00:00')
    return p.parse_args()

def main():
    args=parse_args(); started=utc_now(); rs=stamp(started)
    report={'run_id':f'entry_context_dataset_m15_v1_{rs}','status':'running','database':None,'symbol':None,'start_time':started.isoformat(),'end_time':None,'duration_seconds':None,'collections':{},'config':{},'counts':{},'feature_schema':{},'split_distribution':{},'direction_distribution':{},'ma_match_distribution':{},'errors':[],'args':vars(args)}
    print('=== Book & Quality v2 - Build Entry Context Dataset M15 v1 ===', flush=True)
    try:
        cfg=load_config(); db=connect(cfg)
        report['database']=cfg['database']; report['symbol']=cfg['symbol']
        report['collections']={'book_dataset_source':BOOK_SOURCE_COLLECTION,'ma_predictions_source':MA_PREDICTION_COLLECTION,'target':TARGET_COLLECTION}
        report['config']={'dataset_version':DATASET_VERSION,'book_feature_version':BOOK_FEATURE_VERSION,'book_label_version':BOOK_LABEL_VERSION,'ma_model_version':MA_MODEL_VERSION,'target':TARGET,'train_end':str(TRAIN_END),'valid_end':str(VALID_END),'max_ma_lag_minutes':args.max_ma_lag_minutes,'require_no_leak_ok':True}
        target=db[TARGET_COLLECTION]
        if args.reset:
            print(f'[RESET] Dropping target collection: {TARGET_COLLECTION}', flush=True)
            target.drop()
        target.create_index([('anchor_time', ASCENDING)], unique=True)
        target.create_index([('split', ASCENDING)])
        target.create_index([('dataset_version', ASCENDING), ('split', ASCENDING)])
        print('[LOAD] MA predictions...', flush=True)
        ma_times, ma_docs, skipped_ma_no_time=load_ma_predictions(db, MA_PREDICTION_COLLECTION)
        if not ma_docs: raise RuntimeError('No MA prediction documents loaded.')
        print(f'[LOAD] MA predictions loaded: {len(ma_docs)} skipped_no_time={skipped_ma_no_time}', flush=True)
        book_feature_names, book_feature_names_source=latest_feature_names('candle_book_direction_m15_v3_feature_names_*.json')
        ma_feature_names=build_ma_feature_names()
        feature_names=(book_feature_names if book_feature_names else [f'book_f_{i}' for i in range(253)]) + ma_feature_names
        feature_names_export=exports_dir()/f'entry_context_m15_v1_feature_names_{rs}.json'
        feature_names_export.write_text(json.dumps({'dataset_version':DATASET_VERSION,'feature_names':feature_names,'book_feature_count':len(feature_names)-len(ma_feature_names),'ma_feature_count':len(ma_feature_names),'feature_count':len(feature_names),'book_feature_names_source':book_feature_names_source}, ensure_ascii=False, indent=2), encoding='utf-8')
        counts=Counter(); split_counts=Counter(); dir_counts=Counter(); ma_match_counts=Counter(); label_method_counts=Counter(); examples_errors=[]
        book_query={}
        if args.start_time:
            st=parse_dt(args.start_time)
            if st is None: raise RuntimeError(f'Invalid --start-time: {args.start_time}')
            book_query={'anchor_time': {'$gte': st}}
        cursor=db[BOOK_SOURCE_COLLECTION].find(book_query, projection=None).sort('anchor_time', ASCENDING)
        # --limit is applied after successful rows_built, not on rows scanned.
        bulk_count=0
        for doc in cursor:
            counts['book_rows_seen']+=1
            try:
                no_leak=get_nested(doc,'feature_meta.no_leak_ok', True)
                if no_leak is False:
                    counts['skipped_book_no_leak_not_ok']+=1; continue
                ax=book_anchor_time(doc); dt=book_decision_time(doc)
                if ax is None or dt is None:
                    counts['skipped_missing_time']+=1; continue
                bx=doc.get('x')
                if not isinstance(bx, list) or not bx:
                    counts['skipped_missing_book_x']+=1; continue
                y=book_y(doc)
                if y.get('book_direction') not in ['BUY','SELL','UNCLEAR']:
                    counts['skipped_missing_book_direction']+=1; continue
                ma_time, ma_doc=find_latest_ma(ma_times, ma_docs, dt)
                if ma_doc is None:
                    counts['skipped_missing_ma_prediction']+=1; continue
                lag=(dt-ma_time).total_seconds()/60.0
                if lag < -1e-9:
                    counts['skipped_ma_after_decision']+=1; continue
                if args.max_ma_lag_minutes is not None and lag > args.max_ma_lag_minutes:
                    counts['skipped_ma_lag_too_high']+=1; continue
                mx, ma_meta=build_ma_vector(ma_doc, dt)
                if len(mx)!=len(ma_feature_names):
                    counts['skipped_ma_schema_mismatch']+=1; continue
                x=[sf(v) for v in bx]+mx
                split=split_for(ax)
                out={'dataset_version':DATASET_VERSION,'symbol':cfg['symbol'],'anchor_time':ax,'decision_time':dt,'split':split,'target':TARGET,'x':x,'y':y,'book_feature_version':BOOK_FEATURE_VERSION,'book_label_version':BOOK_LABEL_VERSION,'ma_model_version':MA_MODEL_VERSION,'ma_prediction_ref':{'anchor_time':parse_dt(ma_doc.get('anchor_time')),'decision_time':ma_time,'lag_minutes':lag},'ma_context':ma_meta,'feature_meta':{'book_feature_count':len(bx),'ma_feature_count':len(mx),'feature_count':len(x),'feature_names_export':str(feature_names_export),'no_leak_ok': True}}
                target.replace_one({'anchor_time':ax}, out, upsert=True)
                counts['rows_built']+=1; counts['upserted_or_modified']+=1; split_counts[split]+=1; dir_counts[y['book_direction']]+=1
                if y.get('label_method') is not None: label_method_counts[y.get('label_method')]+=1
                ma_match_counts[f'lag_{int(lag)}m']+=1
                if counts['rows_built']%25000==0: print(f"[PROGRESS] rows_built={counts['rows_built']}", flush=True)
                if args.limit and counts['rows_built'] >= args.limit:
                    print(f"[LIMIT] rows_built limit reached: {args.limit}", flush=True)
                    break
            except Exception as exc:
                counts['row_errors']+=1
                if len(examples_errors)<20: examples_errors.append(str(exc))
        counts['ma_predictions_loaded']=len(ma_docs); counts['ma_predictions_skipped_no_time']=skipped_ma_no_time
        report['counts']=dict(counts)
        report['split_distribution']=dict(split_counts)
        report['direction_distribution']=dict(dir_counts)
        report['label_method_distribution']=dict(label_method_counts)
        # compact lag stats
        lag_values=[]
        try:
            for d in target.find({'dataset_version':DATASET_VERSION},{'_id':0,'ma_prediction_ref.lag_minutes':1}).limit(500000):
                lag_values.append(sf(get_nested(d,'ma_prediction_ref.lag_minutes')))
        except Exception: pass
        if lag_values:
            lag_values=sorted(lag_values); n=len(lag_values)
            report['ma_match_distribution']={'lag_min':round(lag_values[0],3),'lag_p50':round(lag_values[n//2],3),'lag_max':round(lag_values[-1],3),'lag_sample_rows':n}
        report['feature_schema']={'book_feature_count':len(feature_names)-len(ma_feature_names),'ma_feature_count':len(ma_feature_names),'feature_count':len(feature_names),'feature_names_export':str(feature_names_export),'book_feature_names_source':book_feature_names_source}
        if counts.get('row_errors',0)>0: report['errors'].extend(examples_errors)
        report['status']='success' if counts.get('row_errors',0)==0 and counts.get('rows_built',0)>0 else 'success_with_warnings'
    except Exception as exc:
        report['status']='failed'; report['errors'].append(str(exc)); print(f'[ERROR] {exc}', flush=True)
    finally:
        ended=utc_now(); report['end_time']=ended.isoformat(); report['duration_seconds']=round((ended-started).total_seconds(),3); save_report(report, rs)
    print('\n=== Final Summary ===', flush=True)
    print(f"Status          : {report['status']}", flush=True)
    print(f"Book Seen       : {report.get('counts',{}).get('book_rows_seen')}", flush=True)
    print(f"Rows Built      : {report.get('counts',{}).get('rows_built')}", flush=True)
    print(f"Feature Count   : {report.get('feature_schema',{}).get('feature_count')}", flush=True)
    print(f"TXT Report      : {report.get('report_txt_path')}", flush=True)
    print('[DONE]' if report['status']!='failed' else '[FAILED]', flush=True)
    if report['status']=='failed': sys.exit(1)

if __name__=='__main__': main()
