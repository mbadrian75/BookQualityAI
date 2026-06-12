# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Fast Predict MA Scenario Model M30 v1

Location:
    Book_Quality/06_models/ma_quality/01_predict_ma_scenario_m30_v1_fast.py

Purpose:
    Fast batch prediction for MA Scenario Model M30 v1.
    This replaces the slow row-by-row prediction script.

Input:
    06_models/ma_quality/ma_scenario_model_m30_v1_latest.joblib
    bq2_features_ma_quality_m30_v1

Output:
    bq2_predictions_ma_scenario_m30_v1

TEST:
    cd C:/Project/BookQuality/bq2/06_models/ma_quality
    python -u 01_predict_ma_scenario_m30_v1_fast.py --limit 1000 --reset --batch-size 5000

FULL:
    python -u 01_predict_ma_scenario_m30_v1_fast.py --reset --batch-size 5000
"""
from __future__ import annotations

import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.errors import BulkWriteError

SCRIPT_NAME = '01_predict_ma_scenario_m30_v1_fast.py'
MODEL_VERSION = 'ma_scenario_model_m30_v1'
FEATURE_VERSION = 'ma_quality_features_m30_v1'
LABEL_VERSION = 'ma_quality_label_m30_v1'
PREDICTION_VERSION = 'ma_scenario_prediction_m30_v1'
TARGETS = ['predicted_direction','m30_future_direction','h1_future_direction','h4_future_direction','m30_phase','h1_phase','h4_phase']
DEFAULT_CONFIG = {
    'mongo_uri': 'mongodb://localhost:27017',
    'database': 'market_data',
    'symbol': 'XAUUSD',
    'bq2_collections': {
        'features_ma_quality': 'bq2_features_ma_quality_m30_v1',
        'predictions_ma_scenario': 'bq2_predictions_ma_scenario_m30_v1',
    },
}

def utc_now(): return datetime.now(timezone.utc)
def stamp(dt): return dt.strftime('%Y%m%d_%H%M%S')
def project_root(): return Path(__file__).resolve().parents[2]
def reports_dir():
    p = project_root()/'06_models'/'reports'; p.mkdir(parents=True, exist_ok=True); return p
def models_dir(): return project_root()/'06_models'/'ma_quality'

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists(): return None
    try: return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f'[WARN] Could not read json {path}: {exc}', flush=True); return None

def deep_merge(a,b):
    out=dict(a)
    for k,v in b.items(): out[k]=deep_merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out

def load_config():
    cfg=DEFAULT_CONFIG
    fc=read_json(project_root()/'00_config'/'bq2_config.json')
    if fc: cfg=deep_merge(cfg,fc)
    cfg.setdefault('bq2_collections', {})
    cfg['bq2_collections']['features_ma_quality']='bq2_features_ma_quality_m30_v1'
    cfg['bq2_collections']['predictions_ma_scenario']='bq2_predictions_ma_scenario_m30_v1'
    return cfg

def connect(cfg):
    c=MongoClient(cfg['mongo_uri'], serverSelectionTimeoutMS=5000); c.admin.command('ping'); return c[cfg['database']]

def deps():
    try:
        import joblib, numpy as np
    except Exception as exc:
        raise RuntimeError('Missing dependencies. Install: pip install joblib numpy scikit-learn') from exc
    return joblib, np

def parse_dt(v):
    if isinstance(v, datetime): return v.replace(tzinfo=None)
    if isinstance(v, str): return datetime.fromisoformat(v.replace('Z','+00:00')).replace(tzinfo=None)
    raise ValueError(f'Unsupported datetime: {v!r}')

def parse_dt_optional(v):
    if not v: return None
    return datetime.fromisoformat(v.replace('Z','+00:00')).replace(tzinfo=None)

def sf(v, default=0.0):
    try:
        x=float(v)
        return default if math.isnan(x) or math.isinf(x) else x
    except Exception: return default

def load_bundle(args, joblib):
    path=Path(args.model_path) if args.model_path else models_dir()/f'{MODEL_VERSION}_latest.joblib'
    if not path.exists(): raise RuntimeError(f'Model file not found: {path}')
    b=joblib.load(path)
    if b.get('model_version') != MODEL_VERSION: raise RuntimeError(f"Unexpected model_version: {b.get('model_version')}")
    return b

def execute_bulk(db, coll, ops, report):
    if not ops: return
    try:
        r=db[coll].bulk_write(ops, ordered=False)
        report['counts']['upserted_or_modified'] += r.upserted_count + r.modified_count
        report['counts']['inserted'] += r.upserted_count
        report['counts']['modified'] += r.modified_count
        report['counts']['matched'] += r.matched_count
    except BulkWriteError as exc:
        report['status']='failed'; report['errors'].append(str(exc.details)[:20000]); raise

def make_update(doc):
    created_at=doc.pop('created_at', utc_now())
    return UpdateOne({'symbol':doc['symbol'], 'anchor_time':doc['anchor_time']}, {'$set':doc, '$setOnInsert':{'created_at':created_at}}, upsert=True)

def update_stats(report, target, label, conf):
    d=report['prediction_distribution'][target]; d[label]=d.get(label,0)+1
    a=report['confidence_accumulator'][target]; a['sum']+=conf; a['count']+=1
    if conf>=0.70: a['p70']+=1
    if conf>=0.80: a['p80']+=1

def predict_batch(rows, bundle, names, np, report):
    X=np.asarray([[sf(row['features'].get(n,0.0)) for n in names] for row in rows], dtype=np.float32)
    all_pred, all_prob, all_conf = {}, {}, {}
    for t in TARGETS:
        model=bundle['models'][t]; enc=bundle['encoders'][t]
        labels=[str(x) for x in enc.inverse_transform(model.predict(X))]
        probs=[]; confs=[]
        if hasattr(model,'predict_proba'):
            pm=model.predict_proba(X); classes=[str(c) for c in enc.classes_]
            for p in pm:
                pd={c:float(v) for c,v in zip(classes,p)}; probs.append(pd); confs.append(float(max(p)))
        else:
            probs=[{x:1.0} for x in labels]; confs=[1.0]*len(labels)
        all_pred[t]=labels; all_prob[t]=probs; all_conf[t]=confs
    docs=[]; now=utc_now()
    for i,row in enumerate(rows):
        tp={t:all_pred[t][i] for t in TARGETS}
        tb={t:all_prob[t][i] for t in TARGETS}
        for t in TARGETS: update_stats(report,t,tp[t],all_conf[t][i])
        dp=tb['predicted_direction']; p_buy=float(dp.get('BUY',0.0)); p_sell=float(dp.get('SELL',0.0)); p_range=float(dp.get('RANGE',0.0))
        docs.append({
            'symbol':row['symbol'], 'anchor_time':row['anchor_time'], 'decision_time':row['decision_time'],
            'entry_time':row['entry_time'], 'entry_price':row['entry_price'],
            'prediction_version':PREDICTION_VERSION, 'model_version':MODEL_VERSION,
            'feature_version':FEATURE_VERSION, 'label_version':LABEL_VERSION,
            'anchor_timeframe':'M30', 'input_timeframes':['M30','H1','H4'],
            'predicted_direction':tp['predicted_direction'], 'direction_confidence':max(p_buy,p_sell,p_range),
            'p_buy':p_buy, 'p_sell':p_sell, 'p_range':p_range,
            'm30_future_direction':tp['m30_future_direction'], 'h1_future_direction':tp['h1_future_direction'], 'h4_future_direction':tp['h4_future_direction'],
            'm30_phase':tp['m30_phase'], 'h1_phase':tp['h1_phase'], 'h4_phase':tp['h4_phase'],
            'target_predictions':tp, 'target_probabilities':tb, 'feature_meta':row.get('feature_meta',{}),
            'created_at':now, 'updated_at':now,
        })
    return docs

def save_reports(report, rs):
    jp=reports_dir()/f'ma_scenario_predict_m30_v1_fast_report_{rs}.json'
    tp=reports_dir()/f'ma_scenario_predict_m30_v1_fast_report_{rs}.txt'
    report['report_json_path']=str(jp); report['report_txt_path']=str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    lines=['Book & Quality v2 - Fast Predict MA Scenario Model M30 v1 Report','='*74]
    for k in ['run_id','status','database','symbol','start_time','end_time','duration_seconds']:
        lines.append(f'{k:<34}: {report.get(k)}')
    for section,key in [('Collections','collections'),('Model','model'),('Counts','counts')]:
        lines += ['', section, '-'*74]
        for k,v in report[key].items(): lines.append(f'{k:<34}: {v}')
    lines += ['', 'Prediction Distribution', '-'*74]
    for t,d in report['prediction_distribution'].items():
        lines.append(f'{t}:')
        for k,v in d.items(): lines.append(f'  {str(k):<31}: {v}')
    lines += ['', 'Confidence Summary', '-'*74]
    for t,d in report['confidence_summary'].items():
        lines.append(f'{t}:')
        for k,v in d.items(): lines.append(f'  {k:<31}: {v}')
    if report['errors']:
        lines += ['', 'Errors', '-'*74] + [f'- {e}' for e in report['errors'][:50]]
    tp.write_text('\n'.join(lines), encoding='utf-8')

def parse_args():
    p=argparse.ArgumentParser(description='Fast Predict MA Scenario M30 v1.')
    p.add_argument('--limit', type=int, default=None); p.add_argument('--reset', action='store_true')
    p.add_argument('--start', type=str, default=None); p.add_argument('--end', type=str, default=None)
    p.add_argument('--skip-existing', action='store_true'); p.add_argument('--model-path', type=str, default=None)
    p.add_argument('--batch-size', type=int, default=5000)
    return p.parse_args()

def main():
    args=parse_args(); started=utc_now(); rs=stamp(started)
    report={'script_name':SCRIPT_NAME,'run_id':f'fast_predict_ma_scenario_m30_v1_{rs}','status':'running','database':None,'symbol':None,
        'start_time':started.isoformat(),'end_time':None,'duration_seconds':None,'collections':{},'model':{},
        'counts':{'features_seen':0,'predictions_built':0,'upserted_or_modified':0,'inserted':0,'matched':0,'modified':0,'skipped_existing':0,'skipped_missing_features':0,'skipped_feature_schema':0,'row_errors':0},
        'prediction_distribution':{t:{} for t in TARGETS}, 'confidence_accumulator':{t:{'sum':0.0,'count':0,'p70':0,'p80':0} for t in TARGETS}, 'confidence_summary':{}, 'errors':[], 'args':vars(args)}
    print('=== Book & Quality v2 - Fast Predict MA Scenario Model M30 v1 ===', flush=True)
    try:
        joblib,np=deps(); cfg=load_config(); db=connect(cfg); bundle=load_bundle(args, joblib)
        source=cfg['bq2_collections']['features_ma_quality']; target=cfg['bq2_collections']['predictions_ma_scenario']
        report['database']=cfg['database']; report['symbol']=cfg['symbol']; report['collections']={'features_source':source,'target':target}
        report['model']={k:bundle.get(k) for k in ['model_version','dataset_version','feature_version','label_version','trained_at','feature_count','targets']}
        names=bundle.get('feature_names') or []
        if not names: raise RuntimeError('Model bundle does not contain feature_names. Re-train model after dataset export exists.')
        for t in TARGETS:
            if t not in bundle.get('models',{}) or t not in bundle.get('encoders',{}): raise RuntimeError(f'Missing model or encoder for target={t}')
        if args.reset:
            deleted=db[target].delete_many({}); report['reset_deleted_count']=deleted.deleted_count; print(f'[RESET] Deleted {deleted.deleted_count:,} docs from {target}', flush=True)
        db[target].create_index([('symbol',ASCENDING),('anchor_time',ASCENDING)], unique=True, name='uq_symbol_anchor_time')
        db[target].create_index([('prediction_version',ASCENDING),('anchor_time',ASCENDING)], name='ix_prediction_version_anchor_time')
        db[target].create_index([('predicted_direction',ASCENDING),('direction_confidence',ASCENDING)], name='ix_direction_confidence')
        q={'feature_version':FEATURE_VERSION,'label_version':LABEL_VERSION}
        if args.start or args.end:
            tf={}
            if args.start: tf['$gte']=parse_dt_optional(args.start)
            if args.end: tf['$lte']=parse_dt_optional(args.end)
            q['anchor_time']=tf
        cur=db[source].find(q, projection={'_id':0,'symbol':1,'anchor_time':1,'decision_time':1,'entry_time':1,'entry_price':1,'features':1,'feature_count':1,'feature_meta':1}).sort('anchor_time',ASCENDING).batch_size(args.batch_size)
        batch=[]
        def flush():
            nonlocal batch
            if not batch: return
            docs=predict_batch(batch,bundle,names,np,report); ops=[make_update(d) for d in docs]; execute_bulk(db,target,ops,report)
            report['counts']['predictions_built']+=len(docs)
            print(f"[PROGRESS] seen={report['counts']['features_seen']:,} built={report['counts']['predictions_built']:,} written≈{report['counts']['upserted_or_modified']:,}", flush=True)
            batch=[]
        for src in cur:
            try:
                if args.limit is not None and report['counts']['features_seen'] >= args.limit: break
                symbol=src.get('symbol',cfg['symbol']); anchor=parse_dt(src['anchor_time']); decision=parse_dt(src['decision_time'])
                report['counts']['features_seen']+=1
                if args.skip_existing and db[target].find_one({'symbol':symbol,'anchor_time':anchor}, projection={'_id':1}):
                    report['counts']['skipped_existing']+=1; continue
                features=src.get('features')
                if not isinstance(features,dict) or not features: report['counts']['skipped_missing_features']+=1; continue
                if any(n not in features for n in names): report['counts']['skipped_feature_schema']+=1; continue
                batch.append({'symbol':symbol,'anchor_time':anchor,'decision_time':decision,'entry_time':parse_dt(src.get('entry_time',decision)),'entry_price':sf(src.get('entry_price',0.0)),'features':features,'feature_meta':src.get('feature_meta',{})})
                if len(batch)>=args.batch_size: flush()
            except Exception as exc:
                report['counts']['row_errors']+=1
                if len(report['errors'])<100: report['errors'].append(f"anchor={src.get('anchor_time')} | {exc}")
        flush()
        for t,a in report['confidence_accumulator'].items():
            c=a['count']; report['confidence_summary'][t]={'avg_confidence':round(a['sum']/c,6) if c else None,'p70_rate':round(a['p70']/c,6) if c else None,'p80_rate':round(a['p80']/c,6) if c else None}
        report['final_target_count']=db[target].estimated_document_count()
        report['status']='success' if report['counts']['row_errors']==0 and not report['errors'] else 'success_with_row_errors'
    except Exception as exc:
        report['status']='failed'
        if not report['errors']: report['errors'].append(str(exc))
        print(f'[ERROR] {exc}', flush=True)
    finally:
        ended=utc_now(); report['end_time']=ended.isoformat(); report['duration_seconds']=round((ended-started).total_seconds(),3); save_reports(report,rs)
    print('\n=== Final Summary ===', flush=True)
    print(f"Status             : {report['status']}", flush=True)
    print(f"Features Seen      : {report['counts']['features_seen']:,}", flush=True)
    print(f"Predictions Built  : {report['counts']['predictions_built']:,}", flush=True)
    print(f"Upserted/Modified  : {report['counts']['upserted_or_modified']:,}", flush=True)
    print(f"Row Errors         : {report['counts']['row_errors']:,}", flush=True)
    print(f"TXT Report         : {report.get('report_txt_path')}", flush=True)
    print('[DONE]' if report['status']!='failed' else '[FAILED]', flush=True)
    if report['status']=='failed': sys.exit(1)

if __name__ == '__main__': main()
