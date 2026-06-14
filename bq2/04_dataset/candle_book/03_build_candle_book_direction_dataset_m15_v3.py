# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Build Candle Book Direction Dataset M15 v3

Input : bq2_features_candle_book_direction_m15_v3
Output: bq2_dataset_candle_book_direction_m15_v3
Target: book_direction = BUY / SELL / UNCLEAR

TEST:
    cd C:/Project/BookQuality/bq2/04_dataset/candle_book
    python -u 03_build_candle_book_direction_dataset_m15_v3.py --limit 1000 --reset

FULL:
    python -u 03_build_candle_book_direction_dataset_m15_v3.py --reset
"""
from __future__ import annotations

import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError

SCRIPT_NAME='03_build_candle_book_direction_dataset_m15_v3.py'
DATASET_VERSION='candle_book_direction_dataset_m15_v3'
FEATURE_VERSION='candle_book_direction_features_m15_v3'
LABEL_VERSION='candle_book_direction_label_m15_v3'
BATCH_SIZE=1000
DEFAULT_CONFIG={
    'mongo_uri':'mongodb://localhost:27017',
    'database':'market_data',
    'symbol':'XAUUSD',
    'bq2_collections':{
        'features_candle_book_direction':'bq2_features_candle_book_direction_m15_v3',
        'dataset_candle_book_direction':'bq2_dataset_candle_book_direction_m15_v3',
    },
    'candle_book_direction_dataset_m15_v3':{
        'train_end':'2023-01-01 00:00:00',
        'valid_end':'2024-01-01 00:00:00',
        'test_end':None,
        'require_no_leak_ok':True,
        'keep_feature_dict':False,
    },
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
        print(f'[WARN] Could not read config {path}: {exc}', flush=True); return None

def deep_merge(base:Dict[str,Any], incoming:Dict[str,Any]):
    out=dict(base)
    for k,v in incoming.items():
        out[k]=deep_merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out

def load_config():
    cfg=DEFAULT_CONFIG
    file_cfg=read_json(project_root()/'00_config'/'bq2_config.json')
    if file_cfg: cfg=deep_merge(cfg,file_cfg)
    cfg.setdefault('bq2_collections',{})
    cfg['bq2_collections']['features_candle_book_direction']=cfg['bq2_collections'].get('features_candle_book_direction','bq2_features_candle_book_direction_m15_v3')
    cfg['bq2_collections']['dataset_candle_book_direction']=cfg['bq2_collections'].get('dataset_candle_book_direction','bq2_dataset_candle_book_direction_m15_v3')
    return cfg

def connect(cfg):
    client=MongoClient(cfg['mongo_uri'], serverSelectionTimeoutMS=5000); client.admin.command('ping'); return client[cfg['database']]
def parse_dt(v):
    if isinstance(v, datetime): return v.replace(tzinfo=None)
    if isinstance(v, str): return datetime.fromisoformat(v.replace('Z','+00:00')).replace(tzinfo=None)
    raise ValueError(f'Unsupported datetime: {v!r}')
def parse_dt_optional(v): return None if not v else datetime.fromisoformat(v.replace('Z','+00:00')).replace(tzinfo=None)
def sf(v, default=0.0):
    try:
        x=float(v); return default if math.isnan(x) or math.isinf(x) else x
    except Exception: return default

def split_for(anchor_time, train_end, valid_end, test_end):
    if anchor_time < train_end: return 'train'
    if anchor_time < valid_end: return 'valid'
    if test_end is None or anchor_time < test_end: return 'test'
    return 'ignored'

def make_update_op(symbol, doc):
    created_at=doc.pop('created_at', utc_now())
    return UpdateOne({'symbol':symbol,'anchor_time':doc['anchor_time']},{'$set':doc,'$setOnInsert':{'created_at':created_at}},upsert=True)

def execute_bulk(db, coll, ops, report):
    if not ops: return
    try:
        res=db[coll].bulk_write(ops, ordered=False)
        report['counts']['upserted_or_modified']+=res.upserted_count+res.modified_count
        report['counts']['inserted']+=res.upserted_count
        report['counts']['modified']+=res.modified_count
        report['counts']['matched']+=res.matched_count
    except BulkWriteError as exc:
        report['status']='failed'; report['errors'].append(str(exc.details)[:20000]); raise

def save_reports(report, rs):
    jp=reports_dir()/f'candle_book_direction_dataset_m15_v3_report_{rs}.json'
    tp=reports_dir()/f'candle_book_direction_dataset_m15_v3_report_{rs}.txt'
    report['report_json_path']=str(jp); report['report_txt_path']=str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    lines=['Book & Quality v2 - Candle Book Direction Dataset M15 v3 Report','='*74]
    for k in ['run_id','status','database','symbol','start_time','end_time','duration_seconds']:
        lines.append(f'{k:<34}: {report.get(k)}')
    for section,key in [('Collections','collections'),('Config','dataset_config'),('Counts','counts'),('Feature Schema','feature_schema'),('Split Distribution','split_distribution'),('Direction Distribution','direction_distribution'),('Label Method Distribution','label_method_distribution')]:
        lines+=['',section,'-'*74]
        for k,v in report[key].items(): lines.append(f'{k:<34}: {v}')
    if report['errors']:
        lines+=['','Errors','-'*74]+[f'- {e}' for e in report['errors'][:50]]
    tp.write_text('\n'.join(lines), encoding='utf-8')

def parse_args():
    p=argparse.ArgumentParser(description='Build Candle Book Direction Dataset M15 v3.')
    p.add_argument('--limit', type=int, default=None); p.add_argument('--reset', action='store_true')
    p.add_argument('--start', type=str, default=None); p.add_argument('--end', type=str, default=None)
    p.add_argument('--skip-existing', action='store_true')
    return p.parse_args()

def main():
    args=parse_args(); started=utc_now(); rs=stamp(started)
    cfg=load_config(); bq2=cfg['bq2_collections']; dcfg=cfg.get('candle_book_direction_dataset_m15_v3', DEFAULT_CONFIG['candle_book_direction_dataset_m15_v3'])
    source_coll=bq2.get('features_candle_book_direction','bq2_features_candle_book_direction_m15_v3')
    target_coll=bq2.get('dataset_candle_book_direction','bq2_dataset_candle_book_direction_m15_v3')
    train_end=parse_dt_optional(dcfg.get('train_end')) or datetime(2023,1,1)
    valid_end=parse_dt_optional(dcfg.get('valid_end')) or datetime(2024,1,1)
    test_end=parse_dt_optional(dcfg.get('test_end'))
    require_no_leak_ok=bool(dcfg.get('require_no_leak_ok', True)); keep_feature_dict=bool(dcfg.get('keep_feature_dict', False))
    report={'script_name':SCRIPT_NAME,'run_id':f'candle_book_direction_dataset_m15_v3_{rs}','status':'running','database':cfg['database'],'symbol':cfg['symbol'],'start_time':started.isoformat(),'end_time':None,'duration_seconds':None,'collections':{'features_source':source_coll,'target':target_coll},'dataset_config':{'dataset_version':DATASET_VERSION,'feature_version':FEATURE_VERSION,'label_version':LABEL_VERSION,'anchor_timeframe':'M15','input_timeframes':['M1','M5','M15'],'target':'book_direction','train_end':str(train_end),'valid_end':str(valid_end),'test_end':str(test_end) if test_end else None,'require_no_leak_ok':require_no_leak_ok,'keep_feature_dict':keep_feature_dict},'counts':{'features_seen':0,'rows_built':0,'upserted_or_modified':0,'inserted':0,'matched':0,'modified':0,'skipped_existing':0,'skipped_missing_features':0,'skipped_missing_y':0,'skipped_missing_book_direction':0,'skipped_no_leak_not_ok':0,'skipped_schema_mismatch':0,'skipped_ignored_split':0,'row_errors':0},'feature_schema':{},'split_distribution':{},'direction_distribution':{},'label_method_distribution':{},'errors':[],'args':vars(args)}
    print('=== Book & Quality v2 - Build Candle Book Direction Dataset M15 v3 ===', flush=True)
    print(f"Database : {cfg['database']}", flush=True); print(f"Symbol   : {cfg['symbol']}", flush=True); print(f'Source   : {source_coll}', flush=True); print(f'Target   : {target_coll}', flush=True)
    try:
        db=connect(cfg)
        if args.reset:
            deleted=db[target_coll].delete_many({}); report['reset_deleted_count']=deleted.deleted_count; print(f'[RESET] Deleted {deleted.deleted_count:,} docs from {target_coll}', flush=True)
        db[target_coll].create_index([('symbol',ASCENDING),('anchor_time',ASCENDING)], unique=True, name='uq_symbol_anchor_time')
        db[target_coll].create_index([('dataset_version',ASCENDING),('anchor_time',ASCENDING)], name='ix_dataset_version_anchor_time')
        db[target_coll].create_index([('split',ASCENDING),('anchor_time',ASCENDING)], name='ix_split_anchor_time')
        db[target_coll].create_index([('y.book_direction',ASCENDING),('split',ASCENDING)], name='ix_book_direction_split')
        db[target_coll].create_index([('feature_count',ASCENDING)], name='ix_feature_count')
        query={'feature_version':FEATURE_VERSION,'label_version':LABEL_VERSION}
        if args.start or args.end:
            tf={}
            if args.start: tf['$gte']=parse_dt_optional(args.start)
            if args.end: tf['$lte']=parse_dt_optional(args.end)
            query['anchor_time']=tf
        first=db[source_coll].find_one(query, projection={'features':1,'feature_count':1,'_id':0}, sort=[('anchor_time',ASCENDING)])
        if not first or not isinstance(first.get('features'), dict): raise RuntimeError('No valid source feature document found.')
        feature_names=sorted(first['features'].keys()); expected_keys=set(feature_names); feature_count=len(feature_names)
        export_path=exports_dir()/f'candle_book_direction_m15_v3_feature_names_{rs}.json'
        export_path.write_text(json.dumps({'dataset_version':DATASET_VERSION,'feature_version':FEATURE_VERSION,'label_version':LABEL_VERSION,'feature_count':feature_count,'feature_names':feature_names,'created_at':utc_now().isoformat()}, ensure_ascii=False, indent=2), encoding='utf-8')
        report['feature_schema']={'feature_count':feature_count,'first_doc_feature_count_field':first.get('feature_count'),'feature_names_export':str(export_path)}
        cursor=db[source_coll].find(query, projection={'_id':0,'symbol':1,'anchor_time':1,'decision_time':1,'entry_time':1,'entry_price':1,'features':1,'feature_count':1,'feature_meta':1,'y':1}).sort('anchor_time',ASCENDING).batch_size(10000)
        ops=[]
        for src in cursor:
            try:
                if args.limit is not None and report['counts']['features_seen']>=args.limit: break
                report['counts']['features_seen']+=1
                src_symbol=src.get('symbol',cfg['symbol']); anchor_time=parse_dt(src['anchor_time']); decision_time=parse_dt(src['decision_time']); split=split_for(anchor_time,train_end,valid_end,test_end)
                if split=='ignored': report['counts']['skipped_ignored_split']+=1; continue
                if args.skip_existing and db[target_coll].find_one({'symbol':src_symbol,'anchor_time':anchor_time}, projection={'_id':1}): report['counts']['skipped_existing']+=1; continue
                features=src.get('features'); y_full=src.get('y'); meta=src.get('feature_meta',{})
                if not isinstance(features,dict) or not features: report['counts']['skipped_missing_features']+=1; continue
                if not isinstance(y_full,dict) or not y_full: report['counts']['skipped_missing_y']+=1; continue
                book_direction=y_full.get('book_direction')
                if book_direction is None: report['counts']['skipped_missing_book_direction']+=1; continue
                if require_no_leak_ok and meta.get('no_leak_ok') is not True: report['counts']['skipped_no_leak_not_ok']+=1; continue
                if set(features.keys()) != expected_keys: report['counts']['skipped_schema_mismatch']+=1; continue
                x=[sf(features[name]) for name in feature_names]
                y={'book_direction':book_direction,'label_method':y_full.get('label_method'),'soft_buy_score':sf(y_full.get('soft_buy_score',0.0)),'soft_sell_score':sf(y_full.get('soft_sell_score',0.0)),'soft_unclear_score':sf(y_full.get('soft_unclear_score',0.0)),'buy_pressure':sf(y_full.get('buy_pressure',0.0)),'sell_pressure':sf(y_full.get('sell_pressure',0.0)),'pressure_gap':sf(y_full.get('pressure_gap',0.0)),'abs_pressure_gap':sf(y_full.get('abs_pressure_gap',0.0)),'gap_ratio':sf(y_full.get('gap_ratio',0.0))}
                doc={'symbol':src_symbol,'anchor_time':anchor_time,'decision_time':decision_time,'entry_time':parse_dt(src.get('entry_time',decision_time)),'entry_price':sf(src.get('entry_price',0.0)),'dataset_version':DATASET_VERSION,'feature_version':FEATURE_VERSION,'label_version':LABEL_VERSION,'model_type':'candle_book_direction','anchor_timeframe':'M15','input_timeframes':['M1','M5','M15'],'split':split,'x':x,'feature_count':feature_count,'y':y,'feature_meta':{'no_leak_ok':meta.get('no_leak_ok'),'max_feature_time':meta.get('max_feature_time'),'tf_last_closed_time':meta.get('tf_last_closed_time')},'created_at':utc_now(),'updated_at':utc_now()}
                if keep_feature_dict: doc['features']=features
                ops.append(make_update_op(src_symbol,doc)); report['counts']['rows_built']+=1
                report['split_distribution'][split]=report['split_distribution'].get(split,0)+1
                report['direction_distribution'][book_direction]=report['direction_distribution'].get(book_direction,0)+1
                method=y.get('label_method')
                if method: report['label_method_distribution'][method]=report['label_method_distribution'].get(method,0)+1
                if len(ops)>=BATCH_SIZE:
                    execute_bulk(db,target_coll,ops,report); ops=[]
                    print(f"[PROGRESS] seen={report['counts']['features_seen']:,} built={report['counts']['rows_built']:,} written≈{report['counts']['upserted_or_modified']:,}", flush=True)
            except Exception as row_exc:
                report['counts']['row_errors']+=1
                if len(report['errors'])<100: report['errors'].append(f"anchor={src.get('anchor_time')} | {row_exc}")
        if ops: execute_bulk(db,target_coll,ops,report)
        report['final_target_count']=db[target_coll].estimated_document_count()
        report['status']='success' if report['counts']['row_errors']==0 and not report['errors'] else 'success_with_row_errors'
    except Exception as exc:
        report['status']='failed'
        if not report['errors']: report['errors'].append(str(exc))
        print(f'[ERROR] {exc}', flush=True)
    finally:
        ended=utc_now(); report['end_time']=ended.isoformat(); report['duration_seconds']=round((ended-started).total_seconds(),3); save_reports(report,rs)
    print('\n=== Final Summary ===', flush=True)
    print(f"Status            : {report['status']}", flush=True); print(f"Features Seen     : {report['counts']['features_seen']:,}", flush=True); print(f"Rows Built        : {report['counts']['rows_built']:,}", flush=True); print(f"Upserted/Modified : {report['counts']['upserted_or_modified']:,}", flush=True); print(f"Row Errors        : {report['counts']['row_errors']:,}", flush=True); print(f"Feature Count     : {report['feature_schema'].get('feature_count')}", flush=True); print(f"Split Distribution: {report['split_distribution']}", flush=True); print(f"Direction Distribution: {report['direction_distribution']}", flush=True); print(f"TXT Report        : {report.get('report_txt_path')}", flush=True); print('[DONE]' if report['status']!='failed' else '[FAILED]', flush=True)
    if report['status']=='failed': sys.exit(1)

if __name__=='__main__': main()
