# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Train Entry Context Model M15 v1

Input : bq2_dataset_entry_context_m15_v1
Output: 06_models/entry_context/entry_context_model_m15_v1_latest.joblib
Target: book_direction = BUY / SELL / UNCLEAR

TEST:
    cd C:/Project/BookQuality/bq2/05_training/entry_context
    python -u 01_train_entry_context_m15_v1.py --limit-per-split 5000 --max-iter 50

FULL:
    python -u 01_train_entry_context_m15_v1.py --max-iter 120 --learning-rate 0.03 --l2-regularization 1.0 --max-leaf-nodes 15
"""
from __future__ import annotations

import argparse, json, math, sys, shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING

MODEL_VERSION='entry_context_model_m15_v1'
DATASET_VERSION='entry_context_dataset_m15_v1'
BOOK_FEATURE_VERSION='candle_book_direction_features_m15_v3'
BOOK_LABEL_VERSION='candle_book_direction_label_m15_v3'
MA_MODEL_VERSION='ma_scenario_model_m30_v1'
TARGET='book_direction'
SOURCE_COLLECTION='bq2_dataset_entry_context_m15_v1'

DEFAULT_CONFIG={'mongo_uri':'mongodb://localhost:27017','database':'market_data','symbol':'XAUUSD'}

def utc_now(): return datetime.now(timezone.utc)
def stamp(dt): return dt.strftime('%Y%m%d_%H%M%S')
def project_root(): return Path(__file__).resolve().parents[2]
def reports_dir():
    p=project_root()/'05_training'/'reports'; p.mkdir(parents=True, exist_ok=True); return p
def models_dir():
    p=project_root()/'06_models'/'entry_context'; p.mkdir(parents=True, exist_ok=True); return p
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

def check_dependencies():
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
    except Exception as exc:
        raise RuntimeError('Missing dependencies. Install: pip install numpy scikit-learn joblib') from exc
    return locals()

def latest_feature_names()->Tuple[List[str], Optional[str]]:
    files=sorted(exports_dir().glob('entry_context_m15_v1_feature_names_*.json'))
    if not files: return [], None
    path=files[-1]
    data=read_json(path) or {}
    return list(data.get('feature_names', [])), str(path)

def load_split(db, split:str, limit:Optional[int]):
    query={'dataset_version':DATASET_VERSION,'split':split}
    cursor=db[SOURCE_COLLECTION].find(query, projection={'_id':0,'x':1,'y.book_direction':1,'anchor_time':1}).sort('anchor_time', ASCENDING)
    if limit: cursor=cursor.limit(limit)
    X=[]; y=[]
    for doc in cursor:
        x=doc.get('x')
        direction=((doc.get('y') or {}).get('book_direction'))
        if isinstance(x,list) and direction in {'BUY','SELL','UNCLEAR'}:
            X.append([sf(v) for v in x]); y.append(direction)
    return X,y

def class_weights(y_labels:List[str])->Dict[str,float]:
    c=Counter(y_labels); n=len(y_labels); k=max(len(c),1)
    return {cls:n/(k*cnt) for cls,cnt in c.items() if cnt>0}

def make_sample_weights(y_labels:List[str]):
    w=class_weights(y_labels)
    return [w.get(v,1.0) for v in y_labels]

def dist(labels): return dict(Counter([str(x) for x in labels]))

def safe_log_loss(y_true_enc, proba, labels):
    try:
        from sklearn.metrics import log_loss
        return float(log_loss(y_true_enc, proba, labels=labels))
    except Exception: return None

def analyze_probabilities(y_true_labels, pred_labels, proba, classes):
    import numpy as np
    idx={cls:i for i,cls in enumerate(classes)}
    p_buy=proba[:,idx['BUY']] if 'BUY' in idx else np.zeros(len(proba))
    p_sell=proba[:,idx['SELL']] if 'SELL' in idx else np.zeros(len(proba))
    conf=np.max(proba, axis=1)
    gap=np.abs(p_buy-p_sell)
    true=np.asarray(y_true_labels); pred=np.asarray(pred_labels)
    out={'avg_confidence':round(float(np.mean(conf)),6) if len(conf) else None,'p60_rate':round(float(np.mean(conf>=0.60)),6) if len(conf) else None,'p70_rate':round(float(np.mean(conf>=0.70)),6) if len(conf) else None,'p80_rate':round(float(np.mean(conf>=0.80)),6) if len(conf) else None,'avg_buy_sell_gap':round(float(np.mean(gap)),6) if len(gap) else None,'gap10_rate':round(float(np.mean(gap>=0.10)),6) if len(gap) else None,'gap20_rate':round(float(np.mean(gap>=0.20)),6) if len(gap) else None,'gap30_rate':round(float(np.mean(gap>=0.30)),6) if len(gap) else None,'gap_threshold_analysis':{}}
    for th in [0.05,0.10,0.15,0.20,0.25,0.30,0.40]:
        mask=(gap>=th) & (pred!='UNCLEAR')
        rows=int(mask.sum())
        out['gap_threshold_analysis'][f'gap>={th:.2f}']={'coverage':round(float(rows/len(gap)),6) if len(gap) else None,'rows':rows,'actual_direction_match_rate':round(float(np.mean(pred[mask]==true[mask])),6) if rows else None,'pred_buy_rows':int(np.sum(mask & (pred=='BUY'))),'pred_sell_rows':int(np.sum(mask & (pred=='SELL')))}
    return out

def metrics_for_split(model, encoder, X, y, deps):
    np=deps['np']; accuracy_score=deps['accuracy_score']; f1_score=deps['f1_score']; confusion_matrix=deps['confusion_matrix']
    Xn=np.asarray(X,dtype=np.float32); y_enc=encoder.transform(y)
    pred_enc=model.predict(Xn); pred_labels=list(encoder.inverse_transform(pred_enc)); proba=model.predict_proba(Xn); classes=list(encoder.classes_)
    out={'rows':len(y),'accuracy':round(float(accuracy_score(y_enc,pred_enc)),6),'macro_f1':round(float(f1_score(y_enc,pred_enc,average='macro')),6),'weighted_f1':round(float(f1_score(y_enc,pred_enc,average='weighted')),6),'log_loss':safe_log_loss(y_enc,proba,labels=list(range(len(classes)))),'label_distribution':dist(y),'prediction_distribution':dist(pred_labels),'classes':classes,'confusion_matrix_labels':classes,'confusion_matrix':confusion_matrix(y_enc,pred_enc,labels=list(range(len(classes)))).tolist()}
    out.update(analyze_probabilities(y,pred_labels,proba,classes))
    return out

def save_report(report, rs):
    jp=reports_dir()/f'entry_context_train_m15_v1_report_{rs}.json'
    tp=reports_dir()/f'entry_context_train_m15_v1_report_{rs}.txt'
    report['report_json_path']=str(jp); report['report_txt_path']=str(tp)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    lines=['Book & Quality v2 - Train Entry Context Model M15 v1 Report','='*74]
    for k in ['run_id','status','database','symbol','start_time','end_time','duration_seconds']:
        lines.append(f'{k:<34}: {report.get(k)}')
    for title,key in [('Collections','collections'),('Training Config','training_config'),('Counts','counts'),('Model Output','model_output')]:
        lines+=['',title,'-'*74]
        for kk,vv in report.get(key,{}).items(): lines.append(f'{kk:<34}: {vv}')
    lines+=['','Metrics','-'*74]
    for split,m in report.get('metrics',{}).items():
        lines.append(f'{split}:')
        for k,v in m.items():
            if isinstance(v,dict):
                lines.append(f'  {k}:')
                for kk,vv in v.items(): lines.append(f'    {kk:<27}: {vv}')
            else:
                lines.append(f'  {k:<31}: {v}')
    if report.get('errors'):
        lines+=['','Errors','-'*74]+[f'- {e}' for e in report['errors'][:50]]
    tp.write_text('\n'.join(lines), encoding='utf-8')

def parse_args():
    p=argparse.ArgumentParser(description='Train Entry Context M15 v1.')
    p.add_argument('--limit-per-split', type=int, default=None)
    p.add_argument('--max-iter', type=int, default=120)
    p.add_argument('--learning-rate', type=float, default=0.03)
    p.add_argument('--l2-regularization', type=float, default=1.0)
    p.add_argument('--max-leaf-nodes', type=int, default=15)
    p.add_argument('--no-sample-weight', action='store_true')
    return p.parse_args()

def main():
    args=parse_args(); started=utc_now(); rs=stamp(started)
    report={'run_id':f'entry_context_train_m15_v1_{rs}','status':'running','database':None,'symbol':None,'start_time':started.isoformat(),'end_time':None,'duration_seconds':None,'collections':{},'training_config':{},'counts':{},'model_output':{},'metrics':{},'errors':[],'args':vars(args)}
    print('=== Book & Quality v2 - Train Entry Context Model M15 v1 ===', flush=True)
    try:
        deps=check_dependencies(); np=deps['np']; HGB=deps['HistGradientBoostingClassifier']; LabelEncoder=deps['LabelEncoder']; joblib=deps['joblib']
        cfg=load_config(); db=connect(cfg); report['database']=cfg['database']; report['symbol']=cfg['symbol']; report['collections']={'dataset_source':SOURCE_COLLECTION}
        report['training_config']={'model_version':MODEL_VERSION,'dataset_version':DATASET_VERSION,'book_feature_version':BOOK_FEATURE_VERSION,'book_label_version':BOOK_LABEL_VERSION,'ma_model_version':MA_MODEL_VERSION,'target':TARGET,'classifier':'HistGradientBoostingClassifier','max_iter':args.max_iter,'learning_rate':args.learning_rate,'l2_regularization':args.l2_regularization,'max_leaf_nodes':args.max_leaf_nodes,'limit_per_split':args.limit_per_split,'sample_weight':not args.no_sample_weight}
        X_train,y_train=load_split(db,'train',args.limit_per_split); X_valid,y_valid=load_split(db,'valid',args.limit_per_split); X_test,y_test=load_split(db,'test',args.limit_per_split)
        if not X_train or not X_valid or not X_test: raise RuntimeError('Missing train/valid/test rows.')
        feature_count=len(X_train[0])
        report['counts']={'train':len(X_train),'valid':len(X_valid),'test':len(X_test),'rows_loaded':len(X_train)+len(X_valid)+len(X_test),'feature_count':feature_count,'train_label_distribution':dist(y_train),'valid_label_distribution':dist(y_valid),'test_label_distribution':dist(y_test)}
        encoder=LabelEncoder(); encoder.fit(['BUY','SELL','UNCLEAR'])
        y_train_enc=encoder.transform(y_train); X_train_np=np.asarray(X_train,dtype=np.float32); sw=make_sample_weights(y_train) if not args.no_sample_weight else None
        model=HGB(max_iter=args.max_iter,learning_rate=args.learning_rate,l2_regularization=args.l2_regularization,max_leaf_nodes=args.max_leaf_nodes,random_state=42)
        model.fit(X_train_np, y_train_enc, sample_weight=sw)
        for split,X,y in [('train',X_train,y_train),('valid',X_valid,y_valid),('test',X_test,y_test)]: report['metrics'][split]=metrics_for_split(model,encoder,X,y,deps)
        feature_names,feature_names_source=latest_feature_names()
        if feature_names and len(feature_names)!=feature_count:
            report['errors'].append(f'feature_names length mismatch: {len(feature_names)} != {feature_count}')
            feature_names=[]
        model_path=models_dir()/f'{MODEL_VERSION}_{rs}.joblib'; latest_path=models_dir()/f'{MODEL_VERSION}_latest.joblib'
        bundle={'model_version':MODEL_VERSION,'dataset_version':DATASET_VERSION,'book_feature_version':BOOK_FEATURE_VERSION,'book_label_version':BOOK_LABEL_VERSION,'ma_model_version':MA_MODEL_VERSION,'target':TARGET,'trained_at':utc_now().isoformat(),'model':model,'encoder':encoder,'classes':list(encoder.classes_),'feature_count':feature_count,'feature_names':feature_names,'feature_names_source':feature_names_source,'training_config':report['training_config'],'counts':report['counts'],'metrics':report['metrics']}
        joblib.dump(bundle,model_path); shutil.copyfile(model_path,latest_path)
        report['model_output']={'model_path':str(model_path),'latest_model_path':str(latest_path),'classes':list(encoder.classes_),'feature_count':feature_count,'feature_names_source':feature_names_source}
        report['status']='success' if not report['errors'] else 'success_with_warnings'
    except Exception as exc:
        report['status']='failed'; report['errors'].append(str(exc)); print(f'[ERROR] {exc}', flush=True)
    finally:
        ended=utc_now(); report['end_time']=ended.isoformat(); report['duration_seconds']=round((ended-started).total_seconds(),3); save_report(report,rs)
    print('\n=== Final Summary ===', flush=True)
    print(f"Status       : {report['status']}", flush=True)
    print(f"Rows Loaded  : {report.get('counts',{}).get('rows_loaded')}", flush=True)
    print(f"Feature Count: {report.get('counts',{}).get('feature_count')}", flush=True)
    print(f"TXT Report   : {report.get('report_txt_path')}", flush=True)
    print('[DONE]' if report['status']!='failed' else '[FAILED]', flush=True)
    if report['status']=='failed': sys.exit(1)

if __name__=='__main__': main()
