# -*- coding: utf-8 -*-
"""
Book & Quality v2 - Build Entry Context Labels M15 v1

Purpose
-------
Create trade-oriented labels for the combined Entry Context model.
Labels are built from real future M15 candles, while features remain from the combined dataset.

Source anchors:
    bq2_dataset_entry_context_m15_v1
Raw price source:
    xauusd_m15
Target labels:
    bq2_labels_entry_context_m15_v1

Target:
    entry_label = BUY / SELL / NOTRADE
    plus buy_quality_score and sell_quality_score

TEST:
    cd C:/Project/BookQuality/bq2/02_labels/entry_context
    python -u 01_build_entry_context_labels_m15_v1.py --limit 1000 --reset

FULL:
    python -u 01_build_entry_context_labels_m15_v1.py --reset
"""
from __future__ import annotations
import argparse, bisect, json, math, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pymongo import MongoClient, ASCENDING

LABEL_VERSION='entry_context_label_m15_v1'
SOURCE_DATASET_VERSION='entry_context_dataset_m15_v1'
SOURCE_DATASET_COLLECTION='bq2_dataset_entry_context_m15_v1'
RAW_M15_COLLECTION='xauusd_m15'
TARGET_COLLECTION='bq2_labels_entry_context_m15_v1'
TARGET='entry_label'

DEFAULT_CONFIG={'mongo_uri':'mongodb://localhost:27017','database':'market_data','symbol':'XAUUSD'}

def utc_now(): return datetime.now(timezone.utc)
def stamp(dt): return dt.strftime('%Y%m%d_%H%M%S')
def project_root(): return Path(__file__).resolve().parents[2]
def reports_dir():
    p=project_root()/'02_labels'/'reports'; p.mkdir(parents=True, exist_ok=True); return p

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

def parse_dt(v):
    if v is None: return None
    if isinstance(v, datetime): return v.replace(tzinfo=None)
    if isinstance(v, str):
        try: return datetime.fromisoformat(v.replace('Z','+00:00')).replace(tzinfo=None)
        except Exception: return None
    return None

def load_m15(db):
    docs=[]; times=[]
    for d in db[RAW_M15_COLLECTION].find({}, {'_id':0,'datetime':1,'time':1,'open':1,'high':1,'low':1,'close':1}).sort('datetime', ASCENDING):
        t=parse_dt(d.get('datetime')) or parse_dt(d.get('time'))
        if t is None: continue
        o=sf(d.get('open')); h=sf(d.get('high')); l=sf(d.get('low')); c=sf(d.get('close'))
        if h<=0 or l<=0 or o<=0 or c<=0: continue
        docs.append({'datetime':t,'open':o,'high':h,'low':l,'close':c}); times.append(t)
    if not docs:
        raise RuntimeError(f'No M15 raw candles loaded from {RAW_M15_COLLECTION}')
    return times, docs

def get_future_window(times, candles, anchor_time, horizon):
    # Use candles strictly after anchor_time. First future candle open is entry price.
    idx=bisect.bisect_right(times, anchor_time)
    end=idx+horizon
    if idx<0 or end>len(candles): return None
    return candles[idx:end]

def build_label(anchor_time, future, cfg):
    entry=future[0]
    entry_time=entry['datetime']; entry_price=entry['open']
    max_high=max(c['high'] for c in future); min_low=min(c['low'] for c in future); final_close=future[-1]['close']
    up_move=max_high-entry_price
    down_move=entry_price-min_low
    close_move=final_close-entry_price
    buy_runup=max(up_move,0.0); buy_drawdown=max(down_move,0.0)
    sell_runup=max(down_move,0.0); sell_drawdown=max(up_move,0.0)
    # Reward favors both excursion and close confirmation; risk penalizes adverse excursion.
    buy_reward=cfg['runup_weight']*buy_runup + cfg['close_weight']*max(close_move,0.0)
    sell_reward=cfg['runup_weight']*sell_runup + cfg['close_weight']*max(-close_move,0.0)
    buy_quality=buy_reward - cfg['risk_weight']*buy_drawdown
    sell_quality=sell_reward - cfg['risk_weight']*sell_drawdown
    quality_gap=buy_quality-sell_quality
    abs_gap=abs(quality_gap)
    method='notrade'
    label='NOTRADE'
    # Guard: require favorable excursion and positive quality.
    if buy_runup>=cfg['min_favorable_runup_usd'] and buy_quality>=cfg['min_quality_score'] and quality_gap>=cfg['min_quality_gap']:
        label='BUY'; method='buy_quality_dominant'
    elif sell_runup>=cfg['min_favorable_runup_usd'] and sell_quality>=cfg['min_quality_score'] and -quality_gap>=cfg['min_quality_gap']:
        label='SELL'; method='sell_quality_dominant'
    else:
        if max(buy_runup, sell_runup)<cfg['min_favorable_runup_usd']:
            method='low_future_movement'
        elif max(buy_quality, sell_quality)<cfg['min_quality_score']:
            method='low_quality'
        elif abs_gap<cfg['min_quality_gap']:
            method='small_quality_gap'
    # Soft scores for later regression/classification diagnostics.
    # Shift by min over [buy, sell, 0] and normalize.
    vals={'BUY':buy_quality,'SELL':sell_quality,'NOTRADE':0.0}
    mn=min(vals.values())
    shifted={k:max(v-mn,0.0)+1e-9 for k,v in vals.items()}
    total=sum(shifted.values())
    soft={k:shifted[k]/total for k in shifted}
    return {
        'entry_time':entry_time,'entry_price':entry_price,
        'future_horizon_m15':cfg['horizon_m15'],
        'future_max_high':max_high,'future_min_low':min_low,'future_final_close':final_close,
        'close_move_usd':round(close_move,6),
        'buy_runup_usd':round(buy_runup,6),'buy_drawdown_usd':round(buy_drawdown,6),
        'sell_runup_usd':round(sell_runup,6),'sell_drawdown_usd':round(sell_drawdown,6),
        'buy_quality_score':round(buy_quality,6),'sell_quality_score':round(sell_quality,6),
        'quality_gap':round(quality_gap,6),'abs_quality_gap':round(abs_gap,6),
        'entry_label':label,'label_method':method,
        'soft_buy_score':round(soft['BUY'],6),'soft_sell_score':round(soft['SELL'],6),'soft_notrade_score':round(soft['NOTRADE'],6),
    }

def save_report(report, rs):
    jp=reports_dir()/f'entry_context_labels_m15_v1_report_{rs}.json'
    tp=reports_dir()/f'entry_context_labels_m15_v1_report_{rs}.txt'
    report['report_json_path']=str(jp); report['report_txt_path']=str(tp)
    jp.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    lines=['Book & Quality v2 - Entry Context Labels M15 v1 Report','='*74]
    for k in ['run_id','status','database','symbol','start_time','end_time','duration_seconds']:
        lines.append(f'{k:<34}: {report.get(k)}')
    for title,key in [('Collections','collections'),('Config','config'),('Counts','counts'),('Label Distribution','label_distribution'),('Label Method Distribution','label_method_distribution'),('Quality Summary','quality_summary')]:
        lines+=['',title,'-'*74]
        for kk,vv in report.get(key,{}).items(): lines.append(f'{kk:<34}: {vv}')
    if report.get('errors'):
        lines+=['','Errors','-'*74]+[f'- {e}' for e in report['errors'][:50]]
    tp.write_text('\n'.join(lines),encoding='utf-8')

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--reset', action='store_true')
    p.add_argument('--horizon-m15', type=int, default=4)
    p.add_argument('--runup-weight', type=float, default=0.65)
    p.add_argument('--close-weight', type=float, default=0.35)
    p.add_argument('--risk-weight', type=float, default=0.60)
    p.add_argument('--min-favorable-runup-usd', type=float, default=0.80)
    p.add_argument('--min-quality-score', type=float, default=0.20)
    p.add_argument('--min-quality-gap', type=float, default=0.30)
    return p.parse_args()

def main():
    args=parse_args(); started=utc_now(); rs=stamp(started)
    report={'run_id':f'entry_context_labels_m15_v1_{rs}','status':'running','database':None,'symbol':None,'start_time':started.isoformat(),'end_time':None,'duration_seconds':None,'collections':{},'config':{},'counts':{},'label_distribution':{},'label_method_distribution':{},'quality_summary':{},'errors':[],'args':vars(args)}
    print('=== Book & Quality v2 - Build Entry Context Labels M15 v1 ===', flush=True)
    try:
        cfg=load_config(); db=connect(cfg); report['database']=cfg['database']; report['symbol']=cfg['symbol']
        lcfg={'horizon_m15':args.horizon_m15,'runup_weight':args.runup_weight,'close_weight':args.close_weight,'risk_weight':args.risk_weight,'min_favorable_runup_usd':args.min_favorable_runup_usd,'min_quality_score':args.min_quality_score,'min_quality_gap':args.min_quality_gap}
        report['collections']={'source_dataset':SOURCE_DATASET_COLLECTION,'raw_m15':RAW_M15_COLLECTION,'target':TARGET_COLLECTION}
        report['config']={'label_version':LABEL_VERSION,'source_dataset_version':SOURCE_DATASET_VERSION,'target':TARGET,**lcfg}
        target=db[TARGET_COLLECTION]
        if args.reset:
            print(f'[RESET] Dropping target collection: {TARGET_COLLECTION}', flush=True); target.drop()
        target.create_index([('anchor_time',ASCENDING)], unique=True)
        target.create_index([('entry_label',ASCENDING)])
        target.create_index([('label_version',ASCENDING)])
        print('[LOAD] Raw M15 candles...', flush=True)
        times,candles=load_m15(db)
        print(f'[LOAD] M15 candles loaded: {len(candles)}', flush=True)
        counts=Counter(); labels=Counter(); methods=Counter(); errs=[]
        buyq=[]; sellq=[]; gap=[]
        cursor=db[SOURCE_DATASET_COLLECTION].find({'dataset_version':SOURCE_DATASET_VERSION},{'_id':0,'anchor_time':1,'decision_time':1,'split':1}).sort('anchor_time',ASCENDING)
        if args.limit: cursor=cursor.limit(args.limit)
        for doc in cursor:
            counts['anchors_seen']+=1
            try:
                at=parse_dt(doc.get('anchor_time')) or parse_dt(doc.get('decision_time'))
                if at is None:
                    counts['skipped_missing_anchor_time']+=1; continue
                fut=get_future_window(times,candles,at,args.horizon_m15)
                if fut is None:
                    counts['skipped_missing_future']+=1; continue
                lab=build_label(at,fut,lcfg)
                out={'label_version':LABEL_VERSION,'source_dataset_version':SOURCE_DATASET_VERSION,'symbol':cfg['symbol'],'anchor_time':at,'decision_time':doc.get('decision_time'),'split':doc.get('split'),**lab}
                target.replace_one({'anchor_time':at}, out, upsert=True)
                counts['labels_built']+=1; labels[lab['entry_label']]+=1; methods[lab['label_method']]+=1
                buyq.append(lab['buy_quality_score']); sellq.append(lab['sell_quality_score']); gap.append(lab['abs_quality_gap'])
                if counts['labels_built']%50000==0: print(f"[PROGRESS] labels_built={counts['labels_built']}", flush=True)
            except Exception as exc:
                counts['row_errors']+=1
                if len(errs)<20: errs.append(str(exc))
        report['counts']=dict(counts); report['label_distribution']=dict(labels); report['label_method_distribution']=dict(methods)
        def avg(a): return round(sum(a)/len(a),6) if a else None
        def mx(a): return round(max(a),6) if a else None
        report['quality_summary']={'avg_buy_quality':avg(buyq),'avg_sell_quality':avg(sellq),'avg_abs_quality_gap':avg(gap),'max_abs_quality_gap':mx(gap)}
        if errs: report['errors']=errs
        report['status']='success' if counts.get('row_errors',0)==0 and counts.get('labels_built',0)>0 else 'success_with_warnings'
    except Exception as exc:
        report['status']='failed'; report['errors'].append(str(exc)); print(f'[ERROR] {exc}', flush=True)
    finally:
        ended=utc_now(); report['end_time']=ended.isoformat(); report['duration_seconds']=round((ended-started).total_seconds(),3); save_report(report,rs)
    print('\n=== Final Summary ===')
    print('Status       :', report['status'])
    print('Labels Built :', report.get('counts',{}).get('labels_built'))
    print('TXT Report   :', report.get('report_txt_path'))
    print('[DONE]' if report['status']!='failed' else '[FAILED]')
    if report['status']=='failed': sys.exit(1)
if __name__=='__main__': main()
