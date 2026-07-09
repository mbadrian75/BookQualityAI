# -*- coding: utf-8 -*-
r"""
01_build_channel_zone_features_v2.py

Build SCE Channel-Zone entry features.

این فایل جایگزین بخش Featureهای قبلی M5/M15 می‌شود و بخش تایم‌فریم‌های بزرگ
M30/H1/H4 را تغییر نمی‌دهد.

منطق:
- برای هر کندل M5، آخرین کندل‌های بسته‌شده M15 پیدا می‌شود.
- از 20 کندل بسته‌شده قبلی M15، سقف/کف/میانه کانال ساخته می‌شود.
- کندل M5 سازنده سقف و کندل M5 سازنده کف همان بازه پیدا می‌شود.
- از بدنه آن دو کندل، محدوده سقف و محدوده کف ساخته می‌شود.
- برای هر کندل M5 دو رکورد تولید می‌شود: BUY و SELL.

No-Leak:
اگر datetime کندل‌ها زمان Open باشد:
    m15_open + 15min <= m5_open + 5min

دستور اجرا:
python 02_features\channel_zone\01_build_channel_zone_features_v2.py --db market_data --m5 xauusd_m5 --m15 xauusd_m15 --out sce_features_channel_zone_entry_m5_v2 --drop-output
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import ASCENDING, MongoClient, UpdateOne


SIDE_BUY = "BUY"
SIDE_SELL = "SELL"


def parse_dt(value: str | None) -> dt.datetime | None:
    if not value:
        return None

    value = value.strip()
    if not value:
        return None

    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def safe_float(x: Any) -> float | None:
    if x is None:
        return None

    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def ensure_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def load_candles(
    db,
    collection: str,
    datetime_field: str,
    start: dt.datetime | None,
    end: dt.datetime | None,
    projection_fields: list[str],
) -> pd.DataFrame:
    query: dict[str, Any] = {}

    if start or end:
        query[datetime_field] = {}

        if start:
            query[datetime_field]["$gte"] = start

        if end:
            query[datetime_field]["$lt"] = end

    projection = {field: 1 for field in projection_fields}
    projection["_id"] = 0

    rows = list(db[collection].find(query, projection).sort(datetime_field, ASCENDING))

    if not rows:
        raise RuntimeError(f"No rows found in collection={collection} query={query}")

    df = pd.DataFrame(rows)
    df[datetime_field] = pd.to_datetime(df[datetime_field])
    df = df.sort_values(datetime_field).reset_index(drop=True)

    return df


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def add_m5_indicators(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    ma_period: int,
    cci_period: int,
    atr_period: int,
) -> pd.DataFrame:
    df = df.copy()

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)

    typical_price = (high + low + close) / 3.0

    ma_col = f"ma_{ma_period}"
    cci_col = f"cci_{cci_period}"
    atr_col = f"atr_{atr_period}"

    df[ma_col] = close.rolling(ma_period, min_periods=ma_period).mean()
    df[f"{ma_col}_slope"] = df[ma_col] - df[ma_col].shift(1)

    sma_tp = typical_price.rolling(cci_period, min_periods=cci_period).mean()

    def mean_abs_dev(values: np.ndarray) -> float:
        mean_value = float(np.mean(values))
        return float(np.mean(np.abs(values - mean_value)))

    mad = typical_price.rolling(cci_period, min_periods=cci_period).apply(mean_abs_dev, raw=True)
    mad = mad.replace(0, np.nan)

    df[cci_col] = (typical_price - sma_tp) / (0.015 * mad)
    df[f"{cci_col}_slope"] = df[cci_col] - df[cci_col].shift(1)

    tr = true_range(high, low, close)
    df[atr_col] = tr.rolling(atr_period, min_periods=atr_period).mean()
    df[f"{atr_col}_slope"] = df[atr_col] - df[atr_col].shift(1)

    return df


def sign_for_side(side: str) -> int:
    return 1 if side == SIDE_BUY else -1


def make_side_features(base: dict[str, Any], side: str) -> dict[str, Any]:
    side_sign = sign_for_side(side)
    close = base["close"]

    ma = base.get("m5_ma")
    ma_slope = base.get("m5_ma_slope")
    cci = base.get("m5_cci")
    cci_slope = base.get("m5_cci_slope")

    if side == SIDE_BUY:
        primary_break = int(base["low_zone_break_up"] or base["high_zone_break_up"])
        primary_rejection_or_break = int(base["low_zone_rejection_up"] or base["high_zone_break_up"])
        entry_from_low_zone = int(base["low_zone_break_up"] or base["low_zone_rejection_up"])
        entry_from_high_zone = int(base["high_zone_break_up"])
        adverse_break = int(base["high_zone_break_down"] or base["low_zone_break_down"])

        distance_to_mid = base["channel_mid"] - close
        distance_to_far_zone = base["high_zone_high"] - close

    else:
        primary_break = int(base["high_zone_break_down"] or base["low_zone_break_down"])
        primary_rejection_or_break = int(base["high_zone_rejection_down"] or base["low_zone_break_down"])
        entry_from_low_zone = int(base["low_zone_break_down"])
        entry_from_high_zone = int(base["high_zone_break_down"] or base["high_zone_rejection_down"])
        adverse_break = int(base["low_zone_break_up"] or base["high_zone_break_up"])

        distance_to_mid = close - base["channel_mid"]
        distance_to_far_zone = close - base["low_zone_low"]

    side_ma_slope = None if ma_slope is None else side_sign * ma_slope
    side_cci_value = None if cci is None else side_sign * cci
    side_cci_slope = None if cci_slope is None else side_sign * cci_slope
    side_price_vs_ma = None if ma is None else side_sign * (close - ma)

    return {
        "side": side,
        "side_sign": side_sign,

        "side_zone_primary_break": primary_break,
        "side_zone_primary_rejection_or_break": primary_rejection_or_break,
        "side_adverse_zone_break": adverse_break,

        "side_entry_from_low_zone": entry_from_low_zone,
        "side_entry_from_high_zone": entry_from_high_zone,

        "side_distance_to_mid": safe_float(distance_to_mid),
        "side_distance_to_far_zone": safe_float(distance_to_far_zone),

        "side_price_vs_ma": safe_float(side_price_vs_ma),
        "side_ma_slope": safe_float(side_ma_slope),
        "side_cci_value": safe_float(side_cci_value),
        "side_cci_slope": safe_float(side_cci_slope),

        "side_ma_supports": None if side_ma_slope is None else int(side_ma_slope > 0),
        "side_cci_supports": None if side_cci_value is None else int(side_cci_value > 0),
        "side_cci_moving_to_side": None if side_cci_slope is None else int(side_cci_slope > 0),
    }


def build_base_feature(
    *,
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    m5_times: np.ndarray,
    m5_high_values: np.ndarray,
    m5_low_values: np.ndarray,
    i_m5: int,
    j_m15: int,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    lookback = args.lookback_m15

    if i_m5 < 1:
        return None

    if j_m15 < lookback - 1:
        return None

    m15_start_idx = j_m15 - lookback + 1
    m15_end_idx = j_m15 + 1

    m15_window = m15.iloc[m15_start_idx:m15_end_idx]

    if len(m15_window) != lookback:
        return None

    row = m5.iloc[i_m5]
    prev = m5.iloc[i_m5 - 1]

    open_price = float(row[args.open])
    high_price = float(row[args.high])
    low_price = float(row[args.low])
    close_price = float(row[args.close])
    prev_close = float(prev[args.close])

    channel_high = float(m15_window[args.high].max())
    channel_low = float(m15_window[args.low].min())
    channel_range = channel_high - channel_low

    if channel_range <= 0:
        return None

    channel_mid = (channel_high + channel_low) / 2.0

    window_start = m15_window[args.datetime_field].iloc[0]

    if args.datetime_is_open:
        window_end = m15_window[args.datetime_field].iloc[-1] + pd.Timedelta(minutes=15)

        left = int(np.searchsorted(m5_times, np.datetime64(window_start), side="left"))
        right = int(np.searchsorted(m5_times, np.datetime64(window_end), side="left"))
    else:
        window_end = m15_window[args.datetime_field].iloc[-1]

        left = int(np.searchsorted(m5_times, np.datetime64(window_start), side="left"))
        right = int(np.searchsorted(m5_times, np.datetime64(window_end), side="right"))

    if right <= left:
        return None

    window_high_values = m5_high_values[left:right]
    window_low_values = m5_low_values[left:right]

    if len(window_high_values) == 0 or len(window_low_values) == 0:
        return None

    high_m5_idx = left + int(np.nanargmax(window_high_values))
    low_m5_idx = left + int(np.nanargmin(window_low_values))

    high_m5 = m5.iloc[high_m5_idx]
    low_m5 = m5.iloc[low_m5_idx]

    high_zone_low = min(float(high_m5[args.open]), float(high_m5[args.close]))
    high_zone_high = max(float(high_m5[args.open]), float(high_m5[args.close]))

    low_zone_low = min(float(low_m5[args.open]), float(low_m5[args.close]))
    low_zone_high = max(float(low_m5[args.open]), float(low_m5[args.close]))

    body_low = min(open_price, close_price)
    body_high = max(open_price, close_price)

    high_zone_break_up = int(prev_close <= high_zone_high and close_price > high_zone_high)
    high_zone_break_down = int(prev_close >= high_zone_low and close_price < high_zone_low)

    low_zone_break_up = int(prev_close <= low_zone_high and close_price > low_zone_high)
    low_zone_break_down = int(prev_close >= low_zone_low and close_price < low_zone_low)

    high_zone_rejection_down = int(high_price >= high_zone_low and close_price < high_zone_low)
    low_zone_rejection_up = int(low_price <= low_zone_high and close_price > low_zone_high)

    ma_col = f"ma_{args.ma_period}"
    cci_col = f"cci_{args.cci_period}"
    atr_col = f"atr_{args.atr_period}"

    m5_ma = safe_float(row.get(ma_col))
    m5_ma_slope = safe_float(row.get(f"{ma_col}_slope"))
    m5_cci = safe_float(row.get(cci_col))
    m5_cci_slope = safe_float(row.get(f"{cci_col}_slope"))
    m5_atr = safe_float(row.get(atr_col))
    m5_atr_slope = safe_float(row.get(f"{atr_col}_slope"))

    break_distance_high_up = max(0.0, close_price - high_zone_high)
    break_distance_high_down = max(0.0, high_zone_low - close_price)
    break_distance_low_up = max(0.0, close_price - low_zone_high)
    break_distance_low_down = max(0.0, low_zone_low - close_price)

    max_break_distance = max(
        break_distance_high_up,
        break_distance_high_down,
        break_distance_low_up,
        break_distance_low_down,
    )

    zone_break_strength_atr = None

    if m5_atr is not None and m5_atr > 0:
        zone_break_strength_atr = max_break_distance / m5_atr

    return {
        "datetime": row[args.datetime_field].to_pydatetime(),
        "feature_version": "sce_channel_zone_entry_m5_v2",
        "anchor_tf": "M5",
        "context_tf": "M15",
        "lookback_m15": args.lookback_m15,

        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "body_low": body_low,
        "body_high": body_high,

        "last_closed_m15_datetime": m15.iloc[j_m15][args.datetime_field].to_pydatetime(),
        "channel_start_m15_datetime": m15_window.iloc[0][args.datetime_field].to_pydatetime(),
        "channel_end_m15_datetime": m15_window.iloc[-1][args.datetime_field].to_pydatetime(),

        "channel_high": channel_high,
        "channel_low": channel_low,
        "channel_mid": channel_mid,
        "channel_range": channel_range,

        "price_position_in_channel": safe_float((close_price - channel_low) / channel_range),
        "distance_to_channel_high": safe_float(channel_high - close_price),
        "distance_to_channel_low": safe_float(close_price - channel_low),
        "distance_to_channel_mid": safe_float(close_price - channel_mid),

        "high_zone_m5_datetime": high_m5[args.datetime_field].to_pydatetime(),
        "high_zone_low": high_zone_low,
        "high_zone_high": high_zone_high,
        "high_zone_full_low": float(high_m5[args.low]),
        "high_zone_full_high": float(high_m5[args.high]),
        "high_zone_thickness": high_zone_high - high_zone_low,

        "low_zone_m5_datetime": low_m5[args.datetime_field].to_pydatetime(),
        "low_zone_low": low_zone_low,
        "low_zone_high": low_zone_high,
        "low_zone_full_low": float(low_m5[args.low]),
        "low_zone_full_high": float(low_m5[args.high]),
        "low_zone_thickness": low_zone_high - low_zone_low,

        "close_above_high_zone": int(close_price > high_zone_high),
        "close_below_high_zone": int(close_price < high_zone_low),
        "close_inside_high_zone": int(high_zone_low <= close_price <= high_zone_high),
        "body_fully_above_high_zone": int(body_low > high_zone_high),
        "body_fully_below_high_zone": int(body_high < high_zone_low),

        "close_above_low_zone": int(close_price > low_zone_high),
        "close_below_low_zone": int(close_price < low_zone_low),
        "close_inside_low_zone": int(low_zone_low <= close_price <= low_zone_high),
        "body_fully_above_low_zone": int(body_low > low_zone_high),
        "body_fully_below_low_zone": int(body_high < low_zone_low),

        "high_zone_break_up": high_zone_break_up,
        "high_zone_break_down": high_zone_break_down,
        "low_zone_break_up": low_zone_break_up,
        "low_zone_break_down": low_zone_break_down,

        "high_zone_rejection_down": high_zone_rejection_down,
        "low_zone_rejection_up": low_zone_rejection_up,

        "zone_break_max_distance": safe_float(max_break_distance),
        "zone_break_strength_atr": safe_float(zone_break_strength_atr),

        "m5_ma": m5_ma,
        "m5_ma_slope": m5_ma_slope,
        "m5_cci": m5_cci,
        "m5_cci_slope": m5_cci_slope,
        "m5_atr": m5_atr,
        "m5_atr_slope": m5_atr_slope,

        "channel_range_to_atr": safe_float(channel_range / m5_atr)
        if m5_atr is not None and m5_atr > 0
        else None,

        "created_at": dt.datetime.now(),
    }


def write_report(report_path: Path, lines: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db", default="market_data")

    parser.add_argument("--m5", default="xauusd_m5")
    parser.add_argument("--m15", default="xauusd_m15")
    parser.add_argument("--out", default="sce_features_channel_zone_entry_m5_v2")

    parser.add_argument("--drop-output", action="store_true")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")

    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--lookback-m15", type=int, default=20)

    parser.add_argument("--datetime-field", default="datetime")
    parser.add_argument("--datetime-is-open", type=int, default=1)

    parser.add_argument("--open", default="open")
    parser.add_argument("--high", default="high")
    parser.add_argument("--low", default="low")
    parser.add_argument("--close", default="close")

    parser.add_argument("--ma-period", type=int, default=20)
    parser.add_argument("--cci-period", type=int, default=20)
    parser.add_argument("--atr-period", type=int, default=14)

    parser.add_argument("--report-dir", default="02_features/reports")

    args = parser.parse_args()
    args.datetime_is_open = bool(args.datetime_is_open)

    started_at = dt.datetime.now()
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report_dir) / f"sce_build_channel_zone_features_v2_{stamp}.txt"

    start = parse_dt(args.start)
    end = parse_dt(args.end)

    client = MongoClient(args.mongo_uri)
    db = client[args.db]

    fields = [args.datetime_field, args.open, args.high, args.low, args.close]

    load_start = start

    if start:
        load_start = start - dt.timedelta(days=10)

    print("Loading M5 candles...")

    m5 = load_candles(
        db=db,
        collection=args.m5,
        datetime_field=args.datetime_field,
        start=load_start,
        end=end,
        projection_fields=fields,
    )

    print("Loading M15 candles...")

    m15 = load_candles(
        db=db,
        collection=args.m15,
        datetime_field=args.datetime_field,
        start=load_start,
        end=end,
        projection_fields=fields,
    )

    ensure_columns(m5, fields, args.m5)
    ensure_columns(m15, fields, args.m15)

    for col in [args.open, args.high, args.low, args.close]:
        m5[col] = m5[col].astype(float)
        m15[col] = m15[col].astype(float)

    print("Calculating M5 MA/CCI/ATR indicators...")

    m5 = add_m5_indicators(
        df=m5,
        high_col=args.high,
        low_col=args.low,
        close_col=args.close,
        ma_period=args.ma_period,
        cci_period=args.cci_period,
        atr_period=args.atr_period,
    )

    m5_times = m5[args.datetime_field].to_numpy(dtype="datetime64[ns]")
    m15_times = m15[args.datetime_field].to_numpy(dtype="datetime64[ns]")

    m5_high_values = m5[args.high].to_numpy(dtype=float)
    m5_low_values = m5[args.low].to_numpy(dtype=float)

    if args.drop_output:
        print(f"Dropping output collection: {args.out}")
        db.drop_collection(args.out)

    out = db[args.out]

    out.create_index([("datetime", ASCENDING), ("side", ASCENDING)], unique=True)
    out.create_index([("datetime", ASCENDING)])
    out.create_index([("side", ASCENDING)])
    out.create_index([("feature_version", ASCENDING)])

    ops: list[UpdateOne] = []

    written_or_updated = 0
    skipped_no_history = 0
    skipped_build_none = 0
    processed_anchor_rows = 0
    batches = 0

    print("Building Channel-Zone features...")

    for i_m5 in range(len(m5)):
        row_dt = m5.iloc[i_m5][args.datetime_field]

        if start and row_dt.to_pydatetime() < start:
            continue

        if end and row_dt.to_pydatetime() >= end:
            continue

        processed_anchor_rows += 1

        if args.datetime_is_open:
            anchor_close_time = row_dt + pd.Timedelta(minutes=5)
            last_allowed_m15_time = anchor_close_time - pd.Timedelta(minutes=15)
        else:
            last_allowed_m15_time = row_dt

        j_m15 = int(
            np.searchsorted(
                m15_times,
                np.datetime64(last_allowed_m15_time),
                side="right",
            )
            - 1
        )

        if j_m15 < args.lookback_m15 - 1:
            skipped_no_history += 1
            continue

        base_feature = build_base_feature(
            m5=m5,
            m15=m15,
            m5_times=m5_times,
            m5_high_values=m5_high_values,
            m5_low_values=m5_low_values,
            i_m5=i_m5,
            j_m15=j_m15,
            args=args,
        )

        if base_feature is None:
            skipped_build_none += 1
            continue

        for side in (SIDE_BUY, SIDE_SELL):
            doc = dict(base_feature)
            doc.update(make_side_features(base_feature, side))

            ops.append(
                UpdateOne(
                    {"datetime": doc["datetime"], "side": doc["side"]},
                    {"$set": doc},
                    upsert=True,
                )
            )

        if len(ops) >= args.batch_size:
            result = out.bulk_write(ops, ordered=False)
            written_or_updated += result.upserted_count + result.modified_count
            batches += 1
            ops.clear()

            if batches % 20 == 0:
                print(
                    f"batches={batches:,} | "
                    f"processed_anchor_rows={processed_anchor_rows:,} | "
                    f"written_or_updated≈{written_or_updated:,}"
                )

    if ops:
        result = out.bulk_write(ops, ordered=False)
        written_or_updated += result.upserted_count + result.modified_count
        batches += 1
        ops.clear()

    finished_at = dt.datetime.now()
    duration_seconds = (finished_at - started_at).total_seconds()
    output_docs_total = out.estimated_document_count()

    report_lines = [
        "SCE Channel-Zone Feature Build Report",
        "=" * 80,
        f"started_at             : {started_at}",
        f"finished_at            : {finished_at}",
        f"duration_seconds       : {duration_seconds:.2f}",
        "",
        "Mongo",
        "-" * 80,
        f"mongo_uri              : {args.mongo_uri}",
        f"db                     : {args.db}",
        f"m5_collection          : {args.m5}",
        f"m15_collection         : {args.m15}",
        f"output_collection      : {args.out}",
        f"drop_output            : {args.drop_output}",
        "",
        "Parameters",
        "-" * 80,
        f"start                  : {start}",
        f"end                    : {end}",
        f"lookback_m15           : {args.lookback_m15}",
        f"datetime_is_open       : {args.datetime_is_open}",
        f"ma_period              : {args.ma_period}",
        f"cci_period             : {args.cci_period}",
        f"atr_period             : {args.atr_period}",
        "",
        "Counts",
        "-" * 80,
        f"m5_rows_loaded         : {len(m5):,}",
        f"m15_rows_loaded        : {len(m15):,}",
        f"processed_anchor_rows  : {processed_anchor_rows:,}",
        f"bulk_batches           : {batches:,}",
        f"written_or_updated≈    : {written_or_updated:,}",
        f"output_docs_total≈     : {output_docs_total:,}",
        f"skipped_no_history     : {skipped_no_history:,}",
        f"skipped_build_none     : {skipped_build_none:,}",
        "",
        "Feature version",
        "-" * 80,
        "sce_channel_zone_entry_m5_v2",
        "",
        "Design",
        "-" * 80,
        "Lower feature block replaced:",
        "old M5/M15 candle feature block -> 20-M15 Channel-Zone Feature Block",
        "",
        "Higher timeframe context:",
        "unchanged: M30/H1/H4 MA context must be joined by the existing pipeline.",
    ]

    write_report(report_path, report_lines)

    print("\n".join(report_lines))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()