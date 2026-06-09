# Book & Quality

Clean project structure for the Book & Quality trading AI system.

## Architecture

1. Book Entry Model
   - Timeframes: M1 / M5 / M15
   - Target: entry direction
   - Output: p_buy / p_sell / p_notrade

2. Trade Quality Model
   - Timeframes: M30 / H1 / H4
   - Target: trade quality and market phase
   - Output: buy_quality_score / sell_quality_score / fake_score / range_score / trend_phase

3. API Decision Engine
   - Combines Book Entry and Trade Quality outputs
   - Output: final_signal / lot_multiplier / trailing_mode / breakeven_mode / tp_mode / risk_mode / reason

## Project Rules

- Raw MongoDB collections xauusd_* are read-only.
- Project outputs must use the bq_ prefix.
- Every script must generate a report.
- Every project section must be separated into its own folder.
- No-leak rule: each anchor_time may only use candles closed at or before anchor_time.
