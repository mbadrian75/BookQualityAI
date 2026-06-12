# Label Rules

## Book Entry Label v1

Entry:
- anchor_time = M15 close
- entry_time = next M15 open after anchor_time

Initial config:
- TP = 2.0 USD
- SL = 2.0 USD
- Horizon = 8 M15 candles
- Ambiguous = NOTRADE

Outputs:
- BUY
- SELL
- NOTRADE

## Trade Quality Label v1

Purpose:
- Score Buy quality and Sell quality separately.
- Detect fake movement.
- Detect range condition.
- Detect trend phase.

Outputs:
- buy_quality_score
- sell_quality_score
- fake_score
- range_score
- trend_phase
