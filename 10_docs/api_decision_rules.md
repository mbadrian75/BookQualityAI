# API Decision Rules

The API Decision Engine combines:

1. Book Entry Model:
   - p_buy
   - p_sell
   - p_notrade

2. Trade Quality Model:
   - buy_quality_score
   - sell_quality_score
   - fake_score
   - range_score
   - trend_phase

Final output:
- final_signal
- lot_multiplier
- trailing_mode
- breakeven_mode
- tp_mode
- risk_mode
- reason
