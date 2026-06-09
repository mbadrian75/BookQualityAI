# Feature Rules

## Book Entry Features

Allowed timeframes:
- M1
- M5
- M15

Not allowed:
- M30
- H1
- H4

## Trade Quality Features

Allowed timeframes:
- M30
- H1
- H4

Not allowed:
- M1
- M5
- M15

## No-Leak Rule

For every anchor_time:
- only candles already closed at or before anchor_time may be used.
