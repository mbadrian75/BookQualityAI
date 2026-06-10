# Book & Quality v2 - Locked Spec

## Core Rule
Quality predicts. Candle Book confirms.

## MA Phase & Quality Model
Inputs: M30/H1/H4 + MA 20/50/100.
Role: main scenario prediction, probability, phase, quality, risk, lot, TP, SL, trailing and exit.
Important: current alignment is not mandatory. A timeframe can be currently down but at DOWN_END and predicted to reverse up.

## Candle Book Confirmation Model
Inputs: M1/M5/M15 candle behavior.
Role: confirm entry timing and direction.

## API
Quality BUY + Book BUY => BUY.
Quality SELL + Book SELL => SELL.
Quality low probability + Book confirms => small lot.
Quality BLOCK/HIGH_RISK => NOTRADE.
