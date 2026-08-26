# NextCandle AI — V1

A research prototype that estimates the probability of the **next completed candle's direction** using Bybit historical OHLCV data.

## Important

This is **not a guaranteed trading system**. V1 is deliberately paper/research only. Do not connect it to real order execution.

## What V1 does

- Downloads historical USDT perpetual candles from Bybit public market data.
- Creates price-action, momentum, volatility, volume and EMA/RSI features.
- Trains a gradient-boosting classifier.
- Uses chronological holdout validation.
- Runs an expanding-window walk-forward backtest.
- Produces Bullish / Neutral / Bearish probabilities.
- Has a configurable probability threshold and a NO EDGE state.

## Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then choose the pair and timeframe in the sidebar.

## Next upgrades

1. Probability calibration.
2. Better target definitions (direction + movement threshold).
3. More market-state features.
4. Open interest/funding/order-book features.
5. Better walk-forward evaluation and fee/slippage simulation.
6. Multi-pair watchlist selected by the user.
7. Live WebSocket candles with closed-candle confirmation.
8. Paper-trading ledger.
9. Production dashboard and alerts.
