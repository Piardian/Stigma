# Stigma — Statistical / Econometric Quant Engine

Independent research engine based on price-series statistics, Kalman state estimation, Ornstein–Uhlenbeck mean reversion, and walk-forward evaluation. This repository is independent of Begonya, PragmaProject, and TrawmaProject.

## Status

Initial scaffold. Research outputs are not a trading whitelist until out-of-sample tests include realistic spread, commission, slippage, gaps, and walk-forward parameter selection. The requested large MT5/Binance runs require the user's local market data and terminal/API credentials; they cannot be truthfully marked complete from this repository scaffold.

## Install

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -v
```

## Core conventions

- Inputs are chronological OHLC arrays; closed bars only.
- No tick-volume features are used by the quant signal engine.
- Missing/stale bars and invalid numeric values must fail closed.
- Walk-forward results are research evidence, not a guarantee of future performance.
