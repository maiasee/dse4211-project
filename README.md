# DSE4211 Project: LSTM-Based Crypto Portfolio Optimization

This project builds an end-to-end pipeline for **cryptocurrency return forecasting and portfolio optimization**. It trains per-asset LSTM models to predict 7-day-ahead returns for 8 major cryptocurrencies, then feeds those forecasts into a Mean-Variance Optimization (MVO) framework to construct and backtest optimal portfolios.

## Assets Covered
BTC, ETH, BNB, ADA, XRP, LTC, BCH, LINK (daily OHLCV data from Binance, 2020–2025)

## Pipeline Overview

### Pipeline (run in order)

`01_binance_data_cleaning.ipynb` → `02_lstm_preprocessing_pipeline.ipynb` → `03_lstm_training.ipynb` → `04_run_portfolio.py`

| Step | File | Description |
|------|------|-------------|
| 1 | `01_binance_data_cleaning.ipynb` | Fetches raw OHLCV data, computes technical indicators (SMA, EMA, momentum, CCI, regime labels), adds a 7-day return target, and splits into train/val/test sets |
| 2 | `02_lstm_preprocessing_pipeline.ipynb` | Converts cleaned data into per-asset LSTM-ready sequences with a 30-day lookback window and z-score scaling; saves numpy arrays to `lstm_ready_data/` |
| 3 | `03_lstm_training.ipynb` | Trains baseline and regime-aware LSTM models per asset using random hyperparameter search with expanding-window cross-validation; saves forecasts and model weights to `outputs/` |
| 4 | `04_run_portfolio.py` | Orchestration script: loads forecasts, runs backtests for baseline and regime-aware models, computes performance metrics (Sharpe, volatility, max drawdown), and benchmarks against equal-weight |

### Supporting Modules (called by `04_run_portfolio.py`)
```
functions/
  ├── data_processing.py
  ├── mvo.py
  ├── sensitivity_analysis.py
  └── backtest.py
```

| File | Description |
|------|-------------|
| `functions/data_processing.py` | Utility module for loading historical returns and LSTM forecasts, aligning dates/assets, and computing 7-day cumulative returns |
| `functions/mvo.py` | Mean-Variance Optimization: solves a constrained quadratic program to compute optimal portfolio weights given forecasted returns and a covariance matrix |
| `functions/backtest.py` | Backtesting engine that rebalances weekly using MVO weights and tracks realized portfolio returns and weight history |
| `mvo_sensitivity_analysis.ipynb` | Grid search over regularization (lambda) and weight constraint parameters to identify hyperparameter combinations with the best Sharpe ratio, cumulative return, and max drawdown |

## Setup

```bash
pip install -r requirements.txt
```

Run the notebooks in order (01 → 04) for the full backtest and analysis.

## Output

Results (forecasts, model configs, performance metrics, figures) are saved to the `outputs/` directory.
