import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from functions.data_processing import load_data
from functions.backtest import run_backtest

def compute_metrics(log_returns):
    log_returns = log_returns.dropna()

    total_return = np.exp(log_returns.sum()) - 1

    volatility = log_returns.std() * (52 ** 0.5)
    sharpe = np.nan
    if log_returns.std() != 0:
        sharpe = (log_returns.mean() / log_returns.std()) * (52 ** 0.5)

    cumulative = np.exp(log_returns.cumsum())
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }

def run_equal_weight_backtest(daily_log_returns, rebalancing_dates):
    daily_log_returns = daily_log_returns.copy()
    daily_log_returns.index = pd.to_datetime(daily_log_returns.index)
    rebalancing_dates = pd.to_datetime(rebalancing_dates)

    ew_returns = []

    for i in range(len(rebalancing_dates) - 1):
        start = rebalancing_dates[i]
        end = rebalancing_dates[i + 1]

        window = daily_log_returns[
            (daily_log_returns.index > start) & (daily_log_returns.index <= end)
        ]

        if window.empty:
            continue

        w = np.ones(window.shape[1]) / window.shape[1]
        port_ret = window @ w

        ew_returns.append(port_ret.sum())

    return pd.Series(ew_returns, index=rebalancing_dates[1:len(ew_returns)+1])

def main():
    ALL_FCSTS_FP = "outputs/lstm_model_output/all_forecasts.csv"
    HIST_DATA_FP = "outputs/data/binance_data_raw.csv"

    mu_base, mu_regime, daily_log_returns, cumulative_returns, rebalancing_dates = load_data(
        HIST_DATA_FP, ALL_FCSTS_FP
    )

    print("Data Shape Diagnostic:")
    print(f"mu_base shape: {mu_base.shape}")
    print(f"mu_regime shape: {mu_regime.shape}")
    print(f"daily_log_returns shape: {daily_log_returns.shape}")
    print(f"cumulative_returns shape: {cumulative_returns.shape}")
    print(f"number of rebalancing dates: {len(rebalancing_dates)}")

    # Run backtests
    base_ret, base_w = run_backtest(
        mu_base,
        daily_log_returns,
        cumulative_returns,
        rebalancing_dates,
        window=20,
        lamda=5.0
    )

    reg_ret, reg_w = run_backtest(
        mu_regime,
        daily_log_returns,
        cumulative_returns,
        rebalancing_dates,
        window=20,
        lamda=5.0
    )

    ew_ret = run_equal_weight_backtest(daily_log_returns, rebalancing_dates)
    cum_ew = np.exp(ew_ret.cumsum())    

    # print("Check Weekly Structure:")
    # print("Expected for weekly over 1 year: ~52")

    # Cumulative portfolio value from weekly LOG returns
    cum_base = np.exp(base_ret.cumsum())
    cum_reg = np.exp(reg_ret.cumsum())
    ew_metrics = compute_metrics(ew_ret)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(cum_base.index), cum_base.values, label="Baseline Portfolio")
    plt.plot(pd.to_datetime(cum_reg.index), cum_reg.values, label="Regime Portfolio")
    plt.plot(pd.to_datetime(cum_ew.index), cum_ew.values, label="Equal Weight")
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/figures/cumulative_returns.png")
    plt.close()

    # Metrics
    base_metrics = compute_metrics(base_ret)
    reg_metrics = compute_metrics(reg_ret)

    metrics_df = pd.DataFrame(
        [base_metrics, reg_metrics, ew_metrics],
        index=["Baseline", "Regime", "Equal Weight"]
    )
    
    print("\nPerformance Metrics:")
    print(metrics_df)
    
    return metrics_df 

if __name__ == "__main__":
    main()