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

    # Annualized volatility and Sharpe ratio (assuming 52 wks in a year)
    volatility = log_returns.std() * (52 ** 0.5)
    sharpe = np.nan
    if log_returns.std() != 0:
        sharpe = (log_returns.mean() / log_returns.std()) * (52 ** 0.5)

    cumulative = np.exp(log_returns.cumsum())
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    positive_return_rate = (log_returns > 0).mean()

    return {
        "Total Return": total_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Positive Return Rate": positive_return_rate,
    }

def run_buy_and_hold_backtest(daily_log_returns, evaluation_dates):
    daily_log_returns = daily_log_returns.copy()
    daily_log_returns.index = pd.to_datetime(daily_log_returns.index)
    daily_log_returns = daily_log_returns.sort_index()

    evaluation_dates = pd.to_datetime(evaluation_dates)
    start_date = evaluation_dates.min()
    end_date = evaluation_dates.max()

    daily_log_returns = daily_log_returns.loc[
        (daily_log_returns.index >= start_date) & (daily_log_returns.index <= end_date)
    ]

    if daily_log_returns.empty:
        raise ValueError("No returns available for the requested evaluation period.")

    n_assets = daily_log_returns.shape[1]
    w0 = np.ones(n_assets) / n_assets

    daily_simple_returns = np.exp(daily_log_returns) - 1

    asset_values = pd.DataFrame(
        index=daily_simple_returns.index,
        columns=daily_simple_returns.columns,
        dtype=float
    )
    asset_values.iloc[0] = w0 * 1.0

    for t in range(1, len(daily_simple_returns)):
        asset_values.iloc[t] = asset_values.iloc[t - 1] * (1 + daily_simple_returns.iloc[t])

    portfolio_value_daily = asset_values.sum(axis=1)

    # align to exact backtest dates
    portfolio_value = portfolio_value_daily.reindex(evaluation_dates, method="ffill").dropna()

    # renormalise to start at 1
    portfolio_value = portfolio_value / portfolio_value.iloc[0]

    portfolio_log_returns = np.log(portfolio_value / portfolio_value.shift(1)).dropna()

    return portfolio_log_returns, portfolio_value

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

    # === Run backtests
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

    common_dates = base_ret.index.intersection(reg_ret.index)

    ew_ret, cum_ew = run_buy_and_hold_backtest(
        daily_log_returns,
        evaluation_dates=common_dates
    )   

    # print("Check Weekly Structure:")
    # print("Expected for weekly over 1 year: ~52")

    # Cumulative portfolio value from weekly LOG returns
    cum_base_raw = np.exp(base_ret.cumsum())
    cum_reg_raw = np.exp(reg_ret.cumsum())

    cum_base = cum_base_raw / cum_base_raw.iloc[0]
    cum_reg = cum_reg_raw / cum_reg_raw.iloc[0]

    # renormalise so the first plotted value starts at 1
    cum_base = cum_base / cum_base.iloc[0]
    cum_reg = cum_reg / cum_reg.iloc[0]
    ew_metrics = compute_metrics(ew_ret)

    # === Cumulative Portfolio Value Comparison Plot
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(cum_base.index), cum_base.values, label="Baseline Portfolio")
    plt.plot(pd.to_datetime(cum_reg.index), cum_reg.values, label="Regime Portfolio")
    plt.plot(pd.to_datetime(cum_ew.index), cum_ew.values, label="Buy and Hold")
    plt.title("Cumulative Portfolio Value")
    plt.xlabel("Date")
    # plt.ylabel("Portfolio Value")
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
        index=["Baseline", "Regime", "Buy and Hold"]
    )
    
    print("\nPerformance Metrics:")
    print(metrics_df)
    
    # === Save metrics as png
    # Add final portfolio values to metrics
    base_metrics["Final Portfolio Value"] = cum_base.iloc[-1] if len(cum_base) > 0 else np.nan
    reg_metrics["Final Portfolio Value"] = cum_reg.iloc[-1] if len(cum_reg) > 0 else np.nan
    ew_metrics["Final Portfolio Value"] = cum_ew.iloc[-1] if len(cum_ew) > 0 else np.nan

    # Metrics table
    metrics_df = pd.DataFrame({
        "Baseline": base_metrics,
        "Regime": reg_metrics,
        "Buy and Hold": ew_metrics
    })

    # Save table as image
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis("off")

    display_df = metrics_df.copy().map(
    lambda x: f"{x:.4f}" if pd.notnull(x) and isinstance(x, (int, float, np.floating)) else x
    )

    table = ax.table(
        cellText=display_df.values,
        rowLabels=display_df.index,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center"
    )

    # Loop through rows and bold best values
    for i, row_name in enumerate(metrics_df.index):
        row = metrics_df.loc[row_name]

        if row_name == "Volatility":
            best_col = row.idxmin()
        else:
            best_col = row.idxmax()

        col_idx = list(metrics_df.columns).index(best_col)

        # +1 because row 0 is header in matplotlib table
        table[(i + 1, col_idx)].set_text_props(weight='bold')

        table.auto_set_font_size(True)
        # table.set_fontsize(10)
        table.scale(1.1, 1.4)

    plt.tight_layout()
    plt.savefig("outputs/figures/performance_metrics_table.png", dpi=300, bbox_inches="tight")
    plt.close()

    return metrics_df 

if __name__ == "__main__":
    main()