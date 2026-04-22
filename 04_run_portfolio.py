import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from functions.data_processing import load_data
from functions.backtest import run_backtest
from functions.simple_mvo_backtest import run_backtest_historical_mu

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
    positive_rate = (log_returns > 0).mean()

    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Volatility": volatility,
        "Max Drawdown": max_drawdown,
        "Positive Rate": positive_rate
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

    print("\n=== Running Portfolio Backtests ===")
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

    ### Equal Weight Backtest ###
    # ew_ret = run_equal_weight_backtest(daily_log_returns, rebalancing_dates)
    # cum_ew = np.exp(ew_ret.cumsum())
    
    ### Simple MVO Backtest ###
    hist_ret, hist_w = run_backtest_historical_mu(
    daily_log_returns,
    cumulative_returns,
    rebalancing_dates,
    window=20,
    lamda=5.0
    )

    # === Cumulative portfolio value from weekly LOG returns
    cum_base = np.exp(base_ret.cumsum())
    cum_reg = np.exp(reg_ret.cumsum())
    # ew_metrics = compute_metrics(ew_ret)
    cum_hist = np.exp(hist_ret.cumsum())

    # === Cumulative Portfolio Value Comparison Plot
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(cum_base.index), cum_base.values, label="Baseline Portfolio")
    plt.plot(pd.to_datetime(cum_reg.index), cum_reg.values, label="Regime Portfolio")
    # plt.plot(pd.to_datetime(cum_ew.index), cum_ew.values, label="Equal Weight")
    plt.plot(pd.to_datetime(cum_hist.index), cum_hist.values, label="Simple MVO")
    plt.title("Cumulative Portfolio Value")
    plt.xlabel("Date")
    # plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/figures/cum_returns_with_simple_mvo.png")
    plt.close()
    print(f"Saved cumulative returns plot to outputs/figures/cum_returns_with_simple_mvo.png")

    # === Metrics
    base_metrics = compute_metrics(base_ret)
    reg_metrics = compute_metrics(reg_ret)
    # ew_metrics = compute_metrics(ew_ret)
    hist_metrics = compute_metrics(hist_ret) # simple MVO

    base_metrics["Final Portfolio Value"] = cum_base.iloc[-1] if len(cum_base) > 0 else np.nan
    reg_metrics["Final Portfolio Value"] = cum_reg.iloc[-1] if len(cum_reg) > 0 else np.nan
    hist_metrics["Final Portfolio Value"] = cum_hist.iloc[-1] if len(cum_hist) > 0 else np.nan

    # Metrics table
    # metrics_df = pd.DataFrame(
    #     [base_metrics, reg_metrics, ew_metrics],
    #     index=["Baseline", "Regime", "Equal Weight"]
    # )
    metrics_df = pd.DataFrame(
    [base_metrics, reg_metrics, hist_metrics],
    index=["Baseline", "Regime", "Simple MVO"]
)

    os.makedirs("outputs/portfolio_results", exist_ok=True)
    print("\nPerformance Metrics:")
    print(metrics_df)
    metrics_df.to_csv("outputs/portfolio_results/performance_metrics_with_simple_mvo.csv")
    print(f"Saved performance metrics csv to outputs/portfolio_results/performance_metrics_with_simple_mvo.csv")

    # Add final portfolio values to metrics
    base_metrics["Final Portfolio Value"] = cum_base.iloc[-1] if len(cum_base) > 0 else np.nan
    reg_metrics["Final Portfolio Value"] = cum_reg.iloc[-1] if len(cum_reg) > 0 else np.nan
    # ew_metrics["Final Portfolio Value"] = cum_ew.iloc[-1] if len(cum_ew) > 0 else np.nan
    hist_metrics["Final Portfolio Value"] = cum_hist.iloc[-1] if len(cum_hist) > 0 else np.nan

    # Save metrics table as image
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
    table.auto_set_font_size(True)
    table.scale(1.1, 1.8)  

    # Loop through rows and bold best values
    # metrics where lower is better
    lower_is_better = {"Volatility"}

    for j, col_name in enumerate(metrics_df.columns):
        col = metrics_df[col_name]
        
        if col_name in lower_is_better:
            best_idx = col.idxmin()
        else:
            best_idx = col.idxmax()
        
        row_idx = list(metrics_df.index).index(best_idx)
        table[(row_idx + 1, j)].set_text_props(weight='bold')


    plt.tight_layout()
    plt.savefig("outputs/figures/performance_metrics_table_with_simple_mvo.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved performance metrics table as png to outputs/figures/performance_metrics_table_with_simple_mvo.png")

    # === Portfolio weight plots
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/figures/coin_weight_panels", exist_ok=True)

    base_w.index = pd.to_datetime(base_w.index)
    reg_w.index = pd.to_datetime(reg_w.index)
    hist_w.index = pd.to_datetime(hist_w.index)

    colors = ['#E86C2F', '#5BC8D4', '#7B5EA7', '#F5A623', '#2E7D8C',
            '#A0522D', '#C8A882', '#4CAF7D']

    model_weights = {
        "LSTM No Regime": base_w,
        "LSTM Regime": reg_w,
        "Simple MVO": hist_w
    }

    # Combined stacked area figure for all three models
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    for ax, (model_name, weight_df) in zip(axes, model_weights.items()):
        ax.stackplot(
            weight_df.index,
            [weight_df[col].values for col in weight_df.columns],
            labels=weight_df.columns,
            colors=colors[:len(weight_df.columns)],
            alpha=0.9
        )
        ax.set_title(f"{model_name} Portfolio Weights Over Time")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig("outputs/figures/portfolio_weights_all_models.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved combined stacked weights plot to outputs/figures/portfolio_weights_all_models.png")

    # === One figure per coin, with 3 subplots (one for each model)
    for coin in base_w.columns:
        safe_coin = str(coin).lower().replace("/", "_").replace(" ", "_")

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        for ax, (model_name, weight_df) in zip(axes, model_weights.items()):
            ax.plot(weight_df.index, weight_df[coin], linewidth=2)
            ax.set_title(f"{coin} Weight - {model_name}")
            ax.set_ylabel("Weight")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date")
        plt.tight_layout()
        out_fp = f"outputs/figures/coin_weight_panels/{safe_coin}_weights_all_models.png"
        plt.savefig(out_fp, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_fp}")

    # === Save portfolio weights as CSV
    base_w.to_csv("outputs/portfolio_results/weights_baseline.csv")
    reg_w.to_csv("outputs/portfolio_results/weights_regime.csv")
    hist_w.to_csv("outputs/portfolio_results/weights_simple_mvo.csv")
    ew_w = pd.DataFrame(
        np.tile(1 / len(daily_log_returns.columns), (len(rebalancing_dates), len(daily_log_returns.columns))),
        index=rebalancing_dates,
        columns=daily_log_returns.columns
    )
    ew_w.to_csv("outputs/portfolio_results/weights_equal_weight.csv")
    print("Saved portfolio weights to outputs/portfolio_results/")

    return metrics_df

if __name__ == "__main__":
    main()