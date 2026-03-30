import pandas as pd
import matplotlib.pyplot as plt

from data_processing import load_data
from backtest import run_backtest


def compute_metrics(returns):
    """
    Compute basic performance metrics.
    """
    total_return = (1 + returns).prod() - 1
    # volatility = returns.std() * (252 ** 0.5)  # annualised
    # sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)
    
    # EDIT — annualise from weekly (52 periods/year)
    volatility = returns.std() * (52 ** 0.5)
    sharpe = (returns.mean() / returns.std()) * (52 ** 0.5)

    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }


def main():

    # load data
    # mu_base, mu_regime, returns = load_data("outputs_draft/all_forecasts.csv")
    # EDIT — updated load_data to also return test_dates, which we can use to filter the backtest evaluation
    
    mu_base, mu_regime, returns, test_dates = load_data("outputs_draft/all_forecasts.csv")
    # mu_base, mu_regime, returns, test_dates = load_data("all_forecasts_new.csv")

    print("--- Data Shape Diagnostic ---")
    print(f"mu_base shape: {mu_base.shape}")
    print(f"returns shape: {returns.shape}")
    print(f"date range: {returns.index.min()} to {returns.index.max()}")
    print(f"number of test dates: {len(test_dates)}")
    print(f"first test date: {min(test_dates)}")
    print(f"rows before first test date: {(returns.index < min(test_dates)).sum()}")
    print(f"window * 7 required: {20 * 7}")

    # run backtests
    # base_ret, base_w = run_backtest(mu_base, returns)
    # reg_ret, reg_w   = run_backtest(mu_regime, returns)
    # EDIT - run backtests with test_dates
    base_ret, base_w = run_backtest(mu_base, returns, eval_dates=test_dates)
    reg_ret, reg_w   = run_backtest(mu_regime, returns, eval_dates=test_dates)
    
    print("--- Check Weekly Structure ---")
    print(f"Expected for weekly over 1 year: ~52")
    print(f"Date gaps between observations:")
    print(pd.Series(base_ret.index).diff().value_counts().head())

    # compute cumulative returns
    cum_base = (1 + base_ret).cumprod()
    cum_reg  = (1 + reg_ret).cumprod()

    # plot results
    plt.figure(figsize=(10, 6))
    plt.plot(cum_base, label="Baseline Portfolio")
    plt.plot(cum_reg, label="Regime Portfolio")
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig("figures/cumulative_returns.png")
    plt.close()

    # cogit mpute metrics
    base_metrics = compute_metrics(base_ret)
    reg_metrics  = compute_metrics(reg_ret)

    metrics_df = pd.DataFrame([base_metrics, reg_metrics],
                              index=["Baseline", "Regime"])

    print("\nPerformance Metrics:")
    
    print(metrics_df)

    print("---Debugging---")
    print("Returns:")
    print(returns.describe())
    print("Mu Base:")
    print(mu_base.describe())
    print("Mu Regime:")
    print(mu_regime.describe())
    print("Base Weights:")
    print(base_w.describe())
    print("Null Values in Returns:")
    print(returns.isnull().sum().sum())

if __name__ == "__main__":
    main()