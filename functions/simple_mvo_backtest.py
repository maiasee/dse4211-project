from matplotlib import dates
import numpy as np
import pandas as pd
from functions.mvo import optimize_portfolio

def run_backtest_historical_mu(daily_log_returns, cumulative_returns, rebalancing_dates, window=20, lamda=5.0):
    """
    Simple MVO baseline: mu_t estimated as the (rolling) mean of the same
    historical window used to compute cov_t = last 20 wks.
    Everything else is identical to run_backtest().
    """
    portfolio_returns = []
    weights_history = []

    assets = daily_log_returns.columns

    for _ in sorted(rebalancing_dates):
        current_date = _.strftime('%Y-%m-%d')
        t_idx = cumulative_returns.index.get_loc(current_date)
        # print(f"Backtest (historical mu) at {current_date} (index {t_idx})")

        hist_7d = cumulative_returns.iloc[t_idx - (window - 1) * 7 : t_idx + 1 : 7]  # shape (window, n_assets)
        cov_t = hist_7d.cov().values
        mu_t = hist_7d.mean().values  # <-- historical mean of 7d cumulative returns

        w_t = optimize_portfolio(mu_t, cov_t, lamda=lamda)
        # print(f"  weights: {w_t}")

        realised_7d = daily_log_returns.iloc[t_idx + 1 : t_idx + 8].sum(axis=0).values
        portfolio_return = np.dot(w_t, realised_7d)
        # print(f"  portfolio return: {portfolio_return}")

        portfolio_returns.append((current_date, portfolio_return))
        weights_history.append((current_date, w_t))

    portfolio_returns = pd.Series({date: ret for date, ret in portfolio_returns})
    weights_history = pd.DataFrame(
        [w for _, w in weights_history],
        index=[d for d, _ in weights_history],
        columns=assets
    )

    return portfolio_returns, weights_history