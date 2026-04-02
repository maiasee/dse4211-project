from matplotlib import dates
import numpy as np
import pandas as pd
from functions.mvo import optimize_portfolio  

def run_backtest(mu, daily_log_returns, cumulative_returns, rebalancing_dates, window=20, lamda=5.0):
    portfolio_returns = []
    weights_history = []

    dates = mu.index
    assets = mu.columns
    for _ in sorted(rebalancing_dates):
        current_date = _.strftime('%Y-%m-%d')
        t_idx = cumulative_returns.index.get_loc(_.strftime('%Y-%m-%d'))
        print(f"Backtest at {current_date} (index {t_idx})")
        hist_7d = cumulative_returns.iloc[t_idx - (window-1)*7 : t_idx + 1 : 7]
        cov_t = hist_7d.cov().values
        mu_t = mu.loc[current_date].values # should by 7-day cumulative return forecasts (expected returns for the next 7 days)

        w_t = optimize_portfolio(mu_t, cov_t, lamda=lamda)
        print(f"optimal weights at {current_date}: {w_t}")

        realised_7d = daily_log_returns.iloc[t_idx+1 : t_idx+8].sum(axis=0).values # actual return from holding period (next 7 days)
        portfolio_return = np.dot(w_t, realised_7d)
        print(f"portfolio return at {current_date}: {portfolio_return}")

        portfolio_returns.append((current_date, portfolio_return))
        weights_history.append((current_date, w_t))
    
    portfolio_returns = pd.Series(
        {date: ret for date, ret in portfolio_returns}
    )
    weights_history = pd.DataFrame(
        [w for _, w in weights_history],
        index=[d for d, _ in weights_history],
        columns=assets
    )

    return portfolio_returns, weights_history