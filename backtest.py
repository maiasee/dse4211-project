import numpy as np
import pandas as pd
from mvo import optimize_portfolio

def run_backtest(mu, real_returns, window = 20, lamda = 5.0):
    '''
    Run rolling backtest of MVO portfolio and rebalancing
    
    Params:
        mu (pd.Dataframe): Expected returns (date x asset)
        real_returns (pd.DataFrame): Actual returns (date x asset)
        window (int): Rolling window size for covariance
        lamda (float): Risk aversion parameter for MVO

    Returns:
        portfolio_returns (pd.Series): Portfolio returns over time 
        weights_history (pd.DataFrame): Portfolio weights over time
    '''

    dates = mu.index
    assets = mu.columns

    portfolio_returns = []
    weights_list = []
    weights_dates = []

    # loop through time
    for t in range(window, len(dates)):
        
        # expected returns at time t
        mu_t = mu.iloc[t].values

        # historical returns for cov
        hist_returns = real_returns.iloc[t-window:t].values

        # skip insufficient data
        if hist_returns.shape[0] < window:
            continue

        cov_t = np.cov(hist_returns, rowvar=False)

        # optimise weights 
        w_t = optimize_portfolio(mu_t, cov_t, lamda)

        # apply weights to next period
        r_next = real_returns.iloc[t-1].values
        port_ret = np.dot(w_t, r_next)

        # store 
        portfolio_returns.append(port_ret)
        weights_list.append(w_t)
        weights_dates.append(dates[t])

    # convert to pandas
    portfolio_returns = pd.Series(portfolio_returns, index=weights_dates)
    weights_history = pd.DataFrame(weights_list, index=weights_dates, columns=assets)

    return portfolio_returns, weights_history