from matplotlib import dates
import numpy as np
import pandas as pd
from mvo import optimize_portfolio

# def run_backtest(mu, real_returns, window = 20, lamda = 5.0):
#     '''
#     Run rolling backtest of MVO portfolio and rebalancing
    
#     Params:
#         mu (pd.Dataframe): Expected returns (date x asset)
#         real_returns (pd.DataFrame): Actual returns (date x asset)
#         window (int): Rolling window size for covariance
#         lamda (float): Risk aversion parameter for MVO

#     Returns:
#         portfolio_returns (pd.Series): Portfolio returns over time 
#         weights_history (pd.DataFrame): Portfolio weights over time
#     '''

#     dates = mu.index
#     assets = mu.columns

#     portfolio_returns = []
#     weights_list = []
#     weights_dates = []

#     # # loop through time
#     # for t in range(window, len(dates)):
#     # EDIT — weekly loop
#     for t in range(window, len(dates) - 7, 7):
        
#         # expected returns at time t
#         mu_t = mu.iloc[t].values

#         # # historical returns for cov
#         # hist_returns = real_returns.iloc[t-window:t].values

#         # # skip insufficient data
#         # if hist_returns.shape[0] < window:
#         #     continue

#         # cov_t = np.cov(hist_returns, rowvar=False)
#         # EDIT — Use non-overlapping 7-day returns:
#         hist_7d = real_returns.iloc[t-window*7:t].iloc[::7].values  # every 7th row
#         # guard: insufficient history
#         if hist_7d.shape[0] < window:
#             print(f"Warning: insufficient history for cov computationat t={t}, skipping")
#             continue
#         cov_t = np.cov(hist_7d, rowvar=False)
#         # guard: degenerate covariance
#         if np.any(np.isnan(cov_t)) or np.any(np.isinf(cov_t)):
#             print(f"Warning: degenerate cov at t={t}, skipping")
#             continue

#         # optimise weights 
#         w_t = optimize_portfolio(mu_t, cov_t, lamda)

#         # apply weights to next period
#         # r_next = real_returns.iloc[t-1].values
#         # EDIT — apply weights to the 7-day return starting at t
#         r_next = real_returns.iloc[t].values
#         port_ret = np.dot(w_t, r_next)

#         # store 
#         portfolio_returns.append(port_ret)
#         weights_list.append(w_t)
#         weights_dates.append(dates[t])

#     # convert to pandas
#     portfolio_returns = pd.Series(portfolio_returns, index=weights_dates)
#     weights_history = pd.DataFrame(weights_list, index=weights_dates, columns=assets)

#     return portfolio_returns, weights_history

def run_backtest(mu, real_returns, window=20, lamda=5.0, eval_dates=None, max_weight=0.3):
    portfolio_returns = []
    weights_history = []

    dates = mu.index
    assets = mu.columns

    if eval_dates is None or len(eval_dates) == 0:
        raise ValueError("eval_dates must be provided and cannot be empty")

    eval_dates = pd.to_datetime(eval_dates)

    missing = set(eval_dates) - set(mu.index)
    if missing:
        raise ValueError(f"Some eval_dates not found in mu index: {sorted(list(missing))[:5]}")

    for current_date in sorted(eval_dates):
        t = dates.get_loc(current_date)
        
        # need window*7 rows of history before t
        if t < window * 7:
            print(f"Insufficient history at {current_date}, skipping")
            continue
        
        hist_7d = real_returns.iloc[t - window*7 : t].iloc[::7].values
        if hist_7d.shape[0] < window:
            continue
        
        mu_t = mu.iloc[t].values
        cov_t = np.cov(hist_7d.T)
        
        if np.isnan(mu_t).any() or np.isnan(cov_t).any() or np.isinf(cov_t).any():
            continue
        
        cov_t = cov_t + 1e-6 * np.eye(cov_t.shape[0])  # regularisation
        w_t = optimize_portfolio(mu_t, cov_t, lamda=lamda, max_weight=max_weight)
        
        r_next = real_returns.loc[current_date].values
        if np.isnan(r_next).any():
            continue
        
        port_ret = np.dot(w_t, r_next)
        portfolio_returns.append((current_date, port_ret))
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