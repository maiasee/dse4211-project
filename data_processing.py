import pandas as pd

# def load_data(file_path):
#     '''
#     Loads and processes data from all_forecasts.csv for portfolio construction

#     Params:
#         file_path (str): The path to the all_forecasts.csv file

#     Returns:
#         mu_baseline (pd.DataFrame): Expected returns (baseline model) by date and with columns for each cryptocurrency
#         mu_regime (pd.DataFrame): Expected returns (regime model) by date and with columns for each cryptocurrency
#         real_returns (pd.DataFrame): Realised returns by date and with columns for each cryptocurrency
#     '''

#     df = pd.read_csv(file_path)

#     # convert data to date format
#     df["date"] = pd.to_datetime(df["date"])

#     # filter for test set 
#     df_test = df[df["split"]=="test"]

#     # separate baseline and regime model forecasts
#     test_baseline = df_test[df_test["model_type"] == "baseline"]
#     test_regime = df_test[df_test["model_type"] == "regime"]

#     # pivot data into matrix form (date x asset)
#     mu_baseline = test_baseline.pivot(index="date", columns="crypto", values = "y_pred")
#     mu_regime = test_regime.pivot(index="date", columns="crypto", values = "y_pred")
#     real_returns = test_baseline.pivot(index="date", columns="crypto", values = "y_true")

#     # sort indexes by date
#     mu_baseline = mu_baseline.sort_index()
#     mu_regime = mu_regime.sort_index()
#     real_returns = real_returns.sort_index()

#     return mu_baseline, mu_regime, real_returns

def load_data(hist_data_fp, forecast_fp):
    all_forecasts = pd.read_csv(forecast_fp, parse_dates=True)

    mu_baseline = all_forecasts[all_forecasts["model_type"] == "baseline"].pivot(index="date", columns="crypto", values="y_pred")
    mu_regime   = all_forecasts[all_forecasts["model_type"] == "regime"].pivot(index="date", columns="crypto", values="y_pred")

    df_hist = pd.read_csv(hist_data_fp, parse_dates=True)
    df_hist = df_hist[["date", "crypto", "log_return"]] # daily log returns

    # align dates and cryptos between forecasts and historical returns
    start_date = max(mu_baseline.index.min(), mu_regime.index.min(), df_hist["date"].min())
    end_date = min(mu_baseline.index.max(), mu_regime.index.max(), df_hist["date"].max())
    df_hist = df_hist[(df_hist["date"] >= start_date) & (df_hist["date"] <= end_date)]
    daily_log_returns = df_hist.pivot(index="date", columns="crypto", values="log_return")
    cumulative_returns = daily_log_returns.rolling(window=7).sum() # compute backward looking 7d cumulative returns 
    
    # keep only test dates for evaluation
    test_dates = sorted(all_forecasts.loc[all_forecasts["split"] == "test", "date"].unique())
    # take every 7th date to match weekly rebalancing
    rebalancing_dates = test_dates[::7]
    rebalancing_dates = pd.to_datetime(rebalancing_dates)
    return mu_baseline, mu_regime, daily_log_returns, cumulative_returns, rebalancing_dates