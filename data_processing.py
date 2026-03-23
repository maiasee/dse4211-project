import pandas as pd

def load_data(file_path):
    '''
    Loads and processes data from all_forecasts.csv for portfolio construction

    Params:
        file_path (str): The path to the all_forecasts.csv file

    Returns:
        mu_baseline (pd.DataFrame): Expected returns (baseline model) by date and with columns for each cryptocurrency
        mu_regime (pd.DataFrame): Expected returns (regime model) by date and with columns for each cryptocurrency
        real_returns (pd.DataFrame): Realised returns by date and with columns for each cryptocurrency
    '''

    df = pd.read_csv(file_path)

    # convert data to date format
    df["date"] = pd.to_datetime(df["date"])

    # filter for test set 
    df_test = df[df["split"]=="test"]

    # separate baseline and regime model forecasts
    test_baseline = df_test[df_test["model_type"] == "baseline"]
    test_regime = df_test[df_test["model_type"] == "regime"]

    # pivot data into matrix form (date x asset)
    mu_baseline = test_baseline.pivot(index="date", columns="crypto", values = "y_pred")
    mu_regime = test_regime.pivot(index="date", columns="crypto", values = "y_pred")
    real_returns = test_baseline.pivot(index="date", columns="crypto", values = "y_true")

    # sort indexes by date
    mu_baseline = mu_baseline.sort_index()
    mu_regime = mu_regime.sort_index()
    real_returns = real_returns.sort_index()

    return mu_baseline, mu_regime, real_returns