from papermill import execute_notebook
import subprocess

execute_notebook(
    '01_binance_data_cleaning.ipynb',
    '01_binance_data_cleaning_out.ipynb'
)

execute_notebook(
    '02_lstm_preprocessing_pipeline.ipynb',
    '02_lstm_preprocessing_pipeline_out.ipynb'
)

execute_notebook(
    '03_lstm_training.ipynb',
    '03_lstm_training_out.ipynb'
)

subprocess.run(['python', '04_run_portfolio.py'], check=True)