import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from src.data_process import UpdateCsv

data_file = "/Users/vuongdai/Desktop/backup_canari/DATA_HQ_TDSB/GRAF-CASC/DAT/LTU/LTUBAR/EXT/LTU002ESAPRG920.DAT"
df_raw = pd.read_csv(data_file,
                     sep=";",  # Semicolon as delimiter
                     quotechar='"',
                     engine="python",
                     na_values=[""],  # Treat empty strings as NaN
                     skipinitialspace=True,
                     encoding="ISO-8859-1",
                     )

sensor_name = "LTU002ESAPRG920"
df = df_raw[["Ext/Contraction (mm)"]]
df.columns = ["y"]
df.index = pd.to_datetime(df_raw["Date"])
# Resampling to weekly
df = df.resample("W").last()

df_prophet = df.copy()
df_prophet = df_prophet.reset_index()
df_prophet.columns = ["ds","y"]
df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
df_prophet.index = df_prophet["ds"]

nan_idx = np.where(df_prophet["y"].isna())[0]
df_prophet["y"] = df_prophet["y"].interpolate()

# Run Prophet
prophet_model = Prophet()
prophet_model.fit(df_prophet)
phophet_results = prophet_model.predict(df_prophet[["ds"]])
 
trend = phophet_results["trend"]
detrend_data = df_prophet["y"].values - trend.values
detrend_data[nan_idx] = np.nan
df_prophet["y"] = detrend_data

df_detrend = df_prophet.copy()
df_detrend = df_detrend.drop(columns=["ds"], errors="ignore")
df_detrend.columns = [sensor_name]

UpdateCsv(
    df_detrend,
    values_path="/Users/vuongdai/GitHub/bm/detrend_data/weekly/week_values.csv",
    datetimes_path="/Users/vuongdai/GitHub/bm/detrend_data/weekly/week_datetimes.csv",
    rewrite=True,
    )