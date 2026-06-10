import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import json
from src.data_process import GenerateAnomaly, PrepareDf
from src.metric import ProbTimeDetection, FalseRate
import pickle
from src.detector.lstmed_detector import LstmEdDetector

# # Data folder 
# data_folder = "/Users/vuongdai/GitHub/bm/detrend_data/monthly/"

# # Data file
# data_file = "ts_monthly_values_standardize.csv"

# # Time file
# data_file_time = "ts_monthly_datetimes.csv"

# sensor_names = pd.read_csv(data_folder + data_file, nrows=0, delimiter=",").columns.tolist()
# df = pd.read_csv(data_folder + data_file, skiprows=1, delimiter=",", usecols=[0, 1], header=None)
# df_time = pd.read_csv(data_folder + data_file_time, skiprows=1, delimiter=",", usecols=[0, 1], header=None)

# df_dict = PrepareDf(df, df_time)

data_file = "/Users/vuongdai/GitHub/canari/data/benchmark_data/test_2_data.csv"
df = pd.read_csv(data_file, skiprows=0, delimiter=",")
date_time = pd.to_datetime(df["date"])
df = df.drop("date", axis=1)
df.index = date_time
df.index.name = "date_time"

df_dict = {0: df}

# Anomaly info.
anomaly_info_file = "/Users/vuongdai/GitHub/bm/detrend_data/anomaly_info/anomaly_info.json"
with open(anomaly_info_file, "r") as f:
    anomaly_info = json.load(f)

# score_train = {}
for ts, df_temp in df_dict.items(): 
    # Split df into training and test sets
    train_test_split = 0.5
    train_size = int(len(df_temp) * train_test_split)
    _df_train = df_temp.iloc[:train_size, :]
    _df_test  = df_temp.iloc[train_size:, :]

    df_test = {0: _df_test}
    df_train = {0: _df_train}

    # Generate anomalies
    start_anomaly_offset = 0.25
    df_with_anomaly = GenerateAnomaly(
        data=df_test,
        anomaly_info=anomaly_info,
        start_anomaly_offset=start_anomaly_offset,
    )

    model = LstmEdDetector(sequence_len=52)
    model.train(data=_df_train)
    test_score = model.get_anomaly_score(data = df_with_anomaly)

    prob_detection, time_to_detection = ProbTimeDetection(test_score, anomaly_info)

    check  = 1




        

