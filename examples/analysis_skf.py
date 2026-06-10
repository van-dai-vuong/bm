import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import json
from src.tsbenchmark.data_process import GenerateAnomaly, PrepareDf
from src.tsbenchmark.metric import ProbTimeDetection, FalseRate
import pickle

from canari import (
    DataProcess,
    Model,
    SKF,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise

# Data folder 
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


def _anomaly_score(data, covariates, detector, threshold):
    """Run anomaly detection on a single pandas Series."""
    _data = {
        "y": data.values,
        "x": covariates,
        "time": data.index,
    }
    scores, _ = detector.filter(data=_data)
    return pd.DataFrame({"value": scores > threshold}, index=data.index)


def _apply_recursive(data, covariates, detector, threshold):
    """Walk a nested dict of any depth; apply detection at the leaf (a Series/DataFrame)."""
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return _anomaly_score(data, covariates, detector, threshold)
    return {
        key: _apply_recursive(value, covariates, detector, threshold)
        for key, value in data.items()
    }


def detect_anomaly(data, covariates, detector, threshold=0.1):
    return _apply_recursive(data, covariates, detector, threshold)

for ts, df_temp in df_dict.items(): 
    # Split df into training and test sets
    train_test_split = 0.5
    train_size = int(len(df_temp) * train_test_split)
    _df_train = df_temp.iloc[:train_size, :]
    _df_test  = df_temp.iloc[train_size:, :]

    df_test = {0: _df_test}

    data_processor = DataProcess(
        data=df_temp,
        time_covariates=["week_of_year"],
        train_split=train_test_split,
        validation_split=0,
        test_split=1-train_test_split,
        output_col=[0],
    )
    _, _, test_data, _ = data_processor.get_splits()

    # Generate anomalies
    start_anomaly_offset = 0.25
    df_with_anomaly = GenerateAnomaly(
        data=df_test,
        anomaly_info=anomaly_info,
        start_anomaly_offset=start_anomaly_offset,
    )

    models_link = {}
    models_link["seed_1"] = "/Users/vuongdai/GitHub/bm/results/BM2_g.pkl"
    models_link["seed_2"] = "/Users/vuongdai/GitHub/bm/results/BM2_g_1.pkl"

    anom_score = {}

    for i, link in models_link.items():
        with open(link, "rb") as f:
            skf_dict = pickle.load(f)
        skf = SKF.load_dict(skf_dict)

        anom_score[i] = detect_anomaly(
            data = df_with_anomaly,
            covariates = test_data["x"],
            detector=skf,
            threshold=0.1,
            )
        
    prob_detection = {}
    time_to_detection = {}
    for i, link in models_link.items():
        prob_detection[i], time_to_detection[i] = ProbTimeDetection(anom_score[i], anomaly_info)

    check = 1

        

