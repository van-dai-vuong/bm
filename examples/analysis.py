import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import json
from src.data_process import GenerateAnomaly, PrepareDf
from src.metric import ProbTimeDetection, FalseRate

from canari import (
    DataProcess,
    Model,
    Optimizer,
    SKF,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise

# Data folder 
data_folder = "/Users/vuongdai/GitHub/bm/detrend_data/monthly/"

# Data file
data_file = "ts_monthly_values_standardize.csv"

# Time file
data_file_time = "ts_monthly_datetimes.csv"

sensor_names = pd.read_csv(data_folder + data_file, nrows=0, delimiter=",").columns.tolist()
df = pd.read_csv(data_folder + data_file, skiprows=1, delimiter=",", header=None)
df_time = pd.read_csv(data_folder + data_file_time, skiprows=1, delimiter=",", header=None)

df_dict = PrepareDf(df, df_time)

# Anomaly info.
anomaly_info_file = "/Users/vuongdai/GitHub/bm/detrend_data/anomaly_info/anomaly_info.json"
with open(anomaly_info_file, "r") as f:
    anomaly_info = json.load(f)

# Generate anomalies
df_with_anomaly = GenerateAnomaly(data=df_dict, anomaly_info=anomaly_info, col=[0,1])

# SKF model
sigma_v = 5e-2
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=19,
    num_features=1,
    num_layer=1,
    num_hidden_unit=50,
    manual_seed=1,
    smoother=False,
)
noise = WhiteNoise(std_error=sigma_v)

# Normal model
model = Model(
    local_trend,
    lstm_network,
    noise,
)

#  Abnormal model
ab_model = Model(
    local_acceleration,
    lstm_network,
    noise,
)

output_col = [0]
df_temp = df.iloc[:,[1]]
data_processor = DataProcess(
    data=df_temp,
    # time_covariates=["hour_of_day"],
    train_split=0.4,
    validation_split=0.1,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Switching Kalman filter
skf = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-5,
)
skf.auto_initialize_baseline_states(train_data["y"][0:24])

for epoch in range(1):
    skf.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )

def detect_synthetic_anomaly(
    data,
    detector,
    threshold: float = 0.1,
):
    anomaly_score = {}
    for ts in data:
        anomaly_score[ts] = {}
        for anomaly_type in data[ts]:
            anomaly_score[ts][anomaly_type] = {}
            for i in data[ts][anomaly_type]:
                anomaly_score[ts][anomaly_type][i] = {}
                for j in data[ts][anomaly_type][i]:
                    _data = {}
                    _data["y"] = data[ts][anomaly_type][i][j].values
                    _data["x"] = [[] for _ in range(len(_data["y"]))]
                    _data["time"] = data[ts][anomaly_type][i][j].index

                    _anomaly_score, _ = detector.filter(data=_data)
                    
                    _anomaly_score = _anomaly_score > threshold
                    _df_score_temp = pd.DataFrame({"value": _anomaly_score}, index=_data["time"])
                    anomaly_score[ts][anomaly_type][i][j] = _df_score_temp

    return anomaly_score

def detect_anomaly(
    data,
    detector,
    threshold: float = 0.1,
):
    anomaly_score = {}
    for ts in data:
        _df = data[ts] 
        _data = {}
        _data["y"] = _df.values
        _data["x"] = [[] for _ in range(len(_data["y"]))]
        _data["time"] = _df.index
        _anomaly_score, _ = detector.filter(data=_data)
        _anomaly_score = _anomaly_score > threshold
        anomaly_score[ts] = pd.DataFrame({"value": _anomaly_score}, index=_data["time"])

    return anomaly_score

# Estimate anomaly score
anomaly_score_synthetic = detect_synthetic_anomaly(detector=skf, data=df_with_anomaly)
anomaly_score_test_set = detect_anomaly(detector=skf, data=df_dict)

prob, time = ProbTimeDetection(anomaly_score_synthetic, anomaly_info)
false_rate = FalseRate(anomaly_score_test_set)