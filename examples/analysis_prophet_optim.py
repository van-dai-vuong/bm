import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import json
from src.tsbenchmark.data_process import GenerateAnomaly, PrepareDf
from src.tsbenchmark.metric import ProbTimeDetection, FalseRate
import pickle
from src.tsbenchmark.detector.phophet_detector import ProphetDetector
from canari import Optimizer
from ray import tune
import optuna
from ray.tune.search.optuna import OptunaSearch


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

    def detector_with_param(param):
        model = ProphetDetector(anom_threshold=param["threshold"])
        score_train = model.get_anomaly_score(data = df_train)
        false_rate = FalseRate(score_train)

        # tune.report({"metric": false_rate[0]})
        tune.report({"metric": false_rate[0]**2})
    
    # param_space = {"threshold": tune.grid_search([0.1, 0.2, 0.3])}
    # tuner = tune.Tuner(
    #     detector_with_param,
    #     param_space=param_space,
    #     tune_config=tune.TuneConfig(
    #         metric="metric",
    #         mode="min",
    #     ),
    # )

    param_space = {"threshold": tune.uniform(0.1, 0.9)}
    sampler = optuna.samplers.TPESampler()
    optuna_search = OptunaSearch(
        sampler=sampler,
        metric="metric",
        mode="min",
    )

    tuner = tune.Tuner(
        detector_with_param,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            metric="metric",
            mode="min",
            num_samples=20, 
        ),
    )

    results = tuner.fit()

    # Get best parameters
    best_result = results.get_best_result(metric="metric", mode="min")
    param_optim = best_result.config
    print("Best params:", param_optim)
    print("Best false_rate:", best_result.metrics["metric"])

    model_optim = ProphetDetector(anom_threshold=param_optim["threshold"])
    score_train = model_optim.get_anomaly_score(data = df_with_anomaly)

# print(score_train[0])





        

