import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import pandas as pd
import json
import pickle
from tsbenchmark.data_process import GenerateAnomaly, PrepareDf
from tsbenchmark.metric import ProbTimeDetection, FalseRate
from canari import (
    DataProcess,
    SKF,
)
from tsbenchmark.detector import ProphetDetector, SkfDetector, TranAdDetector, LstmEdDetector

# -------------------------------------------------------------------#
# Load data
data_folder = "/Users/vuongdai/GitHub/bm/detrend_data/monthly/"

# Data file
data_file = "ts_monthly_values_standardize.csv"

# Time file
data_file_time = "ts_monthly_datetimes.csv"

sensor_names = pd.read_csv(data_folder + data_file, nrows=0, delimiter=",").columns.tolist()
df = pd.read_csv(data_folder + data_file, skiprows=1, delimiter=",", usecols=[0, 1, 2, 3], header=None)
df_time = pd.read_csv(data_folder + data_file_time, skiprows=1, delimiter=",", usecols=[0, 1, 2, 3], header=None)

df_dict = PrepareDf(df, df_time)

# data_file = "/Users/vuongdai/GitHub/canari/data/benchmark_data/test_2_data.csv"
# df = pd.read_csv(data_file, skiprows=0, delimiter=",")
# date_time = pd.to_datetime(df["date"])
# df = df.drop("date", axis=1)
# df.index = date_time
# df.index.name = "date_time"

# df_dict = {0: df}
# -------------------------------------------------------------------#

# -------------------------------------------------------------------#
# # Load anomaly info.
anomaly_info_file = "/Users/vuongdai/GitHub/bm/detrend_data/anomaly_info/anomaly_info.json"
with open(anomaly_info_file, "r") as f:
    anomaly_info = json.load(f)

# -------------------------------------------------------------------#
# # Loop over time series 
anom_result = {}

for ts, df_temp in df_dict.items():
    print("---------------------------")
    print("---------------------------")
    print(f"ts #{ts} ...... ")
    anom_result[ts] = {}

    # Split data into training and test sets; standardize data
    train_split = 0.5
    data_processor = DataProcess(
        data=df_temp,
        time_covariates=["week_of_year"],
        train_split=train_split,
        validation_split=0,
        test_split=1-train_split,
        output_col=[0],
    )
    _train_data, _, _test_data, _ = data_processor.get_splits()
    test_cov = _test_data["x"].copy()

    train_df = pd.DataFrame(
        data = _train_data["y"],
        index= _train_data["time"],
    )
    train_dict = {0: train_df}

    test_df = pd.DataFrame(
        data = _test_data["y"],
        index= _test_data["time"],
    )
    test_dict = {0: test_df}

    # # Generate anomalies
    start_anomaly_offset = 0.25
    test_data_with_anomaly = GenerateAnomaly(
        data=copy.deepcopy(test_dict),
        anomaly_info=anomaly_info,
        start_anomaly_offset=start_anomaly_offset,
    )

    # -------------------------------------------------------------------#
    # TSAD comparision

    # -------------------------------------------------------------------#
    # # 1. SKF detector
    anom_result[ts]["skf"] = {}
    anom_result[ts]["skf"]["false_rate"] = {}
    anom_result[ts]["skf"]["prob"] = {}
    anom_result[ts]["skf"]["ttd"] = {}

    skf_models_link = {}
    skf_models_link["seed_1"] = "/Users/vuongdai/GitHub/bm/results/BM2_g.pkl"
    skf_models_link["seed_2"] = "/Users/vuongdai/GitHub/bm/results/BM2_g_1.pkl"

    print("SKF detector ...... ")
    for i, link in skf_models_link.items():
        print(f"Seed #{i}")
        with open(link, "rb") as f:
            skf_dict = pickle.load(f)
        skf = SKF.load_dict(skf_dict)

        model = SkfDetector(anom_threshold=0.1, model=skf)

        score_test = model.get_anomaly_score(data=copy.deepcopy(test_dict), covariates=copy.deepcopy(test_cov))

        # score_test_with_anomaly = model.get_anomaly_score(data=test_data_with_anomaly.copy(), covariates=test_cov)

        false_rate = FalseRate(score_test)

        anom_result[ts]["skf"]["false_rate"][i] = false_rate[0]


    # -------------------------------------------------------------------#
    # # 2. Prophet detector
    print("Prophet detector ...... ")

    anom_result[ts]["prophet"] = {}
    anom_result[ts]["prophet"]["false_rate"] = {}
    anom_result[ts]["prophet"]["prob"] = {}
    anom_result[ts]["prophet"]["ttd"] = {}

    model = ProphetDetector(anom_threshold=0.1)

    score_test = model.get_anomaly_score(data=copy.deepcopy(test_dict))

    score_test_with_anomaly = model.get_anomaly_score(data=copy.deepcopy(test_data_with_anomaly))

    false_rate = FalseRate(score_test)
    prob_detection, time_to_detection = ProbTimeDetection(score_test_with_anomaly, anomaly_info)

    anom_result[ts]["prophet"]["false_rate"] = false_rate[0]
    anom_result[ts]["prophet"]["prob"] = prob_detection[0]
    anom_result[ts]["prophet"]["ttd"] = time_to_detection[0]

    # -------------------------------------------------------------------#
    # # 3. LSTM-ED detector
    print("LSTM-ED detector ...... ")

    anom_result[ts]["lstmed"] = {}
    anom_result[ts]["lstmed"]["false_rate"] = {}
    anom_result[ts]["lstmed"]["prob"] = {}
    anom_result[ts]["lstmed"]["ttd"] = {}

    for i in ["seed_1","seed_2"]:
        print(f"Seed #{i}")
        model = LstmEdDetector(sequence_len=52)

        model.train(data=train_df)

        score_test = model.get_anomaly_score(data=copy.deepcopy(test_dict))

        score_test_with_anomaly = model.get_anomaly_score(data=copy.deepcopy(test_data_with_anomaly))

        false_rate = FalseRate(score_test)
        prob_detection, time_to_detection = ProbTimeDetection(score_test_with_anomaly, anomaly_info)

        anom_result[ts]["lstmed"]["false_rate"][i] = false_rate[0]
        anom_result[ts]["lstmed"]["prob"][i] = prob_detection[0]
        anom_result[ts]["lstmed"]["ttd"][i] = time_to_detection[0]

    # -------------------------------------------------------------------#
    # # 4. TranAD detector
    print("TranAD detector ...... ")

    anom_result[ts]["tranad"] = {}
    anom_result[ts]["tranad"]["false_rate"] = {}
    anom_result[ts]["tranad"]["prob"] = {}
    anom_result[ts]["tranad"]["ttd"] = {}

    for i in ["seed_1", "seed_2"]:
        print(f"Seed #{i}")
        model = TranAdDetector(num_epoch=5)

        model.train(data=train_df)

        score_test = model.get_anomaly_score(data=copy.deepcopy(test_dict))

        score_test_with_anomaly = model.get_anomaly_score(data=copy.deepcopy(test_data_with_anomaly))

        false_rate = FalseRate(score_test)
        prob_detection, time_to_detection = ProbTimeDetection(score_test_with_anomaly, anomaly_info)

        anom_result[ts]["tranad"]["false_rate"][i] = false_rate[0]
        anom_result[ts]["tranad"]["prob"][i] = prob_detection[0]
        anom_result[ts]["tranad"]["ttd"][i] = time_to_detection[0]

    
    check = 1

    




    







    



    


