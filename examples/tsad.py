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
# data_folder = "/Users/vuongdai/GitHub/bm/detrend_data/monthly/"

# # Data file
# data_file = "ts_monthly_values_standardize.csv"

# # Time file
# data_file_time = "ts_monthly_datetimes.csv"

# sensor_names = pd.read_csv(data_folder + data_file, nrows=0, delimiter=",").columns.tolist()
# df = pd.read_csv(data_folder + data_file, skiprows=1, delimiter=",", usecols=[0, 1, 2, 3], header=None)
# df_time = pd.read_csv(data_folder + data_file_time, skiprows=1, delimiter=",", usecols=[0, 1, 2, 3], header=None)

# df_dict = PrepareDf(df, df_time)

data_file = "/Users/vuongdai/GitHub/canari/data/toy_time_series/sine.csv"
df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "/Users/vuongdai/GitHub/canari/data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df.index = time_series
df.index.name = "date_time"
df.columns = ["values"]

df_dict = {0: df}
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
        time_covariates=["hour_of_day"],
        train_split=train_split,
        validation_split=0,
        test_split=1-train_split,
        output_col=[0],
    )
    _train_data, _, _test_data, all_data = data_processor.get_splits()

    # Training data
    train_df = pd.DataFrame(
        data = _train_data["y"],
        index= _train_data["time"],
    )
    train_dict = {0: train_df}

    # Test data: not include training data
    _test_df_no_anom = pd.DataFrame(
        data = _test_data["y"],
        index= _test_data["time"],
    )
    test_dict_no_anom = {0: _test_df_no_anom}

    # Test data: include training data
    _test_df_anom = pd.DataFrame(
        data = all_data["y"],
        index= all_data["time"],
    )
    test_dict_anom = {0: _test_df_anom}
    # # Generate anomalies
    anomaly_info["anomaly_start"] = 0.6
    anomaly_info["anomaly_end"] = 1.0

    test_data_with_anomaly, test_data_anom_info = GenerateAnomaly(
        data=copy.deepcopy(test_dict_anom),
        anomaly_info=anomaly_info,
    )

    # ----------------------------------------------
    # ---------------------#
    # TSAD comparision

    # -------------------------------------------------------------------#
    # # 1. SKF detector
    anom_result[ts]["skf"] = {}
    anom_result[ts]["skf"]["false_rate"] = {}
    anom_result[ts]["skf"]["prob"] = {}
    anom_result[ts]["skf"]["ttd"] = {}

    skf_models_link = {}
    skf_models_link["seed_1"] = "/Users/vuongdai/GitHub/canari/saved_params/toy_anomaly_detection_tune.pkl"
    # skf_models_link["seed_2"] = "/Users/vuongdai/GitHub/bm/results/BM2_g_1.pkl"

    print("SKF detector ...... ")
    for i, link in skf_models_link.items():
        print(f"    Seed #{i}")
        with open(link, "rb") as f:
            skf_dict = pickle.load(f)
        skf = SKF.load_dict(skf_dict)

        model = SkfDetector(anom_threshold=0.1, model=skf)

        test_cov_no_anom = _test_data["x"].copy()
        score_test = model.get_anomaly_score(data=copy.deepcopy(test_dict_no_anom), covariates=test_cov_no_anom)

        test_cov_anom = all_data["x"].copy()
        score_test_with_anomaly = model.get_anomaly_score(data=copy.deepcopy(test_data_with_anomaly), covariates=test_cov_anom)

        false_rate = FalseRate(score_test)
        prob_detection, time_to_detection = ProbTimeDetection(
            score_test_with_anomaly, 
            test_data_anom_info,
            max_anom_detect_time=pd.Timedelta("2D"))

        anom_result[ts]["skf"]["false_rate"][i] = false_rate[0]
        anom_result[ts]["skf"]["prob"][i] = prob_detection[0]
        anom_result[ts]["skf"]["ttd"][i] = time_to_detection[0]


    # -------------------------------------------------------------------#
    # # 2. Prophet detector
    print("----------------------------------------------")
    print("Prophet detector ...... ")

    anom_result[ts]["prophet"] = {}
    anom_result[ts]["prophet"]["false_rate"] = {}
    anom_result[ts]["prophet"]["prob"] = {}
    anom_result[ts]["prophet"]["ttd"] = {}

    model = ProphetDetector(anom_threshold=0.1)

    score_test = model.get_anomaly_score(data=copy.deepcopy(test_dict_no_anom))

    score_test_with_anomaly = model.get_anomaly_score(data=copy.deepcopy(test_data_with_anomaly))

    false_rate = FalseRate(score_test)
    prob_detection, time_to_detection = ProbTimeDetection(
        score_test_with_anomaly, 
        test_data_anom_info,
        max_anom_detect_time=pd.Timedelta("2D"))

    anom_result[ts]["prophet"]["false_rate"] = false_rate[0]
    anom_result[ts]["prophet"]["prob"] = prob_detection[0]
    anom_result[ts]["prophet"]["ttd"] = time_to_detection[0]

    # -------------------------------------------------------------------#
    # # 3. LSTM-ED detector
    print("----------------------------------------------")
    print("LSTM-ED detector ...... ")

    anom_result[ts]["lstmed"] = {}
    anom_result[ts]["lstmed"]["false_rate"] = {}
    anom_result[ts]["lstmed"]["prob"] = {}
    anom_result[ts]["lstmed"]["ttd"] = {}

    for i in ["seed_1","seed_2"]:
        print(f"    Seed #{i}")
        model = LstmEdDetector(sequence_len=24)

        model.train(data=train_df)

        score_test = model.get_anomaly_score(data=copy.deepcopy(test_dict_no_anom))

        score_test_with_anomaly = model.get_anomaly_score(data=copy.deepcopy(test_data_with_anomaly))

        false_rate = FalseRate(score_test)
        prob_detection, time_to_detection = ProbTimeDetection(
            score_test_with_anomaly, 
            test_data_anom_info,
            max_anom_detect_time=pd.Timedelta("2D"))

        anom_result[ts]["lstmed"]["false_rate"][i] = false_rate[0]
        anom_result[ts]["lstmed"]["prob"][i] = prob_detection[0]
        anom_result[ts]["lstmed"]["ttd"][i] = time_to_detection[0]

    # -------------------------------------------------------------------#
    # # 4. TranAD detector
    print("----------------------------------------------")
    print("TranAD detector ...... ")

    anom_result[ts]["tranad"] = {}
    anom_result[ts]["tranad"]["false_rate"] = {}
    anom_result[ts]["tranad"]["prob"] = {}
    anom_result[ts]["tranad"]["ttd"] = {}

    for i in ["seed_1", "seed_2"]:
        print(f"     Seed #{i}")
        model = TranAdDetector(num_epoch=5)

        model.train(data=train_df)

        score_test = model.get_anomaly_score(data=copy.deepcopy(test_dict_no_anom))

        score_test_with_anomaly = model.get_anomaly_score(data=copy.deepcopy(test_data_with_anomaly))

        false_rate = FalseRate(score_test)
        prob_detection, time_to_detection = ProbTimeDetection(
            score_test_with_anomaly, 
            test_data_anom_info,
            max_anom_detect_time=pd.Timedelta("2D"))

        anom_result[ts]["tranad"]["false_rate"][i] = false_rate[0]
        anom_result[ts]["tranad"]["prob"][i] = prob_detection[0]
        anom_result[ts]["tranad"]["ttd"][i] = time_to_detection[0]

    
    check = 1

    




    







    



    


