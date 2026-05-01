import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import json
from src.data_process import GenerateAnomaly, PrepareDf
from src.metric import ProbTimeDetection, FalseRate

from pyod.models.abod import ABOD
from pyod.models.ocsvm import OCSVM

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
df_with_anomaly = GenerateAnomaly(data=df_dict, anomaly_info=anomaly_info, col=[0, 1])

# ---------------------------------------------------------------------------
# Detector configuration
# Set DETECTOR_NAME to switch between detectors.
# Supported: "abod", "ocsvm"
# ---------------------------------------------------------------------------
LOOK_BACK_LEN = 19       # sliding-window size, mirrors the original LSTM look_back_len
DETECTOR_NAME = "ocsvm"  # <-- change this to swap detectors


def make_sliding_windows(values: np.ndarray, look_back: int) -> np.ndarray:
    """Convert a 1-D time series into overlapping sliding-window feature rows."""
    values = values.ravel()
    n = len(values)
    if n <= look_back:
        pad = np.full(look_back - n, values[0])
        values = np.concatenate([pad, values])
        n = len(values)
    return np.array([values[i: i + look_back] for i in range(n - look_back)])


def build_detector(train_series: np.ndarray, look_back: int = LOOK_BACK_LEN, name: str = DETECTOR_NAME):
    """
    Fit a pyod detector on sliding windows from the training series.

    Parameters
    ----------
    train_series : array-like 1-D training values
    look_back    : sliding-window width (number of lagged features)
    name         : which detector to use — "abod" or "ocsvm"
    """
    X_train = make_sliding_windows(train_series, look_back)

    if name == "abod":
        detector = ABOD(
            contamination=0.05,  # expected fraction of outliers
            n_neighbors=10,      # neighbours used in angle computation
            method="fast",       # "fast" = FastABOD; "default" = exact
        )
    elif name == "ocsvm":
        detector = OCSVM(
            contamination=0.05,  # expected fraction of outliers
            kernel="rbf",        # kernel: "rbf", "linear", "poly", "sigmoid"
            nu=0.05,             # upper bound on fraction of outliers (≈ contamination)
            gamma="auto",        # kernel coefficient; "auto" = 1/n_features
        )
    else:
        raise ValueError(f"Unknown detector '{name}'. Choose from: 'abod', 'ocsvm'.")

    detector.fit(X_train)
    return detector


# ---------------------------------------------------------------------------
# Train on column index 1 of df (same as original script)
# ---------------------------------------------------------------------------
train_split = 0.4
df_temp = df.iloc[:, [1]]
train_size = int(len(df_temp) * train_split)
train_series = df_temp.values[:train_size]

detector = build_detector(train_series, look_back=LOOK_BACK_LEN, name="abod")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def score_series(values: np.ndarray, det: ABOD, look_back: int = LOOK_BACK_LEN) -> np.ndarray:
    """
    Return a per-timestep binary anomaly flag (True = anomaly).
    The first `look_back` timesteps inherit the label of the first window.
    NaNs are imputed via forward-fill then backward-fill before windowing.
    """
    # Impute NaNs: forward-fill then backward-fill to handle leading/trailing NaNs
    values = pd.Series(values.ravel()).ffill().bfill().values
    X = make_sliding_windows(values, look_back)
    labels = det.predict(X).astype(bool)          # 1 = outlier, 0 = inlier
    prefix = np.full(look_back, labels[0])
    return np.concatenate([prefix, labels])        # shape: (len(values),)


def detect_synthetic_anomaly(
    data: dict,
    det: ABOD,
    look_back: int = LOOK_BACK_LEN,
) -> dict:
    anomaly_score = {}
    for ts in data:
        anomaly_score[ts] = {}
        for anomaly_type in data[ts]:
            anomaly_score[ts][anomaly_type] = {}
            for i in data[ts][anomaly_type]:
                anomaly_score[ts][anomaly_type][i] = {}
                for j in data[ts][anomaly_type][i]:
                    series_df = data[ts][anomaly_type][i][j]
                    flags = score_series(series_df.values, det, look_back)
                    anomaly_score[ts][anomaly_type][i][j] = pd.DataFrame(
                        {"value": flags}, index=series_df.index
                    )
    return anomaly_score


def detect_anomaly(
    data: dict,
    det: ABOD,
    look_back: int = LOOK_BACK_LEN,
) -> dict:
    anomaly_score = {}
    for ts in data:
        series_df = data[ts]
        flags = score_series(series_df.values, det, look_back)
        anomaly_score[ts] = pd.DataFrame({"value": flags}, index=series_df.index)
    return anomaly_score


# ---------------------------------------------------------------------------
# Estimate anomaly scores & evaluate
# ---------------------------------------------------------------------------
anomaly_score_synthetic = detect_synthetic_anomaly(data=df_with_anomaly, det=detector)
anomaly_score_test_set = detect_anomaly(data=df_dict, det=detector)

prob, time = ProbTimeDetection(anomaly_score_synthetic, anomaly_info)
false_rate = FalseRate(anomaly_score_test_set)