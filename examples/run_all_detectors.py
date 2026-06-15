import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import pandas as pd
import json

from src.tsbenchmark.data_process import GenerateAnomaly
from src.tsbenchmark.metric import ProbTimeDetection
from src.tsbenchmark.detector.tranad_detector import TranAdDetector
from src.tsbenchmark.detector.phophet_detector import ProphetDetector
from src.tsbenchmark.detector.lstmed_detector import LstmEdDetector

# ── Config ────────────────────────────────────────────────────────────────────

DATA_FILE             = "/Users/vuongdai/GitHub/canari/data/benchmark_data/test_2_data.csv"
ANOMALY_INFO_FILE     = "/Users/vuongdai/GitHub/bm/detrend_data/anomaly_info/anomaly_info.json"
TRAIN_TEST_SPLIT      = 0.5
START_ANOMALY_OFFSET  = 0.25

# ── Detector registry ─────────────────────────────────────────────────────────
# Add or remove detectors here; no other code needs to change.
#   cls        – detector class
#   kwargs     – constructor arguments
#   needs_fit  – whether the detector requires a .train() call

DETECTORS = [
    {"name": "TranAd",  "cls": TranAdDetector,  "kwargs": {"num_epoch": 5},        "needs_fit": True},
    {"name": "Prophet", "cls": ProphetDetector,  "kwargs": {"anom_threshold": 0.1}, "needs_fit": False},
    {"name": "LstmEd",  "cls": LstmEdDetector,   "kwargs": {"sequence_len": 52},    "needs_fit": True},
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt(v):
    """Pretty-print a scalar, flat dict, or arbitrarily nested structure."""
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (dict, list)):
        return "\n" + json.dumps(v, indent=4, default=str)
    return str(v)

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_FILE, delimiter=",")
df.index = pd.to_datetime(df.pop("date"))
df.index.name = "date_time"
df_dict = {0: df}

with open(ANOMALY_INFO_FILE) as f:
    anomaly_info = json.load(f)

# ── Run ───────────────────────────────────────────────────────────────────────

results = {}

for ts, df_temp in df_dict.items():

    train_size = int(len(df_temp) * TRAIN_TEST_SPLIT)
    _df_train  = df_temp.iloc[:train_size]
    df_with_anomaly = GenerateAnomaly(
        data={0: df_temp.iloc[train_size:]},
        anomaly_info=anomaly_info,
        start_anomaly_offset=START_ANOMALY_OFFSET,
    )

    for det in DETECTORS:
        print(f"\n[ts={ts}] Running {det['name']} …")

        model = det["cls"](**det["kwargs"])
        if det["needs_fit"]:
            model.train(data=_df_train)

        score = model.get_anomaly_score(data=copy.deepcopy(df_with_anomaly))
        prob, ttd = ProbTimeDetection(score, anomaly_info)

#         results[f"ts{ts}_{det['name']}"] = {"prob_detection": prob, "time_to_detection": ttd}
#         print(f"  prob_detection   = {fmt(prob)}")
#         print(f"  time_to_detection= {fmt(ttd)}")

# # ── Summary ───────────────────────────────────────────────────────────────────

# print("\n" + "=" * 55)
# print("RESULTS SUMMARY")
# print("=" * 55)
# for key, m in results.items():
#     print(f"\n{key}")
#     print(f"  prob_detection    = {fmt(m['prob_detection'])}")
#     print(f"  time_to_detection = {fmt(m['time_to_detection'])}")
