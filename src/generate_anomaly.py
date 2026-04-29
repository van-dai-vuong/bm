from typing import Optional
import pandas as pd
import numpy as np


def GenerateAnomaly(
    data,
    time,
    anomaly_info,
    col: Optional[list[str]] = None,
):

    data.columns = data.columns.astype(str)
    time.columns = time.columns.astype(str)
    if col is None:
        col = [str(i) for i in range(data.shape[1])]
        
    data_with_anomaly = {}

    for c in col:
        # Get time and values for this column, drop NaN in time
        valid_mask = time[c].notna()
        t = time[c][valid_mask].values
        v = data[c][valid_mask].values

        # Create df_temp with time as index
        df_temp = pd.DataFrame({"value": v}, index=pd.to_datetime(t))

        data_with_anomaly[c] = {}
        data_with_anomaly[c]["level"] = {}
        data_with_anomaly[c]["trend"] = {}

        for i, val in enumerate(anomaly_info[c]["anomaly_magnitude"]["level"]):
            _anomaly_time = pd.Timestamp(anomaly_info[c]["anomaly_time"][i])
            df_temp_i = df_temp.copy()
            df_temp_i.loc[df_temp_i.index >= _anomaly_time, "value"] += val
            data_with_anomaly[c]["level"][i] = df_temp_i

        for i, val in enumerate(anomaly_info[c]["anomaly_magnitude"]["trend"]):
            _anomaly_time = pd.Timestamp(anomaly_info[c]["anomaly_time"][i])
            df_temp_i = df_temp.copy()
            # Apply trend: linearly increasing from anomaly_time onward
            trend_mask = df_temp_i.index >= _anomaly_time
            n = trend_mask.sum()
            df_temp_i.loc[trend_mask, "value"] += val * np.arange(1, n + 1)
            data_with_anomaly[c]["trend"][i] = df_temp_i

    return data_with_anomaly