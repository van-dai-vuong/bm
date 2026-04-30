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
    data_dict = {}
    num_anomaly_per_magnitude = anomaly_info["num_anomaly_per_magnitude"]

    for c in col:
        # Get time and values for this column, drop NaN in time
        valid_mask = time[c].notna()
        t = time[c][valid_mask].values
        v = data[c][valid_mask].values

        # Create df_temp with time as index
        df_temp = pd.DataFrame({"value": v}, index=pd.to_datetime(t))
        data_dict[int(c)] = df_temp

        data_with_anomaly[int(c)] = {}
        data_with_anomaly[int(c)]["level"] = {}
        data_with_anomaly[int(c)]["trend"] = {}
        np.random.seed(anomaly_info["random_seed"])
        anom_index = np.random.randint(0, len(t), size=num_anomaly_per_magnitude)

        for i, val in enumerate(anomaly_info["anomaly_magnitude"]["level"]):
            data_with_anomaly[int(c)]["level"][i] = {}
            for j in range(num_anomaly_per_magnitude):
                _anomaly_time = df_temp.index[anom_index[j]]
                df_temp_i = df_temp.copy()
                df_temp_i.loc[df_temp_i.index >= _anomaly_time, "value"] += val
                data_with_anomaly[int(c)]["level"][i][j] = df_temp_i

        for i, val in enumerate(anomaly_info["anomaly_magnitude"]["trend"]):
            data_with_anomaly[int(c)]["trend"][i] = {}
            for j in range(num_anomaly_per_magnitude):
                _anomaly_time = df_temp.index[anom_index[j]]
                df_temp_i = df_temp.copy()
                # Apply trend: linearly increasing from anomaly_time onward
                trend_mask = df_temp_i.index >= _anomaly_time
                n = trend_mask.sum()
                df_temp_i.loc[trend_mask, "value"] += val * np.arange(1, n + 1)
                data_with_anomaly[int(c)]["trend"][i][j] = df_temp_i

    return data_with_anomaly, data_dict