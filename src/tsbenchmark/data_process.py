from typing import Optional
import pandas as pd
import numpy as np

import pandas as pd
import os

def UpdateCsv(df, values_path: str, datetimes_path: str, rewrite: bool = False):
    """
    Checks if df's column name exists in values.csv and datetimes.csv.
    If not found, appends:
      - df values as a new column to values.csv
      - df index as a new column to datetimes.csv
 
    Args:
        df: DataFrame with exactly 1 column
        values_path: path to values.csv
        datetimes_path: path to datetimes.csv
        rewrite: if True, overwrite the column even if it already exists
    """
    assert len(df.columns) == 1, "df must have exactly 1 column"
 
    col_name = df.columns[0]
    print(f"Column name to check: '{col_name}'")
 
    # --- Load or initialize values.csv ---
    if os.path.exists(values_path):
        values = pd.read_csv(values_path)
    else:
        print(f"'{values_path}' not found.")
        values = pd.DataFrame()
 
    # --- Load or initialize datetimes.csv ---
    if os.path.exists(datetimes_path):
        datetimes = pd.read_csv(datetimes_path)
    else:
        print(f"'{datetimes_path}' not found.")
        datetimes = pd.DataFrame()
 
    # --- Check if column already exists in both files ---
    in_values = col_name in values.columns
    in_datetimes = col_name in datetimes.columns
 
    if in_values and in_datetimes and not rewrite:
        print(f"Column '{col_name}' already exists in both files. No changes made.")
        return
 
    # --- Add/overwrite values.csv ---
    if not in_values or rewrite:
        if rewrite and in_values:
            values = values.drop(columns=[col_name])
        new_col = pd.DataFrame({col_name: df[col_name].values})
        values = pd.concat([values, new_col], axis=1)
        values.to_csv(values_path, index=False, na_rep="")
        print(f"Added '{col_name}' values to '{values_path}'.")
    else:
        print(f"Column '{col_name}' already in '{values_path}'. Skipped.")
 
    # --- Add/overwrite datetimes.csv ---
    if not in_datetimes or rewrite:
        if rewrite and in_datetimes:
            datetimes = datetimes.drop(columns=[col_name])
        new_col = pd.DataFrame({col_name: df.index.values})
        datetimes = pd.concat([datetimes, new_col], axis=1)
        datetimes.to_csv(datetimes_path, index=False, na_rep="")
        print(f"Added '{col_name}' index to '{datetimes_path}'.")
    else:
        print(f"Column '{col_name}' already in '{datetimes_path}'. Skipped.")


def PrepareDf (data, time):
    data.columns = data.columns.astype(str)
    time.columns = time.columns.astype(str)
    col = [str(i) for i in range(data.shape[1])]
    data_dict = {}

    for c in col:
        # Get time and values for this column, drop NaN in time
        valid_mask = time[c].notna()
        t = time[c][valid_mask].values
        v = data[c][valid_mask].values

        # Create df_temp with time as index
        df_temp = pd.DataFrame({"values": v}, index=pd.to_datetime(t))
        data_dict[int(c)] = df_temp

    return data_dict

def GenerateAnomaly(
    data,
    anomaly_info,
    col: Optional[list[int]] = None,
):
        
    data_with_anomaly = {}
    anomaly_info_out = {}

    num_anomaly_per_magnitude = anomaly_info["num_anomaly_per_magnitude"]
    anom_start = anomaly_info["anomaly_start"]
    anom_end = anomaly_info["anomaly_end"]

    # Anomaly start location
    _anom_start_index = []
    for _anom_start in anomaly_info["random_number"]:
        if anom_start <= _anom_start <= anom_end:
            _anom_start_index.append(_anom_start)
            if len(_anom_start_index) == num_anomaly_per_magnitude:
                break
    _anom_start_index = np.array(_anom_start_index)

    if col is None:
        col = range(len(data))

    for c in col:
        df_temp = data[c]
        anom_start_index = (_anom_start_index * len(df_temp.index)).astype(int).tolist()

        for d in (data_with_anomaly, anomaly_info_out):
            d[int(c)] = {}
            if anomaly_info["anomaly_magnitude"].get("level"):
                d[int(c)]["level"] = {}
            if anomaly_info["anomaly_magnitude"].get("trend"):
                d[int(c)]["trend"] = {}

        if anomaly_info["anomaly_magnitude"].get("level"):
            for _, val in enumerate(anomaly_info["anomaly_magnitude"]["level"]):
                for d in (data_with_anomaly, anomaly_info_out):
                    d[int(c)]["level"][str(val)] = {}
                for j in range(num_anomaly_per_magnitude):
                    data_with_anomaly[int(c)]["level"][str(val)][j] = {}
                    _anomaly_time = df_temp.index[anom_start_index[j]]
                    df_temp_i = df_temp.copy()
                    df_temp_i.iloc[df_temp_i.index >= _anomaly_time,0] += val
                    data_with_anomaly[int(c)]["level"][str(val)][j] = df_temp_i
                    anomaly_info_out[int(c)]["level"][str(val)][j] = _anomaly_time

        if anomaly_info["anomaly_magnitude"].get("trend"):
            for _, val in enumerate(anomaly_info["anomaly_magnitude"]["trend"]):
                for d in (data_with_anomaly, anomaly_info_out):
                    d[int(c)]["trend"][str(val)] = {}
                for j in range(num_anomaly_per_magnitude):
                    data_with_anomaly[int(c)]["trend"][str(val)][j] = {}
                    _anomaly_time = df_temp.index[anom_start_index[j]]
                    df_temp_i = df_temp.copy()
                    # Apply trend: linearly increasing from anomaly_time onward
                    trend_mask = df_temp_i.index >= _anomaly_time
                    n = trend_mask.sum()
                    df_temp_i.iloc[trend_mask, 0] += val * np.arange(1, n + 1)
                    data_with_anomaly[int(c)]["trend"][str(val)][j] = df_temp_i
                    anomaly_info_out[int(c)]["trend"][str(val)][j] = _anomaly_time

    return data_with_anomaly, anomaly_info_out 