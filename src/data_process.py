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
        df_temp = pd.DataFrame({"value": v}, index=pd.to_datetime(t))
        data_dict[int(c)] = df_temp

    return data_dict

def GenerateAnomaly(
    data,
    anomaly_info,
    col: Optional[list[int]] = None,
):
        
    data_with_anomaly = {}
    num_anomaly_per_magnitude = anomaly_info["num_anomaly_per_magnitude"]
    if col is None:
        col = range(len(data))

    for c in col:
        df_temp = data[c]
        data_with_anomaly[int(c)] = {}
        data_with_anomaly[int(c)]["level"] = {}
        data_with_anomaly[int(c)]["trend"] = {}
        # np.random.seed(anomaly_info["random_seed"])
        # anom_index = np.random.randint(0, len(df_temp.index), size=num_anomaly_per_magnitude)
        anom_index = np.array(anomaly_info["random_number"][:num_anomaly_per_magnitude])
        anom_index = (anom_index * len(df_temp.index)).astype(int).tolist()
        for _, val in enumerate(anomaly_info["anomaly_magnitude"]["level"]):
            data_with_anomaly[int(c)]["level"][str(val)] = {}
            for j in range(num_anomaly_per_magnitude):
                _anomaly_time = df_temp.index[anom_index[j]]
                df_temp_i = df_temp.copy()
                df_temp_i.loc[df_temp_i.index >= _anomaly_time, "value"] += val
                data_with_anomaly[int(c)]["level"][str(val)][j] = df_temp_i

        for _, val in enumerate(anomaly_info["anomaly_magnitude"]["trend"]):
            data_with_anomaly[int(c)]["trend"][str(val)] = {}
            for j in range(num_anomaly_per_magnitude):
                _anomaly_time = df_temp.index[anom_index[j]]
                df_temp_i = df_temp.copy()
                # Apply trend: linearly increasing from anomaly_time onward
                trend_mask = df_temp_i.index >= _anomaly_time
                n = trend_mask.sum()
                df_temp_i.loc[trend_mask, "value"] += val * np.arange(1, n + 1)
                data_with_anomaly[int(c)]["trend"][str(val)][j] = df_temp_i

    return data_with_anomaly