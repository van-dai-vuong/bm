import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data folder
data_folder = "/Users/vuongdai/GitHub/bm/detrend_data/weekly/"
data_file = "ts_weekly_values_standardize.csv"
data_file_time = "ts_weekly_datetimes.csv"

fig_folder = "/Users/vuongdai/GitHub/bm/results/fig/data/"

# Columns to read
save_figs = True  # Set to False to just display

sensor_names = pd.read_csv(data_folder + data_file, nrows=0, delimiter=",").columns.tolist()
df_all = pd.read_csv(data_folder + data_file, skiprows=1, delimiter=",", header=None)
df_time_all = pd.read_csv(data_folder + data_file_time, skiprows=1, delimiter=",", header=None)

for i, sensor in enumerate(sensor_names):
    df = df_all.iloc[:, i]
    df_time = df_time_all.iloc[:, i]

    # Trim to last valid index
    last_idx = df.last_valid_index()
    df = df.iloc[:last_idx + 1]
    df_time = df_time.iloc[:last_idx + 1]

    time = pd.to_datetime(df_time.values.flatten())

    plt.figure(figsize=(14, 6))
    plt.plot(time, df.values)
    plt.title(sensor)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_folder + f"{sensor}.png", dpi=150)
        print(f"Saved: {sensor}.png")

    plt.close()