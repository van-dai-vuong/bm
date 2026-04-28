import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data folder 
data_folder = "/Users/vuongdai/GitHub/bm/detrend_data/weekly/"

data_file = "ts_weekly_values.csv"
data_file_time = "ts_weekly_datetimes.csv"

output_file = "ts_weekly_values_standardize.csv"

sensor_names = pd.read_csv(data_folder + data_file, nrows=0, delimiter=",").columns.tolist()
df = pd.read_csv(data_folder + data_file, skiprows=1, delimiter=",", header=None)
df_time = pd.read_csv(data_folder + data_file_time, skiprows=1, delimiter=",", header=None)


# Standardize column-wise (zero mean, unit variance)
df_standardized = (df - df.mean()) / df.std()

# Write output to new CSV file
df_standardized.to_csv(data_folder + output_file, index=False, header=sensor_names)