import kagglehub 
import numpy as np
import os
import pandas as pd

path = kagglehub.dataset_download("maxhorowitz/nflplaybyplay2009to2016")
print(path)

filepath_ = os.path.join(path, "NFL Play by Play 2009-2017 (v4).csv")
nfl_data = pd.read_csv(filepath_)

# Set seed for reproducibility
np.random.seed(0)

# Get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# How many total missing values do we have
rows, cols = nfl_data.shape
total_cells = rows * cols
total_missing = missing_values_count.sum()

# Percentage of data missing
percent_missing = (total_missing/total_cells) * 100

# Drop all rows that have any NaN values
# nfl_data = nfl_data.dropna()

# Remove all the columns that have a missing value
# nfl_data = nfl_data.dropna(axis=1)

# get a small subset of the dataset
small_sub = nfl_data.loc[:, 'EPA':'Season'].head()  
small_sub.fillna(0, inplace=True)  # Fill NaN values with 0 in the small subset

# Replace NaN values with the next valid observation (backward fill)
small_sub.bfill(inplace=True, axis=0).fillna(0, inplace=True)