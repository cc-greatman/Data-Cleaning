# Getting the dataset
import kagglehub
import os

# Module to be used for data manipulation
import numpy as np
import pandas as pd

# For Box-Cox Transformation
from scipy import stats

# For min_max scaling
from mlxtend.preprocessing import minmax_scaling

# Plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)

# Generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# min-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# Normalize the data using box-cox transformation
normalized_data = stats.boxcox(original_data)

# Plot both together to compare
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=True)
ax[0].set_title("Original Data")
sns.histplot(normalized_data, ax=ax[1], kde=True, legend=True)
ax[1].set_title("Normalized Data")
plt.show()