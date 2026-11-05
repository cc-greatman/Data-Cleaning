import kagglehub
import numpy as np
import os
import pandas as pd 
import torch as nn

# Helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
import charset_normalizer

filepath_ = kagglehub.dataset_download("alexisbcook/pakistan-intellectual-capital")
csv_filepath = os.path.join(filepath_, "pakistan_intellectual_capital.csv")

df = pd.read_csv(csv_filepath)

np.random.seed(0)

# Get all the unique values in the country column
countries = df.Country.unique()

# Sort the values gotten in Alphabetical order
countries = countries.sort()

# Due to some inconsistencies in the data we can try to generalize
# the outcome we are trying to get. Say make all entries lowercase and remove
# white-trailing spaces
df.Country = df.Country.str.lower() # lowercase
df.Country = df.Country.str.strip() # Remove white spaces
