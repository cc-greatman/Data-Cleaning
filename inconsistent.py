import kagglehub
import numpy as np
import os
import pandas as pd 

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

countries = df.Country.unique()


# Get the top 10 closest matches to "south korea"
matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# Function to replace rows in the provided column of the provided dataframe
# that matches the provided string above the provided ratio with the provided string
def replace_matches_in_cols(df, column, string_to_match, min_ratio=47):
    # Get a unique list of strings
    strings = df[column].unique()

    # Get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # Only matches with a ration > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # Get the rows of all the close matches in out dataframe
    rows_with_matches = df[column].isin(close_matches)

    # Replace all the rows with the close matches with the input matches
    df.loc[rows_with_matches, column] = string_to_match

    # let us know when it's all done
    print("All Done ✅")

replace_matches_in_cols(df=df, column='Country', string_to_match="south korea")

countries = df.Country.unique()
print(countries)

