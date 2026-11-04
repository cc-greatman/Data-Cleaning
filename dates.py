# Modules we'll use
import datetime
import kagglehub
import numpy as np
import os
import pandas as pd
import seaborn as sns

data_path = kagglehub.dataset_download("nasa/landslide-events")
file_path = os.path.join(data_path, "catalog.csv")
landslide_data = pd.read_csv(file_path)

np.random.seed(0)

# Create a new column, date_parsed, with the parsed dates
landslide_data['date_parsed'] = pd.to_datetime(landslide_data['date'], format="%m/%d/%y")

# we can let pandas infer the format of the date string, but it is better to specify the format explicitly for performance reasons
#landslide_data['date_parsed'] = pd.to_datetime(landslide_data['date'], infer_datetime_format=True)

# Now we can also extract info like day of the month, day of the week, month, year, etc. from the date_parsed column
day_of_month_landslide = landslide_data.date_parsed.dt.day
print(day_of_month_landslide.head())