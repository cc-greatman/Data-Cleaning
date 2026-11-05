# modules we'll use
import pandas as pd
import numpy as np
import kagglehub
import os

# Helpful character encoding module
import charset_normalizer

# Set seed for reproducibility
np.random.seed(0)

# Start with a string
before = "This is the won symbol: ₩"

# Check to see what datatype it is
print("Character type before:", type(before))

# Encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors="replace")

# Check the new type
print("Character type after:", type(after))

# Convert it back to utf-8
after = after.decode("utf-8")
print(after)

file_path = kagglehub.dataset_download("kemical/kickstarter-projects")
csv_file = os.path.join(file_path, "ks-projects-201612.csv")

# Using charset to guess what kind of encoding was used on the file
with open(csv_file, 'rb') as rawdata:
    result = charset_normalizer.detect(rawdata.read(100000))

# Decoding the file directly while reading
kickstarters_2016 = pd.read_csv(csv_file, encoding='cp1006')

# Saving our file(will be saved as UTF-8 by default)
kickstarters_2016.to_csv("ks-projects-201612-utf8.csv")
