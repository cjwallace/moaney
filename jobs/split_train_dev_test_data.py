# # Create a new train/dev/test split

# This job creates a new train/dev/test/split and persists it to disk.

# ## Imports

import os
import csv
import pandas as pd
from complainer.splitter import train_dev_test_split

# ## Params

# The following should be set as environment variables in the CDSW job.

INPUT_FILE = os.environ['INPUT_FILE']
TARGET_DIRECTORY = os.environ['TARGET_DIRECTORY']

# ## Read raw data

df = pd.read_csv(INPUT_FILE)

# ## Filter to data containing complaints only

df = df[df['Consumer complaint narrative'].notnull()]

# ## Split data into train, dev and test subsets

train, dev, test = train_dev_test_split(
  df, dev_fraction=0.2, test_fraction=0.1
)

# ## Create target directory
# If necessary.

if not os.path.exists(TARGET_DIRECTORY):
    os.mkdir(TARGET_DIRECTORY)

# ## Write subsets to disk
# Quote all fields to avoid weird character shenanigans.

train.to_csv(TARGET_DIRECTORY+'/train.csv', quoting=csv.QUOTE_ALL)
dev.to_csv(TARGET_DIRECTORY+'/dev.csv', quoting=csv.QUOTE_ALL)
test.to_csv(TARGET_DIRECTORY+'/test.csv', quoting=csv.QUOTE_ALL)

# ## Print log
print("JOB PARAMS:")
print("INPUT_FILE: {}".format(INPUT_FILE))
print("TARGET_DIRECTORY: {}".format(TARGET_DIRECTORY))