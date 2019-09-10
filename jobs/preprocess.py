# # Preprocess

# This job takes an input directory containing train.csv, dev.csv and test.csv
# and performs some preprocessing.
# Prep is encoding the target variable, retaining only the relevant columns,
# and renaming those columns. Then persist to disk.

# ## Imports

import os
import csv
import pandas as pd
from complainer.preprocessing import (
  filter_rename_mortgages, encode_targets, target_encoding_dict
)

# ## Params

# The following should be set as environment variables in the CDSW job.

INPUT_DIRECTORY = os.environ['INPUT_DIRECTORY']
TARGET_DIRECTORY = os.environ['TARGET_DIRECTORY']

# ## Define procedure for reading, processing and writing
# Use python parsing engine, since messy string data can contain characters
# that the C engine does not like.

def preprocess(split):

    df = pd.read_csv(INPUT_DIRECTORY + '/' + split + '.csv',
                     delimiter=',',
                     engine='python',
                     quoting=csv.QUOTE_ALL)

    # Filter and rename columns

    mortgages = filter_rename_mortgages(df)
    mortgages = encode_targets(
        mortgages,
        target_column='issue',
        target_encoding_dict=target_encoding_dict
    )
    
    mortgages.to_csv(TARGET_DIRECTORY + '/' + split + '.csv',
                     index=False)
    
    return None

# ## Create target directory
# If necessary.

if not os.path.exists(TARGET_DIRECTORY):
    os.mkdir(TARGET_DIRECTORY)

# ## Read, process and write processed data to disk

for split in ['train', 'dev', 'test']:
    preprocess(split)
    print(split + ' complete')

# ## Print log

print("JOB PARAMS:")
print("INPUT_DIRECTORY: {}".format(INPUT_DIRECTORY))
print("TARGET_DIRECTORY: {}".format(TARGET_DIRECTORY))