# # Train classifier

# This job trains a ML algorithm and persists it to disk.

# ## Imports

import os
import joblib
import pandas as pd
from nbsvm import NBSVM
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# ## Params

# The following should be set as environment variables in the CDSW job.

TRAIN_DATA = os.environ['TRAIN_DATA']
MODEL_DIRECTORY = os.environ['MODEL_DIRECTORY']


# ## Read data

train = pd.read_csv(TRAIN_DATA)


# ## Featurize

# We need a computable representation of text.
# For topic classification (which is what we're doing here),
# keywords usually work great, at least as a baseline.
# We'll use scikit's tf-idf.

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train.complaint)
y = train.issue


# ## Train a classifier

# We'll try a simple multinomial naive bayes classifier.
# Technically this should use integer counts (because that's what a multinomial
# distribution represents), but it works with td-idf in practice too.

model = NBSVM()
model.fit(X, y)


# ## Persist model

# Create target directory if necessary.

if not os.path.exists(MODEL_DIRECTORY):
    os.mkdir(MODEL_DIRECTORY)

# And persist vectorizer and classifier objects.

joblib.dump(vectorizer, MODEL_DIRECTORY + 'vectorizer.pkl')
joblib.dump(model, MODEL_DIRECTORY + 'model.pkl')

# ## Print log

print("JOB PARAMS:")
print("TRAIN_DATA: {}".format(TRAIN_DATA))
print("MODEL_DIRECTORY: {}".format(MODEL_DIRECTORY))