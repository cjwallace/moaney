# # Evaluate

# This job evaluates a pre-trained classifier on a train and dev dataset.


# ## Imports

import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, confusion_matrix
)
from complainer.preprocessing import target_encoding_dict

# ## Params

# The following should be set as environment variables in the CDSW job.

DATA = os.environ['DATA']
VECTORIZER = os.environ['VECTORIZER']
MODEL = os.environ['MODEL']


# ## Read data

data = pd.read_csv(DATA)


# ## Read vectorizer and model

vectorizer = joblib.load(VECTORIZER)
model = joblib.load(MODEL)


# ## Featurize

features = vectorizer.transform(data.complaint)
target = data.issue


# ## Predict (batch score)

predictions = model.predict(features)

# ## Metrics

# Calculate a measure of goodness.
# We'll use the area under the ROC curve (true positive vs false positive
# rate), with a weighted average over the multiple classes.
# To do this we must binarize the labels (because that's what the sklearn
# method wants).

labels = [x for x in set(target_encoding_dict.values())]

binary_target = label_binarize(target, classes=labels)
binary_predictions = label_binarize(predictions, classes=labels)

roc = roc_auc_score(
    binary_target,
    binary_predictions,
    average='weighted'
)


print(
  "The weighted average ROC AUC for the train set: {}"
  .format(roc)
)


# Let's also take a look at the precision, recall, f-score and support,
# again weighted by class imbalances.

prfs = precision_recall_fscore_support(
    binary_target,
    binary_predictions,
    average='weighted'
)


# ## Print metrics
# It would be good to log these somewhere persistent too.

print(
    """
    Training set metrics (class weighted averages)
    ---
    roc auc: {}
    precision: {}
    recall: {}
    f-score: {}
  
  """
  .format(roc, *prfs[:3])
)

# ## Confusion matrix
# High level metrics are high level.
# Let's look at a confusion matrix to understand what's going on
# in more detail.

cm = confusion_matrix(target, predictions, labels=model.classes_)

# Normalize by class imbalance
norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

norm_cm_df = pd.DataFrame(norm_cm, index=model.classes_, columns=model.classes_)

# Show result
def plot_confusion_matrix(df):
    ax = sns.heatmap(df)
    ax.set(xlabel="Predicted labels", ylabel="True labels")
    return ax

plot_confusion_matrix(norm_cm_df)

# ## Print log

print("JOB PARAMS:")
print("DATA: {}".format(DATA))
print("VECTORIZER: {}".format(VECTORIZER))
print("MODEL: {}".format(MODEL))