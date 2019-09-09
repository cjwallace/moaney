# # Prototyping on mortgages

# Let's prototype a potentially useful ML app here.
# We'll abstract and test and expand later, but we can try to fail fast.


# ## Imports

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

# ## Read data

# We'll read the data and keep only those complaints with narratives.
# I don't like stateful manipulation of dataframes in the global scope,
# so we're going to wrap it in a function.

def read_complaints(filename):
    """
    Reads the consumer complaints data (in it's standard downloaded csv form)
    from the file `filename', relative to project root directory.
    """
    complaints = pd.read_csv(filename)
    complaints = complaints[complaints['Consumer complaint narrative'].notnull()]
    return complaints

complaints = read_complaints('data/raw/consumer_complaints.csv')


# Let's try to build an "issue classifier" from the free text.
# This could be used as an assistive system for the administrators that
# need to respond to the free text:
# complaints could be automatically routed to the right person
# (with the caveat that they need a method for correcting misclassifications!)

# As described in the field reference, the "Issue" identified for each
# complaint can take certain values depending on the product
# (which is specified by the user).

# Let's at least start simple, and focus on just one product.
# I'm choosing mortgages because I like the cut of it's jib.

# How many issues exist within the mortgage product category?

(
    complaints
    [complaints['Product'] == 'Mortgage'] # filter
    .groupby('Issue')
    ['Complaint ID'] # select column
    .count()
    .sort_values(ascending=False)
)


# Owing to the imbalanced number of entries for each issue, I suggest
# we take only the top 9 (arbitrarily) categories here,
# then introduce an "other" category.
# It's still an imbalanced dataset, but nonetheless, let's go ahead.
# Let's also encode the issue categories for ease of reference.

issues_abbreviations = {
    "Loan servicing, payments, escrow account": "loan_servicing",
    "Loan modification,collection,foreclosure": "loan_modification",
    "Trouble during payment process": "payment_process",
    "Struggling to pay mortgage": "struggling_to_pay",
    "Application, originator, mortgage broker": "application",
    "Settlement process and costs": "settlement",
    "Applying for a mortgage or refinancing an existing mortgage": "applying",
    "Closing on a mortgage": "closing",
    "Credit decision / Underwriting": "underwriting",
    "Incorrect information on your report": "other",
    "Applying for a mortgage": "applying",
    "Problem with a credit reporting company's investigation into an existing problem": "other",
    "Improper use of your report": "other",
    "Credit monitoring or identity theft protection services": "other",
    "Unable to get your credit report or credit score": "other",
    "Problem with fraud alerts or security freezes": "other"
}


# Our `issues_abbreviations` dict is not 1:1.
# Inverting won't work perfectly because we've mapped several categories to
# "other".
# We'll "fix" that by replacing the categories mapping to "other" with
# "Unknown" in the reverse mapping, since we can't be sure which issue they
# should be attributed to.
# Again, I'm going to encapsulate the stateful append to the dict with a
# function.

def create_abbreviations_issues():
    """
    Reverse the issues_abbreviations dict, encapsulating state
    """
    abbreviations_issues = {
        v:k for (k,v) in issues_abbreviations.items()
    }
    abbreviations_issues['other'] = 'Unknown'
    return abbreviations_issues

abbreviations_issues = create_abbreviations_issues()


# Now we can clean up the data.

def create_mortgages_frame():
    mortgages = (
        complaints
            [complaints['Product'] == 'Mortgage']
            [['Consumer complaint narrative', 'Issue']]
            .rename({
                'Consumer complaint narrative': 'complaint',
                'Issue': 'issue'
                },
                axis='columns')
            .reset_index(drop=True)
    )
    mortgages['issue'] = mortgages.issue.apply(
      lambda x: issues_abbreviations[x]
    )
    return mortgages

mortgages = create_mortgages_frame()

# Let's take a look.

mortgages.head()

def plot_mortgage_complaint_lengths():
    # This is the second time I've written this code, so maybe
    # it'd be a good thing to abstract!
    ax = mortgages.complaint.apply(len).plot(kind='hist', bins = 100)
    ax.set_xlabel('Character length of text')
    return ax

plot_mortgage_complaint_lengths()

# Still a wide range of lengths.

complaint_lengths = mortgages.complaint.apply(len)
complaint_lengths.describe()

# I'm suspicious of the extremely long messages.
# Let's look at some of them.

long_messages = (
    mortgages
    .where(mortgages.complaint.apply(lambda c: len(c) > 10000))
    .dropna()
)

long_messages.sample().complaint.values[0]

# Yup, they just seem like really long complaints ¯\\\_(ツ)_/¯

# How imbalanced is the dataset?

def plot_class_balance():
    ax = pd.value_counts(mortgages.issue).plot(kind='bar')
    ax.set_xlabel('Issue')
    ax.set_ylabel('Count')
    return ax

plot_class_balance()

# ## Classifying complaints

# So we have some handle on the data.
# Let's train a classifier to separate complaints into the issues identified.

# First, split into train and test sets, stratified across the classes.
# If we were going to do model selection too, we'd want to split into an
# additional set to give us some measure of expected real world performance
# (ie stop us overfitting the test set).
# We'll do that eventually, but the results of this experiment aren't going
# be used for anything except testing whether this problem is easily amenable
# to ML, so I'm not too concerned about holding out validation data right now.

train_X, test_X, train_y, test_y = train_test_split(
    mortgages.complaint,
    mortgages.issue,
    stratify=mortgages.issue,
    test_size=0.2
)


# We need a computable representation of text.
# For topic classification (which is what we're doing here),
# keywords usually work great, at least as a baseline.
# We'll use scikit's tf-idf.

vectorizer = TfidfVectorizer()
vectorized_train_X = vectorizer.fit_transform(train_X)
vectorized_test_X = vectorizer.transform(test_X)


# Yes, using vectorizer like this has broken my enforced rigour about
# state transforms in the global scope (vectorizer is now different to
# when it was created). I'll deal with it.

# We'll try a simple multinomial naive bayes classifier.
# Technically this should use integer counts (because that's what a multinomial
# distribution represents), but it works with td-idf in practice too.

clf = MultinomialNB()
clf.fit(vectorized_train_X, train_y)


# Predict on train and test using the fitted classifier.

train_y_pred = clf.predict(vectorized_train_X)
test_y_pred = clf.predict(vectorized_test_X)


# Calculate a measure of goodness.
# We'll use the area under the ROC curve (true positive vs false positive
# rate), with a weighted average over the multiple classes.
# To do this we must binarize the labels (because that's what the sklearn
# method wants).

labels = [x for x in set(issues_abbreviations.values())]

binary_true_train_y = label_binarize(train_y, classes=labels)
binary_pred_train_y = label_binarize(train_y_pred, classes=labels)
binary_true_test_y = label_binarize(test_y, classes=labels)
binary_pred_test_y = label_binarize(test_y_pred, classes=labels)

train_roc = roc_auc_score(
    binary_true_train_y,
    binary_pred_train_y,
    average='weighted'
)

test_roc = roc_auc_score(
    binary_true_test_y,
    binary_pred_test_y,
    average='weighted'
)

print(
  "The weighted average ROC AUC for the train set: {}"
  .format(train_roc)
)

print(
  "The weighted average ROC AUC for the test set: {}"
  .format(test_roc)
)


# OK. That's not a great score, but we used a simple
# method with all default params (for both td-idf and naive bayes).

# Let's also take a look at the precision, recall, f-score and support,
# again weighted by class imbalances.

train_prfs = precision_recall_fscore_support(
    binary_true_train_y,
    binary_pred_train_y,
    average='weighted'
)

test_prfs = precision_recall_fscore_support(
    binary_true_test_y,
    binary_pred_test_y,
    average='weighted'
)

# When calling the `precision...` function, we get an UndefinedMetricWarning
# telling us that there are labels with no predicted samples.
# That's likely because the classes in our training set are so imbalanced.
# There are multiple ways of dealing with that which we can investigate later.