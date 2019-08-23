# # First exploration

# ## Imports

import pandas as pd

# ## Data Load

# First, we load the data into a `pd.DataFrame`.
# The data is local and fits in memory, so this is easy.

complaints = pd.read_csv('data/raw/consumer_complaints.csv')

# ## Investigate

# Let's take a peak at the first few rows.

complaints.head()

# Lots of complaints do not have a consumer narrative associated with them.
# Let's drop those so we only retain rows where there's a text column with
# content.

narrative_complaints = (
    complaints[complaints['Consumer complaint narrative'].notnull()]
)

# How much smaller did that make the data set?

print(
    "{:.2f}% of complaints have a narrative"
    .format(100 * len(narrative_complaints) / len(complaints))
)

# Are all the complaints with a narrative submitted via the web?

narrative_complaints['Submitted via'].unique()

# Yep!
# I wonder how companies repsond and how frequently.

(
    narrative_complaints
    .groupby('Company response to consumer')
    ['Complaint ID']
    .count()
)

# How about products?

(
    narrative_complaints
    .groupby('Product')
    ['Complaint ID']
    .count()
    .sort_values(ascending=False)
)


# I tried actually submitting a complaint and there are tick boxes for the
# product the complaint references.
# So writing a "product classifier" probably isn't helpful!

# How many issue types are there, and how balanced are they?

(
  narrative_complaints
  .groupby('Issue')
  ['Complaint ID']
  .count()
  .sort_values(ascending=False)
)

# Hmm. We could try writing a classifier for that, perhaps.
# Let's trim down the dataset.

complaint_issues = (
    narrative_complaints
    [['Consumer complaint narrative', 'Issue']]
    .rename({
        'Consumer complaint narrative': 'complaint',
        'Issue': 'issue'
    }, axis='columns')
)

# What does the free text look like here?
# If it's submitted via a web form, presumably it's pretty messy.

def text_length_plot():
    """
    Function to scope axes for interactive CDSW session.
    """
    ax = complaint_issues.complaint.apply(len).plot(kind='hist', bins = 100)
    ax.set_xlabel('Character length of text')
    return ax
  
text_length_plot()

# Wowzer, that's a pretty wide range of lengths.
# Take a look at the summary stats:

complaint_lengths = complaint_issues.complaint.apply(len)
complaint_lengths.describe()

# Sometimes nothing beats looking at the raw data.
# Just run the code below as often as you like to see the text of a
# randomly selected complaint.

complaint_issues.complaint.sample().values[0]