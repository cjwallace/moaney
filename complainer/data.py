"""
Utilities related to loading and handling complaints data
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def train_dev_test_split(df,
    dev_fraction=0.2,
    test_fraction=0.2,
    stratify=None,
    random_state=None):
    """
    Split a pandas.DataFrame into three subdataframes with distinct samples of
    the data.
    
    Parameters
    ----------
    df : An indexable with same length / shape[0], but probably a
         pandas.DataFrame
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas
        dataframes.
        As sklearn.model_selection.train_test_split.
    dev_fraction : float (default=0.2)
        Fraction of the index to split into the dev set.
        Must be greater than zero and less than 1.
        Sum with `test_fraction` must also be less than 1.
    test_fraction : float (default=0.2)
        Fraction of the index to split into the test set.
        Must be greater than zero and less than 1.
        Sum with `dev_fraction` must also be less than 1.
    stratify: Array-like or None (default=None)
        If not None, data is split in a stratified fashion, using this as the
        class labels.
        Passed directly to sklearn.model_selection.train_test_split.
    random_state : int, RandomState instance or None, optional (default=None)
        Seed for random number generator.
        As sklearn.model_selection.train_test_split.
    Returns
    -------
    train : pandas.DataFrame (or indexable)
        Subset of df, containing fraction
        1 - `dev_fraction` - `test_fraction`
        of the rows.
    dev : pandas.DataFrame (or indexable)
        Subset of df, containing `dev_fraction` of the rows.
    test : pandas.DataFrame (or indexable)
        Subset of df, containing `test_fraction` of the rows.
    """
    
    rest_fraction = dev_fraction + test_fraction
    
    if rest_fraction > 1:
        raise(ValueError(
            """
            The sum of `dev_fraction` and `test_fraction`
            must be less than one.
            """
        ))
    
    train, rest = train_test_split(
        df,
        stratify=stratify,
        test_size=rest_fraction,
        random_state=random_state
    )
    
    dev, test = train_test_split(
        rest,
        stratify=stratify,
        test_size=test_fraction / rest_fraction,
        random_state=random_state
    )

    return train, dev, test


def encode_targets(df, target_column, target_encoding_dict):
    """
    Replaces values in `df` target_column with the corresponding values
    specified in the encoding_dict
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame on which to operate
    target_column : string
        Name of the target column to replace values of
    encoding_dict : string
        Dictionary containing current target value to desired value mapping
    Returns
    -------
    df : pandas.DataFrame (or indexable)
        Input DataFrame, with values of target column replaced
    """
    df = df.copy() # do not mutate original frame
    df[target_column] = df[target_column].apply(
        lambda target: target_encoding_dict[target]
    )
    return df



def filter_rename_mortgages(df):
    """
    Filter dataset down to only mortgage products, only the complaint and issue
    columns (feature and target, respectively), and rename those columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame on which to operate. Must contain "Product", "Issue", and
        "Consumer complaint narrative" columns.
    Returns
    -------
    df : pandas.DataFrame (or indexable)
        Filtered and renamed version of input DataFrame.
    """
    mortgages = (
        df
        [df['Product'] == 'Mortgage']
        [['Consumer complaint narrative', 'Issue']]
        .rename({
            'Consumer complaint narrative': 'complaint',
            'Issue': 'issue'
            },
            axis='columns')
        .reset_index(drop=True)
    )
    return mortgages


target_encoding_dict = {
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