"""
Utility for splitting dataset into train, dev, and test subsets.
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