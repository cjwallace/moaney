"""
Functions for preprocessing mortgage complaints data.
"""

import pandas as pd


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