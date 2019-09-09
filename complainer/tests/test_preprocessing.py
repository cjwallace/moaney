import pytest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from complainer.preprocessing import encode_targets, filter_rename_mortgages


@pytest.fixture
def ff():
    """ff = fruit frame"""
    return pd.DataFrame({'fruit': ['apple', 'orange', 'banana', 'pear'] * 25})


target_encoding = {'apple': 'a',
                   'orange': 'o',
                   'banana': 'b',
                   'pear': 'p'}


reverse_encoding = {v:k for k, v in target_encoding.items()}


class TestEncodeTargets:
    def test_encode_targets_returns_same_shape_df(self, ff):
        encoded = encode_targets(ff, 'fruit', target_encoding)
        assert ff.shape == encoded.shape

    def test_encoding_replaces_values(self, ff):
        encoded = encode_targets(ff, 'fruit', target_encoding)
        new_target_values = set(encoded['fruit'].values)
        desired_target_values = target_encoding.values()
        assert new_target_values.issubset(desired_target_values)

    def test_encoding_and_decoding_return_original(self, ff):
        encoded = encode_targets(ff, 'fruit', target_encoding)
        decoded = encode_targets(encoded, 'fruit', reverse_encoding)
        assert_frame_equal(ff, decoded)


@pytest.fixture
def mf():
    """mf = mortgage frame"""
    return pd.DataFrame(
      {'Product': ['Mortgage', 'Not Mortgage', 'Mortgage'],
       'Consumer complaint narrative': ['Blah', 'Yadda', 'Harumph'],
       'Issue': ['a_mortgage_issue', 'not_an_issue', 'another_mortgage_issue']}
    )


class TestFilterAndRenameMortgages:
    def test_filtering_reduces_number_of_rows(self, mf):
        mortgages = filter_rename_mortgages(mf)
        assert len(mortgages) < len(mf)

    def test_filtered_df_has_only_expected_two_columns(self, mf):
        mortgages = filter_rename_mortgages(mf)
        assert set(mortgages.columns) == {'issue', 'complaint'}

    def test_filtered_df_has_only_mortgage_issues(self, mf):
        mortgages = filter_rename_mortgages(mf)
        assert set(mortgages.issue) == {'a_mortgage_issue',
                                        'another_mortgage_issue'}