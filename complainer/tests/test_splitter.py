import pytest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from complainer.splitter import train_dev_test_split


@pytest.fixture
def df():
    return pd.DataFrame({'X': range(100), 'y': range(100)})


class TestTrainDevTestSplit:
    def test_train_dev_test_split_returns_six_objects(self, df):
        tdts = train_dev_test_split(df)
        assert len(tdts) == 3
    
    def test_total_length_of_splits_unchanged(self, df):
        train, dev, test = train_dev_test_split(df)
        assert (len(train) + len(dev) + len(test)) == len(df)

    def test_splits_have_expected_lengths(self, df):
        train, dev, test = train_dev_test_split(df)
        assert len(train) == 60
        assert len(dev) == 20
        assert len(test) == 20
        
    def test_no_duplicates_in_splits(self, df):
        train, dev, test = train_dev_test_split(df)
        tdt = pd.concat([train, dev, test])
        assert len(tdt.drop_duplicates()) == len(tdt)
        
    def test_same_random_seed_returns_same_split(self, df):
        train_a, dev_a, test_a = train_dev_test_split(df, random_state=42)
        train_b, dev_b, test_b = train_dev_test_split(df, random_state=42)
        assert_frame_equal(train_a, train_b)
        assert_frame_equal(dev_a, dev_b)
        assert_frame_equal(test_a, test_b)
        
    def test_combined_fractions_greater_than_one_throw(self, df):
        with pytest.raises(ValueError):
            train_dev_test_split(df, dev_fraction=0.5, test_fraction = 0.51)