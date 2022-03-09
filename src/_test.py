import input_output as io
import common
import statistical as stat
import numpy as np
from numpy.random import default_rng
import pandas as pd
from collections import Counter
import pytest


class Test_IO:
    @pytest.mark.parametrize(
        "fname,expected_shape",
        [
            ("data/efield.t", (101, 4)),
            ("data/expec.t", (101, 6)),
            ("data/npop.t", (101, 39)),
            ("data/nstate_i.t", (101, 481)),
            ("data/table.dat", (4950, 8)),
        ],
    )
    def test_table_input(self, fname, expected_shape):
        assert io.read_table(fname).shape == expected_shape


class Test_Common:
    """Tests for the Common function mdule"""

    def test_thresholding(self):
        thresholded = common.drop_constants(io.read_table("data/expec.t"))
        assert thresholded.shape == (101, 3)


class Test_Statistical:
    """Tests the statistical module."""

    def two_cols():
        return pd.DataFrame(data={"col1": [1.0, 2.0, 1.0], "col2": [3.0, 4.0, 5.0]})

    def two_cols_and_time():
        return pd.DataFrame(
            data={
                "col1": [1.0, 2.0, 1.0],
                "col2": [3.0, 4.0, 5.0],
                "time": [0.0, 1.0, 2.0],
            }
        )

    def time_and_two_cols():
        return pd.DataFrame(
            data={
                "time": [0.0, 1.0, 2.0],
                "col1": [1.0, 2.0, 1.0],
                "col2": [3.0, 4.0, 5.0],
            }
        )

    def two_cols_and_const():
        return pd.DataFrame(
            data={
                "col1": [1.0, 2.0, 1.0],
                "col2": [3.0, 4.0, 5.0],
                "col3": [6.0, 6.0, 6.0],
            }
        )

    def three_cols():
        return pd.DataFrame(
            data={
                "col1": [1.0, 2.0, 1.0],
                "col2": [3.0, 4.0, 5.0],
                "col3": [-6.0, -7.0, -6.0],
            }
        )

    def four_cols():
        return pd.DataFrame(
            data={
                "col1": [1.0, 2.0, 1.0],
                "col2": [3.0, 4.0, 5.0],
                "col3": [-6.0, -7.0, -6.0],
                "col4": [8.0, 9.0, 10.0],
            }
        )

    def random_five_cols():
        return pd.DataFrame(default_rng().standard_normal((1000, 5)))

    def random_hundred_cols():
        return pd.DataFrame(default_rng().standard_normal((50, 100)))

    @pytest.mark.parametrize(
        "data,effective_cols",
        [
            (two_cols(), 2),
            (two_cols_and_time(), 2),
            (time_and_two_cols(), 2),
            (two_cols_and_const(), 2),
            (three_cols(), 3),
            (four_cols(), 4),
            (random_five_cols(), 5),
            (random_hundred_cols(), 100),
        ],
    )
    def test_sorted_pearson_corr_shape(self, data, effective_cols):
        corrs = stat.sorted_pearson_corr(data)

        assert len(corrs.shape) == 1
        assert corrs.shape[0] == (effective_cols - 1) * effective_cols / 2
        assert np.max(corrs.values) <= 1.0
        assert np.min(corrs.values) >= -1.0
        assert all([a != b for a, b in corrs.index])

        counts = Counter([col for cols in corrs.index for col in cols]).values()
        assert all([count == effective_cols - 1 for count in counts])

        # sorted in descending order
        assert np.all(np.abs(corrs.values[:-1]) >= np.abs(corrs.values[1:]))
