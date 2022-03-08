import input_output as io
import common
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
