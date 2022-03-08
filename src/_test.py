import input_output as io
import common
import numerical
import pytest


class Test_IO:
    """Tests for the input output module."""

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
        """Tests the import of the different test data."""
        assert io.read_table(fname).shape == expected_shape


class Test_Common:
    """Tests for the Common function module"""

    def test_thresholding(self):
        thresholded = common.drop_constants(io.read_table("data/expec.t"))
        assert thresholded.shape == (101, 3)


class Test_Numerical:
    """Tests for the numerical module."""

    @pytest.mark.parametrize("fname", ["./data/efield.t"])
    def test_fft(self, fname):
        df = io.read_table(fname)
        assert numerical.fft(df).shape == (101, 1)

    @pytest.mark.parametrize("fname", ["./data/efield.t"])
    def test_fftfreq(self, fname):
        df = io.read_table(fname)
        len(common.fourier_freq(df)) == 101
