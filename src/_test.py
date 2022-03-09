import input_output as io
import common
import numerical
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt


class Test_Input:
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


class Test_Output:
    @pytest.fixture
    def setup_plot(self):
        df = io.read_table("./data/efield.t")
        fig = plt.figure()
        ax = io.plot_fourier_frequency(
            df,
            x_axis_label="Frequency",
            y_axis_label="something",
        )
        return fig, ax

    def test_fft_graph(self, setup_plot):
        assert type(setup_plot[0]) == mpl.figure.Figure
        assert setup_plot[0] == setup_plot[1].get_figure()
        assert setup_plot[1].xaxis.label.get_text() == "Frequency"
        assert setup_plot[1].yaxis.label.get_text() == "something"


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
        assert numerical.fourier_transform(df).shape == (101,)

    @pytest.mark.parametrize("fname", ["./data/efield.t"])
    def test_fftfreq(self, fname):
        df = io.read_table(fname)
        assert len(numerical.fourier_freq(df)) == 101
