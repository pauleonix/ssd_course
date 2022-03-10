import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import numerical

from typing import Tuple  # added for better type hints


def read_table(
    fname: str, sep: str = r"(?<!\()\s\s+", csv_read_kwargs: dict = {}
) -> pd.DataFrame:
    """Reads the content of a data file as pandas DataFrame.

    Args:
        fname: str -> path to file

        sep: str ->  default: r"(?<!\\()\s\s+"  Regex-pattern marking the delimiter of
        the file.

        csv_read_kwargs: dict -> Dictionary of key word arguments that is passed to
        pandas.read_csv()

    Returns:
        pandas.DataFrame
    """

    return pd.read_csv(fname, header=0, sep=sep, engine="python", **csv_read_kwargs)


def read_table_as_numpy(
    fname: str, sep: str = r"(?<!\()\s\s+", pandas_read_kwargs={}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads a data table as pandas DataFrame and then returns a tuple which contains:
     the values of the frame the index as numpy array and the column names as
     c-contigious numpy array.

    Args:
        fname: str -> path to file

        sep: str ->  default: r"(?<!\\()\s\s+"  Regex-pattern marking the delimiter of
        the file.

        pandas_read_kwargs: dict -> Dictionary of key word arguments that is
        passed to pandas.read_csv()

    Returns:
        Tuple(Values as numpy.ndarray, index as numpy.ndarray, columns as numpy.ndarray)
    """
    df = read_table(fname, sep, csv_read_kwargs=pandas_read_kwargs)
    i_cols = "time"
    if fname.endswith(".dat"):
        i_cols = ["i", "j"]
    df.set_index(i_cols, inplace=True)
    return (
        np.ascontiguousarray(df.values),
        np.array(df.index),
        np.array(df.columns),
    )


def _create_plot_folder() -> None:
    """Creates a plot folder if needed."""
    directory = os.path.join(os.getcwd(), "/plots")
    if not os.path.exists(directory):
        os.mkdir(directory)


def plot_fourier_frequency(
    data: pd.DataFrame,
    ax=None,
    x_axis_label="Frequency",
    y_axis_label=None,
    **plt_kwargs
):
    """Creates a fourier frequency graph.

    Args:
        data: pandas.DataFrame containing the data.
        ax: matplotlib axis (default = None) axis to draw the graph on. Will use
        default axis if None is given.
        x_axis_label: str (default = "Frequency") labet for the x-axis.
        y_axis_label: str (default = None) labet for the y-axis.
        **pltkwargs: key word arguments forwarded to seaborn.plot
    """
    _create_plot_folder()
    if ax is None:
        ax = plt.gca()
    shift_freq = np.fft.fftshift(numerical.fourier_transform(data))
    field_shift_freq = np.fft.fftshift(numerical.fourier_freq(data))
    sns.lineplot(x=field_shift_freq, y=shift_freq, ax=ax, **plt_kwargs)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    return ax


if __name__ == "__main__":
    data = read_table("./data/efield.t")
    fig = plt.figure()
    ax = plot_fourier_frequency(data)
    print(ax.xaxis.label.get_text())
    plt.show()
