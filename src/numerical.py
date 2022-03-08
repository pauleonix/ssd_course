"""This module handles all specifically numerical analysis code."""
import common
import input_output as io
import pandas as pd
import numpy as np


def fourier_transform(
    data: pd.DataFrame,
    drop_time: bool = True,
    drop_constant: bool = True,
    complex: bool = False,
) -> np.array:
    """Calculates the discrete Fourier transforamtion of the given data.

    Args:
        data: pandas Dataframe, containing the data.
        drop_time: bool (default = True) If True the time column will be
        dropped before calculating the transformation.
        drop_constant: bool (default = True) If True any constant column will be
        dropped before calculating the transformation.
        complex: bool (defaut= False) If True the function returns the fourier
        calculation as complex numbers. Otherwise it only returns the real part.

    Returns:
        numpy array with transformed values
    """
    if drop_time and "time" in data.columns:
        data = data.drop("time", axis=1)
    if drop_constant:
        data = common.drop_constants(data)
    fft = np.fft.fft(data.values, axis=1)
    if not complex:
        fft = np.real(fft)
    return fft


def fourier_freq(data: pd.DataFrame, time_column_name="time", fft_kwargs: dict = {}):
    """Calculates the Fourier frequency of the given data.

    Args:
        data: pandas Dataframe, containing the data.
        time_column_name: str (default "time"), column name in data that
        specifies the time.
        fft_kwargs: dict key word arguments for common.fourier_transform.
    """
    if time_column_name not in data.columns:
        raise KeyError(
            """"{}" not found in data.columns.
             The dataset needs a time scale.
            You can specify the time column name
            using the time_column_name keyword parameter.""".format(
                time_column_name
            )
        )
    sample_spacing = data.time[1] - data.time[0]  # calculates 1 time step
    window_length = fourier_transform(data, **fft_kwargs).shape[0]
    fft_freq = np.fft.fftfreq(window_length, sample_spacing)
    return fft_freq


if __name__ == "__main__":
    df = io.read_table("./data/efield.t")
    print(len(fourier_freq(df)))
