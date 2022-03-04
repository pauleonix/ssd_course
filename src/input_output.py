import pandas as pd
import numpy as np
from typing import Tuple  # added for better type hints


def read_table(
    fname: str, sep: str = r"(?<!\()\s\s+", csv_read_kwargs: dict = {}
) -> pd.DataFrame:
    """
    Reads the content of a data file as pandas DataFrame.

    fname: str -> path to file
    sep: str ->  default: r"(?<!\()\s\s+"  Regex-pattern marking the
                delimer of the file.
    csv_read_kwargs dict: -> Dictionary of key word arguments that is
                passed to pandas.read_csv()

    Returns:
    pandas.DataFrame
    """
    if fname.endswith(".dat"):  # special case for the dat file
        idx_col = [0, 1]
    else:
        idx_col = 0

    return pd.read_csv(
        fname, index_col=idx_col, sep=sep, engine="python", **csv_read_kwargs
    )


def read_table_as_numpy(
    fname: str, sep: str = r"(?<!\()\s\s+", pandas_read_kwargs={}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a data table as pandas DataFrame and then return a tuple the
    values of the frame as c-contigious numpy array.


    fname: str -> path to file
    sep: str ->  default: r"(?<!\()\s\s+"  Regex-pattern marking the
                 delimer of the file.
    pandas_read_kwargs dict: -> Dictionary of key word arguments that
                                is passed to pandas.read_csv()

    Returns:
    values as numpy.ndarray, index as numpy.ndarray, columns as numpy.ndarray
    """
    df = read_table(fname, sep, csv_read_kwargs=pandas_read_kwargs)
    return (
        np.ascontiguousarray(df.values),
        np.array(df.index),
        np.array(df.columns),
    )


if __name__ == "__main__":
    pass
