import pandas as pd
import numpy as np
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
