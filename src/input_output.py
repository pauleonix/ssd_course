import pandas as pd


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


if __name__ == "__main__":
    pass
