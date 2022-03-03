import pandas as pd


def read_table(fname: str, sep: str = r"(?<!\()\s\s+") -> pd.DataFrame:
    """

    fname -> str: path to file
    """
    if fname.endswith(".dat"):
        idx_col = [0, 1]
    else:
        idx_col = 0

    # solution for npop
    return pd.read_csv(fname, index_col=idx_col, sep=sep, engine="python")


if __name__ == "__main__":
    x = read_table("data/npop.t")
    print(x)
