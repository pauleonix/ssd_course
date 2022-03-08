import numpy as np
import pandas as pd


def sorted_pearson_corr(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between columns and sort them by absolute value.

    Calculate Pearson correlations between all pairs of distinct columns and
    sort them in descending order by their absolute value.
    If there is a column named 'time' in data, it will be ignored.

    Args:
        data: pandas.DataFrame -> columns will be correlated

    Returns:
        pandas.DataFrame
    """
    corr_matrix = data.corr(method="pearson")

    if "time" in data.columns:
        corr_matrix.drop(index="time", columns="time", inplace=True)

    sorted_corr = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .unstack()
        .dropna()
        .sort_values(key=np.abs, ascending=False)
    )
    return sorted_corr


if __name__ == "__main__":
    pass
