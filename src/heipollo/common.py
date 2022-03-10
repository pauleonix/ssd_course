"""This module handles all common functions that are needed for both statistical
 and numerical analysis"""

import pandas as pd


def drop_constants(data: pd.DataFrame, threshold: float = 1e-4):
    """Takes a pandas Dataframe and drops all columns that have a varaince that
    is lower than a given threshold.

    Args:
        data: pandas Dataframe containingthe dataframe.
        theshold: float (default = 1e-4) ariance threshold.
    Returns:
        pandas Dataframe

    """
    const_column_name = [
        label for label, content in data.items() if content.var() < threshold
    ]
    return data.drop(const_column_name, axis=1)
