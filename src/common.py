"""This module handles all common functions that are needed for both statistical
 and numerical analysis"""

import pandas as pd


def drop_constants(data: pd.DataFrame, threshold: float = 1e-4):
    """ """
    const_column_name = [
        label for label, content in data.items() if content.var() < threshold
    ]
    return data.drop(const_column_name, axis=1)
