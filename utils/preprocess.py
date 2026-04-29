import numpy as np
import pandas as pd
from sklearn import preprocessing


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize all columns; preserves index and column names."""
    scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)


def equalize_length(df: pd.DataFrame, max_length: int) -> pd.DataFrame:
    """Pad a DataFrame to `max_length` rows by repeating the last row."""
    arr = df.values
    if len(arr) < max_length:
        pad = np.tile(arr[-1], (max_length - len(arr), 1))
        arr = np.vstack([arr, pad])
    return pd.DataFrame(arr, columns=df.columns)
