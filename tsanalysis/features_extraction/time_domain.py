"""
All functions below are vectorized, I.E, they can be applied on a whole DataFrame without having to apply
to each column manually.
If we want to add a new function, we need to make sure that it handles DataFrames!
"""
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from PyAstronomy import pyaC
from scipy.signal import butter, filtfilt


def mean(X: pd.DataFrame) -> pd.Series:
    """
    Compute mean for each column and return a Series
    """
    return X.mean().add_suffix("_mean")


def var(X: pd.DataFrame) -> pd.Series:
    """
    Compute variance for each column and return a Series
    """
    return np.var(X).add_suffix("_var")


def std(X: pd.DataFrame) -> pd.Series:
    """
    Compute standard deviation for each column and return a Series
    """
    return np.std(X).add_suffix("_std")


def median(X: pd.DataFrame) -> pd.Series:
    """
    Compute median for each column and return a Series
    """
    return X.median().add_suffix("_median")


def maximum(X: pd.DataFrame) -> pd.Series:
    """
    Find maximum value for each column and return a Series
    """
    return X.max().add_suffix("_max")


def minimum(X: pd.DataFrame) -> pd.Series:
    """
    Find minimum value for each column and return a Series
    """
    return X.min().add_suffix("_min")


def RMS(X: pd.DataFrame) -> pd.Series:
    """
    Compute Root Mean Square for each column of given DataFrame and return a Series
    """
    return np.sqrt(np.mean(X ** 2)).add_suffix("_RMS")


def energy(X: pd.DataFrame) -> pd.Series:
    """
    Compute energy for each column of a given DataFrame and return a Series
    """
    return (X ** 2).sum().add_suffix("_energy")


def IQR(X: pd.DataFrame, axis: int = 0) -> pd.Series:
    """
    Compute IQR for each column of a given DataFrame and return a Series
    """
    res = X.apply(lambda col: stats.iqr(col, interpolation="lower"), axis=axis)
    return res.add_suffix("_IQR")


def MAD(X: pd.DataFrame) -> pd.Series:
    """
    Compute Mean Absolute Difference for each column of a given DataFrame and return a Series
    """
    res = np.mean(np.absolute(X - np.mean(X)))
    return res.add_suffix("_MAD")


def argmax(X: pd.DataFrame) -> pd.Series:
    """
    Find position of maximum value for each column in given DataFrame and return a Series.
    This handles both multi-indexed and single-indexed dataframes.
    """
    res = pd.Series(np.argmax(X.values, axis=0), index=X.columns)
    return res.add_suffix("_argmax")


def argmin(X: pd.DataFrame) -> pd.Series:
    """
    Find position of minimum value for each column in given DataFrame and return a Series
    This handles both multi-indexed and single-indexed dataframes.
    """
    res = pd.Series(np.argmin(X.values, axis=0), index=X.columns)
    return res.add_suffix("_argmin")


def skewness(X: pd.DataFrame) -> pd.Series:
    """
    Compute skewness for each column and return a Series
    """
    return pd.Series(stats.skew(X), index=X.columns).add_suffix("_skewness")


def kurtosis(X: pd.DataFrame) -> pd.Series:
    """
    Compute kurtosis for each column and return a Serie
    """
    return pd.Series(stats.kurtosis(X), index=X.columns).add_suffix("_kurtosis")


def extract_td_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    A function that computes Time Domain features.
    """
    # List all features extraction function
    funcs = [
        mean,
        var,
        std,
        median,
        maximum,
        minimum,
        RMS,
        argmax,
        argmin,
        energy,
        skewness,
        kurtosis,
        IQR,
        MAD
    ]

    out = []
    for func in funcs:
        out.append(pd.Series(data=func(X.T).values, name=func.__name__, index=X.index))
    return pd.concat(out, axis=1)