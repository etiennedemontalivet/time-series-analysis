"""
Time Domain
===============

All functions below are vectorized, I.E, they can be applied on a whole DataFrame
without having to apply to each column manually.
If we want to add a new function, we need to make sure that it handles DataFrames!
"""
import numpy as np
import pandas as pd
import scipy.stats as stats


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


def rms(X: pd.DataFrame) -> pd.Series:
    """
    Compute Root Mean Square for each column of given DataFrame and return a Series
    """
    return np.sqrt(np.mean(X ** 2)).add_suffix("_rms")


def energy(X: pd.DataFrame) -> pd.Series:
    """
    Compute energy for each column of a given DataFrame and return a Series
    """
    return (X ** 2).sum().add_suffix("_energy")


def iqr(X: pd.DataFrame, axis: int = 0) -> pd.Series:
    """
    Compute IQR for each column of a given DataFrame and return a Series
    """
    res = X.apply(lambda col: stats.iqr(col, interpolation="lower"), axis=axis)
    return res.add_suffix("_iqr")


def mad(X: pd.DataFrame) -> pd.Series:
    """
    Compute Mean Absolute Difference for each column of a given DataFrame and return a Series
    """
    res = np.mean(np.absolute(X - np.mean(X)))
    return res.add_suffix("_mad")


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


def _into_subchunks(x, subchunk_length, every_n=1):
    """
    from `tsfresh sources <https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#_into_subchunks>`:
    Split the time series x into subwindows of length "subchunk_length", starting every "every_n".

    For example, the input data if [0, 1, 2, 3, 4, 5, 6] will be turned into a matrix

        0  2  4
        1  3  5
        2  4  6

    with the settings subchunk_length = 3 and every_n = 2
    """
    len_x = len(x)

    assert subchunk_length > 1
    assert every_n > 0

    # how often can we shift a window of size subchunk_length over the input?
    num_shifts = (len_x - subchunk_length) // every_n + 1
    shift_starts = every_n * np.arange(num_shifts)
    indices = np.arange(subchunk_length)

    indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)
    return np.asarray(x)[indexer]


def sample_entropy_single_axis(x: pd.Series, m: int = 2, eta: float = 0.2):
    """
    Calculate and return sample entropy of x.

    Parameters
    ----------
    x : pd.Series
        Time serie to extract sample entropy from.
    m : int, optional
        Length of subvectors. The default is 2.
    eta : float, optional
        Ratio to be multiplied by the std of x to get the tolerance. The default is 0.2.

    Returns
    -------
    float
        Sample entropy of time serie.

    """
    x = np.array(x)

    # if one of the values is NaN, we can not compute anything meaningful
    if np.isnan(x).any():
        return np.nan

    # 0.2 is a common value for r, according to wikipedia...
    tolerance = eta * np.std(x)

    # Split time series and save all templates of length m
    xm = _into_subchunks(x, m)
    B = np.sum(  # pylint: disable=invalid-name
        [np.sum(np.abs(xmi - xm).max(axis=1) <= tolerance) - 1 for xmi in xm]
    )

    # Similar for computing A
    xmp1 = _into_subchunks(x, m + 1)
    A = np.sum(  # pylint: disable=invalid-name
        [np.sum(np.abs(xmi - xmp1).max(axis=1) <= tolerance) - 1 for xmi in xmp1]
    )

    # Return SampEn
    return -np.log(A / B)


def sample_entropy(X: pd.DataFrame, m: int = 4, eta: float = 0.2) -> pd.Series:
    """
    Calculate and return sample entropies of X.

    Parameters
    ----------
    X : pd.DataFrame
        Time series to extract sample entropy from.
    m : int, optional
        Length of subvectors. The default is 4.
    eta : float, optional
        Ratio to be multiplied by the std of x to get the tolerance. The default is 0.2.

    Returns
    -------
    pd.Series
        Sample entropy of time series.

    """
    res = X.apply(lambda col: sample_entropy_single_axis(col, m=m, eta=eta), axis=0)
    return res.add_suffix("_sampen")


def extract_td_features(
    X: pd.DataFrame,
    sampen_m: int=2,
    sampen_eta: float=0.2,
    prefix: str=None
) -> pd.DataFrame:
    """
    A function that computes Time Domain features.

    Parameters
    ----------
    X : pd.DataFrame
        Time series to extract sample entropy from.
    sampen_m : int, default=2
        Length of subvectors for sample entropy computation. The default is 2.
    sampen_eta : float, default=0.2
        Ratio to be multiplied by the std of x to get the tolerance. The default is 0.2.
    prefix : str, default=None
        A prefix to add to features name. If None, no prefix is added. The
        default is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the time domain features per time serie.

    See also
    --------
    extract_cepd_features
        Extract cepstrum domain features

    extract_fd_features
        Extract frequency domain features

    extract_wd_features
        Extract wavelet domain features

    extract_all_features
        Extract all features
    """
    # List all time domain features function that does not have parameters
    funcs = [
        mean,
        var,
        std,
        median,
        maximum,
        minimum,
        rms,
        argmax,
        argmin,
        energy,
        skewness,
        kurtosis,
        iqr,
        mad,
    ]
    out = []
    for func in funcs:
        out.append(pd.Series(data=func(X.T).values, name=func.__name__, index=X.index))

    # Time domain features with parameters
    out.append(
        pd.Series(
            data=sample_entropy(X=X.T, m=sampen_m, eta=sampen_eta).values,
            name="sampen",
            index=X.index,
        )
    )

    # use a prefix in feature name
    if prefix is None or prefix == '':
        prefix=''
    elif prefix[-1] != '_':
        prefix += '_'

    return pd.concat(out, axis=1).add_prefix(prefix)
