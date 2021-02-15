"""
Cepstrum Domain
===============

Cepstrum domain features module
"""
import pandas as pd
import numpy as np
from scipy import signal


def cepstrum_coefs_single_axis(X: pd.Series, nb_coefs: int = 24) -> pd.Series:
    """
    Compute N cepstrum coefficients where N is set to `nb_coefs`.

    Parameters
    ----------
    X : pd.Series
        Signal to extract the cepstrum from.
    nb_coefs : int, optional
        Number of cesptrum coefficients to extract. The default is 24.

    Returns
    -------
    pd.Series
        The nb_coefs values.

    """
    # Get the Hannin
    win = signal.get_window("hann", X.shape[0])
    w_sig = np.multiply(X, win)
    spectrum = np.fft.fft(w_sig)
    ceps = np.fft.ifft(np.log10(np.abs(spectrum)))
    ceps_coefs = ceps.real[:nb_coefs]

    return pd.Series(
        data=ceps_coefs, index=["ceps_" + str(i) for i in range(nb_coefs)], name=X.name
    )


def cepstrum_coefs(
    X: pd.DataFrame, n_cepstrum_coeff: int = 24, axis: int = 0
) -> pd.DataFrame:
    """
    Compute N cepstrum coefficients on each column of given DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        Input containing the time series. Shape has to be (n_signals, time)
    n_cepstrum_coeff : int, optional
        Number of cesptrum coefficients to extract. The default is 24.
    axis : int, optional
        Axis. The default is 0.

    Returns
    -------
    pd.DataFrame
        The n_cepstrum_coeff cesptrum coefficients values per signal.

    """
    res = X.apply(
        lambda col: cepstrum_coefs_single_axis(col, nb_coefs=n_cepstrum_coeff),
        axis=axis,
    )
    return res.T


def extract_cepd_features(X: pd.DataFrame, n_cepstrum_coeff: int = 24) -> pd.DataFrame:
    """
    A function that computes Cepstrum Domain features.

    Parameters
    ----------
    X : pd.DataFrame
        Input containing the time series. Shape has to be (n_signals, time)
    n_cepstrum_coeff : int, default=24
        Number of cesptrum coefficients to extract. The default is 24.

    Returns
    -------
    pd.DataFrame
        The n_cepstrum_coeff cesptrum coefficients values per signal.

    See also
    --------
    extract_fd_features
        Extract frequency domain features

    extract_td_features
        Extract time domain features

    extract_wd_features
        Extract wavelet domain features

    extract_all_features
        Extract all features
    """
    return cepstrum_coefs(X.T, n_cepstrum_coeff=n_cepstrum_coeff)
