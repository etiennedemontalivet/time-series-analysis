"""
Frequency Domain
================

Frequency domain features module
"""
from typing import Callable
import numpy as np
import pandas as pd
from scipy import signal


def powerband_single_axis(
    X: pd.Series,
    fs: int = 1000,
    n_powerband_bins: int = 10,
    powerband_explicit_freq_names: bool = True,
) -> pd.Series:
    """
    Compute power band coefficients of a single Series using Welch method.
    See `<here https://en.wikipedia.org/wiki/Welch%27s_method>__` for more details.

    Parameters
    ----------
    X : pd.Series
        Signal to extract the powerbands from.
    fs : int, default=1000
        Sample rate in Hz. The default is 1000.
    n_powerband_bins : int, default=10
        Number of powerbands to compute. The default is 10.
    powerband_explicit_freq_names : bool, default=True
        If True, the frequency bands are included in the feature name, else
        a counter is used. The default is True.

    Returns
    -------
    pd.Series
        The powerbands features values.

    """
    nperseg = 2 * (n_powerband_bins - 1)
    f, power_band = signal.welch(X, window="hann", nperseg=nperseg, fs=fs)
    if powerband_explicit_freq_names:
        freq_bins = [str(int(f[i])) for i in range(n_powerband_bins)]
    else:
        freq_bins = [str(i) for i in range(n_powerband_bins)]
    return pd.Series(
        data=power_band,
        index=["powerband_" + freq_bins[i] for i in range(len(f))],
        name=X.name,
    )


def powerband(
    X: pd.DataFrame,
    fs=1000,
    n_powerband_bins: int = 10,
    powerband_explicit_freq_names: bool = True,
) -> pd.DataFrame:
    """
    Compute power band coefficients of a DataFrame using Welch method.
    See https://en.wikipedia.org/wiki/Welch%27s_method

    Parameters
    ----------
    X : pd.DataFrame
        Input containing the time series. Shape has to be (n_signals, time)
    fs : int, default=1000
        Sample rate in Hz. The default is 1000.
    n_powerband_bins : int, default=10
        Number of powerbands to compute. The default is 10.
    powerband_explicit_freq_names : bool, default=True
        If True, the frequency bands are included in the feature name, else
        a counter is used. The default is True.

    Returns
    -------
    pd.Series
        The powerbands features values.

    """
    res = X.apply(
        lambda col: powerband_single_axis(
            col,
            fs=fs,
            n_powerband_bins=n_powerband_bins,
            powerband_explicit_freq_names=powerband_explicit_freq_names,
        ),
        axis=0,
    )
    return res.T


def fd_max_argmax_energy_single_axis(
    X: pd.Series,
    window: str = "hann",
    skip_coefs: int = 1,
    last_coeff: int = None,
    filtering_func: Callable = None,
) -> pd.Series:
    """
    Compute the energy, the max and argmax of the magnitude of the fourier
    transform of the time series.

    Parameters
    ----------
    X : pd.Series
        Signal to extract the features from.
    window : str, default="hann"
        The type of window to use for windowing. The default is "hann".
    skip_coefs : int, default=1
        Number of coefficient to skip for the max/argmax computation. The default is 1.
    last_coeff : int, default=None
        The last fft coeff to take into account for the the max/argmax computation.
        If None, no part of the magnitude is removed from the end.
        The default is None.
    filtering_func : Callable, default=None
        A filter on the magnitude could be applied before max/argmax computation.
        The default is None.

    Notes
    -----
    Filtering, skip_coefs and last_coeff are NOT used for energy computation.

    Returns
    -------
    pd.Series
        Series containing max, argmax and energy of the time serie.

    """
    if last_coeff is None:
        last_coeff = X.shape[0] - 1

    win = signal.get_window(window, X.shape[0])
    w_sig = np.multiply(X, win)
    spectrum = np.fft.rfft(w_sig)
    mag = np.abs(spectrum)

    # Apply filtering on magnitude ?
    # ex: signal.filtfilt(b, a, mag)
    if filtering_func is None:
        mag_filtered = mag
    else:
        mag_filtered = filtering_func(mag)

    argmax_coef = np.argmax(mag_filtered[skip_coefs:last_coeff]) + skip_coefs
    max_coef = mag_filtered[argmax_coef]
    energy = np.sum(mag ** 2) / X.shape[0]
    return pd.Series(
        {"fd_max": max_coef, "fd_argmax": argmax_coef, "fd_energy": energy}, name=X.name
    )


def fd_max_argmax_energy(
    X: pd.DataFrame,
    window: str = "hann",
    skip_coefs: int = 1,
    last_coeff: int = None,
    filtering_func: Callable = None,
) -> pd.DataFrame:
    """
    Compute the energy, the max and argmax of the magnitude of the fourier
    transform of the time series.

    Parameters
    ----------
    X : pd.DataFrame
        Input containing the time series. Shape has to be (n_signals, time)
    window : str, default="hann"
        The type of window to use for windowing. The default is "hann".
    skip_coefs : int, default=1
        Number of coefficient to skip for the max/argmax computation. The default is 1.
    last_coeff : int, default=None
        The last fft coeff to take into account for the the max/argmax computation.
        If None, no part of the magnitude is removed from the end.
        The default is None.
    filtering_func : Callable, default=None
        A filter on the magnitude could be applied before max/argmax computation.
        The default is None.

    Notes
    -----
    Filtering, skip_coefs and last_coeff are NOT used for energy computation.

    Returns
    -------
    pd.Series
        Series containing max, argmax and energy of the time serie.

    """
    res = X.apply(
        lambda col: fd_max_argmax_energy_single_axis(
            col,
            window=window,
            skip_coefs=skip_coefs,
            last_coeff=last_coeff,
            filtering_func=filtering_func,
        ),
        axis=0,
    )
    return res.T


# pylint: disable=too-many-arguments
def extract_fd_features(
    X: pd.DataFrame,
    fs: int,
    n_powerband_bins: int = 10,
    powerband_explicit_freq_names: bool = True,
    fft_window: str = "hann",
    fft_max_argmax_skip_coeffs: int = 1,
    fft_max_argmax_last_coeffs: int = None,
    fft_filtering_func: Callable = None,
    prefix: str=None
) -> pd.DataFrame:
    """
    A function that computes Frequency Domain features.

    Parameters
    ----------
    X : pd.DataFrame
        Input containing the time series. Shape has to be (n_signals, time)
    fs : int
        Sample rate in Hz.
    n_powerband_bins : int, default=10
        Number of powerbands to compute. The default is 10.
    powerband_explicit_freq_names : bool, default=True
        If True, the frequency bands are included in the feature name, else
        a counter is used. The default is True.
    fft_window : str, default="hann"
        The type of window to use for windowing. The default is "hann".
    fft_max_argmax_skip_coeffs : int, default=1
        Number of coefficients to skip for the max/argmax computation. The default is 1.
    fft_max_argmax_last_coeffs : int, default=None
        The last fft coeff to take into account for the the max/argmax computation.
        If None, no part of the magnitude is removed from the end.
        The default is None.
    fft_filtering_func : Callable, default=None
        A filter on the magnitude could be applied before max/argmax computation.
        The default is None.
    prefix : str, default=None
        A prefix to add to features name. If None, no prefix is added. The
        default is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the frequency features per time serie.

    See also
    --------
    extract_cepd_features
        Extract cepstrum domain features

    extract_td_features
        Extract time domain features

    extract_wd_features
        Extract wavelet domain features

    extract_all_features
        Extract all features
    """
    # use a prefix in feature name
    if prefix is None or prefix == '':
        prefix=''
    elif prefix[-1] != '_':
        prefix += '_'

    return pd.concat(
        [
            powerband(
                X=X.T,
                fs=fs,
                n_powerband_bins=n_powerband_bins,
                powerband_explicit_freq_names=powerband_explicit_freq_names,
            ),
            fd_max_argmax_energy(
                X=X.T,
                window=fft_window,
                skip_coefs=fft_max_argmax_skip_coeffs,
                last_coeff=fft_max_argmax_last_coeffs,
                filtering_func=fft_filtering_func,
            ),
        ],
        axis=1,
    ).add_prefix(prefix)
