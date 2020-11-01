"""
All functions below are vectorized, I.E, they can be applied on a whole DataFrame without having to apply
to each column manually.
If we want to add a new function, we need to make sure that it handles DataFrames!
"""
import numpy as np
import pandas as pd
from scipy import signal
from typing import Callable

def powerband_single_axis(
    X: pd.Series,
    fs: int = 1000,
    n_powerband_bins: int = 10,
    powerband_explicit_freq_names: bool=True
) -> pd.Series:
    """
    Compute power band coefficients of a single Series
    """
    nperseg = 2 *  ( n_powerband_bins - 1 )
    f, power_band = signal.welch(X, window="hann", nperseg=nperseg, fs=fs)
    if powerband_explicit_freq_names:
        freq_bins = [ str(int(f[i])) for i in range(n_powerband_bins) ]
    else:
        freq_bins = [ str(i) for i in range(n_powerband_bins)]
    return pd.Series(data=power_band, index=["powerband_"+freq_bins[i] for i in range(len(f))], name=X.name)


def powerband(
    X: pd.DataFrame, 
    fs=1000,
    n_powerband_bins: int=10,
    powerband_explicit_freq_names: bool=True
) -> pd.DataFrame:
    """
    Compute powerband coefficient for each column of a DataFrame and returns a Series
    """
    res = X.apply(
        lambda col: powerband_single_axis(
            col,
            fs=fs,
            n_powerband_bins=n_powerband_bins,
            powerband_explicit_freq_names=powerband_explicit_freq_names
        ), 
        axis=0)
    return res.T


def fd_max_argmax_energy_single_axis(
    X: pd.Series,
    window: str = "hann",
    skip_coefs: int = 0,
    last_coeff: int = None,
    filtering_func: Callable = None
) -> pd.Series:
    """
    Compute FFT after applying window on single series and return a series holding:
      - maximum fourier transform coefficient on magnitude
      - position of maximum fourier transform coefficient
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
    return pd.Series({"fd_max": max_coef, "fd_argmax": argmax_coef, "fd_energy": energy}, name=X.name)     


def fd_max_argmax_energy(
    X: pd.DataFrame,
    window: str = "hann",
    skip_coefs: int = 0,
    last_coeff: int = None,
    filtering_func: Callable = None
) -> pd.DataFrame:
    """
    Extract following information:

      - real value of maximum fourier transform coefficient
      - position of maximum fourier transform coefficient
      - energy of spectrum

    for each column of given DataFrame and returns a Series
    """
    res = X.apply(
        lambda col: fd_max_argmax_energy_single_axis(
            col,
            window=window,
            skip_coefs=skip_coefs,
            last_coeff=last_coeff,
            filtering_func=filtering_func
        ), 
        axis=0
    )
    return res.T


def extract_fd_features( 
    X: pd.DataFrame,
    fs: int,
    n_powerband_bins: int = 10,
    powerband_explicit_freq_names: bool = True,
    fft_window: str = "hann",
    fft_max_argmax_skip_coeffs: int = 0,
    fft_max_argmax_last_coeffs: int = None,
    fft_filtering_func: Callable = None
    ) -> pd.DataFrame:
    """
    A function that computes Frequency Domain features.
    """
    return pd.concat(
        [ 
            powerband(
                X=X.T,
                fs=fs, 
                n_powerband_bins=n_powerband_bins,
                powerband_explicit_freq_names=powerband_explicit_freq_names
            ),
            fd_max_argmax_energy(
                X=X.T,
                window=fft_window,
                skip_coefs=fft_max_argmax_skip_coeffs,
                last_coeff=fft_max_argmax_last_coeffs,
                filtering_func=fft_filtering_func
            )
        ],
        axis=1)