"""
All functions below are vectorized, I.E, they can be applied on a whole DataFrame without having to apply
to each column manually.
If we want to add a new function, we need to make sure that it handles DataFrames!
"""
import numpy as np
import pandas as pd
from scipy import signal


def powerband_single_axis(X: pd.Series, nperseg: int = 94, fs: int = 1000) -> pd.Series:
    """
    Compute power band coefficients for a single Series
    """
    _, power_band = signal.welch(X, window="hann", nperseg=nperseg, fs=fs)
    return pd.Series(data=power_band, index=["powerband_"+str(i) for i in range(len(_))], name=X.name)


def powerband(X: pd.DataFrame, axis: int = 0, fs=1000) -> pd.DataFrame:
    """
    Compute powerband coefficient for each column of a DataFrame and returns a Series
    """
    res = X.apply(lambda col: powerband_single_axis(col, fs=fs), axis=axis)
    return res.T


def fd_max_argmax_energy_single_axis(
    X: pd.Series, window: str = "hann", skip_coefs: int = 10, last_coeff: int = 1000
) -> pd.Series:
    """
    Compute FFT after applying window on single series and return a series holding:
      - maximum fourier transform coefficient on magnitude
      - position of maximum fourier transform coefficient
    """
    win = signal.get_window(window, X.shape[0])
    w_sig = np.multiply(X, win)
    spectrum = np.fft.rfft(w_sig)
    mag = np.abs(spectrum)

    # Low pass filter on mag
    b, a = signal.butter(3, 0.1)
    mag_filtered = signal.filtfilt(b, a, mag)

    argmax_coef = np.argmax(mag_filtered[skip_coefs:last_coeff]) + skip_coefs
    max_coef = mag_filtered[argmax_coef]
    energy = np.sum(mag ** 2) / X.shape[0]
    return pd.Series({"fd_max": max_coef, "fd_argmax": argmax_coef, "fd_energy": energy}, name=X.name)     


def fd_max_argmax_energy(
    X: pd.DataFrame, window: str = "hann", axis: int = 0
) -> pd.Series:
    """
    Extract following information:

      - real value of maximum fourier transform coefficient
      - position of maximum fourier transform coefficient
      - energy of spectrum

    for each column of given DataFrame and returns a Series
    """
    res = X.apply(
        lambda col: fd_max_argmax_energy_single_axis(col, window=window), axis=axis
    )
    return res.T


def extract_fd_features( X: pd.DataFrame ) -> pd.DataFrame:
    """
    A function that computes Frequency Domain features.
    """
    # List all features extraction function
    funcs = [powerband, fd_max_argmax_energy]

    out = []
    for func in funcs:
        out.append(func(X.T))
    return pd.concat(out, axis=1)