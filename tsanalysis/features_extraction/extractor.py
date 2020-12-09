"""
This module defines the complete features extraction.
"""
import pandas as pd
from typing import Callable
from typing import List

from cepstrum_domain import extract_cepd_features
from time_domain import extract_td_features
from wavelets_domain import extract_wd_features
from frequency_domain import extract_fd_features


def extract_all_features(
    X: pd.DataFrame,
    fs: int,
    n_cepstrum_coeff: int = 24,
    n_powerband_bins: int = 10,
    powerband_explicit_freq_names: bool = True,
    fft_window: str = "hann",
    fft_max_argmax_skip_coeffs: int = 1,
    fft_max_argmax_last_coeffs: int = None,
    fft_filtering_func: Callable = None,
    n_wavelet_bins: int = 10,
    wavelet_band_cover_ratio: float = 0.5,
    wavelet_types: List[str] = ["db2", "db3"], 
    wavelet_dec_level: List[int] = [5, 5],
    sampen_m: int = 2,
    sampen_eta: float = 0.2
) -> pd.DataFrame:
    # pylint: disable=dangerous-default-value
    """
    A function that computes all features.

    Parameters
    ----------
    X : pd.DataFrame
        Input containing the time series. Shape has to be (n_signals, time)
    fs : int
        Sample rate in Hz.
    n_cepstrum_coeff : int, optional
        Number of cesptrum coefficients to extract. The default is 24.
    n_powerband_bins : int, optional
        Number of powerbands to compute. The default is 10.
    powerband_explicit_freq_names : bool, optional
        If True, the frequency bands are included in the feature name, else
        a counter is used. The default is True.
    fft_window : str, optional
        The type of window to use for windowing. The default is "hann".
    fft_max_argmax_skip_coeffs : int, optional
        Number of coefficients to skip for the magnitude max/argmax computation. The default is 1.
    fft_max_argmax_last_coeffs : int, optional
        The last fft coeff to take into account for the the max/argmax magnitude computation.
        If None, no part of the magnitude is removed from the end.
        The default is None.
    fft_filtering_func : Callable, optional
        A filter on the magnitude could be applied before max/argmax computation.
        The default is None.
    n_wavelet_bins : int, optional
        Number of wavelets power bands to extract. The default is 10.
    wavelet_band_cover_ratio : float, optional
        The cover ration between bands. The default is 0.5.
    wavelet_types : List[str], optional
        Mother wavelet types (cf PyWavelet implementation). The default is ["db2", "db3"].
    wavelet_dec_level : List[int], optional
        Decomposition level. The default is [5, 5].
    sampen_m : int, optional
        Length of subvectors. The default is 2.
    sampen_eta : float, optional
        Ratio to be multiplied by the std of x to get the tolerance. The default is 0.2.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all the extracted features per time serie.

    """
    return pd.concat(
        [
            extract_td_features(X=X, sampen_m=sampen_m, sampen_eta=sampen_eta),
            extract_fd_features(
                X=X,
                fs=fs,
                n_powerband_bins=n_powerband_bins,
                powerband_explicit_freq_names=powerband_explicit_freq_names,
                fft_window=fft_window,
                fft_max_argmax_skip_coeffs=fft_max_argmax_skip_coeffs,
                fft_max_argmax_last_coeffs=fft_max_argmax_last_coeffs,
                fft_filtering_func=fft_filtering_func,
            ),
            extract_cepd_features(X=X, n_cepstrum_coeff=n_cepstrum_coeff),
            extract_wd_features(
                X=X,
                n_wavelet_bins=n_wavelet_bins,
                wavelet_band_cover_ratio=wavelet_band_cover_ratio,
                wavelet_types=wavelet_types,
                wavelet_dec_level=wavelet_dec_level,
            ),
        ],
        axis=1,
    )
