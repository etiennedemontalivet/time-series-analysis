"""
This module defines the complete features extraction.
"""
import pandas as pd
import numpy as np
from typing import Callable
from typing import List

from framework.features_extraction.cepstrum_domain import extract_cepd_features
from framework.features_extraction.time_domain import extract_td_features
from framework.features_extraction.wavelets_domain import extract_wd_features
from framework.features_extraction.frequency_domain import extract_fd_features


def extract_all_features( X: pd.DataFrame,
	fs: int,
	n_cepstrum_coeff: int = 24,
	n_powerband_bins: int = 10,
	powerband_explicit_freq_names: bool = True,
	fft_window: str = "hann",
    fft_max_argmax_skip_coeffs: int = 0,
    fft_max_argmax_last_coeffs: int = None,
    fft_filtering_func: Callable = None,
    n_wavelet_bins: int=10,
    wavelet_band_cover_ratio: float = 0.5,
    wavelet_types: List[str]=["db2", "db3"],
    wavelet_dec_level: List[int]=[5, 5]
	) -> pd.DataFrame:
    """
    A function that computes all features.
    """
    return pd.concat(
    	[ 
    		extract_td_features(X), 
    		extract_fd_features(
      			X=X,
      			fs=fs, 
      			n_powerband_bins=n_powerband_bins,
      			powerband_explicit_freq_names=powerband_explicit_freq_names,
				fft_window=fft_window,
                fft_max_argmax_skip_coeffs=fft_max_argmax_skip_coeffs,
                fft_max_argmax_last_coeffs=fft_max_argmax_last_coeffs,
                fft_filtering_func=fft_filtering_func
      		), 
      		extract_cepd_features(
      			X=X, 
      			n_cepstrum_coeff=n_cepstrum_coeff
      		),
      		extract_wd_features(
      			X=X,
      			n_wavelet_bins=n_wavelet_bins,
                wavelet_band_cover_ratio=wavelet_band_cover_ratio,
                wavelet_types=wavelet_types,
                wavelet_dec_level=wavelet_dec_level
            )
		],
		axis=1 
    )
