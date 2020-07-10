"""
This module defines the complete features extraction.
"""
import pandas as pd
import numpy as np

from framework.features_extraction.cepstrum_domain import extract_cepd_features
from framework.features_extraction.time_domain import extract_td_features
from framework.features_extraction.wavelets_domain import extract_wd_features
from framework.features_extraction.frequency_domain import extract_fd_features


def extract_all_features( X: pd.DataFrame,
	fs: int,
	n_cepstrum_coeff: int = 24,
	n_powerband_bins: int = 10,
	powerband_explicit_freq_names: bool = True 
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
      			powerband_explicit_freq_names=powerband_explicit_freq_names
      		), 
      		extract_cepd_features(
      			X=X, 
      			n_cepstrum_coeff=n_cepstrum_coeff
      		),
      		extract_wd_features(X) 
		],
		axis=1 
    )
