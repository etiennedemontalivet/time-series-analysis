"""
All functions below are vectorized, I.E, they can be applied on a whole DataFrame without having to apply
to each column manually.
If we want to add a new function, we need to make sure that it handles DataFrames!
"""
import pandas as pd
import numpy as np
from scipy import signal

def cepstrum_coefs_single_axis(X: pd.Series, nb_coefs: int = 24) -> pd.Series:
    """
    Compute N cepstrum coefficients where N is set to `nb_coefs`.
    """
    # Get the Hannin
    win = signal.get_window("hann", X.shape[0])
    w_sig = np.multiply(X, win)
    spectrum = np.fft.fft(w_sig)
    ceps = np.fft.ifft(np.log10(np.abs(spectrum)))
    ceps_coefs = ceps.real[:nb_coefs]

    return pd.Series(data=ceps_coefs, index=["ceps_"+str(i) for i in range(nb_coefs)], name=X.name)


def cepstrum_coefs(X: pd.DataFrame, n_cepstrum_coeff: int = 24, axis: int = 0) -> pd.DataFrame:
    """
    Compute N cepstrum coefficients on each column of given DataFrame.
    """
    res = X.apply(
        lambda col: cepstrum_coefs_single_axis(col, nb_coefs=n_cepstrum_coeff), axis=axis
    )
    return res.T

    
def extract_cepd_features( X: pd.DataFrame, n_cepstrum_coeff: int = 24 ) -> pd.DataFrame:
    """
    A function that computes Frequency Domain features.
    """
    return cepstrum_coefs(X.T, n_cepstrum_coeff=n_cepstrum_coeff)
    
