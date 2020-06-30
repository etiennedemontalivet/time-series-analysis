"""
All functions below are vectorized, I.E, they can be applied on a whole DataFrame without having to apply
to each column manually.
If we want to add a new function, we need to make sure that it handles DataFrames!
"""
import pandas as pd
import numpy as np
from scipy import signal
from framework.features_extraction.base import FeatureExtractor


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

    return pd.Series(ceps_coefs).add_prefix("cepstrum_").add_prefix(str(X.name) + "_")


def cepstrum_coefs(X: pd.DataFrame, nb_coefs: int = 24, axis: int = 0) -> pd.Series:
    """
    Compute N cepstrum coefficients on each column of given DataFrame.
    """
    res = X.apply(
        lambda col: cepstrum_coefs_single_axis(col, nb_coefs=nb_coefs), axis=axis
    )
    # FIXME: This is really ugly. I didn't find a better way...
    return res.unstack().dropna().droplevel(0)


class CepstrumDomainFeatureExtractor(FeatureExtractor):
    funcs = [
        cepstrum_coefs,
    ]
