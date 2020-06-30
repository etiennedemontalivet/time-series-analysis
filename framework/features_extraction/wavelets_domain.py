"""
All functions below are vectorized, I.E, they can be applied on a whole DataFrame without having to apply
to each column manually.
If we want to add a new function, we need to make sure that it handles DataFrames!
"""
import numpy as np
import pandas as pd
import pywt
from framework.features_extraction.base import FeatureExtractor


def wavelets_features_single_axis(X: pd.Series) -> pd.Series:
    """
    Compute fequency domain features using Wavelet trasnforms.
    This decomposes the signal into five levels using Daubechies 3 and Daubechies 2.
    It then extract features based on wavelets coefficients (squared sums and absolute sums)

    Note:
    -------
    This is based on paper:
    "Integrating Features for accelerometer-based activity recognition -- Erdas, Atasoy (2016)"
    """
    # We are only interested in detailed coefficients 5 and 4 with Daubechies 3
    _, cD5_db3, cD4_db3, _, _, _ = pywt.wavedec(X, "db3", level=5)
    # We compute the sum of squared detailed coefficients
    cD5_db3_sq = np.sum(cD5_db3 ** 2)
    cD4_db3_sq = np.sum(cD4_db3 ** 2)

    # We are interested in coefficients 5, 4, 3, 2, 1 with Daubechies 2
    _, cD5_db2, cD4_db2, cD3_db2, cD2_db2, cD1_db2 = pywt.wavedec(X, "db2", level=5)
    # We compute the sum of squared detailed coefficients
    cD5_db2_sq = np.sum(cD5_db2 ** 2)
    cD4_db2_sq = np.sum(cD4_db2 ** 2)
    cD3_db2_sq = np.sum(cD3_db2 ** 2)
    cD2_db2_sq = np.sum(cD2_db2 ** 2)
    cD1_db2_sq = np.sum(cD1_db2 ** 2)
    # We compute the absolute sum of coefficients
    CD5_as = np.sum(np.abs(cD5_db2))
    CD4_as = np.sum(np.abs(cD4_db2))
    CD3_as = np.sum(np.abs(cD3_db2))
    CD2_as = np.sum(np.abs(cD2_db2))
    CD1_as = np.sum(np.abs(cD1_db2))

    results = {
        # Daubechies 3 detailed coefficient squared sum
        "cD5_db3_sq": cD5_db3_sq,
        "cD4_db3_sq": cD4_db3_sq,
        # Daubechies 2 detailed coefficients squared sum
        "cD5_db2_sq": cD5_db2_sq,
        "cD4_db2_sq": cD4_db2_sq,
        "cD3_db2_sq": cD3_db2_sq,
        "cD2_db2_sq": cD2_db2_sq,
        "cD1_db2_sq": cD1_db2_sq,
        # Daubechies 2 detailed coefficients absolute sum
        "CD5_db2_as": CD5_as,
        "CD4_db2_as": CD4_as,
        "CD3_db2_as": CD3_as,
        "CD2_db2_as": CD2_as,
        "CD1_db2_as": CD1_as,
    }

    return pd.Series(results).add_prefix(str(X.name) + "_")


def wavelets_features(X: pd.DataFrame, axis: int = 0) -> pd.Series:
    """
    Compute fequency domain features using Wavelet trasnforms.
    This decomposes the signal into five levels using Daubechies 3 and Daubechies 2.
    It then extract features based on wavelets coefficients (squared sums and absolute sums)

    Note:
    -------
    This is based on paper:
    "Integrating Features for accelerometer-based activity recognition -- Erdas, Atasoy (2016)"
    """
    res = X.apply(lambda col: wavelets_features_single_axis(col), axis=axis)
    # FIXME: This is really ugly. I didn't find a better way...
    return res.unstack().dropna().droplevel(0)


class WaveletsDomainFeatureExtractor(FeatureExtractor):
    funcs = [
        wavelets_features,
    ]
