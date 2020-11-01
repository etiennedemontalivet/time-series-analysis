"""
All functions below are vectorized, I.E, they can be applied on a whole DataFrame without having to apply
to each column manually.
If we want to add a new function, we need to make sure that it handles DataFrames!
"""
import numpy as np
import pandas as pd
from scipy import signal
import pywt
from typing import List


def wavelets_acc_coeffs_single_axis(X: pd.Series) -> pd.Series:
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

    return pd.Series(results, name=X.name)


def wavelets_acc_coeffs(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fequency domain features using Wavelet trasnforms.
    This decomposes the signal into five levels using Daubechies 3 and Daubechies 2.
    It then extract features based on wavelets coefficients (squared sums and absolute sums)

    Note:
    -------
    This is based on paper:
    "Integrating Features for accelerometer-based activity recognition -- Erdas, Atasoy (2016)"
    """
    X = X.T
    res = X.apply(lambda col: wavelets_acc_coeffs_single_axis(col), axis=0)
    return res.T


def wavelets_bands_single_axis(
    X: pd.Series,
    n_wavelet_bins: int=10,
    wavelet_band_cover_ratio: float = 0.5,
    wavelet_types: List[str]=["db2", "db3"],
    wavelet_dec_level: List[int]=[5, 5],
) -> pd.Series:
    """
    A function that computes the wavelets power bands on a time serie

    Parameters
    ----------
    X : pd.Series
        The time serie to extract wavelet power bands from.
    n_wavelet_bins : int, optional
        Number of wavelets power bands to extract. The default is 10.
    wavelet_band_cover_ratio : float, optional
        The cover ration between bands. The default is 0.5.
    wavelet_types : List[str], optional
        Mother wavelet types (cf PyWavelet implementation). The default is ["db2", "db3"].
    wavelet_dec_level : List[int], optional
        Decomposition level. The default is [5, 5].

    Returns
    -------
    pd.DataFrame
        Dataframe of shape (1, n_wavelet_bins * size(wavelet_types)) 
        containing the wavelet power bands.

    """
    wav_bands = []
    
    for wave_type, level in zip( wavelet_types, wavelet_dec_level ):
        wav_band_data = []
        wavelet_res = np.concatenate( pywt.wavedec(X, wave_type, level=level) )
        n_samples_per_win = int( ( 1+wavelet_band_cover_ratio ) * len( wavelet_res ) / n_wavelet_bins )
        step_size = int( len( wavelet_res ) / n_wavelet_bins )

        for i in range( n_wavelet_bins - 1 ):
            sub_x = wavelet_res[i*step_size:(i*step_size+n_samples_per_win)]
            sub_x_windowed = signal.get_window("hann", len(sub_x)) * sub_x
            wav_band_data.append( np.sum( sub_x_windowed ** 2) / len(sub_x) )
        sub_x = wavelet_res[(n_wavelet_bins - 1)*step_size:]
        sub_x_windowed = signal.get_window("hann", len(sub_x)) * sub_x
        wav_band_data.append( np.sum( sub_x_windowed ** 2) / len(sub_x) )
        
        wav_bands.append( pd.Series(
            data=wav_band_data, 
            index=[(wave_type + "-l" + str(level)+ "-b" + str(i)) for i in range(n_wavelet_bins) ],
            name=X.name
            )
        )

    return pd.concat(wav_bands, axis=0).rename(X.name)


def wavelets_bands(
    X: pd.DataFrame,
    n_wavelet_bins: int=10,
    wavelet_band_cover_ratio: float = 0.5,
    wavelet_types: List[str]=["db2", "db3"],
    wavelet_dec_level: List[int]=[5, 5]
) -> pd.DataFrame:
    """
    A function that computes the wavelets power bands on a time series dataframe.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe of shape (n_samples, n_times).
    n_wavelet_bins : int, optional
        Number of wavelets power bands to extract. The default is 10.
    wavelet_band_cover_ratio : float, optional
        The cover ration between bands. The default is 0.5.
    wavelet_types : List[str], optional
        Mother wavelet types (cf PyWavelet implementation). The default is ["db2", "db3"].
    wavelet_dec_level : List[int], optional
        Decomposition level. The default is [5, 5].

    Returns
    -------
    pd.DataFrame
        Dataframe of shape (n_samples, n_wavelet_bins * size(wavelet_types)) 
        containing the wavelet power bands.

    """
    X = X.T
    res = X.apply( lambda col: 
        wavelets_bands_single_axis(
            col,
            n_wavelet_bins=n_wavelet_bins,
            wavelet_band_cover_ratio=wavelet_band_cover_ratio,
            wavelet_types=wavelet_types,
            wavelet_dec_level=wavelet_dec_level
        ), 
        axis=0
    )
    return res.T

    
def extract_wd_features(
    X: pd.DataFrame,
    n_wavelet_bins: int=10,
    wavelet_band_cover_ratio: float = 0.5,
    wavelet_types: List[str]=["db2", "db3"],
    wavelet_dec_level: List[int]=[5, 5]
) -> pd.DataFrame:
    """
    A function that computes Wavelets Domain features.
    """
    return pd.concat( 
        [
            wavelets_acc_coeffs(X),
            wavelets_bands(
                X=X,
                n_wavelet_bins=n_wavelet_bins,
                wavelet_band_cover_ratio=wavelet_band_cover_ratio,
                wavelet_types=wavelet_types,
                wavelet_dec_level=wavelet_dec_level
            )
        ],
        axis=1
    )