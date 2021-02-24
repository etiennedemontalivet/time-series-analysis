"""
Extract all features from one sensor
====================================

Learn how to extract all features from time series windows and analyze the
new features dataset.
"""

#%%
# **Load data**
#
# We use fake data generated from :meth:`~tsanalysis.datasets.make_windows_ts_data`.
#

# sphinx_gallery_thumbnail_path = '_static/images/distribution.png'
import pandas as pd
pd.set_option('display.max_columns', 12)

#%%
from tsanalysis.datasets import make_windows_ts_data
data, y = make_windows_ts_data()

#%%
data.head()

#%%
# **Extract all features**  
from tsanalysis.features_extraction import extract_all_features
features = extract_all_features(
    X=data,
    fs=1000,
    n_cepstrum_coeff=12,
    n_powerband_bins=10,
    n_wavelet_bins=10,
    wavelet_types=['db3'],
    wavelet_dec_level=[5],
)
features.head()

#%%
# **Analyze them**: create a FeaturesDataset to easily analyze features distributions, save them, ...
from tsanalysis.datamodels import FeaturesDataset
fds = FeaturesDataset(
    X=features,
    y=y,
    name='demo',
    target_labels= {
        0: "cat",
        1: "dog",
        2: "other"
    }
)
#%%
fds.plot_distribution('std',n_bins=10)
