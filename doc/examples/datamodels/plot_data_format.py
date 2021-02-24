"""
Data format
==========================

Learn how to format the data to use the framework.
"""
import pandas as pd
pd.set_option('display.max_columns', 12)

#%%
# **Your dataset, your choice**
#
# For now, there is no class of raw data. Feel free to code your own class to load,
# save, and analyze your raw data. Our advice is to save your raw data in an event containing
# multiple DataFrames, one per sensor, using *ids* for indexes (such as a filename) for each
# DataFrame. An event might consist of multiple time-series windows from different sizes
# depending on sensors (sampling rate, ...).

#%% **DataFrame example**
#
# Example of a raw data DataFrame format to use :mod:`tsanalysis.features_extraction`
from tsanalysis.datasets import make_windows_ts_data
data, y = make_windows_ts_data()
data.head()

#%%
# **Features format**
#
# Speaking about the features format, this framework uses the following for the common ``X``
# and ``y``.
#
# ``X`` has to be a ``pd.DataFrame`` with:
#
# - ``columns`` have to be the features name
# - ``index`` have to be the ids that defines an event
#
# ``y`` has to be a ``pd.Index`` with:
#
# - ``index`` have to be the ids that defines an event (same as ``X.index``)
#
# For example, we updated the iris data so we conform with this format when loading it:

from tsanalysis.datasets import make_iris_data
# sphinx_gallery_thumbnail_path = '_static/images/data_format.png'
X, y = make_iris_data()

#%%
X.head()

#%%
y.head()

#%%
# **Create fake features data**
#
# We created a fake ``make_classification`` method that you can use to create fake features.
from tsanalysis.datasets import make_classification
X, y = make_classification(n_features=10)

#%%
X.head()

#%%
y.head()
#%%
# .. seealso::
#
#     :func:`make_classification <tsanalysis.datasets.make_classification>`,
#     :func:`make_iris_data <tsanalysis.datasets.make_iris_data>`
