# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Objective

# In this notebook we will see how to extract feature signals from fake data using our features extraction function.

# ### Context

# We use fake data here. Imagine we have 2 sensors, one sampled at 1kHz and the other at 500Hz. We work on a 1 second window, thus we have labeled events with 1000 points for sensor 1 and 500 points for sensor 2. Let say we have 100 events, half of label 0 and half of label 1.

# ### Load your files

import pandas as pd
import numpy as np

Fs_s1 = 1 #kHz
Fs_s2 = 0.5 # kHz

filenames = [ "file" + str(i) for i in range(100)]

sensor1_df = pd.DataFrame( data=np.random.random((100,1000)), index=filenames)

sensor2_df = pd.DataFrame( data=np.random.random((100,500)), index=filenames)

# ### Features extraction

from framework.features_extraction import extract_all_features 

features_s1 = extract_all_features( sensor1_df )
features_s2 = extract_all_features( sensor2_df )

# Keep a track of wich feature belongs to which sensor
all_features = pd.concat( [ features_s1.add_prefix("s1_"), features_s2.add_prefix("s2_") ], axis=1 )

all_features

# ### Features dataset

# Now if we want to analyse these features, we could used the `FeaturesDataset` class to play with it. Let generate some fake labels for our features.

from framework.datamodels.features import FeaturesDataset

y = pd.Series(data=np.concatenate([np.zeros(50), np.ones(all_features.shape[0]-50)]), index=filenames, name="labels")

features_ds  = FeaturesDataset(X=all_features, y=y, name="features_ds")

# ### Save and load features dataset for saving time...

features_ds.dump("./")

new_feat = FeaturesDataset.load("features_ds", "./")

# ### Plot features distribution

# %matplotlib notebook

# Now that we have computed a lot of features, it is time to plot them and start the analysis. Of course they are plenty of librairies that plot distribution, here are just some of them.

# #### Seaborn simple distribution plot

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# We can choose the column to plot to compare the distributions of each label
iCol = 1

sns.distplot(features_ds.X[features_ds.y == 0][ features_ds.X.columns[iCol] ])
sns.distplot(features_ds.X[features_ds.y == 1][ features_ds.X.columns[iCol] ])

# #### Pimp JoyPlot [WIP]

import joypy
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm

X = features_ds.X_scaled

filenames_serie = pd.Series(data=filenames, name="filenames")

filenames_concat = pd.concat([ filenames_serie for _ in X.columns ], axis=0)
filenames_concat = filenames_concat.reset_index()
filenames_concat = filenames_concat['filenames']

values_to_concat_0 = []
values_to_concat_1 = []
for col in X.columns:
    for i in range(len(X[col])):
        if y[X.index[i]] == 0:
            values_to_concat_0.append(X[col].iloc[i])
            values_to_concat_1.append(np.nan)
        else:
            values_to_concat_1.append(X[col].iloc[i])
            values_to_concat_0.append(np.nan)

X_values_concat_0 = pd.Series(data=values_to_concat_0, name='label 0')
X_values_concat_1 = pd.Series(data=values_to_concat_1, name='label 1')

filenames_concat = filenames_concat.reset_index()
filenames_concat = filenames_concat['filenames']

y_concat = pd.concat([ pd.Series(data=features_ds.y, name='label') for _ in X.columns ])
y_concat = y_concat.reset_index()
y_concat = y_concat['label']

X_feat_concat = pd.concat([ pd.Series(data=np.repeat(col, X.shape[0]), name="features") for col in X.columns ])
X_feat_concat = X_feat_concat.reset_index()
X_feat_concat = X_feat_concat['features']

X_full = pd.concat([ X_feat_concat, X_values_concat_0, X_values_concat_1], axis=1)

fig, axes = joypy.joyplot(X_full,
                          by="features",
                          range_style='own',
                          legend=True,
                          grid="y", linewidth=1, figsize=(6,25),
                          title="Features distribution")


