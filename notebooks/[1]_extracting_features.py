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

# In this notebook we will see how to extract feature signals from files.

# ### Context

# Let say you have 3 sensors:
# - sensor 1: sample rate 10kHz
# - sensor 2: sample rate 1kHz
# - sensor 3: sample rate 1kHz
#
# Your file contained 100ms record. You thus have for each event 1000 values of sensor 1 and 100 values of sensor 2 and 3.
#
# To use the framework to extract features, we use pandas multiindex. Let see how to do it.

N_S1 = 1000  # number of values per event for sensor 1
N_S23 = 100 # number of values per event for sensor 2 and 3

# ### Load your files

# You choose how to load your files, here we'll use an example and generate fake data.

import pandas as pd
import numpy as np

# First we generate fake names

fake_name_length = 10
a, z = np.array(["a","z"]).view("int32") 
filenames = np.random.randint(low=a,high=z,size=150*fake_name_length,dtype="int32").view(f"U{fake_name_length}")

# We generate the samples number per event per sensor

index_s1 = list(range(N_S1))   # we have 100 values of sensor 2 & 3 for each event 
index_s23 = list(range(N_S23)) # we have 1000 values of sensor 1 for each event

# +
data_s1 = []          # will be the fake data of sensor 1
data_s2 = []
data_s3 = []
multi_index_s1_0 = [] # will be the dimension 0 of multi-index of sensor 1
multi_index_s1_1 = [] # will be the dimension 1 of multi-index of sensor 1
multi_index_s23_0 = []
multi_index_s23_1 = []

for file in filenames:
    data_s1.append(np.random.rand(N_S1)) # Put your data from sensor 1 here
    data_s2.append(np.random.rand(N_S23))
    data_s3.append(np.random.rand(N_S23))
    
    multi_index_s1_0 +=  [ file ] * N_S1
    multi_index_s1_1 += index_s1
    multi_index_s23_0 +=  [ file ] * N_S23
    multi_index_s23_1 += index_s23
# -

data_s1 = np.array( data_s1 ).flatten()
data_s2 = np.array( data_s2 ).flatten()
data_s3 = np.array( data_s3 ).flatten()

# Generate the multiindexes used for specific format

# +
multi_index_s1 = pd.MultiIndex.from_arrays([
    np.array(multi_index_s1_0, dtype="U64"),
    np.array(multi_index_s1_1, dtype="int32")
], names=["filename", "index"])

multi_index_s23 = pd.MultiIndex.from_arrays([
    np.array(multi_index_s23_0, dtype="U64"),
    np.array(multi_index_s23_1, dtype="int32")
], names=["filename", "index"])
# -

# We can now create the multiindex dataframe that we will use to extarct features. Note that it allows to track filenames.

df_s1 = pd.DataFrame(
    data=np.stack([ data_s1 ], axis=1),
    index=multi_index_s1,
    columns = [
        "sensor1"
    ],
    dtype="float32",
)
df_s23 = pd.DataFrame(
    data=np.stack([data_s2, data_s3], axis=1),
    index=multi_index_s23,
    columns = [
        "sensor2",
        "sensor3"
    ],
    dtype="float32",
)

# ### Features extraction

from framework.datamodels.features import extract_features

features_s1 = extract_features( df=df_s1 )
features_s23 = extract_features( df=df_s23 )

# ### Features dataset

# Now if we want to analyse these features, we could used the `FeaturesDataset` class to play with it. Let generate some fake labels for our features.

from framework.datamodels.features import FeaturesDataset

y = pd.Series(data=np.concatenate([np.zeros(50), np.ones(features_s1.shape[0]-50)]), index=filenames, name="labels")

features_ds_s1  = FeaturesDataset(X=features_s1, y=y, name="sensor1")
features_ds_s23 = FeaturesDataset(X=features_s23, y=y, name="sensor23")

# ### Concatenate our features dataset 

features_ds_s123 = FeaturesDataset(
    X = pd.concat( [features_ds_s1.X, features_ds_s23.X ], axis=1),
    y = y,
    name="all_features"
)

# ### Save and load features dataset for saving time...

features_ds_s123.dump("./")

new_feat = FeaturesDataset.load("all_features", "./")

# ### Plot features distribution

# %matplotlib notebook

# Now that we have computed a lot of features, it is time to plot them and start the analysis. Of course they are plenty of librairies that plot distribution, here are just some of them.

# #### Seaborn simple distribution plot

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

iCol = 1

sns.distplot(features_ds_s123.X[features_ds_s123.y == 0][ features_ds_s123.X.columns[iCol] ])
sns.distplot(features_ds_s123.X[features_ds_s123.y == 1][ features_ds_s123.X.columns[iCol] ])

# #### Pimp JoyPlot [WIP]

import joypy
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm

sub_features = [ x for x in features_ds_s123.X.columns if 'f' in x]

X = features_ds_s123.X_scaled

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

y_concat = pd.concat([ pd.Series(data=features_ds_s123.y, name='label') for _ in X.columns ])
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


