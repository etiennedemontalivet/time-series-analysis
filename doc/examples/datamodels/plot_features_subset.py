"""
Get a subset of features dataset
====================================

Learn how to extract sub-features datasets
"""

import pandas as pd
pd.set_option('display.max_rows', 8)
from tsanalysis.datasets import make_classification
from tsanalysis.datamodels import FeaturesDataset
# sphinx_gallery_thumbnail_path = '_static/images/subset.png'

#%%
X, y = make_classification(n_classes=5, n_samples=100, n_features=10, n_informative=5)
fds = FeaturesDataset(
    X=X,
    y=y,
    name='demo',
    target_labels={
       0:'label1', 1:'label2', 2:'label3', 3:'label4', 4:'label5'
    })
fds.X

#%%
fds.y

#%%
# Extract from a specific label(s)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
fds_03 = fds.get_subset_from_labels([0,3])
fds_03.X

#%%
fds_03.y

#%%
fds_4 = fds.get_subset_from_labels([4])
fds_4.X

#%%
fds_4.y

#%%
# Extract from ids (filenames)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
fds_files = fds.get_subset_from_indexes(['file_' + str(i) for i in range(20,50)])
fds_files.X

#%%
fds_files.y

#%%
# Extract from features
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
fds_feat = fds.get_subset_from_features(['feat_2', 'feat_7', 'feat_9'])
fds_feat.X

#%%
fds_feat.y