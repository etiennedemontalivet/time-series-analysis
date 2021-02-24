"""
Concatenate features dataset
====================================

Learn how to concatenate features datasets
"""

import pandas as pd
from tsanalysis.datasets import make_classification
from tsanalysis.datamodels import FeaturesDataset
# sphinx_gallery_thumbnail_path = '_static/images/concat.png'

#%%
# Same number of features, new samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
X1, y1 = make_classification(n_features=6, n_classes=2, n_samples=50, n_clusters_per_class=1)
fds1 = FeaturesDataset(
    X=X1.rename('f1_{}'.format),
    y=y1.add_prefix("f1_"),
    name='demo1',
    target_labels={
       0:'label1',
       1:'label2',
    })
fds1.X.head()

#%%
X2, y2 = make_classification(n_features=6, n_classes=3, n_samples=100, n_clusters_per_class=1)
fds2 = FeaturesDataset(
    X=X2.rename('f2_{}'.format),
    y=y2.add_prefix("f2_"),
    name='demo2',
    target_labels={
       0:'label1',
       1:'label2',
       2:'label3'
    })
fds2.X.head()

#%%
# **Concatenate 2 features dataset**
#
# Datasets **must have the same features** (same ``fds.X.columns``) **and
# different ids** (``fds.y.index``). If features are different, the merge won't work. If some ids are
# similar, it will automatically remove the duplicates.
from tsanalysis.datamodels import features_concat
fds = features_concat([ fds1, fds2 ])
fds.X

#%%
print("Labels: " + str(fds.target_labels_))

#%%
# Same number of samples, new features
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
X1, y = make_classification(n_features=4, n_classes=2, n_samples=50, n_clusters_per_class=1)
X1.head()

#%%
X2, _ = make_classification(n_features=4, n_classes=2, n_samples=50, n_clusters_per_class=1)
X2 = pd.DataFrame(data=X2.values, index=X2.index, columns=['feat_' + str(i) for i in range(4,8)])
X2.head()

#%%
fds = FeaturesDataset(
    X=pd.concat([X1, X2], axis=1),
    y=y,
    name='demo',
    target_labels={
       0:'label1',
       1:'label2',
    })
fds.X.head()