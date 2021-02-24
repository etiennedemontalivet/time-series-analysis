"""
Load and save features dataset
==============================

Learn how to save and load a features dataset.
"""

from tsanalysis.datasets import make_iris_data
# sphinx_gallery_thumbnail_path = '_static/images/parquet-file.png'
X_df, y_df = make_iris_data()

#%%
# **Create a features dataset**
from tsanalysis.datamodels import FeaturesDataset
fds = FeaturesDataset(
    X=X_df,
    y=y_df,
    name='iris_demo',
    target_labels={
       0:'setosa',
       1:'versicolor',
       2:'virginica'
    })

#%% **Save your features dataset**
fds.dump()

#%%
# **Load a features dataset**
fds_bis = FeaturesDataset.load('iris_demo')

#%%
# .. seealso::
#
#     :func:`make_classification <tsanalysis.datasets.make_classification>`,
#     :func:`make_iris_data <tsanalysis.datasets.make_iris_data>`
