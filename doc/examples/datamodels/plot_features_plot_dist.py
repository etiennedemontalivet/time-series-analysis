"""
Plot a nice feature distribution
================================

Learn how to plot a feature distribution from a features dataset.
"""

from tsanalysis.datasets import make_iris_data
# sphinx_gallery_thumbnail_path = '_static/images/distribution.png'
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
fds.plot_distribution(feature_name='sepal length (cm)')
