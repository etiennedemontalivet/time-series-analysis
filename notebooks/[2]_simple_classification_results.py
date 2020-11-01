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

# In this notebook we will see how to use the results of a classification through the `ClassificationResults` class.

import numpy as np
import pandas as pd

# ### Context

# We use iris and breast_cancer datasets here. We'll do a simple classificiation and explore the results through the `ClassificationResults` class.

# ### IRIS

# #### Load iris

from sklearn.datasets import load_iris

data_iris = load_iris()

X, y = data_iris.data, data_iris.target

# Creating dataframe and serie with fake filenames
files = [ "file_" + str(i) for i in range(len(y))]
y_df = pd.Series(
    data=y,
    index=files
)
X_df = pd.DataFrame(
    data=X,
    index=files)

# #### Classify and use `ClassificationResults`

from sklearn.ensemble import RandomForestClassifier
from tsanalysis.classification.results import ClassificationResults

# ##### Using `numpy`

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X,y)

y_pred = clf.predict(X)

res = ClassificationResults(y_true=pd.Series(y), y_pred=pd.Series(y_pred))

res.plot_confusion_matrix(data_iris.target_names)

res.metrics

res.misclassified

# ##### Using `pandas`

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_df,y_df)

y_pred = clf.predict(X)

res = ClassificationResults(y_true=y_df, y_pred=pd.Series(data=y_pred, index=y_df.index))

res.plot_confusion_matrix(data_iris.target_names)

res.metrics

res.misclassified

# ### Breast Cancer

# #### Load data

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

X_df, y_df = load_breast_cancer(return_X_y=True, as_frame=True)

# Creating dataframe and serie with fake filenames
files = [ "file_" + str(i) for i in range(len(y))]
y_df.index = files
X_df.index

# #### Classify and use `ClassificationResults`

from sklearn.ensemble import RandomForestClassifier
from tsanalysis.classification.results import ClassificationResults

# ##### Using `numpy`

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X,y)

y_pred = clf.predict(X)

res = ClassificationResults(y_true=pd.Series(y), y_pred=pd.Series(y_pred))

res.plot_confusion_matrix(['malignant', 'benign'])

res.metrics

res.misclassified

# ##### Using `pandas`

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_df,y_df)

y_pred = clf.predict(X)

res = ClassificationResults(y_true=y_df, y_pred=pd.Series(data=y_pred,index=y_df.index))

res.plot_confusion_matrix(['malignant', 'benign'])

res.metrics

res.misclassified
