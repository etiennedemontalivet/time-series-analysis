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

# In this notebook we will see how to use the results of a cross-validation through the `CrossValidationResults` class.

import numpy as np
import pandas as pd

# ### Context

# We use iris and breast_cancer datasets here. We'll do a simple classificiation and explore the results through the `CrossValidationResults` class.

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

# #### Cross-val using `CrossValidationResults`

from sklearn.ensemble import RandomForestClassifier
from tsanalysis.classification.results import ClassificationResults, CrossValidationResults
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,random_state=42)
results = []
for train_index, test_index in rskf.split(X_df, y_df):
    X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
    y_train, y_test = y_df[train_index], y_df[test_index]
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = pd.Series(
        data=clf.predict(X_test),
        index=y_test.index
    )
    results.append(ClassificationResults(y_true=y_test, y_pred=y_pred))

cv_res = CrossValidationResults(results)

cv_res

cv_res.confusion_matrix_mean

cv_res.plot_confusion_matrix_mean(data_iris.target_names)

cv_res.metrics

cv_res.plot_metrics()

cv_res.plot_std()

cv_res.plot_tp()

cv_res.misclassified

# ### Breast Cancer

# #### Load data

from sklearn.datasets import load_breast_cancer

X_df, y_df = load_breast_cancer(return_X_y=True, as_frame=True)

# Creating dataframe and serie with fake filenames
files = [ "file_" + str(i) for i in range(len(y_df))]
y_df.index = files
X_df.index

# #### Cross-val using `CrossValidationResults`

from sklearn.ensemble import RandomForestClassifier
from tsanalysis.classification.results import ClassificationResults, CrossValidationResults
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,random_state=42)
results = []
for train_index, test_index in rskf.split(X_df, y_df):
    X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
    y_train, y_test = y_df[train_index], y_df[test_index]
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = pd.Series(
        data=clf.predict(X_test),
        index=y_test.index
    )
    results.append(ClassificationResults(y_true=y_test, y_pred=y_pred))

cv_res = CrossValidationResults(results)

cv_res

cv_res.confusion_matrix_mean

cv_res.plot_confusion_matrix_mean(['malignant', 'benign'])

cv_res.metrics

cv_res.plot_metrics()

cv_res.plot_std()

cv_res.plot_tp()

cv_res.misclassified
