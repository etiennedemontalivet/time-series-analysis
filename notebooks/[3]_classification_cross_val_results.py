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

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Objective" data-toc-modified-id="Objective-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Objective</a></span><ul class="toc-item"><li><span><a href="#Context" data-toc-modified-id="Context-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Context</a></span></li><li><span><a href="#IRIS" data-toc-modified-id="IRIS-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>IRIS</a></span><ul class="toc-item"><li><span><a href="#Load-iris" data-toc-modified-id="Load-iris-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Load iris</a></span></li><li><span><a href="#Cross-val-using-CrossValidationResults" data-toc-modified-id="Cross-val-using-CrossValidationResults-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>Cross-val using <code>CrossValidationResults</code></a></span></li></ul></li><li><span><a href="#Breast-Cancer" data-toc-modified-id="Breast-Cancer-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Breast Cancer</a></span><ul class="toc-item"><li><span><a href="#Load-data" data-toc-modified-id="Load-data-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Load data</a></span></li><li><span><a href="#Cross-val-using-CrossValidationResults" data-toc-modified-id="Cross-val-using-CrossValidationResults-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>Cross-val using <code>CrossValidationResults</code></a></span></li></ul></li></ul></li></ul></div>
# -

# ## Objective

# In this notebook we will see how to use the results of a cross-validation through the `CrossValidationResults` class.

import numpy as np
import pandas as pd

pd.options.plotting.backend = 'plotly' # optional

# ### Context

# We use iris and breast_cancer datasets here. We'll do a simple classificiation and explore the results through the `CrossValidationResults` class.

# ### IRIS

# #### Load iris

from tsanalysis.datasets import make_iris_data
X_df, y_df = make_iris_data()

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

cv_res.confusion_matrix_mean

cv_res.plot_confusion_matrix_mean(['setosa', 'versicolor', 'virginica'])

cv_res.metrics

cv_res.plot_metrics()

cv_res.misclassified

# ### Breast Cancer

# #### Load data

from sklearn.datasets import load_breast_cancer

X_df, y_df = load_breast_cancer(return_X_y=True, as_frame=True)

# Creating dataframe and serie with fake filenames
files = [ "file_" + str(i) for i in range(len(y_df))]
y_df.index = files
X_df.index = files

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

cv_res.misclassified


