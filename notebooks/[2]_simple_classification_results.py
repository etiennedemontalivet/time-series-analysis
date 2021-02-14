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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Objective" data-toc-modified-id="Objective-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Objective</a></span><ul class="toc-item"><li><span><a href="#Context" data-toc-modified-id="Context-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Context</a></span></li><li><span><a href="#IRIS" data-toc-modified-id="IRIS-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>IRIS</a></span><ul class="toc-item"><li><span><a href="#Load-iris" data-toc-modified-id="Load-iris-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Load iris</a></span></li><li><span><a href="#Classify-and-use-ClassificationResults" data-toc-modified-id="Classify-and-use-ClassificationResults-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>Classify and use <code>ClassificationResults</code></a></span><ul class="toc-item"><li><span><a href="#Using-numpy" data-toc-modified-id="Using-numpy-1.2.2.1"><span class="toc-item-num">1.2.2.1&nbsp;&nbsp;</span>Using <code>numpy</code></a></span></li><li><span><a href="#Using-pandas" data-toc-modified-id="Using-pandas-1.2.2.2"><span class="toc-item-num">1.2.2.2&nbsp;&nbsp;</span>Using <code>pandas</code></a></span></li></ul></li></ul></li><li><span><a href="#Breast-Cancer" data-toc-modified-id="Breast-Cancer-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Breast Cancer</a></span><ul class="toc-item"><li><span><a href="#Load-data" data-toc-modified-id="Load-data-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Load data</a></span></li><li><span><a href="#Classify-and-use-ClassificationResults" data-toc-modified-id="Classify-and-use-ClassificationResults-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>Classify and use <code>ClassificationResults</code></a></span><ul class="toc-item"><li><span><a href="#Using-numpy" data-toc-modified-id="Using-numpy-1.3.2.1"><span class="toc-item-num">1.3.2.1&nbsp;&nbsp;</span>Using <code>numpy</code></a></span></li><li><span><a href="#Using-pandas" data-toc-modified-id="Using-pandas-1.3.2.2"><span class="toc-item-num">1.3.2.2&nbsp;&nbsp;</span>Using <code>pandas</code></a></span></li></ul></li></ul></li></ul></li></ul></div>
# -

# ## Objective

# In this notebook we will see how to use the results of a classification through the `ClassificationResults` class.

import numpy as np
import pandas as pd

# ### Context

# We use iris and breast_cancer datasets here. We'll do a simple classificiation and explore the results through the `ClassificationResults` class.

# ### IRIS

# #### Load iris

from tsanalysis.datasets import make_iris_data
X_df, y_df = make_iris_data()

# #### Classify and use `ClassificationResults`

from sklearn.ensemble import RandomForestClassifier
from tsanalysis.classification import ClassificationResults

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_df,y_df)

y_pred = clf.predict(X_df)

res = ClassificationResults(y_true=y_df, y_pred=pd.Series(data=y_pred, index=y_df.index))

res.plot_confusion_matrix(['setosa', 'versicolor', 'virginica'])

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
from tsanalysis.classification import ClassificationResults

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_df,y_df)

y_pred = clf.predict(X)

res = ClassificationResults(y_true=y_df, y_pred=pd.Series(data=y_pred,index=y_df.index))

res.plot_confusion_matrix(['malignant', 'benign'])

res.metrics

res.misclassified
