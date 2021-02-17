"""
Single prediction analysis
==========================

**Objective**  

In this notebook we will see how to analyze the results of a classification through the
:class:`ClassificationResults <tsanalysis.classification.ClassificationResults>` class.
We use iris dataset here. We'll do a simple classificiation and explore the results
through the given methods.
"""

## Remove warning outputs...
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#%%
# **Load data and predict**
#
# We use iris data this example with a `RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`__.
# You could use any model that gives you a ``{y_true, y_pred}`` result. The only specific
# move here is to **convert them to** ``pd.Series``.
#
# *TODO*: see also framework format example

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tsanalysis.datasets import make_iris_data
from tsanalysis.classification import ClassificationResults

X_df, y_df = make_iris_data()

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_df,y_df)
y_pred = clf.predict(X_df)

res = ClassificationResults(y_true=y_df, y_pred=pd.Series(data=y_pred, index=y_df.index))

#%%
# **Plot confusion matrix**  
res.plot_confusion_matrix(['setosa', 'versicolor', 'virginica'])

#%%
# **Metrics**  
# You can get the metrics from the classification prediction
res.metrics

#%%
# **Misclassifieds**  
# 
# You can extract the misclassified filenames by using the method :meth:`~tsanalysis.classification.ClassificationResults.misclassified`
res.misclassified
#%%
# Which means that the misclassifications happend for the given files. Now that you know their ids, you can deepen your
# analyze by plotting them, see if there is any similarities between them, and so on...

#%%
# .. seealso::
#
#     :class:`ClassificationResults <tsanalysis.classification.ClassificationResults>`,
#     :class:`CrossValidationResults <tsanalysis.classification.CrossValidationResults>`
