"""
Single prediction analysis
==========================

Learn how to analyze the results of a single classification.
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
# .. seealso::
#
#       More details in this example about :ref:`sphx_glr_auto_examples_datamodels_plot_data_format.py`

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tsanalysis.datasets import make_iris_data
from tsanalysis.classification import ClassificationResults

X_df, y_df = make_iris_data()

#%%
X_df.head()

#%%
y_df.head()

#%%
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_df,y_df)
y_pred = clf.predict(X_df)

res = ClassificationResults(y_true=y_df, y_pred=pd.Series(data=y_pred, index=y_df.index))

#%%
# **Analyze the results**  
#
# Once you have your ``res``, there are a few interesting methods you can use to analyze
# the results.

#%%
# **Plot confusion matrix:**
# You can easily plot a nice confusion matrix.
res.plot_confusion_matrix(['setosa', 'versicolor', 'virginica'])

#%%
# **Metrics:**  
# You can get the metrics from the classification prediction.
#
# .. note::
#
#   The json manipulation is only needed in this example for a well-rendering output.
import json
print(json.dumps(res.metrics, indent=2))

#%%
# **Misclassifieds:**  
# You can extract the misclassified filenames by using the method :meth:`~tsanalysis.classification.ClassificationResults.misclassified`
res.misclassified
#%%
# Which means that the misclassifications happend for the given files. Now that you know their ids, you can deepen your
# analyzis by plotting them, see if there is any similarity between them, and so on...

#%%
# .. seealso::
#
#     :class:`ClassificationResults <tsanalysis.classification.ClassificationResults>`,
#     :class:`CrossValidationResults <tsanalysis.classification.CrossValidationResults>`
