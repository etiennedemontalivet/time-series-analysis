"""
Cross-validation analysis
==========================

Learn how to analyze the results of a cross-validation classification.

We use iris data in this example with a `RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`__.
You could use any model that gives you a ``{y_true, y_pred}`` result. The only specific
move here is to **convert them to** ``pd.Series``.

.. seealso::

      More details in this example about :ref:`sphx_glr_auto_examples_datamodels_plot_data_format.py`
"""

# Remove warning outputs...
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#%%
# **Load data**
import pandas as pd
from tsanalysis.datasets import make_iris_data
X_df, y_df = make_iris_data()

#%%
X_df.head()

#%%
y_df.head()

#%%
# **Cross-validation**
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

#%%
cv_res = CrossValidationResults(results)

#%%
# **Analyze the results**
#
# Once you have your ``cv_res``, there are a few interesting methods you can use to analyze
# the results.

#%%
# **Plot confusion matrix:**
# You can easily plot a nice confusion matrix.
cv_res.confusion_matrix_mean

#%%
cv_res.plot_confusion_matrix_mean(['setosa', 'versicolor', 'virginica'])

#%%
# **Metrics:**
# You can get the metrics from the classification prediction.
#
# .. note::
#
#   The json manipulation is only needed in this example for a well-rendering output.
import json
print(json.dumps(cv_res.metrics, indent=2))

#%%
cv_res.plot_metrics()

#%%
# **Misclassifieds:**  
# You can extract the misclassified filenames by using the method :meth:`~tsanalysis.classification.ClassificationResults.misclassified`
cv_res.misclassified

#%%
# Which means that the misclassifications happend for the given files. Now that you know their ids, you can deepen your
# analyzis by plotting them, see if there is any similarity between them, and so on...

#%%
# .. seealso::
#
#     :class:`ClassificationResults <tsanalysis.classification.ClassificationResults>`,
#     :class:`CrossValidationResults <tsanalysis.classification.CrossValidationResults>`
