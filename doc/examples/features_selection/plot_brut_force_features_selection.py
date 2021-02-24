"""
Brut Force features selection
=============================

Learn how to do features selection using a brut force approach.
"""

#%%
# **Create fake data**
import numpy as np
import pandas as pd
from tsanalysis.datasets import make_classification
X_df, y_df = make_classification(n_samples=1000, n_features=100, n_informative=10, random_state=42)

#%%
# **Define an objective to minimize**
#
# The objectif could be whatever you want. Only constraints are:
#
# - it must have a ``features_list`` parameter
# - it must return a ``dict`` containing at least a ``score`` key with the score
#   to optimize.
#
# In this example, we use a the mean of matthew correlation coefficient score of
# a cross-validation
from sklearn.naive_bayes import GaussianNB
from tsanalysis.classification.results import ClassificationResults, CrossValidationResults
from sklearn.model_selection import RepeatedStratifiedKFold

def my_objectif(features_list=list):
    """
    My objectif to minimize
    Here we take a simple cross-validation
    """   
    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1,random_state=42)
    results = []
    for train_index, test_index in rskf.split(X_df, y_df):
        X_train, X_test = X_df[features_list].iloc[train_index], \
            X_df[features_list].iloc[test_index]
        y_train, y_test = y_df[train_index], y_df[test_index]
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = pd.Series(
            data=clf.predict(X_test),
            index=y_test.index
        )
        results.append(ClassificationResults(y_true=y_test, y_pred=y_pred))

    cv_res = CrossValidationResults(results)

    return {
        'score': 1 - cv_res.mean['mean_matthews_corrcoef'],
        'accuracy': cv_res.mean['mean_accuracy']
    }

#%%
# **Brut force features selection**
from tsanalysis.features_selection import brut_force_features_selection
df = brut_force_features_selection(
    features_list=X_df.columns,
    objectif=my_objectif,
    max_features=10,
    n_per_epoch=2,
)
df.head()

#%%
best_features = df['feature_name'].tolist()
print(best_features)

#%%
# **Compare results**
print("Without brut force selected features:\n" + str(my_objectif(X_df.columns)))
print("With brut force selected features:\n" + str(my_objectif(best_features)))

