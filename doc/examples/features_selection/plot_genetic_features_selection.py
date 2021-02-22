"""
Genetic features selection
=============================

Learn how to do features selection using a genetic selection approach.
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
# **Genetic features selection**
from tsanalysis.features_selection import genetic_features_selection
df = genetic_features_selection(
    features_list=X_df.columns,
    objectif=my_objectif,
    n_population=100,
    n_features=10,
    n_epochs=4,
    genetic_params={'selection_ratio': 0.1,
                    'mutation_features_ratio': 0.1, 
                    'mutation_population_ratio': 0.9}
)
df.head()

#%%
best_features = (df['population_features'].iloc[-1])[ (df['best_ids'].iloc[-1])[0] ]
print(best_features)

#%%
# **Updating parameters during convergence**
#
# You can define a ``params_updater`` function that update the genetic parameters
# to use. The behaviour could be compared to methods in gradient descent that
# updated some parameters during the backpropagation (such as adam).
def my_params_updater(epoch, genetic_params, objectif_output):
    """ Simple paramaters updater
    When epoch 2 is reached, we divide the number of selected
    features by 2.
    """
    if epoch == 2:
        genetic_params['selection_ratio'] /= 2
    return genetic_params

#%%
df2 = genetic_features_selection(
    features_list=X_df.columns,
    objectif=my_objectif,
    params_updater=my_params_updater, # Call params updater
    n_population=100,
    n_features=10,
    n_epochs=4,
    genetic_params={'selection_ratio': 0.1,
                    'mutation_features_ratio': 0.1, 
                    'mutation_population_ratio': 0.9}
)
df2.head()

#%%
best_features2 = (df2['population_features'].iloc[-1])[ (df2['best_ids'].iloc[-1])[0] ]
print(best_features2)

#%%
# **Compare results**
print("Without genetic selected features:\n" + str(my_objectif(X_df.columns)))
print("With genetic selected features:\n" + str(my_objectif(best_features)))
print("With genetic selected features and params updater:\n" + str(my_objectif(best_features)))

#%%
# .. note::
#
#   Because of the random initial population, results may differs from one run to another.
#   Please increase epochs number if you want to see significant increase in performance.
