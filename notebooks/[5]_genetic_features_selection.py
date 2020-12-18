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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Create-fake-data" data-toc-modified-id="Create-fake-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Create fake data</a></span></li><li><span><a href="#Define-an-objective-to-minimize" data-toc-modified-id="Define-an-objective-to-minimize-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Define an objective to minimize</a></span></li><li><span><a href="#Apply-genetic-features-selection" data-toc-modified-id="Apply-genetic-features-selection-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Apply genetic features selection</a></span></li><li><span><a href="#Using-a-params_updater" data-toc-modified-id="Using-a-params_updater-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Using a params_updater</a></span></li><li><span><a href="#Compare-results" data-toc-modified-id="Compare-results-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Compare results</a></span></li></ul></div>
# -

# In this notebook we will see how to use the brut force method for iterative features selection.

# ### Create fake data

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, random_state=42)

# Transform data into dataframes
X_df = pd.DataFrame(
    data=X,
    columns=['feat_' + str(i) for i in range(100)],
    index=['file_' + str(i) for i in range(1000)])
y_df = pd.Series(
    data=y,
    index=['file_' + str(i) for i in range(1000)])

# ### Define an objective to minimize

# +
from sklearn.naive_bayes import GaussianNB
from tsanalysis.classification.results import ClassificationResults, CrossValidationResults
from sklearn.model_selection import RepeatedStratifiedKFold

def my_objectif(features_list=list):
    """
    My objectif to minimize
    Here we take a simple cross-validation
    """   
    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=3,random_state=42)
    results = []
    for train_index, test_index in rskf.split(X_df, y_df):
        X_train, X_test = X_df[features_list].iloc[train_index], X_df[features_list].iloc[test_index]
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


# -

# ### Apply genetic features selection

from tsanalysis.features_selection.genetic import genetic_features_selection

# +
# genetic_features_selection??
# -

df = genetic_features_selection(
    features_list=X_df.columns,
    objectif=my_objectif,
    n_population=100,
    n_features=10,
    n_epochs=5,
    genetic_params={'selection_ratio': 0.1,
                    'mutation_features_ratio': 0.1, 
                    'mutation_population_ratio': 0.9}
)

df

best_features_list = (df['population_features'].iloc[-1])[ (df['best_ids'].iloc[-1])[0] ]


# ### Using a params_updater

def my_params_updater(epoch, genetic_params, objectif_output):
    if epoch == 2:
        genetic_params['selection_ratio'] /= 2
    return genetic_params


df2 = genetic_features_selection(
    features_list=X_df.columns,
    objectif=my_objectif,
    params_updater=my_params_updater,
    n_population=100,
    n_features=10,
    n_epochs=5,
    genetic_params={'selection_ratio': 0.1,
                    'mutation_features_ratio': 0.1, 
                    'mutation_population_ratio': 0.9}
)

df2

best_features_list2 = (df2['population_features'].iloc[-1])[ (df2['best_ids'].iloc[-1])[0] ]

# ### Compare results

my_objectif(best_features_list)

my_objectif(best_features_list2)

my_objectif(X_df.columns)

# **Note**  
# Because of the random initial population, results may differs from one run to another. Please increase epochs number if you want to see significant increase in performance.


