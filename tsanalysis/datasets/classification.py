"""
Datasets
========

This module defines common datasets to be loaded in the framework specific format.
"""

import pandas as pd
from sklearn.datasets import load_iris

#pylint: disable=no-member
def make_iris_data():
    """
    Make an iris dataset with specific format.

    Returns
    -------
    X_df : pd.DataFrame
        A dataframe with iris data. Columns are the features, indexes are
        fake filenames.

    y_df : pd.Series
        The iris labels with fake filenames as indexes.

    """
    data_iris = load_iris()
    X, y = data_iris.data, data_iris.target

    # Creating dataframe and serie with fake filenames
    files = ["file_" + str(i) for i in range(len(y))]
    y_df = pd.Series(data=y, index=files)
    X_df = pd.DataFrame(data=X, index=files, columns=data_iris.feature_names)
    return X_df, y_df
