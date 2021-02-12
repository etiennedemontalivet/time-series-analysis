"""
This module defines methods to load comon classification datasets
in the needed format
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

def make_iris_data():
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
        index=files,
        columns=data_iris.feature_names)
    return X_df, y_df