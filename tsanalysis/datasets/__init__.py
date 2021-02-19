"""
Datasets
=========

The framework uses mainly dataframes.

**Raw data** have to be formatted using dataframes, indexes being the filenames of
corresponding sensor's acquisitions, columns being the temporality (non column names needed).

**Features data** have to be formatted using dataframes, indexes being the filenames of
corresponding sensor's acquisitions, columns being features names.

"""

from tsanalysis.datasets.classification import make_iris_data, make_classification
from tsanalysis.datasets.raw import make_windows_ts_data
