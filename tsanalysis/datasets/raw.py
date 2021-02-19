"""
Raw data datasets
=================

This module defines a method to create fake raw data in the framework specific format.
"""

import numpy as np
import pandas as pd

# pylint: disable=no-member
def make_windows_ts_data(
    sampling_rate: int = 1000,
    duration: float = 0.256,
    n_events: int = 100,
    id_names: list = None,
    return_y: bool = True,
    n_classes: int = 3,
):
    """
    Make a raw dataset of windows extracted from time-series.

    Parameters
    ----------
    sampling_rate : int, default=1000
        Number of samples per second. The default is 1000.
    duration : float, default=0.256
        The window duration in second(s). The default is 0.256.
    n_events : int, default=100
        The number of events in the created dataset. The default is 100.
    id_names : list of str, default=None
        The ids of the events used as indexes of the returned data. If None, fake
        filenames are used. The default is None.
    return_y : bool, default=True
        If True, fake labels in the framework format are also returned. The default
        is True.
    n_classes : int, default=3
        Number of classes to use in ``y``. The default is 3.

    Returns
    -------
    data : pd.DataFrame
        A dataframe with raw data. Columns are the time, indexes are
        fake filenames.

    y : pd.Series
        The fake labels with fake filenames as indexes.

    """
    values = np.random.random((n_events, int(sampling_rate * duration)))
    if id_names is None:
        id_names = ["file_" + str(i) for i in range(n_events)]
    data = pd.DataFrame(data=values, index=id_names)

    if return_y is True:
        y = pd.Series(data=np.random.randint(0, n_classes, n_events), index=data.index)

    return data, y if return_y is True else data
