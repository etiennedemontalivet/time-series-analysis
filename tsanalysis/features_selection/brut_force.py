"""
This module defines the brut force features selection method
"""

import inspect
import os.path
import time
from typing import List, Callable

import pandas as pd
import numpy as np


# pylint: disable=too-many-locals
def brut_force_features_selection(
    features_list: List[str],
    objectif: Callable,
    max_features: int,
    init_list: List[str] = None,
    n_per_epoch: int = 1,
    accept_duplicates: bool = False,
    verbose: bool = True,
):
    """Brut force features selection

    This method runs a brut force iterative approach on features selection.
    It takes a features list with an objectif to minimize.

    Parameters
    ----------
    features_list : list of str
        The total features list into which the brut force does the selection.
    objectif : Callable
        The objectif to minimize. This must have `features_list` as argument.
        It must return a dictionary with at least a 'score' key corresponding
        to the score value of one objectif call given a features_list.
        The could be for instance a cross-validation.
    max_features : int
        Max number of features to select.
    init_list : list of str, default=None
        List of initial features list to start with. If None, the features
        list starts empty. The default is None.
    n_per_epoch : int, default=1
        Number of best features to select per epoch. The default is 1.
    accept_duplicates : bool, default=False
        If True, a feature can be selected multiple times during the optimization.
        If False, each feature can be selected only once. The default is False.
    verbose : bool, default=True
        If True, verbose is enable. The default is True.

    Raises
    ------
    TypeError
        If objectif does not contain `features_list` as argument.
    TypeError
        If objectif does not return a dictionary containing a 'score' key.

    Returns
    -------
    pd.DataFrame
        History of optimization.

    Note
    ----
    The objectif function could contain as many keys as wanted. All these metrics
    are added in logs and history.

    Because the optimization could take times, logs are saved after each epoch
    in a temporary file, in the directory `.tmp_brutforce`. In case of crash,
    you could retrieve the logs here.

    See also
    --------
    genetic_features_selection

    """

    # Initialization
    if init_list is None:
        init_list = []
    best_features_list = init_list.copy()
    if "features_list" not in str(inspect.signature(objectif)):
        raise TypeError(r"{objectif} must have 'features_list' as argument.")
    if not os.path.exists(".tmp_brutforce"):
        os.makedirs(".tmp_brutforce")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Init variables
    global_res_col = ""
    res_to_save = ""
    all_res = []
    all_res_index = []

    # Loop on epoch
    for epoch in range(0, max_features - len(init_list), n_per_epoch):
        metrics = []
        scores = []
        feature_names = []
        for feat in features_list:
            if accept_duplicates is True or feat not in best_features_list:
                output_dict = objectif(features_list=best_features_list + [feat])
                feature_names.append(feat)
                if output_dict["score"] is None:
                    raise TypeError(
                        "Output of objectif has to be a dictionary containing a 'score' key."
                    )
                scores.append(list(output_dict.values()))
                metrics = list(output_dict.keys())
        scores_df = pd.DataFrame(data=scores, columns=metrics, index=feature_names)

        # Extract best features names using the best clf score
        scores_df = scores_df.sort_values(by="score")
        best_feature_names = scores_df.index[:n_per_epoch].tolist()

        # Update best_features_list
        best_features_list += best_feature_names

        # Save associated metrics in tmp file in case of crash
        with open(".tmp_brutforce/run_" + timestr, "a") as f:
            if epoch == 0:
                f.write(str(init_list) + "\n")
                global_res_col = "epoch,feature_name,"
                for metric in metrics:
                    global_res_col += str(metric) + ","
                global_res_col = global_res_col[:-1]  # remove last comma...
                f.write(global_res_col + "\n")

            for feat in best_feature_names:
                res_to_save = str(epoch) + "," + feat + ","
                for metric in metrics:
                    res_to_save += str(scores_df[metric].loc[feat]) + ","
                res_to_save = res_to_save[:-1]  # remove last comma
                f.write(res_to_save + "\n")

        # Save results with metrics to return them
        for feat in best_feature_names:
            all_res_index.append(feat)
            all_res.append(np.concatenate([[epoch, feat], scores_df.loc[feat].values]))

        if verbose is True:
            print(
                "Epoch "
                + str(epoch)
                + ": best new features are "
                + str(best_feature_names)
                + "(total: "
                + str(len(best_features_list))
                + "/"
                + str(max_features)
                + ")"
            )
            print(scores_df.loc[best_feature_names])
            print("================================================================")
            print()

    return pd.DataFrame(
        data=all_res, index=all_res_index, columns=global_res_col.split(",")
    )
