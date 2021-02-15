"""
This module defines the genetic features selection method
"""

import inspect
import os.path
import time
from typing import List, Callable

import pandas as pd
import numpy as np
from numpy.random import default_rng


# pylint: disable=too-many-locals,too-many-arguments,too-many-nested-blocks,too-many-branches,too-many-statements
def genetic_features_selection(
    features_list: List[str],
    objectif: Callable,
    n_population: int,
    n_features: int,
    genetic_params: dict,
    n_epochs: int = 10,
    init_population: List[List[str]] = None,
    params_updater: Callable = None,
    verbose: bool = True,
):
    """ Genetic features selection

    This method runs a genetic algorithm to perform features selection.
    .. todo: refactor

    Parameters
    ----------
    features_list : list of str
        The total features list into which the genetic algorithm does the selection.
    objectif : Callable
        The objectif to minimize. This must have ``features_list`` as argument.
        It must return a dictionary with at least a 'score' key corresponding
        to the score value of one objectif call given a features_list.
        The could be for instance a cross-validation.
    n_population : int
        The size of the genetic population.
    n_features : int
        The number of features per individual.
    genetic_params : dict
        A dictionary containing the following genetic parameters:
            - ``selection_ratio``: population ratio of selection for cross-over
            - ``mutation_features_ratio``: ratio of features to mutate
            - ``mutation_population_ratio``: population ratio to mutate
    n_epochs : int, default=10
        Number of epochs. The default is 10.
    init_population : list of list of str, default=None
        The initial population. If None, a random population is created. The default is None.
    params_updater : Callable, default=None
        A genetic parameters updater. If not None, this must be a Callable with
        three arguments:
            - ``epoch``: the current epoch number
            - ``genetic_params``: the current genetic parameters
            - ``objectif_output``: the output dictionary of the objectif function
        This could be used to update the genetic paramters in specific contexts.
        It must return a ``genetic_params`` dictionary.
        The default is None.
    verbose : bool, default=True
        If True, verbose is enable. The default is True.

    Raises
    ------
    ValueError
        If ``init_population`` shape does not match ``n_population`` and ``n_features``
        values.
        If ``genetic_params`` misses a key.

    TypeError
        If ``features_list`` is not in ``objectif`` signature.
        If ``epoch`` is not in ``params_updater`` signature.
        If ``genetic_params`` is not in ``params_updater`` signature.
        If ``objectif_output`` is not in ``params_updater`` signature.

    Returns
    -------
    pd.DataFrame
        A dataframe containing, per epoch, the best individual score, the objectif
        metrics (if any) and the whole population.

    Note
    ----
    The objectif function could contain as many keys as wanted. All these metrics
    are added in logs and final result.

    Because the optimization could take times, logs are saved after each epoch
    in a temporary file, in the directory ``.tmp_genetic``. In case of crash,
    you could retrieve the logs of the last population here with associated best
    individual score.

    See also
    --------
    brut_force_features_selection

    """

    # Initialization
    rng = default_rng()
    population_features = []
    if init_population is None:
        # Init features randomly
        for i_pop in range(n_population):
            population_features.append(
                features_list[
                    rng.choice(len(features_list), size=n_features, replace=False)
                ].tolist()
            )
    else:
        if len(init_population) != n_population:
            raise ValueError(
                rf"'init_population' must have {n_population} sublists of features."
            )
        for i_pop in range(n_population):
            if len(init_population[i_pop]) != n_features:
                raise ValueError(
                    rf"'init_population' at {i_pop} must have {n_features} features."
                )

    # Check arguments
    if "selection_ratio" not in genetic_params.keys():
        raise ValueError(r"'genetic_params' must have a 'selection_ratio' key.")
    if "mutation_features_ratio" not in genetic_params.keys():
        raise ValueError(r"'genetic_params' must have a 'mutation_features_ratio' key.")
    if "mutation_population_ratio" not in genetic_params.keys():
        raise ValueError(
            r"'genetic_params' must have a 'mutation_population_ratio' key."
        )
    if "features_list" not in str(inspect.signature(objectif)):
        raise TypeError(r"{objectif} must have 'features_list' as argument.")
    if params_updater is not None:
        if "epoch" not in str(inspect.signature(params_updater)):
            raise TypeError(rf"{params_updater} must have 'epoch' as argument.")
        if "genetic_params" not in str(inspect.signature(params_updater)):
            raise TypeError(
                rf"{params_updater} must have 'genetic_params' as argument."
            )
        if "objectif_output" not in str(inspect.signature(params_updater)):
            raise TypeError(
                rf"{params_updater} must have 'objectif_output' as argument."
            )

    # Save temporary files
    if not os.path.exists(".tmp_genetic"):
        os.makedirs(".tmp_genetic")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Convert percentage to integer:
    genetic_params_epoch = genetic_params
    n_best_individuals = int(genetic_params_epoch["selection_ratio"] * n_population)
    n_mutation_features = int(
        genetic_params_epoch["mutation_features_ratio"] * n_features
    )
    n_mutation_indiviuals = int(
        genetic_params_epoch["mutation_population_ratio"] * n_population
    )
    if verbose is True:
        print("******* INFO *******")
        print(
            rf"CROSS-OVER: {n_best_individuals}/{n_population} best individal(s) selected."
        )
        print(
            rf"MUTATION: {n_mutation_features} randomly changed feature(s) from "
            + rf"{n_mutation_indiviuals}/{n_population} individual(s)."
        )
        print("********************\n")

    # Variables
    all_populations = []
    all_res_index = []
    all_res = []

    for i_epoch in range(n_epochs):
        # Update params ?
        if params_updater is not None and i_epoch > 0:
            genetic_params_epoch_old = genetic_params_epoch.copy()
            genetic_params_epoch = params_updater(
                epoch=i_epoch,
                genetic_params=genetic_params_epoch,
                objectif_output=objectif(
                    features_list=population_features[scores_df.index[0]]
                ),
            )

            if genetic_params_epoch_old != genetic_params_epoch:
                n_best_individuals = int(
                    genetic_params_epoch["selection_ratio"] * n_population
                )
                n_mutation_features = int(
                    genetic_params_epoch["mutation_features_ratio"] * n_features
                )
                n_mutation_indiviuals = int(
                    genetic_params_epoch["mutation_population_ratio"] * n_population
                )
                if verbose is True:
                    print("******* INFO *******")
                    print(
                        rf"CROSS-OVER: {n_best_individuals}/{n_population} best "
                        + "individal(s) selected."
                    )
                    print(
                        rf"MUTATION: {n_mutation_features} randomly changed feature(s) "
                        + rf"from {n_mutation_indiviuals}/{n_population} individual(s)."
                    )
                    print("********************\n")

        # Log the population per epoch
        all_populations.append(population_features)

        # Get score per individual
        metrics = []
        scores = []
        for i_pop in range(n_population):
            output_dict = objectif(features_list=population_features[i_pop])
            if output_dict["score"] is None:
                raise TypeError(
                    "Output of objectif has to be a dictionary containing a 'score' key."
                )
            scores.append(list(output_dict.values()))
            metrics = list(output_dict.keys())
        scores_df = pd.DataFrame(
            data=scores, columns=metrics, index=np.arange(n_population)
        )

        # Extract best individuals using the best objectif score
        scores_df = scores_df.sort_values(by="score")
        best_individuals_ids = scores_df.index[:n_best_individuals].tolist()

        # Create children
        children_features = []
        children_randomness = []
        for i_pop in range(n_population):
            # pick 2 parents
            dad_id, mom_id = np.random.choice(
                best_individuals_ids, size=2, replace=False
            )
            child_feat = []
            common_feat = 0  # number features in common
            for i_feat in range(n_features):
                dad_feat = population_features[dad_id][i_feat]
                mom_feat = population_features[mom_id][i_feat]

                # Check if child has alreday mom and dad features
                if dad_feat in child_feat and mom_feat in child_feat:
                    common_feat += 1
                    # select random feature to add
                    rand_feat = features_list[np.random.randint(len(features_list))]
                    while rand_feat in child_feat:
                        rand_feat = features_list[np.random.randint(len(features_list))]
                    child_feat.append(rand_feat)

                # else if Dad's turn
                elif i_feat % 2 == 1:
                    if dad_feat in child_feat:
                        child_feat.append(mom_feat)
                    else:
                        child_feat.append(dad_feat)
                # else, Mom's turn
                else:
                    if mom_feat in child_feat:
                        child_feat.append(dad_feat)
                    else:
                        child_feat.append(mom_feat)

            children_features.append(child_feat)
            children_randomness.append(common_feat)

        # Mutation
        check_common = 0
        # Update only 100*MUTATION_PER% of population
        children_ids_to_update = rng.choice(
            n_population, size=n_mutation_indiviuals, replace=False
        )
        for i_pop in range(n_population):
            if i_pop in children_ids_to_update:
                # if already introduced more than n_new_features, skip this step
                if children_randomness[i_pop] < n_mutation_features:
                    new_features_ids = np.random.choice(
                        np.arange(n_features),
                        size=(n_mutation_features - children_randomness[i_pop]),
                        replace=False,
                    )
                    for new_feat_id in new_features_ids:
                        rand_feat = features_list[np.random.randint(len(features_list))]
                        while rand_feat in children_features[i_pop]:
                            rand_feat = features_list[
                                np.random.randint(len(features_list))
                            ]
                        children_features[i_pop][new_feat_id] = rand_feat
                else:
                    check_common = +1
        if check_common > n_population / 2:
            print(
                "Warning: "
                + str(check_common)
                + " children have mom and dad common features"
            )

        # Save associated metrics in tmp file in case of crash
        if i_epoch % 2 == 0:
            with open(".tmp_genetic/run_" + timestr + "_0", "w") as file:
                file.write("best score: " + str(scores_df["score"].iloc[0]) + "\n")
                file.write("best ids: " + str(best_individuals_ids) + "\n")
                file.write(str(population_features))
        else:
            with open(".tmp_genetic/run_" + timestr + "_1", "w") as file:
                file.write("best score: " + str(scores_df["score"].iloc[0]) + "\n")
                file.write("best ids: " + str(best_individuals_ids) + "\n")
                file.write(str(population_features))

        # Save results with metrics to return them
        all_res_index.append(i_epoch)
        loc_res = []
        for col in scores_df.columns:
            loc_res.append(scores_df[col].iloc[0])
        loc_res.append(best_individuals_ids)
        loc_res.append(population_features)
        all_res.append(loc_res)
        all_res_col = scores_df.columns.tolist() + ["best_ids", "population_features"]

        if verbose is True:
            print(
                "Epoch "
                + str(i_epoch)
                + ": best score = "
                + str(scores_df["score"].iloc[0])
            )
            print(scores_df.iloc[0])
            print("================================================================")
            print()

        # Update new population...
        population_features = children_features

    return pd.DataFrame(data=all_res, index=all_res_index, columns=all_res_col)
