"""
@author: Etienne de Montalivet
"""
import pandas
from joblib import Parallel, delayed
from typing import List, Callable, Union, Tuple


class FeatureExtractor:
    """
    An abstract class defining mandatory methods for every FeatureExtractor
    """

    funcs = []

    @staticmethod
    def apply_funcs(
        X: pandas.DataFrame,
        funcs: List[Callable[[pandas.Series], pandas.Series]],
        name: Union[str, Tuple[str], None] = None,
    ) -> pandas.Series:
        """
        Ths function takes into parameter:
          - a DataFrame
          - a list of functions that accept a Series and returns a Series
        If applies each function in the list of function to each column of the given dataframe
        """
        features = [func(X) for func in funcs]
        features = pandas.concat(features)
        if name:
            features.name = name
        return features

    def transform(
        self, X: pandas.DataFrame, groupby_col: str = "filename", n_jobs: int = 4,
    ) -> pandas.DataFrame:
        """
        Perfom computation of features on each sub-dataframe after groupping input DataFrame
        using groupby_col.

        This returns a dataframe.
        You can choose how many subdataframes should be handled in parallel using n_jobs.
        """
        # We want to know the columns in order to discard index columns once data is grouped
        colnames = X.columns
        with Parallel(n_jobs=n_jobs, prefer="processes") as parallel:
            res = parallel(
                delayed(self.apply_funcs)(subdf[colnames], self.funcs, name=idx)
                for idx, subdf in X.groupby(groupby_col)
            )
        return pandas.DataFrame(res)
