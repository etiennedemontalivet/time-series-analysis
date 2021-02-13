"""
This module defines the FeaturesDataset class
"""
import os
from typing import Optional, List
from pathlib import Path
from warnings import warn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pyarrow.parquet as pq
import pyarrow as pa

class FeaturesDataset:
    """
    This class aims at easying the features analysis. The object contains
    the common `X` and `y`. The typical operations are:

    - dump / load a set of features
    - extract a subset of features (based on features, indexes or labels)
    - plot a specific feature distribution per label
    - scale a set of feature

    Parameters
    ----------
    X: pd.DataFrame
        A dataframe of shape (n_samples, n_features). Columns names have to be
        the features. Rows names the ids of events (typically filenames).

    y : pd.Series
        A serie containing the labels with the ids of events as indexes.

    scale : boolean, default=True
        If True, scaling is automatically applied on `X`. The fefault is True.
    
    name : str, default=None
        The name of the features dataset. This is used when dumping
        the dataset. If None, 'unamed_features_set' is used. The default
        is None.

    scaler : sklearn scaler, default=StandardScaler()
        The scaler to be used for scaling the features. It could
        be any scaler from `sklearn preprocessing <https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler>`__.
        You could write a custom scaler class by writing custom `fit`
        and `transform` methods. The default is `StandardScaler()`

    Attributes
    ----------
    X_scaled : pd.DataFrame

    index : pd.Index

    shape : tuple

    Notes
    -----

    References
    ----------

    Examples
    --------
    >>> from tsanalysis.datasets import make_iris_data
    >>> X_df, y_df = make_iris_data()

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from tsanalysis.classification.results import ClassificationResults, CrossValidationResults
    >>> from sklearn.model_selection import RepeatedStratifiedKFold

    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale: bool=True,
        name: str=None,
        scaler=None,
    ):
        if name is None:
            self.name = "unamed_features_set"
        else:
            self.name = name
        self.X = X.reindex(y.index)
        self.y = y

        # Remove duplicates
        if any(self.X.index.duplicated(keep="last")):
            self.X = self.X.loc[~self.X.index.duplicated(keep="last")]
        if any(self.y.index.duplicated(keep="last")):
            self.y = self.y.loc[~self.y.index.duplicated(keep="last")]
            print("WARNING: Duplicates has been found and removed.")

        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler
        self._X_scaled = None
        if scale:
            self.scale()

    @property
    def shape(self):
        """
        X Shape.

        Returns
        -------
        tuple
            Shape.

        """
        return self.X.shape

    @property
    def index(self):
        """
        X indexes.

        Returns
        -------
        pd.Index
            X indexes.

        """
        return self.X.index

    @property
    def X_scaled(self):
        """
        The scaled features. :meth:`scale` has to be called before (if not called
        automatically at the init).

        Returns
        -------
        res : pd.DataFrame
            X scaled if scaling has been fitting, else None.

        """
        res = self._X_scaled
        if res is None:
            warn(
                "Scaled has not been fitted yet. You can do that using .scale() method."
            )
        return res

    def scale(self):
        """
        Scale the features using the `scaler`. By default, this method is
        automatically called at init.
        """
        self.scaler.fit(self.X)
        X_scaled = self.scaler.transform(self.X)
        self._X_scaled = pd.DataFrame(
            X_scaled, index=self.X.index, columns=self.X.columns
        )

    def get_subset_from_features(self, columns: List[str]):
        """
        Extract a sub-features dataset.

        Parameters
        ----------
        columns : list of str
            The features to extract.

        Returns
        -------
        FeaturesDataset
        """
        return FeaturesDataset(X=self.X[columns], y=self.y)

    def get_subset_from_indexes(self, indexes: List[str]):
        """
        Extract a features dataset based on subset of indexes.

        Parameters
        ----------
        indexes : list of str
            The indexes to extract.

        Returns
        -------
        FeaturesDataset
        """
        return FeaturesDataset(X=self.X.loc[indexes], y=self.y[indexes])

    def get_subset_from_labels(self, labels, suffix: str=None):
        """
        Extract a features dataset including only the asked label(s).

        Parameters
        ----------
        labels : list of int
            The label to extract.

        suffix : str, default=None
            The suffix that is added to the name of the new FeaturesDataset.
            If None, the `labels` is added to the current name. The default
            is None

        Returns
        -------
        FeaturesDataset

        """
        if not isinstance(labels, list):
            raise ValueError("Argument label should be a list of int")

        if suffix is None:
            new_name = self.name + "_" + str(labels)
        else:
            new_name = self.name + suffix

        new_indexes = [x for x in self.X.index if self.y.loc[x] in label]

        return FeaturesDataset(
            X=self.X.loc[new_indexes], y=self.y.loc[new_indexes], name=new_name
        )

    def dump(self, directory="./"):
        """
        Dump data into specified directory.

        Parameters
        ----------
        directory : str, default="./"
            The destination directory of the features dataset parquet file.

        Returns
        -------
        None
        """
        # If directory did not exist we create it
        if not Path(directory).exists():
            print(f"Creating directory {directory}")
            os.makedirs(directory, exist_ok=True)
        X_dump_path = Path(directory, self.name.split('\\')[-1] + ".features.parquet")
        print(f"Saving the FeaturesDataset  into '{X_dump_path}'")
        tmp = self.X.copy()
        tmp['label'] = self.y
        table = pa.Table.from_pandas(tmp)
        pq.write_table(table, X_dump_path)

    @classmethod
    def load(cls, dataset_name, directory="./", debug:int=1):
        """
        Load features based on dataset name from a parquet file.

        .. warning::
            The `dataset_name` parameter has to not include the extension: '.features.parquet'.

        Parameters
        ----------
        dataset_name : str
            The dataset name to load without the extension.

        directory : str, default="./"
            The directory from which to load the dataset. The default is "./".

        verbose : int, default=1
            Verbosity level to display information. The default is 1.

        Returns
        -------
        FeaturesDataset

        """
        X_load_path = Path(directory, dataset_name + ".features.parquet")
        if X_load_path.exists():
            if verbose > 0:
                print(f"Loading  FeaturesDataset from '{X_load_path}'")
            tmp = pq.read_table(X_load_path)
            tmp = tmp.to_pandas()
            X = tmp.drop(columns = 'label')
            y = tmp['label']
        else:
            raise FileNotFoundError(f"File {X_load_path} does not exist")
        
        return cls(X, y, name=dataset_name)


def features_concat(features: List[FeaturesDataset], name: str = None):
    """
    Return a single features dataset from a list of features datasets
    """
    if name is None:
        name_concat = ""
        for feat in features:
            name_concat = name_concat + feat.name + "__"
    else:
        name_concat = name

    return FeaturesDataset(
        X=pd.concat([feat.X for feat in features]),
        y=pd.concat([feat.y for feat in features]),
        name=name_concat,
    )
