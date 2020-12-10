"""
This module defines the FeaturesDataset class
"""
import os
from typing import Optional, List
from pathlib import Path
from warnings import warn
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeaturesDataset:
    """
    TODO
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale: bool = True,
        name: str = None,
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
        Get X Shape.

        Returns
        -------
        tuple
            Shape.

        """
        return self.X.shape

    @property
    def index(self):
        """
        Get X indexes.

        Returns
        -------
        pd.Index
            X indexes.

        """
        return self.X.index

    @property
    def X_scaled(self):
        """
        Get X scaled

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
        Scale the features
        """
        self.scaler.fit(self.X)
        X_scaled = self.scaler.transform(self.X)
        self._X_scaled = pd.DataFrame(
            X_scaled, index=self.X.index, columns=self.X.columns
        )

    def subset(self, columns: List[str]):
        """
        Return a subset of the features dataset
        """
        return FeaturesDataset(X=self.X[columns], y=self.y)

    def get_subset_from_label(self, label, suffix: Optional[str] = None):
        """
        Return a features dataset including only the asked label.
        """
        if not isinstance(label, (int, list)):
            raise ValueError("Argument label should be an int or a list of int")
        if isinstance(label, list):
            for i in label:
                if (int(i) < 0) or (int(i) > 100):
                    raise ValueError(
                        "Argument label when a list should contain int from 0..100 in list."
                    )

        if suffix is None:
            new_name = self.name + "_" + str(label)
        else:
            new_name = self.name + suffix

        if isinstance(label, list):
            new_filenames = [x for x in self.X.index if self.y.loc[x] in label]
        elif isinstance(label, int):
            new_filenames = [x for x in self.X.index if self.y.loc[x] == label]

        return FeaturesDataset(
            X=self.X.loc[new_filenames], y=self.y.loc[new_filenames], name=new_name
        )

    def dump(self, directory="./"):
        """
        Dump data into specified directory.
        """
        # If directory did not exist we create it
        if not Path(directory).exists():
            print(f"Creating directory {directory}")
            os.makedirs(directory, exist_ok=True)
        X_dump_path = Path(directory, self.name + "_features_X.pkl")
        y_dump_path = Path(directory, self.name + "_features_y.pkl")
        print(f"Saving X features into '{X_dump_path}'")
        print(f"Saving y features into '{y_dump_path}'")
        self.X.to_pickle(X_dump_path)
        self.y.to_pickle(y_dump_path)

    @classmethod
    def load(cls, dataset_name, directory="./", debug: bool = True):
        """
        Load features based on dataset name
        """
        X_load_path = Path(directory, dataset_name + "_features_X.pkl")
        y_load_path = Path(directory, dataset_name + "_features_y.pkl")
        if X_load_path.exists():
            if debug is True:
                print(f"Loading X features from '{X_load_path}'")
            X = pd.read_pickle(X_load_path)
        else:
            raise FileNotFoundError(f"File {X_load_path} does not exist")
        if y_load_path.exists():
            if debug is True:
                print(f"Loading y features from '{y_load_path}'")
            y = pd.read_pickle(y_load_path)
        else:
            raise FileNotFoundError(f"File {y_load_path} does not exist")
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
