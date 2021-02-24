"""
Classification Results
======================

This module defines classes for easy computation of classification results.

"""

from typing import List, Dict
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import metrics


class ClassificationResults:
    """Analyze single classification results

    All metrics are computed using sklearn metrics from `here <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`__.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth (correct) target values.

    y_pred : pd.Series
        Estimated targets as returned by a classifier.

    labels : array-like of shape (`n_classes_`), default=None
        List of labels to index the matrix. This may be used to reorder or select
        a subset of labels. If None is given, those that appear at least once in
        y_true or y_pred are used in sorted order.


    Attributes
    ----------
    n_classes_ : int
        number of classes

    classes_ : ndarray of shape (`n_classes_`)
        class labels

    accurracy_ : float
        Accuracy classification score.
        In multilabel classification, this function computes subset accuracy:
        the set of labels predicted for a sample must exactly match the
        corresponding set of labels in y_true.
        See more details `here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score>`__.

    matthews_corrcoef_ : float
        number of training samples observed in each class.

    f1_weighted_ : float
        F1 score of the F1 scores of each class for the multiclass task.
        Calculate metrics for each label, and find their average weighted
        by support (the number of true instances for each label).
        See more details `here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score>`__.

    f1_micro_ : float
        F1 score of the F1 scores of each class for the multiclass task.
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
        See more details `here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score>`__.

    tp_ : int
        Number of True Positives.

    fp_ : int
        Number of False Positives = number of misclassifications

    confusion_matrix_ : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th column entry indicates the
        number of samples with true label being i-th class and prediced label
        being j-th class.

    See also
    --------
    CrossValidationResults

    Notes
    -----

    References
    ----------

    Examples
    --------
    >>> import pandas as pd
    >>> from tsanalysis.classification import ClassificationResults
    >>> y_true = pd.Series( data=[2, 0, 2, 2, 0, 1], index=['f0', 'f1', 'f3', 'f4', 'f5', 'f6'] )
    >>> y_pred = pd.Series( data=[0, 0, 2, 2, 0, 2], index=['f0', 'f1', 'f3', 'f4', 'f5', 'f6'] )
    >>> cr = ClassificationResults(y_true, y_pred)

    >>> cr.metrics
    {'matthews_corrcoef': 0.45226701686664544,
     'accuracy': 0.6666666666666666,
     'f1_weighted': 0.6,
     'f1_micro': 0.6666666666666666,
     'tp': 4,
     'fp': 2,
     'confusion_matrix': array([[2, 0, 0],
            [0, 0, 1],
            [1, 0, 2]], dtype=int64)}

    >>> cr.classes_
    array([0, 1, 2], dtype=int64)

    >>> cr.misclassified
    Index(['f0', 'f6'], dtype='object')

    >>> cr.plot_confusion_matrix(labels_names=["cat", "dog", "sphinx"])
    # Plot a nice confusion matrix
    """

    def __init__(self, y_true: pd.Series, y_pred: pd.Series, labels: list = None):
        if not isinstance(y_pred, pd.Series):
            raise ValueError("y_pred has to be a pd.Series. Please convert it.")
        if not isinstance(y_true, pd.Series):
            raise ValueError("y_true has to be a pd.Series. Please convert it.")
        self.y_pred = y_pred
        self.y_true = y_true

        # Compute matthews correlation coefficient
        if np.sum(y_pred) > 0:
            matthews_corr_coef = metrics.matthews_corrcoef(self.y_true, self.y_pred)
        else:
            matthews_corr_coef = 0

        self.classes_ = np.unique(self.y_true)
        self.n_classes_ = len(self.classes_)

        # Store it if it is defined, else store 0
        self.matthews_corrcoef_ = matthews_corr_coef if matthews_corr_coef else 0

        # Do the same for F1 Score which can sometimes return None
        f1_score_weighted = metrics.f1_score(
            self.y_true, self.y_pred, average="weighted"
        )
        f1_score_micro = metrics.f1_score(self.y_true, self.y_pred, average="micro")
        # If F1 Score is None we put 0
        self.f1_weighted_ = f1_score_weighted if f1_score_weighted else 0
        self.f1_micro_ = f1_score_micro if f1_score_micro else 0

        # Accuracy can never be None so we do not need to check if it is defined
        self.accuracy_ = metrics.accuracy_score(self.y_true, self.y_pred)

        # Confusion matrix
        self.confusion_matrix_ = metrics.confusion_matrix(
            self.y_true, self.y_pred, labels=labels
        )

        # true positives
        self.tp_ = np.trace(self.confusion_matrix_)
        # false positives
        self.fp_ = np.sum(self.confusion_matrix_) - self.tp_

    @property
    def metrics(self) -> Dict[str, float]:
        """
        Returns a dictionary of all metrics for convenience.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``matthews_corrcoef``: matthew correlation coefficient. See here for more details.
            - ``accuracy``: accuracy. See here for more details.
            - ``f1_weighted``: the f1 weighted score. See here for more details.
            - ``f1_micro``: the f1 micro score. See here for more details.
            - ``tp``: true positives number.
            - ``fp``: false positives number.
            - ``confusion_matrix``: the confusion matrix.

        See also
        --------
        dict

        """
        _metrics = {
            "matthews_corrcoef": self.matthews_corrcoef_,
            "accuracy": self.accuracy_,
            "f1_weighted": self.f1_weighted_,
            "f1_micro": self.f1_micro_,
            "tp": int(self.tp_),
            "fp": int(self.fp_),
            "confusion_matrix": self.confusion_matrix_.tolist(),
        }
        return _metrics

    @property
    def misclassified(self) -> pd.Index:
        """
        Classification misclassifieds.

        Returns
        -------
        pd.Index
            Indexes of misclassifieds.

        See also
        --------
        dict

        """
        y_misclassified = self.y_true[self.y_true != self.y_pred]
        return y_misclassified.index

    def plot_confusion_matrix(
        self, labels_names: list = None, title="Confusion matrix", cmap=None, cbar=True
    ):
        """
        Plot the confusion matrices mean

        Parameters
        ----------
        labels_names : list, optional
            The names of the labels. The default is None.
        title : str, optional
            Figure's title. The default is 'Confusion matrix'.
        cmap : str, optional
            cmap to use. If None, 'Blues' is used. The default is None.
        cbar : boolean, optional
            If true, it plots the colorbar too. The default is True.

        Returns
        -------
        None.

        """
        cm = self.confusion_matrix_

        if labels_names is None:
            labels_names = np.unique(self.y_true).tolist()

        if cmap is None:
            cmap = plt.get_cmap("Blues")

        raw_cm = cm
        cm = 100 * cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=100)
        plt.title(title)
        if cbar:
            plt.colorbar()

        if labels_names is not None:
            tick_marks = np.arange(len(labels_names))
            plt.xticks(tick_marks, labels_names, rotation=45)
            plt.yticks(tick_marks, labels_names)

        thresh = cm.max() / 1.5
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                "{:0.2f}%\n({:d}/{:d})".format(
                    cm[i, j], raw_cm[i, j], np.sum(raw_cm[i, :])
                ),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("True label")
        plt.xlabel(
            "Predicted label\nf1_weighted={:0.2f}; mcc={:0.2f}".format(
                self.f1_weighted_, self.matthews_corrcoef_
            )
        )
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__str__()})"

    def __str__(self) -> str:
        return str(self.dict())

    def dict(self):
        """
        Returns a dict representation of the results

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``metrics``: classification :meth:`metrics`
            - ``predictions``: a dictionary containing ``y_true`` and ``y_pred`` results.
            - ``misclassified``: classification :meth:`misclassified`
        """
        data = {}
        data["metrics"] = self.metrics
        data["predictions"] = {"y_true": self.y_true, "y_pred": self.y_pred}
        data["misclassified"] = self.misclassified
        return data


class CrossValidationResults:
    """Analyze cross-validation results.

    This class should be used in conjunction with :class:`.ClassificationResults`
    whenever cross-validations are performed.

    Parameters
    ----------
    results: List of ClassificationResults

    select_by : str, optional
        Metric to use to sort the results. The default is "matthews_corrcoef".

    Attributes
    ----------
    history : List of ClassificationResults
        The history of classifications that is passed as input parameter.

    misclassifieds : pd.DataFrame
        A dataframe containing the indexes of the misclassifieds with
        statistics: occurence, predicted and true targets.
        **Note**: Number of splits in CV is directly linked to the
        misclassifieds occurences analysis.

    metrics : dict
        Aggregated metrics of CV.

    confusion_matrix_mean : ndarray of shape (`n_classes_`, `n_classes_`)
        Confusion matrix whose i-th row and j-th column entry indicates the
        number of samples with true label being i-th class and prediced label
        being j-th class.

    See also
    --------
    ClassificationResults

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

    >>> rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,random_state=42)
    >>> results = []
    >>> for train_index, test_index in rskf.split(X_df, y_df):
    >>>     X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
    >>>     y_train, y_test = y_df[train_index], y_df[test_index]
    >>>     clf = RandomForestClassifier(max_depth=2, random_state=0)
    >>>     clf.fit(X_train, y_train)
    >>>     y_pred = pd.Series(
    >>>         data=clf.predict(X_test),
    >>>         index=y_test.index
    >>>     )
    >>>     results.append(ClassificationResults(y_true=y_test, y_pred=y_pred))
    >>> cv_res = CrossValidationResults(results)

    >>> cv_res.confusion_matrix_mean

    >>> cv_res.metrics

    >>> cv_res.misclassified

    >>> cv_res.plot_confusion_matrix_mean(['setosa', 'versicolor', 'virginica'])

    >>> cv_res.plot_metrics()

    """

    def __init__(
        self, results: List[ClassificationResults], select_by="matthews_corrcoef"
    ):
        self.history = results
        loc_metrics, misclassified = zip(
            *[(res.metrics, res.misclassified) for res in results]
        )
        self.df = pd.DataFrame(loc_metrics)
        self.__misclassified__ = pd.Series(misclassified)
        self.sorted_df = self.df.sort_values(by=select_by, ascending=False, axis=0)
        self.select_by = select_by
        self._mean = self.df.mean()

        # Compute mean of confusion matrices
        for i in range(len(self.history)):
            if (
                self.history[i].confusion_matrix_.shape[0]
                != self.history[0].confusion_matrix_.shape[0]
            ):
                raise ValueError(
                    "The confusion matrices in you history do not have the same size. Force \
                    its size by using 'labels' in ClassificationResults instantiation."
                )
        cm_sum = self.history[0].confusion_matrix_ * 0
        for res in self.history:
            cm_sum += res.confusion_matrix_
        self.confusion_matrix_mean = cm_sum / len(self.history)

    @property
    def misclassified(self) -> pd.Series:
        """
        A dataframe containing the indexes of the misclassifieds with
        statistcis: occurence, predicted and true targets. **Note**: Number
        of splits in CV is directly linked to the misclassifieds occurences
        analysis.

        Returns
        -------
        pd.DataFrame
        """
        values = pd.Series(np.concatenate(self.__misclassified__)).value_counts()
        values.name = "occurence"

        all_true = pd.concat([x.y_true for x in self.history])
        all_pred = pd.concat([x.y_pred for x in self.history])

        # remove duplicated indexes from true values
        all_true = all_true.loc[~all_true.index.duplicated(keep="first")]

        true_targets = pd.Series(
            [(all_true.loc[idx]) for idx in values.index],
            name="true_target",
            index=values.index,
        )

        predicted_targets = pd.Series(
            [
                (np.unique(all_pred.loc[idx]))[
                    (np.unique(all_pred.loc[idx])) != (all_true.loc[idx])
                ]
                for idx in values.index
            ],
            name="predicted_target",
            index=values.index,
        )

        percentage_error = (values / self.df.shape[0] * 100).rename("percentage_error")
        print(
            "Note: percentage_error is per split. Multiply it by n_split "
            + "to have the full percentage."
        )
        return pd.DataFrame(
            [values, percentage_error, true_targets, predicted_targets]
        ).T

    @property
    def score(self) -> float:
        """
        Return a score for optimization.
        By default, it is equal to `1 - mean(matthews_corr_coef)`
        You can specify the score function when creating a new object.
        """
        return 1 - self._mean[self.select_by]

    def percentile(self, p: float) -> pd.Series:
        """
        Return a series containing the values of p-percentile for each metric.
        Percentile has to be in [0..1].
        """
        series = self.df.quantile(p)
        series.index = series.index.map(lambda x: "p%02d_" % (p * 100) + x)
        return series

    @property
    def mean(self) -> pd.Series:
        """
        Return a series containing the mean values of each metric
        """
        series = self._mean.copy()
        series = series.add_prefix("mean_")
        return series

    @property
    def metrics(self):
        """
        Return aggregated metrics:
            - p50 (median) of scores
            - p05 of scores
            - p95 of scores
            - mean score

        If you are not familiar with percentiles, please visit

        Returns
        -------
        dict: the dictionary containing all the metrics.
        """
        _metrics = {}
        _metrics.update({"score": self.score})
        _metrics.update(self.percentile(0.5))
        _metrics.update(self.percentile(0.05))
        _metrics.update(self.percentile(0.95))
        return _metrics

    def plot_confusion_matrix_mean(
        self,
        labels_names: list = None,
        title: str = "Confusion matrices mean",
        cmap: str = None,
        cbar: bool = True,
    ):
        """
        Plot the confusion matrices mean

        Parameters
        ----------
        labels_names : list, optional
            The names of the labels. The default is None.
        title : str, optional
            Figure's title. The default is 'Confusion matrix'.
        cmap : str, optional
            cmap to use. If None, 'Blues' is used. The default is None.
        cbar : bool, optional
            If True, it plots the colorbar too. The default is True.

        Returns
        -------
        None.

        """

        cm = self.confusion_matrix_mean

        if labels_names is None:
            labels_names = np.unique(
                np.concatenate([x.y_true for x in self.history])
            ).tolist()

        if cmap is None:
            cmap = plt.get_cmap("Blues")

        raw_cm = cm
        cm = 100 * cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=100)
        plt.title(title)
        if cbar:
            plt.colorbar()

        if labels_names is not None:
            tick_marks = np.arange(len(labels_names))
            plt.xticks(tick_marks, labels_names, rotation=45)
            plt.yticks(tick_marks, labels_names)

        thresh = cm.max() / 1.5
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                "{:0.2f}%\n({:0.2f}/{:0.2f})".format(
                    cm[i, j], raw_cm[i, j], np.sum(raw_cm[i, :])
                ),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("True label")
        plt.xlabel(
            "Predicted label\nf1_weighted={:0.2f}; mcc={:0.2f}".format(
                self.mean["mean_f1_weighted"], self.mean["mean_matthews_corrcoef"]
            )
        )
        plt.tight_layout()
        plt.show()

    def plot_metrics(self, figtitle="Classification metrics"):
        """
        Plot metrics for each train/val step during cross-validation

        Parameters
        ----------
        figtitle : str, optional
            Figure title. The default is "Classification metrics".

        Returns
        -------
        fig
            The plotly figure

        """
        fig = go.Figure()
        for metric in ["matthews_corrcoef", "accuracy", "f1_weighted", "f1_micro"]:
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df[metric],
                    name=metric,
                )
            )
        fig.update_layout(
            title=figtitle,
            xaxis_title="Trial no",
            yaxis_title="value",
        )
        return fig
