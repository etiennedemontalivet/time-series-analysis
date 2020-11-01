"""
This module defines classes for easy computation of classification results.

"""

from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import mlflow
import itertools


class ClassificationResults:
    """
    ClassificationResults

    All metrics are computed using sklearn metrics from [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)

    Parameters
    ----------
    y_true : pd.Series
        Ground truth (correct) target values.
        
    y_pred : pd.Series
        Estimated targets as returned by a classifier.
        
    labels_name : list, optional
        List of labels to index the matrix. This may be used to reorder or select a subset of labels. 
        If None is given, those that appear at least once in y_true or y_pred are used in sorted order. The default is None.


    Attributes
    ----------
    n_classes_ : int
        number of classes

    classes_ : ndarray of shape (n_classes_,)
        class labels

    accurracy_ : float
        Accuracy classification score.
        In multilabel classification, this function computes subset accuracy: 
        the set of labels predicted for a sample must exactly match the 
        corresponding set of labels in y_true.
        See more details [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)

    matthews_corrcoef_ : float
        number of training samples observed in each class.

    f1_weighted_ : float
        F1 score of the F1 scores of each class for the multiclass task.
        Calculate metrics for each label, and find their average weighted 
        by support (the number of true instances for each label).
        See more details [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)

    f1_micro_ : float
        F1 score of the F1 scores of each class for the multiclass task.
        Calculate metrics globally by counting the total true positives, 
        false negatives and false positives.
        See more details [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)

    tp_ : int
        Number of True Positives.

    fp_ : int
        Number of False Positives = number of misclassifications

    confusion_matrix_ : ndarray of shape (n_classes_, n_classes_)
        Confusion matrix whose i-th row and j-th column entry indicates the 
        number of samples with true label being i-th class and prediced label 
        being j-th class.
    """
    def __init__(
        self, 
        y_true: pd.Series, 
        y_pred: pd.Series, 
        labels_name: list = None):
        """
        Returns a new instance of ClassificationResult based on given y_true and y_pred.

        Parameters
        ----------
        y_true : pd.Series
            Ground truth (correct) target values.
        y_pred : pd.Series
            Estimated targets as returned by a classifier.
        labels_name : list, optional
            List of labels to index the matrix. This may be used to reorder or select a subset of labels. 
            If None is given, those that appear at least once in y_true or y_pred are used in sorted order. The default is None.

        Returns
        -------
        None.

        """
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Compute matthews correlation coefficient
        if np.sum(y_pred) > 0:
            matthews_corr_coef = metrics.matthews_corrcoef(self.y_true, self.y_pred)
        else:
            matthews_corr_coef = 0
            
        self.n_classes_ = len(np.unique(self.y_true))

        # Store it if it is defined, else store 0
        self.matthews_corrcoef_ = matthews_corr_coef if matthews_corr_coef else 0
        
        # Do the same for F1 Score which can sometimes return None
        f1_score_weighted = metrics.f1_score(self.y_true, self.y_pred, average='weighted')
        f1_score_micro = metrics.f1_score(self.y_true, self.y_pred, average='micro')
        # If F1 Score is None we put 0
        self.f1_weighted_ = f1_score_weighted if f1_score_weighted else 0
        self.f1_micro_ = f1_score_micro if f1_score_micro else 0
        
        # Accuracy can never be None so we do not need to check if it is defined
        self.accuracy_ = metrics.accuracy_score(self.y_true, self.y_pred)

        # Confusion matrix
        self.confusion_matrix_ = metrics.confusion_matrix(self.y_true, self.y_pred)
                
        # true positives
        self.tp_ = np.trace(self.confusion_matrix_)
        # false positives 
        self.fp_ = np.sum(self.confusion_matrix_) - self.tp_
        

    @property
    def metrics(self) -> Dict[str, float]:
        """
        Returns a dictionary of all metrics for convenience
        """
        _metrics = {
            "matthews_corrcoef": self.matthews_corrcoef_,
            "accuracy": self.accuracy_,
            "f1_weighted": self.f1_weighted_,
            "f1_micro": self.f1_micro_,
            "tp": self.tp_,
            "fp": self.fp_,
            "confusion_matrix": self.confusion_matrix_
        }
        return _metrics

    @property
    def misclassified(self) -> pd.Index:
        """
        Returns the index of misclassified 
        """
        if isinstance( self.y_true, pd.Series ):
            y_misclassified = self.y_true[self.y_true != self.y_pred]
            misclassified = y_misclassified.index
        else:
            misclassified = np.where( self.y_true != self.y_pred )[0]

        return misclassified
    
    def plot_confusion_matrix(
        self,
        labels_names: list=None,
        title='Confusion matrix',
        cmap=None,
        cbar=True):
        """
        Plot the confusion matrices mean

        Parameters
        ----------
        labels_names : list, optional
            The names of the labels. The default is None.
        title : TYPE, optional
            Figure's title. The default is 'Confusion matrix'.
        cmap : TYPE, optional
            cmap to use. If None, 'Blues' is used. The default is None.
        cbar : TYPE, optional
            If true, it plots the colorbar too. The default is True.

        Returns
        -------
        None.

        """
        cm = self.confusion_matrix_
        
        if labels_names is None:
            labels_names = np.unique(self.y_true).tolist()
        
        if cmap is None:
            cmap = plt.get_cmap('Blues')
    
        raw_cm = cm
        cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
        plt.title(title)
        if cbar:
            plt.colorbar()
    
        if labels_names is not None:
            tick_marks = np.arange(len(labels_names))
            plt.xticks(tick_marks, labels_names, rotation=45)
            plt.yticks(tick_marks, labels_names)
    
    
        thresh = cm.max() / 1.5
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:0.2f}%\n({:d}/{:d})".format(cm[i, j], raw_cm[i,j], np.sum(raw_cm[i,:])),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
    
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\nf1_weighted={:0.2f}; mcc={:0.2f}'.format(self.f1_weighted_,self.matthews_corrcoef_))
        plt.show()

    def log_metrics(self) -> None:
        """
        Log the metrics using MLFlow
        """
        mlflow.log_metrics(self.metrics)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__str__()})"

    def __str__(self) -> str:
        return str(self.metrics)

    def dict(self):
        """
        Returns a dict representation of the results
        """
        data = {}
        data["metrics"] = self.metrics
        data["predictions"] = {"y_true": self.y_true, "y_pred": self.y_pred}
        data["misclassifications"] = self.misclassified
        return data
