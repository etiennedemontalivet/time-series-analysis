"""
Results
=========

The results from a classification are to be analyzed. 
This module aims at easying the analysis of classification results. Two kinds of
classification results are handled:

- single prediciton
- cross-validation 

.. warning::

  It is assumed that inputs are pandas series with ids as indexes. For example, indexes
  could be filenames.

"""

from classification.results import ClassificationResults, CrossValidationResults
