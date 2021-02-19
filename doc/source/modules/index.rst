.. _api_ref:

=============
API Reference
=============

This is the class and function reference of tsanalysis. 

:mod:`tsanalysis.classification`: Classification
================================================

.. automodule:: tsanalysis.classification
   :no-members:
   :no-inherited-members:

.. currentmodule:: tsanalysis

Classes
-------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   classification.ClassificationResults
   classification.CrossValidationResults

:mod:`tsanalysis.datamodels`: DataModels
================================================

.. automodule:: tsanalysis.datamodels
   :no-members:
   :no-inherited-members:

.. currentmodule:: tsanalysis

Classes
-------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   datamodels.FeaturesDataset

Functions
---------
.. autosummary::
   :toctree: generated/
   :template: function.rst

   datamodels.features_concat

:mod:`tsanalysis.datasets`: Datasets
================================================

.. automodule:: tsanalysis.datasets
   :no-members:
   :no-inherited-members:

.. currentmodule:: tsanalysis

Functions
---------
.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.make_iris_data
   datasets.make_classification
   datasets.make_windows_ts_data

:mod:`tsanalysis.features_extraction`: Features Extraction
==========================================================

.. automodule:: tsanalysis.features_extraction
   :no-members:
   :no-inherited-members:

.. currentmodule:: tsanalysis

Functions
---------
.. autosummary::
   :toctree: generated/
   :template: function.rst

   features_extraction.extract_cepd_features
   features_extraction.extract_td_features
   features_extraction.extract_wd_features
   features_extraction.extract_fd_features
   features_extraction.extract_all_features

:mod:`tsanalysis.features_selection`: Features Selection
========================================================

.. automodule:: tsanalysis.features_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: tsanalysis

Functions
---------
.. autosummary::
   :toctree: generated/
   :template: function.rst

   features_selection.brut_force_features_selection
   features_selection.genetic_features_selection
