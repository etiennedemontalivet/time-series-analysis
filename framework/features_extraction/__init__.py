"""
Feature extraction module.

This module contans refactorized code dedicated to feature computation as well as all legacy code that was used before.

You will find all feature extractors in:
  - cepstrum_domain.py
  - time_domain.py
  - frequency_domain.py
  - wavelets_domain.py
"""
from framework.features_extraction.cepstrum_domain import CepstrumDomainFeatureExtractor
from framework.features_extraction.time_domain import TimeDomainFeatureExtractor
from framework.features_extraction.wavelets_domain import WaveletsDomainFeatureExtractor
from framework.features_extraction.frequency_domain import (
    FrequencyDomainFeatureExtractor,
)
from framework.features_extraction.extractor import CompleteFeatureExtractor
