"""
This module defines the complete Feature Extractor class that is responsible for
feature extraction in Smart Glove project.
"""
from framework.features_extraction.base import FeatureExtractor
from framework.features_extraction.cepstrum_domain import CepstrumDomainFeatureExtractor
from framework.features_extraction.time_domain import TimeDomainFeatureExtractor
from framework.features_extraction.wavelets_domain import WaveletsDomainFeatureExtractor
from framework.features_extraction.frequency_domain import (
    FrequencyDomainFeatureExtractor,
)


class CompleteFeatureExtractor(FeatureExtractor):
    """
    Extract all features from:
      - TimeDomainFeatureExtractor
      - FrequencyDomainFeatureExtractor
      - WaveletsDomainFeatureExtractor
      - CepstrumDomainFeatureExtractor
    """

    funcs = (
        TimeDomainFeatureExtractor.funcs
        + FrequencyDomainFeatureExtractor.funcs
        + WaveletsDomainFeatureExtractor.funcs
        + CepstrumDomainFeatureExtractor.funcs
    )
