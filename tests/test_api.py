"""
Unittests for sklearn estimator consistency.

author: nam
"""
import copy
import os
import unittest

import numpy as np

from sklearn.utils.estimator_checks import check_estimator


class EstimatorAPICompatibility(unittest.TestCase):
    """Check compatibility of transformers and estimators with sklearn's API."""

    @classmethod
    def setUpClass(self):
        """Basic setup."""
        pass

    def test_robustscaler(self):
        """Test RobustScaler"""
        from pychemauth.preprocessing.scaling import RobustScaler

        check_estimator(RobustScaler())

    def test_correctedscaler(self):
        """Test CorrectedScaler"""
        from pychemauth.preprocessing.scaling import CorrectedScaler

        check_estimator(CorrectedScaler())

    def test_lod(self):
        """Test LOD"""
        from pychemauth.preprocessing.missing import LOD

        check_estimator(LOD())

    def test_pcaia(self):
        """Test PCA_IA"""
        from pychemauth.preprocessing.missing import PCA_IA

        check_estimator(PCA_IA())

    def test_plsia(self):
        """Test PLS_IA"""
        from pychemauth.preprocessing.missing import PLS_IA

        check_estimator(PLS_IA())

    def test_msc(self):
        """Test MSC"""
        from pychemauth.preprocessing.filter import MSC

        check_estimator(MSC())

    def test_snv(self):
        """Test SNV"""
        from pychemauth.preprocessing.filter import SNV

        check_estimator(SNV())

    def test_savgol(self):
        """Test SavGol"""
        from pychemauth.preprocessing.filter import SavGol

        check_estimator(SavGol(window_length=2, polyorder=1))

    def test_collinearfeatureselector(self):
        """Test SavGol"""
        from pychemauth.preprocessing.feature_selection import CollinearFeatureSelector

        check_estimator(CollinearFeatureSelector())
        