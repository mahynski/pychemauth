"""
Unittests for sklearn estimator consistency.

author: nam
"""
import copy
import os
import unittest

import numpy as np

from sklearn.utils.estimator_checks import check_estimator


class EstimatorAPICompatibility_Transformers(unittest.TestCase):
    """Check compatibility of transformers with sklearn's API."""

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
        """Test CollinearFeatureSelector"""
        from pychemauth.preprocessing.feature_selection import CollinearFeatureSelector

        check_estimator(CollinearFeatureSelector())

    def test_jensenshannondivergence(self):
        """Test JensenShannonDivergence"""
        from pychemauth.preprocessing.feature_selection import JensenShannonDivergence

        check_estimator(JensenShannonDivergence())
        
    def test_borutashapfeatureselector(self):
        """Test BorutaSHAPFeatureSelector"""
        from pychemauth.preprocessing.feature_selection import BorutaSHAPFeatureSelector

        check_estimator(BorutaSHAPFeatureSelector())

    def test_passthrough(self):
        """Test BorutaSHAPFeatureSelector"""
        from pychemauth.manifold.elliptic import _PassthroughDR

        check_estimator(_PassthroughDR())

class EstimatorAPICompatibility_Classifiers(unittest.TestCase):
    """Check compatibility of classifiers with sklearn's API."""

    def test_ellipticmanifold_authenticator(self):
        """Test EllipticManifold_Authenticator"""
        from pychemauth.manifold.elliptic import EllipticManifold_Authenticator

        check_estimator(EllipticManifold_Authenticator())

    def test_ellipticmanifold_model(self):
        """Test EllipticManifold_Model"""
        from pychemauth.manifold.elliptic import EllipticManifold_Model

        check_estimator(EllipticManifold_Model(0.05))

    def test_pca(self):
        """Test PCA"""
        from pychemauth.classifier.pca import PCA

        check_estimator(PCA())

    def test_plsda(self):
        """Test PLSDA"""
        from pychemauth.classifier.plsda import PLSDA

        check_estimator(PLSDA())

    def test_simca_authenticator(self):
        """Test SIMCA_Authenticator"""
        from pychemauth.classifier.simca import SIMCA_Authenticator

        check_estimator(SIMCA_Authenticator())

    def test_simca_model(self):
        """Test SIMCA_Model"""
        from pychemauth.classifier.simca import SIMCA_Model

        check_estimator(SIMCA_Model(1))

    def test_ddsimca_model(self):
        """Test DDSIMCA_Model"""
        from pychemauth.classifier.simca import DDSIMCA_Model

        check_estimator(DDSIMCA_Model(1))
