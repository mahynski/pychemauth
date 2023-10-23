"""
Unittests for sklearn estimator consistency.

author: nam
"""
import unittest

from sklearn.utils.estimator_checks import check_estimator

from pychemauth.classifier.pca import PCA
from pychemauth.classifier.plsda import PLSDA
from pychemauth.classifier.simca import (
    DDSIMCA_Model,
    SIMCA_Authenticator,
    SIMCA_Model,
)
from pychemauth.manifold.elliptic import (
    EllipticManifold_Authenticator,
    EllipticManifold_Model,
)
from pychemauth.preprocessing.feature_selection import (
    BorutaSHAPFeatureSelector,
    CollinearFeatureSelector,
    JensenShannonDivergence,
)
from pychemauth.preprocessing.filter import MSC, SNV, SavGol
from pychemauth.preprocessing.missing import LOD, PCA_IA, PLS_IA
from pychemauth.preprocessing.scaling import CorrectedScaler, RobustScaler
from pychemauth.regressor.pcr import PCR
from pychemauth.regressor.pls import PLS


class EstimatorAPICompatibility_Transformers(unittest.TestCase):
    """Check compatibility of transformers with sklearn's API."""

    def test_robustscaler(self):
        """Test RobustScaler."""
        check_estimator(RobustScaler())

    def test_correctedscaler(self):
        """Test CorrectedScaler."""
        check_estimator(CorrectedScaler())

    def test_lod(self):
        """Test LOD."""
        check_estimator(LOD())

    def test_pcaia(self):
        """Test PCA_IA."""
        check_estimator(PCA_IA())

    def test_plsia(self):
        """Test PLS_IA."""
        check_estimator(PLS_IA())

    def test_msc(self):
        """Test MSC."""
        check_estimator(MSC())

    def test_snv(self):
        """Test SNV."""
        check_estimator(SNV())

    def test_savgol(self):
        """Test SavGol."""
        check_estimator(SavGol(window_length=2, polyorder=1))

    def test_collinearfeatureselector(self):
        """Test CollinearFeatureSelector."""
        check_estimator(CollinearFeatureSelector())

    def test_jensenshannondivergence(self):
        """Test JensenShannonDivergence."""
        check_estimator(JensenShannonDivergence())

    def test_borutashapfeatureselector(self):
        """Test BorutaSHAPFeatureSelector."""
        check_estimator(BorutaSHAPFeatureSelector())


class EstimatorAPICompatibility_Classifiers(unittest.TestCase):
    """Check compatibility of classifiers with sklearn's API."""

    def test_ellipticmanifold_authenticator(self):
        """Test EllipticManifold_Authenticator."""
        check_estimator(EllipticManifold_Authenticator())

    def test_ellipticmanifold_model(self):
        """Test EllipticManifold_Model."""
        check_estimator(EllipticManifold_Model(0.05))

    def test_pca(self):
        """Test PCA."""
        check_estimator(PCA())

    def test_plsda(self):
        """Test PLSDA."""
        check_estimator(PLSDA())

    def test_simca_authenticator(self):
        """Test SIMCA_Authenticator."""
        check_estimator(SIMCA_Authenticator())

    def test_simca_model(self):
        """Test SIMCA_Model."""
        check_estimator(SIMCA_Model(1))

    def test_ddsimca_model(self):
        """Test DDSIMCA_Model."""
        check_estimator(DDSIMCA_Model(1))


class EstimatorAPICompatibility_Regressors(unittest.TestCase):
    """Check compatibility of regressors with sklearn's API."""

    def test_pcr(self):
        """Test PCR."""
        check_estimator(PCR())

    def test_pls(self):
        """Test PLS."""
        check_estimator(PLS())
