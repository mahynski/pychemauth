"""
Unittests for SIMCA_Model.

author: nam
"""
import unittest

from pychemauth.classifier.simca import SIMCA_Model


class TestSIMCA_Model(unittest.TestCase):
    """Test SIMCA_Model class."""

    """def test_sklearn_compatibility(self):
        #Check compatible with sklearn's estimator API.
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(SIMCA_Model(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)"""
