"""
Unittests for DDSIMCA_Model.

author: nam
"""
import unittest

from classifier.simca import DDSIMCA_Model


class TestDDSIMCA_Model(unittest.TestCase):
    """Test DDSIMCA_Model class."""

    def test_sklearn_compatibility(self):
        """Check compatible with sklearn's estimator API."""
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(DDSIMCA_Model(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)
