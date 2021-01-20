"""
Unittests for SIMCA.

author: nam
"""
import unittest

from classifier.simca import SIMCA


class TestSIMCA(unittest.TestCase):
    """Test SIMCA class."""

    def test_sklearn_compatibility(self):
        """Check compatible with sklearn's estimator API."""
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(SIMCA(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)
