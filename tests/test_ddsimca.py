"""
Unittests for DDSIMCA.

author: nam
"""
import unittest

from classifier.simca import DDSIMCA


class TestDDSIMCA(unittest.TestCase):
    """Test DDSIMCA class."""

    def test_sklearn_compatibility(self):
        """Check compatible with sklearn's estimator API."""
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(DDSIMCA(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)
