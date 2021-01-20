"""
Unittests for PLSDA.

author: nam
"""
import unittest

from classifier.plsda import PLSDA


class TestPLSDA(unittest.TestCase):
    """Test PLSDA class."""

    def test_sklearn_compatibility(self):
        """Check compatible with sklearn's estimator API."""
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(PLSDA())
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)
