"""
Unittests for PLSDA.

author: nam
"""
import os
import unittest

import numpy as np
import pandas as pd

from classifier.plsda import PLSDA


class TestPLSDA(unittest.TestCase):
    """Test PLSDA class."""

    def test_sklearn_compatibility(self):
        """Check compatible with sklearn's estimator API."""
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(PLSDA(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)

    def test_plsda3_hard(self):
        """Test PLSDA on a 3-class example with hard decision boundaries."""
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__)) + "/plsda3_train.csv"
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(df["Class"].values, dtype=str)
        plsda = PLSDA(
            n_components=5,
            alpha=0.05,
            gamma=0.01,
            not_assigned="UNKNOWN",
            style="hard",
            scale_x=True,
        )
        _ = plsda.fit(raw_x, raw_y)

        # Check class centers
        class_centers = np.array(
            [
                [0.93478193, -0.2300635],
                [0.39786506, 1.07826378],
                [-0.46672116, -0.04088351],
            ]
        )
        err = np.all(
            np.abs((plsda._PLSDA__class_centers_ - class_centers)) < 1.0e-6
        )
        self.assertTrue(err)

        # Check X is centered and, in this case, scaled

        # Check Y is centered

        # Check a few transform() values numerically

        # Check a few predictions manually

        # Check some distances from projection to class centers

        # Check FOM on test and train

        # Check some outliers

    def test_plsda3_soft(self):
        """Test PLSDA on a 3-class example with soft decision boundaries."""
        return

    def test_plsda2_hard(self):
        """Test PLSDA on a 2-class example with hard decision boundaries."""
        return

    def test_plsda2_soft(self):
        """Test PLSDA on a 2-class example with soft decision boundaries."""
        return
