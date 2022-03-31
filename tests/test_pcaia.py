"""
Unittests for PCA_IA.

author: nam
"""
import copy
import os
import unittest

import numpy as np
import pandas as pd

from PyChemAuth.preprocessing.missing import PCA_IA


class TestPCA_IA(unittest.TestCase):
    """Test PCA_IA preprocessing."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        df = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + "/data/pls_train.csv"
        )
        self.raw_X = np.array(df.values[:, 3:], dtype=float)
        self.y = np.array(df["Water"].values, dtype=float).reshape(-1, 1)

        # Randomly delete some entries
        n_delete = 10
        np.random.seed(0)
        a = [
            np.random.randint(low=0, high=self.raw_X.shape[0])
            for i in range(n_delete)
        ]
        b = [
            np.random.randint(low=0, high=self.raw_X.shape[1])
            for i in range(n_delete)
        ]
        self.X = self.raw_X.copy()
        for i, j in zip(a, b):
            self.X[i, j] = np.nan

        df = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + "/data/pls_test.csv",
            header=None,
        )
        raw_X = np.array(df.values[:, 3:], dtype=float)  # Extract features
        self.ytest = np.array(df.values[:, 2], dtype=float)
        n_delete = 5
        np.random.seed(0)
        a = [
            np.random.randint(low=0, high=raw_X.shape[0])
            for i in range(n_delete)
        ]
        b = [
            np.random.randint(low=0, high=raw_X.shape[1])
            for i in range(n_delete)
        ]
        self.Xtest = raw_X.copy()
        for i, j in zip(a, b):
            self.Xtest[i, j] = np.nan

    def test_numerical(self):
        """Check some numerical values."""
        itim = PCA_IA(
            n_components=3,
            scale_x=True,
            missing_values=np.nan,
            tol=1.0e-6,
            max_iters=5000,
        )
        a = itim.fit_transform(self.X, self.y)
        np.testing.assert_almost_equal(
            a[np.isnan(self.X)],
            np.array(
                [
                    -1.60831291,
                    -0.55703074,
                    0.47027347,
                    0.5813853,
                    -1.45768718,
                    0.61868865,
                    1.00003394,
                    0.65206436,
                    1.10381975,
                    -1.53995849,
                ]
            ),
            decimal=6,
        )

    def test_transform(self):
        """Check fit/transform order."""
        itim = PCA_IA(
            n_components=3,
            scale_x=True,
            missing_values=np.nan,
            tol=1.0e-6,
            max_iters=5000,
        )
        a = itim.fit_transform(self.X, self.y)

        _ = itim.fit(self.X, self.y)
        b = itim.transform(self.X)
        np.testing.assert_almost_equal(a, b, decimal=6)

    def test_no_impute(self):
        """Check no imputation necessary."""
        itim = PCA_IA(
            n_components=3,
            scale_x=True,
            missing_values=np.nan,
            tol=1.0e-6,
            max_iters=5000,
        )
        a = itim.fit_transform(self.raw_X, self.y)
        np.testing.assert_almost_equal(self.raw_X, a, decimal=6)

    def test_train_vs_test(self):
        """Check test vs. train."""
        itim = PCA_IA(
            n_components=3,
            scale_x=True,
            missing_values=np.nan,
            tol=1.0e-6,
            max_iters=5000,
        )
        a_before = itim.fit_transform(self.X, self.y)
        b = itim.transform(self.Xtest)
        a_after = itim.transform(self.X, self.y)

        np.testing.assert_almost_equal(a_before, a_after, decimal=6)
        np.testing.assert_almost_equal(
            b[:, 1], np.array([1.35127808, 1.24283012, 1.2688408]), decimal=6
        )
        np.testing.assert_almost_equal(
            b[:, 10], np.array([1.1973849, 1.11997621, 1.08804203]), decimal=6
        )

        np.testing.assert_almost_equal(
            b[np.isnan(self.Xtest)],
            np.array(
                [0.47351088, 0.44755744, -0.34282751, -1.45071451, 0.54397966]
            ),
            decimal=6,
        )
