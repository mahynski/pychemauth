"""
Unittests for PLS.

author: nam
"""
import copy
import os
import unittest

import numpy as np
import pandas as pd

from regressor.pls import PLS


class TestPLS_Scaled(unittest.TestCase):
    """Test PLS class with scaling used."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        df = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + "/data/pls_train.csv"
        )
        self.X = np.array(df.values[:, 3:], dtype=float)
        self.y = np.array(df["Water"].values, dtype=float)
        self.model = PLS(n_components=6, alpha=0.05, gamma=0.01, scale_x=True)
        self.model.fit(self.X, self.y)

    """def test_sklearn_compatibility(self):
        #Check compatible with sklearn's estimator API.
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(PLS(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)"""

    def test_transform(self):
        """Check a few x-scores."""
        res = self.model.transform(self.X).ravel()[:3]
        ans = np.array([-11.77911937, -7.40108219, -0.98177486])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_h_q(self):
        """Check some h and q values."""
        h, q = self.model.h_q_(self.X)
        ans_h = np.array([0.08272128, 0.12718815, 0.0304968])
        ans_q = np.array([0.44912905, 0.31931337, 0.10831947])
        np.testing.assert_almost_equal(h[:3], ans_h, decimal=6)
        np.testing.assert_almost_equal(q[:3], ans_q, decimal=6)

    def test_f(self):
        """Check some f values."""
        res = self.model.f_(*self.model.h_q_(self.X))[:3]
        ans = np.array([9.15146807, 10.06191892, 2.75553153])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_z(self):
        """Check some z values."""
        res = self.model.z_(self.X, self.y)[:3]
        ans = np.array([1.48329627e-03, 3.39669025e-03, 1.06630496e-01])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_g(self):
        """Check some g values."""
        res = self.model.g_(self.X, self.y)[:3]
        ans = np.array([9.18612277, 10.14127682, 5.2467719])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_predict(self):
        """Check some predictions."""
        res = self.model.predict(self.X).ravel()[:3]
        ans = np.array([13.13851359, 13.15828113, 12.82654325])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_x_out(self):
        """Check critical distances for X."""
        res = np.array([self.model._PLS__x_crit_, self.model._PLS__x_out_])
        ans = np.array([9.48772904, 23.5870557])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_xy_out(self):
        """Check critical distances for XY."""
        res = np.array([self.model._PLS__xy_crit_, self.model._PLS__xy_out_])
        ans = np.array([11.07049769, 25.82159626])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_score(self):
        """Check score."""
        np.testing.assert_almost_equal(
            self.model.score(self.X, self.y), 0.924088794825304
        )


class TestPLS_Unscaled(unittest.TestCase):
    """Test PLS class without scaling used."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        df = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + "/data/pls_train.csv"
        )
        self.X = np.array(df.values[:, 3:], dtype=float)
        self.y = np.array(df["Water"].values, dtype=float)
        self.model = PLS(n_components=6, alpha=0.05, gamma=0.01, scale_x=False)
        self.model.fit(self.X, self.y)

    """def test_sklearn_compatibility(self):
        #Check compatible with sklearn's estimator API.
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(PLS(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)"""

    def test_transform(self):
        """Check a few x-scores."""
        res = self.model.transform(self.X).ravel()[:3]
        ans = np.array([0.55028919, -0.1011258, -0.03846212])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_h_q(self):
        """Check some h and q values."""
        h, q = self.model.h_q_(self.X)
        ans_h = np.array([0.13355759, 0.13201587, 0.03812691])
        ans_q = np.array([5.08900958e-06, 3.49069667e-04, 2.25974209e-05])
        np.testing.assert_almost_equal(h[:3], ans_h, decimal=6)
        np.testing.assert_almost_equal(q[:3], ans_q, decimal=6)

    def test_f(self):
        """Check some f values."""
        res = self.model.f_(*self.model.h_q_(self.X))[:3]
        ans = np.array([11.61999155, 14.52809849, 3.50415465])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_z(self):
        """Check some z values."""
        res = self.model.z_(self.X, self.y)[:3]
        ans = np.array([3.25158186e-04, 4.34807900e-09, 9.94443229e-02])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_g(self):
        """Check some g values."""
        res = self.model.g_(self.X, self.y)[:3]
        ans = np.array([11.62761631, 14.52809859, 5.83606502])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_predict(self):
        """Check some predictions."""
        res = self.model.predict(self.X).ravel()[:3]
        ans = np.array([13.11803214, 13.09993406, 12.81534794])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_x_out(self):
        """Check critical distances for X."""
        res = np.array([self.model._PLS__x_crit_, self.model._PLS__x_out_])
        ans = np.array([12.59158724, 27.93536325])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_xy_out(self):
        """Check critical distances for XY."""
        res = np.array([self.model._PLS__xy_crit_, self.model._PLS__xy_out_])
        ans = np.array([14.06714045, 29.95863241])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_score(self):
        """Check score."""
        np.testing.assert_almost_equal(
            self.model.score(self.X, self.y), 0.9243675391888061
        )
