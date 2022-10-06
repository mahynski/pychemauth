"""
Unittests for PCR.

author: nam
"""
import copy
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from pychemauth.preprocessing.scaling import CorrectedScaler
from pychemauth.regressor.pcr import PCR


class TestPCR(unittest.TestCase):
    """Test PCR class."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        df = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + "/data/pcr_train.csv",
            header=None,
        )
        self.X = np.array(df.values[:, 1:], dtype=float)
        self.y = np.array(df.values[:, 0], dtype=float)
        self.model = PCR(
            n_components=4,
            alpha=0.05,
            gamma=0.01,
            scale_x=False,
            center_y=True,
            scale_y=False,
            robust="semi",
        )
        self.model.fit(self.X, self.y)

    def test_scaling_params(self):
        """Test scaling parameters."""
        self.assertEqual(self.model._PCR__Nh_, 4)
        self.assertEqual(self.model._PCR__Nq_, 4)
        self.assertEqual(self.model._PCR__Nz_, 1)
        np.testing.assert_almost_equal(
            self.model._PCR__h0_, 0.08547084406857604
        )
        np.testing.assert_almost_equal(
            self.model._PCR__q0_, 0.00020476865764116616
        )
        np.testing.assert_almost_equal(
            self.model._PCR__z0_, 0.06657602899076177
        )

    def test_transform(self):
        """Test transformation works."""
        res = self.model.transform(self.X).ravel()[:3]
        ans = np.array([-0.11708791, 0.10775746, -0.00019102])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_h_q(self):
        """Check some h and q values."""
        h, q = self.model.h_q_(self.X)
        ans_h = np.array([0.05916447, 0.15571287, 0.10156415])
        ans_q = np.array([0.00018677, 0.0003949, 0.0001755])
        np.testing.assert_almost_equal(h[:3], ans_h, decimal=6)
        np.testing.assert_almost_equal(q[:3], ans_q, decimal=6)

    def test_f(self):
        """Check some f values."""
        res = self.model.f_(*self.model.h_q_(self.X))[:3]
        print(res)
        ans = np.array([13.12721385, 16.16208378, 4.21930139])
        np.testing.assert_almost_equal(res, ans, decimal=6)


class TestPCR_sklearn(unittest.TestCase):
    """Test PCR class against scikit-learn."""

    """def test_sklearn_compatibility(self):
        #Check compatible with sklearn's estimator API.
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(PCR(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)"""

    def test_compare_pcr(self):
        """Compare PCR to sklearn pipeline version."""
        # Generate some dummy data
        def generate_data(
            n_samples=500, cov=[[3, 3], [3, 4]], seed=0, mean=[0, 0]
        ):
            rng = np.random.RandomState(seed)
            X = rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)

            return X

        X = generate_data(mean=[3, 6])

        def generate_response(X, dimension, mean=[0, 0], seed=1, y_center=0):
            X_pca = copy.copy(X) - np.array(
                mean
            )  # Do mean shift according to what was specified

            # Generate dummy response on a chosen PC
            pca = PCA(n_components=2).fit(X_pca)
            rng = np.random.RandomState(seed)
            y = (
                X_pca.dot(pca.components_[dimension])
                + mean[dimension]
                + rng.normal(size=X.shape[0]) / 2
                + y_center
            )
            return y, pca

        y, pca_gen = generate_response(X, 0, seed=1, mean=[3, 6], y_center=10)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0
        )

        # Compare this manual implementation with a simple scikit-learn pipeline

        # 1. No y centering/scaling
        pcr_pipe = make_pipeline(
            CorrectedScaler(with_mean=True, with_std=False),
            PCA(n_components=1),
            LinearRegression(fit_intercept=True),
        )
        pcr_pipe.fit(X_train, y_train)
        a = pcr_pipe.predict(X_test)

        manual_pcr = PCR(
            n_components=1, scale_x=False, center_y=False, scale_y=False
        )
        _ = manual_pcr.fit(X_train, y_train)
        b = manual_pcr.predict(X_test).ravel()

        self.assertTrue(np.all(np.abs(a - b) < 1.0e-12))

        # 2. With y centering
        scaler = CorrectedScaler(with_mean=True, with_std=False)
        y_alt = scaler.fit_transform(y_train.reshape(-1, 1))
        pcr_pipe = make_pipeline(
            CorrectedScaler(with_mean=True, with_std=False),
            PCA(n_components=1),
            LinearRegression(fit_intercept=True),
        )
        pcr_pipe.fit(X_train, y_alt)
        a = scaler.inverse_transform(pcr_pipe.predict(X_test)).ravel()

        manual_pcr = PCR(
            n_components=1, scale_x=False, center_y=True, scale_y=False
        )
        _ = manual_pcr.fit(X_train, y_train)
        b = manual_pcr.predict(X_test).ravel()

        self.assertTrue(np.all(np.abs(a - b) < 1.0e-12))

        # 3. With y centering and scaling
        scaler = CorrectedScaler(with_mean=True, with_std=True)
        y_alt = scaler.fit_transform(y_train.reshape(-1, 1))
        pcr_pipe = make_pipeline(
            CorrectedScaler(with_mean=True, with_std=False),
            PCA(n_components=1),
            LinearRegression(fit_intercept=True),
        )
        pcr_pipe.fit(X_train, y_alt)
        a = scaler.inverse_transform(pcr_pipe.predict(X_test)).ravel()

        manual_pcr = PCR(
            n_components=1, scale_x=False, center_y=True, scale_y=True
        )
        _ = manual_pcr.fit(X_train, y_train)
        b = manual_pcr.predict(X_test).ravel()

        self.assertTrue(np.all(np.abs(a - b) < 1.0e-12))

        # 4. With y centering and scaling and X scaling
        scaler = CorrectedScaler(with_mean=True, with_std=True)
        y_alt = scaler.fit_transform(y_train.reshape(-1, 1))
        pcr_pipe = make_pipeline(
            CorrectedScaler(with_mean=True, with_std=True),
            PCA(n_components=1),
            LinearRegression(fit_intercept=True),
        )
        pcr_pipe.fit(X_train, y_alt)
        a = scaler.inverse_transform(pcr_pipe.predict(X_test)).ravel()

        manual_pcr = PCR(
            n_components=1, scale_x=True, center_y=True, scale_y=True
        )
        _ = manual_pcr.fit(X_train, y_train)
        b = manual_pcr.predict(X_test).ravel()

        self.assertTrue(np.all(np.abs(a - b) < 1.0e-12))
