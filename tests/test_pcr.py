"""
Unittests for PCR.

author: nam
"""
import copy
import os
import unittest

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from pychemauth.preprocessing.scaling import CorrectedScaler
from pychemauth.regressor.pcr import PCR


class TestPCR(unittest.TestCase):
    """Test PCR class."""

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
