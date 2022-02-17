"""
Unittests for PLS.

author: nam
"""
import copy
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from classifier.pca import PCA


class TestPCA_Scaled(unittest.TestCase):
    """Test PCA class with scaling used."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        df = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + "/data/pca_train.csv"
        )
        self.X = np.array(df.values[:, 2:], dtype=float)
        df = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + "/data/pca_test.csv",
            header=None,
        )
        self.X_test = np.array(df.values[:, 2:], dtype=float)
        self.model = PCA(n_components=3, alpha=0.05, gamma=0.01, scale_x=True)
        _ = self.model.fit(self.X)

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
        res = self.model.transform(self.X)[:3].ravel()
        ans = np.array(
            [
                [6.91353471, -0.56518687, 0.95539941],
                [4.70333919, 0.70899186, -0.73833557],
                [8.56368316, 0.16753437, 2.44838508],
            ]
        ).ravel()
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_h_q(self):
        """Check some h and q values."""
        h, q = self.model.h_q_(self.X)
        ans_h = np.array([0.07847804, 0.03866518, 0.14480451])
        ans_q = np.array([16.39823482, 24.55049204, 26.46794849])
        np.testing.assert_almost_equal(h[:3], ans_h, decimal=6)
        np.testing.assert_almost_equal(q[:3], ans_q, decimal=6)

    def test_distance(self):
        """Check distances."""
        d = self.model.distance(self.X)
        ans = np.array(
            [
                26.80633683,
                38.31420161,
                43.68579899,
                32.58977472,
                47.75743859,
                61.98356175,
                47.45086191,
                39.9618422,
                28.29575807,
                32.84780065,
                49.84383647,
                42.11886431,
                73.60976492,
                49.76076172,
                38.44562638,
                23.13043372,
                35.76658832,
                49.75717445,
                73.83800702,
                34.86222963,
                47.40044524,
                21.31854092,
                38.89019697,
                40.0873164,
                23.65543939,
                24.04697802,
                41.47048282,
                47.03642974,
                39.75339011,
                58.61370452,
                40.79081675,
                40.28335327,
                34.39375263,
                33.39953629,
                56.36482565,
                34.1339049,
                24.34279869,
                30.49581973,
                34.70835651,
                28.59292231,
                43.59746611,
                30.42881543,
                38.47672709,
                62.70614114,
                39.11429445,
                68.30103268,
                22.16065379,
                28.38072888,
                40.4305351,
                33.06621018,
                33.35975222,
                38.20068674,
                41.18870114,
                42.84983681,
                35.70058611,
                38.85700138,
                26.97781616,
                57.89374839,
                26.8909981,
                34.18537012,
                31.45651176,
                38.32705372,
                46.15469244,
                28.01728273,
                29.32246048,
                39.20806581,
                39.64613882,
                33.79974174,
                29.32262111,
            ]
        )
        np.testing.assert_almost_equal(d, ans, decimal=6)

        d = self.model.distance(self.X_test)
        ans = np.array(
            [
                46.96364376,
                51.88080862,
                36.33651326,
                55.75203074,
                40.03959117,
                49.36479483,
                32.98254988,
                46.61810816,
                41.44415242,
                37.89711645,
                30.65167753,
                45.31729309,
                28.08575664,
                25.35478867,
                29.27473934,
                39.70771182,
                35.38854772,
                34.73384304,
                25.02241245,
                33.93587504,
                27.52740361,
                40.90798492,
                30.66217173,
                36.86744618,
                45.49106343,
                54.88532849,
                30.9133523,
                44.18572496,
            ]
        )
        np.testing.assert_almost_equal(d, ans, decimal=6)

    def test_predict(self):
        """Check some predictions on regular data."""
        res = self.model.predict(self.X)
        ans = np.array(
            [
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            dtype=bool,
        )
        self.assertTrue(np.all(res == ans))

    def test_outliers(self):
        """Check some predictions on extremes and outliers."""
        ext, out = self.model.check_outliers(self.X)
        ans = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            dtype=bool,
        )
        self.assertTrue(np.all(ext == ans))
        ans = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            dtype=bool,
        )
        self.assertTrue(np.all(out == ans))

    def test_c_out(self):
        """Check critical distances for X."""
        res = np.array([self.model._PCA__c_crit_, self.model._PCA__c_out_])
        ans = np.array([54.87550582151137, 79.65318305890995])
        np.testing.assert_almost_equal(res, ans, decimal=6)


class TestPCA_Unscaled(unittest.TestCase):
    """Test PCA class without scaling."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        df = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + "/data/pca_train.csv"
        )
        self.X = np.array(df.values[:, 2:], dtype=float)
        df = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + "/data/pca_test.csv",
            header=None,
        )
        self.X_test = np.array(df.values[:, 2:], dtype=float)
        self.model = PCA(n_components=3, alpha=0.05, gamma=0.01, scale_x=False)
        _ = self.model.fit(self.X)

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
        res = self.model.transform(self.X)[:3].ravel()
        ans = np.array(
            [
                [634.98394834, -97.12147939, -3.66869868],
                [524.78382944, -9.74799749, 5.18016868],
                [622.29974648, -136.69994973, 6.45664819],
            ]
        ).ravel()
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_h_q(self):
        """Check some h and q values."""
        h, q = self.model.h_q_(self.X)
        ans_h = np.array([0.08629, 0.04169701, 0.12818276])
        ans_q = np.array([11.27227657, 18.55001329, 23.00931321])
        np.testing.assert_almost_equal(h[:3], ans_h, decimal=6)
        np.testing.assert_almost_equal(q[:3], ans_q, decimal=6)

    def test_distance(self):
        """Check distances."""
        d = self.model.distance(self.X)
        ans = np.array(
            [
                3.76794144,
                2.64095168,
                5.98936894,
                2.46829811,
                2.33809766,
                2.74636703,
                4.7574767,
                3.58529008,
                2.94520324,
                2.10089442,
                5.18041055,
                3.35367486,
                3.60482453,
                2.41734101,
                2.97840978,
                1.19274989,
                3.05347667,
                5.21135151,
                3.0003599,
                3.8702014,
                0.84067934,
                0.6022872,
                2.90231566,
                3.67076899,
                1.16417912,
                1.95030994,
                1.92412782,
                4.16792125,
                4.36771463,
                4.59884921,
                2.80007702,
                1.54844949,
                1.7255777,
                1.68974707,
                6.49347857,
                3.28914335,
                3.85923973,
                1.12871838,
                2.23873191,
                2.4435445,
                5.84080983,
                3.01971612,
                3.43918183,
                4.62541553,
                3.69422962,
                22.75478832,
                1.60159681,
                2.45618975,
                8.79858982,
                4.65207686,
                2.11343695,
                5.37223592,
                3.20385521,
                3.21688036,
                1.92769769,
                1.50348807,
                3.99621871,
                6.37649581,
                1.16736905,
                3.83575843,
                1.05964083,
                4.12350091,
                7.06183103,
                2.23874387,
                3.08469414,
                2.60483278,
                2.45908106,
                1.56053048,
                1.53759813,
            ]
        )
        np.testing.assert_almost_equal(d, ans, decimal=6)

        d = self.model.distance(self.X_test)
        ans = np.array(
            [
                3.92364852,
                2.00308283,
                2.31137334,
                11.48610499,
                7.82033288,
                5.00193883,
                1.78880534,
                5.22594256,
                3.29642122,
                2.65532502,
                2.56790401,
                2.91678224,
                2.67756356,
                2.75960729,
                2.50006177,
                5.10669449,
                3.73939479,
                0.70030462,
                1.19392802,
                1.50946586,
                1.93076733,
                1.53048956,
                2.76649744,
                5.95633456,
                5.67722464,
                7.18590764,
                0.94706285,
                2.65465154,
            ]
        )
        np.testing.assert_almost_equal(d, ans, decimal=6)

    def test_predict(self):
        """Check some predictions on regular data."""
        res = self.model.predict(self.X)
        ans = np.array(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            dtype=bool,
        )
        self.assertTrue(np.all(res == ans))

    def test_outliers(self):
        """Check some predictions on extremes and outliers."""
        ext, out = self.model.check_outliers(self.X)
        ans = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            dtype=bool,
        )
        self.assertTrue(np.all(ext == ans))
        ans = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            dtype=bool,
        )
        self.assertTrue(np.all(out == ans))

    def test_c_out(self):
        """Check critical distances for X."""
        res = np.array([self.model._PCA__c_crit_, self.model._PCA__c_out_])
        ans = np.array([8.627199537124385, 21.480315700647875])
        np.testing.assert_almost_equal(res, ans, decimal=6)
