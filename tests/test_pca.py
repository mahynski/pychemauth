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

from pychemauth.classifier.pca import PCA


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
                26.63580412,
                38.06455561,
                43.40924329,
                32.37786736,
                47.45184157,
                61.58056371,
                47.14695481,
                39.70333549,
                28.11530628,
                32.63614436,
                49.52339521,
                41.8526048,
                73.12521114,
                49.43440723,
                38.19472763,
                22.9824546,
                35.53256142,
                49.4367446,
                73.35185303,
                34.6331932,
                47.08792864,
                21.18074331,
                38.63421518,
                39.82537437,
                23.50117367,
                23.89016563,
                41.19872047,
                46.72803231,
                39.49456298,
                58.2311268,
                40.52522615,
                40.01773523,
                34.16928944,
                33.18162021,
                56.0053854,
                33.91069375,
                24.18406381,
                30.29501946,
                34.48468692,
                28.40963535,
                43.31093222,
                30.22834451,
                38.22358947,
                62.29776,
                38.8581889,
                67.85611567,
                22.01587686,
                28.1942437,
                40.17052439,
                32.85015713,
                33.14093122,
                37.95184322,
                40.91904398,
                42.56893152,
                35.46612109,
                38.60286356,
                26.80405855,
                57.51987507,
                26.71688095,
                33.96214109,
                31.25116215,
                38.07793418,
                45.85506273,
                27.83560046,
                29.1315257,
                38.9514665,
                39.38719879,
                33.57742978,
                29.13002808,
            ]
        )
        np.testing.assert_almost_equal(d, ans, decimal=6)

        d = self.model.distance(self.X_test)
        ans = np.array(
            [
                46.65917084,
                51.54203214,
                36.09961106,
                55.38448053,
                39.77721218,
                49.04052508,
                32.76520431,
                46.31347217,
                41.17207697,
                37.64871779,
                30.45331204,
                45.02359505,
                27.90318643,
                25.18763689,
                29.08235169,
                39.44679632,
                35.15557396,
                34.50520374,
                24.8581157,
                33.71483452,
                27.34657348,
                40.64021871,
                30.4625607,
                36.62796074,
                45.19449868,
                54.52449047,
                30.71178219,
                43.89495538,
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
        ans = np.array([54.572227758941736, 79.29359439252904])
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
                4.69890585,
                3.11865943,
                7.3856192,
                2.94389185,
                2.90609265,
                2.9984142,
                5.93266629,
                4.43913684,
                3.44305767,
                2.4700274,
                5.63809495,
                4.15339165,
                3.90757742,
                2.77616488,
                3.67196978,
                1.2866092,
                3.41677126,
                5.5719225,
                3.20042685,
                4.84213568,
                0.98472218,
                0.65001782,
                3.23680706,
                3.86279836,
                1.27135551,
                2.06130497,
                2.16065399,
                4.6119927,
                5.1359274,
                5.25584256,
                3.37031549,
                1.91875985,
                1.93845786,
                2.01667835,
                7.04239982,
                4.05586498,
                4.5076014,
                1.39069917,
                2.59678624,
                2.93119496,
                6.96454129,
                3.73189175,
                3.91221882,
                5.70628747,
                4.51302658,
                24.02218944,
                1.82946397,
                2.86152151,
                9.39441693,
                5.87702077,
                2.20366102,
                6.07410308,
                3.47348978,
                3.64971479,
                2.08881608,
                1.68815214,
                4.60228454,
                7.78922548,
                1.39355844,
                4.78472662,
                1.17456706,
                4.58659908,
                7.94472354,
                2.46791878,
                3.74028597,
                3.06719329,
                3.04606586,
                1.73949541,
                1.87112433,
            ]
        )
        np.testing.assert_almost_equal(d, ans, decimal=6)

        d = self.model.distance(self.X_test)
        ans = np.array(
            [
                4.86461855,
                2.39568825,
                2.68600986,
                12.36619664,
                8.98400698,
                5.91205358,
                1.98718071,
                6.07205836,
                3.90813831,
                2.96368384,
                2.92660881,
                3.47635557,
                2.97608843,
                3.04475191,
                2.78225376,
                5.9100393,
                4.07962198,
                0.74249535,
                1.43265286,
                1.64333406,
                2.09150841,
                1.64326701,
                3.19035432,
                6.61003189,
                6.28847278,
                8.06453423,
                1.11425984,
                2.80360516,
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
        ans = np.array([9.48772904, 22.69561408])
        np.testing.assert_almost_equal(res, ans, decimal=6)
