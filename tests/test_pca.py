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
        self.model = PCA(
            n_components=3, alpha=0.05, gamma=0.01, scale_x=True, robust="semi"
        )
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
        ans_h = np.array([0.07847804, 0.03866518, 0.14480451])*(self.X.shape[0]-1)
        ans_q = np.array([16.39823482, 24.55049204, 26.46794849])
        np.testing.assert_almost_equal(h[:3], ans_h, decimal=6)
        np.testing.assert_almost_equal(q[:3], ans_q, decimal=6)

    def test_distance(self):
        """Check distances."""
        d = self.model.distance(self.X)
        ans = np.array(
            [
                31.82341576,
                41.83228123,
                52.70241445,
                35.85423906,
                55.5940998,
                68.21432676,
                55.06903134,
                44.79118036,
                33.31696359,
                37.33143474,
                57.09886583,
                51.04507137,
                77.31835287,
                53.01488267,
                41.73051566,
                26.94525255,
                38.44411439,
                56.66195323,
                77.49762074,
                36.89859287,
                49.48423524,
                23.96204236,
                40.86327034,
                43.3086212,
                25.7454579,
                26.17573442,
                44.31893236,
                50.16783113,
                43.52504164,
                63.58423243,
                44.65534857,
                42.03864122,
                37.32795999,
                36.28516323,
                66.3779305,
                36.77137886,
                26.5018163,
                31.9986692,
                39.43065491,
                33.08526765,
                46.07659609,
                31.85917177,
                40.50470377,
                68.58568288,
                41.93475589,
                74.64424716,
                23.9582095,
                30.01953812,
                46.26179269,
                35.73016038,
                35.52149937,
                41.74722458,
                44.17780511,
                45.72879875,
                37.83492351,
                41.83403831,
                30.7067163,
                65.28113913,
                30.03950959,
                37.02397345,
                34.10569203,
                42.22383443,
                51.07511196,
                31.12906715,
                32.09113754,
                42.10966483,
                42.90575537,
                35.6157772,
                31.0660362,
            ]
        )
        np.testing.assert_almost_equal(d, ans, decimal=6)

        d = self.model.distance(self.X_test)
        ans = np.array(
            [
                52.22296464,
                56.19062732,
                39.58493506,
                58.22113285,
                42.79378619,
                52.27613685,
                34.50191672,
                50.35189974,
                43.99033119,
                40.4672676,
                34.30347315,
                50.45492937,
                30.93050565,
                26.47886604,
                30.94780793,
                41.99900489,
                37.15897016,
                36.48656994,
                26.5424538,
                37.09979825,
                29.14662847,
                43.90959441,
                33.58664097,
                40.70670589,
                49.57158301,
                57.93372021,
                33.66139445,
                46.46970487,
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
        ans = np.array([59.30351203, 84.88690268])
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
        ans_h = np.array([0.08629, 0.04169701, 0.12818276])*(self.X.shape[0]-1)
        ans_q = np.array([11.27227657, 18.55001329, 23.00931321])
        np.testing.assert_almost_equal(h[:3], ans_h, decimal=6)
        np.testing.assert_almost_equal(q[:3], ans_q, decimal=6)

    def test_distance(self):
        """Check distances."""
        d = self.model.distance(self.X)
        ans = np.array(
            [
                7.98108314,
                6.138481,
                12.94675015,
                5.64647432,
                4.98264008,
                7.1458563,
                10.0778954,
                7.69384837,
                6.95434529,
                4.91711505,
                13.53435076,
                7.19360596,
                9.46697201,
                5.86306911,
                6.44059759,
                3.15206361,
                7.68618641,
                13.92622045,
                8.04126502,
                8.14876615,
                1.97902131,
                1.59061025,
                7.3393857,
                10.00232231,
                3.0280724,
                5.28637965,
                4.81972862,
                10.65287161,
                10.22010636,
                11.23404044,
                6.30976376,
                3.31808699,
                4.32001167,
                3.86125884,
                17.04204042,
                7.11003387,
                9.12507139,
                2.44343723,
                5.34967574,
                5.53743251,
                13.36666144,
                6.50189328,
                8.45817432,
                9.99033556,
                8.11762949,
                61.76360763,
                3.9152914,
                5.83038053,
                23.55266578,
                9.61853025,
                5.82216434,
                13.32765412,
                8.41221243,
                7.94144303,
                5.06494041,
                3.7665531,
                9.65239748,
                14.01336712,
                2.6665471,
                8.12084272,
                2.70204333,
                10.46532364,
                17.64298547,
                5.75118238,
                6.86580204,
                6.08198437,
                5.27286301,
                3.94903459,
                3.40133656,
            ]
        )
        np.testing.assert_almost_equal(d, ans, decimal=6)

        d = self.model.distance(self.X_test)
        ans = np.array(
            [
                8.39956838,
                4.56153403,
                5.50779581,
                30.42820838,
                18.95873845,
                11.60959175,
                4.54778543,
                12.45583679,
                7.61390547,
                6.70752724,
                6.29823861,
                6.6800101,
                6.80238183,
                7.08098998,
                6.3406675,
                12.24470594,
                9.73885029,
                1.89092402,
                2.70419541,
                3.94204999,
                5.07498032,
                4.06846239,
                6.6687851,
                15.16447846,
                14.49069322,
                18.01458488,
                2.21411002,
                7.20214528,
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
                False,
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
                True,
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
        ans = np.array([14.06714045, 28.98425643])
        np.testing.assert_almost_equal(res, ans, decimal=6)
