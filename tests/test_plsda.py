"""
Unittests for PLSDA.

author: nam
"""
import os
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from pychemauth.classifier.plsda import PLSDA


class TestPLSDA(unittest.TestCase):
    """Test PLSDA class."""

    """def test_sklearn_compatibility(self):
        #Check compatible with sklearn's estimator API.
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(PLSDA(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)"""

    def test_plsda3_hard(self):
        """Test PLSDA on a 3-class example with hard decision boundaries."""
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/plsda3_train.csv"
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
            score_metric="TEFF",
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
        err = np.all(
            np.abs(
                (raw_x - np.mean(raw_x, axis=0)) / np.std(raw_x, ddof=1, axis=0)
                - plsda._PLSDA__X_
            )
            < 1.0e-6
        )
        self.assertTrue(err)

        # Check Y is centered
        err = np.all(np.abs(np.mean(plsda._PLSDA__y_, axis=0) < 1.0e-6))
        self.assertTrue(err)

        # Check a few transform() values (sPC scores) numerically
        xform = np.array(
            [
                [0.26448772, 0.20920553],
                [0.45877592, 0.43243051],
                [-0.41316007, -0.03986439],
            ]
        )
        err = np.all(
            np.abs(
                (plsda.transform([raw_x[0], raw_x[100], raw_x[150]]) - xform)
            )
            < 1.0e-6
        )
        self.assertTrue(err)

        # Check Mahalanobis distance
        d_plsda = plsda.mahalanobis(
            [raw_x[0], raw_x[100], raw_x[150]]
        )  # Actually d^2
        d_scipy = []
        for t in plsda.transform([raw_x[0], raw_x[100], raw_x[150]]):
            d_ = []
            for i in range(len(plsda._PLSDA__ohencoder_.categories_[0])):
                d_.append(
                    mahalanobis(
                        t,
                        plsda._PLSDA__class_centers_[i],
                        np.linalg.inv(plsda._PLSDA__L_),
                    )
                )
            d_scipy.append(d_)
        d_scipy = np.array(d_scipy)
        self.assertTrue(np.allclose(d_scipy, d_plsda**0.5))

        # Check a few predictions manually
        pred = plsda.predict([raw_x[1], raw_x[125], raw_x[150]])
        self.assertEqual(pred[0], "JPN1")
        self.assertEqual(pred[1], "PHI")
        self.assertEqual(pred[2], "THA1")

        # Check some distances from projection to class centers
        distances = (345 - 1) * np.array(
            [
                [
                    0.004397761999084647,
                    0.04202058409880349,
                    0.01440208288356671,
                ],
                [
                    0.03618876285897883,
                    0.007372767321343739,
                    0.025478671906526282,
                ],
                [
                    0.01694130534356488,
                    0.05756039394563304,
                    2.4417495469312195e-05,
                ],
            ]
        )
        d = plsda.mahalanobis(raw_x)
        d = np.array([d[1], d[125], d[150]])
        err = np.all(np.abs((d - distances)) < 1.0e-12)
        self.assertTrue(err)

        # Check FOM on test and train
        fom = plsda.figures_of_merit(plsda.predict(raw_x), raw_y)
        x = np.array([[87, 0, 10, 0], [2, 24, 3, 0], [0, 0, 219, 0]])
        err = np.all((fom["CM"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([97, 29, 219])
        err = np.all((fom["I"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([0.89690722, 0.82758621, 1.0])
        err = np.all(np.abs((fom["CSNS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.99193548, 1.0, 0.8968254])
        err = np.all(np.abs((fom["CSPS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.94322537, 0.90971765, 0.94700866])
        err = np.all(np.abs((fom["CEFF"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(fom["TSNS"] - 0.9565217391304348) < 1.0e-12)
        self.assertTrue(np.abs(fom["TSPS"] - 0.9565217391304348) < 1.0e-12)
        self.assertTrue(np.abs(fom["TEFF"] - 0.9565217391304348) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y) - 0.9565217391304348) < 1.0e-12
        )

        # Check test set
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/plsda3_test.csv",
            header=None,
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(["THA2"] * len(raw_x), dtype=str)
        fom = plsda.figures_of_merit(plsda.predict(raw_x), raw_y)

        x = np.array(
            [
                [0, 74, 58, 0],
            ]
        )
        err = np.all((fom["CM"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([0, 0, 0, 132])
        err = np.all((fom["I"].values - x) == 0)
        self.assertTrue(err)

        self.assertTrue(np.all([np.isnan(v) for v in fom["CSNS"].values]))

        x = np.array([1.0, 0.43939394, 0.56060606])
        err = np.all(np.abs((fom["CSPS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 0.43939394, 0.56060606])
        err = np.all(np.abs((fom["CEFF"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.isnan(fom["TSNS"]))
        self.assertTrue(np.abs(fom["TSPS"] - 0.0) < 1.0e-12)
        self.assertTrue(np.abs(fom["TEFF"] - fom["TSPS"]) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y) - fom["TEFF"]) < 1.0e-12
        )

    def test_plsda3_soft(self):
        """Test PLSDA on a 3-class example with soft decision boundaries."""
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/plsda3_train.csv"
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(df["Class"].values, dtype=str)
        plsda = PLSDA(
            n_components=5,
            alpha=0.05,
            gamma=0.01,
            not_assigned="UNKNOWN",
            style="soft",
            scale_x=True,
            score_metric="TEFF",
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
        err = np.all(
            np.abs(
                (raw_x - np.mean(raw_x, axis=0)) / np.std(raw_x, ddof=1, axis=0)
                - plsda._PLSDA__X_
            )
            < 1.0e-6
        )
        self.assertTrue(err)

        # Check Y is centered
        err = np.all(np.abs(np.mean(plsda._PLSDA__y_, axis=0) < 1.0e-6))
        self.assertTrue(err)

        # Check a few transform() values (sPC scores) numerically
        xform = np.array(
            [
                [0.26448772, 0.20920553],
                [0.45877592, 0.43243051],
                [-0.41316007, -0.03986439],
            ]
        )
        err = np.all(
            np.abs(
                (plsda.transform([raw_x[0], raw_x[100], raw_x[150]]) - xform)
            )
            < 1.0e-6
        )
        self.assertTrue(err)

        # Check Mahalanobis distance
        d_plsda = plsda.mahalanobis(
            [raw_x[0], raw_x[100], raw_x[150]]
        )  # Actually d^2
        d_scipy = []
        for t in plsda.transform([raw_x[0], raw_x[100], raw_x[150]]):
            d_ = []
            for i in range(len(plsda._PLSDA__ohencoder_.categories_[0])):
                d_.append(
                    mahalanobis(
                        t,
                        plsda._PLSDA__class_centers_[i],
                        np.linalg.inv(plsda._PLSDA__S_[i]),
                    )
                )
            d_scipy.append(d_)
        d_scipy = np.array(d_scipy)
        self.assertTrue(np.allclose(d_scipy, d_plsda**0.5))

        # Check a few predictions manually
        pred = plsda.predict([raw_x[1], raw_x[125], raw_x[150]])
        self.assertTrue(
            np.all([a == b for a, b in zip(pred[0], ["JPN1", "PHI"])])
        )
        self.assertTrue(np.all([a == b for a, b in zip(pred[1], ["PHI"])]))
        self.assertTrue(np.all([a == b for a, b in zip(pred[2], ["THA1"])]))

        # Check some distances from projection to class centers
        distances = np.array(
            [
                [2.53812605, 5.5879958, 37.83387996],
                [17.53237248, 1.09215772, 37.95329066],
                [28.79218466, 18.80333959, 0.06600758],
            ]
        )
        d = plsda.mahalanobis(raw_x)
        d = np.array([d[1], d[125], d[150]])
        err = np.all(np.abs((d - distances)) < 1.0e-6)
        self.assertTrue(err)

        # Check FOM on test and train
        fom = plsda.figures_of_merit(plsda.predict(raw_x), raw_y)
        x = np.array([[91, 17, 0, 2], [0, 27, 0, 2], [2, 2, 203, 13]])
        err = np.all((fom["CM"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([97, 29, 219])
        err = np.all((fom["I"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([0.93814433, 0.93103448, 0.92694064])
        err = np.all(np.abs((fom["CSNS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.991935, 0.939873, 1.0])
        err = np.all(np.abs((fom["CSPS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.96466505, 0.935444, 0.96277756])
        err = np.all(np.abs((fom["CEFF"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(fom["TSNS"] - 0.9304347826086956) < 1.0e-12)
        self.assertTrue(np.abs(fom["TSPS"] - 0.9695652173913043) < 1.0e-12)
        self.assertTrue(
            np.abs(
                fom["TEFF"] - np.sqrt(0.9304347826086956 * 0.9695652173913043)
            )
            < 1.0e-12
        )

        # Check score
        self.assertTrue(
            np.abs(
                plsda.score(raw_x, raw_y)
                - np.sqrt(0.9304347826086956 * 0.9695652173913043)
            )
            < 1.0e-12
        )

        # Check no outliers in this example
        self.assertTrue(np.all(~plsda.check_outliers()))

        # Check test set
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/plsda3_test.csv",
            header=None,
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(["THA2"] * len(raw_x), dtype=str)
        fom = plsda.figures_of_merit(plsda.predict(raw_x), raw_y)

        x = np.array(
            [
                [0, 0, 2, 130],
            ]
        )
        err = np.all((fom["CM"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([0, 0, 0, 132])
        err = np.all((fom["I"].values - x) == 0)
        self.assertTrue(err)

        self.assertTrue(np.all([np.isnan(v) for v in fom["CSNS"].values]))

        x = np.array([1.0, 1.0, 0.98484848])
        err = np.all(np.abs((fom["CSPS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 1.0, 0.98484848])
        err = np.all(np.abs((fom["CEFF"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.isnan(fom["TSNS"]))
        self.assertTrue(np.abs(fom["TSPS"] - 0.9924242424242424) < 1.0e-12)
        self.assertTrue(np.abs(fom["TEFF"] - fom["TSPS"]) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y) - fom["TEFF"]) < 1.0e-12
        )

    def test_plsda2_hard(self):
        """Test PLSDA on a 2-class example with hard decision boundaries."""
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/plsda2_train.csv"
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(df["Class"].values, dtype=str)
        plsda = PLSDA(
            n_components=5,
            alpha=0.05,
            gamma=0.01,
            not_assigned="UNKNOWN",
            style="hard",
            scale_x=False,
            score_metric="TEFF",
        )
        _ = plsda.fit(raw_x, raw_y)

        # Check class centers
        class_centers = np.array(
            [
                [1.11821537],
                [-0.29599819],
            ]
        )
        err = np.all(
            np.abs((plsda._PLSDA__class_centers_ - class_centers)) < 1.0e-6
        )
        self.assertTrue(err)

        # Check X is centered and, in this case, NOT scaled
        err = np.all(
            np.abs((raw_x - np.mean(raw_x, axis=0)) - plsda._PLSDA__X_) < 1.0e-6
        )
        self.assertTrue(err)

        # Check Y is centered
        err = np.all(np.abs(np.mean(plsda._PLSDA__y_, axis=0) < 1.0e-6))
        self.assertTrue(err)

        # Check a few transform() values (sPC scores) numerically
        xform = np.array([[1.0677767], [1.29543202], [-0.08085378]])
        err = np.all(
            np.abs((plsda.transform([raw_x[0], raw_x[10], raw_x[50]]) - xform))
            < 1.0e-6
        )
        self.assertTrue(err)

        # Check Mahalanobis distance
        d_plsda = plsda.mahalanobis(
            [raw_x[0], raw_x[10], raw_x[50]]
        )  # Actually d^2
        d_scipy = []
        for t in plsda.transform([raw_x[0], raw_x[10], raw_x[50]]):
            d_ = []
            for i in range(len(plsda._PLSDA__ohencoder_.categories_[0])):
                d_.append(
                    mahalanobis(
                        t,
                        plsda._PLSDA__class_centers_[i],
                        np.linalg.inv(plsda._PLSDA__L_),
                    )
                )
            d_scipy.append(d_)
        d_scipy = np.array(d_scipy)
        self.assertTrue(np.allclose(d_scipy, d_plsda**0.5))

        # Check a few predictions manually
        pred = plsda.predict([raw_x[0], raw_x[10], raw_x[50]])
        self.assertEqual(pred[0], "Fakes")
        self.assertEqual(pred[1], "Fakes")
        self.assertEqual(pred[2], "Oregano")

        # Check some distances from projection to class centers
        distances = (86 - 1) * np.array(
            [
                [0.00010100838627651073, 0.07384404529354482],
                [0.0012469215301886371, 0.10055537646974555],
                [0.057084547679274567, 0.0018377660482487594],
            ]
        )
        d = plsda.mahalanobis(raw_x)
        d = np.array([d[0], d[10], d[50]])
        err = np.all(np.abs((d - distances)) < 1.0e-12)
        self.assertTrue(err)

        # Check FOM on test and train
        fom = plsda.figures_of_merit(plsda.predict(raw_x), raw_y)
        x = np.array([[18, 0, 0], [0, 68, 0]])
        err = np.all((fom["CM"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([18, 68])
        err = np.all((fom["I"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((fom["CSNS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((fom["CSPS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((fom["CEFF"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(fom["TSNS"] - 1.0) < 1.0e-12)
        self.assertTrue(np.abs(fom["TSPS"] - 1.0) < 1.0e-12)
        self.assertTrue(np.abs(fom["TEFF"] - 1.0) < 1.0e-12)

        # Check score
        self.assertTrue(np.abs(plsda.score(raw_x, raw_y) - 1.0) < 1.0e-12)

        # Check test set
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/plsda2_test.csv",
            header=None,
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(df.values[:, 2], dtype=str)
        fom = plsda.figures_of_merit(plsda.predict(raw_x), raw_y)

        x = np.array([[1, 2, 0], [0, 3, 0], [0, 3, 0]])
        err = np.all((fom["CM"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([3, 0, 3, 0, 3])
        err = np.all((fom["I"].values - x) == 0)
        self.assertTrue(err)

        self.assertTrue(np.all([np.isnan(v) for v in fom["CSNS"].values]))

        x = np.array([0.888889, 0.111111])
        err = np.all(np.abs((fom["CSPS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.888889, 0.111111])
        err = np.all(np.abs((fom["CEFF"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.isnan(fom["TSNS"]))
        self.assertTrue(np.abs(fom["TSPS"] - 0.0) < 1.0e-12)
        self.assertTrue(np.abs(fom["TEFF"] - fom["TSPS"]) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y) - fom["TEFF"]) < 1.0e-12
        )

    def test_plsda2_soft(self):
        """Test PLSDA on a 2-class example with soft decision boundaries."""
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/plsda2_train.csv"
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(df["Class"].values, dtype=str)
        plsda = PLSDA(
            n_components=5,
            alpha=0.05,
            gamma=0.01,
            not_assigned="UNKNOWN",
            style="soft",
            scale_x=False,
            score_metric="TEFF",
        )
        _ = plsda.fit(raw_x, raw_y)

        # Check class centers
        class_centers = np.array(
            [
                [1.11821537],
                [-0.29599819],
            ]
        )
        err = np.all(
            np.abs((plsda._PLSDA__class_centers_ - class_centers)) < 1.0e-6
        )
        self.assertTrue(err)

        # Check X is centered and, in this case, NOT scaled
        err = np.all(
            np.abs((raw_x - np.mean(raw_x, axis=0)) - plsda._PLSDA__X_) < 1.0e-6
        )
        self.assertTrue(err)

        # Check Y is centered
        err = np.all(np.abs(np.mean(plsda._PLSDA__y_, axis=0) < 1.0e-6))
        self.assertTrue(err)

        # Check a few transform() values (sPC scores) numerically
        xform = np.array(
            [
                [1.0677767],
                [1.29543202],
                [-0.08085378],
            ]
        )
        err = np.all(
            np.abs((plsda.transform([raw_x[0], raw_x[10], raw_x[50]]) - xform))
            < 1.0e-6
        )
        self.assertTrue(err)

        # Check Mahalanobis distance
        d_plsda = plsda.mahalanobis(
            [raw_x[0], raw_x[10], raw_x[50]]
        )  # Actually d^2
        d_scipy = []
        for t in plsda.transform([raw_x[0], raw_x[10], raw_x[50]]):
            d_ = []
            for i in range(len(plsda._PLSDA__ohencoder_.categories_[0])):
                d_.append(
                    mahalanobis(
                        t,
                        plsda._PLSDA__class_centers_[i],
                        np.linalg.inv(plsda._PLSDA__S_[i]),
                    )
                )
            d_scipy.append(d_)
        d_scipy = np.array(d_scipy)
        self.assertTrue(np.allclose(d_scipy, d_plsda**0.5))

        # Check a few predictions manually
        pred = plsda.predict([raw_x[0], raw_x[10], raw_x[50]])
        self.assertTrue(np.all([a == b for a, b in zip(pred[0], ["Fakes"])]))
        self.assertTrue(np.all([a == b for a, b in zip(pred[1], ["Fakes"])]))
        self.assertTrue(np.all([a == b for a, b in zip(pred[2], ["Oregano"])]))

        # Check some distances from projection to class centers
        distances = np.array(
            [
                [4.93491630e-02, 5.38051427e01],
                [6.09202227e-01, 7.32678763e01],
                [2.78895125e01, 1.33905536e00],
            ]
        )
        d = plsda.mahalanobis(raw_x)
        d = np.array([d[0], d[10], d[50]])
        err = np.all(np.abs((d - distances)) < 1.0e-6)
        self.assertTrue(err)

        # Check FOM on test and train
        fom = plsda.figures_of_merit(plsda.predict(raw_x), raw_y)
        x = np.array([[17, 0, 1], [0, 65, 3]])
        err = np.all((fom["CM"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([18, 68])
        err = np.all((fom["I"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([0.944444, 0.955882])
        err = np.all(np.abs((fom["CSNS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((fom["CSPS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.971825, 0.977692])
        err = np.all(np.abs((fom["CEFF"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(fom["TSNS"] - 0.9534883720930233) < 1.0e-12)
        self.assertTrue(np.abs(fom["TSPS"] - 1.0) < 1.0e-12)
        self.assertTrue(np.abs(fom["TEFF"] - 0.9764672918705589) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y) - 0.9764672918705589) < 1.0e-12
        )

        # Check no outliers in this example
        self.assertTrue(np.all(~plsda.check_outliers()))

        # Check test set
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/plsda2_test.csv",
            header=None,
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(df.values[:, 2], dtype=str)
        fom = plsda.figures_of_merit(plsda.predict(raw_x), raw_y)

        x = np.array([[0, 0, 3], [0, 0, 3], [0, 0, 3]])
        err = np.all((fom["CM"].values - x) == 0)
        self.assertTrue(err)

        x = np.array([3, 0, 3, 0, 3])
        err = np.all((fom["I"].values - x) == 0)
        self.assertTrue(err)

        self.assertTrue(np.all([np.isnan(v) for v in fom["CSNS"].values]))

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((fom["CSPS"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((fom["CEFF"].values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.isnan(fom["TSNS"]))
        self.assertTrue(np.abs(fom["TSPS"] - 1.0) < 1.0e-12)
        self.assertTrue(np.abs(fom["TEFF"] - fom["TSPS"]) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y) - fom["TEFF"]) < 1.0e-12
        )
