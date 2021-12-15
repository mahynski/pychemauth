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

        # Check a few predictions manually
        pred = plsda.predict([raw_x[1], raw_x[125], raw_x[150]])
        self.assertEqual(pred[0][0], "JPN1")
        self.assertEqual(pred[1][0], "PHI")
        self.assertEqual(pred[2][0], "THA1")

        # Check some distances from projection to class centers
        distances = np.array(
            [
                [1.5128301276851186, 14.455080929988402, 4.954316511946947],
                [12.44893442348872, 2.5362319585422464, 8.76466313584504],
                [5.827809038186319, 19.800775517297765, 0.008399618441443395],
            ]
        )
        err = np.all(np.abs((plsda._PLSDA__distances_ - distances)) < 1.0e-12)
        self.assertTrue(err)

        # Check FOM on test and train
        df, Itot, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = plsda.figures_of_merit(
            plsda.predict(raw_x), raw_y
        )
        x = np.array([[87, 0, 10, 0], [2, 24, 3, 0], [0, 0, 219, 0]])
        err = np.all((df.values - x) == 0)
        self.assertTrue(err)

        x = np.array([97, 29, 219])
        err = np.all((Itot.values - x) == 0)
        self.assertTrue(err)

        x = np.array([0.89690722, 0.82758621, 1.0])
        err = np.all(np.abs((CSNS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.99193548, 1.0, 0.8968254])
        err = np.all(np.abs((CSPS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.94322537, 0.90971765, 0.94700866])
        err = np.all(np.abs((CEFF.values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(TSNS - 0.9565217391304348) < 1.0e-12)
        self.assertTrue(np.abs(TSPS - 0.9565217391304348) < 1.0e-12)
        self.assertTrue(np.abs(TEFF - 0.9565217391304348) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y, use="TEFF") - 0.9565217391304348)
            < 1.0e-12
        )

        # Check test set
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__)) + "/plsda3_test.csv",
            header=None,
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(["THA2"] * len(raw_x), dtype=str)
        df, Itot, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = plsda.figures_of_merit(
            plsda.predict(raw_x), raw_y
        )

        x = np.array(
            [
                [0, 74, 58, 0],
            ]
        )
        err = np.all((df.values - x) == 0)
        self.assertTrue(err)

        x = np.array([0, 0, 0, 132])
        err = np.all((Itot.values - x) == 0)
        self.assertTrue(err)

        self.assertTrue(np.all([np.isnan(v) for v in CSNS.values]))

        x = np.array([1.0, 0.43939394, 0.56060606])
        err = np.all(np.abs((CSPS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 0.43939394, 0.56060606])
        err = np.all(np.abs((CEFF.values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(TSNS - 0.0) < 1.0e-12)
        self.assertTrue(np.abs(TSPS - 0.0) < 1.0e-12)
        self.assertTrue(np.abs(TEFF - 0.0) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y, use="TEFF") - 0.0) < 1.0e-12
        )

    def test_plsda3_soft(self):
        """Test PLSDA on a 3-class example with soft decision boundaries."""
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
            style="soft",
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
                [2.538126045790289, 5.587995799331847, 37.83387995734197],
                [17.532372477230968, 1.0921577192191358, 37.953290659545495],
                [28.792184657319407, 18.803339592324903, 0.06600758061716426],
            ]
        )
        err = np.all(np.abs((plsda._PLSDA__distances_ - distances)) < 1.0e-12)
        self.assertTrue(err)

        # Check FOM on test and train
        df, Itot, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = plsda.figures_of_merit(
            plsda.predict(raw_x), raw_y
        )
        x = np.array([[91, 17, 0, 2], [0, 27, 0, 2], [2, 2, 203, 13]])
        err = np.all((df.values - x) == 0)
        self.assertTrue(err)

        x = np.array([97, 29, 219])
        err = np.all((Itot.values - x) == 0)
        self.assertTrue(err)

        x = np.array([0.93814433, 0.93103448, 0.92694064])
        err = np.all(np.abs((CSNS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.99193548, 0.93987342, 1.0])
        err = np.all(np.abs((CSPS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.96466505, 0.93544351, 0.96277756])
        err = np.all(np.abs((CEFF.values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(TSNS - 0.9304347826086956) < 1.0e-12)
        self.assertTrue(np.abs(TSPS - 0.9391304347826087) < 1.0e-12)
        self.assertTrue(np.abs(TEFF - 0.9347724974175087) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y, use="TEFF") - 0.9347724974175087)
            < 1.0e-12
        )

        # Check no outliers in this example
        self.assertTrue(np.all(~plsda.check_outliers()))

        # Check test set
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__)) + "/plsda3_test.csv",
            header=None,
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(["THA2"] * len(raw_x), dtype=str)
        df, Itot, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = plsda.figures_of_merit(
            plsda.predict(raw_x), raw_y
        )

        x = np.array(
            [
                [0, 0, 2, 130],
            ]
        )
        err = np.all((df.values - x) == 0)
        self.assertTrue(err)

        x = np.array([0, 0, 0, 132])
        err = np.all((Itot.values - x) == 0)
        self.assertTrue(err)

        self.assertTrue(np.all([np.isnan(v) for v in CSNS.values]))

        x = np.array([1.0, 1.0, 0.98484848])
        err = np.all(np.abs((CSPS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 1.0, 0.98484848])
        err = np.all(np.abs((CEFF.values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(TSNS - 0.0) < 1.0e-12)
        self.assertTrue(np.abs(TSPS - 0.9848484848484849) < 1.0e-12)
        self.assertTrue(np.abs(TEFF - 0.9848484848484849) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y, use="TEFF") - 0.9848484848484849)
            < 1.0e-12
        )

    def test_plsda2_hard(self):
        """Test PLSDA on a 2-class example with hard decision boundaries."""
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__)) + "/plsda2_train.csv"
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

        # Check a few predictions manually
        pred = plsda.predict([raw_x[0], raw_x[10], raw_x[50]])
        self.assertEqual(pred[0][0], "Fakes")
        self.assertEqual(pred[1][0], "Fakes")
        self.assertEqual(pred[2][0], "Oregano")

        # Check some distances from projection to class centers
        distances = np.array(
            [
                [0.008585712833503411, 6.276743849951309],
                [0.10598833006603414, 8.54720699992837],
                [4.852186552738337, 0.15621011410114455],
            ]
        )
        err = np.all(np.abs((plsda._PLSDA__distances_ - distances)) < 1.0e-12)
        self.assertTrue(err)

        # Check FOM on test and train
        df, Itot, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = plsda.figures_of_merit(
            plsda.predict(raw_x), raw_y
        )
        x = np.array([[18, 0, 0], [0, 68, 0]])
        err = np.all((df.values - x) == 0)
        self.assertTrue(err)

        x = np.array([18, 68])
        err = np.all((Itot.values - x) == 0)
        self.assertTrue(err)

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((CSNS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((CSPS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((CEFF.values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(TSNS - 1.0) < 1.0e-12)
        self.assertTrue(np.abs(TSPS - 1.0) < 1.0e-12)
        self.assertTrue(np.abs(TEFF - 1.0) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y, use="TEFF") - 1.0) < 1.0e-12
        )

        # Check test set
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__)) + "/plsda2_test.csv",
            header=None,
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(df.values[:, 2], dtype=str)
        df, Itot, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = plsda.figures_of_merit(
            plsda.predict(raw_x), raw_y
        )

        x = np.array([[1, 2, 0], [0, 3, 0], [0, 3, 0]])
        err = np.all((df.values - x) == 0)
        self.assertTrue(err)

        x = np.array([3, 0, 3, 0, 3])
        err = np.all((Itot.values - x) == 0)
        self.assertTrue(err)

        self.assertTrue(np.all([np.isnan(v) for v in CSNS.values]))

        x = np.array([0.888889, 0.111111])
        err = np.all(np.abs((CSPS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.888889, 0.111111])
        err = np.all(np.abs((CEFF.values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(TSNS - 0.0) < 1.0e-12)
        self.assertTrue(np.abs(TSPS - 0.0) < 1.0e-12)
        self.assertTrue(np.abs(TEFF - 0.0) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y, use="TEFF") - 0.0) < 1.0e-12
        )

    def test_plsda2_soft(self):
        """Test PLSDA on a 2-class example with soft decision boundaries."""
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__)) + "/plsda2_train.csv"
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

        # Check a few predictions manually
        pred = plsda.predict([raw_x[0], raw_x[10], raw_x[50]])
        self.assertTrue(np.all([a == b for a, b in zip(pred[0], ["Fakes"])]))
        self.assertTrue(np.all([a == b for a, b in zip(pred[1], ["Fakes"])]))
        self.assertTrue(np.all([a == b for a, b in zip(pred[2], ["Oregano"])]))

        # Check some distances from projection to class centers
        distances = np.array(
            [
                [0.04934916301333544, 53.80514269526369],
                [0.6092022269286212, 73.26787634972074],
                [27.88951247330385, 1.3390553574558735],
            ]
        )
        err = np.all(np.abs((plsda._PLSDA__distances_ - distances)) < 1.0e-12)
        self.assertTrue(err)

        # Check FOM on test and train
        df, Itot, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = plsda.figures_of_merit(
            plsda.predict(raw_x), raw_y
        )
        x = np.array([[17, 0, 1], [0, 65, 3]])
        err = np.all((df.values - x) == 0)
        self.assertTrue(err)

        x = np.array([18, 68])
        err = np.all((Itot.values - x) == 0)
        self.assertTrue(err)

        x = np.array([0.944444, 0.955882])
        err = np.all(np.abs((CSNS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((CSPS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([0.971825, 0.977692])
        err = np.all(np.abs((CEFF.values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(TSNS - 0.9534883720930233) < 1.0e-12)
        self.assertTrue(np.abs(TSPS - 1.0) < 1.0e-12)
        self.assertTrue(np.abs(TEFF - 0.9764672918705589) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y, use="TEFF") - 0.9764672918705589)
            < 1.0e-12
        )

        # Check no outliers in this example
        self.assertTrue(np.all(~plsda.check_outliers()))

        # Check test set
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__)) + "/plsda2_test.csv",
            header=None,
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(df.values[:, 2], dtype=str)
        df, Itot, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = plsda.figures_of_merit(
            plsda.predict(raw_x), raw_y
        )

        x = np.array([[0, 0, 3], [0, 0, 3], [0, 0, 3]])
        err = np.all((df.values - x) == 0)
        self.assertTrue(err)

        x = np.array([3, 0, 3, 0, 3])
        err = np.all((Itot.values - x) == 0)
        self.assertTrue(err)

        self.assertTrue(np.all([np.isnan(v) for v in CSNS.values]))

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((CSPS.values - x)) < 1.0e-6)
        self.assertTrue(err)

        x = np.array([1.0, 1.0])
        err = np.all(np.abs((CEFF.values - x)) < 1.0e-6)
        self.assertTrue(err)

        self.assertTrue(np.abs(TSNS - 0.0) < 1.0e-12)
        self.assertTrue(np.abs(TSPS - 1.0) < 1.0e-12)
        self.assertTrue(np.abs(TEFF - 1.0) < 1.0e-12)

        # Check score
        self.assertTrue(
            np.abs(plsda.score(raw_x, raw_y, use="TEFF") - 1.0) < 1.0e-12
        )
