"""
Unittests for DDSIMCA_Model.

author: nam
"""
import os
import unittest

import numpy as np
import pandas as pd

from classifier.simca import DDSIMCA_Model, SIMCA_Classifier


class TestDDSIMCA(unittest.TestCase):
    """Test DDSIMCA."""

    """def test_sklearn_compatibility(self):
        #Check compatible with sklearn's estimator API.
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(DDSIMCA_Model(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)"""

    def test_ddsimca_model(self):
        """Test DDSIMCA_Model explicitly."""
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/simca_train.csv"
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(df["Class"].values, dtype=str)

        dds = DDSIMCA_Model(
            n_components=7, alpha=0.05, gamma=0.01, scale_x=False
        )
        _ = dds.fit(raw_x, raw_y)

        # Check DoF
        self.assertEqual(dds._DDSIMCA_Model__Nq_, 4)
        self.assertEqual(dds._DDSIMCA_Model__Nh_, 3)

        # Check distances
        h, q = dds.h_q_(raw_x)

        h_test = np.array(
            [0.05547138082139136, 0.02909437276782147, 0.09955524352200402]
        )
        q_test = np.array(
            [0.012305979489635022, 0.010783302341881816, 0.006410844205906461]
        )

        self.assertTrue(
            np.all(np.abs(h_test - np.array([h[0], h[10], h[50]])) < 1.0e-12)
        )
        self.assertTrue(
            np.all(np.abs(q_test - np.array([q[0], q[10], q[50]])) < 1.0e-12)
        )

        self.assertTrue(
            np.abs(dds._DDSIMCA_Model__h0_ - 0.09722222222222224) < 1.0e-12
        )
        self.assertTrue(
            np.abs(dds._DDSIMCA_Model__q0_ - 0.01785123484489048) < 1.0e-12
        )

        dist = dds.distance(raw_x)
        dist2 = (
            h_test * dds._DDSIMCA_Model__Nh_ / dds._DDSIMCA_Model__h0_
            + q_test * dds._DDSIMCA_Model__Nq_ / dds._DDSIMCA_Model__q0_
        )
        self.assertTrue(
            np.all(
                np.abs(dist2 - np.array([dist[0], dist[10], dist[50]]))
                < 1.0e-12
            )
        )

        # Check critical distances
        self.assertTrue(
            np.abs(dds._DDSIMCA_Model__c_crit_ - 14.067140449340169) < 1.0e-12
        )
        self.assertTrue(
            np.abs(dds._DDSIMCA_Model__c_out_ - 29.08559157741169) < 1.0e-12
        )

        # Check predictions of target class
        self.assertTrue(
            np.all(
                np.where(~dds.predict(raw_x))[0]
                == np.array([14, 18, 20, 21, 22, 65])
            )
        )

        # Check predictions of extreme/outliers
        ext, out = dds.check_outliers(raw_x)
        self.assertTrue(
            np.all(np.where(ext)[0] == np.array([14, 18, 20, 21, 22, 65]))
        )
        self.assertTrue(np.all(np.all(~out)))

        # Check test set (all same class)
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/simca_test.csv",
            header=None,
        )
        raw_x_t = np.array(df.values[:, 3:], dtype=float)
        raw_y_t = np.array(df.values[:, 1], dtype=str)

        self.assertTrue(np.all(dds.predict(raw_x_t)))
        ext, out = dds.check_outliers(raw_x_t)
        self.assertTrue(np.all(~ext))
        self.assertTrue(np.all(~out))

        # Check alternative class
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/simca_test_alt.csv",
            header=None,
        )
        raw_x_a = np.array(df.values[:, 3:], dtype=float)
        raw_y_a = np.array(df.values[:, 1], dtype=str)
        self.assertTrue(
            np.all(
                np.where(dds.predict(raw_x_a))[0]
                == np.array([0, 3, 6, 7, 9, 15])
            )
        )
        ext, out = dds.check_outliers(raw_x_a)
        self.assertTrue(np.all(np.where(ext)[0] == np.array([1, 4, 8, 12, 16])))
        self.assertTrue(
            np.all(np.where(out)[0] == np.array([2, 5, 10, 11, 13, 14, 17]))
        )

    def test_ddsimca_classifier(self):
        """Test SIMCA_Classifier using the DDSIMCA_Model."""
        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/simca_train.csv"
        )
        raw_x = np.array(df.values[:, 3:], dtype=float)
        raw_y = np.array(df["Class"].values, dtype=str)

        df = pd.read_csv(
            os.path.dirname(os.path.realpath(__file__))
            + "/data/simca_test_alt.csv",
            header=None,
        )
        raw_x_a = np.array(df.values[:, 3:], dtype=float)
        raw_y_a = np.array(df.values[:, 1], dtype=str)

        sc = SIMCA_Classifier(
            n_components=7,
            alpha=0.05,
            scale_x=False,
            style="dd-simca",
            target_class="Pure",
        )

        # Fit on 2 classes - only uses Pure to train
        _ = sc.fit(np.vstack((raw_x, raw_x_a)), np.hstack((raw_y, raw_y_a)))
        self.assertTrue(
            np.abs(sc.score(raw_x_a, raw_y_a) - sc.TSPS) < 1.0e-12
        )  # Test only alt class
        self.assertTrue(
            np.abs(sc.score(raw_x, raw_y) - sc.TSNS) < 1.0e-12
        )  # Test only target class
        self.assertTrue(
            np.abs(
                sc.score(
                    np.vstack((raw_x, raw_x_a)), np.hstack((raw_y, raw_y_a))
                )
                - np.sqrt(sc.TSPS * sc.TSNS)
            )
            < 1.0e-12
        )
        self.assertTrue(np.abs(sc.TEFF - np.sqrt(sc.TSPS * sc.TSNS)) < 1.0e-12)
