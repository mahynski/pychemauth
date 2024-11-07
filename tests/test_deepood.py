"""
Unittests for DeepOOD.

author: nam
"""
import unittest
import keras
import os

import numpy as np

from pychemauth import utils
from pychemauth.classifier import osr


class Test_DeepOOD_DIME(unittest.TestCase):
    """Test DIME OOD detector."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        orig_model = utils.NNTools.load(
            os.path.dirname(os.path.abspath(__file__)) + "/data/cnn-model.keras"
        )
        self.featurizer = keras.Sequential(orig_model.layers[:-2])
        self.data = utils.NNTools.build_loader(
            os.path.dirname(os.path.abspath(__file__)) + "/data/2d-pgaa/",
            batch_size=30,
            shuffle=True,
        )[0][0]

    def test_no_featurization(self):
        """Test the model when no pre-featurization is performed."""
        try:
            ood = osr.DeepOOD.DIME(model=self.featurizer, alpha=0.05, k=20)
            ood.fit(self.data)
        except Exception as e:
            raise Exception(f"DeepOOD.DIME failed to fit : {e}")
        else:
            np.testing.assert_almost_equal(ood.alpha, 0.05)
            np.testing.assert_equal(ood.k, 20)
            np.testing.assert_almost_equal(ood.threshold, -0.1551388055086136)

            score_test = ood.score_samples(self.data)
            np.testing.assert_almost_equal(
                score_test,
                [
                    -0.04337332,
                    -0.02954598,
                    -0.10246915,
                    -0.1190246,
                    -0.12294023,
                    -0.11626241,
                    -0.02860988,
                    -0.13005805,
                    -0.12091209,
                    -0.10463271,
                    -0.1128294,
                    -0.15786138,
                    -0.08676875,
                    -0.10447733,
                    -0.0273346,
                    -0.099686,
                    -0.12603335,
                    -0.06632295,
                    -0.08149995,
                    -0.11846231,
                    -0.11264917,
                    -0.17057934,
                    -0.15181121,
                    -0.09657414,
                    -0.05000485,
                    -0.01853535,
                    -0.11792821,
                    -0.10253341,
                    -0.09434501,
                    -0.0347723,
                ],
                decimal=5
            )

            np.testing.assert_equal(
                ood.predict(self.data),
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
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            )

    def test_featurization(self):
        """Test the model when pre-featurization is performed."""
        try:
            ood = osr.DeepOOD.DIME(model=None, alpha=0.05, k=20)
            X_feature = self.featurizer.predict(self.data)
            ood.fit(X_feature)
        except Exception as e:
            raise Exception(f"DeepOOD.DIME failed to fit : {e}")
        else:
            np.testing.assert_almost_equal(ood.alpha, 0.05)
            np.testing.assert_equal(ood.k, 20)
            np.testing.assert_almost_equal(ood.threshold, -0.1551388055086136)

            score_test = ood.score_samples(X_feature)
            np.testing.assert_almost_equal(
                score_test,
                [
                    -0.04337332,
                    -0.02954598,
                    -0.10246915,
                    -0.1190246,
                    -0.12294023,
                    -0.11626241,
                    -0.02860988,
                    -0.13005805,
                    -0.12091209,
                    -0.10463271,
                    -0.1128294,
                    -0.15786138,
                    -0.08676875,
                    -0.10447733,
                    -0.0273346,
                    -0.099686,
                    -0.12603335,
                    -0.06632295,
                    -0.08149995,
                    -0.11846231,
                    -0.11264917,
                    -0.17057934,
                    -0.15181121,
                    -0.09657414,
                    -0.05000485,
                    -0.01853535,
                    -0.11792821,
                    -0.10253341,
                    -0.09434501,
                    -0.0347723,
                ],
                decimal=5
            )

            np.testing.assert_equal(
                ood.predict(X_feature),
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
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            )


class Test_DeepOOD_Energy(unittest.TestCase):
    """Test Energy-based OOD detector."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        self.orig_model = utils.NNTools.load(
            os.path.dirname(os.path.abspath(__file__)) + "/data/cnn-model.keras"
        )
        self.featurizer = utils.NNTools.load(
            os.path.dirname(os.path.abspath(__file__)) + "/data/cnn-model.keras"
        )
        self.featurizer.layers[-1].activation = None  # Deactivate to get logits
        self.data = utils.NNTools.build_loader(
            os.path.dirname(os.path.abspath(__file__)) + "/data/2d-pgaa/",
            batch_size=30,
            shuffle=True,
        )[0][0]

    def test_no_featurization(self):
        """Test the model when no pre-featurization is performed."""
        try:
            ood = osr.DeepOOD.Energy(model=self.orig_model, alpha=0.05, T=1.0)
            ood.fit(self.data)
        except Exception as e:
            raise Exception(f"DeepOOD.Energy failed to fit : {e}")
        else:
            np.testing.assert_almost_equal(ood.alpha, 0.05)
            np.testing.assert_almost_equal(ood.T, 1.0)
            np.testing.assert_almost_equal(ood.threshold, 10.107596, decimal=6)

            score_test = ood.score_samples(self.data)
            np.testing.assert_almost_equal(
                score_test,
                [
                    12.946023,
                    11.335374,
                    14.812372,
                    10.651358,
                    15.296914,
                    15.11723,
                    22.082363,
                    14.755956,
                    10.959855,
                    12.201889,
                    12.442291,
                    11.419479,
                    13.329335,
                    14.484063,
                    13.031969,
                    13.195831,
                    13.243632,
                    12.100874,
                    10.260317,
                    10.592216,
                    13.964326,
                    10.107596,
                    10.720448,
                    13.025512,
                    15.540151,
                    15.419064,
                    15.217169,
                    11.417565,
                    13.224181,
                    7.7136126,
                ],
                decimal=6,
            )

            np.testing.assert_equal(
                ood.predict(self.data),
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
                    False,
                ],
            )

    def test_featurization(self):
        """Test the model when pre-featurization is performed."""
        try:
            ood = osr.DeepOOD.Energy(model=None, alpha=0.05, T=1.0)
            X_feature = self.featurizer.predict(self.data)
            ood.fit(X_feature)
        except Exception as e:
            raise Exception(f"DeepOOD.Energy failed to fit : {e}")
        else:
            np.testing.assert_almost_equal(ood.alpha, 0.05)
            np.testing.assert_almost_equal(ood.T, 1.0)
            np.testing.assert_almost_equal(ood.threshold, 10.107596, decimal=6)

            score_test = ood.score_samples(X_feature)
            np.testing.assert_almost_equal(
                score_test,
                [
                    12.946023,
                    11.335374,
                    14.812372,
                    10.651358,
                    15.296914,
                    15.11723,
                    22.082363,
                    14.755956,
                    10.959855,
                    12.201889,
                    12.442291,
                    11.419479,
                    13.329335,
                    14.484063,
                    13.031969,
                    13.195831,
                    13.243632,
                    12.100874,
                    10.260317,
                    10.592216,
                    13.964326,
                    10.107596,
                    10.720448,
                    13.025512,
                    15.540151,
                    15.419064,
                    15.217169,
                    11.417565,
                    13.224181,
                    7.7136126,
                ],
                decimal=6,
            )

            np.testing.assert_equal(
                ood.predict(X_feature),
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
                    False,
                ],
            )


class Test_DeepOOD_Softmax(unittest.TestCase):
    """Test Softmax OOD detector."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        self.orig_model = utils.NNTools.load(
            os.path.dirname(os.path.abspath(__file__)) + "/data/cnn-model.keras"
        )
        self.data = utils.NNTools.build_loader(
            os.path.dirname(os.path.abspath(__file__)) + "/data/2d-pgaa/",
            batch_size=30,
            shuffle=True,
        )[0][0]

    def test_no_featurization(self):
        """Test the model when no pre-featurization is performed."""
        try:
            ood = osr.DeepOOD.Softmax(model=self.orig_model, alpha=0.05)
            ood.fit(self.data)
        except Exception as e:
            raise Exception(f"DeepOOD.Softmax failed to fit : {e}")
        else:
            np.testing.assert_almost_equal(ood.alpha, 0.05)
            np.testing.assert_almost_equal(ood.threshold, 0.99701893)

            score_test = ood.score_samples(self.data)
            np.testing.assert_almost_equal(
                score_test,
                [
                    0.99701893,
                    0.99888146,
                    0.9999965,
                    0.9995356,
                    0.9999954,
                    0.9999985,
                    0.99994904,
                    0.999996,
                    0.99946654,
                    0.9999332,
                    0.9984669,
                    0.9998238,
                    0.99782866,
                    0.9999766,
                    0.9998986,
                    0.99929833,
                    0.9999718,
                    0.99917585,
                    0.99985504,
                    0.9992924,
                    0.99996847,
                    0.9990323,
                    0.9993457,
                    0.9999524,
                    0.9999354,
                    0.998708,
                    0.9999938,
                    0.99422425,
                    0.99996245,
                    0.99799013,
                ],
                decimal=6,
            )

            np.testing.assert_equal(
                ood.predict(self.data),
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
                    False,
                    True,
                    True,
                ],
            )

    def test_featurization(self):
        """Test the model when pre-featurization is performed."""
        try:
            ood = osr.DeepOOD.Softmax(model=None, alpha=0.05)
            X_feature = self.orig_model.predict(self.data)
            ood.fit(X_feature)
        except Exception as e:
            raise Exception(f"DeepOOD.Softmax failed to fit : {e}")
        else:
            np.testing.assert_almost_equal(ood.alpha, 0.05)
            np.testing.assert_almost_equal(ood.threshold, 0.99701893)

            score_test = ood.score_samples(X_feature)
            np.testing.assert_almost_equal(
                score_test,
                [
                    0.99701893,
                    0.99888146,
                    0.9999965,
                    0.9995356,
                    0.9999954,
                    0.9999985,
                    0.99994904,
                    0.999996,
                    0.99946654,
                    0.9999332,
                    0.9984669,
                    0.9998238,
                    0.99782866,
                    0.9999766,
                    0.9998986,
                    0.99929833,
                    0.9999718,
                    0.99917585,
                    0.99985504,
                    0.9992924,
                    0.99996847,
                    0.9990323,
                    0.9993457,
                    0.9999524,
                    0.9999354,
                    0.998708,
                    0.9999938,
                    0.99422425,
                    0.99996245,
                    0.99799013,
                ],
                decimal=6,
            )

            np.testing.assert_equal(
                ood.predict(X_feature),
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
                    False,
                    True,
                    True,
                ],
            )
