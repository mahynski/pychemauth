"""
Unittests for OSR models.

author: nam
"""
import os
import unittest
import keras
import pytest

import numpy as np

from pychemauth import utils
from pychemauth.classifier.osr import OpenSetClassifier, DeepOOD
from pychemauth.classifier.plsda import PLSDA
from pychemauth.manifold.elliptic import EllipticManifold_Model

from sklearn.datasets import load_iris as load_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier


class TestOSC_HardMulticlass_PrefitDeep(unittest.TestCase):
    """Test OSC with Hard Multiclass Classifiers which are Deep."""

    @classmethod
    def setUpClass(self):
        """Configure data."""
        self.orig_model = utils.NNTools.load(
            os.path.dirname(os.path.abspath(__file__)) + "/data/cnn-model.keras"
        )

        self.X_iter = utils.NNTools.build_loader(
            os.path.dirname(os.path.abspath(__file__)) + "/data/2d-pgaa/",
            batch_size=30,
            shuffle=False,
        )

        X_data, y_data = [], []
        for X_batch_, y_batch_ in self.X_iter:
            X_data.append(X_batch_)
            y_data.append(y_batch_)
        self.X_data = np.concatenate(X_data)
        self.y_data = np.concatenate(y_data)

    @pytest.mark.skip(
        reason="This takes a very long time - reenable for debugging"
    )
    def test_energy(self):
        """Test energy-based OOD with a prefit deep classifier."""
        # Featurization
        model = OpenSetClassifier(
            clf_model=DeepOOD.EnergyPrefitClf(),
            clf_prefit=True,
            outlier_model=DeepOOD.Energy,
            outlier_kwargs={
                "model": None,  # This tells it to expect X as X_feature
                "alpha": 0.05,
                "T": 1.0,
            },
            known_classes=np.arange(10),
            inlier_value=True,
            unknown_class=-1,
            score_metric="TEFF",
            clf_style="hard",
            score_using="all",
        )
        featurizer = keras.Sequential(self.orig_model.layers[:])
        featurizer.layers[-1].activation = None  # Deactivate to get logits

        X_feature = featurizer.predict(self.X_iter)
        _ = model.fit(X_feature, self.y_data)
        feat_preds = model.predict(X_feature)
        feat_threshold = model.fitted_outlier_model.threshold

        # No featurization
        model = OpenSetClassifier(
            clf_model=utils.NNTools.load(
                os.path.dirname(os.path.abspath(__file__))
                + "/data/cnn-model.keras"
            ),
            clf_prefit=True,
            outlier_model=DeepOOD.Energy,
            outlier_kwargs={
                "model": utils.NNTools.load(
                    os.path.dirname(os.path.abspath(__file__))
                    + "/data/cnn-model.keras"
                ),
                "alpha": 0.05,
                "T": 1.0,
            },
            known_classes=np.arange(10),
            inlier_value=True,
            unknown_class=-1,
            score_metric="TEFF",
            clf_style="hard",
            score_using="all",
        )
        _ = model.fit(self.X_iter)
        nofeat_preds = model.predict(self.X_iter)
        nofeat_threshold = model.fitted_outlier_model.threshold

        # Compare the two results
        correct = [
            7,
            1,
            0,
            1,
            0,
            0,
            8,
            0,
            1,
            4,
            7,
            1,
            7,
            2,
            9,
            6,
            2,
            7,
            4,
            1,
            2,
            1,
            1,
            2,
            8,
            5,
            0,
            6,
            2,
            -1,
            4,
            1,
            1,
            5,
            1,
            5,
            2,
            1,
            6,
            8,
            3,
            1,
            1,
            4,
            9,
            7,
            -1,
            1,
            3,
            2,
            -1,
            5,
            7,
            7,
            -1,
            1,
            5,
            7,
            3,
            2,
            1,
        ]

        np.testing.assert_almost_equal(feat_threshold, 8.357131958007812)
        np.testing.assert_almost_equal(nofeat_threshold, 8.357131958007812)
        np.testing.assert_equal(correct, feat_preds)
        np.testing.assert_equal(correct, nofeat_preds)

    @pytest.mark.skip(
        reason="This takes a very long time - reenable for debugging"
    )
    def test_softmax(self):
        """Test softmax OOD with a prefit deep classifier."""
        # Featurization
        model = OpenSetClassifier(
            clf_model=DeepOOD.SoftmaxPrefitClf(),
            clf_prefit=True,
            outlier_model=DeepOOD.Softmax,
            outlier_kwargs={
                "model": None,  # This tells it to expect X as X_feature
                "alpha": 0.05,
            },
            known_classes=np.arange(10),
            inlier_value=True,
            unknown_class=-1,
            score_metric="TEFF",
            clf_style="hard",
            score_using="all",
        )
        X_feature = self.orig_model.predict(self.X_iter)
        _ = model.fit(X_feature, self.y_data)
        feat_preds = model.predict(X_feature)
        feat_threshold = model.fitted_outlier_model.threshold

        # No featurization
        model = OpenSetClassifier(
            clf_model=utils.NNTools.load(
                os.path.dirname(os.path.abspath(__file__))
                + "/data/cnn-model.keras"
            ),
            clf_prefit=True,
            outlier_model=DeepOOD.Softmax,
            outlier_kwargs={
                "model": utils.NNTools.load(
                    os.path.dirname(os.path.abspath(__file__))
                    + "/data/cnn-model.keras"
                ),
                "alpha": 0.05,
            },
            known_classes=np.arange(10),
            inlier_value=True,
            unknown_class=-1,
            score_metric="TEFF",
            clf_style="hard",
            score_using="all",
        )
        _ = model.fit(self.X_iter)
        nofeat_preds = model.predict(self.X_iter)
        nofeat_threshold = model.fitted_outlier_model.threshold

        # Compare the two results
        correct = [
            -1,
            1,
            0,
            1,
            0,
            0,
            8,
            0,
            1,
            4,
            7,
            1,
            -1,
            2,
            9,
            6,
            2,
            7,
            4,
            1,
            2,
            1,
            1,
            2,
            8,
            5,
            0,
            -1,
            2,
            1,
            4,
            1,
            1,
            5,
            1,
            5,
            2,
            1,
            6,
            8,
            3,
            1,
            1,
            4,
            9,
            7,
            1,
            1,
            3,
            2,
            -1,
            5,
            7,
            7,
            0,
            1,
            5,
            7,
            3,
            2,
            1,
        ]

        np.testing.assert_almost_equal(feat_threshold, 0.9978286623954773)
        np.testing.assert_almost_equal(nofeat_threshold, 0.9978286623954773)
        np.testing.assert_equal(correct, feat_preds)
        np.testing.assert_equal(correct, nofeat_preds)


class TestOSC_HardMulticlass(unittest.TestCase):
    """Test OSC with Hard Multiclass System."""

    @classmethod
    def setUpClass(self):
        """Configure data."""
        self.X, y = load_data(return_X_y=True, as_frame=True)
        names = dict(zip(np.arange(3), ["setosa", "versicolor", "virginica"]))
        self.y = y.apply(lambda x: names[x])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X.values,
            self.y.values,
            shuffle=True,
            random_state=42,
            test_size=0.2,
            stratify=self.y,
        )

    def test_isolationforest(self):
        """Use IsolationForest as an outlier detector."""
        # Leave out virginica on purpose
        X_train_ = self.X_train[self.y_train != "versicolor"]
        y_train_ = self.y_train[self.y_train != "versicolor"]

        osc = OpenSetClassifier(
            clf_model=PLSDA,
            clf_kwargs={
                "n_components": 3,
                "alpha": 0.05,
                "gamma": 0.01,
                "style": "hard",
                "scale_x": True,
            },
            outlier_model=IsolationForest,
            outlier_kwargs={
                "n_estimators": 100,
                "max_samples": 1.0,
                "contamination": 0.15,
                "max_features": 1.0,
                "bootstrap": True,
                "random_state": 42,
            },
            score_metric="TEFF",
            clf_style="hard",
            unknown_class="UNKNOWN FLOWER",
        )

        osc.fit(X_train_, y_train_)

        pred_ = osc.predict(X_train_)[:10]
        correct_ = [
            "setosa",
            "UNKNOWN FLOWER",
            "setosa",
            "virginica",
            "virginica",
            "virginica",
            "UNKNOWN FLOWER",
            "virginica",
            "setosa",
            "setosa",
        ]
        np.testing.assert_equal(pred_, correct_)

        np.testing.assert_almost_equal(
            osc.score(X_train_, y_train_), 0.9219544457292888, decimal=9
        )

        np.testing.assert_equal(
            osc.figures_of_merit(osc.predict(X_train_), y_train_)[
                "CM"
            ].values.ravel(),
            [35, 0, 5, 0, 33, 7],
        )

        np.testing.assert_equal(
            osc.figures_of_merit(osc.predict(self.X_train), self.y_train)[
                "CM"
            ].values.ravel(),
            [35, 0, 5, 0, 3, 37, 0, 33, 7],
        )

    def test_elliptical(self):
        """Use EllipticalManifold as an outlier detector."""
        # Leave out virginica on purpose
        X_train_ = self.X_train[self.y_train != "versicolor"]
        y_train_ = self.y_train[self.y_train != "versicolor"]

        osc = OpenSetClassifier(
            clf_model=PLSDA,
            clf_kwargs={
                "n_components": 3,
                "alpha": 0.05,
                "gamma": 0.01,
                "style": "hard",
                "scale_x": True,
            },
            outlier_model=EllipticManifold_Model,
            outlier_kwargs={"alpha": 0.05, "robust": True, "center": "score"},
            score_metric="TEFF",
            clf_style="hard",
            unknown_class="UNKNOWN FLOWER",
        )

        osc.fit(X_train_, y_train_)

        pred_ = osc.predict(X_train_)[:10]
        correct_ = [
            "setosa",
            "UNKNOWN FLOWER",
            "setosa",
            "virginica",
            "virginica",
            "UNKNOWN FLOWER",
            "UNKNOWN FLOWER",
            "UNKNOWN FLOWER",
            "setosa",
            "setosa",
        ]
        np.testing.assert_equal(pred_, correct_)

        np.testing.assert_almost_equal(
            osc.score(X_train_, y_train_), 0.8874119674649424, decimal=9
        )

        np.testing.assert_equal(
            osc.figures_of_merit(osc.predict(self.X_train), self.y_train)[
                "CM"
            ].values.ravel(),
            [39, 0, 1, 0, 32, 8, 0, 24, 16],
        )


class TestOSC_SoftMulticlass(unittest.TestCase):
    """Test OSC with Soft Multiclass System."""

    @classmethod
    def setUpClass(self):
        """Configure data."""
        self.X, y = load_data(return_X_y=True, as_frame=True)
        names = dict(zip(np.arange(3), ["setosa", "versicolor", "virginica"]))
        self.y = y.apply(lambda x: names[x])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X.values,
            self.y.values,
            shuffle=True,
            random_state=42,
            test_size=0.2,
            stratify=self.y,
        )

    def test_isolationforest(self):
        """Use IsolationForest as an outlier detector."""
        # Leave out virginica on purpose
        X_train_ = self.X_train[self.y_train != "versicolor"]
        y_train_ = self.y_train[self.y_train != "versicolor"]

        osc = OpenSetClassifier(
            clf_model=PLSDA,
            clf_kwargs={
                "n_components": 3,
                "alpha": 0.05,
                "gamma": 0.01,
                "style": "soft",
                "not_assigned": "UNKNOWN FLOWER",
                "scale_x": True,
            },
            outlier_model=IsolationForest,
            outlier_kwargs={
                "n_estimators": 100,
                "max_samples": 1.0,
                "contamination": 0.15,
                "max_features": 1.0,
                "bootstrap": True,
                "random_state": 42,
            },
            score_metric="TEFF",
            clf_style="soft",
            unknown_class="OUTLIER",
        )

        osc.fit(X_train_, y_train_)

        pred_ = osc.predict(X_train_)[:10]
        correct_ = [
            ["setosa"],
            ["OUTLIER"],
            ["setosa"],
            ["virginica"],
            ["UNKNOWN FLOWER"],
            ["virginica"],
            ["OUTLIER"],
            ["virginica"],
            ["setosa"],
            ["setosa"],
        ]
        np.testing.assert_equal(pred_, correct_)

        np.testing.assert_almost_equal(
            osc.score(X_train_, y_train_), 0.9013878188659973, decimal=9
        )

        np.testing.assert_equal(
            osc.figures_of_merit(osc.predict(X_train_), y_train_)[
                "CM"
            ].values.ravel(),
            [33, 0, 7, 0, 32, 8],
        )

        np.testing.assert_equal(
            osc.figures_of_merit(osc.predict(self.X_train), self.y_train)[
                "CM"
            ].values.ravel(),
            [33, 0, 7, 0, 1, 39, 0, 32, 8],
        )


class TestOSC_BinaryOvA_Class(unittest.TestCase):
    """Test OSC with Binary OvA System to create Class Model."""

    @classmethod
    def setUpClass(self):
        """Configure data."""
        self.X, y = load_data(return_X_y=True, as_frame=True)
        names = dict(zip(np.arange(3), ["setosa", "versicolor", "virginica"]))
        self.y = y.apply(lambda x: names[x])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X.values,
            self.y.values,
            shuffle=True,
            random_state=42,
            test_size=0.2,
            stratify=self.y,
        )

        target = "setosa"
        known_alternative = "virginica"
        unknown_alternative = "versicolor"
        self.mask = (self.y_train == target) | (
            self.y_train == known_alternative
        )

    def test_isolationforest(self):
        """Use IsolationForest as an outlier detector."""
        osc = OpenSetClassifier(
            clf_model=RandomForestClassifier,
            clf_kwargs={
                "n_estimators": 100,
                "max_features": 1,
                "random_state": 42,
                "class_weight": "balanced",
            },
            outlier_model=IsolationForest,
            outlier_kwargs={
                "n_estimators": 100,
                "max_samples": 1.0,
                "contamination": 0.15,
                "max_features": 1.0,
                "bootstrap": True,
                "random_state": 42,
            },
            inlier_value=1,
            unknown_class="UNKNOWN",
            score_metric="TEFF",
            clf_style="hard",
            score_using="all",
        )

        osc.fit(self.X_train[self.mask], self.y_train[self.mask])

        pred_ = osc.predict(self.X_train[self.mask])[:10]
        correct_ = [
            "setosa",
            "UNKNOWN",
            "setosa",
            "virginica",
            "virginica",
            "virginica",
            "UNKNOWN",
            "virginica",
            "setosa",
            "setosa",
        ]
        np.testing.assert_equal(pred_, correct_)

        np.testing.assert_almost_equal(
            osc.score(self.X_train[self.mask], self.y_train[self.mask]),
            0.9219544457292888,
            decimal=9,
        )

        np.testing.assert_equal(
            osc.figures_of_merit(
                osc.predict(self.X_train[self.mask]), self.y_train[self.mask]
            )["CM"].values.ravel(),
            [35, 0, 5, 0, 33, 7],
        )

        np.testing.assert_equal(
            osc.figures_of_merit(osc.predict(self.X_train), self.y_train)[
                "CM"
            ].values.ravel(),
            [35, 0, 5, 0, 3, 37, 0, 33, 7],
        )


class TestOSC_BinaryOvA_OCC(unittest.TestCase):
    """Test OSC with Binary OvA System to create OCC."""

    @classmethod
    def setUpClass(self):
        """Configure data."""
        self.X, y = load_data(return_X_y=True, as_frame=True)
        names = dict(zip(np.arange(3), ["setosa", "versicolor", "virginica"]))
        self.y = y.apply(lambda x: names[x])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X.values,
            self.y.values,
            shuffle=True,
            random_state=42,
            test_size=0.2,
            stratify=self.y,
        )

        self.target = "setosa"
        known_alternative = "virginica"
        unknown_alternative = "versicolor"
        self.mask = (self.y_train == self.target) | (
            self.y_train == known_alternative
        )

        def convert_y_to_binary_format(
            y, target, alternative_class="KNOWN ALTERNATIVE"
        ):
            y_binary = y.copy()
            y_binary[y_binary != target] = alternative_class

            return y_binary

        self.y_train_binary = convert_y_to_binary_format(
            self.y_train[self.mask], self.target
        )
        self.X_train_binary = self.X_train[self.mask]

    def test_isolationforest(self):
        """Use IsolationForest as an outlier detector."""
        osc = OpenSetClassifier(
            clf_model=RandomForestClassifier,
            clf_kwargs={
                "n_estimators": 100,
                "max_features": 1,
                "random_state": 42,
                "class_weight": "balanced",
            },
            outlier_model=IsolationForest,
            outlier_kwargs={
                "n_estimators": 100,
                "max_samples": 1.0,
                "contamination": 0.15,
                "max_features": 1.0,
                "bootstrap": True,
                "random_state": 42,
            },
            inlier_value=1,
            unknown_class="UNKNOWN",
            score_metric="TEFF",
            clf_style="hard",
            score_using=self.target,
        )

        osc.fit(self.X_train_binary, self.y_train_binary)

        pred_ = osc.predict(self.X_train_binary)[:10]
        correct_ = [
            "setosa",
            "UNKNOWN",
            "setosa",
            "KNOWN ALTERNATIVE",
            "KNOWN ALTERNATIVE",
            "KNOWN ALTERNATIVE",
            "UNKNOWN",
            "KNOWN ALTERNATIVE",
            "setosa",
            "setosa",
        ]
        np.testing.assert_equal(pred_, correct_)

        np.testing.assert_almost_equal(
            osc.score(self.X_train, self.y_train), 0.9354143466934853, decimal=9
        )

        np.testing.assert_equal(
            osc.figures_of_merit(osc.predict(self.X_train), self.y_train)[
                "CM"
            ].values.ravel(),
            [0, 35, 5, 3, 0, 37, 33, 0, 7],
        )
