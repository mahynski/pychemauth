"""
Unittests for tools associated with creating NN models.

author: nam
"""
import unittest
import keras
import tempfile
import pytest

import numpy as np

from sklearn.preprocessing import LabelEncoder
from pychemauth import utils, datasets
from keras import layers


class TestNNTools(unittest.TestCase):
    """Test utilities in NNTools."""

    @classmethod
    def setUpClass(self):
        """Load sample data."""
        self._X, self._y = datasets.load_pgaa(return_X_y=True)
        self.image_size = (self._X.shape[1], 1)
        self._X = np.expand_dims(self._X, axis=-1)
        self.n_classes = len(np.unique(self._y))
        self._y_str = self._y
        self._enc = LabelEncoder()
        self._y = self._enc.fit_transform(self._y)

    @pytest.mark.dependency(name="test_make_npy_xloader")
    def test_make_npy_xloader(self):
        """Build an XLoader from data on disk."""
        dir_ = tempfile.TemporaryDirectory()

        # Create a dataset on disk
        x_f_, y_f_ = utils.write_dataset(
            dir_.name + "/dummy",
            self._X,
            self._y,
            overwrite=False,
            augment=False,
        )

        # Build the loader
        batch_size = 10
        data = utils.NNTools.build_loader(
            dir_.name + "/dummy", batch_size=batch_size, loader="x", fmt="npy"
        )

        # Check it is the same as the original data
        idx = 0
        for b in range(int(np.ceil(self._X.shape[0] / batch_size))):
            X_, y_ = data[b]
            for i in range(min(batch_size, self._X.shape[0] - idx)):
                np.testing.assert_allclose(self._X[idx], X_[i])
                np.testing.assert_allclose(self._y[idx], y_[i])
                idx += 1

        dir_.cleanup()

    @pytest.mark.dependency(depends=["test_make_npy_xloader"])
    def test_xloader_filter(self):
        """Build an XLoader from data on disk."""
        dir_ = tempfile.TemporaryDirectory()

        # Create a dataset on disk
        x_f_, y_f_ = utils.write_dataset(
            dir_.name + "/dummy",
            self._X,
            self._y_str,  # Write y as strings
            overwrite=False,
            augment=False,
        )

        filter = self._y_str == "Coal and Coke"

        # Build the loader
        batch_size = 10
        data = utils.NNTools.build_loader(
            dir_.name + "/dummy",
            batch_size=batch_size,
            loader="x",
            fmt="npy",
            filter=filter,
        )

        # Indices where 'Coal and Coke' is
        correct_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            30,
            31,
            32,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            225,
            226,
            227,
            228,
            229,
            230,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
            266,
            267,
            268,
            269,
            270,
            271,
            272,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            280,
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            288,
            289,
        ]
        idx = 0
        for b in range(len(data)):
            X_, y_ = data[b]
            for i in range(len(X_)):
                self.assertEqual(y_[i], "Coal and Coke")
                np.testing.assert_allclose(self._X[correct_indices[idx]], X_[i])
                self.assertEqual(self._y_str[correct_indices[idx]], y_[i])
                idx += 1

        dir_.cleanup()

    @pytest.mark.dependency(depends=["test_make_npy_xloader"])
    def test_xloader_filter_include(self):
        """Build an XLoader from data on disk."""
        dir_ = tempfile.TemporaryDirectory()

        # Create a dataset on disk
        x_f_, y_f_ = utils.write_dataset(
            dir_.name + "/dummy",
            self._X,
            self._y_str,  # Write y as strings
            overwrite=False,
            augment=False,
        )

        filter = [False] * (len(self._y) // 2) + [True] * (
            len(self._y) - len(self._y) // 2
        )  # Arbitrarily filter out the first half

        # Build the loader
        batch_size = 10
        data = utils.NNTools.build_loader(
            dir_.name + "/dummy",
            batch_size=batch_size,
            loader="x",
            fmt="npy",
            filter=filter,
            include=["Coal and Coke", "Biomass"],
        )

        # Indices where ['Coal and Coke', 'Biomass'] is in the second "half" of the data
        correct_indices = [
            183,
            184,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
            266,
            267,
            268,
            269,
            270,
            271,
            272,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            280,
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            288,
            289,
        ]
        idx = 0
        for b in range(len(data)):
            X_, y_ = data[b]
            for i in range(len(X_)):
                self.assertIn(y_[i], ["Coal and Coke", "Biomass"])
                np.testing.assert_allclose(self._X[correct_indices[idx]], X_[i])
                self.assertEqual(self._y_str[correct_indices[idx]], y_[i])
                idx += 1

        dir_.cleanup()

    @pytest.mark.dependency(depends=["test_make_npy_xloader"])
    def test_xloader_include_str(self):
        """Build an XLoader from data on disk."""
        dir_ = tempfile.TemporaryDirectory()

        # Create a dataset on disk
        x_f_, y_f_ = utils.write_dataset(
            dir_.name + "/dummy",
            self._X,
            self._y_str,  # Write y as strings
            overwrite=False,
            augment=False,
        )

        # Build the loader
        batch_size = 10
        data = utils.NNTools.build_loader(
            dir_.name + "/dummy",
            batch_size=batch_size,
            loader="x",
            fmt="npy",
            include=["Coal and Coke", "Biomass"],
        )

        # Indices where ['Coal and Coke', 'Biomass'] are
        correct_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            183,
            184,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
            266,
            267,
            268,
            269,
            270,
            271,
            272,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            280,
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            288,
            289,
        ]
        idx = 0
        for b in range(len(data)):
            X_, y_ = data[b]
            for i in range(len(X_)):
                self.assertIn(y_[i], ["Coal and Coke", "Biomass"])
                np.testing.assert_allclose(self._X[correct_indices[idx]], X_[i])
                self.assertEqual(self._y_str[correct_indices[idx]], y_[i])
                idx += 1

        dir_.cleanup()

    @pytest.mark.dependency(depends=["test_make_npy_xloader"])
    def test_xloader_include_int(self):
        """Build an XLoader from data on disk."""
        dir_ = tempfile.TemporaryDirectory()

        # Create a dataset on disk
        x_f_, y_f_ = utils.write_dataset(
            dir_.name + "/dummy",
            self._X,
            self._y,  # Write y as integers
            overwrite=False,
            augment=False,
        )

        # Build the loader
        batch_size = 10
        data = utils.NNTools.build_loader(
            dir_.name + "/dummy",
            batch_size=batch_size,
            loader="x",
            fmt="npy",
            include=self._enc.transform(["Coal and Coke", "Biomass"]),
        )

        # Indices where ['Coal and Coke', 'Biomass'] are
        correct_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            183,
            184,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
            266,
            267,
            268,
            269,
            270,
            271,
            272,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            280,
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            288,
            289,
        ]
        idx = 0
        for b in range(len(data)):
            X_, y_ = data[b]
            for i in range(len(X_)):
                self.assertIn(
                    y_[i], self._enc.transform(["Coal and Coke", "Biomass"])
                )
                np.testing.assert_allclose(self._X[correct_indices[idx]], X_[i])
                self.assertEqual(self._y[correct_indices[idx]], y_[i])
                idx += 1

        dir_.cleanup()

    @pytest.mark.dependency(depends=["test_make_npy_xloader"])
    def test_xloader_exclude_str(self):
        """Build an XLoader from data on disk."""
        dir_ = tempfile.TemporaryDirectory()

        # Create a dataset on disk
        x_f_, y_f_ = utils.write_dataset(
            dir_.name + "/dummy",
            self._X,
            self._y_str,  # Write y as strings
            overwrite=False,
            augment=False,
        )

        # Build the loader
        batch_size = 10
        data = utils.NNTools.build_loader(
            dir_.name + "/dummy",
            batch_size=batch_size,
            loader="x",
            fmt="npy",
            exclude=[
                "Carbon Powder",
                "Concrete",
                "Dolomitic Limestone",
                "Forensic Glass",
                "Fuel Oil",
                "Graphite/Urea Mixture",
                "Lubricating Oil",
                "Phosphate Rock",
                "Steel",
                "Titanium Alloy",
                "Zircaloy",
            ],
        )

        # Indices where only the remaining classes ['Coal and Coke', 'Biomass'] are
        correct_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            183,
            184,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
            266,
            267,
            268,
            269,
            270,
            271,
            272,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            280,
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            288,
            289,
        ]
        idx = 0
        for b in range(len(data)):
            X_, y_ = data[b]
            for i in range(len(X_)):
                self.assertIn(y_[i], ["Coal and Coke", "Biomass"])
                np.testing.assert_allclose(self._X[correct_indices[idx]], X_[i])
                self.assertEqual(self._y_str[correct_indices[idx]], y_[i])
                idx += 1

        dir_.cleanup()

    @pytest.mark.dependency(depends=["test_make_npy_xloader"])
    def test_xloader_exclude_int(self):
        """Build an XLoader from data on disk."""
        dir_ = tempfile.TemporaryDirectory()

        # Create a dataset on disk
        x_f_, y_f_ = utils.write_dataset(
            dir_.name + "/dummy",
            self._X,
            self._y,  # Write y as integers
            overwrite=False,
            augment=False,
        )

        # Build the loader
        batch_size = 10
        data = utils.NNTools.build_loader(
            dir_.name + "/dummy",
            batch_size=batch_size,
            loader="x",
            fmt="npy",
            exclude=self._enc.transform(
                [
                    "Carbon Powder",
                    "Concrete",
                    "Dolomitic Limestone",
                    "Forensic Glass",
                    "Fuel Oil",
                    "Graphite/Urea Mixture",
                    "Lubricating Oil",
                    "Phosphate Rock",
                    "Steel",
                    "Titanium Alloy",
                    "Zircaloy",
                ]
            ),
        )

        # Indices where only the remaining classes ['Coal and Coke', 'Biomass'] are
        correct_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            183,
            184,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
            266,
            267,
            268,
            269,
            270,
            271,
            272,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            280,
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            288,
            289,
        ]
        idx = 0
        for b in range(len(data)):
            X_, y_ = data[b]
            for i in range(len(X_)):
                self.assertIn(
                    y_[i], self._enc.transform(["Coal and Coke", "Biomass"])
                )
                np.testing.assert_allclose(self._X[correct_indices[idx]], X_[i])
                self.assertEqual(self._y[correct_indices[idx]], y_[i])
                idx += 1

        dir_.cleanup()

    def _make_model(self, activation="relu"):
        """Make a simple model to use."""
        input_ = keras.layers.Input(shape=self.image_size)
        conv1 = layers.Conv1D(
            filters=16,
            kernel_size=16,
            activation=activation,
            strides=8,
            padding="same",
            use_bias=True,
        )(input_)
        pool1 = layers.MaxPool1D(2)(conv1)
        conv2 = layers.Conv1D(
            filters=16 * 8,
            kernel_size=8,
            activation=activation,
            strides=4,
            padding="same",
            use_bias=True,
        )(pool1)
        pool2 = layers.MaxPool1D(2)(conv2)
        conv3 = layers.Conv1D(
            filters=16 * 8 * 4,
            kernel_size=8,
            activation=activation,
            strides=2,
            padding="same",
            use_bias=True,
        )(pool2)
        flat = layers.Flatten()(conv3)
        output = layers.Dense(self.n_classes, activation="softmax")(flat)
        model = keras.Model(inputs=[input_], outputs=[output])

        return model

    @pytest.mark.dependency(name="test_find_learning_rate")
    def test_find_learning_rate(self):
        """Test learning rate finder loop with callback."""
        try:
            finder = utils.NNTools.find_learning_rate(
                self._make_model(),
                (self._X, self._y),
                n_updates=100,
                start_lr=1.0e-8,
                end_lr=10.0,
            )
        except Exception as e:
            raise Exception(f"LearningRateFinder failed to run : {e}")

        np.testing.assert_almost_equal(
            finder.lr_mult, (10.0 / 1.0e-8) ** (1 / 100.0)
        )
        np.testing.assert_almost_equal(finder.start_lr, 1.0e-8)
        np.testing.assert_almost_equal(finder.end_lr, 10.0)

    @pytest.mark.dependency(
        depends=["test_make_npy_xloader", "test_find_learning_rate"]
    )
    def test_find_learning_rate_iter(self):
        """Test learning rate finder loop with callback using a data iterator."""
        dir_ = tempfile.TemporaryDirectory()

        # Write dataset to disk and create data loader
        _ = utils.write_dataset(
            dir_.name + "/train",
            self._X,
            self._y,
            overwrite=False,
            augment=False,
        )
        data = utils.NNTools.build_loader(
            dir_.name + "/train", batch_size=10, shuffle=True
        )

        try:
            finder = utils.NNTools.find_learning_rate(
                self._make_model(),
                data,
                n_updates=100,
                start_lr=1.0e-8,
                end_lr=10.0,
            )
        except Exception as e:
            raise Exception(f"LearningRateFinder failed to run : {e}")

        np.testing.assert_almost_equal(
            finder.lr_mult, (10.0 / 1.0e-8) ** (1 / 100.0)
        )
        np.testing.assert_almost_equal(finder.start_lr, 1.0e-8)
        np.testing.assert_almost_equal(finder.end_lr, 10.0)

        dir_.cleanup()

    @pytest.mark.dependency(name="test_clr_triangular_train")
    def test_clr_triangular_train(self):
        """Test CLR triangular policy during training."""
        clr = utils.NNTools.CyclicalLearningRate(
            base_lr=0.001,
            max_lr=0.01,
            step_size=10,
            mode="triangular",
        )

        model = utils.NNTools.train(
            model=self._make_model(),
            data=(self._X, self._y),
            fit_kwargs={
                "batch_size": 50,
                "epochs": 20,
                "validation_split": 0.2,
                "shuffle": True,
                "callbacks": [clr],
            },
            model_filename=None,
            history_filename=None,
            wandb_project=None,
        )

        # Should go back to min
        np.testing.assert_almost_equal(model.optimizer.lr, 0.001)
        np.testing.assert_almost_equal(
            clr.history["iterations"], np.arange(1, 20 + 1)
        )
        np.testing.assert_almost_equal(
            clr.history["lr"],
            [
                0.001,
                0.0019,
                0.0028,
                0.0037,
                0.0046,
                0.0055,
                0.0064,
                0.0073,
                0.0082,
                0.0091,
                0.01,
                0.0091,
                0.0082,
                0.0073,
                0.0064,
                0.0055,
                0.0046,
                0.0037,
                0.0028,
                0.0019,
            ],
        )

    @pytest.mark.dependency(
        depends=["test_make_npy_xloader", "test_clr_triangular_train"]
    )
    def test_clr_triangular_train_iter(self):
        """Test CLR triangular policy during training using a data iterator."""
        dir_ = tempfile.TemporaryDirectory()

        # Write dataset to disk and create data loader
        _ = utils.write_dataset(
            dir_.name + "/train",
            self._X,
            self._y,
            overwrite=False,
            augment=False,
        )
        data = utils.NNTools.build_loader(
            dir_.name + "/train", batch_size=10, shuffle=True
        )

        clr = utils.NNTools.CyclicalLearningRate(
            base_lr=0.001,
            max_lr=0.01,
            step_size=10,
            mode="triangular",
        )

        model = utils.NNTools.train(
            model=self._make_model(),
            data=data,
            fit_kwargs={
                "epochs": 20,
                "shuffle": True,
                "callbacks": [clr],
            },
            model_filename=None,
            history_filename=None,
            wandb_project=None,
        )

        # Should go back to min
        np.testing.assert_almost_equal(model.optimizer.lr, 0.001)
        np.testing.assert_almost_equal(
            clr.history["iterations"], np.arange(1, 20 + 1)
        )
        np.testing.assert_almost_equal(
            clr.history["lr"],
            [
                0.001,
                0.0019,
                0.0028,
                0.0037,
                0.0046,
                0.0055,
                0.0064,
                0.0073,
                0.0082,
                0.0091,
                0.01,
                0.0091,
                0.0082,
                0.0073,
                0.0064,
                0.0055,
                0.0046,
                0.0037,
                0.0028,
                0.0019,
            ],
        )

        dir_.cleanup()

    def test_clr_triangular2_train(self):
        """Test CLR triangular2 policy during training."""
        clr = utils.NNTools.CyclicalLearningRate(
            base_lr=0.001,
            max_lr=0.01,
            step_size=10,
            mode="triangular2",
        )

        model = utils.NNTools.train(
            model=self._make_model(),
            data=(self._X, self._y),
            fit_kwargs={
                "batch_size": 50,
                "epochs": 40,
                "validation_split": 0.2,
                "shuffle": True,
                "callbacks": [clr],
            },
            model_filename=None,
            history_filename=None,
            wandb_project=None,
        )

        # Should go back to min
        np.testing.assert_almost_equal(model.optimizer.lr, 0.001)
        np.testing.assert_almost_equal(
            clr.history["iterations"], np.arange(1, 40 + 1)
        )
        np.testing.assert_almost_equal(
            clr.history["lr"],
            [
                0.001,
                0.0019,
                0.0028,
                0.0037,
                0.0046,
                0.0055,
                0.0064,
                0.0073,
                0.0082,
                0.0091,
                0.01,
                0.0091,
                0.0082,
                0.0073,
                0.0064,
                0.0055,
                0.0046,
                0.0037,
                0.0028,
                0.0019,
                0.001,
                0.00145,
                0.0019,
                0.00235,
                0.0028,
                0.00325,
                0.0037,
                0.00415,
                0.0046,
                0.00505,
                0.0055,
                0.00505,
                0.0046,
                0.00415,
                0.0037,
                0.00325,
                0.0028,
                0.00235,
                0.0019,
                0.00145,
            ],
        )

    def test_clr_exp_range_train(self):
        """Test CLR exp_range policy during training."""
        clr = utils.NNTools.CyclicalLearningRate(
            base_lr=0.001,
            max_lr=0.01,
            step_size=10,
            mode="exp_range",
            gamma=0.9,
        )

        model = utils.NNTools.train(
            model=self._make_model(),
            data=(self._X, self._y),
            fit_kwargs={
                "batch_size": 50,
                "epochs": 40,
                "validation_split": 0.2,
                "shuffle": True,
                "callbacks": [clr],
            },
            model_filename=None,
            history_filename=None,
            wandb_project=None,
        )

        # Should go back to min
        np.testing.assert_almost_equal(model.optimizer.lr, 0.001)
        np.testing.assert_almost_equal(
            clr.history["iterations"], np.arange(1, 40 + 1)
        )
        np.testing.assert_almost_equal(
            clr.history["lr"],
            [
                0.001,
                0.00181,
                0.002458,
                0.0029683,
                0.00336196,
                0.003657205,
                0.0038697815,
                0.0040132706,
                0.004099364,
                0.004138106,
                0.004138106,
                0.0035418659,
                0.0030334927,
                0.0026013756,
                0.0022353467,
                0.0019265101,
                0.0016670873,
                0.0014502839,
                0.0012701703,
                0.0011215766,
                0.001,
                0.001098477,
                0.0011772588,
                0.0012392993,
                0.0012871592,
                0.0013230541,
                0.0013488984,
                0.0013663433,
                0.0013768103,
                0.0013815204,
                0.0013815204,
                0.0013090315,
                0.0012472252,
                0.0011946899,
                0.0011501893,
                0.001112642,
                0.0010811023,
                0.001054744,
                0.0010328464,
                0.0010147808,
            ],
        )
