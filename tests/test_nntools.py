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
        self._y = LabelEncoder().fit_transform(self._y)

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
