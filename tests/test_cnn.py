"""
Unittests for Convolutional Neural Networks (CNNs).

author: nam
"""
import unittest
import tqdm
import pytest

import numpy as np

from pyts.image import GramianAngularField
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from pychemauth.datasets import load_pgaa
from pychemauth.classifier.cnn import CNNFactory
from pychemauth.utils import NNTools


class TestCNNFactory(unittest.TestCase):
    """Test CNNFactory models."""

    @classmethod
    def setUpClass(self):
        """Image some example data for testing."""
        limit = 2631  # Cutoff at E ~ 7650 keV

        # Clip and renormalize X
        self._X, self._y = load_pgaa(return_X_y=True)
        self._X = self._X[:, :limit]
        self._X = (self._X.T / np.sum(self._X, axis=1)).T
        self._X = np.log(np.clip(self._X, a_min=1.0e-7, a_max=None))

        # Use sklearn to get a subsample with a consistent fraction of different classes
        _, self._X, _, self._y = train_test_split(
            self._X,
            self._y,
            test_size=0.1,
            random_state=42,
            shuffle=True,
            stratify=self._y,
        )

        # Perform imaging
        X_gaf = np.zeros(
            shape=(
                self._X.shape[0],
                self._X.shape[1],
                self._X.shape[1],
            )
        )
        for i in range(self._X.shape[0]):
            gaf = GramianAngularField(method="difference")
            X_gaf[i] = gaf.fit_transform(self._X[i : i + 1])[0]
        self._X = np.expand_dims(
            X_gaf, axis=-1
        )  # Convert to a "single channeled" image

        self.n_classes = len(np.unique(self._y))
        self.image_size = self._X.shape[1:]
        print(self._X)

        # For sparse categorical encoding
        enc = OrdinalEncoder(dtype=int)
        self._y = enc.fit_transform(self._y.reshape(-1, 1)).reshape(-1)

    def test_base_fixed(self):
        """Test weights in the BASE of model do not change during training."""
        cnn_builder = CNNFactory(
            name="mobilenet",
            input_size=self.image_size,
            pixel_range=(-1, 1),
            n_classes=self.n_classes,
            cam=True,
            dropout=0.2,
        )

        untrained_model = cnn_builder.build()

        trained_model = NNTools.train(
            model=cnn_builder.build(),
            data=(self._X, self._y),
            fit_kwargs={
                "batch_size": 5,  # Keep low to avoid memory overflow
                "epochs": 1,  # Single epoch for testing
                "validation_split": 0.0,
                "shuffle": True,
                "callbacks": [],
            },
            model_filename=None,
            history_filename=None,
            wandb_project=None,
        )

        # Check the CNN "adapter" layer has fixed weights
        np.testing.assert_allclose(
            trained_model.layers[1].weights[0].numpy().ravel(), np.ones(3)
        )

        # Check that the layers in the BASE haven't changed
        for layer_t, layer_u in zip(
            trained_model.layers[4].weights, untrained_model.layers[4].weights
        ):
            np.testing.assert_allclose(layer_t.numpy(), layer_u.numpy())

    @pytest.mark.skip(
        reason="This takes a very long time - reenable for debugging"
    )
    def test_init(self):
        """Test all the models in the factory can be looked up and initialized."""
        for name in tqdm.tqdm(
            [
                "xception",
                "efficientnetb0",
                "efficientnetb1",
                "efficientnetb2",
                "efficientnetb3",
                "efficientnetb4",
                "efficientnetb5",
                "efficientnetb6",
                "efficientnetb7",
                "efficientnetv2b0",
                "efficientnetv2b1",
                "efficientnetv2b2",
                "efficientnetv2b3",
                "efficientnetv2s",
                "efficientnetv2m",
                "efficientnetv2l",
                "convnexttiny",
                "convnextsmall",
                "convnextbase",
                "convnextlarge",
                "convnextxlarge",
                "vgg16",
                "vgg19",
                "resnet50",
                "resnet101",
                "resnet152",
                "resnet50v2",
                "resnet101v2",
                "resnet152v2",
                "resnetrs50",
                "resnetrs101",
                "resnetrs152",
                "resnetrs200",
                "resnetrs270",
                "resnetrs350",
                "resnetrs420",
                "regnetx002",
                "regnetx004",
                "regnetx006",
                "regnetx008",
                "regnetx016",
                "regnetx032",
                "regnetx040",
                "regnetx064",
                "regnetx080",
                "regnetx120",
                "regnetx160",
                "regnetx320",
                "regnety002",
                "regnety004",
                "regnety006",
                "regnety008",
                "regnety016",
                "regnety032",
                "regnety040",
                "regnety064",
                "regnety080",
                "regnety120",
                "regnety160",
                "regnety320",
                "mobilenet",
                "mobilenetv2",
                "mobilenetv3small",
                "mobilenetv3large",
                "densenet121",
                "densenet169",
                "densenet201",
                "nasnetlarge",
                "nasnetmobile",
                "inceptionv3",
                "inceptionresnetv2",
            ]
        ):
            try:
                CNNFactory(
                    name=name,
                    input_size=self.image_size,
                    pixel_range=(-1, 1),
                    n_classes=self.n_classes,
                ).build()
            except Exception as e:
                raise Exception(f"Unable to build {name} model : {e}")
