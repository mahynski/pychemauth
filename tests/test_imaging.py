"""
Unittests for tools associated with creating CNN models for working with imaging.

author: nam
"""
import unittest
import keras
import pytest

import numpy as np

from pychemauth.analysis import explain
from pychemauth.classifier.cnn import CNNFactory

from keras import layers


def _model_factory(idx):
    """Create various models for testing."""
    num_classes = 3  # Arbitrary choice > 2

    def _make_base():
        # Basic functional base that could be used for transfer learning
        input_ = keras.layers.Input(shape=(224, 224, 1))
        x = keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            padding="same",
            strides=1,
            activation=keras.activations.tanh,
            use_bias=False,
        )(input_)
        base_model = keras.applications.MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
            pooling=None,
            include_preprocessing=True,
        )
        base_model.trainable = False
        x = base_model(x, training=False)

        return input_, x

    if idx == 0:
        # Basic manual, non CAM architecture with dropout
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 1:
        # Basic manual, non CAM architecture without dropout
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 2:
        # Basic manual, non CAM architecture for binary
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
    elif idx == 3:
        # Basic manual, non CAM architecture with manual activation
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(num_classes, activation=None),
                layers.Activation("softmax"),
            ]
        )
    elif idx == 4:
        # Basic manual, CAM architecture with dropout
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 5:
        # Basic manual, CAM architecture without dropout
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.GlobalAveragePooling2D(),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 6:
        # Basic manual, CAM architecture for binary
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.GlobalAveragePooling2D(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
    elif idx == 7:
        # Basic manual, CAM architecture with manual activation
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.GlobalAveragePooling2D(),
                layers.Dense(num_classes, activation=None),
                layers.Activation("softmax"),
            ]
        )
    elif idx == 8:
        # Functional Base with dropout
        input_, x = _make_base()
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        output = keras.layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs=[input_], outputs=[output])
    elif idx == 9:
        # Functional Base without dropout
        input_, x = _make_base()
        x = keras.layers.GlobalAveragePooling2D()(x)
        output = keras.layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs=[input_], outputs=[output])
    elif idx == 10:
        # Functional Base for binary
        input_, x = _make_base()
        x = keras.layers.GlobalAveragePooling2D()(x)
        output = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs=[input_], outputs=[output])
    elif idx == 11:
        # Functional Base with manual activation
        input_, x = _make_base()
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(num_classes, activation=None)(x)
        output = keras.layers.Activation("softmax")(x)
        model = keras.Model(inputs=[input_], outputs=[output])
    elif idx == 12:
        # 1D: Basic manual, non CAM architecture with dropout
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 13:
        # 1D: Basic manual, non CAM architecture without dropout
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Flatten(),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 14:
        # 1D: Basic manual, non CAM architecture for binary
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Flatten(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
    elif idx == 15:
        # 1D: Basic manual, non CAM architecture with manual activation
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Flatten(),
                layers.Dense(num_classes, activation=None),
                layers.Activation("softmax"),
            ]
        )
    elif idx == 16:
        # 1D: Basic manual, CAM architecture with dropout
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.GlobalAveragePooling1D(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 17:
        # 1D: Basic manual, CAM architecture without dropout
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.GlobalAveragePooling1D(),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 18:
        # 1D: Basic manual, CAM architecture for binary
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.GlobalAveragePooling1D(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
    elif idx == 19:
        # 1D: Basic manual, CAM architecture with manual activation
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.GlobalAveragePooling1D(),
                layers.Dense(num_classes, activation=None),
                layers.Activation("softmax"),
            ]
        )
    elif idx == 20:
        # Basic manual, non CAM architecture with dropout - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 21:
        # Basic manual, non CAM architecture without dropout  - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 22:
        # Basic manual, non CAM architecture for binary - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
    elif idx == 23:
        # Basic manual, non CAM architecture with manual activation - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dense(num_classes, activation=None),
                layers.Activation("softmax"),
            ]
        )
    elif idx == 24:
        # Basic manual, CAM architecture with dropout - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 25:
        # Basic manual, CAM architecture without dropout - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.GlobalAveragePooling2D(),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 26:
        # Basic manual, CAM architecture for binary - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.GlobalAveragePooling2D(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
    elif idx == 27:
        # Basic manual, CAM architecture with manual activation - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.GlobalAveragePooling2D(),
                layers.Dense(num_classes, activation=None),
                layers.Activation("softmax"),
            ]
        )
    elif idx == 28:
        # 1D: Basic manual, non CAM architecture with dropout - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 29:
        # 1D: Basic manual, non CAM architecture without dropout - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.Flatten(),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 30:
        # 1D: Basic manual, non CAM architecture for binary - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.Flatten(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
    elif idx == 31:
        # 1D: Basic manual, non CAM architecture with manual activation - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.Flatten(),
                layers.Dense(num_classes, activation=None),
                layers.Activation("softmax"),
            ]
        )
    elif idx == 32:
        # 1D: Basic manual, CAM architecture with dropout - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.GlobalAveragePooling1D(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 33:
        # 1D: Basic manual, CAM architecture without dropout - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.GlobalAveragePooling1D(),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 34:
        # 1D: Basic manual, CAM architecture for binary - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.GlobalAveragePooling1D(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
    elif idx == 35:
        # 1D: Basic manual, CAM architecture with manual activation - no final max pool
        model = keras.Sequential(
            [
                keras.Input((100, 3)),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.GlobalAveragePooling1D(),
                layers.Dense(num_classes, activation=None),
                layers.Activation("softmax"),
            ]
        )
    elif idx == 36:
        # Dense net (not conv base)
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Dense(10, activation="relu"),
                layers.Dense(100, activation="relu"),
                layers.Dense(10, activation="relu"),
                layers.Flatten(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
    elif idx == 37:
        # Multiple dense layers at the end
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(10, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 38:
        # Multiple dense layers at the end, another example
        model = keras.Sequential(
            [
                keras.Input((100, 100, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(10, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif idx == 39:
        # Test the CNNFactory with CAM
        cnn_builder = CNNFactory(
            name='mobilenet',
            input_size=(2631, 2631, 1),
            n_classes=10,
            dim=2,
            cam=True,
            dropout=0.0
        ) 
        model = cnn_builder.build()
    elif idx == 40:
        # Test the CNNFactory with CAM
        cnn_builder = CNNFactory(
            name='xception',
            input_size=(2631, 2631, 1),
            n_classes=10,
            dim=2,
            cam=True,
            dropout=0.0
        ) 
        model = cnn_builder.build()
    elif idx == 41:
        # Test the CNNFactory with CAM + dropout
        cnn_builder = CNNFactory(
            name='mobilenet',
            input_size=(2631, 2631, 1),
            n_classes=10,
            dim=2,
            cam=True,
            dropout=0.2
        ) 
        model = cnn_builder.build()
    elif idx == 42:
        # Test the CNNFactory with CAM + dropout
        cnn_builder = CNNFactory(
            name='xception',
            input_size=(2631, 2631, 1),
            n_classes=10,
            dim=2,
            cam=True,
            dropout=0.2
        ) 
        model = cnn_builder.build()
    elif idx == 43:
        # Test the CNNFactory with CAM
        cnn_builder = CNNFactory(
            name='mobilenet',
            input_size=(2631, 2631, 1),
            n_classes=10,
            dim=2,
            cam=False,
            dropout=0.0
        ) 
        model = cnn_builder.build()
    elif idx == 44:
        # Test the CNNFactory with CAM
        cnn_builder = CNNFactory(
            name='xception',
            input_size=(2631, 2631, 1),
            n_classes=10,
            dim=2,
            cam=False,
            dropout=0.0
        ) 
        model = cnn_builder.build()
    elif idx == 45:
        # Test the CNNFactory with CAM + dropout
        cnn_builder = CNNFactory(
            name='mobilenet',
            input_size=(2631, 2631, 1),
            n_classes=10,
            dim=2,
            cam=False,
            dropout=0.2
        ) 
        model = cnn_builder.build()
    elif idx == 46:
        # Test the CNNFactory with CAM + dropout
        cnn_builder = CNNFactory(
            name='xception',
            input_size=(2631, 2631, 1),
            n_classes=10,
            dim=2,
            cam=False,
            dropout=0.2
        ) 
        model = cnn_builder.build()
    else:
        raise Exception("Unknown model idx")

    return model


class CheckCAMArchitecture(unittest.TestCase):
    """Check the CNN models follow an approved architecture for CAM explanations."""

    @classmethod
    def setUpClass(self):
        """Store the correct answers for all models."""
        self.correct_answers = {
            0: (True, -2, 2, "max_pooling2d_1", "output"),
            1: (True, -1, 2, "max_pooling2d_3", "output"),
            2: (True, -1, 2, "max_pooling2d_5", "output"),
            3: (True, -2, 2, "max_pooling2d_7", "output"),
            4: (True, -2, 2, "max_pooling2d_9", "output"),
            5: (True, -1, 2, "max_pooling2d_11", "output"),
            6: (True, -1, 2, "max_pooling2d_13", "output"),
            7: (True, -2, 2, "max_pooling2d_15", "output"),
            8: (True, -2, 2, "global_average_pooling2d_4", "input"),
            9: (True, -1, 2, "global_average_pooling2d_5", "input"),
            10: (True, -1, 2, "global_average_pooling2d_6", "input"),
            11: (True, -2, 2, "global_average_pooling2d_7", "input"),
            12: (True, -2, 1, "max_pooling1d_1", "output"),
            13: (True, -1, 1, "max_pooling1d_3", "output"),
            14: (True, -1, 1, "max_pooling1d_5", "output"),
            15: (True, -2, 1, "max_pooling1d_7", "output"),
            16: (True, -2, 1, "max_pooling1d_9", "output"),
            17: (True, -1, 1, "max_pooling1d_11", "output"),
            18: (True, -1, 1, "max_pooling1d_13", "output"),
            19: (True, -2, 1, "max_pooling1d_15", "output"),
            20: (True, -2, 2, "conv2d_21", "output"),
            21: (True, -1, 2, "conv2d_23", "output"),
            22: (True, -1, 2, "conv2d_25", "output"),
            23: (True, -2, 2, "conv2d_27", "output"),
            24: (True, -2, 2, "conv2d_29", "output"),
            25: (True, -1, 2, "conv2d_31", "output"),
            26: (True, -1, 2, "conv2d_33", "output"),
            27: (True, -2, 2, "conv2d_35", "output"),
            28: (True, -2, 1, "conv1d_17", "output"),
            29: (True, -1, 1, "conv1d_19", "output"),
            30: (True, -1, 1, "conv1d_21", "output"),
            31: (True, -2, 1, "conv1d_23", "output"),
            32: (True, -2, 1, "conv1d_25", "output"),
            33: (True, -1, 1, "conv1d_27", "output"),
            34: (True, -1, 1, "conv1d_29", "output"),
            35: (True, -2, 1, "conv1d_31", "output"),
            36: (False, 0, 0, "", ""),
            37: (False, 0, 0, "", ""),
            38: (False, 0, 0, "", ""),
            39: (True, -1, 2, 'global_average_pooling2d_28', 'input'),
            40: (True, -1, 2, 'global_average_pooling2d_29', 'input'),
            41: (True, -2, 2, 'global_average_pooling2d_30', 'input'),
            42: (True, -2, 2, 'global_average_pooling2d_31', 'input'),
            43: (True, -1, 2, 'flatten_42', 'input'),
            44: (True, -1, 2, 'flatten_43', 'input'),
            45: (True, -2, 2, 'flatten_44', 'input'),
            46: (True, -2, 2, 'flatten_45', 'input'),
        }

    def test_hires(self):
        """Test the models for HiResCAM."""

        def _renumber(answer, incr):
            # Keras automatically numbers layers _X behind the scenes. This is to compensate.
            prefix = answer[3].split("_")[:-1]
            result = "_".join(
                prefix + [str(incr)]
            )  # int(answer[3].split('_')[-1]) +
            return (answer[0], answer[1], answer[2], result, answer[4])

        def _extract(layer_name):
            return int(layer_name.split("_")[-1])

        for model_idx in sorted(self.correct_answers.keys()):
            res = explain.check_cam_model_architecture(
                _model_factory(model_idx),
                conv_layer_name=None,
                mode=None,
                style="hires",
            )

            # This assumes the layer name for these idx are correct, but at least it checks all the others
            if model_idx in [0, 8, 12, 20, 28]:
                start = model_idx
                incr = 2 if model_idx != 8 else 1
                base = _extract(res[3])

            # Renumber the layer if necessary using the start of a "chunk" as reference since this is run in a loop
            correct = (
                _renumber(
                    self.correct_answers[model_idx],
                    base + (model_idx - start) * incr,
                )
                if model_idx <= 35
                else self.correct_answers[model_idx]
            )
            np.testing.assert_equal(res, correct)

    def test_grad(self):
        """Test the models for GradCAM."""
        for model_idx in self.correct_answers.keys():
            if model_idx in [
                0,
                1,
                2,
                3,
                12,
                13,
                14,
                15,
                20,
                21,
                22,
                23,
                28,
                29,
                30,
                31,
                43,
                44,
                45,
                46
            ]:  # Non-CAM architectures should not allow GradCAM
                with pytest.raises(
                    Exception,
                    match="Cannot safely explain this model with GradCAM; use HiResCAM instead.",
                ):
                    explain.check_cam_model_architecture(
                        _model_factory(model_idx),
                        conv_layer_name=None,
                        mode=None,
                        style="grad",
                    )
            else:  # CAM architectures should be fine with GradCAM
                try:
                    explain.check_cam_model_architecture(
                        _model_factory(model_idx),
                        conv_layer_name=None,
                        mode=None,
                        style="grad",
                    )
                except Exception as e:
                    raise Exception(f"Unexpected exception : {e}")
