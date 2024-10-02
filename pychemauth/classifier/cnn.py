"""
Tools for using Convolutional Neural Networks (CNNs).

Author: nam
"""
import keras
import copy

import tensorflow as tf

from tensorflow.keras import backend as K

from typing import Any, ClassVar


class CNNFactory:
    """Factory function to create 2D CNNs for classification."""
    name: ClassVar[str]
    input_size: ClassVar[tuple[int, int, int]]
    pixel_range: ClassVar[tuple[float, float]]
    n_classes: ClassVar[int]
    cam: ClassVar[bool]
    dropout: ClassVar[float]
    kwargs: ClassVar[dict[str, Any]]

    def __init__(
        self,
        name: str,
        input_size: tuple[int, int, int],
        pixel_range: tuple[float, float],
        n_classes: int,
        cam: bool = True,
        dropout: float = 0.0,
        kwargs: dict[str, Any] = {},
    ) -> None:
        """
        Instantiate the factory.

        Parameters
        ----------
        name : str
            Name of the keras.Applications model to use as the base.

        input_size : tuple(int, int, int)
            Size of the image being classified.  Should be in channel-last format; e.g., (N, N, 1) or (N, N, 3).

        pixel_range : tuple(float, float)
            Tuple of (min, max) values inclusive that will be encountered in the images.  For 2D "imaged" signals this might be (-1.0, 1.0) or (0.0, 1.0), whereas for conventional images this is usually (0, 255). All base models expect data in the [0, 255] range so inputs are scaled to match this; these are subsequently preprocessed in different ways by different base models, but all start from this input so range so we automatically scale this to match.

        n_classes : int
            Number of classes to learn.

        cam : bool, optional(default=True)
            Whether to use a "CAM" architecture or not.
            CAM refers to BASE -> [GAP -> Dropout (optional) -> Dense].  Otherwise BASE -> [Flatten -> Dropout (optional) -> Dense] is used.
            CAM architectures can be explained using HiResCAM or GradCAM, while the latter requires HiResCAM.

        dropout : float, optional(default=0.0)
            If positive, a Dropout layer is added to the architecture behind the final Dense layer with this dropout rate.

        kwargs : dict(str, object), optional(default={})
            Optional keyword arguments to the model. Important keywords will be overriden internally, but some models have additional tuning parameters (e.g., MobileNet `alpha`) that can be adjusted.

        Notes
        -----
        Some base models have minimum image sizes; e.g., Xception requires images to be atleast (71, 71, 3).  An appropriate python exception should be generated if you try to use an image which is too small; if you receive such an error either increase the size of your image, if possible, or select a different model.

        Base models also require different sorts of preprocessing.  Read the documentation on Keras' website for more information.  However, the necessary preprocessing is provided in Keras and automatically added to the model behind the scenes so you do not need to do anything about this.
        """
        self.set_params(
            **{
                "name": name,
                "input_size": input_size,
                "pixel_range": pixel_range,
                "n_classes": n_classes,
                "cam": cam,
                "dropout": dropout,
                "kwargs": kwargs,
            }
        )

    def set_params(self, **parameters: Any) -> "CNNFactory":
        """Set the parameters."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the parameters."""
        return {
            "name": self.name,
            "input_size": self.input_size,
            "pixel_range": self.pixel_range,
            "n_classes": self.n_classes,
            "cam": self.cam,
            "dropout": self.dropout,
            "kwargs": self.kwargs,
        }

    def _validate_inputs(self) -> None:
        """Check the input has the right size and format, etc."""
        if len(self.input_size) != 3:
            raise ValueError(
                "input should be a 3 dimensional, channels-last tensor"
            )
        if self.input_size[0] != self.input_size[1]:
            raise ValueError("input should be square")

        if self.input_size[0] <= 0:
            raise ValueError("Invalid input size")

        if self.n_classes < 2:
            raise ValueError("n_classes should be at least 2")

        if self.dropout < 0:
            raise ValueError("dropout rate should be non-negative")

        if len(self.pixel_range) != 2:
            raise ValueError("pixel_range should be a tuple of length 2")

    def _model_lookup(self, name: str) -> tuple[Any, Any]:
        """Lookup a model and its preprocessor in keras.applications."""
        name = name.lower()

        if name == "xception":
            base_model, preprocessor = (
                tf.keras.applications.Xception,
                tf.keras.applications.xception.preprocess_input,  # Expects data in [0, 255] -> [-1, +1]
            )
        elif name == "efficientnetb0":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetB0,
                tf.keras.applications.efficientnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetb1":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetB1,
                tf.keras.applications.efficientnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetb2":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetB2,
                tf.keras.applications.efficientnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetb3":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetB3,
                tf.keras.applications.efficientnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetb4":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetB4,
                tf.keras.applications.efficientnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetb5":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetB5,
                tf.keras.applications.efficientnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetb6":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetB6,
                tf.keras.applications.efficientnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetb7":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetB7,
                tf.keras.applications.efficientnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetv2b0":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetV2B0,
                tf.keras.applications.efficientnet_v2.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetv2b1":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetV2B1,
                tf.keras.applications.efficientnet_v2.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetv2b2":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetV2B2,
                tf.keras.applications.efficientnet_v2.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetv2b3":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetV2B3,
                tf.keras.applications.efficientnet_v2.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetv2s":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetV2S,
                tf.keras.applications.efficientnet_v2.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetv2m":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetV2M,
                tf.keras.applications.efficientnet_v2.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "efficientnetv2l":
            base_model, preprocessor = (
                tf.keras.applications.EfficientNetV2L,
                tf.keras.applications.efficientnet_v2.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "convnexttiny":
            base_model, preprocessor = (
                tf.keras.applications.ConvNeXtTiny,
                tf.keras.applications.convnext.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "convnextsmall":
            base_model, preprocessor = (
                tf.keras.applications.ConvNeXtSmall,
                tf.keras.applications.convnext.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "convnextbase":
            base_model, preprocessor = (
                tf.keras.applications.ConvNeXtBase,
                tf.keras.applications.convnext.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "convnextlarge":
            base_model, preprocessor = (
                tf.keras.applications.ConvNeXtLarge,
                tf.keras.applications.convnext.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "convnextxlarge":
            base_model, preprocessor = (
                tf.keras.applications.ConvNeXtXLarge,
                tf.keras.applications.convnext.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "vgg16":
            base_model, preprocessor = (
                tf.keras.applications.VGG16,
                tf.keras.applications.vgg16.preprocess_input,  # Expects data in [0, 255]; RGB -> BGR and centers based on imagenet
            )
        elif name == "vgg19":
            base_model, preprocessor = (
                tf.keras.applications.VGG19,
                tf.keras.applications.vgg19.preprocess_input,  # Expects data in [0, 255]; RGB -> BGR and centers based on imagenet
            )
        elif name == "resnet50":
            base_model, preprocessor = (
                tf.keras.applications.ResNet50,
                tf.keras.applications.resnet.preprocess_input,  # Expects data in [0, 255]; RGB -> BGR and centers based on imagenet
            )
        elif name == "resnet101":
            base_model, preprocessor = (
                tf.keras.applications.ResNet101,
                tf.keras.applications.resnet.preprocess_input,  # Expects data in [0, 255]; RGB -> BGR and centers based on imagenet
            )
        elif name == "resnet152":
            base_model, preprocessor = (
                tf.keras.applications.ResNet152,
                tf.keras.applications.resnet.preprocess_input,  # Expects data in [0, 255]; RGB -> BGR and centers based on imagenet
            )
        elif name == "resnet50v2":
            base_model, preprocessor = (
                tf.keras.applications.ResNet50V2,
                tf.keras.applications.resnet_v2.preprocess_input,  # Expects data in [0, 255] -> [-1, +1]
            )
        elif name == "resnet101v2":
            base_model, preprocessor = (
                tf.keras.applications.ResNet101V2,
                tf.keras.applications.resnet_v2.preprocess_input,  # Expects data in [0, 255] -> [-1, +1]
            )
        elif name == "resnet152v2":
            base_model, preprocessor = (
                tf.keras.applications.ResNet152V2,
                tf.keras.applications.resnet_v2.preprocess_input,  # Expects data in [0, 255] -> [-1, +1]
            )
        elif name == "resnetrs50":
            base_model, preprocessor = (
                tf.keras.applications.ResNetRS50,
                tf.keras.applications.resnet_rs.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "resnetrs101":
            base_model, preprocessor = (
                tf.keras.applications.ResNetRS101,
                tf.keras.applications.resnet_rs.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "resnetrs152":
            base_model, preprocessor = (
                tf.keras.applications.ResNetRS152,
                tf.keras.applications.resnet_rs.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "resnetrs200":
            base_model, preprocessor = (
                tf.keras.applications.ResNetRS200,
                tf.keras.applications.resnet_rs.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "resnetrs270":
            base_model, preprocessor = (
                tf.keras.applications.ResNetRS270,
                tf.keras.applications.resnet_rs.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "resnetrs350":
            base_model, preprocessor = (
                tf.keras.applications.ResNetRS350,
                tf.keras.applications.resnet_rs.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "resnetrs420":
            base_model, preprocessor = (
                tf.keras.applications.ResNetRS420,
                tf.keras.applications.resnet_rs.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx002":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX002,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx004":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX004,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx006":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX006,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx008":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX008,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx016":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX016,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx032":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX032,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx040":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX040,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx064":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX064,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx080":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX080,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx120":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX120,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx160":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX160,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnetx320":
            base_model, preprocessor = (
                tf.keras.applications.RegNetX320,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety002":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY002,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety004":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY004,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety006":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY006,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety008":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY008,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety016":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY016,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety032":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY032,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety040":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY040,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety064":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY064,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety080":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY080,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety120":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY120,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety160":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY160,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "regnety320":
            base_model, preprocessor = (
                tf.keras.applications.RegNetY320,
                tf.keras.applications.regnet.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "mobilenet":
            base_model, preprocessor = (
                tf.keras.applications.MobileNet,
                tf.keras.applications.mobilenet.preprocess_input,  # Expects data in [0, 255] -> [-1, +1]
            )
        elif name == "mobilenetv2":
            base_model, preprocessor = (
                tf.keras.applications.MobileNetV2,
                tf.keras.applications.mobilenet_v2.preprocess_input,  # Expects data in [0, 255] -> [-1, +1]
            )
        elif name == "mobilenetv3small":
            base_model, preprocessor = (
                tf.keras.applications.MobileNetV3Small,
                tf.keras.applications.mobilenet_v3.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "mobilenetv3large":
            base_model, preprocessor = (
                tf.keras.applications.MobileNetV3Large,
                tf.keras.applications.mobilenet_v3.preprocess_input,  # Expects data in [0, 255] - passthrough
            )
        elif name == "densenet121":
            base_model, preprocessor = (
                tf.keras.applications.DenseNet121,
                tf.keras.applications.densenet.preprocess_input,  # Expects data in [0, 255] -> [0, 1] then autoscales with predefined scales
            )
        elif name == "densenet169":
            base_model, preprocessor = (
                tf.keras.applications.DenseNet169,
                tf.keras.applications.densenet.preprocess_input,  # Expects data in [0, 255] -> [0, 1] then autoscales with predefined scales
            )
        elif name == "densenet201":
            base_model, preprocessor = (
                tf.keras.applications.DenseNet201,
                tf.keras.applications.densenet.preprocess_input,  # Expects data in [0, 255] -> [0, 1] then autoscales with predefined scales
            )
        elif name == "nasnetlarge":
            base_model, preprocessor = (
                tf.keras.applications.NASNetLarge,
                tf.keras.applications.nasnet.preprocess_input,  # Expected data in [0, 255] -> [-1, +1]
            )
        elif name == "nasnetmobile":
            base_model, preprocessor = (
                tf.keras.applications.NASNetMobile,
                tf.keras.applications.nasnet.preprocess_input,  # Expected data in [0, 255] -> [-1, +1]
            )
        elif name == "inceptionv3":
            base_model, preprocessor = (
                tf.keras.applications.InceptionV3,
                tf.keras.applications.inception_v3.preprocess_input,  # Expected data in [0, 255] -> [-1, +1]
            )
        elif name == "inceptionresnetv2":
            base_model, preprocessor = (
                tf.keras.applications.InceptionResNetV2,
                tf.keras.applications.inception_resnet_v2.preprocess_input,  # Expected data in [0, 255] -> [-1, +1]
            )
        else:
            raise ValueError(
                f"Could not find a pre-trained model in keras.applications named {name}"
            )

        return base_model, preprocessor

    def build(self) -> keras.Model:
        """
        Build the model using weights learned from ImageNet.

        Returns
        -------
        model : keras.Model
            Uncompiled Keras model to be used for image classification.

        Notes
        -----
        Models are loaded using weights trained on the ImageNet dataset.

        Please refer to Keras' documentation for appropriate citations, documentation, and other details about each base model.
        """
        self._validate_inputs()

        # These keywords are required to be these values - do not allow user to modify them.
        # Keras seems to follow a convention where if "include_preprocessing" is a valid input then the
        # preprocessing function is just a passthrough.  Thus, we should try to instantiate the model
        # but if it fails from this keyword, add the preprocessor manually.  If the argument is valid
        # it is still ok to manually add this function since it is a passthrough.
        base_kwargs = {
            "include_top": False,
            "weights": "imagenet",
            "input_shape": (
                self.input_size[0],
                self.input_size[1],
                3,
            ),  # Force channels-last format
            "pooling": None,
            "include_preprocessing": True,  # Will scale all inputs to [0, 255] so preprocessors will all work
        }

        # Remove any keywords that would conflict with above
        additional_kwargs = copy.deepcopy(self.kwargs)
        for key in base_kwargs.keys():
            additional_kwargs.pop(key, [])
        base_kwargs.update(additional_kwargs)

        base_model_, preprocessor = self._model_lookup(self.name)
        try:
            base_model = base_model_(**base_kwargs)
        except (TypeError, ValueError) as e:
            # Some models do not have an 'include_preprocessing' kwarg, so if not valid remove and try again
            if "include_preprocessing" in str(e):
                base_kwargs.pop("include_preprocessing")
                base_model = base_model_(**base_kwargs)
            else:
                raise Exception(e)
        base_model.trainable = False

        # Build model
        input_ = keras.layers.Input(shape=self.input_size)

        # Always rescale inputs to [0, 255]
        px_min, px_max = sorted(self.pixel_range)
        s_ = 255.0 / (px_max - px_min)
        rescaler = keras.layers.Rescaling(scale=s_, offset=-px_min * s_)

        if self.input_size[2] == 1:
            # https://discuss.pytorch.org/t/best-way-to-deal-with-1-channel-images/26699
            # Much better for memory than copying the channel in the input
            connector = keras.layers.Conv2D(  # "Connector" layer
                filters=3,  # Convert (N, N, 1) --> (N, N, 3)
                kernel_size=1,  # Focus on 1 pixel at a time
                padding="same",
                strides=1,
                activation=None,  # This just multiplies the input pixel by some weight and applies no activation function
                use_bias=False,  # No bias to shift input_
                kernel_initializer=tf.keras.initializers.Ones(),  # Set all weights to "1" so input_ is essentially just copied
                data_format="channels_last",  # Should serve as a basic check that input is provided in this format
                trainable=False,
            )
            x = connector(input_)
            x = rescaler(x)
            x = preprocessor(
                x, data_format="channels_last"
            )  # Prepare input for model
        elif self.input_size[2] == 3:
            x = rescaler(input_)
            x = preprocessor(
                x, data_format="channels_last"
            )  # Prepare input for model
        else:
            raise Exception(
                f"Unexpected input_size ({self.input_size}). This should end in either 1 or 3 channels."
            )
        x = base_model(x, training=False)

        if self.cam:
            # BASE -> [GAP -> Dropout (optional) -> Dense]
            x = keras.layers.GlobalAveragePooling2D()(x)
            if self.dropout > 0:
                x = keras.layers.Dropout(self.dropout)(x)
        else:
            # BASE -> [Flatten -> Dropout (optional) -> Dense]
            x = keras.layers.Flatten()(x)
            if self.dropout > 0.0:
                x = keras.layers.Dropout(self.dropout)(x)

        output = keras.layers.Dense(
            self.n_classes,
            activation="softmax"
            if self.n_classes > 2
            else "sigmoid",  # Already checked self.n_classes >= 2
        )(x)

        return keras.Model(inputs=[input_], outputs=[output])
