"""
Tools to explain predictions.

Author: nam
"""
import PIL
import scipy
import keras
import matplotlib

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, show

from matplotlib.collections import LineCollection


class _CAMExplainer:
    """Base class for explaining classifications of 1D or 2D (imaged) spectra with CAM methods."""

    def __init__(self, style="hires"):
        """
        Instantiate the class.

        An architecture ending in a global average pooling (GAP), followed by a
        single dense layer is recommended for the most consistent explanations. HiResCAM will be guaranteed to reflect areas that increase the most prediction's likelihood only when a single dense layer is used at the end, but a GAP is not required (and the last CNN layer is used for explanation). Grad-CAM does not carry such guarantees.

        Parameters
        ----------
        style : str
            'grad' uses Grad-CAM algorithm; 'hires' uses HiResCAM algorithm instead.  If the model has a CAM architecture
            (it ends in GAP + 1 Dense layer) these should yield identical results.

        References
        -----------
        1. HiResCAM Method: https://arxiv.org/abs/2011.08891
        2. Grad-CAM Method: https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf

        Notes
        -----
        This only supports Keras models at the moment.
        """
        self.set_params(
            **{
                "style": style.lower(),
            }
        )

    def set_params(self, **parameters):
        """Set parameters."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep=True):
        """Get parameters."""
        return {
            "style": self.style,
        }
    
    def importances(self, *args, **kwargs):
        """Compute the feature importances for a single input."""
        raise NotImplementedError
    
    def explain(self, *args, **kwargs):
        """Compute an explanation."""
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        """Visualize the explanation."""
        raise NotImplementedError

class CAM1D(_CAMExplainer):
    """Explain 1D spectra classified with CNN using CAM methods."""

    def __init__(self, style="hires"):
        """Instantiate the class."""
        super().__init__(style=style)

    def importances(self,
        spectra,
        centers,
        model,
    ):  
        """
        Compute the feature importances for a single spectra.

        The importances are bounded between [0, 1] and reflect the 
        explanation of the class predicted for this spectra.

        Parameters
        ----------
        spectra : ndarray(float, ndim=2)
            A single (N, 1) spectra.
            
        centers : ndarray(float, ndim=1)
            Location spectra were measured at in an (N,) array.

        model : keras.Model
            Model being used.

        Returns
        -------
        importances : ndarray(float, ndim=1)
            A vector of feature importances the same length as the spectra.

        Raises
        ------
        Exception
            The Keras model architecture is incorrect.

        ValueError
            The spectra is provided in the incorrect shape.
        """
        result = self.explain(
            spectra=spectra,
            centers=centers,
            model=model
        )

        return result[1].get_array().data

    def explain(self,
        spectra,
        centers,
        model,
    ):
        """
        Compute a detailed explanation for a single spectra.

        Parameters
        ----------
        spectra : ndarray(float, ndim=2)
            A single (N, 1) spectra.
            
        centers : ndarray(float, ndim=1)
            Location spectra were measured at in an (N,) array.

        model : keras.Model
            Model being used.

        Returns
        -------
        class_act_map : ndarray
            Class activation map in the range of [0, 1].

        lc : matplotlib.collections.LineCollection
            Line collection colored according to class activation map.

        preds : ndarray(float, ndim=1)
            A vector of class probabilities.

        pred_index : int
            The index of the most likely class.

        conv_layer_name : str
            Name of the last CNN layer found in the network, which is used for explanations.
        
        Raises
        ------
        Exception
            The Keras model architecture is incorrect.

        ValueError
            The spectra is provided in the incorrect shape.
        """
        if len(spectra.shape) == 2:
            if spectra.shape[1] != 1:
                raise ValueError("Spectra should have a single channel and have shape (N, 1).")
            
            # Checks that architecture is correct internally
            asymm_class_act_map, _, preds, pred_index, conv_layer_name = _make_cam(
                style=self.style,
                input=np.expand_dims(spectra, axis=0),
                model=model,
                conv_layer_name=None, # Auto-detect
            )

            lc = _cam_1d(
                centers, 
                np.squeeze(spectra), 
                asymm_class_act_map, 
                cmap="Reds", 
                interp=False
            )
        else:
            raise ValueError("Unexpected shape of spectra")

        return asymm_class_act_map, lc, preds, pred_index, conv_layer_name
    
    def visualize(self):
        # Add features like cmaps to explain as options so we can 
        # specify them for this function to get nice explanations
        return


class CAM2D(_CAMExplainer):
    """Explain 2D spectra classified with CNN using CAM methods."""

    def __init__(self, style="hires"):
        """Instantiate the class."""
        super().__init__(style=style)

    def visualize(
        self,
        centers,
        spectra,
        image,
        correct_label,
        model,
        conv_layer_name,
        mode,
        image_cmap=matplotlib.colormaps["jet"],
        spectra_cmap="Reds",
        encoder=None,
        symmetrize=False,
    ):
        """
        Visualize the explaination.

        Parameters
        ----------
        centers : ndarray(float, ndim=1)
            Location spectra were measured at in an (N,) array.

        spectra : ndarray(float, ndim=1)
            A single (N,) spectra.

        image : ndarray(float, ndim=3)
            Imaged spectra as a single (N, N, 1) tensor where the image is NxN with 1 channel (must be in 'channels_last' format).

        correct_label : str
            Correct label for the spectra (e.g., from y_train).

        model : keras.Model
            Model being used. 

        conv_layer_name : str
            Name of the Keras layer being explained. Sometimes these layers are hidden as a function so it easier to reference the
            input to the subsequent layer, rather than the output of the layer desired.  `mode` controls this.

        mode : str
            Whether to explain the output or input of the `conv_layer_name` layer. Expects either {'output', 'input'}.

        image_cmap : matplotlib.colormaps, optional(default=matplotlib.colormaps["jet"])
            Matplotlib colormap to use for the 2D heatmap.

        spectra_cmap : matplotlib.colormaps, optional(default="Reds")
            Matplotlib colormap to use for the spectra heatmap. Best if perceptually uniform.

        encoder :  sklearn.preprocessing.OrdinalEncoder, optional(default=None)
            Encodes classes (strings) as integers.

        symmetrize : bool, optional(default=False)
            Whether or not to symmetrize the class activation map before creating the heat map for explanations.

        Returns
        -------
        fig1 : matplotlib.pyplot.figure
            Figure showing the score and probabilities.

        fig2 : matplotlib.pyplot.figure
            Figure showing the colored heatmaps.
        """
        cmap_heatmap, lc, preds, pred_index = self.explain(
            centers,
            spectra,
            image,
            model,
            conv_layer_name,
            mode=mode,
            image_cmap=image_cmap,
            spectra_cmap=spectra_cmap,
            symmetrize=symmetrize,
        )

        # 1. Plot raw score output
        if encoder is not None:
            classes_ = encoder.inverse_transform(
                [[i] for i in range(len(encoder.categories_[0]))]
            ).ravel()
            if len(classes_) != len(preds):
                raise ValueError("Encoder has the wrong number of classes")
        else:
            classes_ = ["Class {}".format(i) for i in np.arange(len(preds))]

        fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 2))
        ax[0].bar(classes_, preds, color="black", width=0.9)
        ax[0].set_xticks(
            ticks=np.arange(len(classes_)), labels=classes_, rotation=90
        )
        ax[0].set_title(
            "Prediction = {}".format(
                encoder.inverse_transform([[pred_index]])[0][0]
            )
        )
        ax[0].set_ylabel("Raw Scores")

        clipped = np.clip(preds, a_min=-250, a_max=250)
        ax[1].bar(
            classes_,
            np.exp(clipped) / np.sum(np.exp(clipped)),
            color="black",
            width=0.9,
        )
        ax[1].set_xticks(
            ticks=np.arange(len(classes_)), labels=classes_, rotation=90
        )
        ax[1].set_yscale("log")
        ax[1].set_title("Actual = {}".format(correct_label))
        ax[1].set_ylabel("Softmax Probability")

        # 2. Plot explanation
        width_ratios = (2, 7, 0.4, 7, 0.4)
        height_ratios = (2, 7)
        width = 16
        height = width * sum(height_ratios) / sum(width_ratios)
        fig2 = plt.figure(figsize=(width, height))
        gs = fig2.add_gridspec(
            2,
            5,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.01,
            hspace=0.01,
        )

        ax_left = fig2.add_subplot(gs[1, 0])
        ax_left.plot(spectra, centers)
        ax_left.set_ylim([centers.min(), centers.max()])
        ax_left.set_xlim([spectra.min(), spectra.max()])
        plt.setp(ax_left.spines.values(), color="white")
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        ax_left.invert_xaxis()

        ax_top1 = fig2.add_subplot(gs[0, 1])
        ax_top1.plot(centers, spectra)
        ax_top2 = fig2.add_subplot(gs[0, 3])
        ax_top2.add_collection(lc)
        for ax in (ax_top1, ax_top2):
            ax.set_xlim([centers.min(), centers.max()])
            ax.set_ylim([spectra.min(), spectra.max()])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.setp(ax.spines.values(), color="white")

        # Plot the image
        ax_gaf = fig2.add_subplot(gs[1, 1])
        im1 = ax_gaf.imshow(image[:, :, 0], cmap=image_cmap, origin="lower")
        ax_gaf.set_xticks([])
        ax_gaf.set_yticks([])
        ax_gaf.set_title("Imaged Spectra", y=-0.09)

        # Plot class activations
        ax_cam = fig2.add_subplot(gs[1, 3])
        im2 = ax_cam.imshow(cmap_heatmap, cmap=image_cmap, origin="lower")
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])
        if self.style == "grad":
            ax_cam.set_title("Grad-CAM", y=-0.09)
        else:
            ax_cam.set_title("HiResCAM", y=-0.09)

        # Add colorbars
        cbar = fig2.colorbar(im1, cax=fig2.add_subplot(gs[1, 2]), pad=-1)
        cbar.set_ticks([])
        cbar.ax.text(
            0.5, 1.1, "1.0", ha="center", va="center"
        )  # Imaged spectra bounded b/w [0,1]
        cbar.ax.text(0.5, -1.1, "-1.0", ha="center", va="center")
        cbar = fig2.colorbar(im2, cax=fig2.add_subplot(gs[1, 4]))
        cbar.set_ticks([])
        cbar.ax.text(
            0.5, 255 + 25.5 / 2.0, "Max", ha="center", va="center"
        )  # Heatmap normalized from [0,1] then x255 to make image
        cbar.ax.text(0.5, 0 - 25.5 / 2.0, "Min", ha="center", va="center")

        return fig1, fig2

    def explain(
        self,
        centers,
        spectra,
        image,
        model,
        conv_layer_name,
        mode="output",
        image_cmap=matplotlib.colormaps["jet"],
        spectra_cmap="Reds",
        symmetrize=False,
    ):
        """
        Produce a color-coded explanation for the classification of a spectra.

        Parameters
        ----------
        centers : ndarray(float, ndim=1)
            Location spectra were measured at in an (N,) array.

        spectra : ndarray(float, ndim=1)
            A single (N,) spectra.

        image : ndarray(float, ndim=3)
            Imaged spectra as a single (N, N, 1) tensor where the image is NxN with 1 channel (must be in 'channels_last' format).

        model : keras.Model
            Model being used.

        conv_layer_name : str
            Name of the Keras layer being explained. Sometimes these layers are hidden as a function so it easier to reference the
            input to the subsequent layer, rather than the output of the layer desired.  `mode` controls this.

        mode : str, optional(default='output')
            Whether to explain the output or input of the `conv_layer_name` layer. Expects either {'output', 'input'}.

        image_cmap : matplotlib.colormaps, optional(default=matplotlib.colormaps["jet"])
            Matplotlib colormap to use for the 2D heatmap.

        spectra_cmap : matplotlib.colormaps, optional(default="Reds")
            Matplotlib colormap to use for the spectra heatmap. Best if perceptually uniform.

        symmetrize : bool, optional(default=False)
            Whether to use the symmetric CAM for 2D visualization. Symmetrization is applied before normalizing (incl. applying a ReLU).
            This means that if (i,j) helps but (j,i) hurts, then the averaging done by symmetrizing should be done before ReLU since
            the parts of the image that might decrease class probability are rounded up to 0, artificially inflating the representation
            of how much the model really interacts with the i-j peaks.

        Returns
        -------
        cmap_heatmap : ndarray(float, ndim=3)
            (N, N, 3) RBG colormap of class activations.

        lc : matplotlib.collections.LineCollection
            Line collection colored according to "condensed" class activation map.

        preds : ndarray(float, ndim=1)
            A vector of class probabilities.

        pred_index : int
            The index of the most likely class.
        """
        if len(image.shape) == 3:
            (
                cmap_heatmap,
                _,
                symm_class_act_map,
                preds,
                pred_index,
            ) = self._explain_2d(
                image=np.expand_dims(
                    image, axis=0
                ),  # Make into a batch with only 1 entry
                model=model,
                conv_layer_name=conv_layer_name,
                mode=mode,
                cmap=image_cmap,
                symmetrize=symmetrize,
            )
            lc = _cam_1d(
                centers=centers,
                spectra=spectra,  # may need to np.squeeze(spectra, axis=-1)
                # symm_class_act_map=symm_class_act_map,
                heatmap=np.mean(
                    symm_class_act_map, axis=0
                ),  # Average across "columns"
                cmap=spectra_cmap,
                interp=False,
            )
        else:
            raise ValueError("Unexpected shape of image")

        return cmap_heatmap, lc, preds, pred_index

    def _explain_2d(
        self, image, model, conv_layer_name, mode, cmap, symmetrize
    ):
        """
        Explain a 2D image.

        Parameters
        ----------
        image : ndarray(float, ndim=3)
            Imaged spectra as a single (1, N, N, C) tensor where the image is NxN with C=1 channels (must be in 'channels_last' format).

        model : keras.Model
            Model being used.

        conv_layer_name : str
            Name of the Keras layer being explained. Sometimes these layers are hidden as a function so it easier to reference the
            input to the subsequent layer, rather than the output of the layer desired.  `mode` controls this.

        mode : str
            Whether to explain the output or input of the `conv_layer_name` layer. Expects either {'output', 'input'}.

        cmap : matplotlib.colormaps
            Matplotlib colormap to use for the 2D heatmap.

        symmetrize : bool
            Whether to use the symmetric CAM for 2D visualization. Symmetrization is applied before normalizing (incl. applying a ReLU).
            This means that if (i,j) helps but (j,i) hurts, then the averaging done by symmetrizing should be done before ReLU since
            the parts of the image that might decrease class probability are rounded up to 0, artificially inflating the representation
            of how much the model really interacts with the i-j peaks.

        Returns
        -------
        cmap_heatmap : ndarray(float, ndim=3)
            (N, N, 3) RGB heatmap of class activations.

        asymm_class_act_map : ndarray(float, ndim=2)
            Asymmetric class activation map.

        symm_class_act_map : ndarray(float, ndim=2)
            Symmetric class activation map.

        preds : ndarray(float, ndim=1)
            A vector of class probabilities.

        pred_index : int
            The index of the most likely class.
        """
        asymm_class_act_map, symm_class_act_map, preds, pred_index, conv_layer_name = _make_cam(
            style=self.style,
            input=image,
            model=model,
            conv_layer_name=conv_layer_name,
            mode=mode,
        )

        cmap_heatmap = self._fit_heatmap(
            image_shape=(image.shape[2], image.shape[1]),
            class_act_map=(
                symm_class_act_map if symmetrize else asymm_class_act_map
            ),
            cmap=cmap,
        )

        return (
            cmap_heatmap,
            asymm_class_act_map,
            symm_class_act_map,
            preds,
            pred_index,
        )

    def _fit_heatmap(
        self, image_shape, class_act_map, cmap=matplotlib.colormaps["jet"]
    ):
        """
        Compute colored heatmap and rescale to size of input image.

        This is based on https://keras.io/examples/vision/grad_cam/.

        Parameters
        ----------
        image_shape : tuple(int, int)
            (Width, Height) to resize the image to.

        class_act_map : ndarray(float, ndim=2)
            2D array in the range of [0, 1].

        cmap : matplotlib.colormaps, optional(default=matplotlib.colormaps["jet"])
            Matplotlib colormap to use.

        Returns
        -------
        heatmap : ndarray(float, ndim=2)
            Linearly interpolated resized heatmap.
        """
        # Create a scaled heatmap in a range 0-255 (CAM is in range [0, 1] already)
        heatmap = np.uint8(255 * class_act_map)

        # Use colormap to colorize heatmap (use only RGB values of the colormap)
        cmap_colors = cmap(np.arange(256))[:, :3]
        cmap_heatmap = cmap_colors[heatmap]

        # Create an image with RGB colorized heatmap
        cmap_heatmap = keras.utils.array_to_img(cmap_heatmap)
        cmap_heatmap = cmap_heatmap.resize(
            image_shape,
            resample=PIL.Image.BILINEAR,  # Do linear interpolation to be consistent with interpolation of 1D
        )
        cmap_heatmap = keras.utils.img_to_array(cmap_heatmap)

        return np.uint8(cmap_heatmap)


def _make_cam(
    style, input, model, conv_layer_name=None, mode="output", pred_index=None
):
    """
    Compute activation map for a given 1D or 2D input.

    This is based on https://keras.io/examples/vision/grad_cam/.  To ensure explainability, 
    the Keras model is checked that it conforms to a "CAM" architecture (CNN -> GAP -> Dense),
    or simply terminates in a single Dense layer (CNN -> Dense).  The former can be explained
    with either 'grad' or 'hires' methods, but the latter requires 'hires'.

    Parameters
    ----------
    style : str
        Should be {'grad', 'hires'} to indicate the style of CAM.

    input : ndarray(float, ndim=3 or 4)
        When explaining a 2D image (1, N, N, C) tensor where the image is NxN with C=1 channels. This must be in 'channels_last' format. When explaining a 1D spectra (1, N, C) tensor where the image is NxN with C=1 channels.

    model : keras.Model
        Model being used. 

    conv_layer_name : str, optional(default=None)
        Name of the Keras layer being explained. Sometimes these layers are hidden as a function so it easier to reference the
        input to the subsequent layer, rather than the output of the layer desired.  `mode` controls this.
        If None, defaults to the last convolutional layer before the top.

    mode : str, optional(default='output')
        Whether to explain the output or input of the `conv_layer_name` layer. Expects either {'output', 'input'}. If `conv_layer_name`
        is None, this is ignored.

    pred_index : int, optional(default=None)
        Index of class to compute the activation map with respect to.  If `None`, the most likely class is used.

    Returns
    -------
    asymm_class_act_map : ndarray
        Asymmmetric CAM in the range of [0, 1].

    symm_class_act_map : ndarray
        Symmmetric CAM in the range of [0, 1].

    preds : ndarray(float, ndim=1)
        A vector of class probabilities.

    pred_index : int
        The index of the most likely class.

    conv_layer_name : str
        Name of the last CNN layer found in the network, which is used for explanations.

    Raises
    ------
    Exception
        The Keras model architecture is incorrect.
    """

    def _is_conv(layer, return_dim=False):
        # Check if a layer is convolutional
        for type_ in [keras.layers.Conv1D]:
            if isinstance(layer, type_):
                return (True, 1) if return_dim else True

        for type_ in [keras.layers.Conv2D]:
            if isinstance(layer, type_):
                return (True, 2) if return_dim else True

        return (False, 0) if return_dim else False

    def _is_gap(layer):
        # Check if a layer does global average pooling
        valid = [
            keras.layers.pooling.global_average_pooling1d.GlobalAveragePooling1D,
            keras.layers.pooling.global_average_pooling2d.GlobalAveragePooling2D,
        ]
        for type_ in valid:
            if isinstance(layer, type_):
                return True
        return False

    def _ends_with_dense(model):
        # Check the model ends with a dense layer
        if isinstance(model.layers[-1], keras.layers.Dense):
            # Ends with a dense layer with a softmax activation
            if model.layers[-1].activation == keras.activations.softmax:
                return True, -1
        elif isinstance(
            model.layers[-1], keras.layers.core.activation.Activation
        ):
            # Activation specified manually after a linear dense layer
            if model.layers[-1].activation == keras.activations.softmax:
                if isinstance(model.layers[-2], keras.layers.Dense):
                    if model.layers[-2].activation == keras.activations.linear:
                        return True, -2
        return False, 0

    def _check_last_cnn(dense_position, conv_layer_name, mode):
        # Check we are explaining the last CNN layer in the network
        # Assumes CNN -> GAP -> Dense
        if (
            mode == "output"
            and model.layers[dense_position - 2].name != conv_layer_name
        ): # Output of CNN
            raise Exception(
                "You are not explaining the last CNN layer in the network."
            )
        if (
            mode == "input"
            and mode.layers[dense_position - 1].name != conv_layer_name
        ): # Input to GAP layer
            raise Exception(
                "You are not explaining the last CNN layer in the network."
            )
        return True

    def _is_valid(model, conv_layer_name):
        # Check the overall model architecture
        check, dense_position = _ends_with_dense(model)
        if (
            check
            and _is_gap(model.layers[dense_position - 1])
            and _is_conv(model.layers[dense_position - 2])
        ):
            # CAM architecture - can explain with either method and should give identical results
            _, dim = _is_conv(model.layers[dense_position - 2], return_dim=True)
            if conv_layer_name is None:
                # Choose default
                conv_layer_name = model.layers[dense_position - 2].name
                _ = _check_last_cnn(
                    dense_position, 
                    conv_layer_name=conv_layer_name, 
                    mode='output'
                )
            else:
                # Check user-specified layer and mode
                _ = _check_last_cnn(
                    dense_position, 
                    conv_layer_name=conv_layer_name, 
                    mode=mode
                )
            if style in ["grad", "hires"]:
                return True, dense_position, dim, conv_layer_name
        elif check and _is_conv(model.layers[dense_position - 1]):
            # CNN -> Dense without GAP can still be explained with HiResCAM
            _, dim = _is_conv(model.layers[dense_position - 1], return_dim=True)
            if conv_layer_name is None:
                # Choose default
                conv_layer_name = model.layers[dense_position - 1].name
                _ = _check_last_cnn(
                    dense_position + 1, # Hack to get logic to understand CNN right before Dense
                    conv_layer_name=conv_layer_name, 
                    mode='output'
                )
            else:
                # Check user-specified layer and mode
                _ = _check_last_cnn(
                    dense_position + 1, # Hack to get logic to understand CNN right before Dense
                    conv_layer_name=conv_layer_name, 
                    mode=mode
                )
            if style == "hires":
                return True, dense_position, dim, conv_layer_name
            else:
                raise Exception(
                    "Cannot safely explain this model with GradCAM; use HiResCAM instead."
                )
        else:
            pass
        return False, dense_position, 0, conv_layer_name

    # Check the model has the right architecture to be safely explained
    valid, _, dim, conv_layer_name = _is_valid(model, conv_layer_name)
    if valid:
        last_layer_act = model.layers[-1].activation

        # 1. Deactivate the activation to get the raw score value
        model.layers[-1].activation = None

        # 2. Create a model that maps the input image to the activations
        # of the layer of interest, as well as the output predictions.
        grad_model = keras.models.Model(
            model.inputs,
            [
                model.get_layer(conv_layer_name).output
                if mode == "output"
                else model.get_layer(conv_layer_name).input,
                model.output,
            ],
        )

        # 3. Compute the gradient of the top predicted class for our input image
        # with respect to the activations.
        with tf.GradientTape() as tape:
            conv_layer_output, preds = grad_model(input)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, conv_layer_output)

        conv_layer_output = conv_layer_output[0]

        if style == "grad":
            # Use standard 'Grad-CAM' algorithm.

            # Mean intensity of the gradient over a specific feature map channel
            pooled_grads = tf.reduce_mean(
                grads, axis=(0, 1, 2) if dim == 2 else (0, 1)
            )

            # We multiply each channel in the feature map array
            # by "how important this channel is" with regard to the top predicted class
            # then sum all the channels to obtain the class activation map.
            class_act_map = conv_layer_output @ pooled_grads[..., tf.newaxis]
        elif style == "hires":
            # Use the 'HiResCAM' algorithm instead.
            class_act_map = tf.math.multiply(grads, conv_layer_output)
            class_act_map = tf.reduce_sum(
                class_act_map, axis=-1
            )  # "channels last" means feature maps in last dim
        else:
            raise ValueError("Unrecognized CAM style")

        class_act_map = tf.squeeze(class_act_map)

        # 4. Reactivate final layer activation.
        model.layers[-1].activation = last_layer_act

        # For visualization purpose, it is conventional to normalize the heatmap between 0 & 1 after
        # a ReLU so we only focus on what positively affects the CAM with respect to the class.
        asymm_class_act_map = tf.maximum(class_act_map, 0) / tf.math.reduce_max(
            class_act_map
        )
        if dim == 2:
            symm_class_act_map = (
                class_act_map + tf.transpose(class_act_map)
            ) / 2.0
            symm_class_act_map = tf.maximum(
                symm_class_act_map, 0
            ) / tf.math.reduce_max(symm_class_act_map)
        else:
            symm_class_act_map = asymm_class_act_map

        return (
            asymm_class_act_map.numpy(),
            symm_class_act_map.numpy(),
            preds[0].numpy(),
            pred_index.numpy(),
            conv_layer_name
        )
    else:
        raise Exception(
            f"Model does not have the right architecture to be explained with the {style} method."
        )

def _cam_1d(centers, spectra, heatmap, cmap="Reds", interp=False):
    """
    Explain 1D spectra by coloring it according to a heatmap using upsampling.

    Parameters
    ----------
    centers : ndarray(float, ndim=1)
        Location spectra were measured at in an (N,) array.

    spectra : ndarray(float, ndim=1)
        A single (N,) spectra.

    heatmap : ndarray(float, ndim=1)
        1D heatmap vector.

    cmap : matplotlib.colormaps, optional(default="Reds")
        Matplotlib colormap to use for spectra heatmap. Best if perceptually uniform.

    interp : bool, optional(default="False")
        Whether or not to interpolate the coloring.

    Returns
    -------
    lc : matplotlib.collections.LineCollection
        Line collection colored according to class activation map.
    """
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html.
    points = np.array([centers, spectra]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=plt.Normalize(
            heatmap.min(), heatmap.max()
        ),  # For improved visualization
    )

    if interp:
        # Linearly interpolate
        lc.set_array(
            np.interp(
                centers,
                np.linspace(0, 1, len(heatmap)) * (centers[-1] - centers[0])
                + centers[0],
                heatmap,
            )
        )
    else:
        # Just nearest neighbor - this assumes 'same' padding everywhere
        tree = scipy.spatial.KDTree(
            np.expand_dims(
                np.linspace(0, 1, len(heatmap)) * (centers[-1] - centers[0])
                + centers[0],
                axis=1,
            )
        )
        indices = tree.query(np.expand_dims(centers, axis=1))[1]
        lc.set_array(heatmap[indices])

    return lc

def color_spectrum(
    x,
    y,
    importance_values,
    cmap="coolwarm",
    figsize=None,
    bounds=None,
    background=True,
):
    """
    Color a 1D spectrum based on feature importance values.

    Parameters
    ----------
    x : array_like(float, ndim=1)
        Wavelengths (channel) measured at.

    y : array_like(float, ndim=1)
        Spectral (signal) intensities.

    importance_values : array_like(float, ndim=1)
        Importance value assigned. Should have the same length as x and y.

    cmap : str, optional(default="coolwarm")
        Name of matplotlib colormap to use.

    figsize : tuple(int, int), optional(default=None)
        Size of figure to plot.

    bounds : tuple(float, float), optional(default=None)
        Bounds to color based on; if unspecified uses min/max of importance_values.

    background : scalar(bool), optional(default=True)
        Whether or not to plot the uncolored (gray) spectrum behind the colored points.

    Returns
    -------
    axes : matplotlib.pyplot.axes
        Axes the result is plotted on.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    importance_values = np.asarray(importance_values, dtype=np.float64).ravel()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if background:
        axes.plot(x, y, color="k", alpha=0.10)

    if bounds is None:
        min_, max_ = importance_values.min(), importance_values.max()
    else:
        min_, max_ = bounds[0], bounds[1]

    norm = plt.Normalize(min_, max_)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(importance_values)

    line = axes.add_collection(lc)
    fig.colorbar(line, ax=axes)

    y_range = y.max() - y.min()
    axes.set_xlim(x.min(), x.max())
    axes.set_ylim(y.min() - 0.05 * y_range, y.max() + 0.05 * y_range)

    return axes


def bokeh_color_spectrum(
    x, y, importance_values, palette=Spectral10, y_axis_type=None
):
    """
    Color a 1D spectrum based on feature importance values in Bokeh.

    Parameters
    ----------
    x : array_like(float, ndim=1)
        Wavelengths (channel) measured at.

    y : array_like(float, ndim=1)
        Spectral (signal) intensities.

    importance_values : array_like(float, ndim=1)
        Importance value assigned to each feature. Should have the same length as x and y.

    palette : bokeh.palettes, optional(default=Spectral10)
        Color palette to use (https://docs.bokeh.org/en/latest/docs/reference/palettes.html).

    y_axis_type : str, optional(default=None)
        Optional transformation of y axis, e.g., y_axis_type="log".
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    importance_values = np.asarray(importance_values, dtype=np.float64).ravel()

    spectrum_df = pd.DataFrame(
        np.vstack((x, y, importance_values)).T,
        columns=("Channel", "Signal", "Importance"),
    )

    datasource = ColumnDataSource(spectrum_df)
    color_mapping = LinearColorMapper(
        low=spectrum_df["Importance"].min(),
        high=spectrum_df["Importance"].max(),
        palette=palette,
    )

    plot_figure = figure(
        title="Importance-Colored Signal",
        plot_width=900,
        plot_height=600,
        tools=("pan, wheel_zoom, reset"),
        x_axis_label="Channel",
        y_axis_label="Signal",
        y_axis_type=y_axis_type,
    )

    plot_figure.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div>
            <span style='font-size: 16px; color: #224499'>Channel:</span>
            <span style='font-size: 18px'>@Channel</span>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>Importance Value:</span>
            <span style='font-size: 18px'>@Importance</span>
        </div>
    </div>
    """
        )
    )

    plot_figure.line(
        "Channel",
        "Signal",
        source=datasource,
        color="black",
        line_width=1,
        line_alpha=0.25,
    )
    plot_figure.circle(
        "Channel",
        "Signal",
        source=datasource,
        color=dict(field="Importance", transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4,
    )
    show(plot_figure)
