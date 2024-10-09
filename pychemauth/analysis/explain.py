"""
Tools to explain predictions.

Author: nam
"""
import PIL
import matplotlib.figure
import scipy
import keras
import matplotlib
import sklearn

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, show

from matplotlib.collections import LineCollection

from typing import Union, Sequence, Any, ClassVar
from numpy.typing import NDArray


class CAMBaseExplainer:
    """Base class for explaining classifications of 1D or 2D (imaged) series with class activation map (CAM) methods."""

    style: ClassVar[str]

    def __init__(self, style: str = "hires") -> None:
        """Instantiate the class."""
        self.set_params(
            **{
                "style": style.lower(),
            }
        )

    def set_params(self, **parameters: Any) -> "CAMBaseExplainer":
        """Set parameters."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters."""
        return {
            "style": self.style,
        }

    def importances(self, *args, **kwargs):
        """Compute the feature importances for a single input."""
        raise NotImplementedError

    def explain_(self, *args, **kwargs):
        """Compute an explanation."""
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        """Visualize the explanation."""
        raise NotImplementedError


class CAM1D(CAMBaseExplainer):
    """
    Use Class Activation Map (CAM) methods to explain 1D series (e.g., spectra) classified with a Convolutional Neural Network (CNN).

    A network architecture ending in a global average pooling (GAP), followed by a single dense layer is recommended for the most consistent explanations. HiResCAM [1] will be guaranteed to reflect areas that increase the prediction's likelihood only when a single dense layer is used at the end, but a GAP is not required (and the last convolutional layer is used for explanation). Grad-CAM [2] does not carry such guarantees.

    Feature importances are bounded between [0, 1] and reflect the explanation of the class predicted for this series. CAM methods compute the importance value of each point in output of the last convolutional layer, which is much smaller than the input; e.g., a 1D spectra with 1000 input points may end up with 10 outputs from the last convolutional layer. This coarse-grained explanation vector is upsampled to match the size of the input.  The downsampled vector being explained, has a receptive field [3] (portion of the input) that mathematically contributes to its value.  Input points contribute differently to each point in the CAM which is approximated by interpolating the importance values during upsampling, though this is not rigorous.

    References
    -----------
    1. HiResCAM Method: https://arxiv.org/abs/2011.08891
    2. Grad-CAM Method: https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf
    3. https://github.com/google-research/receptive_field

    Notes
    -----
    This only supports Keras models at the moment.
    """

    def __init__(self, style: str = "hires") -> None:
        """
        Instantiate the class.

        Parameters
        ----------
        style : str
            'grad' uses Grad-CAM algorithm; 'hires' uses HiResCAM algorithm instead.  If the model has a CAM architecture (it ends in global average pooling then 1 dense layer) these should yield identical results.

        References
        -----------
        1. HiResCAM Method: https://arxiv.org/abs/2011.08891
        2. Grad-CAM Method: https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf
        """
        super().__init__(style=style)

    def importances(
        self,
        y: NDArray[np.floating],
        x: NDArray[np.floating],
        model: keras.Model,
        interp: bool = True,
    ) -> NDArray[np.floating]:
        """
        Compute the feature importances for a single series, such as a spectra.

        Parameters
        ----------
        y : ndarray(float, ndim=2)
            A single (N, 1) series.

        x : ndarray(float, ndim=1)
            Location series were measured at in an (N,) array.

        model : keras.Model
            Model being used.

        interp : bool, optional(default=True)
            Whether or not to interpolate the class activation map during upsampling
            to produce the feature importances.  If False, the importances are reported
            as the value in the CAM they are closest to (nearest neighbor) in `x`.

        Returns
        -------
        importances : ndarray(float, ndim=1)
            A vector of feature importances the same length as the input, N.

        Raises
        ------
        Exception
            The Keras model architecture is incorrect.

        ValueError
            The series is provided in the incorrect shape.
        """
        result = self.explain_(y=y, x=x, model=model, interp=interp)

        return result[1].get_array().data

    def explain_(
        self,
        y: NDArray[np.floating],
        x: NDArray[np.floating],
        model: keras.Model,
        cmap: Union[str, matplotlib.colors.LinearSegmentedColormap] = "Reds",
        interp: bool = True,
    ) -> tuple[
        NDArray[np.floating],
        matplotlib.collections.LineCollection,
        NDArray[np.floating],
        int,
        str,
    ]:
        """
        Compute a detailed explanation for a single series, such as a spectra.

        Parameters
        ----------
        y : ndarray(float, ndim=2)
            A single (N, 1) series.

        x : ndarray(float, ndim=1)
            Location series were measured at in an (N,) array.

        model : keras.Model
            Model being used.

        cmap : str or matplotlib.colors.LinearSegmentedColormap, optional(default="Reds")
            Matplotlib colormap to use for the class activation map. Best if perceptually uniform.

        interp : bool, optional(default=True)
            Whether or not to interpolate the class activation map during upsampling
            to produce the feature importances reflected in the LineCollection returned (`lc`).  If False, the importances are reported as the value in the class activation map they are closest to (nearest neighbor) in `x`.

        Returns
        -------
        class_act_map : ndarray(float, ndim=1)
            Class activation map in the range of [0, 1]. The size is determined by the
            size of the output from the last CNN layer in the model.

        lc : matplotlib.collections.LineCollection
            Line collection colored according to class activation map.  This is an
            upsampled version of `class_act_map` and may or may not be interpolated
            depending on `interp`.

        preds : ndarray(float, ndim=1)
            A vector of class probabilities.

        pred_index : int
            The index of the most likely class.

        conv_layer_name : str
            Name of the last convolutional layer found in the network, which is used for explanations.

        Raises
        ------
        Exception
            The Keras model architecture is incorrect.

        ValueError
            The series is provided in the incorrect shape.
        """
        if len(y.shape) == 2:
            if y.shape[1] != 1:
                raise ValueError(
                    "Series should have a single channel and have shape (N, 1)."
                )

            # Checks that architecture is correct internally
            (
                asymm_class_act_map,
                _,
                preds,
                pred_index,
                conv_layer_name,
            ) = _make_cam(
                style=self.style,
                input=np.expand_dims(y, axis=0),
                model=model,
                conv_layer_name=None,  # Auto-detect
            )

            lc = _color_1d(
                x=x,
                y=np.squeeze(y),
                importances=asymm_class_act_map,
                cmap=cmap,
                interp=interp,
            )
        else:
            raise ValueError("Unexpected shape of series")

        return asymm_class_act_map, lc, preds, pred_index, conv_layer_name

    def visualize(
        self,
        y: NDArray[np.floating],
        x: NDArray[np.floating],
        model: keras.Model,
        cmap: Union[str, matplotlib.colors.LinearSegmentedColormap] = "Reds",
        interp: bool = True,
        encoder: Union[
            sklearn.preprocessing.OrdinalEncoder,
            sklearn.preprocessing.LabelEncoder,
            None,
        ] = None,
        figsize: Union[tuple[int, int], None] = None,
        fontsize: Union[int, None] = None,
        show_lines: bool = False,
    ) -> tuple[
        matplotlib.pyplot.Axes,
        tuple[
            NDArray[np.floating],
            matplotlib.collections.LineCollection,
            NDArray[np.floating],
            int,
            str,
        ],
    ]:
        """
        Visualize the predictions and class activation map for a series, such as a spectra.

        Parameters
        ----------
        y : ndarray(float, ndim=2)
            A single (N, 1) series.

        x : ndarray(float, ndim=1)
            Location series were measured at in an (N,) array.

        model : keras.Model
            Model being used.

        cmap : str or matplotlib.colors.LinearSegmentedColormap, optional(default="Reds")
            Matplotlib colormap to use for the class activation map. Best if perceptually uniform.

        interp : bool, optional(default=True)
            Whether or not to interpolate the class activation map during upsampling
            to produce the feature importances reflected in the LineCollection returned (`lc`).  If False, the importances are reported as the value in the CAM they are closest to (nearest neighbor).

        encoder : sklearn.preprocessing.OrdinalEncoder or sklearn.preprocessing.LabelEncoder, optional(default=None)
            Encoder used to translate class names into integers. If None labels are reported
            as integers.

        figsize : tuple(int, int), optional(default=None)
            Size of the output figure.

        fontsize : int, optional(default=None)
            Control the fontsize in the output figure.

        show_lines : bool, optional(default=False)
            Whether or not to display lines which divide the regions of the series based on
            where the nearest (in `x`) class activation map point is used for upsampling.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Axes the result is plotted on.

        explanation : tuple
            class_act_map : ndarray(float, ndim=1)
                Class activation map in the range of [0, 1]. The size is determined by the
                size of the output from the last CNN layer in the model.

            lc : matplotlib.collections.LineCollection
                Line collection colored according to class activation map.  This is an
                upsampled version of `class_act_map` and may or may not be interpolated
                depending on `interp`.

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
            The series is provided in the incorrect shape.
        """
        explanation = self.explain_(
            y=y, x=x, model=model, cmap=cmap, interp=interp
        )
        class_act_map, lc, preds, pred_index, _ = explanation

        if isinstance(encoder, sklearn.preprocessing.OrdinalEncoder):
            classes_ = encoder.inverse_transform(
                [[i] for i in range(len(encoder.categories_[0]))]
            ).ravel()
            pred_ = encoder.inverse_transform([[pred_index]])[0][0]
        elif isinstance(encoder, sklearn.preprocessing.LabelEncoder):
            classes_ = encoder.inverse_transform(
                [i for i in range(len(encoder.classes_))]
            )
            pred_ = encoder.inverse_transform([pred_index])[0]
        else:
            classes_ = [str(c_) for c_ in range(len(preds))]
            pred_ = pred_index

        fig, ax = plt.subplots(
            nrows=1, ncols=2, figsize=(8, 2) if figsize is None else figsize
        )

        # Plot raw scores before softmax is applied to compute probabilities
        ax[0].bar(classes_, preds, color="black", width=0.9)
        ax[0].set_xticks(
            ticks=np.arange(len(classes_)),
            labels=classes_,
            rotation=90,
            fontsize=fontsize - 2 if fontsize is not None else fontsize,
        )
        ax[0].set_title("Prediction = {}".format(pred_), fontsize=fontsize)
        ax[0].set_ylabel("Raw Scores", fontsize=fontsize)

        # Plot colored series from CAM
        ax[1].set_xlim(
            np.min((ax[1].get_xlim()[0], x.min())),
            np.max((ax[1].get_xlim()[1], x.max())),
        )
        ax[1].set_ylim(
            np.min((ax[1].get_ylim()[0], y.min())),
            np.max((ax[1].get_ylim()[1], y.max())),
        )
        line = ax[1].add_collection(lc)
        cbar = fig.colorbar(line, ax=ax[1])
        cbar.ax.tick_params(
            labelsize=fontsize - 2 if fontsize is not None else fontsize
        )
        # ax[1].set_xticklabels(ax[1].get_xticklabels(), fontsize=fontsize-2 if fontsize is not None else fontsize)
        # ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize=fontsize-2 if fontsize is not None else fontsize)

        if show_lines:
            # Draw the lines where the colors change (heatmap pixels)
            for i in range(len(class_act_map) - 1):
                dx = (x[-1] - x[0]) / float(len(class_act_map) - 1)
                ax[1].axvline((0.5 + i) * dx + x[0], color="k", ls="--")

        return ax, explanation


class CAM2D(CAMBaseExplainer):
    """
    Use Class Activation Map (CAM) methods to explain 2D images classified with a Convolutional Neural Network (CNN).

    A network architecture ending in a global average pooling (GAP), followed by a single dense layer is recommended for the most consistent explanations. HiResCAM [1] will be guaranteed to reflect areas that increase the prediction's likelihood only when a single dense layer is used at the end, but a GAP is not required (and the last convolutional layer is used for explanation). Grad-CAM [2] does not carry such guarantees.

    Feature importances are bounded between [0, 1] and reflect the explanation of the class predicted for this series. CAM methods compute the importance value of each pixel in output of the last convolutional layer, which is much smaller than the input; e.g., a 256x256 image may end up being 10x10 after the last convolutional layer. This coarse-grained explanation image is upsampled to match the size of the input.  The downsampled image being explained, has a receptive field [3] (portion of the input) that mathematically contributes to its value.  Input pixels contribute differently to each pixel in the CAM which is approximated by interpolating the importance values during upsampling, though this is not rigorous.

    References
    -----------
    1. HiResCAM Method: https://arxiv.org/abs/2011.08891
    2. Grad-CAM Method: https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf
    3. https://github.com/google-research/receptive_field

    Notes
    -----
    This only supports Keras models at the moment.
    """

    def __init__(self, style: str = "hires") -> None:
        """
        Instantiate the class.

        Parameters
        ----------
        style : str
            'grad' uses Grad-CAM algorithm; 'hires' uses HiResCAM algorithm instead.  If the model has a CAM architecture (it ends in global average pooling then 1 dense layer) these should yield identical results.

        References
        -----------
        1. HiResCAM Method: https://arxiv.org/abs/2011.08891
        2. Grad-CAM Method: https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf
        """
        super().__init__(style=style)

    def importances(
        self,
        image: NDArray[np.floating],
        model: keras.Model,
        symmetrize: bool = True,
        dim: int = 2,
        series_summary: str = "mean",
    ) -> NDArray[np.floating]:
        """
        Compute the feature importances for a single series, such as a spectra.

        Parameters
        ----------
        image : ndarray(float, ndim=3)
            Square image to explain as a single (N, N, 1) tensor where the image is NxN with 1 channel (must be in 'channels_last' format).

        model : keras.Model
            Model being used to classify the image.

        symmetrize : bool, optional(default=True)
            Whether to use the symmetric CAM for 2D visualization. Symmetrization is applied before normalizing (incl. applying a ReLU). This means that if (i,j) helps but (j,i) hurts, then the averaging done by symmetrizing should be done before ReLU since the parts of the image that might decrease class probability are rounded up to 0, artificially inflating the representation of how much the model really interacts with the i-j peaks.

        dim : int, optional(default=2)
            Dimensionality of explanation to return.  If `dim=2` returns a 2D array where each pixel is given an importance value; if `dim=1` then return the "condensed" explanation for a series that presumably was used to generate the image being explained.

        series_summary : str, optional(default='mean')
            Method to summarize the 2D explanation on a 1D series.  The default, 'mean', takes the average across the rows (equivalently, columns) of the symmetrized image.  Must be in {'mean', 'max'}.

        Returns
        -------
        importances : ndarray(float, ndim=1 or 2)
            A vector of feature importances the same size as the image; either (N,) or (N, N) depending on the value of `dim`.

        Raises
        ------
        Exception
            The Keras model architecture is incorrect.

        ValueError
            The series is provided in the incorrect shape.
            The value of `dim` is not in {1, 2}.
        """
        _, _, _, lc, _, _, _, importances = self.explain_(
            x=np.arange(0, image.shape[1], dtype=np.float64),  # Dummy values
            y=np.arange(0, image.shape[0], dtype=np.float64),  # Dummy values
            image=image,
            model=model,
            symmetrize=symmetrize,
            series_summary=series_summary,
        )

        if dim == 1:
            # This condenses CAM then linearly interpolates to full size spectra
            return lc.get_array().data  # type: ignore[union-attr]
        elif dim == 2:
            # (bi)Linearly interpolate CAM to larger image size
            return importances
        else:
            raise ValueError("Invalid dim value; must be in {1, 2}.")

    def explain_(
        self,
        image: NDArray[np.floating],
        model: keras.Model,
        symmetrize: bool = True,
        y: Union[NDArray[np.floating], None] = None,
        x: Union[NDArray[np.floating], None] = None,
        image_cmap: Union[
            str, matplotlib.colors.LinearSegmentedColormap
        ] = matplotlib.colormaps["jet"],
        series_summary: str = "mean",
        series_cmap: Union[
            str, matplotlib.colors.LinearSegmentedColormap
        ] = "Reds",
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        Union[matplotlib.collections.LineCollection, None],
        NDArray[np.floating],
        int,
        str,
        NDArray[np.floating],
    ]:
        """
        Produce a CAM explanation for a 2D, single-channel image.

        Parameters
        ----------
        image : ndarray(float, ndim=3)
            Square image to explain as a single (N, N, 1) tensor where the image is NxN with 1 channel (must be in 'channels_last' format).

        model : keras.Model
            Model being used to classify the image.

        symmetrize : bool, optional(default=True)
            Whether to use the symmetric CAM for 2D visualization. Symmetrization is applied before normalizing (incl. applying a ReLU). This means that if (i,j) helps but (j,i) hurts, then the averaging done by symmetrizing should be done before ReLU since the parts of the image that might decrease class probability are rounded up to 0, artificially inflating the representation of how much the model really interacts with the i-j peaks.

        y : ndarray(float, ndim=1), optional(default=None)
            A single (N,) series, such as a 1D spectra. If provided, it is assumed the `image` is a transformation of this series.

        x : ndarray(float, ndim=1), optional(default=None)
            Locations series were measured at in an (N,) array. If provided, it is assumed the `image` is a transformation of the series this corresponds to.

        image_cmap : str or matplotlib.colors.LinearSegmentedColormap, optional(default=matplotlib.colormaps["jet"])
            Matplotlib colormap to use for the 2D image and CAM explanation.

        series_summary : str, optional(default='mean')
            Method to summarize the 2D explanation on a 1D series.  The default, 'mean', takes the average across the rows (equivalently, columns) of the symmetrized image.  Must be in {'mean', 'max'}.

        series_cmap : str or matplotlib.colors.LinearSegmentedColormap, optional(default="Reds")
            Matplotlib colormap to use on the condensed 2D CAM explanation for the series. Best if perceptually uniform.

        Returns
        -------
        asymm_class_act_map : ndarray(float, ndim=2)
            Asymmetric class activation map in the range of [0, 1]. The size is determined by the size of the output from the last CNN layer in the model.

        symm_class_act_map : ndarray(float, ndim=2)
            Symmetric class activation map in the range of [0, 1]. The size is determined by the size of the output from the last CNN layer in the model.

        cmap_heatmap : ndarray(float, ndim=3)
            (N, N, 3) RBG colormap of class activations.

        lc : matplotlib.collections.LineCollection or None
            Line collection colored according to "condensed" class activation map if a 1D series is provided. This is an
            upsampled version of `symm_class_act_map`. If a series was not provided, this is returned as `None`.

        preds : ndarray(float, ndim=1)
            A vector of class probabilities.

        pred_index : int
            The index of the most likely class.

        conv_layer_name : str
            Name of the last convolutional layer found in the network, which is used for explanations.

        importances : ndarray(float, ndim=2)
            Feature importance of each pixel [0, 1] with the original image size (N, N).

        Raises
        ------
        ValueError
            If the image has the wrong shape.
            If the image is not square (N, N, 1).
        """
        if len(image.shape) == 3:
            if image.shape[0] != image.shape[1]:
                raise ValueError("Image should be square.")

            (
                cmap_heatmap,
                asymm_class_act_map,
                symm_class_act_map,
                preds,
                pred_index,
                conv_layer_name,
            ) = self._explain_2d(
                image=np.expand_dims(
                    image, axis=0
                ),  # Make into a batch with only 1 entry
                model=model,
                cmap=image_cmap,
                symmetrize=symmetrize,
            )

            importances = self._resize(
                image=np.expand_dims(
                    asymm_class_act_map
                    if not symmetrize
                    else symm_class_act_map,
                    axis=-1,
                ),
                image_shape=(image.shape[1], image.shape[0]),
                rescale=True,
            )

            if y is None:
                # Explaining a 2D image, or we do not have access to the original spectra
                lc = None
            else:
                # Explaining an "imaged" series such as a spectra
                lc = _color_1d(
                    x=x,  # type: ignore[arg-type]
                    y=y,
                    importances=self._condense(
                        series_summary=series_summary,
                        symm_map=symm_class_act_map,
                    ),
                    cmap=series_cmap,
                    interp=True,
                )
        else:
            raise ValueError("Unexpected shape of image")

        return (  # type: ignore[return-value]
            asymm_class_act_map,
            symm_class_act_map,
            cmap_heatmap,
            lc,
            preds,
            pred_index,
            conv_layer_name,
            importances,
        )

    def visualize(
        self,
        image: NDArray[np.floating],
        model: keras.Model,
        symmetrize: bool = True,
        y: Union[NDArray[np.floating], None] = None,
        x: Union[NDArray[np.floating], None] = None,
        image_cmap: Union[
            str, matplotlib.colors.LinearSegmentedColormap
        ] = matplotlib.colormaps["jet"],
        series_cmap: Union[
            str, matplotlib.colors.LinearSegmentedColormap
        ] = "Reds",
        encoder: Union[
            sklearn.preprocessing.OrdinalEncoder,
            sklearn.preprocessing.LabelEncoder,
            None,
        ] = None,
        correct_label: Union[str, None] = None,
        origin: str = "upper",
        fontsize: int = 12,
    ) -> tuple[
        matplotlib.figure.Figure,
        matplotlib.figure.Figure,
        matplotlib.figure.Figure,
        matplotlib.figure.Figure,
    ]:
        """
        Visualize the CAM explaination of a 2D, single-channel image.

        Parameters
        ----------
        image : ndarray(float, ndim=3)
            Square image to explain as a single (N, N, 1) tensor where the image is NxN with 1 channel (must be in 'channels_last' format).

        model : keras.Model
            Model being used to classify the image.

        symmetrize : bool, optional(default=True)
            Whether to use the symmetric CAM for 2D visualization. Symmetrization is applied before normalizing (incl. applying a ReLU). This means that if (i,j) helps but (j,i) hurts, then the averaging done by symmetrizing should be done before ReLU since the parts of the image that might decrease class probability are rounded up to 0, artificially inflating the representation of how much the model really interacts with the i-j peaks.

        y : ndarray(float, ndim=1), optional(default=None)
            A single (N,) series, such as a 1D spectra. If provided, it is assumed the `image` is a transformation of this series.

        x : ndarray(float, ndim=1), optional(default=None)
            Locations series were measured at in an (N,) array. If provided, it is assumed the `image` is a transformation of the series this corresponds to.

        image_cmap : str or matplotlib.colors.LinearSegmentedColormap, optional(default=matplotlib.colormaps["jet"])
            Matplotlib colormap to use for the 2D image and CAM explanation.

        series_cmap : str or matplotlib.colors.LinearSegmentedColormap, optional(default="Reds")
            Matplotlib colormap to use on the condensed 2D CAM explanation for the series. Best if perceptually uniform.

        encoder : sklearn.preprocessing.OrdinalEncoder or sklearn.preprocessing.LabelEncoder, optional(default=None)
            Encodes classes (strings) as integers.

        correct_label : str, optional(default=None)
            Correct label for the series (e.g., from y_train). Used to label the output figures.

        origin : str, optional(default='upper')
            Origin convention for 2D images; see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html.

        fontsize : int, optional(default=12)
            Font size for titles; font sizes for labels and axes are scaled accordingly.

        Returns
        -------
        fig1 : matplotlib.pyplot.figure
            Figure showing the score and probabilities.

        fig2 : matplotlib.pyplot.figure
            Figure showing the colored image.

        fig3 : matplotlib.pyplot.figure
            Figure showing the class activation map.

        fig4 : matplotlib.pyplot.figure
            Figure showing the explained image where the alpha value (transparency) reflects the importance.
        """
        (
            _,
            _,
            cmap_heatmap,
            lc,
            preds,
            pred_index,
            _,
            importances,
        ) = self.explain_(
            x=x,
            y=y,
            image=image,
            model=model,
            image_cmap=image_cmap,
            series_cmap=series_cmap,
            symmetrize=symmetrize,
        )

        # Set fontsizes
        title_fontsize = fontsize
        label_fontsize = title_fontsize - 2

        # 1. Plot raw score output
        fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 2))

        if isinstance(encoder, sklearn.preprocessing.OrdinalEncoder):
            classes_ = encoder.inverse_transform(
                [[i] for i in range(len(encoder.categories_[0]))]
            ).ravel()
            if len(classes_) != len(preds):
                raise ValueError("Encoder has the wrong number of classes")
            ax[0].set_title(
                "Prediction = {}".format(
                    encoder.inverse_transform([[pred_index]])[0][0],
                ),
                fontsize=title_fontsize,
            )
        elif isinstance(encoder, sklearn.preprocessing.LabelEncoder):
            classes_ = encoder.inverse_transform(
                [i for i in range(len(encoder.classes_))]
            )
            if len(classes_) != len(preds):
                raise ValueError("Encoder has the wrong number of classes")
            ax[0].set_title(
                "Prediction = {}".format(
                    encoder.inverse_transform([pred_index])[0],
                ),
                fontsize=title_fontsize,
            )
        else:
            classes_ = ["Class {}".format(i) for i in np.arange(len(preds))]
            ax[0].set_title(
                "Prediction = Class {}".format(pred_index),
                fontsize=title_fontsize,
            )

        ax[0].bar(classes_, preds, color="black", width=0.9)
        ax[0].set_xticks(
            ticks=np.arange(len(classes_)),
            labels=classes_,
            rotation=90,
            fontsize=label_fontsize,
        )
        ax[0].set_ylabel("Raw Scores", fontsize=label_fontsize)

        clipped = np.clip(preds, a_min=-250, a_max=250)
        ax[1].bar(
            classes_,
            np.exp(clipped) / np.sum(np.exp(clipped)),
            color="black",
            width=0.9,
        )
        ax[1].set_xticks(
            ticks=np.arange(len(classes_)),
            labels=classes_,
            rotation=90,
            fontsize=label_fontsize,
        )
        ax[1].set_yscale("log")
        ax[1].set_title(
            "Actual = {}".format(correct_label)
            if correct_label is not None
            else "",
            fontsize=title_fontsize,
        )
        ax[1].set_ylabel("Softmax Probability", fontsize=label_fontsize)

        def build_fig(colorize=False):
            width_ratios = (2, 7, 0.4)
            height_ratios = (2, 7)
            width = 5
            height = width * sum(height_ratios) / sum(width_ratios)
            fig = plt.figure(figsize=(width, height))
            gs = fig.add_gridspec(
                2,
                3,
                width_ratios=width_ratios,
                height_ratios=height_ratios,
                left=0.1,
                right=0.9,
                bottom=0.1,
                top=0.9,
                wspace=0.01,
                hspace=0.01,
            )

            if (y is not None) and (x is not None):
                ax_left = fig.add_subplot(gs[1, 0])
                ax_left.plot(y, x, color="k")
                ax_left.set_ylim([x.min(), x.max()])
                ax_left.set_xlim([y.min(), y.max()])
                plt.setp(ax_left.spines.values(), color="white")
                ax_left.set_xticks([])
                ax_left.set_yticks([])
                ax_left.invert_xaxis()

                if origin == "upper":
                    ax_left.invert_yaxis()

                ax_top = fig.add_subplot(gs[0, 1])
                if colorize:
                    ax_top.add_collection(
                        lc
                    )  # Plot series colored by importance on top
                    cbar = fig.colorbar(lc, cax=fig.add_subplot(gs[0, 2]))
                    cbar.ax.tick_params(labelsize=label_fontsize)
                else:
                    ax_top.plot(x, y, color="k")  # Just plot the series on top

                ax_top.set_xlim([x.min(), x.max()])
                ax_top.set_ylim([y.min(), y.max()])
                ax_top.set_xticks([])
                ax_top.set_yticks([])
                plt.setp(ax_top.spines.values(), color="white")

            return fig, gs

        # Build image
        fig2, gs = build_fig(colorize=False)
        ax_gaf = fig2.add_subplot(gs[1, 1])
        im2 = ax_gaf.imshow(image[:, :, 0], cmap=image_cmap, origin=origin)
        ax_gaf.set_xticks([])
        ax_gaf.set_yticks([])
        ax_gaf.set_title(
            "Image",
            y=-0.09 - (title_fontsize - 12) * 0.005,
            fontsize=title_fontsize,
        )
        cbar = fig2.colorbar(im2, cax=fig2.add_subplot(gs[1, 2]), pad=-1)
        cbar.ax.tick_params(labelsize=label_fontsize)
        # cbar.set_ticks([])

        # Build CAM
        fig3, gs = build_fig(colorize=True)
        ax_cam = fig3.add_subplot(gs[1, 1])
        im3 = ax_cam.imshow(cmap_heatmap, cmap=image_cmap, origin=origin)
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])
        if self.style == "grad":
            ax_cam.set_title(
                "GradCAM",
                y=-0.09 - (title_fontsize - 12) * 0.005,
                fontsize=title_fontsize,
            )
        else:
            ax_cam.set_title(
                "HiResCAM",
                y=-0.09 - (title_fontsize - 12) * 0.005,
                fontsize=title_fontsize,
            )
        cbar = fig3.colorbar(im3, cax=fig3.add_subplot(gs[1, 2]))
        cbar.ax.tick_params(labelsize=label_fontsize)
        # cbar.set_ticks([])

        # Build the image with alpha to indicate relevance / explanation
        fig4, gs = build_fig(colorize=False)
        ax_expl = fig4.add_subplot(gs[1, 1])
        im4 = ax_expl.imshow(
            image[:, :, 0], cmap=image_cmap, origin=origin, alpha=importances
        )
        ax_expl.set_xticks([])
        ax_expl.set_yticks([])
        ax_expl.set_title(
            "Explained Image",
            y=-0.09 - (title_fontsize - 12) * 0.005,
            fontsize=title_fontsize,
        )
        cbar = fig4.colorbar(im4, cax=fig4.add_subplot(gs[1, 2]))
        cbar.ax.tick_params(labelsize=label_fontsize)
        # cbar.set_ticks([])

        return fig1, fig2, fig3, fig4

    def _explain_2d(
        self,
        image: NDArray[np.floating],
        model: keras.Model,
        cmap: Union[str, matplotlib.colors.LinearSegmentedColormap],
        symmetrize: bool,
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        int,
        str,
    ]:
        """
        Explain a 2D image.

        Parameters
        ----------
        image : ndarray(float, ndim=3)
            Image as a single (1, N, N, C) tensor where the image is NxN with C=1 channels (must be in 'channels_last' format).

        model : keras.Model
            Model being used to classify the image.

        cmap : str or matplotlib.colors.LinearSegmentedColormap
            Matplotlib colormap to use for the 2D heatmap.

        symmetrize : bool
            Whether to use the symmetric CAM for 2D visualization. Symmetrization is applied before normalizing (incl. applying a ReLU). This means that if (i,j) helps but (j,i) hurts, then the averaging done by symmetrizing should be done before ReLU since the parts of the image that might decrease class probability are rounded up to 0, artificially inflating the representation of how much the model really interacts with the i-j peaks.

        Returns
        -------
        cmap_heatmap : ndarray(float, ndim=3)
            (N, N, 3) RGB heatmap of class activations.

        asymm_class_act_map : ndarray(float, ndim=2)
            Asymmetric class activation map in the range of [0, 1]. The size is determined by the size of the output from the last CNN layer in the model.

        symm_class_act_map : ndarray(float, ndim=2)
            Symmetric class activation map in the range of [0, 1]. The size is determined by the size of the output from the last CNN layer in the model.

        preds : ndarray(float, ndim=1)
            A vector of class probabilities.

        pred_index : int
            The index of the most likely class.

        conv_layer_name : str
            Name of the last convolutional layer found in the network, which is used for explanations.
        """
        # Checks that architecture is correct internally
        (
            asymm_class_act_map,
            symm_class_act_map,
            preds,
            pred_index,
            conv_layer_name,
        ) = _make_cam(
            style=self.style,
            input=image,
            model=model,
            conv_layer_name=None,  # Auto-detect
        )

        class_act_map = (
            symm_class_act_map if symmetrize else asymm_class_act_map
        )
        heatmap = np.uint8(
            255 * class_act_map
        )  # Create a scaled heatmap in a range 0-255 (CAM is in range [0, 1] already)
        if not isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
            cmap = matplotlib.colormaps(
                cmap
            )  # cmap is a string, so get the map from the name
        cmap_colors = cmap(np.arange(256))[:, :3]
        cmap_heatmap = cmap_colors[heatmap]

        cmap_heatmap = self._resize(
            image_shape=(image.shape[2], image.shape[1]),
            image=cmap_heatmap,
        )

        return (
            cmap_heatmap,
            asymm_class_act_map,
            symm_class_act_map,
            preds,
            pred_index,
            conv_layer_name,
        )

    def _resize(
        self,
        image_shape: tuple[int, int],
        image: NDArray[np.floating],
        rescale: bool = False,
    ) -> Union[NDArray[np.floating], NDArray[np.uint8]]:
        """
        Compute colored heatmap and rescale to size of input image using (bi)linear interpolation.

        This is based on https://keras.io/examples/vision/grad_cam/.

        Parameters
        ----------
        image_shape : tuple(int, int)
            (Width, Height) to resize the image to.

        image : ndarray(float, ndim=3)
            Image to rescale to have `image_shape`; should have a shape of (N, N, 1).

        rescale : bool, optional(default=False)
            If True, rescale the image to [0, 1] (instead of [0, 255]) and trims off the channel dimension returning a 2D vector instead of 3D tensor.

        Returns
        -------
        resized : ndarray(float, ndim=2) or ndarray(uint8, ndim=3)
            Linearly interpolated resized image.
        """
        # Create an image to take advantange of built-in tools
        image_ = keras.utils.array_to_img(image)
        image_ = image_.resize(
            image_shape,
            resample=PIL.Image.BILINEAR,  # Do linear interpolation to be consistent with interpolation of 1D
        )
        image = keras.utils.img_to_array(image_)

        if rescale:
            # Convert to 2D array in [0, 1]
            return np.squeeze(
                image / 255.0, axis=-1  # Convert back to [0, 1] range
            )
        else:
            # Convert to uint8 for image convention
            return image.astype(np.uint8)

    def _condense(
        self, series_summary: str, symm_map: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Condense a 2D symmetric CAM to a 1D version.

        Parameters
        ----------
        series_summary : str, optional(default='mean')
            Method to summarize the 2D explanation on a 1D series.  The default, 'mean', takes the average across the rows (equivalently, columns) of the symmetrized image.  Must be in {'mean', 'max'}.

        symm_map : ndarray(float, ndim=2)
            Symmetric class activation map.

        Returns
        -------
        condensed : ndarray(float, ndim=1)
            Condensed version of `symm_map`.
        """
        if series_summary.lower() == "mean":
            condensed = np.mean(
                symm_map, axis=0
            )  # Average across "columns" - always use symmetric version, otherwise ambiguous if we should use columns or rows since they could differ
        elif series_summary.lower() == "max":
            condensed = np.max(
                symm_map, axis=0
            )  # Take max across "columns" - always use symmetric version, otherwise ambiguous if we should use columns or rows since they could differ
        else:
            raise ValueError(
                'Unrecognized series_summary method; must be in {"mean", "max"}.'
            )

        return condensed


def check_cam_model_architecture(
    model: keras.Model,
    style: str,
    conv_layer_name: Union[str, None] = None,
    mode: Union[str, None] = None,
) -> tuple[bool, int, int, str, str]:
    """
    Check if a Keras model has a valid architecture for CAM explanation.

    There are 2 valid architectures:
    1. BASE -> [GAP -> Dropout (optional) -> Dense]
    2. BASE -> [Flatten -> Dropout (optional) -> Dense]
    where BASE = CNN or a pretrained convolutional keras.src.engine.functional.Functional and the [bracketed] part is the
    TOP.

    If BASE is pretrained then it can be complicated, in practice, to point to the "output" of the last convolutional
    layer.  Instead, we can use the "input" to the next layer (first in the TOP) which would be either the GAP or Flatten.

    Parameters
    ----------
    model : keras.Model
        Model being used.

    style : str
        Should be {'grad', 'hires'} to indicate the style of CAM.

    conv_layer_name : str, optional(default=None)
        Name of the Keras layer being explained. Sometimes these layers are hidden as a function so it easier to reference the input to the subsequent layer, rather than the output of the layer desired.  `mode` controls this. If None, defaults to the last convolutional layer before the TOP.

    mode : str, optional(default=None)
        Whether to explain the output or input of the `conv_layer_name` layer. Expects either {'output', 'input'}. If `conv_layer_name` is None, this is ignored.

    Returns
    -------
    valid : bool
        Whether the model has a valid architecture.

    position : int
        Layer position, relative to the model end, where the GAP or Flatten is.

    dim : int
        Dimensionality of the model (either 1D or 2D).

    layer_name : str
        Name of the terminal convolutional layer being explained.  If `effective_mode` is input, this will be the name of the next layer.

    effective_mode : str
        Whether to explain the output or input of the `conv_layer_name` layer.
    """

    def _is_conv(layer, return_dim=False):
        # Check if a layer is convolutional - meant to check the model's base.
        # If the base is a pretrained Keras application we need to use the 'input' to
        # the next layer, not the 'output' of this part for the gradcam algorithm.
        # This performs these checks.

        # Pooling after other convolutional layers is also acceptable, just not Global pooling
        for type_ in [
            keras.layers.Conv1D,
            keras.layers.MaxPooling1D,
            keras.layers.AveragePooling1D,
        ]:
            if isinstance(layer, type_):
                return (True, 1, None) if return_dim else True

        for type_ in [
            keras.layers.Conv2D,
            keras.layers.MaxPooling2D,
            keras.layers.AveragePooling2D,
        ]:
            if isinstance(layer, type_):
                return (True, 2, None) if return_dim else True

        # No easy way to check if we are using a pre-trained CNN base
        # from Keras applications; best way I can think of for now is
        # to recognize this as a functional.Functional as assume it is
        # for 2D since at the moment only 2D image classifiers are
        # available in Keras.
        for type_ in [keras.src.engine.functional.Functional]:
            if isinstance(layer, type_):
                return (True, 2, "input") if return_dim else True

        return (False, 0, None) if return_dim else False

    def _is_gap(layer):
        # Check if a layer does global average pooling
        valid = [
            keras.layers.GlobalAveragePooling1D,
            keras.layers.GlobalAveragePooling2D,
        ]
        for type_ in valid:
            if isinstance(layer, type_):
                return True
        return False

    def _ends_with_dense(model):
        # Check the model ends with a dense layer.
        valid = [
            keras.activations.softmax,
            keras.activations.sigmoid,
        ]  # Could be softmax (multiclass) or logistic (binary)
        if isinstance(model.layers[-1], keras.layers.Dense):
            # Ends with a dense layer with a softmax activation
            if model.layers[-1].activation in valid:
                return True, -1
        elif isinstance(model.layers[-1], keras.layers.Activation):
            # Activation specified manually after a linear dense layer
            if model.layers[-1].activation in valid:
                if isinstance(model.layers[-2], keras.layers.Dense):
                    if (
                        model.layers[-2].activation == keras.activations.linear
                    ):  # Must not be an activation here
                        return True, -2
        return False, 0

    def _dropout_adjust(dense_position, model):
        # Positional adjustment if there is an optional dropout layer before the final Dense layer
        if isinstance(model.layers[dense_position - 1], keras.layers.Dropout):
            return -1
        else:
            return 0

    def _check_last_cnn(dense_position, conv_layer_name, mode):
        # Check we are explaining the last CNN layer in the network
        if (
            mode == "output"
            and model.layers[dense_position - 2].name != conv_layer_name
        ):  # Output of BASE
            raise Exception(
                "You are not explaining the last convolutional layer in the network."
            )
        if (
            mode == "input"
            and model.layers[dense_position - 1].name != conv_layer_name
        ):  # Input to GAP/Flatten layer
            raise Exception(
                "You are not explaining the last convolutional layer in the network."
            )
        return True

    def _certify_mode(
        model, dense_position, terminal_dropout, conv_layer_name, mode
    ):
        # Check the mode of operation is consistent with where the last convolutional layer is
        _, dim, mode_override = _is_conv(
            model.layers[dense_position + terminal_dropout - 2], return_dim=True
        )

        effective_mode = mode
        if mode_override is not None:
            effective_mode = mode_override

        if conv_layer_name is None:
            # Choose default
            if (
                effective_mode == "input"
            ):  # Functional Keras base so need to use input mode to next layer
                conv_layer_name = model.layers[
                    dense_position + terminal_dropout - 2 + 1
                ].name
            else:
                # Point directly to output of the BASE
                conv_layer_name = model.layers[
                    dense_position + terminal_dropout - 2
                ].name
                effective_mode = "output"

            _ = _check_last_cnn(
                dense_position + terminal_dropout,
                conv_layer_name=conv_layer_name,
                mode=effective_mode,
            )
        else:
            # Check user-specified layer and mode are correct
            _ = _check_last_cnn(
                dense_position + terminal_dropout,
                conv_layer_name=conv_layer_name,
                mode=effective_mode,
            )

        return dim, conv_layer_name, effective_mode

    # Check the overall model architecture
    check, dense_position = _ends_with_dense(model)
    terminal_dropout = _dropout_adjust(dense_position, model)

    if (
        check
        and _is_gap(model.layers[dense_position + terminal_dropout - 1])
        and _is_conv(model.layers[dense_position + terminal_dropout - 2])
    ):
        # CAM architecture
        # Can explain with either method and should give identical results
        dim, layer_name, effective_mode = _certify_mode(
            model, dense_position, terminal_dropout, conv_layer_name, mode
        )
        if style in ["grad", "hires"]:
            return (
                True,
                dense_position + terminal_dropout,
                dim,
                layer_name,
                effective_mode,
            )
    elif (
        check
        and isinstance(
            model.layers[dense_position + terminal_dropout - 1],
            keras.layers.Flatten,
        )
        and _is_conv(model.layers[dense_position + terminal_dropout - 2])
    ):
        # Only alternative valid archictecture is BASE -> Flatten -> Dropout (optional) -> Dense
        # Can still be explained with HiResCAM
        dim, layer_name, effective_mode = _certify_mode(
            model, dense_position, terminal_dropout, conv_layer_name, mode
        )
        if style == "hires":
            return (
                True,
                dense_position + terminal_dropout,
                dim,
                layer_name,
                effective_mode,
            )
        else:
            raise Exception(
                "Cannot safely explain this model with GradCAM; use HiResCAM instead."
            )
    else:
        pass

    return False, 0, 0, "", ""


def _make_cam(
    style: str,
    input: NDArray[np.floating],
    model: keras.Model,
    conv_layer_name: Union[str, None] = None,
    mode: str = "output",
    pred_index: Union[int, None] = None,
) -> tuple[
    NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], int, str
]:
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
        When explaining a 2D image (1, N, N, C) tensor where the image is NxN with C=1 channels. This must be in 'channels_last' format. When explaining a 1D series (1, N, C) tensor where the image is NxN with C=1 channels.

    model : keras.Model
        Model being used.

    conv_layer_name : str, optional(default=None)
        Name of the Keras layer being explained. Sometimes these layers are hidden as a function so it easier to reference the input to the subsequent layer, rather than the output of the layer desired.  `mode` controls this. If None, defaults to the last convolutional layer before the top.

    mode : str, optional(default='output')
        Whether to explain the output or input of the `conv_layer_name` layer. Expects either {'output', 'input'}. If `conv_layer_name` is None, this is ignored.

    pred_index : int, optional(default=None)
        Index of class to compute the activation map with respect to.  If `None`, the most likely class is used.

    Returns
    -------
    asymm_class_act_map : ndarray(float, ndim=1 or 2)
        Asymmmetric CAM in the range of [0, 1]. The size is determined by the size of the output from the last CNN layer in the model.

    symm_class_act_map : ndarray(float, ndim=1 or 2)
        Symmmetric CAM in the range of [0, 1]. The size is determined by the size of the output from the last CNN layer in the model.

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
    # Due to issues with running this on certain gpus, we can force this to operate on the CPU.
    # This seems to arise when using certain CNN Bases which have, e.g., batch norms inside.
    from tensorflow.python.client import device_lib

    cpu_name = None
    for dev in device_lib.list_local_devices():
        if dev.device_type == "CPU":
            cpu_name = dev.name
            break
    if cpu_name is None:
        raise Exception(
            "Could not locate the CPU to compute class activation map."
        )

    with tf.device(cpu_name):
        (
            valid,
            _,
            dim,
            conv_layer_name,
            effective_mode,
        ) = check_cam_model_architecture(
            model=model, style=style, conv_layer_name=conv_layer_name, mode=mode
        )
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
                    if effective_mode == "output"
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
                class_act_map = (
                    conv_layer_output @ pooled_grads[..., tf.newaxis]
                )
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
            asymm_class_act_map = tf.maximum(
                class_act_map, 0
            ) / tf.math.reduce_max(class_act_map)
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
                pred_index.numpy(),  # type: ignore[union-attr]
                conv_layer_name,
            )
        else:
            raise Exception(
                f"Model does not have the right architecture to be explained with the {style} method."
            )


def _color_1d(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    importances: NDArray[np.floating],
    cmap: Union[str, matplotlib.colors.LinearSegmentedColormap] = "Reds",
    interp: bool = True,
    bounds: Union[tuple[float, float], None] = None,
) -> matplotlib.collections.LineCollection:
    """
    Explain 1D series by coloring it according to a heatmap using upsampling.

    Parameters
    ----------
    x : ndarray(float, ndim=1)
        Location series were measured at in an (N,) array.

    y : ndarray(float, ndim=1)
        A single (N,) series, such as a 1D spectra.

    importances : ndarray(float, ndim=1)
        1D vector to color based on, e.g., the class activation map.

    cmap : str or matplotlib.colors.LinearSegmentedColormap, optional(default="Reds")
        Matplotlib colormap to use for the `importances`. Best if perceptually uniform.

    interp : bool, optional(default=True)
        Whether or not to interpolate the coloring.

    bounds : tuple(float, float), optional(default=None)
        Specific bounds to apply to color normalization for improved visualization.

    Returns
    -------
    lc : matplotlib.collections.LineCollection
        Line collection colored according to class activation map.
    """
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html.
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=plt.Normalize(importances.min(), importances.max())
        if bounds is None
        else plt.Normalize(bounds[0], bounds[1]),  # For improved visualization
    )

    if interp:
        # Linearly interpolate
        lc.set_array(
            np.interp(
                x,
                np.linspace(0, 1, len(importances)) * (x[-1] - x[0]) + x[0],
                importances,
            )
        )
    else:
        # Just nearest neighbor - this assumes 'same' padding everywhere
        tree = scipy.spatial.KDTree(
            np.expand_dims(
                np.linspace(0, 1, len(importances)) * (x[-1] - x[0]) + x[0],
                axis=1,
            )
        )
        indices = tree.query(np.expand_dims(x, axis=1))[1]
        lc.set_array(importances[indices])

    return lc


def color_series(
    y: Union[NDArray[np.floating], Sequence[float]],
    x: Union[NDArray[np.floating], Sequence[float]],
    importance_values: Union[NDArray[np.floating], Sequence[float]],
    cmap: Union[str, matplotlib.colors.LinearSegmentedColormap] = "coolwarm",
    figsize: Union[tuple[int, int], None] = None,
    bounds: Union[tuple[float, float], None] = None,
    background: bool = True,
) -> matplotlib.pyplot.Axes:
    """
    Color a 1D series based on feature importance values.

    Parameters
    ----------
    y : array-like(float, ndim=1)
        Series values, such as spectral (signal) intensities.

    x : array-like(float, ndim=1)
        Location series was measured at, for example, wavelengths or energy.

    importance_values : array-like(float, ndim=1)
        Importance value assigned. Should have the same length as `x` and `y`.

    cmap : str or matplotlib.colors.LinearSegmentedColormap, optional(default="coolwarm")
        Name of matplotlib colormap to use.

    figsize : tuple(int, int), optional(default=None)
        Size of figure to plot.

    bounds : tuple(float, float), optional(default=None)
        Bounds to color based on; if unspecified uses min/max of importance_values.

    background : bool, optional(default=True)
        Whether or not to plot the uncolored (gray) spectrum behind the colored points.

    Returns
    -------
    axes : matplotlib.pyplot.Axes
        Axes the result is plotted on.

    Raises
    ------
    ValueError
        If the lengths of x, y, and importance_values do not match.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    importance_values = np.asarray(importance_values, dtype=np.float64).ravel()
    if not (len(x) == len(y) and len(x) == len(importance_values)):
        raise ValueError(
            f"Lengths of x ({len(x)}), y ({len(y)}), and importance_values ({len(importance_values)}) should match."
        )

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    if background:
        axes.plot(x, y, color="k", alpha=0.10)

    lc = _color_1d(
        x=x,
        y=y,
        importances=np.asarray(importance_values, dtype=np.float64).ravel(),
        cmap=cmap,
        interp=True,  # Shouldn't matter if x.shape == y.shape
        bounds=bounds,
    )
    line = axes.add_collection(lc)
    fig.colorbar(line, ax=axes)

    y_range = y.max() - y.min()
    axes.set_xlim(x.min(), x.max())
    axes.set_ylim(y.min() - 0.05 * y_range, y.max() + 0.05 * y_range)

    return axes


def bokeh_color_spectrum(
    y: Union[NDArray[np.floating], Sequence[float]],
    x: Union[NDArray[np.floating], Sequence[float]],
    importance_values: Union[NDArray[np.floating], Sequence[float]],
    palette=Spectral10,
    y_axis_type: Union[str, None] = None,
) -> None:
    """
    Color a 1D spectrum based on feature importance values using Bokeh.

    Parameters
    ----------
    y : array-like(float, ndim=1)
        Series values, such as spectral (signal) intensities.

    x : array-like(float, ndim=1)
        Location series was measured at, for example, wavelengths or energy.

    importance_values : array-like(float, ndim=1)
        Importance value assigned to each feature. Should have the same length as `x` and `y`.

    palette : bokeh.palettes, optional(default=Spectral10)
        Color palette to use (https://docs.bokeh.org/en/latest/docs/reference/palettes.html).

    y_axis_type : str, optional(default=None)
        Optional transformation of y axis, e.g., y_axis_type="log".

    Notes
    -----
    If using this in a Jupyter Norebook be sure to set the output correctly (see example below)

    Examples
    --------
    >>> from bokeh.io import output_notebook()
    >>> output_notebook()
    >>> bokeh_color_spectrum(...)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    importance_values = np.asarray(importance_values, dtype=np.float64).ravel()
    if not (len(x) == len(y) and len(x) == len(importance_values)):
        raise ValueError("Lengths of x, y, and importance_values should match.")

    spectrum_df = pd.DataFrame(
        np.vstack((x, y, importance_values)).T,
        columns=("Center", "Value", "Importance"),
    )

    datasource = ColumnDataSource(spectrum_df)
    color_mapping = LinearColorMapper(
        low=spectrum_df["Importance"].min(),
        high=spectrum_df["Importance"].max(),
        palette=palette,
    )

    plot_figure = figure(
        title="Importance-Colored Value",
        tools=(
            "pan, wheel_zoom, box_select, box_zoom, lasso_select, crosshair, tap, examine, reset"
        ),
        x_axis_label="Center",
        y_axis_label="Value",
        y_axis_type=y_axis_type,
        width=900,
        aspect_ratio=1.5,
    )

    plot_figure.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div>
            <span style='font-size: 16px; color: #224499'>Center:</span>
            <span style='font-size: 18px'>@Center</span>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>Importance:</span>
            <span style='font-size: 18px'>@Importance</span>
        </div>
    </div>
    """
        )
    )

    plot_figure.line(
        x="Center",
        y="Value",
        source=datasource,
        color="black",
        line_width=1,
        line_alpha=0.25,
    )
    plot_figure.scatter(
        x="Center",
        y="Value",
        source=datasource,
        color=dict(field="Importance", transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4,
    )
    show(plot_figure)
