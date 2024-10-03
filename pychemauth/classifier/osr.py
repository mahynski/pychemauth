"""
Open Set Recognition Models.

author: nam
"""
import sys
import copy
import keras
import scipy
import torch
import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dime import DIME

from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.preprocessing import LabelEncoder

from pychemauth.utils import _multi_cm_metrics, _occ_cm_metrics
from pychemauth import utils

from typing import Union, Sequence, Callable, Any, ClassVar, TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pychemauth.utils.NNTools import XLoader


class DeepOOD:
    """Deep neural network out-of-distribution (OOD) tools and models."""

    class ProbaFeatureClf:
        """
        Base class for prefit classifiers in `OpenSetClassifier` when using featurized data.

        This class is a wrapper which modifies the behavior of a probabilistic classifier to return a single prediction.
        Deep classification models end in a softmax layer predicting a floating point probability for each class when model.predict is called, akin to model.predict_proba in sklearn. To make these have the same behaviors, this class is needed.
        """

        def __init__(self, model=None) -> None:
            """
            Instantiate the class.

            Parameters
            ----------
            model : object, optional(default=None)
                Predictive model to use; this assumes `.predict` is implemented.
            """
            self.model = model

        def convert(
            self, probabilities: NDArray[np.floating]
        ) -> NDArray[np.integer]:
            """
            Convert an array of probabilities to a single prediction.

            Parameters
            ----------
            probabilities : ndarray(float, ndim=2)
                2D array of class proabilities for each input observation.

            Returns
            -------
            prediction : ndarray(int, ndim=1)
                Index of highest probability for each row in `probabilities`.
            """
            return np.argmax(np.asarray(probabilities), axis=1)

        def predict(self, X_feature: NDArray[Any]) -> NDArray[np.integer]:
            """
            Make a prediction given a featurized input.

            Parameters
            ----------
            X_feature : ndarray
                Input that the model can accept.

            Returns
            -------
            prediction : ndarray(int, ndim=1)
                Index of highest probability for each row in `X_feature`.
            """
            return self.convert(self.model.predict(X_feature))

    class SoftmaxFeatureClf(ProbaFeatureClf):
        """Classification model for featurized data to use with `OpenSetClassifier` when the classifier is a prefit, deep model and `DeepOOD.Softmax` is the outlier model chosen."""

        def predict(self, X_feature: NDArray[Any]) -> NDArray[np.integer]:
            """
            Make a prediction given a featurized input.

            Parameters
            ----------
            X_feature : ndarray
                Input that the model can accept.  Data iterators are not supported at this time.

            Returns
            -------
            prediction : ndarray(int, ndim=1)
                Index of highest probability for each row in `X_feature`.
            """
            return self.convert(X_feature)

    class EnergyFeatureClf(ProbaFeatureClf):
        """Classification model for featurized data to use with `OpenSetClassifier` when the classifier is a prefit, deep model and `DeepOOD.Energy` is the outlier model chosen."""

        def __init__(self) -> None:
            """Instantiate the class."""
            self.model = keras.layers.Softmax()

        def predict(self, X_feature: NDArray[Any]) -> NDArray[np.integer]:
            """
            Make a prediction given a featurized input.

            Parameters
            ----------
            X_feature : ndarray
                Input that the model can accept.

            Returns
            -------
            prediction : ndarray(int, ndim=1)
                Index of highest probability for each row in `X_feature`.
            """
            return self.convert(self.model(X_feature))

    class DIMEFeatureClf(ProbaFeatureClf):
        """Classification model for featurized data to use with `OpenSetClassifier` when the classifier is a prefit, deep model and `DeepOOD.DIME` is the outlier model chosen."""

        def __init__(
            self,
            model_loader: Callable[[], keras.Model],
            input_layer: int = 0,
            output_layer: int = -1,
        ) -> None:
            """
            To make this object picklable (e.g., for use in GridSearchCV) we load the model only when it is needed using a function which is picklable.

            Parameters
            ----------
            model_loader : callable
                Function that instantiates a new keras.Model when called (with no inputs).  Should be picklable if combined with GridSearchCV, etc.

            input_layer : int, optional(default=0)
                Index of the `model.layers` whose input is the featurized data. If no featurization is used, the default of 0 is the correct value.

            output_layer : int, optional(default=-1)
                Index of the `model.layers` whose output is the class probabilities.

            Example
            -------
            >>> DIMEFeatureClf(model_loader=utils.HuggingFace.from_pretrained(model_id="mahynski/2d-cnn-demo", input_layer=-2)
            """
            self.model_loader = model_loader
            self.input_layer = input_layer
            self.output_layer = output_layer
            self.model_ = None

        @property
        def model(self) -> keras.Model:
            """Overload the `.model` property so that it is not loaded until it is needed."""
            if self.model_ is None:
                m_ = self.model_loader()
                self.model_ = keras.Model(
                    inputs=m_.layers[
                        self.input_layer
                    ].input,  # Featurization goes up to this layer
                    outputs=m_.layers[self.output_layer].output,
                )
            return self.model_

    class _ODBase:
        """Base model for out-of-distribution detectors for deep neural networks."""

        model: ClassVar[Union[keras.Model, None]]
        threshold: ClassVar[float]
        alpha: ClassVar[float]
        is_fitted_: bool
        _X_train_scores: ClassVar[NDArray[np.floating]]

        def set_params(self, **parameters: Any) -> "DeepOOD._ODBase":
            """Set parameters; for consistency with scikit-learn's estimator API."""
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self

        def get_params(self, deep: bool = False) -> dict[str, Any]:
            """Get parameters; for consistency with scikit-learn's estimator API."""
            raise NotImplementedError

        def predict(
            self, X: Union[NDArray[np.floating], "XLoader"]
        ) -> NDArray[np.bool_]:
            """
            Predict if samples belong to the known distribution.

            Parameters
            ----------
            X : ndarray(float) or data iterator
                Data the `model` can accept as input.

            Returns
            -------
            inlier : ndarray(bool, ndim=1)
                Array of booleans where inliers are considered `True`.
            """
            check_is_fitted(self, "is_fitted_")
            return (
                self.score_samples(X) >= self.threshold
            )  # When numerical precision is an issue, ">=" helps ensure points that should be accepted are vs. just ">"

        def score_samples(
            self,
            X: Union[NDArray[np.floating], "XLoader"],
            featurized: bool = False,
        ) -> NDArray[np.floating]:
            """Score the samples."""
            raise NotImplementedError

        def fit(
            self, X: Union[NDArray[np.floating], "XLoader"]
        ) -> "DeepOOD._ODBase":
            """
            Fit the detector.

            Parameters
            ----------
            X : ndarray(float) or data iterator
                Training data the model can accept as input.

            Returns
            -------
            self
            """
            if not (0.0 < self.alpha < 1.0):
                raise ValueError("alpha should be between 0 and 1")

            # 1. Check the model ends in a softmax or logistic so that the output is a probability.
            if self.model is not None:
                _ = DeepOOD._check_end(self.model, disable=False)

            # 2. Compute threshold cutoff.
            self.set_params(
                **{
                    "_X_train_scores": self.score_samples(
                        X, featurized=True if self.model is None else False
                    ),
                }
            )

            self.set_params(
                **{
                    "threshold": np.percentile(  # Score below this will be outlier
                        self._X_train_scores,
                        self.alpha * 100.0,
                        method="lower",  # When numerical precision is an issue, this helps ensure the boundary is less than the score for points that should be accepted
                    ),
                    "is_fitted_": True,
                }
            )

            return self

        def visualize(
            self,
            bins: Union[int, Sequence[Any]] = 25,
            ax: Union[matplotlib.pyplot.Axes, None] = None,
            X_test: Union[NDArray[np.floating], "XLoader", None] = None,
            test_label: Union[str, None] = None,
            no_train: bool = False,
            no_threshold: bool = False,
            density: bool = True,
        ) -> matplotlib.pyplot.Axes:
            """
            Visualize the distribution of training scores, and others if desired.

            Parameters
            ----------
            bins : int or sequence
                Number of bins to use in the histogram; if it is a sequence, this defines the edges of the bins where the left edge of the first bin is the first entry, and the right edge of last bin is the last entry.

            ax : matplotlib.pyplot.Axes, optional(default=None)
                Axes to plot the results on. This is created if not provided.

            X_test : ndarray(float) or data iterator, optional(default=None)
                Alternate data the model can accept as input. If `X_test=None` just plot the training data.

            test_label : str, optional(default=None)
                Label for any test data provided.

            no_train : bool, optional(default=False)
                If `no_train=True` do not plot the training data.

            no_threshold : bool, optional(default=False)
                If `no_threshold=True` do not plot the threshold.

            density : bool, optional(default=True)
                Whether to normalize the historgram to a probability distribution.

            Returns
            -------
            ax : matplotlib.pyplot.Axes
                Axes the histogram is plotted on.
            """
            check_is_fitted(self, "is_fitted_")

            if ax is None:
                fig, ax = plt.subplots(nrows=1, ncols=1)

            if not no_train:
                ax.hist(
                    self._X_train_scores,
                    bins=bins,
                    label="Training Set",
                    density=density,
                    alpha=0.5,
                )

            if not no_threshold:
                ax.axvline(
                    self.threshold,
                    color="red",
                    label="Threshold ("
                    + r"$\alpha$ "
                    + f'= {"%.4f"%(self.alpha)})',  # {"%.2f"%(100.0*self.alpha)}%)',
                )

            if X_test is not None:
                ax.hist(
                    self.score_samples(
                        X_test,
                        featurized=True if self.model is None else False,
                    ),
                    bins=bins,
                    label=test_label,
                    density=density,
                    alpha=0.5,
                )

            ax.legend(loc="best")
            ax.set_xlabel("Score")

            return ax

    def _check_end(
        model: keras.Model, disable: bool = False
    ) -> tuple[bool, Callable[..., Any]]:
        """
        Check a Keras model ends with a softmax or logistic function.

        Parameters
        ----------
        model : keras.Model
            Keras model to check.

        disable : bool, optional(default=False)
            Whether to disable the final activation function in the model, if and only if the
            model architecture is valid.

        Returns
        -------
        valid : bool
            If the model architecture is valid.

        original_activation : function
            Activation function used in the original model.
        """
        valid_activations = [
            keras.activations.softmax,
            keras.activations.sigmoid,
        ]  # Could be softmax (multiclass) or logistic (binary)

        original_activation = model.layers[-1].activation
        valid = False
        if isinstance(model.layers[-1], keras.layers.Dense):
            # Ends with a dense layer with a softmax activation
            if original_activation in valid_activations:
                valid = True
        elif isinstance(model.layers[-1], keras.layers.Activation):
            # Activation specified manually after a linear dense layer
            if original_activation in valid_activations:
                if isinstance(model.layers[-2], keras.layers.Dense):
                    if (
                        model.layers[-2].activation == keras.activations.linear
                    ):  # Must not be an activation here
                        valid = True

        if valid:
            if disable:
                model.layers[-1].activation = None

        return valid, original_activation

    class DIME(OutlierMixin, _ODBase):
        """
        Use DIME to detect out-of-distribution points.

        This is just a wrapper for the code originally developed in [1].
        The code is available at https://github.com/sartorius-research/dime.pytorch

        Notes
        -----
        From [1]: "By approximating the training set embedding into feature space as a linear hyperplane,
        we derive a simple, unsupervised, highly performant and computationally efficient method [to detect
        out-of-distribution examples during prediction time]."  Essentially, truncated SVD is performed on the
        featurized training data; these features are assumed here to be the 2D output of the model at some
        intermediate stage.  For example, the output of a convolutional base before global average pooling
        and subsequent classification with a dense head.  The user can create a model from a pretrained
        one as in the example below; this allows the user to select the stage at which OOD detection is
        performed.

        References
        ----------
        1. Sjoegren and Trygg, arXiv (2021) https://arxiv.org/pdf/2108.10673.pdf

        Examples
        --------
        A simple example of using a pretrained model as a featurizer.
        >>> model = keras.models.load_model('my_model.pkl') # Model: [CNN Base] -> GAP -> Dense (see CNNFactory)
        >>> featurizer = keras.Sequential(orig_model.layers[:-1]) # Exclude the final Dense layer
        >>> ood = DeepOOD.DIME(
        ...     model=featurizer,
        ...     alpha=0.01,
        ...     k=20
        ... )
        >>> ood.fit(X_train)
        >>> ax = ood.visualize(X_test=X_test)

        We can also train the model by pre-featurizing the dataset.
        >>> ood = DeepOOD.DIME(
        ...     model=None,
        ...     alpha=0.01,
        ...     k=20
        ... )
        >>> X_feature = featurizer.predict(X_train) # Expensive step
        >>> _ = ood.fit( # Essentially instantaneous
        ...     X_feature
        ... )
        >>> ax = ood.visualize(X_test=featurizer.predict(X_test))
        """

        k: ClassVar[int]

        def __init__(
            self, model: Union[keras.Model, None], k: int, alpha: float = 0.05
        ) -> None:
            """
            Instantiate the class.

            Parameters
            ----------
            model : keras.Model or None
                A trained Keras model which outputs a transformed version of the input. For example, a CNN base
                used for transfer learning.  This must implement a `predict` method. Setting `model=None`
                will make the detector assume that all `X` passed to it have already been featurized. It can be
                advantageous to pre-featurize the data by running it through this model before optimizing a
                detector since it can dramatically increase the speed of training by avoiding this (usually
                expensive) calculation.

            k : scalar(int)
                Number of dimensions in the final embedding.

            alpha : scalar(float), optional(default=0.05)
                Type I error rate.

            """
            self.set_params(**{"model": model, "k": k, "alpha": alpha})

        def get_params(self, deep: bool = False) -> dict[str, Any]:
            """Get parameters; for consistency with scikit-learn's estimator API."""
            if deep:
                raise NotImplementedError
            return {
                "model": self.model,  # This can't really be deep copied easily
                "k": self.k,
                "alpha": self.alpha,
            }

        def score_samples(
            self,
            X: Union[NDArray[np.floating], "XLoader"],
            featurized: bool = False,
        ) -> NDArray[np.floating]:
            """
            Compute the (negative) distance to the modeled embedding for each observation.

            Parameters
            ----------
            X : ndarray(float) or data iterator
                Training data the `model` can accept as input.  If `model=None` assume that `X` is
                already featurized.  It can be advantageous to pre-featurize the data by running it through
                this model before optimizing a DIME OOD detector since it can dramatically increase the speed
                of training by avoiding this (usually expensive) calculation.

            featurized : bool, optional(default=False)
                Whether `X` has already been featurized already.  If `model=None` this is ignored
                and it is assumed `X` has been featurized.

            Returns
            -------
            scores : ndarray(float, ndim=1)
                Negative distance for each observation in X. The lower, the more abnormal.
            """

            def _scores(X):
                """Compute the scores for a batch of data."""
                if featurized or (self.model is None):
                    X_ = torch.tensor(X)
                else:
                    X_ = torch.tensor(self._featurize(X))
                scores = -self.dime.distance_to_hyperplane(X_).numpy()

                return scores

            if utils.NNTools._is_data_iter(X):
                scores = []
                for X_batch_, _ in X:
                    if X_batch_.size > 0:  # Check if batch is empty
                        scores.append(_scores(X_batch_))
                return np.concatenate(scores)
            else:
                return _scores(X)

        def _featurize(
            self, X: Union[NDArray[np.floating], "XLoader"]
        ) -> NDArray[Any]:
            """Featurize the data."""
            if utils.NNTools._is_data_iter(X):
                X_feature = []
                for X_batch_, _ in X:
                    if X_batch_.size > 0:  # Check if batch is empty
                        X_feature.append(self.model.predict(X_batch_))  # type: ignore[union-attr]
                return np.concatenate(X_feature)
            else:
                return self.model.predict(X)  # type: ignore[union-attr]

        def fit(self, X: Union[NDArray[np.floating], "XLoader"]) -> "DIME":
            """
            Fit the detector.

            Parameters
            ----------
            X : ndarray(float) or data iterator
                Training data the `model` can accept as input.  If `model=None` assume that `X` is
                already featurized.  It can be advantageous to pre-featurize the data by running it through
                this model before optimizing a DIME OOD detector since it can dramatically increase the speed
                of training by avoiding this (usually expensive) calculation.

            Returns
            -------
            self
            """
            if not (0.0 < self.alpha < 1.0):
                raise ValueError("alpha should be between 0 and 1")
            if self.k < 1:
                raise ValueError("k should be at least 1")

            # 1. Fit DIME on features
            if self.model is None:
                X_feature = X  # Assume X is already featurized
            else:
                X_feature = self._featurize(X)

            self.dime = DIME(
                explained_variance_threshold=self.k, n_percentiles=10000
            ).fit(
                torch.tensor(X_feature),
                calibrate_against_trainingset=True,
            )

            self.set_params(
                **{
                    "_X_train_scores": self.score_samples(
                        X_feature, featurized=True
                    ),
                }
            )

            self.set_params(
                **{
                    "threshold": np.percentile(  # Score below this will be outlier
                        self._X_train_scores, self.alpha * 100.0
                    ),
                    "is_fitted_": True,
                }
            )

            return self

    class Energy(OutlierMixin, _ODBase):
        """
        Use an energy-based score to detect out-of-distribution points.

        Notes
        -----
        The softmax logits are used to compute a "Helmholtz free energy" for each observation.  These free
        energies are collected over the training data, which should be composed only of samples which are,
        "in-distribution", and a threshold is established. When a test point is predicted, if the (negative)
        free energy score is below this threshold the point is considered out-of-distribution.

        References
        ----------
        1. Liu et al., NIPS (2020) https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf

        Examples
        --------
        A simple example of using a pretrained model as a featurizer.
        >>> model = keras.models.load_model('my_model.pkl') # Model: [CNN Base] -> GAP -> Dense (see CNNFactory)
        >>> ood = DeepOOD.Energy(
        ...     model=model,
        ...     alpha=0.01,
        ...     T=1.0
        ... )
        >>> ood.fit(X_train)
        >>> ax = ood.visualize(X_test=X_test)

        We can also train the model by pre-featurizing the dataset.
        >>> ood = DeepOOD.Energy(
        ...     model=None,
        ...     alpha=0.01,
        ...     T=1.0
        ... )
        >>> featurizer = model
        >>> featurizer.layers[-1].activation = None # Remove softmax activation to get logits as output
        >>> X_logits = featurizer.predict(X_train) # Expensive step
        >>> _ = ood.fit( # Essentially instantaneous
        ...     X_logits
        ... )
        >>> ax = ood.visualize(X_test=featurizer.predict(X_test))
        """

        def __init__(
            self, model: keras.Model, alpha: float = 0.05, T: float = 1.0
        ) -> None:
            """
            Instantiate the class.

            Parameters
            ----------
            model : keras.Model
                A trained Keras classification model which outputs a probability.  This model should terminate
                in a softmax or logistic activation function. Setting `model=None` will make the detector assume
                that all `X` passed to it have already been featurized (i.e., are logits not raw inputs). It can be
                advantageous to pre-featurize the data by running it through this model before optimizing a
                detector since it can dramatically increase the speed of training by avoiding this (usually
                expensive) calculation.

            alpha : scalar(float), optional(default=0.05)
                The free energy of each obseravtion is computed over the training set; the lower alpha percentile
                of this is taken as a threshold below which a prediction is considered an outlier.

            T : scalar(float), optional(default=1.0)
                Temperature scale to use.
            """
            self.set_params(**{"model": model, "alpha": alpha, "T": T})

        def get_params(self, deep: bool = False) -> dict[str, Any]:
            """Get parameters; for consistency with scikit-learn's estimator API."""
            if deep:
                raise NotImplementedError
            return {
                "model": self.model,  # This can't really be deep copied easily
                "alpha": self.alpha,
                "T": self.T,
            }

        def score_samples(
            self,
            X: Union[NDArray[np.floating], "XLoader"],
            featurized: bool = False,
        ) -> NDArray[np.floating]:
            """
            Compute the energy score for each observation.

            Parameters
            ----------
            X : ndarray(float) or data iterator
                Data the model can accept as input.

            featurized : bool, optional(default=False)
                Whether `X` has already been featurized already; i.e., if the `X` matrix is the logits for
                each observation not the raw data.  If `model=None` this is ignored and it is assumed `X`
                has been featurized.

            Returns
            -------
            scores : ndarray(float, ndim=1)
                Negative energy score for each observation in X. The lower, the more abnormal.
            """
            if self.T < 0:
                raise ValueError("T should be non-negative")

            if featurized or (self.model is None):
                featurized = True

            # If valid, deactivate activation function to obtain raw logits
            if self.model is not None:
                valid, original_activation = DeepOOD._check_end(
                    self.model, disable=True
                )
            else:
                valid = True

            if valid:

                def negative_energy(logits) -> NDArray[np.floating]:
                    return self.T * scipy.special.logsumexp(
                        np.asarray(logits) / self.T, axis=1
                    )

                try:
                    if utils.NNTools._is_data_iter(X):
                        scores_ = []
                        for X_batch_, _ in X:
                            if X_batch_.size > 0:  # Check if batch is empty
                                scores_.append(
                                    negative_energy(
                                        X_batch_
                                        if featurized
                                        else self.model.predict(X_batch_)
                                    )
                                )
                        scores = np.concatenate(scores_)
                    else:
                        scores = negative_energy(
                            X if featurized else self.model.predict(X)
                        )
                except Exception as e:
                    if self.model is not None:
                        self.model.layers[
                            -1
                        ].activation = original_activation  # Return model to original state
                    raise Exception(e)
                else:
                    if self.model is not None:
                        self.model.layers[
                            -1
                        ].activation = original_activation  # Return model to original state
                    return scores
            else:
                raise Exception("Invalid model; cannot compute energy scores.")

    class Softmax(OutlierMixin, _ODBase):
        """
        Use the softmax confidence score to detect out-of-distribution points.

        Notes
        -----
        From [1]: "Correctly classified examples tend to have greater maximum softmax probabilities
        than erroneously classified and out-of-distribution examples, allowing for their detection."
        Essentially, class probabilities are computed for each training observation.  The maximum
        probability is the predicted class.  These maximums are collected over the training data, which
        should be composed only of samples which are "in-distribution", and a threshold is established.
        When a test point is predicted, if the maximum softmax score is below this threshold the
        point is considered out-of-distribution.

        References
        ----------
        1. Hendrycks and Gimpel, ICLR (2017) https://arxiv.org/pdf/1610.02136.pdf

        Examples
        --------
        A simple example of using a pretrained model as a featurizer.
        >>> model = keras.models.load_model('my_model.pkl') # Model: [CNN Base] -> GAP -> Dense (see CNNFactory)
        >>> ood = DeepOOD.Softmax(
        ...     model=model,
        ...     alpha=0.01,
        ... )
        >>> ood.fit(X_train)
        >>> ax = ood.visualize(X_test=X_test)

        We can also train the model by pre-featurizing the dataset.
        >>> ood = DeepOOD.Softmax(
        ...     model=None,
        ...     alpha=0.01,
        ... )
        >>> X_probabilities = model.predict(X_train) # Expensive step
        >>> _ = ood.fit( # Essentially instantaneous
        ...     X_probabilities
        ... )
        >>> ax = ood.visualize(X_test=model.predict(X_test))
        """

        def __init__(self, model: keras.Model, alpha: float = 0.05) -> None:
            """
            Instantiate the class.

            Parameters
            ----------
            model : keras.Model
                A trained Keras classification model which outputs a probability.  This model should terminate
                in a softmax or logistic activation function. Setting `model=None` will make the detector assume
                that all `X` passed to it have already been featurized (i.e., are class probabilities not raw inputs).
                It can be advantageous to pre-featurize the data by running it through this model before optimizing
                a detector since it can dramatically increase the speed of training by avoiding this (usually
                expensive) calculation.

            alpha : float, optional(default=0.05)
                The probabilities of the predicted class (max probability for an observation) is computed
                over the training set; the lower alpha percentile of this is taken as a threshold below which
                a prediction is considered an outlier.
            """
            self.set_params(**{"model": model, "alpha": alpha})

        def get_params(self, deep: bool = False) -> dict[str, Any]:
            """Get parameters; for consistency with scikit-learn's estimator API."""
            if deep:
                raise NotImplementedError
            return {
                "model": self.model,  # This can't really be deep copied easily
                "alpha": self.alpha,
            }

        def score_samples(
            self,
            X: Union[NDArray[np.floating], "XLoader"],
            featurized: bool = False,
        ) -> NDArray[np.floating]:
            """
            Compute the softmax score for each observation.

            Parameters
            ----------
            X : ndarray(float) or iterator
                Data the model can accept as input.

            featurized : bool, optional(default=False)
                Whether `X` has already been featurized already; i.e., if the `X` matrix is the class
                probabilities for each observation not the raw data.  If `model=None` this is ignored
                and it is assumed `X` has been featurized.

            Returns
            -------
            scores : ndarray(float, ndim=1)
                Maximum class probability predicted for each observation in X. The lower, the more abnormal.
            """
            if featurized or (self.model is None):
                featurized = True

            def softmax_confidence(probabilities):
                """Compute softmax confidence for a batch of data."""
                return np.max(probabilities, axis=1)

            if utils.NNTools._is_data_iter(X):
                scores = []
                for X_batch_, _ in X:
                    if X_batch_.size > 0:  # Check if batch is empty
                        scores.append(
                            softmax_confidence(
                                X_batch_
                                if featurized
                                else self.model.predict(X_batch_)  # type: ignore[union-attr]
                            )
                        )
                return np.concatenate(scores)
            else:
                return softmax_confidence(
                    X if featurized else self.model.predict(X)  # type: ignore[union-attr]
                )


class OpenSetClassifier(ClassifierMixin, BaseEstimator):
    """
    Train a composite classifier with a reject option to work under open-set conditions.

    Parameters
    ----------
    clf_model : object, optional(default=None)
        Unfitted or fitted classification model. Must support `.fit()` and `.predict()` methods. Only
        the latter is required if the `clf_prefit=True`.

    outlier_model : object, optional(default=None)
        Unfitted outlier detection model. Must support `.fit()` and `.predict()` methods.
        This should return a value of `inlier_value` for points which are considered inliers.
        If `None` then all points will be passed to the classifier.

    clf_prefit : bool, optional(default=False)
        Whether the `clf_model` is already fit or not; if `clf_prefit=True` the model will not be
        refit.  This is advantageous when using a model which is expensive to train, such as a
        deep neural network.  In fact, if `clf_model` is a Keras model, `clf_prefit=True` is required.

    clf_kwargs : dict, optional(default={})
        Keyword arguments to instantiate the classification model with.  If `clf_prefit=True` these
        are ignored since they are not used.

    outlier_kwargs : dict, optional(default={})
        Keyword arguments to instantiate the outlier model with.  If `outlier_model=None` these are
        ignored since they are not used.

    known_classes : array_like(int or str, ndim=1), optional(default=None)
        A list of classes which the classifier is responsible for recognizing. If `None`,
        all unique values of `y` are used; otherwise, `y` is filtered to only include these
        instances when training the classifier, if training is performed.

    inlier_value : scalar(float, int, or str), optional(default=1)
        The value `outlier_model.predict()` returns for inlier class(es).  Many sklearn routines
        return +1 for inlier vs. -1 for outlier; other routines sometimes use 0 for outlier.
        As a result, we simply check for the inlier value (+1 by default) for greater flexibility.
        Ignored if `outlier_model=None`.

    unknown_class : scalar(int or str), optional(default="Unknown")
        The name or index to assign to points which are considered unknown according to the
        `outlier_model`. Ignored if `outlier_model=None`.

    score_metric : scalar(str), optional(default="TEFF")
        Default scoring metric to use. See `figures_of_merit` outputs for options.

    clf_style : scalar(str), optional(default="hard")
        Style of classification model; "hard" models assign each point to a single category, while
        "soft" models can make multiple assignments (multi-label), including to an unknown category.

    score_using : scalar(int or str), optional(default="all")
        Which classes to use for scoring.  The default "all" computes TEFF, etc. using all
        `known_classes`; intead, if a single class name is provided the metrics are computed
        to reflect this model as a one-class classifier (OCC), such as SIMCA.  OCC models
        return a binary yes/no membership decision, but not both, so these are 'hard' models.
        An error will be thrown if this is incorrectly specified.

    Note
    ----
    This is composed of an outlier model, which is called first to determine which points
    are considered inliers and which are outliers, and a classification model, which is
    resposible for classifying the inliers (closed set).

    The type of `unknown_class` should mimic that of the raw data; i.e., if classes in y are
    strings unknown_class should be a string (default="Unknown"). Integers may also be used.

    Warning
    -------
    The TSPS formula changes depending on whether a hard or soft classifier is being used.
    Also see :class:`pychemauth.classifier.plsda.PLSDA`.
    """

    def __init__(
        self,
        clf_model: Union[object, None] = None,
        outlier_model: Union[object, None] = None,
        clf_prefit: bool = False,
        clf_kwargs: dict[str, Any] = {},
        outlier_kwargs: dict[str, Any] = {},
        known_classes: Union[Sequence[int], Sequence[str], None] = None,
        inlier_value: Union[float, int, str] = 1,
        unknown_class: Union[int, str] = "Unknown",
        score_metric: str = "TEFF",
        clf_style: str = "hard",
        score_using: Union[int, str] = "all",
    ) -> None:
        """Initialize the class."""
        self.set_params(
            **{
                "clf_model": clf_model,
                "outlier_model": outlier_model,
                "clf_prefit": clf_prefit,
                "clf_kwargs": clf_kwargs,
                "outlier_kwargs": outlier_kwargs,
                "known_classes": known_classes,
                "inlier_value": inlier_value,
                "unknown_class": unknown_class,
                "score_metric": score_metric,
                "clf_style": clf_style,
                "score_using": score_using,
            }
        )

    def set_params(self, **parameters: Any) -> "OpenSetClassifier":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "clf_model": self.clf_model,
            "outlier_model": self.outlier_model,
            "clf_prefit": self.clf_prefit,
            "clf_kwargs": self.clf_kwargs,
            "outlier_kwargs": self.outlier_kwargs,
            "known_classes": self.known_classes,
            "inlier_value": self.inlier_value,
            "unknown_class": self.unknown_class,
            "score_metric": self.score_metric,
            "clf_style": self.clf_style,
            "score_using": self.score_using,
        }

    def _check_category_type(self, y: Any) -> None:
        """Check that categories are same type as `unknown_class` variable."""
        t_ = None
        for t_ in [(int, np.int32, np.int64), (str,)]:
            if isinstance(self.unknown_class, t_):
                use_type = t_
                break
        if t_ is None:
            raise TypeError("unknown_class must be an integer or string")
        if not np.all([isinstance(y_, use_type) for y_ in y]):
            raise ValueError(
                "You must set the 'unknown_class' variable type ({}) the same \
                as y, e.g., both must be int or str".format(
                    [type(y_) for y_ in y]
                )
            )

    def fit(
        self, X: Union[NDArray[np.floating], "XLoader"], y=None
    ) -> "OpenSetClassifier":
        """
        Fit the composite model.

        Parameters
        ----------
        X : array_like(float, ndim=2) or data iterator
            Input feature matrix. Data iterators are only supported if `clf_model` is deep.  Data iterators supply both `X` and `y` values.

        y : array_like(str or int, ndim=1), optional(default=None)
            Class labels or indices. If `X` is a data iterator, leave this as None since the iterator will provide this information.

        Returns
        -------
        self : OpenSetClassifier
            Fitted model.
        """
        self.deep_ = False
        if isinstance(self.clf_model, keras.Model) or isinstance(
            self.clf_model, DeepOOD.ProbaFeatureClf
        ):
            self.deep_ = True

        self.knowns_ = None
        y_check_ = None
        if not self.deep_:
            # Shallow models should have 2D input, deep models might have higher dimensional tensors
            if utils.NNTools._is_data_iter(X):
                raise NotImplementedError(
                    "Data iterators are only supported for deep models."
                )

            X, y = check_X_y(X, y, accept_sparse=False)
            self.n_features_in_ = X.shape[1:]
            self._check_category_type(y.ravel())  # type: ignore[union-attr]
            assert self.unknown_class not in set(
                y
            ), "unknown_class value is already taken."
            y_check_ = y
        else:
            # Deep models support data iterators
            if utils.NNTools._is_data_iter(X):
                (
                    _,
                    _,
                    unique_targets,
                    X_batch,
                    _,
                ) = utils.NNTools._summarize_batches(X)
                self.n_features_in_ = X_batch.shape[1:]
                self._check_category_type(unique_targets.keys())
                assert (
                    self.unknown_class
                    not in set(  # Closed-set clf will not know about these
                        unique_targets.keys()
                    )
                ), "unknown_class value is already taken."
                y_check_ = np.array(list(unique_targets.keys()))
            else:
                X, y = np.asarray(X), np.asarray(y)
                assert X.shape[0] == y.shape[0]
                self.n_features_in_ = X.shape[1:]
                self._check_category_type(y.ravel())
                assert self.unknown_class not in set(
                    y
                ), (  # Closed-set clf will not know about these
                    "unknown_class value is already taken."
                )
                y_check_ = y

        if not (self.clf_style in ["soft", "hard"]):
            raise ValueError("clf_style should be either 'hard' or 'soft'.")

        # Remove any classes the classification model should not be responsible for
        # learning.
        if self.known_classes is None:  # Consider all training examples.
            self.knowns_ = np.unique(y_check_)
        else:  # Manually specified - possibly overwrite
            self.knowns_ = np.unique(self.known_classes)

        # Filter for only the known classes
        if utils.NNTools._is_data_iter(X):
            # Create a new iterator which is filtered by the known classes
            X = copy.deepcopy(X)
            X.set_include(self.knowns_)  # type: ignore[union-attr]
        else:
            known_mask = np.array(
                [y_ in self.knowns_ for y_ in y_check_], dtype=bool
            )
            if np.sum(known_mask) == 0:
                raise Exception(
                    "There are no known classes in the training set."
                )

        # For sklearn compatibility - not used
        self.classes_ = self.knowns_.tolist() + [self.unknown_class]

        # Check that self.score_using is valid
        if (self.score_using not in self.knowns_) and (
            self.score_using.lower() != "all"
        ):
            raise ValueError(
                "score_using should be 'all' or one of the classes trained on."
            )

        # Train outlier detection first, since this how it will work at prediction
        # time.  This needs to remember the data that the classifier will use for
        # training and flag anything different (covariate/semantic shift).  Thus, this needs
        # to train on the knowns_ only.
        if self.outlier_model is not None:
            try:
                self.od_ = self.outlier_model(**self.outlier_kwargs)
                if utils.NNTools._is_data_iter(X):
                    self.od_.fit(X)
                else:
                    self.od_.fit(X[known_mask, :])
            except:
                raise Exception(
                    f"Unable to fit outlier model : {sys.exc_info()[0]}"
                )
        else:
            self.od_ = None

        # Now fit the classifier
        if not self.clf_prefit:
            # Deep models must be prefit and data iterators are only supported for deep models
            # so models here are "shallow" and data is raw 2D input here.
            if self.deep_:
                raise Exception("Deep models must be prefit.")

            # Predict for all X for simplicity. The composite mask will only allow knowns
            # which are not outliers through.
            if self.od_ is not None:
                inlier_mask = self.od_.predict(X) == self.inlier_value
                composite_mask = known_mask & inlier_mask
            else:
                composite_mask = known_mask

            if np.sum(composite_mask) == 0:
                raise Exception(
                    "There are no inlying known classes in the training set."
                )

            if len(np.unique(y[composite_mask])) < 2:
                raise Exception(
                    "There are less than 2 distinct classes available for training."
                )

            try:
                self.clf_ = self.clf_model(**self.clf_kwargs)  # type: ignore[operator]
                self.clf_.fit(X[composite_mask, :], y[composite_mask])
            except:
                raise Exception(
                    f"Unable to fit classification model : {sys.exc_info()[0]}"
                )
        else:
            # Deep or shallow models could be prefit
            if self.deep_:
                # Deep models need to have their .predict() method wrapped to output the prediction index instead of probabilities.
                if isinstance(self.clf_model, DeepOOD.ProbaFeatureClf):
                    # User has already wrapped the model
                    self.clf_ = self.clf_model
                else:
                    # User is providing some generic probabilistic model that needs to have its output wrapped
                    self.clf_ = DeepOOD.ProbaFeatureClf(model=self.clf_model)
            else:
                # Otherwise just use the model provided.
                self.clf_ = self.clf_model

        self.is_fitted_ = True
        return self

    def predict(
        self, X: Union[NDArray[np.floating], "XLoader"]
    ) -> Union[Sequence[Any], NDArray[Any]]:
        """
        Make a prediction.

        Parameters
        ----------
        X : array_like(float, ndim=2) or data iterator.
            Input feature matrix. Data iterators are only supported if `clf_model` is deep.

        Returns
        -------
        predictions : array_like(int or str, ndim=2)
            Class, or classes, assigned to each point.  Points considered outliers are assigned the value `unknown_class`. This is returned as a list to accommodate multi-label, or soft, classifiers which can return multiple predictions for each observation.
        """
        check_is_fitted(self, "is_fitted_")

        if not utils.NNTools._is_data_iter(X):
            assert X.shape[1:] == self.n_features_in_

        # 1. Check for outliers
        if self.od_ is None:
            if not self.deep_ and utils.NNTools._is_data_iter(X):
                raise NotImplementedError(
                    "Data iterators are only supported for deep models."
                )
            return self.clf_.predict(X)
        else:
            inlier_mask = self.od_.predict(X) == self.inlier_value

            # 2. Predict on points considered inliers
            predictions: Sequence[Any]
            if not utils.NNTools._is_data_iter(X):
                predictions = [[]] * len(X)

                if np.sum(inlier_mask) > 0:
                    pred = self.clf_.predict(X[inlier_mask, :])
            else:
                predictions = [[]] * (
                    (len(X) - 1) * X.batch_size + len(X[len(X) - 1][0])  # type: ignore[union-attr]
                )

                X = copy.deepcopy(X)
                X._set_filter(inlier_mask)  # type: ignore[union-attr]

                if np.sum(inlier_mask) > 0:
                    pred = self.clf_.predict(X)

            j = 0
            for i in range(len(predictions)):
                if not inlier_mask[i]:
                    predictions[i] = (
                        [self.unknown_class]
                        if self.clf_style == "soft"
                        else self.unknown_class
                    )
                else:
                    predictions[i] = pred[j]
                    j += 1

        return predictions

    def fit_predict(
        self, X: Union[NDArray[np.floating], "XLoader"], y=None
    ) -> Union[Sequence[Any], NDArray[Any]]:
        """
        Fit then predict.

        Parameters
        ----------
        X : array_like(float, ndim=2) or data iterator
            Input feature matrix.

        y : array_like(str or int, ndim=1)
            Class labels or indices. If `X` is a data iterator, leave this as None since the iterator will provide this information.

        Returns
        -------
        predictions : array_like(int or str, ndim=2)
            Class, or classes, assigned to each point.  Points considered outliers are
            assigned the value `unknown_class`.
        """
        self.fit(X=X, y=y)
        return self.predict(X=X)

    def score(self, X: Union[NDArray[np.floating], "XLoader"], y=None) -> float:
        """
        Score the prediction.

        Parameters
        ----------
        X : array_like(float, ndim=2) or data iterator
            Columns of features; observations are rows - will be converted to numpy array automatically.

        y : array_like(str or int, ndim=1)
            Ground truth classes - will be converted to numpy array automatically. If `X` is a data iterator, leave this as None since the iterator will provide this information.

        Returns
        -------
        score : scalar(float)
            Score.
        """
        check_is_fitted(self, "is_fitted_")

        if utils.NNTools._is_data_iter(X):
            y = []
            for _, y_batch_ in X:
                if y_batch_.size > 0:  # Check if batch is empty
                    y.append(y_batch_)
            y = np.concatenate(y)
        else:
            X, y = np.asarray(X), np.asarray(y)
            self._check_category_type(y.ravel())
        y_pred = self.predict(X)

        metrics = self.figures_of_merit(y_pred, y)
        if self.score_metric.upper() not in metrics:
            raise ValueError(
                "Unrecognized metric : {}".format(self.score_metric.upper())
            )
        else:
            return metrics[self.score_metric.upper()]

    @property
    def fitted_classification_model(self):
        """Return the fitted classification model."""
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.clf_)

    @property
    def fitted_outlier_model(self):
        """Return the fitted outlier model."""
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.od_)

    def figures_of_merit(
        self,
        predictions: Union[Sequence[Any], NDArray[Any]],
        actual: Union[Sequence[Any], NDArray[Any]],
    ) -> dict[str, Any]:
        """
        Compute figures of merit.

        Parameters
        ----------
        predictions : array_like(str or int, ndim=2)
            Array of values containing the predicted class of points (in
            order). Each row may have multiple entries corresponding to
            multiple class predictions.

        actual : array_like(str or int, ndim=1)
            Array of ground truth classes for the predicted points.  Should
            have only one class per point.

        Returns
        -------
        fom : dict
            Dictionary object with the following attributes. Note that CSPS and CEFF
            are absent in the case of a one-class classifier (`score_using` is a single
            class instead of 'all').

            CM : pandas.DataFrame
                Inputs (index) vs. predictions (columns); akin to a confusion matrix.

            I : pandas.Series
                Number of each class asked to classify.

            CSNS : pandas.Series
                Class sensitivity.

            CSPS : pandas.Series
                Class specificity.

            CEFF : pandas.Series
                Class efficiency.

            TSNS : scalar(float)
                Total sensitivity.

            TSPS : scalar(float)
                Total specificity.

            TEFF : scalar(float)
                Total efficiency.

            ACC : scalar(float)
                Accuracy.

        Note
        ----
        When making predictions about purely extraneous classes (not in training set)
        class efficiency (CEFF) is given as simply class specificity (CSPS)
        since class sensitivity (CSNS) cannot be calculated.  For a one-class
        classifier, TSNS = CSNS.

        References
        ----------
        [1] "Multiclass partial least squares discriminant analysis: Taking the
        right way - A critical tutorial," Pomerantsev and Rodionova, Journal of
        Chemometrics (2018). https://doi.org/10.1002/cem.3030.
        """
        check_is_fitted(self, "is_fitted_")

        # Dummy check that not_assigned and y have same data types
        actual = np.asarray(actual).ravel()
        self._check_category_type(actual)
        assert self.unknown_class not in set(
            actual
        ), "unknown_class value is already taken"

        all_classes = [self.unknown_class] + np.unique(
            np.concatenate((np.unique(actual), self.knowns_))  # type: ignore[arg-type]
        ).tolist()
        encoder = LabelEncoder()
        encoder.fit(all_classes)
        n_classes = len(all_classes)
        use_classes = encoder.classes_[encoder.classes_ != self.unknown_class]

        n = np.zeros((n_classes, n_classes), dtype=int)
        for row, actual_class in zip(predictions, actual):
            kk = encoder.transform([actual_class])[0]
            if isinstance(row, np.ndarray) or isinstance(row, list):
                if self.clf_style.lower() == "hard":
                    raise Exception(
                        "Found multiple class assignments - perhaps you are using a soft model?"
                    )
                for entry in row:
                    try:
                        ll = encoder.transform([entry])[0]
                    except:
                        # Assume that if the encoder does not recognize the entry it is
                        # from the model returning an "unknown" assignment.  This string/value
                        # won't be in the data and is usually specified when the model
                        # is trained so it is difficult to build consistently into this
                        # workflow; this seems to be the best approach.
                        assert (
                            len(row) == 1
                        )  # If "unknown" then this should be the only assignment made
                        ll = encoder.transform([self.unknown_class])[0]
                    n[kk, ll] += 1
            else:
                if self.clf_style.lower() == "soft":
                    raise Exception(
                        "Class assignments not provided as list - perhaps you are using a hard model or OCC?"
                    )
                ll = encoder.transform([row])[0]
                n[kk, ll] += 1

        df = pd.DataFrame(
            data=n, columns=encoder.classes_, index=encoder.classes_
        )
        df = df[df.index != self.unknown_class]  # Trim off row of "UNKNOWN"
        Itot = pd.Series(
            [np.sum(np.array(actual) == kk) for kk in use_classes],
            index=use_classes,
        )
        assert np.sum(Itot) == len(actual)

        if self.score_using.lower() == "all":
            # Evaluate as a multiclass, possibly multilabel (depends on clf_style) classifier
            fom = _multi_cm_metrics(
                df=df,
                Itot=Itot,
                trained_classes=self.knowns_,
                use_classes=use_classes,
                style=self.clf_style,
                not_assigned=self.unknown_class,
                actual=actual,
            )
        else:
            # Evaluate as a OCC where the score_using class is the target class.
            fom = _occ_cm_metrics(
                df=df,
                Itot=Itot,
                target_class=self.score_using,
                trained_classes=self.knowns_,
                not_assigned=self.unknown_class,
                actual=actual,
            )

        return fom

    def _get_tags(self):
        """For compatibility with scikit-learn >=0.21."""
        return {
            "allow_nan": False,
            "array_api_support": False,
            "binary_only": False,
            "multilabel": False,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": True,
            "requires_positive_y": False,
            "_skip_test": ["check_estimators_dtypes"],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
