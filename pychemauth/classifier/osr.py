"""
Open Set Recognition Models.

author: nam
"""
import sys
import copy
import keras
import scipy
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dime import DIME

from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.preprocessing import LabelEncoder

from pychemauth import utils

class DeepOOD:
    """Deep neural network out-of-distribution (OOD) tools and models."""
    class _ODBase:
        """Base model for out-of-distribution detectors for deep neural networks."""
        def set_params(self, **parameters):
            """Set parameters; for consistency with scikit-learn's estimator API."""
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self
        
        def get_params(self, deep=False):
            """Get parameters; for consistency with scikit-learn's estimator API."""
            raise NotImplementedError
            
        def predict(self, X):
            """
            Predict if samples belong to the known distribution.
            
            Parameters
            ----------
            X : ndarray(float) or iterator
                Data the `model` can accept as input.
                
            Returns
            -------
            inlier : ndarray(bool, ndim=1)
                Array of booleans where inliers are considered `True`.
            """
            check_is_fitted(self, "is_fitted_")
            return self.score_samples(X) > self.threshold
        
        def score_samples(self, X, featurized=False):
            """Score the samples."""
            raise NotImplementedError
        
        def fit(self, X):
            """
            Fit the detector.
            
            Parameters
            ----------
            X : ndarray(float) or iterator
                Training data the model can accept as input.
                
            Returns
            -------
            self
            """
            if not (0.0 < self.alpha < 1.0):
                raise ValueError('alpha should be between 0 and 1')
            
            # 1. Check the model ends in a softmax or logistic so that the output is a probability.
            if self.model is not None:
                _  = DeepOOD._check_end(
                    self.model, 
                    disable=False
                )

            # 2. Compute threshold cutoff.
            self._X_train_scores = self.score_samples(
                X, 
                featurized=True if self.model is None else False
            )
            self.threshold = np.percentile( # Score below this will be outlier
                self._X_train_scores,
                self.alpha*100.0
            )
    
            self.is_fitted_ = True

            return self
        
        def visualize(self, bins=25, ax=None, X_test=None, test_label=None, no_train=False, density=True):
            """
            Visualize the distribution of training scores, and others if desired.
            
            Parameters
            ----------
            bins : int
                Number of bins to use in the histogram.
                
            ax : matplotlib.pyplot.Axes, optional(default=None)
                Axes to plot the results on. This is created if not provided.
            
            X_test : ndarray(float) or iterator, optional(default=None)
                Alternate data the model can accept as input. If `X_test=None` just plot the training data.
                
            test_label : str, optional(default=None)
                Label for any test data provided.
                
            no_train : bool, optional(default=False)
                If `no_train=True` do not plot the training data.
                
            density : bool, optional(default)
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
                    label='Training Set', 
                    density=density, 
                    alpha=0.5
                )
            ax.axvline(
                self.threshold, 
                color='red', 
                label=f'Threshold ({"%.2f"%(100.0*self.alpha)}%)'
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
                    alpha=0.5
                )
            
            ax.legend(loc='best')
            ax.set_xlabel('Score')
            
            return ax
        
    def _check_end(model, disable=False):
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
            
        original_activation : keras.activations
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
        out-of-distribution examples during prediction time]."  Essentially, PCA is performed on the 
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
        def __init__(self, model, k, alpha=0.05):
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
            self.set_params(
                **{
                    'model': model, 
                    'k': k, 
                    'alpha': alpha
                }
            )

        def get_params(self, deep=False):
            """Get parameters; for consistency with scikit-learn's estimator API."""
            if deep:
                raise NotImplementedError
            return {
                "model": self.model, # This can't really be deep copied easily
                "k": self.k,
                "alpha": self.alpha
            }
        
        def score_samples(self, X, featurized=False):
            """
            Compute the (negative) distance to the modeled embedding for each observation.
            
            Parameters
            ----------
            X : ndarray(float) or iterator
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
                    scores.append(_scores(X_batch_))
                return np.concatenate(scores)
            else:
                return _scores(X)
            
        def _featurize(self, X):
            """Featurize the data."""
            if utils.NNTools._is_data_iter(X):
                X_feature = []
                for X_batch_, _ in X:
                    X_feature.append(self.model.predict(X_batch_))
                return np.concatenate(X_feature)
            else:
                return self.model.predict(X)
           
        def fit(self, X):
            """
            Fit the detector.
            
            Parameters
            ----------
            X : ndarray(float) or iterator
                Training data the `model` can accept as input.  If `model=None` assume that `X` is 
                already featurized.  It can be advantageous to pre-featurize the data by running it through
                this model before optimizing a DIME OOD detector since it can dramatically increase the speed
                of training by avoiding this (usually expensive) calculation.
                
            Returns
            -------
            self
            """
            if not (0.0 < self.alpha < 1.0):
                raise ValueError('alpha should be between 0 and 1')
            if self.k < 1:
                raise ValueError('k should be at least 1')
        
            # 1. Fit DIME on features
            if self.model is None:
                X_feature = X # Assume X is already featurized
            else:
                X_feature = self._featurize(X)
                
            self.dime = DIME(
                explained_variance_threshold=self.k, 
                n_percentiles=10000
            ).fit(
                torch.tensor(X_feature),
                calibrate_against_trainingset=True,
            )
            self._X_train_scores = self.score_samples(X_feature, featurized=True)

            self.threshold = np.percentile( # Score below this will be outlier 
                self._X_train_scores,
                self.alpha*100.0
            )

            self.is_fitted_ = True

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
        def __init__(self, model, alpha=0.05, T=1.0):
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
            self.set_params(
                **{
                    'model': model, 
                    'alpha': alpha,
                    'T': T
                }
            )

        def get_params(self, deep=False):
            """Get parameters; for consistency with scikit-learn's estimator API."""
            if deep:
                raise NotImplementedError
            return {
                "model": self.model, # This can't really be deep copied easily
                "alpha": self.alpha,
                "T": self.T
            }
        
        def score_samples(self, X, featurized=False):
            """
            Compute the energy score for each observation.
            
            Parameters
            ----------
            X : ndarray(float) or iterator
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
                
            # If valid, deactivate activation function to obtian raw logits
            if self.model is not None:
                valid, original_activation = DeepOOD._check_end(self.model, disable=True)
            else:
                valid = True
                
            if valid:
                def negative_energy(logits):
                    return self.T * scipy.special.logsumexp(np.asarray(logits)/self.T, axis=1)
                
                try:
                    if utils.NNTools._is_data_iter(X):
                        scores = []
                        for X_batch_, _ in X:
                            scores.append(negative_energy(X_batch_ if featurized else self.model.predict(X_batch_)))
                        scores = np.concatenate(scores)
                    else:
                        scores = negative_energy(X if featurized else self.model.predict(X))
                except Exception as e:
                    if self.model is not None:
                        self.model.layers[-1].activation = original_activation # Return model to original state
                    raise Exception(e)
                else:
                    if self.model is not None:
                        self.model.layers[-1].activation = original_activation # Return model to original state
                    return scores
            else:
                raise Exception('Invalid model; cannot compute energy scores.')

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
        def __init__(self, model, alpha=0.05):
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
            self.set_params(
                **{
                    'model': model, 
                    'alpha': alpha
                }
            )

        def get_params(self, deep=False):
            """Get parameters; for consistency with scikit-learn's estimator API."""
            if deep:
                raise NotImplementedError
            return {
                "model": self.model, # This can't really be deep copied easily
                "alpha": self.alpha
            }
        
        def score_samples(self, X, featurized=False):
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
                    scores.append(softmax_confidence(X_batch_ if featurized else self.model.predict(X_batch_)))
                return np.concatenate(scores)
            else:
                return softmax_confidence(X if featurized else self.model.predict(X))
            
class OpenSetClassifier(ClassifierMixin, BaseEstimator):
    """
    Train a composite classifier with a reject option to work under open-set conditions.

    Parameters
    ----------
    clf_model : object, optional(default=None)
        Unfitted or fitted classification model. Must support `.fit()` and `.predict()` methods.

    outlier_model : object, optional(default=None)
        Unfitted outlier detection model. Must support `.fit()` and `.predict()` methods.
        This should return a value of `inlier_value` for points which are considered inliers.
        If `None` then all points will be passed to the classifier.

    clf_prefit : bool, optional(default=False)
        Whether the `clf_model` is already fit or not.  If `clf_prefit=True` the model will not be
        refit.  This is advantageous when using a model which is expensive to train, such as a
        deep neural network.

    clf_kwargs : dict, optional(default={})
        Keyword arguments to instantiate the classification model with.  If `clf_prefit=True` these
        are ignored since they are not used.

    outlier_kwargs : dict, optional(default={})
        Keyword arguments to instantiate the outlier model with.

    known_classes : array_like(int or str, ndim=1), optional(default=None)
        A list of classes which the classifier is responsible for recognizing. If `None`,
        all unique values of `y` are used; otherwise, `y` is filtered to only include these
        instances when training the classifier.

    inlier_value : scalar, optional(default=1)
        The value `outlier_model.predict()` returns for inlier class(es).  Many sklearn routines
        return +1 for inlier vs. -1 for outlier; other routines sometimes use 0 for outlier.
        As a result, we simply check for the inlier value (+1 by default) for greater flexibility.

    unknown_class : scalar(int or str), optional(default="Unknown")
        The name or index to assign to points which are considered unknown according to the
        `outlier_model`.

    score_metric : scalar(str), optional(default="TEFF")
        Default scoring metric to use. See `figures_of_merit` outputs for options.

    clf_style : scalar(str), optional(default="hard")
        Style of classification model; "hard" models assign each point to a single category, while
        "soft" models can make multiple assignments, including to an unknown category.

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
        clf_model=None,
        outlier_model=None,
        clf_prefit=False,
        clf_kwargs={},
        outlier_kwargs={},
        known_classes=None,
        inlier_value=1,
        unknown_class="Unknown",
        score_metric="TEFF",
        clf_style="hard",
        score_using="all",
    ):
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

    def set_params(self, **parameters):
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
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

    def _check_category_type(self, y):
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

    def fit(self, X, y):
        """
        Fit the composite model.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Input feature matrix.

        y : array_like(str or int, ndim=1)
            Class labels or indices.

        Returns
        -------
        self : OpenWorldClassifier
            Fitted model.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        self._check_category_type(y.ravel())
        assert self.unknown_class not in set(
            y
        ), "unknown_class value is already taken"

        if not (self.clf_style in ["soft", "hard"]):
            raise ValueError("clf_style should be either 'hard' or 'soft'.")

        # Remove any classes the classification model should not be responsible for
        # learning.
        if self.known_classes is None:
            # Consider all training examples.
            self.knowns_ = np.unique(y)
        else:
            self.knowns_ = np.unique(self.known_classes)

        # For sklearn compatibility - not used
        self.classes_ = self.knowns_.tolist() + [self.unknown_class]

        known_mask = np.array([y_ in self.knowns_ for y_ in y], dtype=bool)

        if np.sum(known_mask) == 0:
            raise Exception("There are no known classes in the training set.")

        # Check that self.score_using is valid
        if (self.score_using not in self.knowns_) and (
            self.score_using.lower() != "all"
        ):
            raise ValueError(
                "score_using should be 'all' or one of the classes trained on."
            )

        # Train outlier detection first, since this how it will work at prediction
        # time.  This needs to remember the data that the classifier will use for
        # training and flag anything different (covariate shift).  Thus, this needs
        # to train on the knowns_ only.
        try:
            self.od_ = self.outlier_model(**self.outlier_kwargs)
            self.od_.fit(X[known_mask, :])
        except:
            raise Exception(
                f"Unable to fit outlier model : {sys.exc_info()[0]}"
            )

        # Predict for all X for simplicity. The composite mask will only allow knowns
        # which are not outliers through.
        inlier_mask = self.od_.predict(X) == self.inlier_value

        composite_mask = known_mask & inlier_mask

        if np.sum(composite_mask) == 0:
            raise Exception(
                "There are no inlying known classes in the training set."
            )

        if len(np.unique(y[composite_mask])) < 2:
            raise Exception(
                "There are less than 2 distinct classes available for training."
            )

        try:
            if not clf_prefit:
                self.clf_ = self.clf_model(**self.clf_kwargs)
                self.clf_.fit(X[composite_mask, :], y[composite_mask])
        except:
            raise Exception(
                f"Unable to fit classification model : {sys.exc_info()[0]}"
            )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Make a prediction.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Input feature matrix.

        Returns
        -------
        predictions : array_like(int or str, ndim=2)
            Class, or classes, assigned to each point.  Points considered outliers are
            assigned the value `unknown_class`.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, accept_sparse=False)
        assert X.shape[1] == self.n_features_in_

        # 1. Check for outliers
        inlier_mask = self.od_.predict(X) == self.inlier_value

        # 2. Predict on points considered inliers
        if np.sum(inlier_mask) > 0:
            pred = self.clf_.predict(X[inlier_mask, :])

        predictions = [[]] * len(X)
        j = 0
        for i in range(X.shape[0]):
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

    def fit_predict(self, X, y):
        """
        Fit then predict.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Input feature matrix.

        y : array_like(str or int, ndim=1)
            Class labels or indices.

        Returns
        -------
        predictions : array_like(int or str, ndim=2)
            Class, or classes, assigned to each point.  Points considered outliers are
            assigned the value `unknown_class`.
        """
        self.fit(X, y)
        return self.predict(X)

    def score(self, X, y):
        """
        Score the prediction.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array_like(str or int, ndim=1)
            Ground truth classes - will be converted to numpy array
            automatically.

        Returns
        -------
        score : scalar(float)
            Score.
        """
        check_is_fitted(self, "is_fitted_")

        X, y = np.asarray(X), np.asarray(y)
        self._check_category_type(y.ravel())
        metrics = self.figures_of_merit(self.predict(X), y)
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

    def figures_of_merit(self, predictions, actual):
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
            np.concatenate((np.unique(actual), self.knowns_))
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
            correct_ = 0.0
            for class_ in df.index:  # All input classes
                if (
                    class_ in self.knowns_
                ):  # Things to classifier knows about (TP)
                    correct_ += df[class_][class_]
                else:
                    # Consider an assignment as "unknown" a correct assignment (TN)
                    correct_ += df[self.unknown_class][class_]
            ACC = correct_ / df.sum().sum()

            # Class-wise FoM
            # Sensitivity is "true positive" rate and is only defined for
            # trained/known classes
            CSNS = pd.Series(
                [
                    df[kk][kk] / Itot[kk] if Itot[kk] > 0 else np.nan
                    for kk in self.knowns_
                ],
                index=self.knowns_,
            )

            # Specificity is the fraction of points that are NOT a given class that
            # are correctly predicted to be something besides the class. Thus,
            # specificity can only be computed for the columns that correspond to
            # known classes since we have only trained on them. These are "true
            # negatives". This is always >= 0.
            CSPS = pd.Series(
                [
                    1.0
                    - np.sum(df[kk][df.index != kk])  # Column sum
                    / np.sum(Itot[Itot.index != kk])
                    for kk in self.knowns_
                ],
                index=self.knowns_,
            )

            # If CSNS can't be calculated, using CSPS as efficiency;
            # Oliveri & Downey introduced this "efficiency" used in [1]
            CEFF = pd.Series(
                [
                    np.sqrt(CSNS[c] * CSPS[c])
                    if not np.isnan(CSNS[c])
                    else CSPS[c]
                    for c in self.knowns_
                ],
                index=self.knowns_,
            )

            # Total FoM

            # Evaluates overall ability to recognize a class is itself.  If you
            # show the model some class it hasn't trained on, it can't be predicted
            # so no contribution to the diagonal.  We will normalize by total
            # number of points shown [1].  If some classes being tested were seen in
            # training they contribute, otherwise TSNS goes down for a class never
            # seen before.  This might seem unfair, but TSNS only makes sense if
            # (1) you are examining what you have trained on or (2) you are
            # examining extraneous objects so you don't calculate this at all.
            TSNS = np.sum([df[kk][kk] for kk in self.knowns_]) / np.sum(Itot)

            # If any untrained class is correctly predicted to be "NOT_ASSIGNED" it
            # won't contribute to df[use_classes].sum().sum().  Also, unseen
            # classes can't be assigned to so the diagonal components for those
            # entries is also 0 (df[k][k]).
            TSPS = 1.0 - (
                df[use_classes].sum().sum()
                - np.sum([df[kk][kk] for kk in use_classes])
            ) / np.sum(Itot) / (
                1.0
                if self.clf_style.lower() == "hard"
                else len(self.knowns_) - 1.0
            )
            # Soft models can assign a point to all categories which would make this
            # sum > 1, meaning TSPS < 0 would be possible.  By scaling by the total
            # number of classes, TSPS is always positive; TSPS = 0 means all points
            # assigned to all classes (trivial result) vs. TSPS = 1 means no mistakes.

            # Sometimes TEFF is reported as TSPS when TSNS cannot be evaluated (all
            # previously unseen samples).
            TEFF = np.sqrt(TSPS * TSNS)

            fom = dict(
                zip(
                    [
                        "CM",
                        "I",
                        "CSNS",
                        "CSPS",
                        "CEFF",
                        "TSNS",
                        "TSPS",
                        "TEFF",
                        "ACC",
                    ],
                    (
                        df[
                            [c for c in df.columns if c in self.knowns_]
                            + [self.unknown_class]
                        ][
                            [x in np.unique(actual) for x in df.index]
                        ],  # Re-order for easy visualization
                        Itot,
                        CSNS,
                        CSPS,
                        CEFF,
                        TSNS,
                        TSPS,
                        TEFF,
                        ACC,
                    ),
                )
            )
        else:
            # Evaluate as a OCC where the score_using class is the target class.
            alternatives = [
                class_ for class_ in df.index if class_ != self.score_using
            ]

            correct_ = df[self.score_using][self.score_using]  # (TP)
            for class_ in alternatives:  # All "negative" classes
                # Number of times an observation NOT from score_using was correctly not assigned to score_using
                # Assigning to multiple alternatives does not influence this in the spirit of OCC
                correct_ += Itot[class_] - df[self.score_using][class_]  # (TN)
            ACC = correct_ / float(Itot.sum())

            CSPS = {}
            for class_ in alternatives:
                if np.sum(Itot[class_]) > 0:
                    CSPS[class_] = 1.0 - df[class_][self.score_using] / np.sum(
                        Itot[class_]
                    )
                else:
                    CSPS[class_] = np.nan

            if np.all(actual == self.score_using):
                # Testing on nothing but the target class, can't evaluate TSPS
                TSPS = np.nan
            else:
                TSPS = 1.0 - (
                    df[self.score_using].sum()
                    - df[self.score_using][self.score_using]
                ) / (Itot.sum() - Itot[self.score_using])

            # TSNS = CSNS
            if self.score_using not in set(actual):
                # Testing on nothing but alternative classes, can't evaluate TSNS
                TSNS = np.nan
            else:
                TSNS = (
                    df[self.score_using][self.score_using]
                    / Itot[self.score_using]
                )

            if np.isnan(TSNS):
                TEFF = TSPS
            elif np.isnan(TSPS):
                TEFF = TSNS
            else:
                TEFF = np.sqrt(TSNS * TSPS)

            fom = dict(
                zip(
                    ["CM", "I", "CSPS", "TSNS", "TSPS", "TEFF", "ACC"],
                    (
                        df[
                            [c for c in df.columns if c in self.knowns_]
                            + [self.unknown_class]
                        ][
                            [x in np.unique(actual) for x in df.index]
                        ],  # Re-order for easy visualization
                        Itot,
                        CSPS,
                        TSNS,
                        TSPS,
                        TEFF,
                        ACC,
                    ),
                )
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
