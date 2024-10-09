"""
Soft independent modeling of class analogies.

author: nam
"""
import copy
import warnings
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pychemauth.preprocessing.scaling import CorrectedScaler
from pychemauth.utils import estimate_dof, _logistic_proba, _occ_metrics

from typing import Any, ClassVar, Union, Sequence
from numpy.typing import NDArray


class SIMCA_Authenticator(ClassifierMixin, BaseEstimator):
    r"""
    Train a SIMCA model for a target class.

    Parameters
    ----------
    n_components : scalar(int), optional(default=1)
        Number of components to use in the SIMCA model.

    alpha : scalar(float), optional(default=0.05)
        Significance level for SIMCA model. Only used for DD-SIMCA at the moment.

    gamma : scalar(float), optional(default=None)
        Outlier significance level for SIMCA model. Only used for DD-SIMCA at the moment. Typically, `gamma=0.01` for many applications.  If `None` then no outlier detection is performed on the training data; this also disables the `sft` option which only makes sense if you specify a `gamma` value.

    target_class : scalar(str or int), optional(default=None)
        The class used to fit the SIMCA model; the rest are used to test specificity.

    style : str, optional(default="dd-simca")
        Type of SIMCA to use ("simca" or "dd-simca")

    use : str, optional(default="rigorous")
        Which methodology to use to evaluate the model ("rigorous", "compliant", "acc") (default="rigorous"). See Ref. [1] for more details.  The "acc" option uses accuracy instead of TSNS or TEFF metrics; this enables more straightforward comparison to discriminative classifiers, but is not conventional.

    scale_x : bool, optional(default=True)
        Whether or not to scale `X` by its sample standard deviation or not. This depends on the meaning of `X` and is up to the user to determine if scaling it (by the standard deviation) makes sense. Note that `X` is always centered.

    robust : str, optional(default="semi")
        Whether or not to apply robust methods to estimate degrees of freedom. This is only used with DD-SIMCA. "full" is not implemented yet, but involves robust PCA and robust degrees of freedom estimation; "semi" (default) is described in [3] and uses classical PCA but robust DoF estimation; all other values revert to classical PCA and classical DoF estimation. If the dataset is clean (no outliers) it is best practice to use a classical method [3], however, to initially test for and potentially remove these points, a robust variant is recommended. This is why "semi: is the default value. If `sft`=True then this value is ignored and a robust method is applied to iteratively clean the dataset, while the final fitting uses the classical approach.

    sft : bool, optional(default=False)
        Whether or not to use the iterative outlier removal scheme described in Ref. [2], called "sequential focused trimming."  This is only used with DD-SIMCA. If not used (default) robust estimates of parameters may be attempted; if the iterative approach is used, these robust estimates are only computed during the outlier removal loop(s) while the final "clean" data uses classical estimates.  This option may throw away data it is originally provided for training; it keeps only "regular" samples (inliers and extremes) to train the model.  If a `gamma` value is not specified then this option is not available.

    Note
    ----
    Essentially, a SIMCA model is trained for one target class. The target is set when this class is instantiated and must be one of (but doesn't need to be the only) class found in the training set (this is checked automatically).  During testing (.score()), points are broken up by class and based on the method, may be used to see how well the target class can be distinguished from each alternative.  This allows you to pass points that belong to other classes during training - they are just ignored.  This is important for integration with other scikit-learn, etc. workflows.

    Note that when you are optimizing the model using TEFF, TSPS is computed by using the alternative classes.  In that case, the model choice is influenced by these alternatives.  This is a "compliant" approach, however, if you use TSNS instead of TEFF the model only uses information about the target class itself.  This is a "rigorous" approach which can be important to consider to avoid bias in the model.

    When TEFF is used to choose a model, this is a "compliant" approach, whereas when TSNS is used instead, this is a "rigorous" approach. In rigorous models, alpha should be fixed as other hyperparameters are adjusted to match this target; in compliant approaches this can be allowed to vary and the model with the best efficiency is selected. [1]

    References
    ----------
    [1] "Rigorous and compliant approaches to one-class classification," Rodionova, O., Oliveri, P., and Pomerantsev, A. Chem. and Intell. Lab. Sys. (2016) 89-96.

    [2] "Detection of outliers in projection-based modeling," Rodionova, O., and Pomerantsev, A., Anal. Chem. 92 (2020) 2656-2664.

    [3] "Concept and role of extreme objects in PCA/SIMCA," Pomerantsev, A. and Rodionova, O., Journal of Chemometrics 28 (2014) 429-438.
    """

    n_components: ClassVar[int]
    alpha: ClassVar[float]
    gamma: ClassVar[Union[float, None]]
    target_class: ClassVar[Union[str, int, None]]
    style: ClassVar[str]
    use: ClassVar[str]
    scale_x: ClassVar[bool]
    robust: ClassVar[str]
    sft: ClassVar[bool]

    def __init__(
        self,
        n_components: int = 1,
        alpha: float = 0.05,
        gamma: Union[float, None] = None,
        target_class: Union[str, int, None] = None,
        style: str = "dd-simca",
        use: str = "rigorous",
        scale_x: bool = True,
        robust: str = "semi",
        sft: bool = False,
    ) -> None:
        """Instantiate the classifier."""
        self.set_params(
            **{
                "n_components": n_components,
                "alpha": alpha,
                "gamma": gamma,
                "target_class": target_class,
                "style": style,
                "use": use,
                "scale_x": scale_x,
                "robust": robust,
                "sft": sft,
            }
        )

    def set_params(self, **parameters: Any) -> "SIMCA_Authenticator":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "n_components": self.n_components,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "target_class": self.target_class,
            "style": self.style,
            "use": self.use,
            "scale_x": self.scale_x,
            "robust": self.robust,
            "sft": self.sft,
        }

    def fit(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[
            Sequence[int], Sequence[str], NDArray[np.integer], NDArray[np.str_]
        ],
    ) -> "SIMCA_Authenticator":
        """
        Fit the SIMCA authenticator model.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Input feature matrix.

        y : array-like(str or int, ndim=1)
            Class labels or indices. Should include some examples of "target_class".

        Returns
        -------
        self : SIMCA_Authenticator
            Fitted model.

        Note
        ----
        Only data of the target class will be used for fitting, though more
        can be provided. This is important in pipelines, for example, when
        SMOTE is used to up-sampled minority classes; in that case, those
        must be part of the pipeline for those steps to work automatically.
        However, a user may manually provide only the data of interest.
        """
        X_, y_ = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
            copy=False,
        )
        self.n_features_in_ = X_.shape[1]
        self.classes_ = [True, False]  # For sklearn compatibility

        # Fit model to target data
        if self.style == "simca":
            self.__model_ = SIMCA_Model(
                n_components=self.n_components,
                alpha=self.alpha,
                scale_x=self.scale_x,
            )
        elif self.style == "dd-simca":
            self.__model_ = DDSIMCA_Model(
                n_components=self.n_components,
                alpha=self.alpha,
                gamma=self.gamma,
                scale_x=self.scale_x,
                robust=self.robust,
                sft=self.sft,
            )
        else:
            raise ValueError("{} is not a recognized style.".format(self.style))

        assert self.target_class in np.unique(
            y_
        ), "target_class not in training set"
        self.__model_.fit(
            X_[y_ == self.target_class], y_[y_ == self.target_class]
        )
        self.is_fitted_ = True

        return self

    def transform(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Transform into the SIMCA subspace.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Input feature matrix.

        Returns
        -------
        projected : ndarray(float, ndim=2)
            X transformed into the SIMCA subspace.

        Note
        ----
        This is necessary for scikit-learn compatibility.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        return self.__model_.transform(X)

    def fit_transform(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[
            Sequence[int], Sequence[str], NDArray[np.integer], NDArray[np.str_]
        ],
    ) -> NDArray[np.floating]:
        """
        Fit then transform.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Input feature matrix.

        y : array-like(str or int, ndim=1)
            Class labels or indices. Should include some examples of "target_class".

        Returns
        -------
        projected : ndarray(float, ndim=2)
            X transformed into the SIMCA subspace.

        Note
        ----
        This is necessary for scikit-learn compatibility.
        """
        _ = self.fit(X, y)
        return self.transform(X)

    def predict(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.bool_]:
        """
        Make a prediction.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Input feature matrix.

        Returns
        -------
        predictions : ndarray(bool, ndim=1)
            Whether or not each point is an inlier.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        return self.__model_.predict(X)

    def predict_proba(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Predict the probability that observations are inliers.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Input feature matrix.

        Returns
        -------
        probability : ndarray(float, ndim=2)
            Probability that each point is an inlier.  First column
            is NOT inlier, 1-p(x), second column is inlier probability, p(x).
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        return self.__model_.predict_proba(X)

    def decision_function(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Compute the decision function for each sample.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        decision_function : ndarray(float, ndim=1)
            Shifted, negative distance for each sample.

        Note
        ----
        Following scikit-learn's EllipticEnvelope, this returns the negative
        sqrt(distance) shifted by the cutoff distance, so f < 0 implies
        an extreme or outlier while f > 0 implies an inlier.

        References
        ----------
        See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-decision_function
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        return self.__model_.decision_function(X)

    @property
    def model(self) -> Union["SIMCA_Model", "DDSIMCA_Model"]:
        """
        Return the trained undelying SIMCA Model.

        Returns
        -------
        model : SIMCA_Model or DDSIMCA_Model
            The trained SIMCA Model being used.
        """
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.__model_)

    def score(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[
            Sequence[int], Sequence[str], NDArray[np.integer], NDArray[np.str_]
        ],
    ):
        r"""
        Score the model.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Input feature matrix.

        y : array-like(str or int, ndim=1)
            Class labels or indices.

        Returns
        -------
        score : scalar(float)
            The model's score. For rigorous models this is the negative
            squared deviation of TSNS from (1-:math:`\alpha`); for compliant
            models this is the TEFF.

        Note
        ----
        The "rigorous" approach uses only the target class to score
        the model and returns -(TSNS - (1-:math:`\alpha`))^2; the "compliant"
        approach returns TEFF. In both cases, a larger output is a
        "better" model.
        """
        check_is_fitted(self, "is_fitted_")
        if self.use.lower() == "rigorous":
            # Make sure we have the target class to test on
            assert self.target_class in set(np.unique(y))

            m = self.metrics(X, y)
            return -((m["TSNS"] - (1 - self.alpha)) ** 2)
        elif self.use.lower() == "compliant":
            # Make sure we have alternatives to test on
            a = set(np.unique(y))
            a.discard(self.target_class)
            assert len(a) > 0

            # Make sure we have the target class to test on
            assert self.target_class in set(np.unique(y))

            m = self.metrics(X, y)
            return m["TEFF"]
        elif self.use.lower() == "acc":
            # Technically we do not need the target class to be present to
            # compute accuracy.
            m = self.metrics(X, y)
            return m["ACC"]
        else:
            raise ValueError("Unrecognized setting use = " + str(self.use))

    def metrics(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[
            Sequence[int], Sequence[str], NDArray[np.integer], NDArray[np.str_]
        ],
    ) -> dict[str, float]:
        """
        Compute figures of merit for the model.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Input feature matrix.

        y : array-like(str or int, ndim=1)
            Class labels or indices.

        Returns
        -------
        metrics : dict(str, float)
            Dictionary of {"TSNS", "TSPS", "TEFF", "CSPS", "ACC"}.

        Note
        ----
        If using a set with only the target class present, then
        TSPS = np.nan and TEFF = TSNS.  If only alternatives present,
        sets TSNS = np.nan and TEFF = TSPS. Otherwise returns TEFF
        as a geometric mean of TSNS and TSPS.
        """
        check_is_fitted(self, "is_fitted_")
        X_, y_ = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
        )
        if X_.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        metrics, self.__alternatives_ = _occ_metrics(
            X=X_,
            y=y_,
            target_class=self.target_class,  # type: ignore[arg-type]
            predict_function=self.predict,
        )

        return metrics

    def _get_tags(self):
        """For compatibility with scikit-learn >=0.21."""
        return {
            "allow_nan": False,
            "array_api_support": False,
            "binary_only": False,
            "multilabel": True,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "preserves_dtype": [np.float64],  # Only for transformers
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": True,
            "requires_positive_y": False,
            "_skip_test": [
                "check_estimators_dtypes"  # Target class in training set issues
            ],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }


class SIMCA_Model(ClassifierMixin, BaseEstimator):
    """
    SIMCA model for a single class.

    Parameters
    ----------
    n_components : scalar(int)
        Number of PCA components to use to model this class.

    alpha : scalar(float), optional(default=0.05)
        Significance level.

    scale_x : bool, optional(default=True)
        Whether or not to scale `X` by its sample standard deviation or not. This depends on the meaning of `X` and is up to the user to determine if scaling it (by the standard deviation) makes sense. Note that `X` is always centered.

    In general, you need a separate SIMCA object for each class in the dataset you wish to characterize. This code is based on implementation described in [1].  An F-test is performed based on the squared orthogonal distance (OD); if it is in excess of some critical value a point is not assigned to a class, otherwise it is.  Since a different SIMCA object is trained to characterize different classes, it is possible that testing a point on a different SIMCA class will result in multiple class assignments; however, each individual SIMCA class is binary.

    Improvements have been suggested, including the use of Mahalanobis distance instead of the OD (see discussion in [3]), however we implement a more "classic" version here.

    References
    ----------
    [1] "Robust classification in high dimensions based on the SIMCA Method," Vanden Branden and Hubert, Chemometrics and Intelligent Laboratory Systems 79 (2005) 10-21.

    [2] "Pattern recognition by means of disjoint principal components models," S. Wold, Pattern Recognition 8 (1976) 127–139.

    [3] "Decision criteria for soft independent modelling of class analogy applied to near infrared data" De Maesschalk et al., Chemometrics and Intelligent Laboratory Systems 47 (1999) 65-77.
    """

    n_components: ClassVar[int]
    alpha: ClassVar[float]
    scale_x: ClassVar[bool]

    def __init__(
        self, n_components: int, alpha: float = 0.05, scale_x: bool = True
    ) -> None:
        """Instantiate the class."""
        self.set_params(
            **{"n_components": n_components, "alpha": alpha, "scale_x": scale_x}
        )

    def set_params(self, **parameters: Any) -> "SIMCA_Model":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "n_components": self.n_components,
            "alpha": self.alpha,
            "scale_x": self.scale_x,
        }

    def fit(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[
            Sequence[int],
            Sequence[str],
            NDArray[np.integer],
            NDArray[np.str_],
            None,
        ] = None,
    ) -> "SIMCA_Model":
        """
        Fit the SIMCA model.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        y : array-like(str or int, ndim=1), optional(default=None)
            Ignored. If passed, it is checked that they are all identical and this
            label is used; otherwise the name "Training Class" is assigned.

        Returns
        -------
        self : SIMCA_Model
            Fitted model.
        """
        self.__X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        if y is not None:  # Just so this passes sklearn api checks
            self.__X_, _ = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_2d=True,
                force_all_finite=True,
                y_numeric=False,
                copy=True,
            )
        self.n_features_in_ = self.__X_.shape[1]

        if y is None:
            self.__label_ = "Training Class"
        else:
            label = np.unique(y)
            if len(label) > 1:
                raise Exception("More than one class passed during training.")
            else:
                self.__label_ = str(label[0])

        max_components = np.min([self.n_features_in_, self.__X_.shape[0]]) - 1
        if self.n_components > max_components:
            warnings.warn(
                "The number of PCA components exceeds maximum allowable ({}) - changing to this value".format(
                    np.max([1, max_components])
                )
            )
            self.set_params(**{"n_components": np.max([1, max_components])})

        # 1. Standardize X
        self.__ss_ = CorrectedScaler(with_mean=True, with_std=self.scale_x)

        # 2. Perform PCA on standardized coordinates
        self.__pca_ = PCA(n_components=self.n_components, random_state=0)
        self.__pca_.fit(self.__ss_.fit_transform(self.__X_))

        # 3. Compute critical F value
        II, JJ, KK = self.__X_.shape[0], self.__X_.shape[1], self.n_components
        if II > JJ:  # See [3]
            self.__a_ = JJ
        else:
            self.__a_ = II - 1
        self.__f_crit_ = scipy.stats.f.ppf(
            1.0 - self.alpha, self.__a_ - KK, (self.__a_ - KK) * (II - KK - 1)
        )

        self.is_fitted_ = True
        return self

    def transform(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Project X into the feature subspace.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        t-scores : ndarray(float, ndim=2)
            Projection of X via PCA into a score space.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        return self.__pca_.transform(self.__ss_.transform(X))

    def fit_transform(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[
            Sequence[int],
            Sequence[str],
            NDArray[np.integer],
            NDArray[np.str_],
            None,
        ] = None,
    ) -> NDArray[np.floating]:
        """
        Fit and transform.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        y : array-like(str or int, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        t-scores : array-like(float, ndim=2)
            Projection of X via PCA into a score space.
        """
        self.fit(X, y)
        return self.transform(X)

    def distance(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Compute the F score (distance) for a given set of observations.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        F : ndarray(float, ndim=1)
            F value for each observation.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        II, _, KK = self.__X_.shape[0], self.__X_.shape[1], self.n_components

        X_pred = self.__ss_.inverse_transform(
            self.__pca_.inverse_transform(self.transform(X))
        )
        numer = np.sum((X - X_pred) ** 2, axis=1) / (self.__a_ - KK)  # See [3]

        X_pred = self.__ss_.inverse_transform(
            self.__pca_.inverse_transform(self.transform(self.__X_))
        )
        OD2 = np.sum((self.__X_ - X_pred) ** 2, axis=1)  # See [3]
        denom = np.sum(OD2) / ((self.__a_ - KK) * (II - KK - 1))

        # F-score for each distance
        F = numer / denom

        return F

    def decision_function(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Compute the decision function for each sample.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        decision_function : ndarray(float, ndim=1)
            Shifted, negative distance for each sample.

        Note
        ----
        Following scikit-learn's EllipticEnvelope, this returns the negative
        sqrt(distance) shifted by the cutoff distance, so f < 0 implies an
        extreme or outlier while f > 0 implies an inlier.

        References
        ----------
        See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-decision_function
        """
        check_is_fitted(self, "is_fitted_")
        return -np.sqrt(self.distance(X)) - (-np.sqrt(self.__f_crit_))

    def predict_proba(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Predict the probability that observations are inliers.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        probability : ndarray(float, ndim=2)
            2D array as logistic function of the decision_function().
            Probability that each point is an inlier.  First column
            is NOT inlier, 1-p(x), second column is inlier probability, p(x).

        Note
        ----
        Computes the logistic(decision_function(X, y)) as the
        transformation of the decision function.  This function is > 0
        for inliers so predict_proba(X) > 0.5 means inlier, < 0.5 means
        outlier or extreme.

        References
        ----------
        See SHAP documentation for a discussion on the utility and impact
        of "squashing functions": https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Squashing%20Effect.html#Probability-space-explaination

        See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-predict_proba
        """
        check_is_fitted(self, "is_fitted_")

        return _logistic_proba(self.decision_function(X))

    def predict(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.bool_]:
        """
        Predict whether each point is an inlier.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        inliers : ndarray(bool, ndim=1)
            Bolean array of whether a point belongs to this class.
        """
        check_is_fitted(self, "is_fitted_")
        F = self.distance(X)

        # If f < f_crit, it belongs to the class
        return F < self.__f_crit_

    def loss(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[Sequence[bool], NDArray[np.bool_]],
        eps: float = 1.0e-15,
    ) -> float:
        r"""
        Compute the negative log-loss, or logistic/cross-entropy loss.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array-like(bool, ndim=1)
            Correct labels; True for inlier, False for outlier.

        eps : scalar(float), optional(default=1.0e-15)
            Numerical addition to enable evaluation when log(p ~ 0).

        Returns
        -------
        loss : scalar(float)
            Negative, normalized log loss; :math:`\frac{1}{N} \sum_i \left( y_{in}(i) {\rm ln}(p_{in}(i)) + (1-y_{in}(i)) {\rm ln}(1-p_{in}(i)) \right)`

        References
        ----------
        See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss.
        """
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        assert np.all(
            [a in [True, False] for a in y]
        ), "y should contain only True or False labels"

        # Inlier = True (positive class), p[:,0]
        # Not inlier = False (negative class), p[:,1]
        prob = self.predict_proba(X)

        y_in = np.array([1.0 if y_ == True else 0.0 for y_ in y])
        p_in = np.clip(prob[:, 1], a_min=eps, a_max=1.0 - eps)

        # Return the negative, normalized log-loss
        return -np.sum(
            y_in * np.log(p_in) + (1.0 - y_in) * np.log(1.0 - p_in)
        ) / len(X)

    def score(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[Sequence[bool], NDArray[np.bool_]],
    ) -> float:
        """
        Return the accuracy of the model.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array-like(bool, ndim=1)
            Correct labels; True for inlier, False for outlier.

        Returns
        -------
        accuracy : scalar(float)
            Accuracy of predictions.
        """
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        return self.accuracy(X, y)

    def accuracy(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[Sequence[bool], NDArray[np.bool_]],
    ) -> float:
        """
        Get the fraction of correct predictions.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array-like(bool, ndim=1), optional(default=None)
            Correct labels; True for inlier, False for outlier.

        Returns
        -------
        accuracy : scalar(float)
            Accuracy of predictions.
        """
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        y = y.reshape(-1, 1)  # type: ignore[union-attr]
        if not isinstance(y[0][0], (np.bool_, bool)):
            raise ValueError("y must be provided as a Boolean array")
        X_pred = self.predict(X)

        return np.sum(X_pred == y.ravel()) / X_pred.shape[0]

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
            "preserves_dtype": [np.float64],  # Only for transformers
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": [
                "check_estimators_dtypes"  # More than one class in training set issues
            ],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }


class DDSIMCA_Model(ClassifierMixin, BaseEstimator):
    """
    A Data-driven SIMCA Model.

    Parameters
    ----------
    n_components : scalar(int)
        Number of PCA components to use to model this class.

    alpha : scalar(float), optional(default=0.05)
        Significance level.

    gamma : scalar(float), optional(default=None)
        Outlier significance level.  If `None` then no outlier detection is performed on the training data; this also disables the `sft` option which only makes sense if you specify a `gamma` value.

    scale_x : bool, optional(default=True)
        Whether or not to scale X by its sample standard deviation or not. This depends on the meaning of X and is up to the user to determine if scaling it (by the standard deviation) makes sense. Note that X is always centered.

    robust : str, optional(default="semi")
        Whether or not to apply robust methods to estimate degrees of freedom. "full" is not implemented yet, but involves robust PCA and robust degrees of freedom estimation; "semi" (default) is described in [2] and uses classical PCA but robust DoF estimation; all other values revert to classical PCA and classical DoF estimation. If the dataset is clean (no outliers) it is best practice to use a classical method [2], however, to initially test for and potentially remove these points, a robust variant is recommended. This is why "semi" is the default value. If `sft`=True then this value is ignored and a robust method is applied to iteratively clean the dataset, while the final fitting uses the classical approach.

    sft : bool, optional(default=False)
        Whether or not to use the iterative outlier removal scheme described in [3], called "sequential focused trimming."  If not used (default) robust estimates of parameters may be attempted; if the iterative approach is used, these robust estimates are only computed during the outlier removal loop(s) while the final "clean" data uses classical estimates.  This option may throw away data it is originally provided for training; it keeps only "regular" samples (inliers and extremes) to train the model.  If a `gamma` value is not specified then this option is not available.

    Note
    ----
    DD-SIMCA uses a combination of OD and SD, modeled by a chi-squared distribution, to determine the acceptance criteria to belong to a class. The degrees of freedom in this model are estimated from a data-driven approach. This implementation follows [1].

    As in SIMCA, this is designed to be a binary classification tool (yes/no) for a single class.  A separate object must be trained for each class you wish to model.

    References
    ----------
    [1] "Acceptance areas for multivariate classification derived by projection methods," Pomerantsev, A., Journal of Chemometrics 22 (2008) 601-609.

    [2] "Concept and role of extreme objects in PCA/SIMCA," Pomerantsev, A. and Rodionova, O., Journal of Chemometrics 28 (2014) 429-438.

    [3] "Detection of outliers in projection-based modeling," Rodionova, O., and Pomerantsev, A., Anal. Chem. 92 (2020) 2656-2664.
    """

    n_components: ClassVar[int]
    alpha: ClassVar[float]
    gamma: ClassVar[Union[float, None]]
    scale_x: ClassVar[bool]
    robust: ClassVar[str]
    sft: ClassVar[bool]

    def __init__(
        self,
        n_components: int,
        alpha: float = 0.05,
        gamma: Union[float, None] = None,
        scale_x: bool = True,
        robust: str = "semi",
        sft: bool = False,
    ) -> None:
        """Instantiate the class."""
        self.set_params(
            **{
                "n_components": n_components,
                "alpha": alpha,
                "gamma": gamma,
                "scale_x": scale_x,
                "robust": robust,
                "sft": sft,
            }
        )

    def set_params(self, **parameters: Any) -> "DDSIMCA_Model":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "n_components": self.n_components,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "scale_x": self.scale_x,
            "robust": self.robust,
            "sft": self.sft,
        }

    def fit(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[
            Sequence[int],
            Sequence[str],
            NDArray[np.integer],
            NDArray[np.str_],
            None,
        ] = None,
    ) -> "DDSIMCA_Model":
        """
        Fit the SIMCA model.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        y : array-like(str or int, ndim=1), optional(default=None)
            Ignored. If passed, it is checked that they are all identical and this
            label is used; otherwise the name "Training Class" is assigned.

        Returns
        -------
        self : DDSIMCA_Model
            Fitted model.
        """
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if y is not None:  # Just so this passes sklearn api checks
            X, _ = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_2d=True,
                force_all_finite=True,
                y_numeric=False,
            )

        if y is None:
            self.__label_ = "Training Class"
        else:
            label = np.unique(y)
            if len(label) > 1:
                raise Exception("More than one class passed during training.")
            else:
                self.__label_ = str(label[0])

        def train(X, robust):
            """
            Train the model.

            Parameters
            ----------
            X : ndarray(float, ndim=2)
                Data to train on.
            robust : str
                "full" = robust PCA + robust parameter estimation in [2] (not yet implemented);
                "semi" = classical PCA + robust parameter estimation in [2] ("RDD-SIMCA");
                otherwise = classical PCA + classical parameter estimation in [2] ("CDD-SIMCA");
            """
            self.__X_ = np.array(
                X
            ).copy()  # This is needed so self._h_q() works correctly
            self.n_features_in_ = self.__X_.shape[1]

            max_components = (
                np.min([self.n_features_in_, self.__X_.shape[0]]) - 1
            )
            if self.n_components > max_components:
                warnings.warn(
                    "The number of PCA components exceeds maximum allowable ({}) - changing to this value".format(
                        np.max([1, max_components])
                    )
                )
                self.set_params(**{"n_components": np.max([1, max_components])})

            if robust == "full":
                raise NotImplementedError
            else:
                # 1. Standardize X
                self.__ss_ = CorrectedScaler(
                    with_mean=True, with_std=self.scale_x
                )
                self.__ss_.fit(self.__X_)

                # 2. Perform PCA on standardized coordinates
                self.__pca_ = PCA(
                    n_components=self.n_components, random_state=0
                )
                self.__pca_.fit(self.__ss_.transform(self.__X_))

            self.is_fitted_ = True

            # 3. Compute critical distances
            h_vals, q_vals = self._h_q(self.__X_)

            # As in the conclusions of [1], Nh ~ n_components is expected so good initial guess
            self.__Nh_, self.__h0_ = estimate_dof(
                h_vals,
                robust=(
                    True if (robust == "semi" or robust == "full") else False
                ),
                initial_guess=self.n_components,
            )

            # As in the conclusions of [1], Nq ~ rank(X)-n_components is expected;
            # assuming near full rank then this is min(I,J)-n_components
            # (n_components<=J)
            self.__Nq_, self.__q0_ = estimate_dof(
                q_vals,
                robust=(
                    True if (robust == "semi" or robust == "full") else False
                ),
                initial_guess=np.min([len(q_vals), self.n_features_in_])
                - self.n_components,
            )

            # Eq. 19 in [2]
            self.__c_crit_ = scipy.stats.chi2.ppf(
                1.0 - self.alpha, self.__Nh_ + self.__Nq_
            )

            # Eq. 20 in [2]
            if self.gamma is None:
                self.__c_out_ = self.__c_crit_
            else:
                self.__c_out_ = scipy.stats.chi2.ppf(
                    (1.0 - self.gamma) ** (1.0 / self.__X_.shape[0]),
                    self.__Nh_ + self.__Nq_,
                )

        # This is based on [3]
        if not self.sft:
            train(X, robust=self.robust)
            self.__sft_history_ = {}
        else:
            # SFT only really makes sense if you have an outlier significance level
            if self.gamma is None:
                raise Exception("Specify a gamma value to use sft.")

            X_tmp = np.array(X).copy()
            total_data_points = X_tmp.shape[0]
            X_out = np.empty((0, X_tmp.shape[1]), dtype=type(X_tmp))
            outer_iters = 0
            max_outer = 100
            max_inner = 100
            sft_tracker = {}
            while True:  # Outer loop
                if outer_iters >= max_outer:
                    raise Exception(
                        "Unable to iteratively clean data; exceeded maximum allowable outer loops (to eliminate swamping)."
                    )
                train(X_tmp, robust="semi")
                _, outliers = self.check_outliers(X_tmp)
                X_delete_ = X_tmp[outliers, :]
                inner_iters = 0
                while np.sum(outliers) > 0:
                    if inner_iters >= max_inner:
                        raise Exception(
                            "Unable to iteratively clean data; exceeded maximum allowable inner loops (to eliminate masking)."
                        )
                    X_tmp = X_tmp[~outliers, :]
                    if len(X_tmp) == 0:
                        raise Exception(
                            "Unable to iteratively clean data; all observations are considered outliers."
                        )
                    train(X_tmp, robust="semi")
                    _, outliers = self.check_outliers(X_tmp)
                    X_delete_ = np.vstack((X_delete_, X_tmp[outliers, :]))
                    inner_iters += 1
                X_out = np.vstack((X_out, X_delete_))
                assert (
                    X_tmp.shape[0] + X_out.shape[0] == total_data_points
                )  # Sanity check

                # All inside X_tmp are inliers or extremes (regular objects) now.
                # Check that all outliers are predicted to be outliers in the latest version trained
                # on only inlier and extremes.
                outer_iters += 1
                sft_tracker[outer_iters] = {
                    "initially removed": X_delete_,
                    "returned": None,
                }
                if len(X_out) > 0:
                    _, outliers = self.check_outliers(X_out)
                    X_return = X_out[~outliers, :]
                    X_out = X_out[outliers, :]
                    if len(X_return) == 0:
                        break
                    else:
                        sft_tracker[outer_iters]["returned"] = X_return
                        X_tmp = np.vstack((X_tmp, X_return))
                else:
                    break

            # Outliers have been iteratively found, and X_tmp is a consistent set of data to use
            # which is considered "clean" so should try to use classical estimates of the parameters.
            # train() assigns X_tmp to self.__X_ also. See [3].
            assert (
                X_out.shape[0] + self.__X_.shape[0] == total_data_points
            )  # Sanity check
            train(X_tmp, robust=False)
            self.__sft_history_ = {
                "outer_loops": outer_iters,
                "removed": {"X": X_out},
                "iterations": sft_tracker,
            }

        return self

    @property
    def sft_history(self) -> dict[str, Any]:
        """Return the sequential focused trimming history."""
        return copy.deepcopy(self.__sft_history_)

    def transform(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Project X into the feature subspace.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted
            to numpy array automatically.

        Returns
        -------
        t-scores : ndarray(float, ndim=2)
            Projection of X via PCA into a score space.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        return self.__pca_.transform(self.__ss_.transform(X))

    def fit_transform(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[
            Sequence[int],
            Sequence[str],
            NDArray[np.integer],
            NDArray[np.str_],
            None,
        ] = None,
    ) -> NDArray[np.floating]:
        """
        Fit and transform.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted
            to numpy array automatically.

        y : array-like(str or int, ndim=1), optional(default=None)
            Ignored. If passed, it is checked that they are all identical and this
            label is used; otherwise the name "Training Class" is assigned.

        Returns
        -------
        t-scores : ndarray(float, ndim=2)
            Projection of X via PCA into a score space.
        """
        self.fit(X, y)
        return self.transform(X)

    def _h_q(
        self, X_raw: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute the h (SD) and q (OD) distances."""
        check_is_fitted(self, "is_fitted_")
        X_raw = check_array(
            X_raw,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X_raw.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        X_raw_std = self.__ss_.transform(X_raw)
        T = self.__pca_.transform(X_raw_std)
        X_pred = self.__pca_.inverse_transform(T)

        # OD - this is to produce identical results to mdatools 0.14.1
        q_vals = np.sum((X_raw_std - X_pred) ** 2, axis=1)

        # SD
        h_vals = np.sum(T**2 / self.__pca_.explained_variance_, axis=1)

        return h_vals, q_vals

    def distance(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Compute how far away points are from this class.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted
            to a numpy array automatically.

        Returns
        -------
        sq_distance : ndarray(float, ndim=1)
            Squared distance to class.

        Note
        ----
        This is computed as a sum of the OD and OD to be used with acceptance
        rule II from [1].  This is really a "squared" distance.

        """
        check_is_fitted(self, "is_fitted_")
        h, q = self._h_q(X)

        return self.__Nh_ * h / self.__h0_ + self.__Nq_ * q / self.__q0_

    def decision_function(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Compute the decision function for each sample.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        decision_function : ndarray(float, ndim=1)
            Shifted, negative distance for each sample.

        Note
        ----
        Following scikit-learn's EllipticEnvelope, this returns the negative
        sqrt(distance) shifted by the cutoff distance, so f < 0 implies an
        extreme or outlier while f > 0 implies an inlier.

        References
        ----------
        See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-decision_function
        """
        check_is_fitted(self, "is_fitted_")
        return -np.sqrt(self.distance(X)) - (-np.sqrt(self.__c_crit_))

    def predict_proba(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Predict the probability that observations are inliers.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        probability : ndarray(float, ndim=2)
            2D array as logistic function of the decision_function().
            First column is NOT inlier, 1-p(x), second column is inlier
            probability, p(x).

        Note
        ----
        Computes the logistic(decision_function(X, y)) as the
        transformation of the decision function.  This function is > 0
        for inliers so predict_proba(X) > 0.5 means inlier, < 0.5 means
        outlier or extreme.

        References
        ----------
        See SHAP documentation for a discussion on the utility and impact
        of "squashing functions": https://shap.readthedocs.io/en/latest/\
        example_notebooks/tabular_examples/model_agnostic/Squashing%20Effect.html\
        #Probability-space-explaination

        See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-predict_proba
        """
        check_is_fitted(self, "is_fitted_")

        return _logistic_proba(self.decision_function(X))

    def predict(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.bool_]:
        """
        Predict whether each point is an inlier.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        inliers : ndarray(bool, ndim=1)
            Bolean array of whether a point belongs to this class.
        """
        check_is_fitted(self, "is_fitted_")

        # If c < c_crit, it belongs to the class
        return self.distance(X) < self.__c_crit_

    def loss(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[Sequence[bool], NDArray[np.bool_]],
        eps: float = 1.0e-15,
    ) -> float:
        r"""
        Compute the negative log-loss, or logistic/cross-entropy loss.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array-like(bool, ndim=1)
            Correct labels; True for inlier, False for outlier.

        eps : scalar(float), optional(default=1.0e-15)
            Numerical addition to enable evaluation when log(p ~ 0).

        Returns
        -------
        loss : scalar(float)
            Negative, normalized log loss; :math:`\frac{1}{N} \sum_i \left( y_{in}(i) {\rm ln}(p_{in}(i)) + (1-y_{in}(i)) {\rm ln}(1-p_{in}(i)) \right)`

        References
        ----------
        See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss.
        """
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        assert np.all(
            [a in [True, False] for a in y]
        ), "y should contain only True or False labels"

        # Inlier = True (positive class), p[:,0]
        # Not inlier = False (negative class), p[:,1]
        prob = self.predict_proba(X)

        y_in = np.array([1.0 if y_ == True else 0.0 for y_ in y])
        p_in = np.clip(prob[:, 1], a_min=eps, a_max=1.0 - eps)

        # Return the negative, normalized log-loss
        return -np.sum(
            y_in * np.log(p_in) + (1.0 - y_in) * np.log(1.0 - p_in)
        ) / len(X)

    def score(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[Sequence[bool], NDArray[np.bool_]],
    ) -> float:
        """
        Return the accuracy of the model.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array-like(bool, ndim=1), optional(default=None)
            Correct labels; True for inlier, False for outlier.

        Returns
        -------
        accuracy : scalar(float)
            Accuracy of predictions.
        """
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        return self.accuracy(X, y)

    def accuracy(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[Sequence[bool], NDArray[np.bool_]],
    ) -> float:
        """
        Get the fraction of correct predictions.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array-like(bool, ndim=1), optional(default=None)
            Correct labels; True for inlier, False for outlier.

        Returns
        -------
        accuracy : scalar(float)
            Accuracy of predictions.
        """
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        y = y.reshape(-1, 1)  # type: ignore[union-attr]
        if not isinstance(y[0][0], (bool, np.bool_)):
            raise ValueError("y must be provided as a Boolean array")
        X_pred = self.predict(X)

        return np.sum(X_pred == y.ravel()) / X_pred.shape[0]

    def check_outliers(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """
        Check if extremes and outliers exist in the data.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        extremes : ndarray(bool, ndim=1)
            Boolean mask of X if each point falls between acceptance threshold
            (belongs to class) and the outlier threshold (extreme).

        outliers : ndarray(bool, ndim=1)
            Boolean mask of X if each point falls beyond the outlier (outlier)
            threshold.
        """
        check_is_fitted(self, "is_fitted_")

        if self.__c_out_ < self.__c_crit_:
            warnings.warn(
                "Outlier threshold < inlier threshold; increase alpha value or decrease gamma value."
            )

        dX_ = self.distance(X)
        extremes = (self.__c_crit_ <= dX_) & (dX_ < self.__c_out_)
        outliers = dX_ >= self.__c_out_

        return extremes, outliers

    def extremes_plot(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        upper_frac: float = 0.25,
        ax: Union[matplotlib.pyplot.Axes, None] = None,
    ) -> matplotlib.pyplot.Axes:
        r"""
        Plot an "extremes plot" [2] to evaluate the quality of the model.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Data to evaluate the number of outliers + extremes in.

        upper_frac : scalar(float), optional(default=0.25)
            Count the number of extremes and outliers for :math:`\alpha` values corresponding
            to :math:`n_{\rm exp}` = [1, X.shape[0]*upper_frac], where :math:`\alpha = n_{\rm exp} / N_{\rm tot}`.

        ax : matplotlib.pyplot.Axes, optional(default=None)
            Axes to plot on.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Axes results are plotted.

        Note
        ----
        This modifies the alpha value (type I error rate), keeping all other parameters
        fixed, and computes the number of expected extremes (n_exp) vs. the number
        observed (n_obs).  Theoretically, :math:`n_{\rm exp} = `alpha*N_{\rm tot}`.

        The 95% tolerance limit is given in black.  Points which fall outside these
        bounds are highlighted.

        Warning
        -------
        Both extreme points and outliers are considered "extremes" here.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        N_tot = X.shape[0]  # type: ignore[union-attr]
        n_values = np.arange(1, int(upper_frac * N_tot) + 1)
        alpha_values = n_values / N_tot

        n_observed_ = []
        for a in alpha_values:
            params = self.get_params()
            params["alpha"] = a
            model_ = DDSIMCA_Model(**params)
            model_.fit(X)
            extremes, outliers = model_.check_outliers(X)
            n_observed_.append(np.sum(extremes) + np.sum(outliers))
        n_observed = np.array(n_observed_)

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        n_upper = n_values + 2.0 * np.sqrt(n_values * (1.0 - n_values / N_tot))
        n_lower = n_values - 2.0 * np.sqrt(n_values * (1.0 - n_values / N_tot))
        ax.plot(n_values, n_upper, "-", color="k", alpha=0.5)
        ax.plot(n_values, n_values, "-", color="k", alpha=0.5)
        ax.plot(n_values, n_lower, "-", color="k", alpha=0.5)
        ax.fill_between(
            n_values, y1=n_upper, y2=n_lower, color="gray", alpha=0.25
        )

        mask = (n_lower <= n_observed) & (n_observed <= n_upper)
        ax.plot(n_values[mask], n_observed[mask], "o", color="green")
        ax.plot(n_values[~mask], n_observed[~mask], "o", color="red")

        ax.set_xlabel("Expected")
        ax.set_ylabel("Observed")

        return ax

    def plot_loadings(
        self,
        feature_names: Union[Sequence[str], NDArray[np.str_], None] = None,
        ax: Union[matplotlib.pyplot.Axes, None] = None,
        fontsize: Union[int, None] = None,
        feature_offset: tuple[int, int] = (0, 0),
    ) -> matplotlib.pyplot.Axes:
        """
        Make a 2D loadings plot.

        Parameters
        ----------
        feature_names : array-like(str, ndim=1), optional(default=None)
            List of names of each columns in X. Otherwise displays indices.

        ax : matplotlib.pyplot.Axes, optional(default=None)
            Axes to plot on.

        fontsize : scalar(int), optional(default=None)
            Fontsize to use for features.

        feature_offset : tuple(float, float), optional(default=(0,0))
            Label offset for each feature.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Axes results are plotted on.

        Note
        ----
        This uses the top 2 eigenvectors regardless of the model dimensionality. If it
        is less than 2 a ValueError is returned.
        """
        check_is_fitted(self, "is_fitted_")
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        if self.n_components < 2:
            raise ValueError(
                "Cannot visualize when using less than 2 components."
            )

        if feature_names is None:
            feature_names = [str(i) for i in range(self.n_features_in_)]

        if len(feature_names) != self.n_features_in_:
            raise ValueError("Must provide a name for each column.")

        # Some communities multiply by the sqrt(eigenvalues) - for consistency with mdatools 0.14.1 we do not.
        a = (
            self.__pca_.components_.T
        )  # * np.sqrt(self.__pca_.explained_variance_)
        ax.plot(a[:, 0], a[:, 1], "o")
        ax.axvline(0, ls="--", color="k")
        ax.axhline(0, ls="--", color="k")
        for i, label in zip(range(len(a)), feature_names):
            ax.text(
                a[i, 0] + feature_offset[0],
                a[i, 1] + feature_offset[1],
                label,
                fontsize=fontsize,
            )
        ax.set_xlabel(
            "PC 1 ({}%)".format(
                "%.4f" % (self.__pca_.explained_variance_ratio_[0] * 100.0)
            )
        )
        ax.set_ylabel(
            "PC 2 ({}%)".format(
                "%.4f" % (self.__pca_.explained_variance_ratio_[1] * 100.0)
            )
        )

        return ax

    def visualize(
        self,
        X: Union[Sequence[bool], NDArray[np.bool_]],
        y: Union[
            Sequence[int], Sequence[str], NDArray[np.integer], NDArray[np.str_]
        ],
        ax: Union[matplotlib.pyplot.Axes, None] = None,
        log: bool = True,
        outlier_curve: bool = True,
    ) -> matplotlib.pyplot.Axes:
        r"""
        Plot the :math:`\Chi^{2}` acceptance area with observations on distance plot.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array-like(str or int, ndim=1)
            Labels for observations in X.

        ax : matplotlib.pyplot.Axes, optional(default=None)
            Axis object to plot on.

        log : bool, optional(default=True)
            Whether or not to transform the axes using a natural logarithm.

        outlier_curve : bool, optional(default=True)
            Whether or not to display the outlier threshold curve and characterize
            any points as "extreme" if you specified a `gamma` value.  If False,
            then all points will be labeled as inliers vs. outliers only, based
            on your alpha value.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Axes results are plotted on.
        """
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        h_lim = np.linspace(0, self.__c_crit_ * self.__h0_ / self.__Nh_, 1000)
        h_lim_out = np.linspace(
            0, self.__c_out_ * self.__h0_ / self.__Nh_, 1000
        )
        q_lim = (
            (self.__c_crit_ - self.__Nh_ / self.__h0_ * h_lim)
            * self.__q0_
            / self.__Nq_
        )
        q_lim_out = (
            (self.__c_out_ - self.__Nh_ / self.__h0_ * h_lim_out)
            * self.__q0_
            / self.__Nq_
        )

        if ax is None:
            fig = plt.figure()
            axis = fig.gca()
        else:
            axis = ax

        axis.plot(
            np.log(1.0 + h_lim / self.__h0_) if log else h_lim / self.__h0_,
            np.log(1.0 + q_lim / self.__q0_) if log else q_lim / self.__q0_,
            "g-",
        )
        if outlier_curve and self.gamma is not None:
            axis.plot(
                np.log(1.0 + h_lim_out / self.__h0_)
                if log
                else h_lim_out / self.__h0_,
                np.log(1.0 + q_lim_out / self.__q0_)
                if log
                else q_lim_out / self.__q0_,
                "r-",
            )
        xlim, ylim = (
            1.1
            * np.max(
                np.log(1.0 + h_lim_out / self.__h0_)
                if log
                else h_lim_out / self.__h0_
            ),
            1.1
            * np.max(
                np.log(1.0 + q_lim_out / self.__q0_)
                if log
                else q_lim_out / self.__q0_
            ),
        )
        markers = [
            "o",
            "v",
            "s",
            "*",
            "+",
            "x",
            "^",
            "<",
            ">",
            "1",
            "2",
            "3",
            "4",
        ]
        for i, class_ in enumerate(sorted(np.unique(y))):
            h_, q_ = self._h_q(X[y == class_])
            in_mask = self.predict(X[y == class_])
            ext_mask, out_mask = self.check_outliers(X[y == class_])

            if outlier_curve and self.gamma is not None:
                categories_ = [
                    (
                        "g",
                        in_mask,
                        "{} = Inlier ({})".format(class_, np.sum(in_mask)),
                    ),
                    (
                        "orange",
                        ext_mask,
                        "{} = Extreme ({})".format(class_, np.sum(ext_mask)),
                    ),
                    (
                        "r",
                        out_mask,
                        "{} = Outlier ({})".format(class_, np.sum(out_mask)),
                    ),
                ]
            else:
                categories_ = [
                    (
                        "g",
                        in_mask,
                        "{} = Inlier ({})".format(class_, np.sum(in_mask)),
                    ),
                    (
                        "r",
                        (ext_mask | out_mask),
                        "{} = Outlier ({})".format(
                            class_, np.sum(out_mask) + np.sum(ext_mask)
                        ),
                    ),
                ]

            for c, mask, label in categories_:
                if np.sum(mask) > 0:
                    axis.plot(
                        np.log(1.0 + h_[mask] / self.__h0_)
                        if log
                        else h_[mask] / self.__h0_,
                        np.log(1.0 + q_[mask] / self.__q0_)
                        if log
                        else q_[mask] / self.__q0_,
                        label=label,
                        marker=markers[i % len(markers)],
                        lw=0,
                        color=c,
                        alpha=0.35,
                    )
            xlim = np.max(
                [
                    xlim,
                    1.1
                    * np.max(
                        np.log(1.0 + h_ / self.__h0_)
                        if log
                        else h_ / self.__h0_
                    ),
                ]
            )
            ylim = np.max(
                [
                    ylim,
                    1.1
                    * np.max(
                        np.log(1.0 + q_ / self.__q0_)
                        if log
                        else q_ / self.__q0_
                    ),
                ]
            )
        axis.legend(bbox_to_anchor=(1, 1))
        axis.set_xlim(0, xlim)
        axis.set_ylim(0, ylim)
        axis.set_xlabel(r"${\rm ln(1 + h/h_0)}$" if log else r"${\rm h/h_0}$")
        axis.set_ylabel(r"${\rm ln(1 + q/q_0)}$" if log else r"${\rm q/q_0}$")

        return axis

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
            "preserves_dtype": [np.float64],  # Only for transformers
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": [
                "check_estimators_dtypes"  # More than one class in training set issues
            ],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
