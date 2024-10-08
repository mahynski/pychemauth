"""
Scale data.

author: nam
"""
import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from typing import Any, Union, Sequence, ClassVar
from numpy.typing import NDArray


class RobustScaler(TransformerMixin, BaseEstimator):
    """
    Perform "robust" autoscaling on the data.

    Parameters
    ----------
    with_median : bool, optional(default=True)
        Center the data using the median.

    with_iqr : bool, optional(default=True)
        Scale the data using the interquantile range (IQR).

    pareto : bool, optional(default=False)
        Scale by the square root of the IQR instead.

    rng : tuple(float, float), optional(default=(25.0, 75.0))
        Quantiles to use; default is (25.0, 75.0) corresponding the interquartile range.

    Note
    ----
    Akin to scikit-learn's RobustScaler: this centers the data using the median instead of the mean, and scales by the interquantile range (IQR) instead of the standard deviation. The quantile can also be changed.

    A "pareto" setting is also available which will use the square root of the IQR instead.
    """

    with_median: ClassVar[bool]
    with_iqr: ClassVar[bool]
    pareto: ClassVar[bool]
    rng: ClassVar[tuple[float, float]]

    def __init__(
        self,
        with_median: bool = True,
        with_iqr: bool = True,
        pareto: bool = False,
        rng: tuple[float, float] = (25.0, 75.0),
    ) -> None:
        """Instantiate the class."""
        self.set_params(
            **{
                "with_median": with_median,
                "with_iqr": with_iqr,
                "pareto": pareto,
                "rng": rng,
            }
        )

    def set_params(self, **parameters: Any) -> "RobustScaler":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "with_median": self.with_median,
            "with_iqr": self.with_iqr,
            "pareto": self.pareto,
            "rng": self.rng,
        }

    def fit(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]], y=None
    ) -> "RobustScaler":
        """
        Fit the scaler using some training data.

        Parameter
        ---------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        self : RobustScaler
            Fitted model.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
        )
        if y is not None:  # Just so this passes sklearn api checks
            X_, y_ = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_2d=True,
                force_all_finite=True,
                y_numeric=True,
            )

        self.n_features_in_ = X_.shape[1]
        self.__median_ = np.median(X_, axis=0)
        self.__iqr_ = scipy.stats.iqr(X_, rng=self.rng, axis=0)
        self.is_fitted_ = True

        return self

    def transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """
        Transform (center and possibly scale) the data after fitting.

        Parameter
        ---------
        X : array_like(float, ndim=2)
            Unscaled feature matrix.

        Returns
        -------
        X_scaled : ndarray(float, ndim=2)
            Scaled feature matrix.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        check_is_fitted(self, "is_fitted_")
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        if self.with_median:
            X_ -= self.__median_
        if self.with_iqr:
            X_ /= np.sqrt(self.__iqr_) if self.pareto else self.__iqr_

        return X_

    def inverse_transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """
        Invert the transformation.

        Parameter
        ---------
        X : array_like(float, ndim=2)
            Scaled feature matrix.

        Returns
        -------
        X_unscaled : ndarray(float, ndim=2)
            Unscaled feature matrix.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        check_is_fitted(self, "is_fitted_")
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        if self.with_iqr:
            X_ *= np.sqrt(self.__iqr_) if self.pareto else self.__iqr_
        if self.with_median:
            X_ += self.__median_

        return X_

    def fit_transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]], y=None
    ) -> NDArray[np.floating]:
        """
        Fit and then transform some data.

        Parameter
        ---------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        X_scaled : ndarray(float, ndim=2)
            Scaled feature matrix.
        """
        self.fit(X)

        return self.transform(X)

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
            "_skip_test": [],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }


class CorrectedScaler(TransformerMixin, BaseEstimator):
    """
    Perform variations of autoscaling on the data.

    Parameters
    ----------
    with_mean : bool, optional(default=True)
        Center the data using the mean.

    with_std : bool, optional(default=True)
        Scale the data using the (corrected) sample standard deviation
        which uses N-1 degrees of freedom instead of N.

    pareto : bool, optional(default=False)
        Scale by the square root of the standard deviation instead.

    biased : bool, optional(default=False)
        If set to True, computes the uncorrected standard deviation with N
        degrees of freedom (like StandardScaler). This can be used in
        conjunction with `pareto` as well.

    Note
    ----
    This is a "StandardScaler" which by default reduces the degrees of freedom by one in the standard deviation, unlike scikit-learn's default StandardScaler.

    A "pareto" setting is also available which will use the square root of this corrected standard deviation instead.
    """

    with_mean: ClassVar[bool]
    with_std: ClassVar[bool]
    pareto: ClassVar[bool]
    biased: ClassVar[bool]

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        pareto: bool = False,
        biased: bool = False,
    ) -> None:
        """Instantiate the class."""
        self.set_params(
            **{
                "with_mean": with_mean,
                "with_std": with_std,
                "pareto": pareto,
                "biased": biased,
            }
        )

    def set_params(self, **parameters: Any) -> "CorrectedScaler":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "with_mean": self.with_mean,
            "with_std": self.with_std,
            "pareto": self.pareto,
            "biased": self.biased,
        }

    def fit(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]], y=None
    ) -> "CorrectedScaler":
        """
        Fit the scaler using some training data.

        Parameter
        ---------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        self : CorrectedScaler
            Fitted model.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
        )
        if y is not None:  # Just so this passes sklearn api checks
            X_, y_ = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_2d=True,
                force_all_finite=True,
                y_numeric=True,
            )

        self.n_features_in_ = X_.shape[1]
        self.__mean_ = np.mean(X_, axis=0, dtype=np.float64)
        self.__std_ = np.std(
            X_, axis=0, ddof=(0 if self.biased else 1), dtype=np.float64
        )
        self.is_fitted_ = True

        return self

    def transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """
        Transform (center and possibly scale) the data after fitting.

        Parameter
        ---------
        X : array_like(float, ndim=2)
            Unscaled feature matrix.

        Returns
        -------
        X_scaled : ndarray(float, ndim=2)
            Scaled feature matrix.
        """
        check_is_fitted(self, "is_fitted_")
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        if self.with_mean:
            X_ -= self.__mean_
        if self.with_std:
            tol = 1.0e-18
            if np.any(self.__std_ < tol):
                raise Exception(
                    "Cannot standardize. The X matrix has std ~ 0 for columns : {}".format(
                        np.where(self.__std_ < tol)
                    )
                )
            X_ /= np.sqrt(self.__std_) if self.pareto else self.__std_

        return X_

    def inverse_transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """
        Invert the transformation.

        Parameter
        ---------
        X : array_like(float, ndim=2)
            Scaled feature matrix.

        Returns
        -------
        X_unscaled : ndarray(float, ndim=2)
            Unscaled feature matrix.
        """
        check_is_fitted(self, "is_fitted_")
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        if self.with_std:
            X_ *= np.sqrt(self.__std_) if self.pareto else self.__std_
        if self.with_mean:
            X_ += self.__mean_

        return X_

    def fit_transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]], y=None
    ) -> NDArray[np.floating]:
        """
        Fit and then transform some data.

        Parameter
        ---------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        X_scaled : ndarray(float, ndim=2)
            Scaled feature matrix.
        """
        self.fit(X)

        return self.transform(X)

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
            "_skip_test": [],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
