"""
Scale data.

@author: nam
"""
import numpy as np
import scipy
from sklearn.utils.validation import check_array, check_is_fitted


class RobustScaler:
    """
    Perform "robust" autoscaling on the data.

    Akin to scikit-learn's RobustScaler: this centers the data using the median
    instead of the mean, and scales by the interquantile range (IQR)
    instead of the standard deviation. The quantile can also be changed.

    A "pareto" setting is also available which will use the square root of
    the IQR instead.
    """

    def __init__(
        self, with_median=True, with_iqr=True, pareto=False, rng=(25.0, 75.0)
    ):
        """
        Instantiate the class.

        Parameters
        ----------
        with_median : bool
            Center the data using the median.
        with_iqr : bool
            Scale the data using the interquantile range (IQR).
        pareto : bool
            Scale by the square root of the IQR instead.
        rng : tuple(float, float)
            Quantiles to use; default is (25, 75) corresponding the interquartile range.
        """
        self.set_params(
            **{
                "with_median": with_median,
                "with_iqr": with_iqr,
                "pareto": pareto,
                "rng": rng,
            }
        )
        self.is_fitted_ = False

    def set_params(self, **parameters):
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "with_median": self.with_median,
            "with_iqr": self.with_iqr,
            "pareto": self.pareto,
            "rng": self.rng,
        }

    def fit(self, X, y=None):
        """Fit the scaler using some training data."""
        X = check_array(X, accept_sparse=False)

        self.n_features_in_ = X.shape[1]
        self.__median_ = np.median(X, axis=0)
        self.__iqr_ = scipy.stats.iqr(X, rng=self.rng, axis=0)
        self.is_fitted_ = True

        return self

    def transform(self, X):
        """Transform (center and possibly scale) the data after fitting."""
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, "is_fitted_")
        assert X.shape[1] == self.n_features_in_

        result = np.array(X, dtype=float)
        if self.with_median:
            result -= self.__median_
        if self.with_iqr:
            result /= np.sqrt(self.__iqr_) if self.pareto else self.__iqr_

        return result

    def inverse_transform(self, X):
        """Invert the transformation."""
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, "is_fitted_")
        assert X.shape[1] == self.n_features_in_

        result = np.array(X, dtype=float)
        if self.with_iqr:
            result *= np.sqrt(self.__iqr_) if self.pareto else self.__iqr_
        if self.with_median:
            result += self.__median_

        return result

    def fit_transform(self, X, y=None):
        """Fit and then transform some data."""
        self.fit(X)

        return self.transform(X)


class CorrectedScaler:
    """
    Perform variations of autoscaling on the data.

    This is a "StandardScaler" which by default reduces the degrees of freedom
    by one in the standard deviation, unlike scikit-learn's default StandardScaler.

    A "pareto" setting is also available which will use the square root of
    this corrected standard deviation instead.
    """

    def __init__(
        self, with_mean=True, with_std=True, pareto=False, biased=False
    ):
        """
        Instantiate the class.

        Parameters
        ----------
        with_mean : bool
            Center the data using the mean.
        with_std : bool
            Scale the data using the (corrected) sample standard deviation
            which uses N-1 degrees of freedom instead of N.
        pareto : bool
            Scale by the square root of the standard deviation instead.
        biased : bool
            If set to True, computes the uncorrected standard deviation with N
            degrees of freedom (like StandardScaler). This can be used in
            conjunction with pareto as well.
        """
        self.set_params(
            **{
                "with_mean": with_mean,
                "with_std": with_std,
                "pareto": pareto,
                "biased": biased,
            }
        )
        self.is_fitted_ = False

    def set_params(self, **parameters):
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "with_mean": self.with_mean,
            "with_std": self.with_std,
            "pareto": self.pareto,
            "biased": self.biased,
        }

    def fit(self, X, y=None):
        """Fit the scaler using some training data."""
        X = check_array(X, accept_sparse=False)

        self.n_features_in_ = X.shape[1]
        self.__mean_ = np.mean(X, axis=0)
        self.__std_ = np.std(X, axis=0, ddof=(0 if self.biased else 1))
        self.is_fitted_ = True

        return self

    def transform(self, X):
        """Transform (center and possibly scale) the data after fitting."""
        X = check_array(X, accept_sparse=False)
        assert(X.shape[1] == self.n_features_in_)
        check_is_fitted(self, "is_fitted_")
        result = np.array(X, dtype=np.float64)

        if self.with_mean:
            result -= self.__mean_
        if self.with_std:
            tol = 1.0e-18
            if np.any(self.__std_ < tol):
                raise Exception('Cannot standardize. The X matrix has std ~ 0 for columns : {}'.format(np.where(self.__std_ < tol)))
            result /= np.sqrt(self.__std_) if self.pareto else self.__std_

        return result

    def inverse_transform(self, X):
        """Invert the transformation."""
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, "is_fitted_")
        assert X.shape[1] == self.n_features_in_

        result = np.array(X, dtype=float)
        if self.with_std:
            result *= np.sqrt(self.__std_) if self.pareto else self.__std_
        if self.with_mean:
            result += self.__mean_

        return result

    def fit_transform(self, X, y=None):
        """Fit and then transform some data."""
        self.fit(X)

        return self.transform(X)
