"""
Utility functions for classifiers.

author: nam
"""
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted


class CustomScaler:
    """
    Perform standard scaling on data.

    Custom standard scaler which reduces the degrees of freedom
    by one in the standard deviation, unlike sklearn's default.
    """

    def __init__(self, with_mean=True, with_std=True):
        """
        Instantiate the class.

        Parameters
        ----------
        with_mean : bool
            Center the data using the mean.
        with_std : bool
            Scale the data using the (corrected) sample standard deviation
            which uses N-1 degrees of freedom instead of N.
        """
        self.set_params(**{"with_mean": with_mean, "with_std": with_std})
        self.is_fitted_ = False

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y=None):
        """Fit the scaler using some training data."""
        X = check_array(X, accept_sparse=False)
        # X = np.array(X)

        self.__mean_ = np.mean(X, axis=0)
        self.__std_ = np.std(X, axis=0, ddof=1)
        self.is_fitted_ = True

    def transform(self, X):
        """Transform (center and possibly scale) the data after fitting."""
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, "is_fitted_")

        result = np.array(X, dtype=np.float64)
        if self.with_mean:
            result -= self.__mean_
        if self.with_std:
            result /= self.__std_

        return result

    def inverse_transform(self, X):
        """Invert the transformation."""
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, "is_fitted_")

        result = np.array(X, dtype=np.float64)
        if self.with_std:
            result *= self.__std_
        if self.with_mean:
            result += self.__mean_

        return result

    def fit_transform(self, X, y=None):
        """Fit and then transform some data."""
        self.fit(X)

        return self.transform(X)
