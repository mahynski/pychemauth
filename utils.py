"""
Utility functions for classifiers.

author: nam
"""
import numpy as np
import scipy
from sklearn.utils.validation import check_array, check_is_fitted


def estimate_dof(h_vals, q_vals, n_components, n_features_in):
    """
    Estimate the degrees of freedom for the chi-squared distribution.

    This follows from Ref. 1.

    [1] "Acceptance areas for multivariate classification derived by projection
    methods," Pomerantsev, Journal of Chemometrics 22 (2008) 601-609.
    """

    def err2(N, vals):
        """
        Use a "robust" method for estimating DoF.

        In [1] Eq. 14 suggests the IQR should be divided by the mean (h0),
        however, the citation they provide suggests the median might be
        a better choice; in practice, it seems that is favored since it
        is more robust against outliers, so this is used below in that
        spirit.
        """
        x0 = np.median(vals)  # np.mean(vals)
        a = (scipy.stats.chi2.ppf(0.75, N) - scipy.stats.chi2.ppf(0.25, N)) / N
        b = scipy.stats.iqr(vals, rng=(25, 75)) / x0

        return (a - b) ** 2

    # As in conclusions of [1], Nh ~ n_components is expected
    res = scipy.optimize.minimize(
        err2, n_components, args=(h_vals), method="Nelder-Mead"
    )
    if res.success:
        # Robust method, if possible
        Nh = res.x[0]
    else:
        # Use simple estimate if this fails (Eq. 13 in [1])
        Nh = 2.0 * np.mean(h_vals) ** 2 / np.std(h_vals, ddof=1) ** 2

    # As in conclusions of [1], Nq ~ rank(X)-n_components is expected;
    # assuming near full rank then this is min(I,J)-n_components
    # (n_components<=J)
    res = scipy.optimize.minimize(
        err2,
        np.min([len(q_vals), n_features_in]) - n_components,
        args=(q_vals),
        method="Nelder-Mead",
    )
    if res.success:
        # Robust method, if possible
        Nq = res.x[0]
    else:
        # Use simple estimate if this fails (Eq. 23 in [1])
        Nq = 2.0 * np.mean(q_vals) ** 2 / np.std(q_vals, ddof=1) ** 2

    return Nh, Nq


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
