"""
Filter data.

@author: nam
"""
import numpy as np
import scipy.signal
from sklearn.utils.validation import check_array, check_is_fitted


class SavGol:
    """Perform a Savitzky-Golay filtering."""

    def __init__(
        self,
        window_length,
        polyorder,
        deriv=0,
        delta=1.0,
        axis=-1,
        mode="interp",
        cval=0.0,
    ):
        """
        Instantiate the class.

        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
        for description of parameters.
        """
        self.set_params(
            **{
                "window_length": window_length,
                "polyorder": polyorder,
                "deriv": deriv,
                "delta": delta,
                "axis": axis,
                "mode": mode,
                "cval": cval,
            }
        )
        self.is_fitted_ = False

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {
            "window_length": self.window_length,
            "polyorder": self.polyorder,
            "deriv": self.deriv,
            "delta": self.delta,
            "axis": self.axis,
            "mode": self.mode,
            "cval": self.cval,
        }

    def fit(self, X, y=None):
        """Fit the filter using some training data."""
        X = check_array(X, accept_sparse=False)

        try:
            _ = scipy.signal.savgol_filter(
                X,
                window_length=self.window_length,
                polyorder=self.polyorder,
                deriv=self.deriv,
                delta=self.delta,
                axis=self.axis,
                mode=self.mode,
                cval=self.cval,
            )
        except Exception as e:
            raise ValueError(
                "Cannot perform Savitzky-Golay filtering with these parameters : {}".format(
                    e
                )
            )

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def transform(self, X):
        """Transform (center and possibly scale) the data after fitting."""
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, "is_fitted_")
        assert X.shape[1] == self.n_features_in_

        X_filtered = scipy.signal.savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            axis=self.axis,
            mode=self.mode,
            cval=self.cval,
        )

        return X_filtered

    def fit_transform(self, X, y=None):
        """Fit and then transform some data."""
        self.fit(X)

        return self.transform(X)
