"""
Fill in missing data.

author: nam
"""
import sys

import numpy as np
from sklearn.decomposition import PCA


class PCA_IA:
    """
    Use PCA to iteratively estimate missing data values.

    Notes
    -----
    This implementation follows:

    [1] Walczak, B. and Massart, D. L., "Dealing with missing data: Part I," Chemometrics and Intelligent Laboratory Systems 58 (2001) 15-27.
    [2] Walczak, B. and Massart, D. L., "Dealing with missing data: Part II," Chemometrics and Intelligent Laboratory Systems 58 (2001) 29-42.
    """

    def __init__(
        self,
        n_components=1,
        scale_x=False,
    ):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of dimensions to project into. Should be in the range
            [1, num_features].
        scale_x : bool
            Whether or not to scale X columns by the standard deviation.
        """
        self.set_params(
            **{
                "n_components": n_components,
                "scale_x": scale_x,
            }
        )

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {
            "n_components": self.n_components,
            "scale_x": self.scale_x,
        }

    def matrix_X_(self, X):
        """Check that observations are rows of X."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert (
            X.shape[1] == self.n_features_in_
        ), "Incorrect number of features given in X."

        return X

    def fit(self, X, y=None):
        """
        Estimate the missing data.

        Missing data should be set to numpy.nan.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        self
        """
        if scipy.sparse.issparse(X) or scipy.sparse.issparse(y):
            raise ValueError("Cannot use sparse data.")
        self.__X_ = self.matrix_X_(X)

        return self

    def transform(self, X):
        """
        Project X into the PCA subspace.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        X : matrix-like
            Matrix with missing data filled in.
        """
        check_is_fitted(self, "is_fitted_")

        return X_filled

    def fit_transform(self, X, y):
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)
