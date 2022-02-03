"""
Principal Components Analysis (PCA).

author: nam
"""
import numpy as np
import scipy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from utils import CustomScaler, estimate_dof


class PCA(RegressorMixin, BaseEstimator):
    """
    Create a Principal Components Analysis (PCA) model.

    This enables deeper inspection of data through outlier analysis, etc. as
    detailed in the references below.  PCA only creates a quantitive model
    of the X data; no responses are considered (y).

    Notes
    -----
    See references such as:

    [1] Pomerantsev AL., Chemometrics in Excel, John Wiley & Sons, Hoboken NJ, 20142.
    [2] Rodionova OY., Pomerantsev AL. "Detection of Outliers in Projection-Based Modeling", Anal. Chem. 2020, 92, 2656−2664.
    """

    def __init__(self, n_components=1, alpha=0.05, gamma=0.01, scale_x=False):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of dimensions to project into. Should be in the range
            [1, num_features].
        alpha : float
            Type I error rate (signficance level).
        gamma : float
            Significance level for determining outliers.
        scale_x : bool
            Whether or not to scale X columns by the standard deviation.
        """
        self.set_params(
            **{
                "n_components": n_components,
                "alpha": alpha,
                "gamma": gamma,
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
            "alpha": self.alpha,
            "gamma": self.gamma,
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
