"""
Projection to Latent Structures (PLS).

author: nam
"""
import numpy as np
import scipy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from classifier.utils import CustomScaler, estimate_dof


class PLS(RegressorMixin, BaseEstimator):
    """
    Perform a Partial Least Squares Regression (PLS) aka Projection to Latent Structures Regression.

    Notes
    -----
    * X and y are always centered internally.
    * A single, scalar output (y) is expected for each observation. This is to allow
    for outlier detection and analysis following [1].

    [1] "Acceptance areas for multivariate classification derived by projection
    methods," Pomerantsev, Journal of Chemometrics 22 (2008) 601-609.
    """

    def __init__(
        self,
        n_components=1,
        alpha=0.05,
        gamma=0.01,
        scale_x=False,
        scale_y=False,
    ):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of dimensions to project into. Should be in the range
            [1, min(n_samples-1, n_features)].
        alpha : float
            Type I error rate (signficance level).
        gamma : float
            Significance level for determining outliers.
        scale_x : bool
            Whether or not to scale X columns by the standard deviation.
        scale_y : bool
            Whether or not to scale Y by its standard deviation.
        """
        self.set_params(
            **{
                "n_components": n_components,
                "alpha": alpha,
                "gamma": gamma,
                "scale_x": scale_x,
                "scale_y": scale_y,
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
            "scale_y": self.scale_y,
        }

    def column_y_(self, y):
        """Convert y to column format."""
        y = np.array(y)
        if y.ndim != 2:
            y = y[:, np.newaxis]

        return y

    def fit(self, X, y):
        """
        Fit the PLS model.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : array-like
            Response values. Should only have a single scalar response for each
            observation.

        Returns
        -------
        self
        """
        if scipy.sparse.issparse(X) or scipy.sparse.issparse(y):
            raise ValueError("Cannot use sparse data.")
        self.__X_ = np.array(X).copy()
        self.__X_, y = check_X_y(self.__X_, y, accept_sparse=False)
        # check_array(y, accept_sparse=False, dtype=None, force_all_finite=True)
        self.__y_ = self.column_y_(
            y
        )  # sklearn expects 1D array, convert to columns
        assert self.__y_.shape[1] == 1

        if self.__X_.shape[0] != self.__y_.shape[0]:
            raise ValueError(
                "X ({}) and y ({}) shapes are not compatible".format(
                    self.__X_.shape, self.__y_.shape
                )
            )
        self.n_features_in_ = self.__X_.shape[1]

        # 1. Preprocess X data
        self.__x_scaler_ = CustomScaler(
            with_mean=True, with_std=self.scale_x
        )  # Always center and maybe scale X
        self.__Xt_ = self.__x_scaler_.fit_transform(self.__X_)

        # 2. Preprocess Y data
        self.__y_scaler_ = CustomScaler(
            with_mean=True, with_std=self.scale_y
        )  # Always center and maybe scale Y
        self.__yt_ = self.__y_scaler_.fit_transform(self.__y_)

        # 3. Perform PLS
        upper_bound = np.min(
            [
                self.__X_.shape[0] - 1,
                self.__X_.shape[1],
            ]
        )
        lower_bound = 1
        if self.n_components > upper_bound or self.n_components < lower_bound:
            raise Exception(
                "n_components must [{}, min(n_samples-1 [{}], \
n_features [{}])] = [{}, {}].".format(
                    lower_bound,
                    self.__X_.shape[0] - 1,
                    self.__X_.shape[1],
                    lower_bound,
                    upper_bound,
                )
            )

        self.__pls_ = PLSRegression(
            n_components=self.n_components,
            scale=self.scale_x,  # False
            max_iter=10000,
            random_state=0,
        )

        # 4. Fit the projection
        T = self.__pls_.fit_transform(self.__Xt_, self.__yt_)

        # 5. Characterize the shape of that projected space
        self.__space_scaler_ = CustomScaler(with_mean=True, with_std=False)
        T_ = self.__space_scaler_.fit_transform(T)  # Center the projected space
        self.__eval_, self.__evec_ = np.linalg.eig(np.matmul(T_.T, T_))

        h_vals, q_vals = self.h_q_(self.__X_)
        self.__h0_, self.__q0_ = np.mean(h_vals), np.mean(q_vals)
        self.__Nh_, self.__Nq_ = estimate_dof(
            h_vals, q_vals, self.n_components, self.n_features_in_
        )

        self.__c_crit_ = scipy.stats.chi2.ppf(
            1.0 - self.alpha, self.__Nh_ + self.__Nq_
        )

        self.__c_out_ = scipy.stats.chi2.ppf(
            (1.0 - self.gamma) ** (1.0 / self.__X_.shape[0]),
            self.__Nh_ + self.__Nq_,
        )

        self.is_fitted_ = True
        return self

    def h_q_(self, X):
        """Compute inner and outer distances."""
        # h (SD) is computed relative to center of space (zeroed for simplicity)
        h = np.sum(
            self.__space_scaler_.transform(self.transform(X)) ** 2
            / self.__eval_,
            axis=1,
        )

        # Centering should cancel out during q (OD) calculation
        q = np.sum(
            (
                self.__x_scaler_.inverse_transform(
                    self.__pls_.inverse_transform(self.transform(X)) - X
                )
            )
            ** 2,
            axis=1,
        )

        return h, q

    def transform(self, X):
        """
        Project X into the PLS subspace.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        t-scores : matrix-like
            Projection of X via PLS into a score space.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, accept_sparse=False)
        X = np.array(X)
        assert X.shape[1] == self.n_features_in_

        return self.__pls_.transform(self.__x_scaler_.transform(X))

    def fit_transform(self, X, y):
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """
        Predict the values for a given set of features.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        predictions : ndarray
            Predicted output for each observation.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, accept_sparse=False)
        assert X.shape[1] == self.n_features_in_

        return self.__y_scaler_.inverse_transform(
            self.__pls_.predict(self.__x_scaler_.transform(X))
        )

    def score(self, X, y):
        """
        Compute the coefficient of determination (R^2) as the score.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : array-like
            Response values.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, accept_sparse=False)
        assert X.shape[1] == self.n_features_in_
        check_array(y, accept_sparse=False, dtype=None, force_all_finite=True)
        y = self.column_y_(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X ({}) and y ({}) shapes are not compatible".format(
                    X.shape, y.shape
                )
            )

        ss_res = np.sum((self.predict(X) - y) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot

    def _get_tags(self):
        """For compatibility with sklearn >=0.21."""
        return {
            "allow_nan": False,
            "binary_only": False,
            "multilabel": False,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "poor_score": False,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": True,
            "requires_positive_y": False,
            "_skip_test": True,  # Skip since get_tags is unstable anyway
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
