"""
Principal Components Regression (PCR).

author: nam
"""
import numpy as np
import scipy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from classifier.utils import CustomScaler


class PCR(RegressorMixin, BaseEstimator):
    """
    Perform a Principal Components Regression (PCR).

    Notes
    -----
    This is almost identical to
    >>> pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression(fit_intercept=True))
    >>> pcr.fit(X_train, y_train)

    This implementation is more explicit. See references such as:

    1. Pomerantsev AL., Chemometrics in Excel, John Wiley & Sons, Hoboken NJ, 20142.
    2. Rodionova OY., Pomerantsev AL. "Detection of Outliers in Projection-Based Modeling", Anal. Chem. 2020, 92, 2656−2664.
    """

    def __init__(
        self, n_components=1, scale_x=False, center_y=False, scale_y=False
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
        center_y : bool
            Whether ot not to center the Y responses.
        scale_y : bool
            Whether or not to scale Y by its standard deviation.
        """
        self.set_params(
            **{
                "n_components": n_components,
                "scale_x": scale_x,
                "center_y": center_y,
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
            "scale_x": self.scale_x,
            "center_y": self.center_y,
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
        Fit the PCR model.

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

        # 2. Preprocess Y data
        self.__y_scaler_ = CustomScaler(
            with_mean=self.center_y, with_std=self.scale_y
        )  # Maybe center and maybe scale Y
        self.__yt_ = self.__y_scaler_.fit_transform(self.__y_)

        # 3. Perform PCA on X data
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

        self.__pca_ = PCA(
            n_components=self.n_components,
            random_state=0,
        )
        self.__T_train_ = self.__pca_.fit_transform(
            self.__x_scaler_.fit_transform(self.__X_)
        )

        # 4. Fit the projection
        # Add column to include an intercept term in the projected space
        t_new = np.hstack(
            (np.ones((self.__T_train_.shape[0], 1)), self.__T_train_)
        )
        Q = np.matmul(
            np.matmul(np.linalg.inv(np.matmul(t_new.T, t_new)), t_new.T),
            self.__yt_,
        )
        self.__intercept_ = Q[0]
        self.__coefs_ = Q[1:]

        self.is_fitted_ = True
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
        t-scores : matrix-like
            Projection of X via PCA into a score space.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, accept_sparse=False)
        X = np.array(X)
        assert X.shape[1] == self.n_features_in_

        return self.__pca_.transform(self.__x_scaler_.transform(X))

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

        T_test = self.transform(X)

        return self.__y_scaler_.inverse_transform(
            self.__intercept_ + np.matmul(T_test, self.__coefs_)
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
