"""
Soft independent modeling of class analogies.

author: nam
"""
import numpy as np
import scipy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SIMCA:
    """
    SIMCA classifier for a single class.

    In general, you need a separate SIMCA object for each class in the dataset
    you wish to characterize. This code is based on implementation described in
    [1].  An F-test is performed based on the squared orthogonal distance (OD);
    if it is in excess of some critical value a point is not assigned to a
    class, otherwise it is.  Since a different SIMCA object is trained to
    characterize different classes, it is possible that testing a point on a
    different SIMCA class will result in multiple class assignments; however,
    each individual SIMCA class is binary.

    1. "Robust classification in high dimensions based on the SIMCA Method,"
    Vanden Branden and Hubert, Chemometrics and Intelligent Laboratory Systems
    79 (2005) 10-21.
    2. "Pattern recognition by means of disjoint principal components models,"
    S. Wold, Pattern Recognition 8 (1976) 127–139.
    """

    def __init__(self, n_components, alpha=0.05):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of PCA components to use to model this class.
        alpha : float
            Significance level.
        """
        self.set_params(**{"n_components": n_components, "alpha": alpha})

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {"n_components": self.n_components, "alpha": self.alpha}

    def column_y_(self, y):
        """Convert y to column format."""
        y = np.array(y)
        if y.ndim != 2:
            y = y.reshape(-1, 1)

        return y

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
        Fit the SIMCA model.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            clas being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        self
        """
        self.__X_ = np.array(X).copy()
        assert self.__X_.ndim == 2, "Expect 2D feature (X) matrix."
        self.n_features_in_ = self.__X_.shape[1]

        # 1. Standardize X
        self.__ss_ = StandardScaler(with_mean=True, with_std=True)

        # 2. Perform PCA on standardized coordinates
        self.__pca_ = PCA(n_components=self.n_components, random_state=0)
        self.__pca_.fit(self.__ss_.fit_transform(self.__X_))

        # 3. Compute critical F value
        II, JJ, KK = self.__X_.shape[0], self.__X_.shape[1], self.n_components
        self.__f_crit_ = scipy.stats.f.ppf(
            1.0 - self.alpha, JJ - KK, (JJ - KK) * (II - KK - 1)
        )

        return self

    def transform(self, X):
        """
        Project X into the feature subspace.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            clas being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        t-scores : matrix-like
            Projection of X via PCA into a score space.
        """
        return self.__pca_.transform(self.__ss_.transform(self.matrix_X_(X)))

    def predict(self, X):
        """
        Predict the class(es) for a given set of features.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        predictions : ndarray
            Bolean array of whether a point belongs to this class.
        """
        II, JJ, KK = self.__X_.shape[0], self.__X_.shape[1], self.n_components

        X = self.matrix_X_(X)

        X_pred = np.matmul(self.transform(X), self.__pca_.components_)
        numer = np.sum((self.__ss_.transform(X) - X_pred) ** 2, axis=1) / (
            JJ - KK
        )

        X_pred = np.matmul(self.transform(self.__X_), self.__pca_.components_)
        OD2 = np.sum((self.__ss_.transform(self.__X_) - X_pred) ** 2, axis=1)
        denom = np.sum(OD2) / ((JJ - KK) * (II - KK - 1))

        # F-score for each distance
        F = numer / denom

        # If f < f_crit, it belongs to the class
        return F < self.__f_crit_

    def score(self, X, y):
        """
        Score the prediction.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : array-like
            Boolean array of whether or not each point belongs to the class.

        Returns
        -------
        score : float
            Accuracy
        """
        y = self.column_y_(y)
        if not isinstance(y[0], bool):
            raise ValueError("y must be provided as a Boolean array")
        X_pred = self.predict(X)
        assert (
            y.shape[0] == X_pred.shape[0]
        ), "X and y do not have the same dimensions."

        return np.sum(X_pred == y.ravel()) / X_pred.shape[0]

    def _get_tags(self):
        """For compatibility with sklearn >=0.21."""
        return {
            "allow_nan": False,
            "binary_only": True,
            "multilabel": False,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "preserves_dtype": [np.float64],
            "poor_score": False,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": True,  # Skip since get_tags is unstable anyway
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }


class DDSIMCA:
    """
    Data-driven SIMCA.

    DD-SIMCA uses a combination of OD and SD, modeled by a chi-squared
    distribution, to determine the acceptance criteria to belong to a class.
    The degrees of freedom in this model are estimated from a data-driven
    approach. This implementation follows [1].

    As in SIMCA, this is designed to be a binary classification tool (yes/no)
    for a single class.  A separate object must be trained for each class you
    wish to model.

    1. "Acceptance areas for multivariate classification derived by projection
    methods," Pomerantsev, Journal of Chemometrics 22 (2008) 601-609.
    """

    def __init__(self, n_components, alpha=0.05):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of PCA components to use to model this class.
        alpha : float
            Significance level.
        """
        self.set_params(**{"n_components": n_components, "alpha": alpha})

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {"n_components": self.n_components, "alpha": self.alpha}

    def column_y_(self, y):
        """Convert y to column format."""
        y = np.array(y)
        if y.ndim != 2:
            y = y.reshape(-1, 1)

        return y

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
        Fit the SIMCA model.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            clas being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        self
        """
        self.__X_ = np.array(X).copy()
        assert self.__X_.ndim == 2, "Expect 2D feature (X) matrix."
        self.n_features_in_ = self.__X_.shape[1]

        # 1. Standardize X
        self.__ss_ = StandardScaler(with_mean=True, with_std=True)

        # 2. Perform PCA on standardized coordinates
        self.__pca_ = PCA(n_components=self.n_components, random_state=0)
        self.__pca_.fit(self.__ss_.fit_transform(self.__X_))

        # 3. Compute critical distance
        h_vals, q_vals = self.h_q_(self.__X_)
        self.__h0_, self.__q0_ = np.mean(h_vals), np.mean(q_vals)
        self.__Nh_, self.__Nq_ = self.estimate_dof_(h_vals, q_vals)

        self.__c_crit_ = scipy.stats.chi2.ppf(
            1.0 - self.alpha, self.__Nh_ + self.__Nq_
        )

        return self

    def transform(self, X):
        """
        Project X into the feature subspace.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            clas being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        t-scores : matrix-like
            Projection of X via PCA into a score space.
        """
        return self.__pca_.transform(self.__ss_.transform(self.matrix_X_(X)))

    def h_q_(self, X_raw):
        """Compute the h (OD) and q (SD) distances."""
        X_raw_std = self.__ss_.transform(X_raw)
        T = self.__pca_.transform(X_raw_std)
        X_pred = np.matmul(T, self.__pca_.components_)

        # OD
        q_vals = np.sum((X_raw_std - X_pred) ** 2, axis=1)

        # SD
        h_vals = np.sum(T ** 2 / self.__pca_.explained_variance_, axis=1)

        return h_vals, q_vals

    def estimate_dof_(self, h_vals, q_vals):
        """Estimate the degrees of freedom for the chi-squared distribution."""

        def err2(N, vals):  # Use "robust" method for estimating DoF
            x0 = np.mean(vals)
            a = (
                scipy.stats.chi2.ppf(0.75, N) - scipy.stats.chi2.ppf(0.25, N)
            ) / N
            b = scipy.stats.iqr(vals, rng=(25, 75)) / x0

            return (a - b) ** 2

        # As in conclusions of [1], Nh ~ n_components is expected
        res = scipy.optimize.minimize(
            err2, self.n_components, args=(h_vals), method="Nelder-Mead"
        )
        if res.success:
            Nh = res.x[0]
        else:
            raise Exception("Could not compute N_h : {}".format(res.message))

        # As in conclusions of [1], Nq ~ rank(X)-n_components is expected;
        # assuming near full rank then this is min(I,J)-n_components
        # (n_components<=J)
        res = scipy.optimize.minimize(
            err2,
            np.min([len(q_vals), self.n_features_in_]) - self.n_components,
            args=(q_vals),
            method="Nelder-Mead",
        )
        if res.success:
            Nq = res.x[0]
        else:
            raise Exception("Could not compute N_q : {}".format(res.message))

        return Nh, Nq

    def distance(self, X):
        """
        Compute the distance of points to this class.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            clas being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        distance : ndarray
            Distance to class.
        """
        h, q = self.h_q_(self.matrix_X_(X))

        return self.__Nh_ * h / self.__h0_ + self.__Nq_ * q / self.__q0_

    def predict(self, X):
        """
        Predict the class(es) for a given set of features.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        predictions : ndarray
            Bolean array of whether a point belongs to this class.
        """
        # If c < c_crit, it belongs to the class
        return self.distance(self.matrix_X_(X)) < self.__c_crit_

    def _get_tags(self):
        """For compatibility with sklearn >=0.21."""
        return {
            "allow_nan": False,
            "binary_only": True,
            "multilabel": False,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "preserves_dtype": [np.float64],
            "poor_score": False,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": True,  # Skip since get_tags is unstable anyway
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
