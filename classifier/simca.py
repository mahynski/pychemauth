"""
Soft independent modeling of class analogies.

author: nam
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA

from .utils import CustomScaler


class SIMCA_Classifier(ClassifierMixin, BaseEstimator):
    """
    SIMCA against an ensemble of classes.

    Essentially, a SIMCA model is trained for each classe provided in the
    .fit() step (training set).  During testing (.score()) each point is
    run through each model to see if it is predicted to belong to the
    target class or not.  The target is set when the class is instantiated
    and must be one of the classes found in the training set (this is
    checked automatically).  This allows you to pass points that belong to
    other classes, they are just ignored.  This is important so this can
    integrate with other scikit-learn, etc. workflows.
    """

    def __init__(
        self, n_components=1, alpha=0.05, target_class=None, style="simca"
    ):
        """
        Instantiate the classifier.

        Parameters
        ----------
        n_components : int
            Number of components to use in the SIMCA model.
        alpha : float
            Significance level for SIMCA model.
        target_class : str or int
            The class used to fit the SIMCA model; the rest are used
            to test specificity.
        style : str
            Type of SIMCA to use ("simca" or "dd-simca")
        """
        self.set_params(
            **{
                "n_components": n_components,
                "alpha": alpha,
                "target_class": target_class,
                "style": style,
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
            "target_class": self.target_class,
            "style": self.style,
        }

    def fit(self, X, y):
        """
        Fit the SIMCA model.

        Only data of the target class will be used for fitting, though more
        can be provided. This is important in the case that, for example,
        SMOTE is used to up-sampled minority classes; in that case, those
        must be part of the pipeline for those steps to work automatically.
        However, a user may manually provide only the data of interest.

        Parameters
        ----------
        X : ndarray
            Inputs
        y : ndarray
            Class labels or indices. Should include some data of
            'target_class'.
        """
        if scipy.sparse.issparse(X) or scipy.sparse.issparse(y):
            raise ValueError("Cannot use sparse data.")
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = self.__X_.shape[1]

        # Fit model to target data
        if self.style == "simca":
            self.__model_ = SIMCA(
                n_components=self.n_components, alpha=self.alpha
            )
        elif self.style == "dd-simca":
            self.__model_ = DDSIMCA(
                n_components=self.n_components, alpha=self.alpha
            )
        else:
            raise ValueError("{} is not a recognized style.".format(self.style))

        assert self.target_class in np.unique(
            y
        ), "target_class not in training set"
        self.__model_.fit(X[y == self.target_class])
        self.is_fitted_ = True

    @property
    def CSPS(self):
        """Class specificities."""
        return copy.deepcopy(self.__CSPS_)

    @property
    def TSNS(self):
        """Total sensitivity of the model."""
        return copy.deepcopy(self.__TSNS_)

    @property
    def TSPS(self):
        """Total specificity of the model."""
        return copy.deepcopy(self.__TSPS_)

    def score(self, X, y):
        """
        Score the model (uses total efficiency as score).

        Parameters
        ----------
        X : ndarray
            Inputs
        y : ndarray
            Class labels or indices
        """
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(X, y, accept_sparse=False)

        self.__alternatives_ = [
            c for c in sorted(np.unique(y)) if c != self.target_class
        ]

        mask = y == self.target_class
        self.__TSNS_ = np.sum(self.__model_.predict(X[mask])) / np.sum(
            mask
        )  # TSNS = CSNS for SIMCA

        self.__CSPS_ = {}
        for class_ in self.__alternatives_:
            mask = y == class_
            self.__CSPS_[class_] = 1.0 - np.sum(
                self.__model_.predict(X[mask])
            ) / np.sum(mask)

        mask = y != self.target_class
        self.__TSPS_ = 1.0 - np.sum(self.__model_.predict(X[mask])) / np.sum(
            mask
        )

        TEFF = np.sqrt(self.__TSNS_ * self.__TSPS_)

        return TEFF


class SIMCA(ClassifierMixin, BaseEstimator):
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
    3. De Maesschalk et al., Chemometrics and Intelligent Laboratory Systems
    47 (1999) 65-77.
    """

    def __init__(self, n_components, alpha=0.05, scale_x=True):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of PCA components to use to model this class.
        alpha : float
            Significance level.
        scale_x : bool
            Whether or not to scale X by its sample standard deviation or not.
            This depends on the meaning of X and is up to the user to
            determine if scaling it (by the standard deviation) makes sense.
            Note that X is always centered.
        """
        self.set_params(
            **{"n_components": n_components, "alpha": alpha, "scale_x": scale_x}
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
            "scale_x": self.scale_x,
        }

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
            class being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        self
        """
        self.__X_ = np.array(X).copy()
        assert self.__X_.ndim == 2, "Expect 2D feature (X) matrix."
        self.n_features_in_ = self.__X_.shape[1]

        if (
            self.n_components
            > np.min([self.n_features_in_, self.__X_.shape[0]]) - 1
        ):
            raise Exception("Reduce the number of PCA components")

        # 1. Standardize X
        self.__ss_ = CustomScaler(with_mean=True, with_std=self.scale_x)

        # 2. Perform PCA on standardized coordinates
        self.__pca_ = PCA(n_components=self.n_components, random_state=0)
        self.__pca_.fit(self.__ss_.fit_transform(self.__X_))

        # 3. Compute critical F value
        II, JJ, KK = self.__X_.shape[0], self.__X_.shape[1], self.n_components
        if II > JJ:  # See De Maesschalk et al. Chem. Intell. Lab. Sys. 47 1999
            self.__a_ = JJ
        else:
            self.__a_ = II - 1
        self.__f_crit_ = scipy.stats.f.ppf(
            1.0 - self.alpha, self.__a_ - KK, (self.__a_ - KK) * (II - KK - 1)
        )

        return self

    def transform(self, X):
        """
        Project X into the feature subspace.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        t-scores : matrix-like
            Projection of X via PCA into a score space.
        """
        return self.__pca_.transform(self.__ss_.transform(self.matrix_X_(X)))

    def distance(self, X):
        """
        Compute the F score (distance) for a given set of observations.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        predictions : ndarray
            F value for each observation.
        """
        II, _, KK = self.__X_.shape[0], self.__X_.shape[1], self.n_components

        X = self.matrix_X_(X)

        X_pred = np.matmul(self.transform(X), self.__pca_.components_)
        # See De Maesschalk et al. Chem. Intell. Lab. Sys. 47 1999
        numer = np.sum((self.__ss_.transform(X) - X_pred) ** 2, axis=1) / (
            self.__a_ - KK
        )

        X_pred = np.matmul(self.transform(self.__X_), self.__pca_.components_)
        # See De Maesschalk et al. Chem. Intell. Lab. Sys. 47 1999
        OD2 = np.sum((self.__ss_.transform(self.__X_) - X_pred) ** 2, axis=1)
        denom = np.sum(OD2) / ((self.__a_ - KK) * (II - KK - 1))

        # F-score for each distance
        F = numer / denom

        return F

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
        F = self.distance(X)

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


class DDSIMCA(ClassifierMixin, BaseEstimator):
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

    def __init__(self, n_components, alpha=0.05, scale_x=True):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of PCA components to use to model this class.
        alpha : float
            Significance level.
        scale_x : bool
            Whether or not to scale X by its sample standard deviation or not.
            This depends on the meaning of X and is up to the user to
            determine if scaling it (by the standard deviation) makes sense.
            Note that X is always centered.
        """
        self.set_params(
            **{"n_components": n_components, "alpha": alpha, "scale_x": scale_x}
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
            "scale_x": self.scale_x,
        }

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

        if (
            self.n_components
            > np.min([self.n_features_in_, self.__X_.shape[0]]) - 1
        ):
            raise Exception("Reduce the number of PCA components")

        # 1. Standardize X
        self.__ss_ = CustomScaler(with_mean=True, with_std=self.scale_x)

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
            class being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        t-scores : matrix-like
            Projection of X via PCA into a score space.
        """
        return self.__pca_.transform(self.__ss_.transform(self.matrix_X_(X)))

    def h_q_(self, X_raw):
        """Compute the h (OD) and q (SD) distances."""
        X_raw_std = self.__ss_.transform(self.matrix_X_(X_raw))
        T = self.__pca_.transform(X_raw_std)
        X_pred = np.matmul(T, self.__pca_.components_)

        # OD
        q_vals = np.sum((X_raw_std - X_pred) ** 2, axis=1)

        # SD
        h_vals = np.sum(T ** 2 / self.__pca_.explained_variance_, axis=1) / (
            self.__X_.shape[0] - 1
        )

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
        Compute how far away points are from this class.

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

    def visualize(self, X, y, ax=None):
        """
        Plot the chi-squared acceptance area with various observations.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : matrix-like
            Labels for observations in X.
        ax : matplotlib.pyplot.axes
            Axis object to plot on (optional).
        """
        h_lim = np.linspace(0, self.__c_crit_ * self.__h0_ / self.__Nh_, 1000)
        q_lim = (
            (self.__c_crit_ - self.__Nh_ / self.__h0_ * h_lim)
            * self.__q0_
            / self.__Nq_
        )

        if ax is None:
            fig = plt.figure()
            axis = fig.gca()
        else:
            axis = ax

        axis.plot(
            np.log(1.0 + h_lim / self.__h0_),
            np.log(1.0 + q_lim / self.__q0_),
            "r-",
        )
        xlim, ylim = 0, 0
        X_ = self.matrix_X_(X)
        y_ = np.array(y)
        for i, class_ in enumerate(sorted(np.unique(y_))):
            h_, q_ = self.h_q_(X_[y_ == class_])
            axis.plot(
                np.log(1.0 + h_ / self.__h0_),
                np.log(1.0 + q_ / self.__q0_),
                color="C{}".format(i),
                label=class_,
                lw=0,
                marker="o",
            )
            xlim = np.max([xlim, 1.1 * np.max(np.log(1.0 + h_ / self.__h0_))])
            ylim = np.max([ylim, 1.1 * np.max(np.log(1.0 + q_ / self.__q0_))])
        axis.legend(loc="best")
        axis.set_xlim(0, xlim)
        axis.set_ylim(0, ylim)
        axis.set_xlabel(r"${\rm ln(1 + h/h_0)}$")
        axis.set_ylabel(r"${\rm ln(1 + q/q_0)}$")

        return axis

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
