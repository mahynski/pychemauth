"""
Principal Components Analysis (PCA).

author: nam
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.decomposition
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

sys.path.append("../")
from utils import CustomScaler, estimate_dof


class PCA(ClassifierMixin, BaseEstimator):
    """
    Create a Principal Components Analysis (PCA) model.

    This enables deeper inspection of data through outlier analysis, etc. as
    detailed in the references below.  PCA only creates a quantitive model
    of the X data; no responses are considered (y). The primary use case for
    this is to inspect the data to classify/detect any extremes or outliers.

    Notes
    -----
    See references such as:

    [1] Pomerantsev AL., Chemometrics in Excel, John Wiley & Sons, Hoboken NJ, 20142.
    [2] Rodionova OY., Pomerantsev AL. "Detection of Outliers in Projection-Based Modeling", Anal. Chem. 2020, 92, 2656−2664.
    [3] "Acceptance areas for multivariate classification derived by projection
    methods," Pomerantsev, Journal of Chemometrics 22 (2008) 601-609.
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

    def fit(self, X, y=None):
        """
        Fit the PCR model.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : array-like
            Response values. Ignored - this is here for compatability with
            scikit-learn.

        Returns
        -------
        self
        """
        if scipy.sparse.issparse(X) or scipy.sparse.issparse(y):
            raise ValueError("Cannot use sparse data.")
        self.__X_ = np.array(X).copy()
        self.__X_ = check_array(self.__X_, accept_sparse=False)
        self.n_features_in_ = self.__X_.shape[1]

        # 1. Preprocess X data
        self.__x_scaler_ = CustomScaler(
            with_mean=True, with_std=self.scale_x
        )  # Always center and maybe scale X

        # 2. Perform PCA on X data
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

        self.__pca_ = sklearn.decomposition.PCA(
            n_components=self.n_components, svd_solver="auto"
        )
        self.__pca_.fit(self.__x_scaler_.fit_transform(self.__X_))

        self.is_fitted_ = True

        # 5. Characterize outliers
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
        check_is_fitted(self, "is_fitted_")
        return self.__pca_.transform(
            self.__x_scaler_.transform(self.matrix_X_(X))
        )

    def fit_transform(self, X, y=None):
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def h_q_(self, X):
        """Compute the h (SD) and q (OD) distances."""
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, accept_sparse=False)
        X = self.matrix_X_(X)
        assert X.shape[1] == self.n_features_in_

        X_raw_std = self.__x_scaler_.transform(X)
        T = self.__pca_.transform(X_raw_std)
        X_pred = self.__pca_.inverse_transform(
            T
        )  # np.matmul(T, self.__pca_.components_)

        # OD
        q_vals = np.sum((X_raw_std - X_pred) ** 2, axis=1)

        # SD
        h_vals = np.sum(T ** 2 / self.__pca_.explained_variance_, axis=1) / (
            self.__X_.shape[0] - 1
        )

        return h_vals, q_vals

    def distance(self, X):
        """
        Compute how far away points are from this class.

        This is computed as a sum of the OD and OD to be used with acceptance
        rule II from [3].

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        distance : ndarray
            Distance to class.
        """
        h, q = self.h_q_(X)

        return self.__Nh_ * h / self.__h0_ + self.__Nq_ * q / self.__q0_

    def predict(self, X):
        """
        Predict if the data are "regular" (NOT extremes or outliers).

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
        d = self.distance(X)

        # If d < c_crit, it is not extreme not outlier
        return d < self.__c_crit_

    def check_outliers(self, X):
        """
        Check where, if ever, extemes and outliers occur in the data.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        extremes, outliers : ndarray, ndarray
            Boolean mask of X if each point falls between acceptance threshold
            (belongs to class) and the outlier threshold (extreme), or beyond
            the outlier (outlier) threshold.
        """
        dX_ = self.distance(X)
        extremes = (self.__c_crit_ <= dX_) & (dX_ < self.__c_out_)
        outliers = dX_ >= self.__c_out_
        return extremes, outliers

    def visualize(self, X, ax=None):
        """
        Plot the chi-squared acceptance area with observations.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        """
        check_is_fitted(self, "is_fitted_")

        if ax is None:
            fig = plt.figure()
            axis = plt.gca()
        else:
            axis = ax

        h_, q_ = self.h_q_(X)
        h_lim = np.linspace(0, self.__c_crit_ * self.__h0_ / self.__Nh_, 1000)
        h_lim_out = np.linspace(
            0, self.__c_out_ * self.__h0_ / self.__Nh_, 1000
        )
        q_lim = (
            (self.__c_crit_ - self.__Nh_ / self.__h0_ * h_lim)
            * self.__q0_
            / self.__Nq_
        )
        q_lim_out = (
            (self.__c_out_ - self.__Nh_ / self.__h0_ * h_lim_out)
            * self.__q0_
            / self.__Nq_
        )

        axis.plot(
            np.log(1.0 + h_lim / self.__h0_),
            np.log(1.0 + q_lim / self.__q0_),
            "g-",
        )
        axis.plot(
            np.log(1.0 + h_lim_out / self.__h0_),
            np.log(1.0 + q_lim_out / self.__q0_),
            "r-",
        )
        xlim, ylim = (
            1.1 * np.max(np.log(1.0 + h_lim_out / self.__h0_)),
            1.1 * np.max(np.log(1.0 + q_lim_out / self.__q0_)),
        )

        ext_mask, out_mask = self.check_outliers(X)
        in_mask = (~ext_mask) & (~out_mask)
        for c, mask, label in [
            (
                "g",
                in_mask,
                "Regular =" + " ({})".format(np.sum(in_mask)),
            ),
            (
                "orange",
                ext_mask,
                "Extreme =" + " ({})".format(np.sum(ext_mask)),
            ),
            (
                "r",
                out_mask,
                "Outlier =" + " ({})".format(np.sum(out_mask)),
            ),
        ]:
            axis.plot(
                np.log(1.0 + h_[mask] / self.__h0_),
                np.log(1.0 + q_[mask] / self.__q0_),
                label=label,
                marker="o",
                lw=0,
                color=c,
                alpha=0.35,
            )
        xlim = np.max([xlim, 1.1 * np.max(np.log(1.0 + h_ / self.__h0_))])
        ylim = np.max([ylim, 1.1 * np.max(np.log(1.0 + q_ / self.__q0_))])
        axis.legend(loc="upper right")
        axis.set_xlim(0, xlim)
        axis.set_ylim(0, ylim)
        axis.set_xlabel(r"${\rm ln(1 + h/h_0)}$")
        axis.set_ylabel(r"${\rm ln(1 + q/q_0)}$")

        return axis

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
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": True,  # Skip since get_tags is unstable anyway
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
