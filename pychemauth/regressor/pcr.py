"""
Principal Components Regression (PCR).

author: nam
"""
import copy
import scipy
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pychemauth.preprocessing.scaling import CorrectedScaler
from pychemauth.utils import estimate_dof

from typing import Any, Union, Sequence, ClassVar
from numpy.typing import NDArray


class PCR(RegressorMixin, BaseEstimator):
    """
    Perform a Principal Components Regression (PCR).

    Parameters
    ----------
    n_components : scalar(int), optional(default=1)
        Number of dimensions to project into. Should be in the range [1, `num_features`].

    alpha : scalar(float), optional(default=0.05)
        Type I error rate (signficance level).

    gamma : scalar(float), optional(default=0.01)
        Significance level for determining outliers.

    scale_x : bool, optional(default=False)
        Whether or not to scale X columns by the standard deviation.

    center_y : bool, optional(default=False)
        Whether ot not to center the Y responses.

    scale_y : bool, optional(default=False)
        Whether or not to scale Y by its standard deviation.

    robust : str, optional(default="semi")
        Whether or not to apply robust methods to estimate degrees of freedom. "full" is not implemented yet, but involves robust PCA and robust degrees of freedom estimation; "semi" (default) is described in [3] and uses classical PCA but robust DoF estimation; all other values revert to classical PCA and classical DoF estimation. If the dataset is clean (no outliers) it is best practice to use a classical method [3], however, to initially test for and potentially remove these points, a robust variant is recommended. This is why "semi" is the default value.

    sft : bool, optional(default=False)
        Whether or not to use the iterative outlier removal scheme described in [2], called "sequential focused trimming."  If not used (default) robust estimates of parameters may be attempted; if the iterative approach is used, these robust estimates are only computed during the outlier removal loop(s) while the final "clean" data uses classical estimates.  This option may throw away data it is originally provided for training; it keeps only "regular" samples (inliers and extremes) to train the model.

    Note
    ----
    This is designed to regress a single, scalar target for each observation. `X` data is always (column) centered, but may or may not be scaled by its standard deviation. `Y` data may or may not be centered and/or scaled.

    This is almost identical to
    >>> pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression(fit_intercept=True))
    >>> pcr.fit(X_train, y_train)

    References
    -----------
    This implementation is more explicit and enables a more detailed handling
    of outliers. See references such as:

    [1] Pomerantsev AL., Chemometrics in Excel, John Wiley & Sons, Hoboken NJ, 20142.

    [2] Rodionova OY., Pomerantsev AL. "Detection of Outliers in Projection-Based Modeling", Anal. Chem. 2020, 92, 2656−2664.

    [3] "Concept and role of extreme objects in PCA/SIMCA," Pomerantsev, A. and Rodionova, O., Journal of Chemometrics 28 (2014) 429-438.

    [4] "Acceptance areas for multivariate classification derived by projection methods," Pomerantsev, A., Journal of Chemometrics 22 (2008) 601-609.
    """

    n_components: ClassVar[int]
    alpha: ClassVar[float]
    gamma: ClassVar[float]
    scale_x: ClassVar[bool]
    center_y: ClassVar[bool]
    scale_y: ClassVar[bool]
    robust: ClassVar[str]
    sft: ClassVar[bool]

    def __init__(
        self,
        n_components: int = 1,
        alpha: float = 0.05,
        gamma: float = 0.01,
        scale_x: bool = False,
        center_y: bool = False,
        scale_y: bool = False,
        robust: str = "semi",
        sft: bool = False,
    ) -> None:
        """Instantiate the class."""
        self.set_params(
            **{
                "n_components": n_components,
                "alpha": alpha,
                "gamma": gamma,
                "scale_x": scale_x,
                "center_y": center_y,
                "scale_y": scale_y,
                "robust": robust,
                "sft": sft,
            }
        )

    def set_params(self, **parameters: Any) -> "PCR":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "n_components": self.n_components,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "scale_x": self.scale_x,
            "center_y": self.center_y,
            "scale_y": self.scale_y,
            "robust": self.robust,
            "sft": self.sft,
        }

    def fit(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.floating]],
        y: Union[Sequence[float], NDArray[np.floating]],
    ) -> "PCR":
        """
        Fit the PCR model.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        y : array-like(float, ndim=1)
            Response values. Should only have a single scalar response for each observation.

        Returns
        -------
        self : PCR
            Fitted model.
        """

        def train(X, y, robust):
            """Train the model."""
            self.__X_, self.__y_ = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_2d=True,
                force_all_finite=True,
                y_numeric=True,
                copy=True,
            )
            self.__y_ = self.__y_.reshape(-1, 1)
            self.n_features_in_ = self.__X_.shape[1]

            if robust == "full":
                raise NotImplementedError
            else:
                # 1. Preprocess X data
                self.__x_scaler_ = CorrectedScaler(
                    with_mean=True, with_std=self.scale_x
                )  # Always center and maybe scale X

                # 2. Preprocess Y data
                self.__y_scaler_ = CorrectedScaler(
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
                if (
                    self.n_components > upper_bound
                    or self.n_components < lower_bound
                ):
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

            # 5. Characterize outliers
            h_vals, q_vals = self._h_q(self.__X_)

            # As in the conclusions of [4], Nh ~ n_components is expected so good initial guess
            self.__Nh_, self.__h0_ = estimate_dof(
                h_vals,
                robust=(
                    True if (robust == "semi" or robust == "full") else False
                ),
                initial_guess=self.n_components,
            )

            # As in the conclusions of [4], Nq ~ rank(X)-n_components is expected;
            # assuming near full rank then this is min(I,J)-n_components
            # (n_components<=J)
            self.__Nq_, self.__q0_ = estimate_dof(
                q_vals,
                robust=(
                    True if (robust == "semi" or robust == "full") else False
                ),
                initial_guess=np.min([len(q_vals), self.n_features_in_])
                - self.n_components,
            )

            self.__Nf_ = self.__Nh_ + self.__Nq_
            self.__f0_ = (
                self.__Nf_
            )  # This term is a matter of convention to match the literature

            z_vals = self._z(
                self.__X_, self.__y_.ravel()
            )  # Must come after fitting is otherwise complete
            self.__Nz_, self.__z0_ = estimate_dof(
                z_vals,
                robust=(
                    True if (robust == "semi" or robust == "full") else False
                ),
                initial_guess=self.__y_.shape[1],
            )

            self.__x_crit_ = scipy.stats.chi2.ppf(1.0 - self.alpha, self.__Nf_)
            self.__x_out_ = scipy.stats.chi2.ppf(
                (1.0 - self.gamma) ** (1.0 / self.__X_.shape[0]),
                self.__Nf_,
            )

            self.__xy_crit_ = scipy.stats.chi2.ppf(
                1.0 - self.alpha, self.__Nf_ + self.__Nz_
            )
            self.__xy_out_ = scipy.stats.chi2.ppf(
                (1.0 - self.gamma) ** (1.0 / self.__X_.shape[0]),
                self.__Nf_ + self.__Nz_,
            )

        # This is based on [2]
        if not self.sft:
            train(X, y, robust=self.robust)
            self.__sft_history_ = {}
        else:
            X_tmp = np.asarray(X).copy()
            y_tmp = np.asarray(y).copy().ravel()
            total_data_points = X_tmp.shape[0]
            X_out = np.empty((0, X_tmp.shape[1]), dtype=type(X_tmp))
            y_out = np.array([], dtype=type(y_tmp))
            outer_iters = 0
            max_outer = 100
            max_inner = 100
            sft_tracker = {}
            while True:  # Outer loop
                if outer_iters >= max_outer:
                    raise Exception(
                        "Unable to iteratively clean data; exceeded maximum allowable outer loops (to eliminate swamping)."
                    )
                train(X_tmp, y_tmp, robust="semi")
                _, outliers = self.check_xy_outliers(X_tmp, y_tmp)
                X_delete_ = X_tmp[outliers, :]
                y_delete_ = y_tmp[outliers]
                inner_iters = 0
                while np.sum(outliers) > 0:
                    if inner_iters >= max_inner:
                        raise Exception(
                            "Unable to iteratively clean data; exceeded maximum allowable inner loops (to eliminate masking)."
                        )
                    X_tmp = X_tmp[~outliers, :]
                    y_tmp = y_tmp[~outliers]
                    if len(X_tmp) == 0:
                        raise Exception(
                            "Unable to iteratively clean data; all observations are considered outliers."
                        )
                    train(X_tmp, y_tmp, robust="semi")
                    _, outliers = self.check_xy_outliers(X_tmp, y_tmp)
                    X_delete_ = np.vstack((X_delete_, X_tmp[outliers, :]))
                    y_delete_ = np.concatenate((y_delete_, y_tmp[outliers]))
                    inner_iters += 1
                X_out = np.vstack((X_out, X_delete_))
                y_out = np.concatenate((y_out, y_delete_))
                assert (
                    X_tmp.shape[0] + X_out.shape[0] == total_data_points
                )  # Sanity check
                assert (
                    len(y_tmp) + len(y_out) == total_data_points
                )  # Sanity check

                # All inside X_tmp are inliers or extremes (regular objects) now.
                # Check that all outliers are predicted to be outliers in the latest version trained
                # on only inlier and extremes.
                outer_iters += 1
                sft_tracker[outer_iters] = {
                    "initially removed X": X_delete_,
                    "initially removed y": y_delete_,
                    "returned X": None,
                    "returned y": None,
                }
                if len(X_out) > 0:
                    _, outliers = self.check_xy_outliers(X_out, y_out)
                    X_return = X_out[~outliers, :]
                    y_return = y_out[~outliers]
                    X_out = X_out[outliers, :]
                    y_out = y_out[outliers]
                    if len(X_return) == 0:
                        break
                    else:
                        sft_tracker[outer_iters]["returned X"] = X_return
                        sft_tracker[outer_iters]["returned y"] = y_return
                        X_tmp = np.vstack((X_tmp, X_return))
                        y_tmp = np.concatenate((y_tmp, y_return))
                else:
                    break

            # Outliers have been iteratively found, and X_tmp is a consistent set of data to use
            # which is considered "clean" so should try to use classical estimates of the parameters.
            # train() assigns X_tmp to self.__X_ also. See [3].
            assert (
                X_out.shape[0] + self.__X_.shape[0] == total_data_points
            )  # Sanity check
            assert (
                len(y_out) + self.__y_.shape[0] == total_data_points
            )  # Sanity check
            train(X_tmp, y_tmp, robust=False)
            self.__sft_history_ = {
                "outer_loops": outer_iters,
                "removed": {"X": X_out, "y": y_out},
                "iterations": sft_tracker,
            }

        return self

    @property
    def sft_history(self) -> dict[str, Any]:
        """Return the sequential focused trimming history."""
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.__sft_history_)

    def _h_q(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute inner and outer (X) distances."""
        check_is_fitted(self, "is_fitted_")
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        # These h, q should give identical results as formulas used with SIMCA
        x_scores = self.transform(X_)
        h = np.diagonal(
            np.matmul(
                np.matmul(
                    x_scores,
                    np.linalg.inv(
                        np.matmul(self.__T_train_.T, self.__T_train_)
                        / (
                            self.__T_train_.shape[0] - 1
                        ),  # For consistency with PLS from mdatools 0.14.1
                    ),
                ),
                x_scores.T,
            )
        )

        # Compare in transformed space, same as SD
        q = np.sum(
            (
                self.__pca_.inverse_transform(x_scores)
                - self.__x_scaler_.transform(X_)
            )
            ** 2,
            axis=1,
        )

        return h, q

    def _f(
        self,
        h: Union[Sequence[float], NDArray[np.floating]],
        q: Union[Sequence[float], NDArray[np.floating]],
    ) -> NDArray[np.floating]:
        """Full (X) distance, Eq. 3 in [2]."""
        check_is_fitted(self, "is_fitted_")
        return (
            self.__Nh_ * np.array(h).ravel() / self.__h0_
            + self.__Nq_ * np.array(q).ravel() / self.__q0_
        )

    def _z(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.floating]],
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Y residual squared, Eq. 7 in [2]."""
        check_is_fitted(self, "is_fitted_")
        return ((self.predict(X) - y) ** 2.0).ravel()

    def _g(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.floating]],
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """XY total distance, Eq. 9 in [2]."""
        check_is_fitted(self, "is_fitted_")
        h, q = self._h_q(X)
        f = self._f(h, q)
        z = self._z(X, y)
        g = (
            self.__Nf_ * f / self.__f0_ + self.__Nz_ * z / self.__z0_
        )  # = f + Nz*z/z0
        return g

    def transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """
        Project X into the PCA subspace.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        Returns
        -------
        t-scores : ndarray(float, ndim=2)
            Projection of X via PCA into a score space.
        """
        check_is_fitted(self, "is_fitted_")
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        return self.__pca_.transform(self.__x_scaler_.transform(X_))

    def fit_transform(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.floating]],
        y: Union[Sequence[float], NDArray[np.floating]],
    ) -> NDArray[np.floating]:
        """
        Fit and transform.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        y : array-like(float, ndim=1)
            Response values. Should only have a single scalar response for each observation.

        Returns
        -------
        t-scores : ndarray(float, ndim=2)
            Projection of `X` via PCA into a score space.
        """
        self.fit(X, y)
        return self.transform(X)

    def predict(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """
        Predict the values for a given set of features.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        Returns
        -------
        predictions : ndarray(float, ndim=1)
            Predicted output for each observation.
        """
        check_is_fitted(self, "is_fitted_")
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        T_test = self.transform(X_)

        return self.__y_scaler_.inverse_transform(
            self.__intercept_ + np.matmul(T_test, self.__coefs_)
        ).ravel()

    def score(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.floating]],
        y: Union[Sequence[float], NDArray[np.floating]],
    ) -> float:
        r"""
        Compute the coefficient of determination (:math:`R^2`) as the score.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        y : array-like(float, ndim=1)
            Response values. Should only have a single scalar response for each observation.

        Returns
        -------
        score : scalar(float)
            Coefficient of determination (:math:`R^2`).
        """
        check_is_fitted(self, "is_fitted_")
        X_, y_ = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=True,
            copy=True,
        )

        ss_res = np.sum((self.predict(X_) - y_) ** 2)
        ss_tot = np.sum((y_ - np.mean(y_)) ** 2)
        return 1.0 - ss_res / ss_tot

    def check_x_outliers(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """
        Check if outliers and extremes exist in the X data.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        Returns
        -------
        extremes : ndarray(bool, ndim=1)
            Boolean mask of `X` if each point falls between acceptance threshold (belongs to class) and the outlier threshold.

        outliers : ndarray(bool, ndim=1)
            Boolean mask of `X` if each point falls beyond the outlier threshold.

        Note
        ----
        This uses the `X` matrix's "full distance" in [2] (cf. Eq. 3).
        """
        check_is_fitted(self, "is_fitted_")
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        f = self._f(*self._h_q(X_))

        extremes = (self.__x_crit_ <= f) & (f < self.__x_out_)
        outliers = f >= self.__x_out_

        return extremes, outliers

    def check_xy_outliers(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.floating]],
        y: Union[Sequence[float], NDArray[np.floating]],
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """
        Check if outliers and extremes exist in the XY data.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        y : array-like(float, ndim=1)
            Response values. Should only have a single scalar response for each observation.

        Returns
        -------
        extremes : ndarray(bool, ndim=1)
            Boolean mask of X if each point falls between acceptance threshold (belongs to class) and the outlier threshold.

        outliers : ndarray(bool, ndim=1)
            Boolean mask of X if each point falls beyond the outlier threshold.

        Note
        ----
        This uses the system's "total distance" in [2] (cf. Eq. 9).
        """
        check_is_fitted(self, "is_fitted_")
        X_, y_ = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=True,
            copy=True,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        g = self._g(X_, y_)

        extremes = (self.__xy_crit_ <= g) & (g < self.__xy_out_)
        outliers = g >= self.__xy_out_

        return extremes, outliers

    def visualize(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.floating]],
        y: Union[Sequence[float], NDArray[np.floating]],
        figsize: Union[tuple[int, int], None] = None,
        log: bool = True,
    ) -> matplotlib.pyplot.Axes:
        r"""
        Plot the :math:`\Chi^{2}` acceptance area with observations on distance plot.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        y : array-like(float, ndim=1)
            Response values. Should only have a single scalar response for each observation.

        figsize : tuple(int, int), optional(default=None)
            Figure size.

        log : bool, optional(default=True)
            Whether or not to transform the axes using a natural logarithm.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Axes the results are plotted on.
        """
        check_is_fitted(self, "is_fitted_")
        X_, y_ = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=True,
            copy=True,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        # 1. X plot
        h_, q_ = self._h_q(X_)
        h_lim = np.linspace(0, self.__x_crit_ * self.__h0_ / self.__Nh_, 1000)
        h_lim_out = np.linspace(
            0, self.__x_out_ * self.__h0_ / self.__Nh_, 1000
        )
        q_lim = (
            (self.__x_crit_ - self.__Nh_ / self.__h0_ * h_lim)
            * self.__q0_
            / self.__Nq_
        )
        q_lim_out = (
            (self.__x_out_ - self.__Nh_ / self.__h0_ * h_lim_out)
            * self.__q0_
            / self.__Nq_
        )

        axes[0].plot(
            np.log(1.0 + h_lim / self.__h0_) if log else h_lim / self.__h0_,
            np.log(1.0 + q_lim / self.__q0_) if log else q_lim / self.__q0_,
            "g-",
        )
        axes[0].plot(
            np.log(1.0 + h_lim_out / self.__h0_)
            if log
            else h_lim_out / self.__h0_,
            np.log(1.0 + q_lim_out / self.__q0_)
            if log
            else q_lim_out / self.__q0_,
            "r-",
        )
        xlim, ylim = (
            1.1
            * np.max(
                np.log(1.0 + h_lim_out / self.__h0_)
                if log
                else h_lim_out / self.__h0_
            ),
            1.1
            * np.max(
                np.log(1.0 + q_lim_out / self.__q0_)
                if log
                else q_lim_out / self.__q0_
            ),
        )

        ext_mask, out_mask = self.check_x_outliers(X_)
        in_mask = (~ext_mask) & (~out_mask)
        for c, mask, label in [
            (
                "g",
                in_mask,
                "Inlier =" + " ({})".format(np.sum(in_mask)),
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
            axes[0].plot(
                np.log(1.0 + h_[mask] / self.__h0_),
                np.log(1.0 + q_[mask] / self.__q0_),
                label=label,
                marker="o",
                lw=0,
                color=c,
                alpha=0.35,
            )
        xlim = np.max(
            [
                xlim,
                1.1
                * np.max(
                    np.log(1.0 + h_ / self.__h0_) if log else h_ / self.__h0_
                ),
            ]
        )
        ylim = np.max(
            [
                ylim,
                1.1
                * np.max(
                    np.log(1.0 + q_ / self.__q0_) if log else q_ / self.__q0_
                ),
            ]
        )
        axes[0].legend(loc="upper right")
        axes[0].set_xlim(0, xlim)
        axes[0].set_ylim(0, ylim)
        axes[0].set_xlabel(
            r"${\rm ln(1 + h/h_0)}$" if log else r"${\rm h/h_0}$"
        )
        axes[0].set_ylabel(
            r"${\rm ln(1 + q/q_0)}$" if log else r"${\rm q/q_0}$"
        )
        axes[0].set_title("Full Distance (X)")

        # 2. XY plot
        f_, z_ = self._f(h_, q_), self._z(X_, y_)
        f_lim = np.linspace(0, self.__xy_crit_ * self.__f0_ / self.__Nf_, 1000)
        f_lim_out = np.linspace(
            0, self.__xy_out_ * self.__f0_ / self.__Nf_, 1000
        )
        z_lim = (
            (self.__xy_crit_ - self.__Nf_ / self.__f0_ * f_lim)
            * self.__z0_
            / self.__Nz_
        )
        z_lim_out = (
            (self.__xy_out_ - self.__Nf_ / self.__f0_ * f_lim_out)
            * self.__z0_
            / self.__Nz_
        )

        axes[1].plot(
            np.log(1.0 + f_lim / self.__f0_) if log else f_lim / self.__f0_,
            np.log(1.0 + z_lim / self.__z0_) if log else z_lim / self.__z0_,
            "g-",
        )
        axes[1].plot(
            np.log(1.0 + f_lim_out / self.__f0_)
            if log
            else f_lim_out / self.__f0_,
            np.log(1.0 + z_lim_out / self.__z0_)
            if log
            else z_lim_out / self.__z0_,
            "r-",
        )
        xlim, ylim = (
            1.1
            * np.max(
                np.log(1.0 + f_lim_out / self.__f0_)
                if log
                else f_lim_out / self.__f0_
            ),
            1.1
            * np.max(
                np.log(1.0 + z_lim_out / self.__z0_)
                if log
                else z_lim_out / self.__z0_
            ),
        )

        ext_mask, out_mask = self.check_xy_outliers(X_, y_)
        in_mask = (~ext_mask) & (~out_mask)
        for c, mask, label in [
            (
                "g",
                in_mask,
                "Inlier =" + " ({})".format(np.sum(in_mask)),
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
            axes[1].plot(
                np.log(1.0 + f_[mask] / self.__f0_)
                if log
                else f_[mask] / self.__f0_,
                np.log(1.0 + z_[mask] / self.__z0_)
                if log
                else z_[mask] / self.__z0_,
                label=label,
                marker="o",
                lw=0,
                color=c,
                alpha=0.35,
            )
        xlim = np.max(
            [
                xlim,
                1.1
                * np.max(
                    np.log(1.0 + f_ / self.__f0_) if log else f_ / self.__f0_
                ),
            ]
        )
        ylim = np.max(
            [
                ylim,
                1.1
                * np.max(
                    np.log(1.0 + z_ / self.__z0_) if log else z_ / self.__z0_
                ),
            ]
        )
        axes[1].legend(loc="upper right")
        axes[1].set_xlim(0, xlim)
        axes[1].set_ylim(0, ylim)
        axes[1].set_xlabel(
            r"${\rm ln(1 + f/f_0)}$" if log else r"${\rm f/f_0}$"
        )
        axes[1].set_ylabel(
            r"${\rm ln(1 + z/z_0)}$" if log else r"${\rm z/z_0}$"
        )
        axes[1].set_title("Total Distance (XY)")
        plt.tight_layout()

        return axes

    def _get_tags(self):
        """For compatibility with scikit-learn >=0.21."""
        return {
            "allow_nan": False,
            "array_api_support": False,
            "binary_only": False,
            "multilabel": False,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "preserves_dtype": [np.float64],  # Only for transformers
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": True,
            "requires_positive_y": False,
            "_skip_test": [
                "check_fit2d_1sample",  # Can't fit 1D data
            ],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
