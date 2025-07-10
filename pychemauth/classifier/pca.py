"""
Principal Components Analysis (PCA).

author: nam
"""
import copy
import warnings
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.decomposition
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pychemauth.preprocessing.scaling import CorrectedScaler
from pychemauth.utils import estimate_dof, _logistic_proba

from typing import Union, Sequence, Any, ClassVar
from numpy.typing import NDArray


class PCA(BaseEstimator):  # Not a proper classifer by sklearn standards
    """
    Create a Principal Components Analysis (PCA) model.

    Parameters
    ----------
    n_components : scalar(int), optional(default=1)
        Number of dimensions to project into. Should be in the range
        [1, num_features].

    alpha : scalar(float), optional(default=0.05)
        Type I error rate (significance level).

    gamma : scalar(float), optional(default=0.01)
        Significance level for determining outliers.

    scale_x : bool, optional(default=False)
        Whether or not to scale X columns by the standard deviation.

    robust : str, optional(default="semi")
        Whether or not to apply robust methods to estimate degrees of freedom.
        "full" is not implemented yet, but involves robust PCA and robust
        degrees of freedom estimation; "semi" (default) is described in [4] and
        uses classical PCA but robust DoF estimation; all other values
        revert to classical PCA and classical DoF estimation.
        If the dataset is clean (no outliers) it is best practice to use a classical
        method [2], however, to initially test for and potentially remove these
        points, a robust variant is recommended. This is why "semi" is the
        default value.

    sft : bool, optional(default=False)
        Whether or not to use the iterative outlier removal scheme described
        in [2], called "sequential focused trimming."  If not used (default)
        robust estimates of parameters may be attempted; if the iterative
        approach is used, these robust estimates are only computed during the
        outlier removal loop(s) while the final "clean" data uses classical
        estimates.  This option may throw away data it is originally provided
        for training; it keeps only "regular" samples (inliers and extremes)
        to train the model.

    Note
    ----
    This enables deeper inspection of data through outlier analysis, etc. as
    detailed in the references below.  PCA only creates a quantitive model
    of the X data; no responses are considered (y). The primary use case for
    this is to inspect the data to classify/detect any extremes or outliers.

    References
    ----------
    See references such as:

    [1] Pomerantsev AL., Chemometrics in Excel, John Wiley & Sons, Hoboken NJ, 20142.

    [2] "Detection of Outliers in Projection-Based Modeling", Rodionova OY., Pomerantsev AL.,
    Anal. Chem. 2020, 92, 2656−2664.

    [3] "Acceptance areas for multivariate classification derived by projection
    methods," Pomerantsev, Journal of Chemometrics 22 (2008) 601-609.

    [4] "Concept and role of extreme objects in PCA/SIMCA," Pomerantsev, A. and
    Rodionova, O., Journal of Chemometrics 28 (2014) 429-438.
    """

    n_components: ClassVar[int]
    alpha: ClassVar[float]
    gamma: ClassVar[float]
    scale_x: ClassVar[bool]
    robust: ClassVar[str]
    sft: ClassVar[bool]

    def __init__(
        self,
        n_components: int = 1,
        alpha: float = 0.05,
        gamma: float = 0.01,
        scale_x: bool = False,
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
                "robust": robust,
                "sft": sft,
            }
        )

    def set_params(self, **parameters: Any) -> "PCA":
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
            "robust": self.robust,
            "sft": self.sft,
        }

    def fit(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[
            NDArray[np.floating], NDArray[np.integer], NDArray[np.str_], None
        ] = None,
    ) -> "PCA":
        """
        Fit the PCA model.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array-like(int, str or float, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        self : PCA
        """
        if y is not None:  # Just so this passes sklearn api checks
            X, _ = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_2d=True,
                force_all_finite=True,
                y_numeric=False,
                copy=False,
            )

        def train(X, robust):
            """
            Train the model.

            Parameters
            ----------
            X : ndarray(float, ndim=2)
                Data to train on.
            robust : str
                "full" = robust PCA + robust parameter estimation in [4] (not yet implemented);
                "semi" = classical PCA + robust parameter estimation in [4];
                otherwise = classical PCA + classical parameter estimation in [4];
            """
            self.__X_ = check_array(
                X,
                accept_sparse=False,
                dtype=np.float64,
                ensure_2d=True,
                force_all_finite=True,
                copy=True,
            )
            self.n_features_in_ = self.__X_.shape[1]

            if robust == "full":
                raise NotImplementedError
            else:
                # 1. Preprocess X data
                self.__x_scaler_ = CorrectedScaler(
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

                self.__pca_ = sklearn.decomposition.PCA(
                    n_components=self.n_components, svd_solver="auto"
                )
                self.__pca_.fit(self.__x_scaler_.fit_transform(self.__X_))

            self.is_fitted_ = True

            # 3. Compute critical distances
            h_vals, q_vals = self._h_q(self.__X_)

            # As in the conclusions of [1], Nh ~ n_components is expected so good initial guess
            self.__Nh_, self.__h0_ = estimate_dof(
                h_vals,
                robust=(
                    True if (robust == "semi" or robust == "full") else False
                ),
                initial_guess=self.n_components,
            )

            # As in the conclusions of [1], Nq ~ rank(X)-n_components is expected;
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

            self.__c_crit_ = scipy.stats.chi2.ppf(
                1.0 - self.alpha, self.__Nh_ + self.__Nq_
            )
            self.__c_out_ = scipy.stats.chi2.ppf(
                (1.0 - self.gamma) ** (1.0 / self.__X_.shape[0]),
                self.__Nh_ + self.__Nq_,
            )

        # This is based on [2]
        if not self.sft:
            train(X, robust=self.robust)
            self.__sft_history_ = {}
        else:
            X_tmp = np.asarray(X, dtype=np.float64).copy()
            total_data_points = X_tmp.shape[0]
            X_out = np.empty((0, X_tmp.shape[1]), dtype=type(X_tmp))
            outer_iters = 0
            max_outer = 100
            max_inner = 100
            sft_tracker = {}
            while True:  # Outer loop
                if outer_iters >= max_outer:
                    raise Exception(
                        "Unable to iteratively clean data; exceeded maximum allowable outer loops (to eliminate swamping)."
                    )
                train(X_tmp, robust="semi")
                _, outliers = self.check_outliers(X_tmp)
                X_delete_ = X_tmp[outliers, :]
                inner_iters = 0
                while np.sum(outliers) > 0:
                    if inner_iters >= max_inner:
                        raise Exception(
                            "Unable to iteratively clean data; exceeded maximum allowable inner loops (to eliminate masking)."
                        )
                    X_tmp = X_tmp[~outliers, :]
                    if len(X_tmp) == 0:
                        raise Exception(
                            "Unable to iteratively clean data; all observations are considered outliers."
                        )
                    train(X_tmp, robust="semi")
                    _, outliers = self.check_outliers(X_tmp)
                    X_delete_ = np.vstack((X_delete_, X_tmp[outliers, :]))
                    inner_iters += 1
                X_out = np.vstack((X_out, X_delete_))
                assert (
                    X_tmp.shape[0] + X_out.shape[0] == total_data_points
                )  # Sanity check

                # All inside X_tmp are inliers or extremes (regular objects) now.
                # Check that all outliers are predicted to be outliers in the latest version trained
                # on only inlier and extremes.
                outer_iters += 1
                sft_tracker[outer_iters] = {
                    "initially removed": X_delete_,
                    "returned": None,
                }
                if len(X_out) > 0:
                    _, outliers = self.check_outliers(X_out)
                    X_return = X_out[~outliers, :]
                    X_out = X_out[outliers, :]
                    if len(X_return) == 0:
                        break
                    else:
                        sft_tracker[outer_iters]["returned"] = X_return
                        X_tmp = np.vstack((X_tmp, X_return))
                else:
                    break

            # Outliers have been iteratively found, and X_tmp is a consistent set of data to use
            # which is considered "clean" so should try to use classical estimates of the parameters.
            # train() assigns X_tmp to self.__X_ also. See [2].
            assert (
                X_out.shape[0] + self.__X_.shape[0] == total_data_points
            )  # Sanity check
            train(X_tmp, robust=False)
            self.__sft_history_ = {
                "outer_loops": outer_iters,
                "removed": {"X": X_out},
                "iterations": sft_tracker,
            }

        return self

    @property
    def sft_history(self) -> dict[str, Any]:
        """Return the sequential focused trimming history."""
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.__sft_history_)

    def transform(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Project X into the feature subspace.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        t-scores : ndarray(float, ndim=2)
            Projection of X via PCA into a score space.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        return self.__pca_.transform(self.__x_scaler_.transform(X))

    def fit_transform(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[
            NDArray[np.floating], NDArray[np.integer], NDArray[np.str_], None
        ] = None,
    ) -> NDArray[np.floating]:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def _h_q(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute the h (SD) and q (OD) distances."""
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        X_raw_std = self.__x_scaler_.transform(X)
        T = self.__pca_.transform(X_raw_std)
        X_pred = self.__pca_.inverse_transform(T)

        # OD
        q_vals = np.sum((X_raw_std - X_pred) ** 2, axis=1)

        # SD
        h_vals = np.sum(T**2 / self.__pca_.explained_variance_, axis=1)

        return h_vals, q_vals

    def distance(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.floating]:
        """
        Compute how far away points are from this class.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        distance : ndarray(float, ndim=2)
            Distance to class.

        Note
        ----
        This is computed as a sum of the OD and OD to be used with acceptance
        rule II from [3].
        """
        check_is_fitted(self, "is_fitted_")
        h, q = self._h_q(X)

        return self.__Nh_ * h / self.__h0_ + self.__Nq_ * q / self.__q0_

    def decision_function(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]], y=None
    ) -> NDArray[np.floating]:
        """
        Compute the decision function for each sample.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array-like(float, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        decision_function : ndarray(float, ndim=1)
            Shifted, negative distance for each sample.

        Note
        ----
        Following scikit-learn's EllipticEnvelope, this returns the negative
        sqrt(distance) shifted by the cutoff distance, so f < 0 implies an
        extreme or outlier while f > 0 implies an inlier.

        References
        ----------
        See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-decision_function
        """
        check_is_fitted(self, "is_fitted_")
        return -np.sqrt(self.distance(X)) - (-np.sqrt(self.__c_crit_))

    def predict_proba(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]], y=None
    ) -> NDArray[np.floating]:
        """
        Predict the probability that observations are inliers.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array-like(float, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        probabilities : ndarray(float, ndim=2)
            2D array as logistic function of the decision_function(). First column
            is NOT inlier, 1-p(x), second column is inlier probability, p(x).

        Note
        ----
        Computes the logistic(decision_function(X, y)) as the
        transformation of the decision function.  This function is > 0
        for inliers so predict_proba(X) > 0.5 means inlier, < 0.5 means
        outlier or extreme.

        See SHAP documentation for a discussion on the utility and impact
        of "squashing functions": https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Squashing%20Effect.html#Probability-space-explaination

        References
        ----------
        See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-predict_proba
        """
        check_is_fitted(self, "is_fitted_")

        return _logistic_proba(self.decision_function(X, y))

    def predict(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> NDArray[np.bool_]:
        """
        Predict if the data are "inliers" (NOT extremes or outliers).

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        predictions : ndarray(bool, ndim=1)
            Boolean array of whether a point belongs to this class.
        """
        check_is_fitted(self, "is_fitted_")
        d = self.distance(X)

        # If d < c_crit, it is not extreme not outlier
        return d < self.__c_crit_

    def check_outliers(
        self, X: Union[NDArray[np.floating], Sequence[Sequence[float]]]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """
        Check where, if ever, extemes and outliers occur in the data.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        extremes : ndarray(bool, ndim=1)
             Boolean mask of X if each point falls between acceptance threshold
            (belongs to class) and the outlier threshold.

        outliers : ndarray(bool, ndim=1)
            Boolean mask of X if each point falls beyond the outlier threshold.
        """
        check_is_fitted(self, "is_fitted_")

        if self.__c_out_ < self.__c_crit_:
            warnings.warn(
                "Outlier threshold < inlier threshold; increase alpha value or decrease gamma value."
            )

        dX_ = self.distance(X)
        extremes = (self.__c_crit_ <= dX_) & (dX_ < self.__c_out_)
        outliers = dX_ >= self.__c_out_

        return extremes, outliers

    def extremes_plot(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        upper_frac: float = 0.25,
        ax: Union[matplotlib.pyplot.Axes, None] = None,
    ) -> matplotlib.pyplot.Axes:
        r"""
        Plot an "extremes plot" [4] to evalute the quality of the model.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Data to evaluate the number of non-inliers (outliers + extremes) in.

        upper_frac : scalar(float), optional(default=0.25)
            Count the number of extremes and outliers for alpha values corresponding
            to :math:`n_{\rm exp}` = [1, X.shape[0]*upper_frac], where :math:`\alpha = n_{\rm exp} / N_{\rm tot}`.

        ax : matplotlib.pyplot.Axes, optional(default=None)
            Axes to plot on.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Axes results are plotted.

        Note
        ----
        This modifies the alpha value (type I error rate), keeping all other parameters
        fixed, and computes the number of expected extremes (n_exp) vs. the number
        observed (n_obs).  Theoretically, n_exp = alpha*N_tot.

        The 95% tolerance limit is given in black.  Points which fall outside these
        bounds are highlighted.

        Warning
        -------
        Both extreme points and outliers are considered "extremes" here.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        N_tot = X.shape[0]  # type: ignore[union-attr]
        n_values = np.arange(1, int(upper_frac * N_tot) + 1)
        alpha_values = n_values / N_tot

        n_observed_ = []
        for a in alpha_values:
            params = self.get_params()
            params["alpha"] = a
            model_ = PCA(**params)
            model_.fit(X)
            extremes, outliers = model_.check_outliers(X)
            n_observed_.append(np.sum(extremes) + np.sum(outliers))
        n_observed = np.array(n_observed_)

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        n_upper = n_values + 2.0 * np.sqrt(n_values * (1.0 - n_values / N_tot))
        n_lower = n_values - 2.0 * np.sqrt(n_values * (1.0 - n_values / N_tot))
        ax.plot(n_values, n_upper, "-", color="k", alpha=0.5)
        ax.plot(n_values, n_values, "-", color="k", alpha=0.5)
        ax.plot(n_values, n_lower, "-", color="k", alpha=0.5)
        ax.fill_between(
            n_values, y1=n_upper, y2=n_lower, color="gray", alpha=0.25
        )

        mask = (n_lower <= n_observed) & (n_observed <= n_upper)
        ax.plot(n_values[mask], n_observed[mask], "o", color="green")
        ax.plot(n_values[~mask], n_observed[~mask], "o", color="red")

        ax.set_xlabel("Expected")
        ax.set_ylabel("Observed")

        return ax

    def loss(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        y: Union[NDArray[np.bool_], Sequence[bool]],
        eps: float = 1.0e-15,
    ) -> float:
        r"""
        Compute the negative log-loss, or logistic/cross-entropy loss.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array-like(bool, ndim=1)
            Correct labels; True for inlier, False for outlier.

        eps : scalar(float), optional(default=1.0e-15)
            Numerical addition to enable evaluation when log(p ~ 0).

        Returns
        -------
        loss : scalar(float)
            Negative, normalized log loss; :math:`\frac{1}{N} \sum_i \left( y_{in}(i) {\rm ln}(p_{in}(i)) + (1-y_{in}(i)) {\rm ln}(1-p_{in}(i)) \right)`

        References
        ----------
        See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss.
        """
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        assert np.all(
            [a in [True, False] for a in y]
        ), "y should contain only True or False labels"

        # Inlier = True (positive class), p[:,0]
        # Not inlier = False (negative class), p[:,1]
        prob = self.predict_proba(X, y)

        y_in = np.array([1.0 if y_ == True else 0.0 for y_ in y])
        p_in = np.clip(prob[:, 1], a_min=eps, a_max=1.0 - eps)

        # Return the negative, normalized log-loss
        return -np.sum(
            y_in * np.log(p_in) + (1.0 - y_in) * np.log(1.0 - p_in)
        ) / len(X)

    def plot_loadings(
        self,
        feature_names: Union[NDArray[np.str_], Sequence[str], None] = None,
        ax: Union[matplotlib.pyplot.Axes, None] = None,
    ) -> matplotlib.pyplot.Axes:
        """
        Make a 2D loadings plot.

        Parameters
        ----------
        feature_names : array-like(str, ndim=1), optional(default=None)
            List of names of each columns in X. Otherwise displays indices.

        ax : matplotlib.pyplot.Axes, optional(default=None)
            Axes to plot on.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Axes results are plotted on.

        Note
        ----
        This uses the top 2 eigenvectors regardless of the model dimensionality. If it
        is less than 2 a ValueError is returned.
        """
        check_is_fitted(self, "is_fitted_")
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        if feature_names is None:
            feature_names = [str(i) for i in range(self.n_features_in_)]

        if self.n_components < 2:
            raise ValueError(
                "Cannot visualize when using less than 2 components."
            )

        if len(feature_names) != self.n_features_in_:
            raise ValueError("Must provide a name for each column.")

        # Some communities multiply by the sqrt(eigenvalues) - for consistency with mdatools 0.14.1 we do not.
        a = (
            self.__pca_.components_.T
        )  # * np.sqrt(self.__pca_.explained_variance_)
        ax.plot(a[:, 0], a[:, 1], "o")
        ax.axvline(0, ls="--", color="k")
        ax.axhline(0, ls="--", color="k")
        for i, label in zip(range(len(a)), feature_names):
            ax.text(a[i, 0], a[i, 1], label)
        ax.set_xlabel(
            "PC 1 ({}%)".format(
                "%.4f" % (self.__pca_.explained_variance_ratio_[0] * 100.0)
            )
        )
        ax.set_ylabel(
            "PC 2 ({}%)".format(
                "%.4f" % (self.__pca_.explained_variance_ratio_[1] * 100.0)
            )
        )

        return ax

    def visualize(
        self,
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        ax: Union[matplotlib.pyplot.Axes, None] = None,
        log: bool = True,
    ) -> matplotlib.pyplot.Axes:
        r"""
        Plot the :math:`\Chi^{2}` acceptance area with observations on distance plot.

        Parameters
        ----------
        X : array-like(str, ndim=2)
            Columns are features, observations are rows - will be converted to
            numpy array automatically.

        ax : matplotlib.pyplot.Axes, optional(default=None)
            Axes to plot on.

        log : bool, optional(default=True)
            Whether or not to transform the axes using a natural logarithm.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Axes results are plotted on.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:  # type: ignore[union-attr]
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        if ax is None:
            fig = plt.figure()
            axis = plt.gca()
        else:
            axis = ax

        h_, q_ = self._h_q(X)
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
            np.log(1.0 + h_lim / self.__h0_) if log else h_lim / self.__h0_,
            np.log(1.0 + q_lim / self.__q0_) if log else q_lim / self.__q0_,
            "g-",
        )
        axis.plot(
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

        ext_mask, out_mask = self.check_outliers(X)
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
            axis.plot(
                np.log(1.0 + h_[mask] / self.__h0_)
                if log
                else h_[mask] / self.__h0_,
                np.log(1.0 + q_[mask] / self.__q0_)
                if log
                else q_[mask] / self.__q0_,
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
        axis.legend(loc="upper right")
        axis.set_xlim(0, xlim)
        axis.set_ylim(0, ylim)
        axis.set_xlabel(r"${\rm ln(1 + h/h_0)}$" if log else r"${\rm h/h_0}$")
        axis.set_ylabel(r"${\rm ln(1 + q/q_0)}$" if log else r"${\rm q/q_0}$")

        return axis

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
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": ["check_fit2d_1sample"],  # Can't fit 1D
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
