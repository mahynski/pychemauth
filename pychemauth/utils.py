"""
General utility functions.

author: nam
"""
import copy
import scipy
import joblib
import sklearn
import pychemauth
import imblearn
import pickle
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, show

from matplotlib.collections import LineCollection

from sklearn.cross_decomposition import PLSRegression
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.utils.validation import check_array

from matplotlib.patches import Ellipse, Rectangle

from pathlib import Path
from tempfile import TemporaryDirectory
from huggingface_hub import hf_hub_download, HfApi, ModelCard, ModelCardData


class HuggingFace:
    """Tools to help store and load models on Hugging Face Hub."""

    @staticmethod
    def from_pretrained(
        model_id: str,
        filename="model.pkl",
        revision=None,
        token=None,
        library_version=None,
    ):
        """
        Load a pre-trained model from Hugging Face.

        Parameters
        ----------
        model_id : str
            Model ID, for example "hf-user/my-awesome-model"

        filename : str, optional(default="model.pkl")
            The name of the model file in the repo, e.g., "model.pkl". This is the default name
            used when pushing to Hugging Face hub (`push_to_hub`), but if you change it or
            use another repo with a different name, change it to match here.

        revision : str, optional(default=None)
            Model revision; if None, the latest version is retrieved.

        token : str, optional(default=None)
            Your Hugging Face access token. Refer to https://huggingface.co/settings/tokens.
            Ungated, public models do not require this to be specified.

        library_version : str, optional(default=None)
            The version of the PyChemAuth library to use; if None, the latest version is used.

        Returns
        -------
        model : sklearn.base.BaseEstimator
            Model, or pipeline, from PyChemAuth that is compatible with sklearn's estimator API.
        """
        filename = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            token=token,
            library_version=library_version,
            library_name="PyChemAuth",
        )
        return joblib.load(filename)

    @staticmethod
    def push_to_hub(
        model: sklearn.base.BaseEstimator,
        namespace: str,
        repo_name: str,
        token: str,
        revision=None,
        private=True,
    ) -> None:
        """
        Push a PyChemAuth model, or pipeline, to Hugging Face.

        If no repo (namespace/repo_name) exists on Hugging Face this creates a minimal model
        card and repo to hold a PyChemAuth model or pipeline. By default, all new repos are
        set to private.

        It is strongly recommended that you visit the link below for instructions on how to
        fill out an effective model card that accurately and completely describes your model.

        https://huggingface.co/docs/hub/model-card-annotated

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            Model, or pipeline, from PyChemAuth that is compatible with sklearn's estimator API.

        namespace : str
            User or organization name on Hugging Face.

        repo_name : str
            Name of Hugging Face repository, e.g., "my-awesome-model".

        token : str
            Your Hugging Face access token.  Be sure it has write access.
            Refer to https://huggingface.co/settings/tokens.

        revision : str, optional(default=None)
            The git revision to commit from. Defaults to the head of the `"main"` branch.

        private : bool, optional(default=True)
            If a new repo is created, this indicates if it should be private.

        Notes
        -----
        All models are serialized using pickle(protocol=4).
        """

        def _check_model_type(model):
            """Determine if the model is a regressor or classifier."""
            from pychemauth.classifier import osr, plsda, simca
            from pychemauth.manifold import elliptic

            _type = type(model)
            if (_type is sklearn.pipeline.Pipeline) or (
                _type is imblearn.pipeline.Pipeline
            ):
                _type = type(model.steps[-1][1])

            if _type in [
                pychemauth.classifier.osr.OpenSetClassifier,
                pychemauth.classifier.plsda.PLSDA,
                pychemauth.classifier.simca.SIMCA_Authenticator,
                pychemauth.classifier.simca.SIMCA_Model,
                pychemauth.classifier.simca.DDSIMCA_Model,
                pychemauth.manifold.elliptic.EllipticManifold_Authenticator,
                pychemauth.manifold.elliptic.EllipticManifold_Model,
            ]:
                # Tag as classifier
                return "tabular-classification"
            elif _type in [
                pychemauth.regressor.pcr.PCR,
                pychemauth.regressor.pls.PLS,
            ]:
                # Tag as regressor
                return "tabular-regression"
            else:
                # No tags - e.g., PCA.
                return "other"

        # Save all files in a temporary directory and push them in a single commit
        try:
            repo_id = f"{namespace}/{repo_name}"

            # Create repo
            api = HfApi()

            def _create_repo(exist_ok=False):
                return api.create_repo(
                    repo_id=repo_id,
                    token=token,
                    private=private,
                    repo_type="model",
                    exist_ok=exist_ok,
                )

            try:
                _ = _create_repo(exist_ok=False)
                _new_repo = True
            except:
                _ = _create_repo(exist_ok=True)
                _new_repo = False

            with TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Serialize the model
                with open(os.path.join(tmpdir, "model.pkl"), mode="bw") as f:
                    pickle.dump(model, file=f, protocol=4)

                # Create the model card for new repos only - otherwise this will overwrite
                if _new_repo:
                    card_data = ModelCardData(
                        library_name="PyChemAuth",
                        license="other",
                        license_name="nist",
                        license_link="https://github.com/mahynski/pychemauth/blob/main/LICENSE.md",
                        pipeline_tag=_check_model_type(model),
                        tags=["PyChemAuth"],
                    )
                    content = f"""
---
{ card_data.to_yaml() }
---

# Model Card

This is a default card created by PyChemAuth.

Refer to [this link](https://huggingface.co/docs/hub/model-card-annotated) for best practices on filling this out.
"""
                    card = ModelCard(content)
                    card.validate()
                    (tmpdir / "README.md").write_text(card.content)

                return api.upload_folder(
                    repo_id=repo_id,
                    folder_path=tmpdir,
                    token=token,
                    commit_message="Pushing model on {}".format(
                        datetime.datetime.now()
                    ),
                    revision=revision,
                    repo_type="model",
                )

        except Exception as e:
            raise Exception(
                "Unable to create temporary directory and save model information : {}".format(
                    e
                )
            )

        return


class ControlBoundary:
    """Base class for plotting statistical control boundaries."""

    def __init__(self):
        """Initialize class."""
        self.boundary_ = None

    def set_params(self, **parameters):
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        """Plot the control boundary."""
        raise NotImplementedError

    @property
    def boundary(self):
        """Return the boundary."""
        return copy.deepcopy(self.boundary_)


def _adjusted_covariance(X, method, center, dim):
    """Compute the covariance of data around a fixed center."""
    if center is None:
        # Not forcing the center, leave
        adjust = np.array([0.0 for i in range(dim)])
    else:
        adjust = check_array(
            center,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=False,
            force_all_finite=True,
            copy=True,
        )
    if adjust.shape != (dim,):
        raise Exception("Invalid center.")

    X = X[:, :dim] - adjust
    if method.lower() == "empirical":
        cov = EmpiricalCovariance(
            assume_centered=False if center is None else True
        ).fit(X)
    elif method.lower() == "mcd":
        cov = MinCovDet(
            assume_centered=False if center is None else True, random_state=42
        ).fit(X)
    else:
        raise ValueError("Unrecognized method for determining the covariance.")

    return cov.covariance_, cov.location_ + adjust


class CovarianceEllipse(ControlBoundary):
    """Draw chi-squared limits of a two dimensional distribution as an ellipse."""

    def __init__(self, method="empirical", center=None):
        """
        Instantiate the class.

        Parameters
        ----------
        method : str, optional(default='empirical')
            How to compute the covariance matrix.  The default 'empirical' uses the
            empirical covariance, if 'mcd' the minimum covariance determinant
            is computed.

        center : array_like(float, ndim=1), optional(default=None)
            Shifts the training data to make this the center.  If None, no shifting
            is done, and the data is not assumed to be centered when the ellipse is
            calculated.
        """
        super(CovarianceEllipse, self).__init__()
        self.set_params(**{"method": method, "center": center})

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {"method": self.method, "center": self.center}

    def fit(self, X):
        """
        Fit the covariance ellipse to the data.

        Only the first 2 dimensions, or columns, of the data will be used.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix with at least 2 features (columns).

        Returns
        -------
        self

        Raises
        ------
        Exception if X has less than 2 columns.
        ValueError if the covariance method is unrecognized.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        if X_.shape[1] < 2:
            raise Exception(
                "Can only draw 2D covariance ellipse if there are at least 2 features."
            )

        self.__S_, self.__class_center_ = _adjusted_covariance(
            X_, self.method, self.center, dim=2
        )

        evals, evecs = np.linalg.eig(self.__S_)
        ordered = sorted(zip(evals, evecs.T), key=lambda x: x[0], reverse=True)
        self.__l1_, self.__l2_ = ordered[0][0], ordered[1][0]
        largest_evec = ordered[0][1]
        self.__angle_ = (
            np.arctan2(largest_evec[1], largest_evec[0]) * 180.0 / np.pi
        )

        return self

    def visualize(self, ax, alpha=0.05, ellipse_kwargs={"alpha": 0.3}):
        """
        Draw a covariance ellipse boundary at a certain threshold.

        Parameters
        ----------
        ax : matplotlib.Axes.axes
            Axes object to plot the ellipse on.

        alpha : float, optional(default=0.05)
            Significance level (Type I error rate).

        ellipse_kwargs: dict, optional(default={'alpha':0.3})
            Dictionary of formatting arguments for the ellipse.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Ellipse.html.

        Returns
        -------
        ax : matplotlib.Axes.axes
            Axes object with ellipse plotted on it.
        """
        k = np.sqrt(
            -2 * np.log(alpha)
        )  # https://www.kalmanfilter.net/background2.html
        self.boundary_ = Ellipse(
            xy=self.__class_center_,
            width=np.sqrt(self.__l1_) * k * 2,
            height=np.sqrt(self.__l2_) * k * 2,
            angle=self.__angle_,
            **ellipse_kwargs,
        )
        ax.add_artist(self.boundary_)

        return ax


class OneDimLimits(ControlBoundary):
    """Draw chi-squared limits of a one dimensional distribution as a rectangle."""

    def __init__(self, method="empirical", center=None):
        """
        Instantiate the class.

        Parameters
        ----------
        method : str, optional(default='empirical')
            How to compute the covariance matrix.  The default 'empirical' uses the
            empirical covariance, if 'mcd' the minimum covariance determinant
            is computed.

        center : array_like(float, ndim=1), optional(default=None)
            Shifts the training data to make this the center.  If None, no shifting
            is done, and the data is not assumed to be centered when the ellipse is
            calculated.
        """
        super(OneDimLimits, self).__init__()
        self.set_params(**{"method": method, "center": center})

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {"method": self.method, "center": self.center}

    def fit(self, X):
        """
        Fit the covariance to the data.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix with a single feature (column).

        Returns
        -------
        self

        Raises
        ------
        Exception if X has more than 1 column.
        ValueError if the covariance method is unrecognized.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        if X_.shape[1] != 1:
            raise Exception(
                "Can only draw one dimensional boundary if there is a single feature."
            )

        self.__S_, self.__class_center_ = _adjusted_covariance(
            X_, self.method, self.center, dim=1
        )

        return self

    def visualize(
        self, ax, x, alpha=0.05, rectangle_kwargs={"alpha": 0.3}, vertical=True
    ):
        """
        Draw a covariance boundary as a rectangle at a certain threshold.

        Parameters
        ----------
        ax : matplotlib.Axes.axes
            Axes object to plot the ellipse on.

        x : float
            X coordinate to center the covariance "bar" on. If `vertical` is True, this is
            a y coordinate instead.

        alpha : float, optional(default=0.05)
            Significance level (Type I error rate).

        rectangle_kwargs: dict, optional(default={'alpha':0.3})
            Dictionary of formatting arguments for the rectangle_kwargs.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html.

        vertical : scalar(bool), optional(default=True)
            Whether or not to plot the boundary vertically (True) or horizontally (False).

        Returns
        -------
        ax : matplotlib.Axes.axes
            Axes object with rectangle plotted on it.
        """
        d_crit = scipy.stats.chi2.ppf(1.0 - alpha, 1)

        if vertical:
            self.boundary_ = Rectangle(
                xy=[
                    x,
                    self.__class_center_[0] - np.sqrt(d_crit * self.__S_[0][0]),
                ],
                width=0.6,
                height=2 * np.sqrt(d_crit * self.__S_[0][0]),
                **rectangle_kwargs,
            )
        else:
            dy = 2.0 / 3.0
            self.boundary_ = Rectangle(
                xy=[
                    self.__class_center_[0] - np.sqrt(d_crit * self.__S_[0][0]),
                    x - 0.5 * dy,
                ],
                width=2 * np.sqrt(d_crit * self.__S_[0][0]),
                height=dy,
                **rectangle_kwargs,
            )
        ax.add_artist(self.boundary_)

        return ax


def color_spectrum(
    x,
    y,
    importance_values,
    cmap="coolwarm",
    figsize=None,
    bounds=None,
    background=True,
):
    """
    Color a spectrum based on feature importance values.

    Parameters
    ----------
    x : array_like(float, ndim=1)
        Wavelengths (channel) measured at.

    y : array_like(float, ndim=1)
        Spectral (signal) intensities.

    importance_values : array_like(float, ndim=1)
        Importance value assigned to each feature.

    cmap : str, optional(default="coolwarm")
        Name of colormap to use (https://matplotlib.org/stable/gallery/color/colormap_reference.html).

    figsize : tuple(int, int), optional(default=None)
        Size of figure to plot.

    bounds : tuple(float, float), optional(default=None)
        Bounds to color based on; if unspecified uses min/max of importance_values.

    background : scalar(bool), optional(default=True)
        Whether or not to plot the uncolored (gray) spectrum behind the colored points.

    Returns
    -------
    axes : matplotlib.pyplot.axes
        Axes the result is plotted on.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    importance_values = np.asarray(importance_values, dtype=np.float64).ravel()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if background:
        axes.plot(x, y, color="k", alpha=0.10)

    if bounds is None:
        min_, max_ = importance_values.min(), importance_values.max()
    else:
        min_, max_ = bounds[0], bounds[1]

    norm = plt.Normalize(min_, max_)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(importance_values)

    line = axes.add_collection(lc)
    fig.colorbar(line, ax=axes)

    y_range = y.max() - y.min()
    axes.set_xlim(x.min(), x.max())
    axes.set_ylim(y.min() - 0.05 * y_range, y.max() + 0.05 * y_range)

    return axes


def bokeh_color_spectrum(
    x, y, importance_values, palette=Spectral10, y_axis_type=None
):
    """
    Color a spectrum based on feature importance values in Bokeh.

    Parameters
    ----------
    x : array_like(float, ndim=1)
        Wavelengths (channel) measured at.

    y : array_like(float, ndim=1)
        Spectral (signal) intensities.

    importance_values : array_like(float, ndim=1)
        Importance value assigned to each feature.

    palette : bokeh.palettes, optional(default=Spectral10)
        Color palette to use (https://docs.bokeh.org/en/latest/docs/reference/palettes.html).

    y_axis_type : str, optional(default=None)
        Optional transformation of y axis, e.g., y_axis_type="log".
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    importance_values = np.asarray(importance_values, dtype=np.float64).ravel()

    spectrum_df = pd.DataFrame(
        np.vstack((x, y, importance_values)).T,
        columns=("Channel", "Signal", "Importance"),
    )

    datasource = ColumnDataSource(spectrum_df)
    color_mapping = LinearColorMapper(
        low=spectrum_df["Importance"].min(),
        high=spectrum_df["Importance"].max(),
        palette=palette,
    )

    plot_figure = figure(
        title="Importance-Colored Signal",
        plot_width=900,
        plot_height=600,
        tools=("pan, wheel_zoom, reset"),
        x_axis_label="Channel",
        y_axis_label="Signal",
        y_axis_type=y_axis_type,
    )

    plot_figure.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div>
            <span style='font-size: 16px; color: #224499'>Channel:</span>
            <span style='font-size: 18px'>@Channel</span>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>Importance Value:</span>
            <span style='font-size: 18px'>@Importance</span>
        </div>
    </div>
    """
        )
    )

    plot_figure.line(
        "Channel",
        "Signal",
        source=datasource,
        color="black",
        line_width=1,
        line_alpha=0.25,
    )
    plot_figure.circle(
        "Channel",
        "Signal",
        source=datasource,
        color=dict(field="Importance", transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4,
    )
    show(plot_figure)


def estimate_dof(u_vals, robust=True, initial_guess=None):
    """
    Estimate the degrees of freedom for projection-based modeling.

    Parameters
    ----------
    u_vals : array_like(float, ndim=1)
        Observation values.

    robust : scalar(bool), optional(default=True)
        Whether to use a statistically robust approach or not.

    initial_guess : scalar(float or None), optional(default=None)
        Initial guess for the degrees of freedom.

    Returns
    -------
    Nu : scalar(int)
        Number of degrees of freedom.

    u0 : scalar(float)
        Associated scaling factor.

    References
    ----------
    [1] "Acceptance areas for multivariate classification derived by projection
    methods," Pomerantsev, Journal of Chemometrics 22 (2008) 601-609.

    [2] "Concept and role of extreme objects in PCA/SIMCA," Pomerantsev A.,
    Rodionova, O., Journal of Chemometrics 28 (2014) 429-438.

    [3] "Detection of outliers in projection-based modeling," Rodionova, O., and
    Pomerantsev, A., Anal. Chem. 92 (2020) 2656-2664.
    """
    if not robust:
        # Eq. 12 in [2]
        u0 = np.mean(u_vals)
        Nu = int(
            np.max([round(2.0 * u0**2 / np.std(u_vals, ddof=1) ** 2, 0), 1])
        )
    else:

        def err2(N, vals):
            # Use a "robust" method for estimating DoF - solve Eq. 14 in [2].
            if N < 1:
                N = 1
            a = (
                (scipy.stats.chi2.ppf(0.75, N) - scipy.stats.chi2.ppf(0.25, N))
                * np.median(vals)
                / scipy.stats.chi2.ppf(0.5, N)
            )
            b = scipy.stats.iqr(vals, rng=(25, 75))

            return (a - b) ** 2

        def approximate(vals):
            # Eq. 16 in [2]
            a = 0.72414
            b = 2.68631
            c = 0.84332
            M = np.median(vals)
            S = scipy.stats.iqr(vals, rng=(25, 75))

            arg = b * M / S
            if arg < 1:
                return 1
            else:
                return int(
                    round(np.exp(((1.0 / a) * np.log(arg)) ** (1.0 / c)), 0)
                )

        def averaged_estimator(N, vals):
            # Eq. 17 in [2]
            M = np.median(vals)
            S = scipy.stats.iqr(vals, rng=(25, 75))

            return (
                0.5
                * N
                * (
                    M / scipy.stats.chi2.ppf(0.5, N)
                    + S
                    / (
                        scipy.stats.chi2.ppf(0.75, N)
                        - scipy.stats.chi2.ppf(0.25, N)
                    )
                )
            )

        res = scipy.optimize.minimize(
            err2,
            (1 if initial_guess is None else initial_guess),
            args=(u_vals),
            method="Nelder-Mead",
        )
        if res.success:
            # Direct method, if possible
            Nu = int(np.max([round(res.x[0], 0), 1]))
        else:
            # Else, use analytical approximation
            Nu = approximate(u_vals)

        u0 = averaged_estimator(Nu, u_vals)

    return Nu, u0


def pos_def_mat(S, inner_max=10, outer_max=100):
    """
    Create a positive definite approximation of a square, symmetric matrix.

    Parameters
    ----------
    S : array_like(float, ndim=2)
        2D square, symmetric matrix to make positive definite.

    inner_max : scalar(int), optional(default=10)
        Number of iterations at a fixed tolerance to try.

    outer_max : scalar(int), optional(default=100)
        Number of different tolerances to try.

    Returns
    -------
    recon : ndarray(float, ndim=2)
        Symmetric, positive definite matrix approximation of S.
    """
    S = np.asarray(S, dtype=np.float64)
    assert S.shape[0] == S.shape[1]  # Check square
    assert np.allclose(S, (S + S.T) / 2.0)  # Check symmetric

    for j in range(outer_max):
        min_ = np.min(np.abs(S)) / 1000.0  # Drop down by 3 orders of magnitude
        max_ = np.min(np.abs(S)) * 10.0  # Within one order of magnitude of min
        tol = min_ + j * (max_ - min_) / float(outer_max)

        recon = copy.copy(S)

        # Compute evecs, evals, set all evals to tol, reconstruct
        for i in range(inner_max):
            evals, evecs = np.linalg.eig(recon)
            if np.any(np.abs(evals) < tol):
                evals[np.abs(evals) < tol] = tol
                recon = np.matmul(
                    evecs, np.matmul(np.diag(evals), np.linalg.inv(evecs))
                )
            else:
                break

        safe = True
        try:
            # Try even if inner loop reached its limit
            np.linalg.cholesky(recon)
        except np.linalg.LinAlgError:
            safe = False

        if np.max(np.abs(S - recon)) > tol:
            # If the maximum difference is more than the eigenvalue
            # tolerance, reject this.
            safe = False

        if safe:
            return recon

    raise Exception("Unable to create a symmetric, positive definite matrix")


def pls_vip(pls: PLSRegression, mode="weights"):
    """
    Compute the variable importance in projection (VIP) in a PLS(1) model.

    Parameters
    ----------
    pls : sklearn.cross_decomposition.PLSRegression
        Trained PLS model.

    mode : scalar(str), optional(default='weights')
        Whether to use the weights or the rotations to compute the VIP.

    Returns
    -------
    vip : ndarray(float, ndim=1)
        Variable importances.

    Note
    ----
    Often, both VIP and the PLS coefficients are used to remove features. [1]

    References
    ----------
    [1] Wold, S., Sjoestroem, M., & Eriksson, L. (2001). PLS-regression: a basic tool of
    chemometrics. Chemometrics and intelligent laboratory systems, 58(2), 109-130.

    [2] Chong, I.-G., Jun, C.-H. (2005). Performance of some variable selection methods
    when multicollinearity is present. Chemometrics and intelligent laboratory systems,
    78(1), 103-112.
    """
    t = pls.x_scores_
    q = pls.y_loadings_

    if mode == "weights":
        w = pls.x_weights_
    else:
        w = pls.x_rotations_
    w /= np.linalg.norm(w, axis=0)

    n, _ = w.shape
    s = np.diag(t.T @ t @ q.T @ q)

    return np.sqrt(n * (w**2 @ s) / np.sum(s))


def _logistic_proba(x):
    """
    Compute the logistic function of a given input.

    This is designed to work on margin space "distances" from classifiers
    or authenticators to predict probabilities. See scikit-learn convention:
    https://scikit-learn.org/stable/glossary.html#term-predict_proba

    Parameters
    ----------
    x : ndarray(float, ndim=1)
        Array of distances.

    Returns
    -------
    probabilities : ndarray(float, ndim=2)
        2D array as logistic function of the the input, x. First column
        is NOT inlier, 1-p(x), second column is inlier probability, p(x).
    """
    p_inlier = p_inlier = 1.0 / (
        1.0 + np.exp(-np.clip(x, a_max=None, a_min=-500))
    )
    prob = np.zeros((p_inlier.shape[0], 2), dtype=np.float64)
    prob[:, 1] = p_inlier
    prob[:, 0] = 1.0 - p_inlier

    return prob

def _multiclass_cm_metrics(df, Itot, trained_classes, use_classes, style, not_assigned, actual):
    """
    Compute metrics for a multiclass classifier / authenticator using the confusion matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Inputs (index) vs. predictions (columns); akin to a confusion matrix.

    Itot : pandas.Series
        Number of each class asked to classify.

    trained_classes : numpy.ndarray(str or int)
        Classes seen during training.
     
    use_classes : numpy.ndarray(str or int)
        Classes to use when computing metrics; this includes all classes seen during testing excluding the "unknown" class.
    
    style : str
        Either "hard" or "soft' denoting whether a point can be assigned to one or multiple classes, respectively.
     
    not_assigned : str or int
        The designation for an "unknown" or unrecognized class.
     
    actual : numpy.ndarray(str or int)
        True target (y) values.

    Returns
    -------
    fom : dict
        Dictionary object with the following attributes.

        CM : pandas.DataFrame
            Inputs (index) vs. predictions (columns); akin to a confusion matrix.

        I : pandas.Series
            Number of each class asked to classify.  This is the same as the `Itot` input.

        CSNS : pandas.Series
            Class sensitivity.

        CSPS : pandas.Series
            Class specificity.

        CEFF : pandas.Series
            Class efficiency.

        TSNS : scalar(float)
            Total sensitivity.

        TSPS : scalar(float)
            Total specificity.

        TEFF : scalar(float)
            Total efficiency.

        ACC : scalar(float)
            Accuracy.

    Note
    ----
    When making predictions about extraneous classes (not in training set)
    class efficiency (CEFF) is given as simply class specificity (CSPS)
    since class sensitivity (CSNS) cannot be calculated.

    References
    ----------
    [1] "Multiclass partial least squares discriminant analysis: Taking the
    right way - A critical tutorial," Pomerantsev and Rodionova, Journal of
    Chemometrics (2018). https://doi.org/10.1002/cem.3030.
    """

    # Define accuracy so it works in hard or soft cases
    correct_ = 0.0
    for class_ in df.index:  # All input classes
        if (
            class_ in trained_classes
        ):  # Things to classifier knows about (TP)
            correct_ += df[class_][class_]
        else:
            # Consider an assignment as "unknown" a correct assignment (TN)
            correct_ += df[not_assigned][class_]
    ACC = correct_ / df.sum().sum() # This normalization guarantees 0 <= ACC <= 1

    # Class-wise FoM
    # Sensitivity is "true positive" rate and is only defined for trained/known classes
    CSNS = pd.Series(
        [
            df[kk][kk] / Itot[kk] if Itot[kk] > 0 else np.nan
            for kk in trained_classes
        ],
        index=trained_classes,
    )

    # Specificity is the fraction of points that are NOT a given class that
    # are correctly predicted to be something besides the class. Thus,
    # specificity can only be computed for the columns that correspond to
    # known classes since we have only trained on them. These are "true
    # negatives". This is always >= 0.
    CSPS = pd.Series(
        [
            1.0
            - np.sum(df[kk][df.index != kk])  # Column sum
            / np.sum(Itot[Itot.index != kk])
            for kk in trained_classes
        ],
        index=trained_classes,
    )

    # If CSNS can't be calculated, using CSPS as efficiency;
    # Oliveri & Downey introduced this "efficiency" used in [1]
    CEFF = pd.Series(
        [
            np.sqrt(CSNS[c] * CSPS[c]) if not np.isnan(CSNS[c]) else CSPS[c]
            for c in trained_classes
        ],
        index=trained_classes,
    )

    # Total FoM

    # Evaluates overall ability to recognize a class is itself.  If you
    # show the model some class it hasn't trained on, it can't be predicted
    # so no contribution to the diagonal.  We will normalize by total
    # number of points shown [1].  If some classes being tested were seen in
    # training they contribute, otherwise TSNS goes down for a class never
    # seen before.  This might seem unfair, but TSNS only makes sense if
    # (1) you are examining what you have trained on or (2) you are
    # examining extraneous objects so you don't calculate this at all.
    TSNS = np.sum([df[kk][kk] for kk in trained_classes]) / np.sum(Itot)

    # If any untrained class is correctly predicted to be "NOT_ASSIGNED" it
    # won't contribute to df[use_classes].sum().sum().  Also, unseen
    # classes can't be assigned to so the diagonal components for those
    # entries is also 0 (df[k][k]).
    TSPS = 1.0 - (
        df[use_classes].sum().sum()
        - np.sum([df[kk][kk] for kk in use_classes])
    ) / np.sum(Itot) / (
        1.0 if style.lower() == "hard" else len(trained_classes) - 1.0
    )
    # Soft models can assign a point to all categories which would make this
    # sum > 1, meaning TSPS < 0 would be possible.  By scaling by the total
    # number of classes, TSPS is always positive; TSPS = 0 means all points
    # assigned to all classes (trivial result) vs. TSPS = 1 means no mistakes.

    # Sometimes TEFF is reported as TSPS when TSNS cannot be evaluated (all
    # previously unseen samples).
    TEFF = np.sqrt(TSPS * TSNS)

    return dict(
        zip(
            ["CM", "I", "CSNS", "CSPS", "CEFF", "TSNS", "TSPS", "TEFF", "ACC"],
            (
                df[
                    [c for c in df.columns if c in trained_classes]
                    + [not_assigned]
                ][
                    [x in np.unique(actual) for x in df.index]
                ],  # Re-order for easy visualization
                Itot,
                CSNS,
                CSPS,
                CEFF,
                TSNS,
                TSPS,
                TEFF,
                ACC,
            ),
        )
    )

def _occ_cm_metrics(df, Itot, target_class, trained_classes, not_assigned, actual):
    """
    Compute one-class classifier metrics from the confusion matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Inputs (index) vs. predictions (columns); akin to a confusion matrix.

    Itot : pandas.Series
        Number of each class asked to classify.

    trained_classes : numpy.ndarray(str or int)
        Classes seen during training.
     
    not_assigned : str or int
        The designation for an "unknown" or unrecognized class.
     
    actual : numpy.ndarray(str or int)
        True target (y) values.

    Returns
    -------
    fom : dict
        Dictionary object with the following attributes.

        CM : pandas.DataFrame
            Inputs (index) vs. predictions (columns); akin to a confusion matrix.

        I : pandas.Series
            Number of each class asked to classify.  This is the same as the `Itot` input.

        CSPS : pandas.Series
            Class specificity.

        TSNS : scalar(float)
            Total sensitivity.  For OCC this is also the CSNS.

        TSPS : scalar(float)
            Total specificity.

        TEFF : scalar(float)
            Total efficiency.

        ACC : scalar(float)
            Accuracy.
    """

    alternatives = [class_ for class_ in df.index if class_ != target_class]

    correct_ = df[target_class][target_class]  # (TP)
    for class_ in alternatives:  # All "negative" classes
        # Number of times an observation NOT from target_class was correctly not assigned to target_class
        # Assigning to multiple alternatives does not influence this in the spirit of OCC
        correct_ += Itot[class_] - df[target_class][class_]  # (TN)
    ACC = correct_ / float(Itot.sum())

    CSPS = {}
    for class_ in alternatives:
        if np.sum(Itot[class_]) > 0:
            CSPS[class_] = 1.0 - df[class_][target_class] / np.sum(
                Itot[class_]
            )
        else:
            CSPS[class_] = np.nan

    if np.all(actual == target_class):
        # Testing on nothing but the target class, can't evaluate TSPS
        TSPS = np.nan
    else:
        TSPS = 1.0 - (
            df[target_class].sum()
            - df[target_class][target_class]
        ) / (Itot.sum() - Itot[target_class])

    # TSNS = CSNS
    if target_class not in set(actual):
        # Testing on nothing but alternative classes, can't evaluate TSNS
        TSNS = np.nan
    else:
        TSNS = (
            df[target_class][target_class]
            / Itot[target_class]
        )

    if np.isnan(TSNS):
        TEFF = TSPS
    elif np.isnan(TSPS):
        TEFF = TSNS
    else:
        TEFF = np.sqrt(TSNS * TSPS)

    fom = dict(
        zip(
            ["CM", "I", "CSPS", "TSNS", "TSPS", "TEFF", "ACC"],
            (
                df[
                    [c for c in df.columns if c in trained_classes]
                    + [not_assigned]
                ][
                    [x in np.unique(actual) for x in df.index]
                ],  # Re-order for easy visualization
                Itot,
                CSPS,
                TSNS,
                TSPS,
                TEFF,
                ACC,
            ),
        )
    )

    return fom

def _occ_metrics(X, y, target_class, predict_function):
    """
    Compute one-class classifier (OCC) metrics directly from data.

    Parameters
    ----------
    X : array_like(float, ndim=2)
        Input feature matrix.

    y : array_like(str or int, ndim=1)
        Class labels or indices.

    target_class : str or int
        Target class being modeled by the OCC; should have the same type as `y`.
        
    predict_function : callable
        Should return a 1D numpy array of booleans corresponding to whether a point
        is an inlier.

    Returns
    -------
    fom : dict
        Dictionary object with the following attributes.

        CSPS : pandas.Series
            Class specificity.

        TSNS : scalar(float)
            Total sensitivity.  For OCC this is also the CSNS.

        TSPS : scalar(float)
            Total specificity.

        TEFF : scalar(float)
            Total efficiency.

        ACC : scalar(float)
            Accuracy.
    
    alternatives : list(str or int)
        Classes besides the target class present in `y`.
    """
    alternatives = [c for c in sorted(np.unique(y)) if c != target_class]

    CSPS = {}
    for class_ in alternatives:
        mask = y == class_
        CSPS[class_] = 1.0 - np.sum(predict_function(X[mask])) / np.sum(mask)

    mask = y != target_class
    if np.sum(mask) == 0:
        # Testing on nothing but the target class, can't evaluate TSPS
        TSPS = np.nan
    else:
        TSPS = 1.0 - np.sum(predict_function(X[mask])) / np.sum(mask)

    mask = y == target_class
    if np.sum(mask) == 0:
        # Testing on nothing but alternative classes, can't evaluate TSNS
        TSNS = np.nan
    else:
        TSNS = np.sum(predict_function(X[mask])) / np.sum(mask)  # TSNS = CSNS

    if np.isnan(TSNS):
        TEFF = TSPS
    elif np.isnan(TSPS):
        TEFF = TSNS
    else:
        TEFF = np.sqrt(TSNS * TSPS)

    # Compute accuracy
    y_in = y == target_class
    ACC = np.sum(predict_function(X) == y_in) / X.shape[0]

    return {"CSPS": CSPS, "TSNS": TSNS, "TSPS": TSPS, "TEFF": TEFF, "ACC": ACC}, alternatives