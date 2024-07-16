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

import numpy as np

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
