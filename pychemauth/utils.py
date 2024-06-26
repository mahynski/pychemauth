"""
General utility functions.

author: nam
"""
import copy
import bokeh
import scipy

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

class ControlBoundary:
    """
    Base class for plotting statistical control boundaries.
    """
    def __init__(self):
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

    X = X[:,:dim] - adjust
    if method.lower() == 'empirical':
        cov = EmpiricalCovariance(assume_centered=False if center is None else True).fit(X)
    elif method.lower() == 'mcd':
        cov = MinCovDet(assume_centered=False if center is None else True, random_state=42).fit(X)
    else:
        raise ValueError("Unrecognized method for determining the covariance.")
        
    return cov.covariance_, cov.location_ + adjust

class CovarianceEllipse(ControlBoundary):
    """
    Draw chi-squared limits of a two dimensional distribution as an ellipse.
    """
    def __init__(self, method='empirical', center=None):
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
        self.set_params(
            **{
                "method": method,
                "center": center
            }
        )

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "method": self.method,
            "center": self.center
        }

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
            raise Exception("Can only draw 2D covariance ellipse if there are at least 2 features.")

        self.__S_, self.__class_center_ = _adjusted_covariance(X_, self.method, self.center, dim=2)

        evals, evecs = np.linalg.eig(self.__S_)
        ordered = sorted(zip(evals, evecs.T), key=lambda x:x[0], reverse=True)
        self.__l1_, self.__l2_ = ordered[0][0], ordered[1][0]
        largest_evec = ordered[0][1]
        self.__angle_ = np.arctan2(largest_evec[1], largest_evec[0])*180.0/np.pi

        return self

    def visualize(self, ax, alpha=0.05, ellipse_kwargs={'alpha':0.3}):
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
        k = np.sqrt(-2*np.log(alpha)) # https://www.kalmanfilter.net/background2.html
        self.boundary_ = Ellipse(
            xy=self.__class_center_, 
            width=np.sqrt(self.__l1_)*k*2, 
            height=np.sqrt(self.__l2_)*k*2, 
            angle=self.__angle_,
            **ellipse_kwargs
        )
        ax.add_artist(self.boundary_)

        return ax
        
class OneDimLimits(ControlBoundary):
    """
    Draw chi-squared limits of a one dimensional distribution as a rectangle.
    """
    def __init__(self, method='empirical', center=None):
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
        self.set_params(
            **{
                "method": method,
                "center": center
            }
        )

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "method": self.method,
            "center": self.center
        }

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
            raise Exception("Can only draw one dimensional boundary if there is a single feature.")

        self.__S_, self.__class_center_ = _adjusted_covariance(X_, self.method, self.center, dim=1)

        return self

    def visualize(self, ax, x, alpha=0.05, rectangle_kwargs={'alpha':0.3}, vertical=True):
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
                xy=[x, self.__class_center_[0] - np.sqrt(d_crit*self.__S_[0][0])], 
                width=0.6, 
                height=2*np.sqrt(d_crit*self.__S_[0][0]), 
                **rectangle_kwargs
            )
        else:
            dy = 2.0/3.0
            self.boundary_ = Rectangle(
                xy=[self.__class_center_[0] - np.sqrt(d_crit*self.__S_[0][0]), x-0.5*dy], 
                width=2*np.sqrt(d_crit*self.__S_[0][0]), 
                height=dy, 
                **rectangle_kwargs
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

def pls_vip(pls: PLSRegression, mode='weights'):
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

    if mode == 'weights':
      w = pls.x_weights_
    else:
      w = pls.x_rotations_
    w /= np.linalg.norm(w, axis=0)

    n, _ = w.shape
    s = np.diag(t.T @ t @ q.T @ q)

    return np.sqrt(n*(w**2 @ s)/ np.sum(s))

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
        1.0
        + np.exp(
            -np.clip(x, a_max=None, a_min=-500)
        )
    )
    prob = np.zeros((p_inlier.shape[0], 2), dtype=np.float64)
    prob[:, 1] = p_inlier
    prob[:, 0] = 1.0 - p_inlier
        
    return prob