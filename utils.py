"""
General utility functions.

author: nam
"""
import bokeh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, show
from matplotlib.collections import LineCollection
from sklearn.model_selection import learning_curve


def plot_learning_curve(
    model, X, y, cv=3, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0]
):
    """
    Plot the learning curve for a model.

    Parameters
    ----------
    X : matrix-like
        Columns of features; observations are rows - will be converted to
        numpy array automatically.
    y : array-like
        Response values.
    cv : int or sklearn.model_selection object
        Cross-validation strategy; uses k-fold CV if an integer is provided.
    train_sizes : array-like
        Fractions of provided data to use for training.

    Returns
    -------
    matplotlib.pyplot.Axes
        Axes the figure is plotted on.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        n_jobs=1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.plot(
        train_sizes,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training",
    )

    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )

    plt.plot(
        train_sizes,
        test_mean,
        color="green",
        linestyle="--",
        marker="s",
        markersize=5,
        label="Validation",
    )

    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )

    plt.grid()
    plt.xlabel("Number of training samples")
    plt.ylabel("TEFF")
    plt.legend(loc="best")
    plt.tight_layout()

    return fig.gca()


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
    x : array-like
        Wavelengths (channel) measured at.
    y : array-like
        Spectral (signal) intensities.
    importance_values : ndarray
        Importance value assigned to each feature.
    cmap : str
        Name of colormap to use (https://matplotlib.org/stable/gallery/color/colormap_reference.html).
    figsize : tuple
        Size of figure to plot.
    bounds : tuple
        Bounds to color based on; if unspecified uses min/max of importance_values.
    background : bool
        Whether or not to plot the uncolored (gray) spectrum behind the colored points.

    Returns
    -------
    axes : matplotlib.pyplot.Axes
        Axes the result is plotted on.
    """
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    importance_values = np.array(importance_values).ravel()
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


def bokeh_color_spectrum(x, y, importance_values, palette=Spectral10):
    """
    Color a spectrum based on feature importance values in Bokeh.

    Parameters
    ----------
    x : array-like
        Wavelengths (channel) measured at.
    y : array-like
        Spectral (signal) intensities.
    importance_values : ndarray
        Importance value assigned to each feature.
    palette : str
        Name of colormap to use (https://docs.bokeh.org/en/latest/docs/reference/palettes.html).
    """
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    importance_values = np.array(importance_values).ravel()

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


def estimate_dof(h_vals, q_vals, n_components, n_features_in):
    """
    Estimate the degrees of freedom for the chi-squared distribution.

    This follows from Ref. 1.

    [1] "Acceptance areas for multivariate classification derived by projection
    methods," Pomerantsev, Journal of Chemometrics 22 (2008) 601-609.
    """

    def err2(N, vals):
        """
        Use a "robust" method for estimating DoF.

        In [1] Eq. 14 suggests the IQR should be divided by the mean (h0),
        however, the citation they provide suggests the median might be
        a better choice; in practice, it seems that is favored since it
        is more robust against outliers, so this is used below in that
        spirit.
        """
        x0 = np.median(vals)  # np.mean(vals)
        a = (scipy.stats.chi2.ppf(0.75, N) - scipy.stats.chi2.ppf(0.25, N)) / N
        b = scipy.stats.iqr(vals, rng=(25, 75)) / x0

        return (a - b) ** 2

    # As in conclusions of [1], Nh ~ n_components is expected
    res = scipy.optimize.minimize(
        err2, n_components, args=(h_vals), method="Nelder-Mead"
    )
    if res.success:
        # Robust method, if possible
        Nh = res.x[0]
    else:
        # Use simple estimate if this fails (Eq. 13 in [1])
        Nh = 2.0 * np.mean(h_vals) ** 2 / np.std(h_vals, ddof=1) ** 2

    # As in conclusions of [1], Nq ~ rank(X)-n_components is expected;
    # assuming near full rank then this is min(I,J)-n_components
    # (n_components<=J)
    res = scipy.optimize.minimize(
        err2,
        np.min([len(q_vals), n_features_in]) - n_components,
        args=(q_vals),
        method="Nelder-Mead",
    )
    if res.success:
        # Robust method, if possible
        Nq = res.x[0]
    else:
        # Use simple estimate if this fails (Eq. 23 in [1])
        Nq = 2.0 * np.mean(q_vals) ** 2 / np.std(q_vals, ddof=1) ** 2

    return Nh, Nq
