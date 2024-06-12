"""Some basic functions for generating data for examples."""
import copy

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def generate_data(n_samples=500, cov=[[3, 3], [3, 4]], seed=0, mean=[0, 0]):
    """Create a randomized 2D dataset that is already centered."""
    rng = np.random.RandomState(seed)
    X = rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)

    return X


def generate_response(
    X, dimension, mean=[0, 0], seed=1, y_center=0, display=True
):
    """Create response variables on the x coordinates."""
    X_pca = copy.copy(X) - np.array(
        mean
    )  # Do mean shift according to what was specified

    # Generate dummy response on a chosen PC
    pca = PCA(n_components=2).fit(X_pca)
    rng = np.random.RandomState(seed)
    y = (
        X_pca.dot(pca.components_[dimension])
        + mean[dimension]
        + rng.normal(size=X.shape[0]) / 2
        + y_center
    )

    if display:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        cmap = plt.get_cmap("viridis")
        rgba = cmap((y - np.min(y)) / (np.max(y) - np.min(y)))

        #     axes[0].scatter(X_pca.dot(pca.components_[0])+mean[0], y, c=rgba)
        axes[0].scatter(pca.transform(X_pca)[:, 0] + mean[0], y, c=rgba)

        axes[0].set(xlabel="Projected data onto PC 1", ylabel="y")
        #     axes[1].scatter(X_pca.dot(pca.components_[1])+mean[1], y, c=rgba)
        axes[1].scatter(pca.transform(X_pca)[:, 1] + mean[1], y, c=rgba)

        axes[1].set(xlabel="Projected data onto PC 2", ylabel="y")
        axes[2].scatter(X[:, 0], X[:, 1], c=rgba, alpha=0.3)
        axes[2].set_xlabel("Feature 1")
        axes[2].set_ylabel("Feature 2")

        axes[int(dimension)].set_title("Strong correlation")
        axes[int(not dimension)].set_title("No correlation")
        plt.tight_layout()

    return y, pca


def visualize_data(X, mean=[0, 0]):
    """Visualize the synthetic data."""
    X_pca = copy.copy(X) - np.array(
        mean
    )  # Do mean shift only for visualization purposes
    pca = PCA(n_components=2).fit(X_pca)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    axes[0].scatter(X[:, 0], X[:, 1], alpha=0.3, label="Samples")
    for i, (comp, var) in enumerate(
        zip(pca.components_, pca.explained_variance_)
    ):
        comp = comp * np.sqrt(var)  # scale component to get loading
        axes[0].plot(
            [mean[0], comp[0] + mean[0]],
            [mean[1], comp[1] + mean[1]],
            label=f"Principal Component {i+1} (Loaded)",
            linewidth=3,
            color=f"C{i + 2}",
        )
    axes[0].set(aspect="equal", xlabel="Feature 1", ylabel="Feature 2")
    _ = axes[0].legend()

    # Since we have unit eigenvectors, we can easily project
    axes[1].scatter(X[:, 0], X[:, 1], alpha=0.3, label="Raw Data")
    projection_0 = np.dot(X_pca, pca.components_[0]).reshape(
        -1, 1
    ) * pca.components_[0] + np.array(mean)
    projection_1 = np.dot(X_pca, pca.components_[1]).reshape(
        -1, 1
    ) * pca.components_[1] + np.array(mean)
    axes[1].plot(
        projection_0[:, 0],
        projection_0[:, 1],
        "g*",
        alpha=0.1,
        label="Projection to PC 1",
    )
    axes[1].plot(
        projection_1[:, 0],
        projection_1[:, 1],
        "r*",
        alpha=0.1,
        label="Projection to PC 2",
    )
    axes[1].set(aspect="equal", xlabel="Feature 1", ylabel="Feature 2")
    _ = axes[1].legend()

    # Visualize the projection
    viz_pt = 20  # I chose point 20 randomly
    axes[1].plot(X[viz_pt][0], X[viz_pt][1], "k*")
    axes[1].plot(projection_0[viz_pt][0], projection_0[viz_pt][1], "k*")
    axes[1].plot(projection_1[viz_pt][0], projection_1[viz_pt][1], "k*")
    axes[1].arrow(
        x=X[viz_pt][0],
        dx=projection_0[viz_pt][0] - X[viz_pt][0],
        y=X[viz_pt][1],
        dy=projection_0[viz_pt][1] - X[viz_pt][1],
        head_width=0.3,
        length_includes_head=True,
        color="g",
    )
    axes[1].arrow(
        x=X[viz_pt][0],
        dx=projection_1[viz_pt][0] - X[viz_pt][0],
        y=X[viz_pt][1],
        dy=projection_1[viz_pt][1] - X[viz_pt][1],
        head_width=0.3,
        length_includes_head=True,
        color="r",
    )


def add_line(coeffs):
    """Put a line on the figure."""
    intercept, slope = coeffs
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "k--", label="Linear regression")
