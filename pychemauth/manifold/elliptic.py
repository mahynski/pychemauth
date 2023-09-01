"""
Non-linear manifold-based dimensionality reduction methods classified with an elliptic boundary.

author: nam
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class EllipticManifold(ClassifierMixin, BaseEstimator):
    r"""
    Perform a dimensionality reduction with decision boundary determined by an ellipse.

    Parameters
    ----------
    alpha : scalar(float)
        Type I error rate (signficance level).

    dr_model : object
        Dimensionality reduction model, such as PCA; must support fit() and transform().

    kwargs : dict
        Keyword arguments for model; EllipticManifold.model = model(**kwargs). Must
        contain the `ndims` keyword.

    ndims : str, optional(default="n_components")
        Keyword in kwargs that corresponds to the dimensionality of the final space.

    robust : scalar(bool), optional(default=True)
        Whether or not use a robust estimate of the covariance matrix [2,3] to compute the
        Mahalanobis distances.

    center : str, optional(default="score")
        If "score", center the ellipse in the score space; otherwise go from transformation
        of the mean in the original data space.

    Note
    ----
    This process can be summarized as follows.  For each individual class:

    Step 0: Outlier removal.  In principle, it may be optimal to remove outliers before any analysis
    so that they do not impact the development of the manifold / dimensionality reduction.  For high
    dimensional data, an isolation forest is often very efficient.  If the dimensionality reduction
    step is robust against outliers (e.g., ROBPCA) then this might be not be necessary; however, in
    general, consider adding this to the pipeline before training since this is not handled 
    internally by this model.

    Step 1: Perform a dimensionality reduction (DR) step using some model to obtain
    a projection ("scores" or "embedding").

    Step 2: Following [1], assume each class forms a normally distributed subset
    with the known means (*see Note below) in the score/embedding space.  The covariance matrix is used to
    compute the `Mahalanobis distance <https://en.wikipedia.org/wiki/Mahalanobis_distance>`_.
    to its class center.

    Step 3: Assuming that these distances follow the :math:`\Chi{^2}` distribution, a
    soft discrimination rule can be constructed: sample i belongs to class k if the
    Mahalanobis distance is less than a threshold:

    :math:`$c_{crit} = \Chi^{-2}(1 - \alpha; d)$`

    where d is the dimensionality of the score space.

    This is similar to using scikit-learn's EllipticEnvelope (EE) after some dimensionality
    reduction step; EE essentially learns an ellipse around a known class, with
    some predefined amount of the observations set to be "included."  The basic
    difference is that a statistical confidence limit (:math:`\alpha`) can be directly
    specified here. EE's "contamination" is approximately :math:`\alpha` (type I error
    rate); however, this seems to lack some rigor. Here the :math:`\Chi{^2}`
    distribution has some degrees of freedom associated with based on the
    dimensionality of the space, whereas EE just takes the fraction as a hyperparameter
    with no context. As a result, many of the class members and functions follow a
    similar API.

    However, Mahalanobis distances are known to suffer from "masking" (multiple
    outliers skew the training so no individual seems "too bad"). EE uses the
    robust algorithm from [2,3] which aims to prevent this.

    scikit-learn has an excellent demonstration of this here:
    https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_\
    distances.html?highlight=outlier%20detection

    As a result, you can select either the conventional empirical method or the
    robust method to compute the Mahalanobis distances.

    The class center can be computed in 2 ways.  First, the mean of the
    X (and y) data can be taken and projected.  This is common in other approaches such
    as in [1] where there is justification to enforce that the class SHOULD have
    a mean at a fixed location (i.e., in a projection of a one-hot-encoding).
    Alternatively, the empirical mean in the score space/embedding can be
    taken.  We perform the latter by default; this operates more closely to a
    standard scikit-learn pipeline where a DR step occurs, then an EE boundary is
    fit to the embedding, but contrasts with some other published approaches.

    Data standardization/autoscaling is generally recommended since manifolds
    are determined based on distances (metrics may vary) between points in the
    training data, so this should be meaningful or at least "fair."

    References
    ----------
    [1] "Multiclass partial least squares discriminant analysis: Taking the
    right way - A critical tutorial," Pomerantsev and Rodionova, Journal of
    Chemometrics (2018). https://doi.org/10.1002/cem.3030.

    [2] "A fast algorithm for the minimum covariance determinant estimator,"
    Rousseeuw, Peter J., and Katrien Van Driessen, Technometrics 41 (1999)
    212-223. https://www.tandfonline.com/doi/abs/10.1080/00401706.1999.10485670

    [3] "Least median of squares regression," P. J. Rousseeuw., J. Am Stat Ass.,
    79 (1984).

    [4] "Concept and role of extreme objects in PCA/SIMCA," Pomerantsev, A. and
    Rodionova, O., Journal of Chemometrics 28 (2014) 429-438.
    """

    def __init__(
        self,
        alpha,
        dr_model,
        kwargs,
        ndims="n_components",
        robust=True,
        center="score",
    ):
        """Instantiate the class."""
        self.set_params(
            **{
                "alpha": alpha,
                "dr_model": dr_model,
                "kwargs": kwargs,
                "ndims": ndims,
                "robust": robust,
                "center": center,
            }
        )
        self.is_fitted_ = False

    def set_params(self, **parameters):
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "alpha": self.alpha,
            "dr_model": self.dr_model,
            "kwargs": self.kwargs,
            "ndims": self.ndims,
            "robust": self.robust,
            "center": self.center,
        }

    def _column_y(self, y):
        """Convert y to column format."""
        y = np.array(y)
        if y.ndim != 2:
            y = y[:, np.newaxis]

        return y

    def _sanity(self, X, y, init=False):
        """Check data format and sanity."""
        if init:
            self.n_features_in_ = X.shape[1]
        else:
            assert X.shape[1] == self.n_features_in_, "Incorrect X matrix shape"

        if y is None:
            X = check_array(X, accept_sparse=False, copy=True)
        else:
            X, y = check_X_y(X, y, accept_sparse=False, copy=True)
            y = self._column_y(y)

        return X, y

    def fit(self, X, y=None):
        """
        Fit the dimensionality reduction model.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array_like(float, ndim=1), optional(default=None)
            Response. Ignored if it is not used by :py:func:`dr_model.fit` (unsupervised methods).
            If passed, it is checked that they are all identical and this 
            label is used; otherwise the name "Training Class" is assigned.

        Returns
        -------
        self : EllipticManifold
            Fitted model.

        Note
        ----
        Only examples of a single, known class should be use to fit the model.
        """
        # Sanity checks
        X, y = self._sanity(np.asarray(X, dtype=np.float64), y, init=True)

        if y is None:
            self.__label_ = "Training Class"
        else:
            label = np.unique(y)
            if len(label) > 1:
                raise Exception("More than one class passed during training.")
            else:
                self.__label_ = str(label[0])
                
        # Fit the model
        self.model_ = self.dr_model(**self.kwargs)
        self.model_.fit(X, y)
        self.is_fitted_ = True

        # Compute (squared) Mahalanobis critical distance
        if not (self.ndims in self.kwargs):
            raise ValueError(
                "Cannot determined score space dimensionality, {} not in kwargs.".format(
                    self.ndims
                )
            )
        self.__d_crit_ = scipy.stats.chi2.ppf(
            1.0 - self.alpha, self.kwargs[self.ndims]
        )

        # Compute scatter matrix
        t_train = self.transform(X)
        if self.center != "score":
            # Put center of ellipse on projection of the empiricial data mean
            # in the original space
            self.__class_center_ = self.transform(
                X=np.mean(X, axis=0).reshape(1, -1)
            )[0]
        else:
            # Center ellipse in the score space
            self.__class_center_ = np.mean(t_train, axis=0)
        t = t_train - self.__class_center_

        # See https://scikit-learn.org/stable/auto_examples/\
        # covariance/plot_mahalanobis_distances.html?highlight=outlier%20detection
        # Do NOT perform additional centering to respect the decision above.
        if self.robust:
            cov = MinCovDet(assume_centered=True).fit(t)
        else:
            cov = EmpiricalCovariance(assume_centered=True).fit(t)
        self.__S_ = cov.covariance_

        return self

    def transform(self, X):
        """
        Perform dimensionality reduction on X into the score space.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        scores : ndarray(float, ndim=2)
            Coordinates of X in lower dimensional space.
        """
        check_is_fitted(self, "is_fitted_")
        X, _ = self._sanity(np.asarray(X, dtype=np.float64), y=None)

        return self.model_.transform(X)

    def fit_transform(self, X, y=None):
        """
        Fit the dimensionality reduction model and reduce X into the score space.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array_like(float, ndim=1), optional(default=None)
            Response. Ignored if it is not used by :py:func:`dr_model.fit` (unsupervised methods).

        Returns
        -------
        scores : ndarray(float, ndim=2)
            Coordinates of X in lower dimensional space.
        """
        _ = self.fit(X, y)

        return self.transform(X)

    def mahalanobis(self, X):
        """
        Compute the Mahalanobis distance for each sample.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        distances : ndarray(float, ndim=1)
            Mahalanobis distance for each sample.
        """
        check_is_fitted(self, "is_fitted_")
        t_test = self.transform(X)
        S_inv = np.linalg.inv(self.__S_)

        # Compute distances
        distances2 = np.diag(
            np.matmul(
                np.matmul(
                    (t_test - self.__class_center_),
                    S_inv,
                ),
                (t_test - self.__class_center_).T,
            )
        )
        assert np.all(distances2 >= 0), "All distances must be >= 0"

        return np.sqrt(distances2)

    def predict(self, X):
        """
        Predict if points are inliers (+1) or not (0).

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        results : ndarray(int, ndim=1)
            Array of +1 inliers else 0.
        """
        d = self.mahalanobis(X)
        mask = d > np.sqrt(self.__d_crit_)

        # +1/-1 follows scikit-learn's EllipticEnvelope API - this is different
        # to be more consistent with other APIs such as how SIMCA works.
        results = np.ones(len(X))  # Inliers
        results[mask] = 0  # Outliers

        return results

    def fit_predict(self, X, y=None):
        """
        Fit the dimensionality reduction model and then predict inliers.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array_like(float, ndim=1), optional(default=None)
            Response. Ignored if it is not used by :py:func:`dr_model.fit` (unsupervised methods).

        Returns
        -------
        results : ndarray(int, ndim=1)
            Array of +1 inliers else 0.
        """
        _ = self.fit(X, y)

        return self.predict(X)

    def score_samples(self, X):
        """
        Score observations.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        scores : ndarray(float, ndim=1)
            Negative Mahalanobis distance for each sample.

        Note
        ----
        Following scikit-learn's EllipticEnvelope, this returns the negative Mahalanobis
        distance.
        """
        return -self.mahalanobis(X)

    def decision_function(self, X):
        """
        Compute the decision function for each sample.
        
        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        decision_function : ndarray(float, ndim=1)
            Shifted, negative Mahalanobis distance for each sample.

        Note
        ----
        Following scikit-learn's EllipticEnvelope, this returns the negative Mahalanobis
        distance shifted by the cutoff distance, so f < 0 implies an outlier
        while f > 0 implies an inlier.

        References
        ----------
        See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-decision_function
        """
        return self.score_samples(X) - (-np.sqrt(self.__d_crit_))

    def predict_proba(self, X):
        """
        Predict the probability that observations are inliers.
        
        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        probability : ndarray(float, ndim=2)
            2D array as sigmoid function of the decision_function(). First column
            is for inliers, p(x), second columns is NOT an inlier, 1-p(x).

        Note
        ----
        Computes the sigmoid(decision_function(X, y)) as the
        transformation of the decision function.  This function is > 0
        for inliers so predict_proba(X, y) > 0.5 means inlier, < 0.5 means
        outlier.

        References
        ----------
        See SHAP documentation for a discussion on the utility and impact
        of "squashing functions": https://shap.readthedocs.io/en/latest/\
        example_notebooks/tabular_examples/model_agnostic/Squashing%20Effect.html\
        #Probability-space-explaination

        See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-predict_proba
        """
        p_inlier = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        prob = np.zeros((p_inlier.shape[0], 2), dtype=np.float64)
        prob[:, 0] = p_inlier
        prob[:, 1] = 1.0 - p_inlier

        return prob

    def loss(self, X, y, eps=1.0e-15):
        r"""
        Compute the negative log-loss, or logistic/cross-entropy loss.
        
        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array_like(int, ndim=1)
            Correct labels; +1 for inlier, 0 otherwise.
           
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
        assert len(X) == len(y)
        assert np.all(
            [a in [0, 1] for a in y]
        ), "y should contain only 0 or 1 labels"

        # Inlier = +1 (positive class), p[:,0]
        # Not inlier = 0 (negative class), p[:,1]
        prob = self.predict_proba(X)

        y_in = np.array([1.0 if y_ == +1 else 0.0 for y_ in y])
        p_in = np.clip(prob[:, 0], a_min=eps, a_max=1.0 - eps)

        # Return the negative, normalized log-loss
        return -np.sum(
            y_in * np.log(p_in) + (1.0 - y_in) * np.log(1.0 - p_in)
        ) / len(X)

    def extremes_plot(self, X, upper_frac=0.25, ax=None):
        r"""
        Plot an "extremes plot" [4] to evaluate the quality of the model.
        
        Parameters
        ----------
        X : array_like(float, ndim=2)
            Data to evaluate the number of outliers + extremes in.

        upper_frac : scalar(float), optional(default=0.25)
            Count the number of extremes and outliers for alpha values corresponding
            to :math:`n_{\rm exp}` = [1, X.shape[0]*upper_frac], where :math:`\alpha = n_{\rm exp} / N_{\rm tot}`.

        ax : matplotlib.pyplot.axes, optional(default=None)
            Axes to plot results on.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Axes results are plotted.

        Note
        ----
        This modifies the alpha value (type I error rate), keeping all other parameters
        fixed, and computes the number of expected extremes (:math:`n_{\rm exp}`) vs. the number
        observed (:math:`n_{\rm obs}`).  Theoretically, :math:`n_{\rm exp} = \alpha*N_{\rm tot}`.

        The 95% tolerance limit is given in black.  Points which fall outside these
        bounds are highlighted.

        Warning
        -------
        Both extreme points and outliers are considered "extremes" here.  
        """
        X_ = check_array(np.asarray(X, np.float64), accept_sparse=False)
        N_tot = X_.shape[0]
        n_values = np.arange(1, int(upper_frac * N_tot) + 1)
        alpha_values = n_values / N_tot

        n_observed = []
        for a in alpha_values:
            params = self.get_params()
            params["alpha"] = a
            model_ = EllipticManifold(**params)
            model_.fit(X)
            extremes, outliers = model_.check_outliers(X)
            n_observed.append(np.sum(extremes) + np.sum(outliers))
        n_observed = np.array(n_observed)

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

    def visualize(self, X_mats, labels, ax=None):
        """
        Plot the results automatically in 1 or 2 dimensions.

        Parameters
        ----------
        X_mats : list(array_like(float, ndim=2))
            List of different feature matrices to plot.

        labels : list(str)
            Labels for each feature matrix.

        ax : matplotlib.pyplot.axes, optional(default=None)
            Axes to plot results on.  If None, a new figure is created.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Axes results are plotted on.
        """
        check_is_fitted(self, "is_fitted_")
        n = self.kwargs[self.ndims]
        if n == 1:
            return self._visualize_1d(X_mats, labels, ax)
        elif n == 2:
            return self._visualize_2d(X_mats, labels, ax)
        else:
            raise Exception("Cannot visualize {} dimensions".format(n))

    def _visualize_2d(self, X_mats, labels, axes=None):
        """
        Plot 2D results.

        Parameters
        ----------
        X_mats : list(array_like(float, ndim=2))
            List of different feature matrices to plot.

        labels : list(str)
            Labels for each feature matrix.

        axes : matplotlib.pyplot.axes, optional(default=None)
            Axes to plot results on.  If None, a new figure is created.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Axes results are plotted on.
        """
        check_is_fitted(self, "is_fitted_")
        if self.kwargs[self.ndims] != 2:
            raise Exception(
                "Cannot perform 2D visualization for a {} dimensional score space.".format(
                    self.kwargs[self.ndims]
                )
            )
        if len(labels) != len(X_mats):
            raise ValueError("Must provide a label for each set of X")

        def soft_boundary_2d(rmax=10.0, rbins=1000, tbins=180):
            """
            Compute the bounding ellipse.

            Parameters
            ----------
            rmax : float
                Radius to g from class center to look for boundary.
            rbins : int
                Number of points to search from class center (r=0 to r=rmax) for
                boundary.
            tbins : int
                Number of bins to split [0, 2*pi) into around the class center.

            Returns
            -------
            ndarray
                Cutoff boundary.
            """

            def estimate_boundary(rmax, rbins, tbins):
                cutoff = []
                c = self.__class_center_
                for theta in np.linspace(0, 2 * np.pi, tbins):
                    # Walk "outward" until you meet the threshold
                    for r in np.linspace(0, rmax, rbins):
                        sPC = c + r * np.array([np.cos(theta), np.sin(theta)])

                        d = np.matmul(
                            np.matmul(
                                (sPC - c),
                                np.linalg.inv(self.__S_),
                            ),
                            (sPC - c).reshape(-1, 1),
                        )[0]
                        if d > self.__d_crit_:
                            cutoff.append(sPC)
                            break

                return np.array(cutoff)

            cutoff = estimate_boundary(rmax=rmax, rbins=rbins, tbins=tbins)

            return cutoff

        if axes is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = axes

        cutoff = soft_boundary_2d(
            rmax=np.sqrt(self.__d_crit_ * np.max(np.diag(self.__S_))) * 1.2,
            rbins=1000,
            tbins=90,
        )
        ax.plot(cutoff[:, 0], cutoff[:, 1], color="k")

        for i, (X, l) in enumerate(zip(X_mats, labels)):
            T = self.transform(X)
            ax.plot(
                T[:, 0],
                T[:, 1],
                "o",
                alpha=0.5,
                color="C{}".format(i),
                label=l,
            )

            ax.axis("equal")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.legend(loc="best")

        return ax

    def _visualize_1d(self, X_mats, labels, axes=None):
        """
        Plot 1D results.

        Parameters
        ----------
        X_mats : list(array_like(float, ndim=2))
            List of different feature matrices to plot.

        labels : list(str)
            Labels for each feature matrix.

        axes : matplotlib.pyplot.axes, optional(default=None)
            Axes to plot results on.  If None, a new figure is created.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Axes results are plotted on.
        """
        check_is_fitted(self, "is_fitted_")
        if self.kwargs[self.ndims] != 1:
            raise Exception(
                "Cannot perform 1D visualization for a {} dimensional score space.".format(
                    self.kwargs[self.ndims]
                )
            )
        if len(labels) != len(X_mats):
            raise ValueError("Must provide a label for each set of X")

        def soft_boundary_1d(rmax=10.0, rbins=1000):
            """
            Compute the bounding ellipse around for "soft" classification.

            Parameters
            ----------
            rmax : float
                Radius to go from class center to look for boundary.
            rbins : int
                Number of points to search from class center (r=0 to r=rmax) for
                boundary.

            Returns
            -------
            ndarray
                [high, low] boundary values.
            """

            def estimate_boundary(rmax, rbins):
                cutoff = []
                c = self.__class_center_
                # For each center, choose a systematic orientation
                for direction in [+1, -1]:
                    # Walk "outward" until you meet the threshold
                    for r in np.linspace(0, rmax, rbins):
                        sPC = c + r * direction
                        d = np.matmul(
                            np.matmul(
                                (sPC - c),
                                np.linalg.inv(self.__S_),
                            ),
                            (sPC - c).reshape(-1, 1),
                        )[0]
                        if d > self.__d_crit_:
                            cutoff.append(sPC)
                            break

                return np.array(cutoff)

            cutoff = estimate_boundary(rmax=rmax, rbins=rbins)

            return cutoff

        if axes is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = axes

        cutoff = soft_boundary_1d(
            rmax=np.sqrt(self.__d_crit_ * np.max(np.diag(self.__S_))) * 1.2,
            rbins=1000,
        )
        ax.axvline(cutoff[0], color="k")
        ax.axvline(cutoff[1], color="k")

        for i, (X, l) in enumerate(zip(X_mats, labels)):
            T = self.transform(X)
            ax.plot(
                T[:, 0],
                [i] * len(X),
                "o",
                alpha=0.5,
                color="C{}".format(i),
                label=l,
            )
        ax.set_xlabel("PC 1")

        ax.legend(loc="best")
        ax.set_yticks([])

        return ax
