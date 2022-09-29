"""
Feature selection algorithms.

@author: nam
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y


def optimize_jsd_bins(
    X, y, top_k=1, min_bins=5, max_bins=1000, bin_stride=10, show=False
):
    """
    Optimize bins by looking at how the average JSD (from top_k features) changes with n_bins.

    This is fit to a curve such that y = 1/<JSD> - 1, x = n_bins where y is modeled as (A/x)^B.
    We select x_optimal = A so that <JSD> = 0.5 where the average is over the top_k features.
    Generally, I recommend using the top half or quarter of features available.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    y : ndarray
        Response categories (1D).
    top_k : int
        Number of features to average the JSD over.
    min_bins : int
        Minimum number of bins to create distributions.
    max_bins : int
        Maximum number of bins to create distributions.
    bin_stride : int
        Stride between min_ and max_bins.
    show : bool
        Whether or not to display the fitting curve.

    Returns
    -------
    chosen : int
        Chosen number of bins (guaranteed to be at least 2).
    """
    X, y = check_X_y(X, y, copy=True)

    # Average the JSD over the top_k features (overall, not per class)
    average_top_jsd = []
    for bins in np.arange(min_bins, max_bins, bin_stride):
        jsd = JensenShannonDivergence(bins=bins, per_class=False, top_k=top_k)
        _ = jsd.fit(X, y)
        mean = []
        for k in range(top_k):
            mean += jsd.divergence[k][1].values()
        average_top_jsd.append([bins, np.mean(mean)])
    average_top_jsd = np.array(average_top_jsd)

    # Fit this to a y = (A/x)**B curve
    lr = LinearRegression(
        fit_intercept=True, copy_X=True, n_jobs=1, positive=False
    )
    x_ = average_top_jsd[:, 0].reshape(-1, 1)
    eps = 1.0e-6  # To prevent overflow
    y_ = 1.0 / np.clip(average_top_jsd[:, 1], a_max=1 - eps, a_min=eps) - 1.0
    lr.fit(np.log(x_), np.log(y_))
    B = -lr.coef_[0]
    A = np.exp(lr.intercept_ / B)

    chosen = np.max(
        [2, np.round(A, 0).astype(int)]
    )  # Take optimal guess as x = bins = A; should be at least 2
    # At bins (x) = A, y = 1 = 1/<JSD> - 1, so <JSD> = 0.5

    if show:
        fig = plt.figure()
        plt.plot(x_, y_, color="k", label="Measured")
        plt.plot(
            x_,
            (A / x_) ** B,
            color="red",
            label="Fit, " + r"$y = \left( \frac{A}{x} \right)^B$",
        )
        plt.legend(loc=1)
        plt.ylabel(r"y = $\frac{1}{\langle JSD \rangle} - 1$", fontsize=12)
        plt.xlabel("x = Number of Bins", fontsize=12)
        plt.axvline(chosen, color="blue", alpha=0.5)

    return chosen


class JensenShannonDivergence:
    """
    Compute the Jensen-Shanon (JS) divergence as a feature selection routine.

    The JS divergence is a measure of similarity between 2 distributions;
    it essentially describes the mean Kullback-Leibler divergence of two
    distributions from their mean.  When using a base of 2, the value is
    bounded between 0 (identical) and 1 (maximally difference).

    This is computed for each feature by comparing the distribution of
    that feature for observations of a class vs. those of all other classes.
    This "one-vs-all" comparison means that if the JS divergence is high
    (close to 1) for a feature for a certain class, then it is likely quite
    easy to construct some bounds that define intervals of that feature which
    characterize that class, and only that class.  For example, class A could
    have higher levels of some analyte than all other classes, or it could
    fall in range min < class A < max, where all other classes exist outside
    of this range.

    Features with high JS divergences for given classes mean that decision
    trees are likely to be successful if trained using those features.
    Therefore, this can be used as a feature selection method; alternatively,
    this can be used during the initial data analysis or exploratory data
    analysis phases to generate hypotheses.

    A suggestion for choosing the number of bins is to ensure the average
    number of points per bin is >= 5.  So if you have n_samples, the number
    of bins ~ n_samples/5.  This is a heuristic.  Also note n_samples is the
    total number of samples, regardless of class - since the distribution is
    going to be divided up to do a OvA comparison and both the "O" and "A"
    must be histogrammed using the same bins, it makes sense to look at the
    overall distribution.  However, this can cause issues if you have a
    minority class that is poorly sampled.  One solution could be do SMOTE
    in advance of any such calculation.  Since the JS divergence is (also)
    supervised, these should both appear in the pipeline for
    cross-validation anyway.  The example below illustrates this.

    Notes
    -----
    * Because class information is utilized, this is a supervised method and
    should be fit on the training sets/folds only.
    * If you request the top_k features for n_classes, you may receive less
    than top_k*n_classes features because classes may share analytes. This
    is only an upper bound.
    * You can search for the top_k features overall, but it may happen that
    one class has k or more features that have very different distributions
    from the rest, and therefore the top_k are only useful for distinguishing
    one (or only a few) classes, so per_class=True is set to True by default.
    * Using too many bins makes individual measurements all start to
    look unique and therefore 2 distributions appear to have a large
    JS divergence.  Be sure to try using a different number of bins
    to check your results qualitatively.  This also means outliers
    can be very problematic because they cause the the (max-min)
    range to be amplified artificially, which might actually make
    divergences look small because the bins are now too coarse. Use the robust
    option to investigate this.
    * As a counterargument, if a minority class is well separated by a
    feature, it could be that the entire class is considered an outlier
    and the code will return a NaN for that feature.  Investigate further
    and avoid using the `robust` option in that case.


    References
    -----
    * https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    * https://machinelearningmastery.com/divergence-between-probability-
    distributions/

    Example
    -------
    >>> pipeline = imblearn.pipeline.Pipeline(steps=[
    ...    ("smote", ScaledSMOTEENN(random_state=1)),
    ...    ("selector", JensenShannonDivergence(top_k=1,
    ...                                         per_class=True,
    ...                                         feature_names=X.columns)),
    ...    ("tree", DecisionTreeClassifier(random_state=1))
    ... ])
    >>> param_grid = [{
    ...    'smote__k_enn':[3, 5, 7, 10],
    ...    'smote__k_smote':[3, 5, 7, 10],
    ...    'smote__kind_sel_enn':['all', 'mode'],
    ...    'tree__max_depth':np.arange(1,4+1),
    ... }]
    >>> ncv = NestedCV(k_inner=2, k_outer=5)
    >>> results = ncv.grid_search(pipeline, param_grid, X.values, y.values)
    """

    def __init__(
        self,
        top_k=1,
        threshold=0.0,
        epsilon=1.0e-12,
        per_class=True,
        feature_names=None,
        bins=25,
        robust=False,
    ):
        """
        Instantiate the class.

        Parameters
        ----------
        top_k : int
            Return this many top ranked (closest to 1) features.
        threshold : float
            If specified, it should be between [0, 1]; this enforces that only
            the top_k features with at least this value are returned.
        epsilon : float
            A noise term is added to the distributions so probabilities are
            non-zero when not measured; this is numerically required to
            compute the KL divergence as part of the JS divergence.
        per_class : bool
            Do we want to return the top_k analytes (above threshold) on average
            (False) or the top_k per class (True)?  If True, up to
            top_k*n_classes can be returned, otherwise just top_k above
            threshold.
        feature_names : array-like
            List of names of features (columns of X) in order.
        bins : int
            Number of bins to use to construct the probability distribution.
        robust : bool
            Whether or not to use Q[1/3] +/- 1.5*IQR as a threshold to limit
            the range over which probability distributions are generated by
            histograms. This can make the results robust against skewed
            distributions with outliers which might cause the binning to
            become too coarse, but it might mask real differences or exclude
            important points if they aren't really outliers.
        """
        self.set_params(
            **{
                "epsilon": epsilon,
                "threshold": threshold,
                "top_k": top_k,
                "per_class": per_class,
                "feature_names": feature_names,
                "bins": bins,
                "robust": robust,
            }
        )
        return

    def set_params(self, **parameters):
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "epsilon": self.epsilon,
            "threshold": self.threshold,
            "top_k": self.top_k,
            "per_class": self.per_class,
            "feature_names": self.feature_names,
            "bins": self.bins,
            "robust": self.robust,
        }

    def make_prob_(self, p, ranges, bins=25):
        """Make the probability distribution for a feature."""
        prob, _ = np.histogram(p, bins=bins, range=ranges)
        return (prob + self.epsilon) / np.sum(prob)

    def jensen_shannon_(self, p, q):
        """Compute the JS divergence between p and q in base 2."""
        m = 0.5 * (p + q)
        return 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)

    def fit(self, X, y):
        """
        Compute the JS divergence for each class, for each feature.

        If per_class is True, then the ranking is done for each class
        and the top_k are chosen from each.

        Parameters
        ----------
        X : array-like
            Observations (columns as features); this is converted to
            a numpy array behind the scenes.
        y : array-like
            Classes; this is converted to a numpy array behind the scenes.
        """
        self.__X_ = np.array(X).copy()
        self.__y_ = np.array(y).copy()

        def compute_(column):
            if self.robust:
                # Use IQR to determine outliers - don't go past min/max
                # of data though
                iqr = scipy.stats.iqr(self.__X_[:, column], rng=(25, 75))
                ranges = (
                    np.max(
                        [
                            np.percentile(
                                self.__X_[:, column],
                                25,
                                interpolation="midpoint",
                            )
                            - 1.5 * iqr,
                            np.min(self.__X_[:, column]),
                        ]
                    ),
                    np.min(
                        [
                            np.percentile(
                                self.__X_[:, column],
                                75,
                                interpolation="midpoint",
                            )
                            + 1.5 * iqr,
                            np.max(self.__X_[:, column]),
                        ]
                    ),
                )
            else:
                ranges = (
                    np.min(self.__X_[:, column]),
                    np.max(self.__X_[:, column]),
                )
            div = {}
            for class_ in np.unique(self.__y_):
                p = self.make_prob_(
                    self.__X_[self.__y_ == class_, column], ranges, self.bins
                )
                q = self.make_prob_(
                    self.__X_[self.__y_ != class_, column], ranges, self.bins
                )
                div[class_] = self.jensen_shannon_(p, q)
            return div

        self.__divergence_ = {}
        for i in range(X.shape[1]):
            self.__divergence_[i] = compute_(i)

        self.__mask_ = np.array([False] * self.__X_.shape[1])
        if not self.per_class:
            # Just take top analytes regardless
            # Sort based on mean divergence across all classes
            self.__divergence_ = sorted(
                self.__divergence_.items(),
                key=lambda x: np.mean(list(x[1].values())),
                reverse=True,
            )
            for i in range(len(self.__divergence_)):
                divs = self.__divergence_[i][1]
                if (
                    np.mean(list(divs.values())) > self.threshold
                    and i < self.top_k
                ):
                    # If in top k and above threshold accept
                    index = self.__divergence_[i][0]
                    self.__mask_[index] = True

            if self.feature_names is not None:
                assert (
                    len(self.feature_names) == self.__X_.shape[1]
                ), "The size of feature_names disagrees with X"
                tmp = []
                for i in range(len(self.__divergence_)):
                    tmp.append(
                        (
                            self.feature_names[self.__divergence_[i][0]],
                            self.__divergence_[i][1],
                        )
                    )
                self.__divergence_ = tmp
        else:
            # Rank divergences by class
            top_class_div = {}
            for class_ in np.unique(self.__y_):
                # Sort based on divergence of a chosen class
                top_class_div[class_] = sorted(
                    self.__divergence_.items(),
                    key=lambda x: x[1][class_],
                    reverse=True,
                )
                for i in range(len(top_class_div[class_])):
                    divs = top_class_div[class_][i][1]
                    if divs[class_] > self.threshold and i < self.top_k:
                        # If in top k and above threshold accept
                        index = top_class_div[class_][i][0]
                        self.__mask_[index] = True

            if self.feature_names is not None:
                assert (
                    len(self.feature_names) == self.__X_.shape[1]
                ), "The size of feature_names disagrees with X"
                self.__divergence_ = {}
                for class_ in top_class_div.keys():
                    tmp = []
                    for i in range(len(top_class_div[class_])):
                        tmp.append(
                            (
                                self.feature_names[top_class_div[class_][i][0]],
                                top_class_div[class_][i][1],
                            )
                        )
                    self.__divergence_[class_] = tmp  # Class-based results
            else:
                self.__divergence_ = top_class_div

        return self

    def transform(self, X):
        """
        Select analytes with the highest divergences.

        Parameters
        ----------
        X : array-like
            Observations (columns as features); this is converted to
            a numpy array behind the scenes.  Should be in the same
            column order as the X trained on.

        Returns
        -------
        X_selected : ndarray
            Matrix with only the features selected.
        """
        X = np.array(X)
        if X.shape[1] != len(self.__mask_):
            raise ValueError("The shape of X has changed")

        return X[:, self.__mask_]

    def visualize(
        self, by_class=True, classes=None, threshold=None, ax=None, figsize=None
    ):
        """
        Plot divergences for each class.

        Parameters
        ----------
        by_class : bool
            Whether to plot feature divergences sorted by class
            (per_class=True) or by their overall mean (per_class=False). This
            is independent of the choice for per_class made at instantiation.
        classes : array-like
            List of classes to plot; if None, plot all trained on. Only
            relevant for when by_class is true.
        threshold : float
            Draws a horizontal red line to visualize this threshold.
        ax : list(matplotlib.pyplot.axes.Axes) or None
            If None, creates its own plots, otherwise plots on the axes
            given.
        figsize : tuple(int,int) or None
            Figure size to produce.

        Example
        -------
        >>> js = JensenShannonDivergence(top_k=3,
        ...                              feature_names=X.columns,
        ...                              per_class=False)
        >>> _ = js.fit(X, y)
        >>> js.visualize(by_class=False, threshold=0.7)
        """
        if by_class:  # Plot results sorted by class
            disp_classes = np.unique(self.__y_) if classes is None else classes
            if ax is None:
                fig, ax = plt.subplots(
                    nrows=len(disp_classes), ncols=1, figsize=figsize
                )
                ax = ax.ravel()
            else:
                try:
                    iter(ax)
                except TypeError:
                    ax = [ax]
            for class_, ax_ in zip(disp_classes, ax):
                ax_.set_title(class_)
                if self.per_class:
                    xv = [a[0] for a in self.divergence[class_]]
                    yv = [a[1][class_] for a in self.divergence[class_]]
                else:
                    xv = [a[0] for a in self.divergence]
                    yv = [a[1][class_] for a in self.divergence]
                    resorted = sorted(
                        zip(xv, yv), key=lambda x: x[1], reverse=True
                    )
                    xv = [a[0] for a in resorted]
                    yv = [a[1] for a in resorted]

                ax_.bar(x=np.arange(1, len(xv) + 1), height=yv)
                if threshold is not None:
                    ax_.axhline(threshold, color="r")
                ax_.set_xticks(np.arange(1, len(xv) + 1))
                _ = ax_.set_xticklabels(xv, rotation=90)
                plt.tight_layout()
        else:  # Plot results by mean across all classes
            _ = plt.figure(figsize=figsize)
            ax = plt.gca()
            if self.per_class:
                arbitrary_class = list(self.divergence.keys())[0]
                xv = [a[0] for a in self.divergence[arbitrary_class]]
                yv = [
                    np.mean(list(a[1].values()))
                    for a in self.divergence[arbitrary_class]
                ]
                resorted = sorted(zip(xv, yv), key=lambda x: x[1], reverse=True)
                xv = [a[0] for a in resorted]
                yv = [a[1] for a in resorted]
            else:
                xv = [a[0] for a in self.divergence]
                yv = [np.mean(list(a[1].values())) for a in self.divergence]

            ax.bar(x=np.arange(1, len(xv) + 1), height=yv)
            if threshold is not None:
                ax.axhline(threshold, color="r")
            ax_.set_xticks(np.arange(1, len(xv) + 1))
            _ = ax.set_xticklabels(xv, rotation=90)
            plt.tight_layout()

    @property
    def accepted(self):
        """
        Return the selected features.

        This will return a boolean mask if feature_names was not specified,
        otherwise they are converted to column names.
        """
        if self.feature_names is not None:
            return np.array(self.feature_names)[self.__mask_]
        else:
            return self.__mask_

    @property
    def divergence(self):
        """
        Return the JS divergences computed.

        If per_class is True, this returns a dictionary with sorted entries
        for each class, otherwise an array of (feature, {class:JS divergence})
        is returned sorted by the highest average JS divergence for that
        feature.

        Example
        -------
        >>> js.divergence # per_class=False
        [('trans-chlordane', {'Larus (gull)': 0.7666381975219294,
                              'Phoebastria (albatross)': 0.7899250953304608,
                              'Uria (murre)': 0.9103582224128998}),
        ('trans-nonachlor', {'Larus (gull)': 0.6710164255792366,
                             'Phoebastria (albatross)': 0.6581336196968821,
                             'Uria (murre)': 0.8520633525927908}),
        ...
        >>> js.divergence # per_class=True
        {'Larus (gull)': [
                ('PCB 110',
                         {'Larus (gull)': 0.8600457234182239,
                          'Phoebastria (albatross)': 0.2216666656792119,
                          'Uria (murre)': 0.4488878856962035}),
                ('PCB 52',
                         {'Larus (gull)': 0.7746536806371277,
                          'Phoebastria (albatross)': 0.14537496042116752,
                          'Uria (murre)': 0.3151990265662022}),
                ...
                ]
        'Phoebastria (albatross)' : [
                ...
                ]
        'Uria (murre)' : [
                ...
                ]
        }
        """
        return self.__divergence_.copy()


class PipeBorutaSHAP:
    """
    BorutaSHAP feature selector for pipelines.

    Create a BorutaSHAP instance that is compatible with
    scikit-learn's estimator API and can be used in scikit-learn and
    imblearn's pipelines.

    This is essentially a wrapper for
    [BorutaSHAP](https://github.com/Ekeany/Boruta-Shap). See
    documentation therein for additional details. This requires input as a
    Pandas DataFrame so an internal conversion will be performed.  Also,
    you must provide the names of the original columns (in order) at
    instantiation.

    BorutaSHAP works with tree-based models which do not require scaling or
    other preprocessing, therefore this stage can actually be put in the
    pipeline either before or after standard scaling (see example below).

    Notes
    -----
    BorutaSHAP is expensive; default parameters are set to be gentle but it can
    dramatically increase the cost of nested CV or grid searching.

    Leave `column_names` as None in pipelines which have feature engineers that
    can change the number of components. PipeBorutaSHAP will just label columns
    with integers to handle things consistently internally.

    Example
    -------
    >>> X, y = pd.read_csv(...), pd.read_csv(...)
    >>> pipeline = imblearn.pipeline.Pipeline(steps=[
    ...     ("smote", ScaledSMOTEENN(k_enn=5, kind_sel_enn='mode')),
    ...     ("scaler", StandardScaler()),
    ...     ("boruta", PipeBorutaSHAP(column_names=X.columns)),
    ...     ('tree', DecisionTreeClassifier(random_state=0))
    ...     ])
    >>> param_grid = [
    ...     {'smote__k_enn':[3, 5],
    ...     'smote__kind_sel_enn':['all', 'mode'],
    ...     'tree__max_depth':[3,5],
    ...     'boruta__pvalue':[0.05, 0.1]
    ...     }]
    >>> gs = GridSearchCV(estimator=pipeline,
    ...     param_grid=param_grid,
    ...     n_jobs=-1,
    ...     cv=StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
    ...     )
    >>> gs.fit(X.values, y.values)
    >>> # OR, ...
    >>> NestedCV().grid_search(pipeline, param_grid, X.values, y.values)
    """

    def __init__(
        self,
        column_names=None,
        model=RF(
            n_estimators=100,
            criterion="entropy",
            random_state=0,
            class_weight="balanced",
        ),
        classification=True,
        percentile=100,
        pvalue=0.05,
    ):
        """Instantiate the class."""
        self.set_params(
            **{
                "column_names": column_names,
                "model": model,
                "classification": classification,
                "percentile": percentile,
                "pvalue": pvalue,
            }
        )
        return

    def set_params(self, **parameters):
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "column_names": self.column_names,
            "model": self.model,
            "classification": self.classification,
            "percentile": self.percentile,
            "pvalue": self.pvalue,
        }

    def fit(self, X, y):
        """Fit BorutaSHAP to data."""
        # Convert X and y to pandas.DataFrame and series
        from BorutaShap import BorutaShap

        self.__boruta_ = BorutaShap(
            model=self.model,
            importance_measure="shap",
            classification=self.classification,
            percentile=self.percentile,
            pvalue=self.pvalue,
        )

        if self.column_names is None:
            self.column_names = [str(i) for i in range(X.shape[1])]
        else:
            assert X.shape[1] == len(
                self.column_names
            ), "X is not compatible \
            with column names provided."
        # BorutaSHAP is expensive so try to keep these to reasonable values.
        # If used in kfold CV the cost goes up very quickly.
        self.__boruta_.fit(
            X=pd.DataFrame(data=X, columns=self.column_names),
            y=pd.Series(data=y),
            n_trials=20,
            sample=False,
            train_or_test="test",  # Does internal 70:30 train/test
            normalize=True,
            verbose=False,
            random_state=0,
        )

        return self

    def transform(self, X):
        """Select the columns that were deemed important."""
        # Could reorder X relative to original input?
        return pd.DataFrame(data=X, columns=self.column_names)[
            self.accepted
        ].values

    @property
    def accepted(self):
        """Get the columns that are important."""
        return self.__boruta_.accepted

    @property
    def rejected(self):
        """Get the columns that are not important."""
        return self.__boruta_.rejected
