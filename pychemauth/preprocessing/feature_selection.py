"""
Feature selection algorithms.

author: nam
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import entropy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pychemauth.eda.explore import InspectData


class CollinearFeatureSelector(TransformerMixin, BaseEstimator):
    """
    Select features from different clusters determined by their collinearity.

    Parameters
    ----------
    t : scalar(float), optional(default=0.0)
        See `InspectData.cluster_collinear; Ward clustering threshold to
        determine the number of clusters.

    seed : scalar(int), optional(default=42)
        Random number generator seed.  Set this value for reproducible
        calculations.

    minimize_label_entropy : scalar(bool)
        Whether or not to refine choices based on a lookup function which
        defines the similarity of features.

    kwargs : dict()
        A dictionary of keyword arguments to `InspectData.minimize_cluster_label_entropy`.
        If `minimize_label_entropy` is true, this should at least contain a `lookup` function, but may also
        include: {cutoff_factor, n_restarts, max_iters, early_stopping, T}.

    Note
    ----
    This is intended for use with `sklearn.feature_selection.SelectFromModel`.

    Warning
    -------
    If you are using `minimize_label_entropy` the lookup function has an important restriction.
    The `lookup` function may take multiple default arguments, but the first should be an
    integer corresponding to the column index (starting from 0).  Data column names are not present if
    data is a numpy array, and not stored if it is provided as as pandas.DataFrame so that operations
    are consistent with `sklearn.feature_selection.SelectFromModel`. Fortunately, name information
    can be encoded in the default parameters when it is defined at the scope of the SelectFromModel
    object. This is important if you want to categorize features based on their name.
    See the example below for illustration.

    Example
    -------
    >>> from sklearn.feature_selection import SelectFromModel
    >>> column_names = ['ash', 'malic acid', ...]
    >>> def lookup(feature_idx, column_names=column_names):
    ...    if 'l' in column_names[feature_idx].lower():
    ...        return 'Contains letter L'
    ...    else:
    ...        return 'No letter L'
    >>> se = SelectFromModel(
    ...    estimator=ClusterSelector(t=0.9, seed=42, minimize_label_entropy=True,
    ...    kwargs={"lookup":lookup, "n_restarts":5, "max_iters":100, "T":1.0}),
    ...    threshold=0.5,
    ...    prefit=False)
    >>> se.fit(X_train)
    """

    def __init__(self, t=0.0, seed=42, minimize_label_entropy=False, kwargs={}):
        """Instantiate the selector."""
        self.set_params(
            **{
                "t": t,
                "seed": seed,
                "minimize_label_entropy": minimize_label_entropy,
                "kwargs": kwargs,
            }
        )

    def set_params(self, **parameters):
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "t": self.t,
            "seed": self.seed,
            "minimize_label_entropy": self.minimize_label_entropy,
            "kwargs": self.kwargs,
        }

    def get_feature_names_out(self, input_features=None):
        """
        Return the selected features.

        Parameters
        ----------
        input_features : array_like(str, ndim=1), optional(default=None)
            List of input features to mask.

        Returns
        -------
        features_out : ndarray(str or bool, ndim=1)
            Array of features if `input_features` is provided, otherwise boolean mask.

        """
        check_is_fitted(self, "is_fitted_")
        mask = np.asarray(self.feature_importances_, dtype=bool)
        if input_features is not None:
            assert (
                len(input_features) == self.n_features_in_
            ), "input_features has the wrong size."
            return np.array(input_features, dtype=str)[mask]
        else:
            return mask

    def get_support(self):
        """
        Return the selected features.

        Returns
        -------
        support : ndarray(bool, ndim=1)
            Boolean array of columns that were selected.
        """
        check_is_fitted(self, "is_fitted_")
        return np.asarray(self.feature_importances_, dtype=bool)

    def fit(self, X, y=None):
        """
        Fit the selector.

        Parameters
        ----------
        X : array_like(float, ndim=2) or pandas.DataFrame
            Input feature matrix.

        y : array_like(float, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        self : CollinearFeatureSelector
            Fitted selector.
        """
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        if y is not None:  # Just so this passes sklearn api checks
            X, y = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_2d=True,
                force_all_finite=True,
                y_numeric=False,
            )
        self.n_features_in_ = X.shape[1]

        (
            selected_features,
            cluster_id_to_feature_ids,
            _,
        ) = InspectData.cluster_collinear(
            X=X,
            feature_names=None,  # We use indices here not names to be consistent with sklearn.feature_selection.SelectFromModel
            t=self.t,
            display=False,  # We can change this to False so we don't have these plots
        )
        self.is_fitted_ = True

        self.feature_importances_ = np.zeros(X.shape[1], dtype=np.float64)
        if self.minimize_label_entropy is True:
            # Optimize choices in the cluster
            best_choices = InspectData.minimize_cluster_label_entropy(
                cluster_id_to_feature_ids=cluster_id_to_feature_ids,
                X=pd.DataFrame(data=X, columns=np.arange(X.shape[1])),
                seed=self.seed,
                **self.kwargs
            )

            for idx in best_choices:
                self.feature_importances_[idx] = 1
        else:
            # Select a random feature from each cluster
            np.random.seed(self.seed)
            for cluster in cluster_id_to_feature_ids.keys():
                feature_idx = cluster_id_to_feature_ids[cluster][
                    np.random.randint(
                        0, high=len(cluster_id_to_feature_ids[cluster])
                    )
                ]
                self.feature_importances_[feature_idx] = 1

        return self

    def transform(self, X):
        """
        Transform X by removing features not selected.

        Parameters
        ----------
        X : array_like(float, ndim=2) or pandas.DataFrame
            Input feature matrix.

        Returns
        -------
        X_selected : array_like(float, ndim=2) or pandas.DataFrame
            X with all features not selected removed. If X is provided as a DataFrame
            a DataFrame is returned.
        """
        check_is_fitted(self, "is_fitted_")
        mask = np.asarray(self.feature_importances_, dtype=bool)
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        X_ = X_[:, mask]

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(data=X_, columns=X.columns[mask])
        else:
            return X_

    def fit_transform(self, X, y=None):
        """
        Fit then transform.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Input feature matrix.

        y : array_like(float, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        X_selected : array_like(float, ndim=2)
            X with all features not selected removed.
        """
        self.fit(X, y)
        return self.transform(X)

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
            "_skip_test": [
                "check_estimators_fit_returns_self"  # Unfortunately, this test seems to generate bad dummy data for this
            ],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }


class JensenShannonDivergence(TransformerMixin, BaseEstimator):
    r"""
    Compute the Jensen-Shanon divergence (JSD).

    Parameters
    ----------
    top_k : scalar(int), optional(default=1)
        Return this many top ranked (closest to 1) features.

    threshold : scalar(float), optional(default=0.0)
        If specified, it should be between [0, 1]; this enforces that only
        the `top_k` features with at least this value are returned.

    epsilon : scalar(float), optional(default=1.0e-12)
        A noise term is added to the distributions so probabilities are
        non-zero when not measured; this is numerically required to
        compute the KL divergence as part of the JS divergence.

    per_class : scalar(bool), optional(default=True)
        Do we want to return the top_k analytes (above threshold) on average
        (False) or the top_k per class (True)?  If True, up to
        top_k*n_classes can be returned, otherwise just top_k above
        threshold.

    feature_names : array_like(str, ndim=1), optional(default=None)
        List of names of features (columns of X) in order.

    bins : scalar(int), optional(default=25)
        Number of bins to use to construct the probability distribution.

    robust : scalar(bool), optional(default=False)
        Whether or not to use Q[1 or 3] +/- 1.5*IQR as a threshold to limit
        the range over which probability distributions are generated by
        histograms. This can make the results robust against skewed
        distributions with outliers which might cause the binning to
        become too coarse, but it might mask real differences or exclude
        important points if they aren't really outliers.

    Note
    ----
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
    ----------
    * https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

    * https://machinelearningmastery.com/divergence-between-probability-distributions/

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
    >>> ncv = BiasedNestedCV(k_inner=2, k_outer=5)
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
        """Instantiate the class."""
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

    def _make_prob(self, p, ranges, bins=25):
        """Make the probability distribution for a feature."""
        prob, _ = np.histogram(p, bins=bins, range=ranges)
        return (prob + self.epsilon) / np.sum(prob)

    def _jensen_shannon(self, p, q):
        """Compute the JS divergence between p and q in base 2."""
        m = 0.5 * (p + q)
        return 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)

    def fit(self, X, y):
        r"""
        Compute the JSD for each class, for each feature.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Observations (columns as features); this is converted to
            a numpy array behind the scenes.

        y : array_like(str or int, ndim=1)
            Classes; this is converted to a numpy array behind the scenes.

        Returns
        -------
        self : JensenShannonDivergence
            Fitted model.

        Note
        ----
        If `per_class` is True, then the ranking is done for each class
        and the `top_k` are chosen from each.
        """
        self.__X_, self.__y_ = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
            copy=True,
        )
        self.n_features_in_ = self.__X_.shape[1]

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
                                method="midpoint",
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
                                method="midpoint",
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
                p = self._make_prob(
                    self.__X_[self.__y_ == class_, column], ranges, self.bins
                )
                q = self._make_prob(
                    self.__X_[self.__y_ != class_, column], ranges, self.bins
                )
                div[class_] = self._jensen_shannon(p, q)
            return div

        self.__divergence_ = {}
        for i in range(self.n_features_in_):
            self.__divergence_[i] = compute_(i)

        self.__mask_ = np.array([False] * self.n_features_in_)
        if not self.per_class:
            # Just take top analytes regardless
            # Sort based on mean divergence across all classes
            self.__divergence_ = sorted(
                self.__divergence_.items(),
                key=lambda x: np.mean(list(x[1].values())),
                reverse=True,
            )
            for i in range(self.n_features_in_):
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
                    len(self.feature_names) == self.n_features_in_
                ), "The size of feature_names disagrees with X"
                tmp = []
                for i in range(self.n_features_in_):
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
                    len(self.feature_names) == self.n_features_in_
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

        # So this is compatible with sklearn.feature_selection.SelectFromModel
        self.feature_importances_ = np.asarray(self.__mask_, dtype=np.float64)

        self.is_fitted_ = True

        return self

    def transform(self, X):
        """
        Select analytes with the highest divergences.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Observations (columns as features); this is converted to
            a numpy array behind the scenes.Should be in the same
            column order as the X trained on.

        Returns
        -------
        X_selected : ndarray(float, ndim=2)
            Matrix with only the features selected.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        return X[:, self.__mask_]

    def visualize(
        self, by_class=True, classes=None, threshold=None, ax=None, figsize=None
    ):
        """
        Plot divergences for each class.

        Parameters
        ----------
        by_class : scalar(bool), optional(default=True)
            Whether to plot feature divergences sorted by class
            (per_class=True) or by their overall mean (per_class=False). This
            is independent of the choice for per_class made at instantiation.

        classes : array_like(str, ndim=1), optional(default=None)
            List of classes to plot; if None, plot all trained on. Only
            relevant for when by_class is true.

        threshold : scalar(float), optional(default=None)
            Draws a horizontal red line to visualize this threshold.

        ax : list(matplotlib.pyplot.axes), optional(default=None)
            If None, creates its own plots, otherwise plots on the axes
            given.

        figsize : tuple(int,int), optional(default=None)
            Figure size to produce.

        Example
        -------
        >>> js = JensenShannonDivergence(top_k=3,
        ...                              feature_names=X.columns,
        ...                              per_class=False)
        >>> _ = js.fit(X, y)
        >>> js.visualize(by_class=False, threshold=0.7)
        """
        check_is_fitted(self, "is_fitted_")
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
            if ax is None:
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
            ax.set_xticks(np.arange(1, len(xv) + 1))
            _ = ax.set_xticklabels(xv, rotation=90)
            plt.tight_layout()

    def get_feature_names_out(self):
        """
        Return the selected features.

        Note
        ----
        This will return a boolean mask if feature_names was not specified,
        otherwise they are converted to column names.
        """
        check_is_fitted(self, "is_fitted_")
        if self.feature_names is not None:
            return np.array(self.feature_names)[self.__mask_]
        else:
            return self.__mask_

    def get_support(self):
        """
        Return the selected features.

        Returns
        -------
        support : ndarray(bool, ndim=1)
            Boolean array of columns that were selected.
        """
        check_is_fitted(self, "is_fitted_")
        return self.__mask_

    @property
    def divergence(self):
        """
        Return the JS divergences computed.

        Note
        ----
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
        check_is_fitted(self, "is_fitted_")
        return self.__divergence_.copy()

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
            "_skip_test": [],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }


class BorutaSHAPFeatureSelector(TransformerMixin, BaseEstimator):
    r"""
    BorutaSHAP feature selector for use in pipelines.

    Parameters
    ----------
    column_names : array_like(str, ndim=1), optional(default=None)
        Name of the columns in the feature matrix.

    model : sklearn.base.BaseEstimator, optional(default=sklearn.ensemble.RandomForestClassifier)
        An unfitted model for estimating SHAP values.

    classification : scalar(bool), optional(default=True)
        Whether this is a classification or regression problem.  Make sure the `model` you are using
        is consistent with this choice.

    percentile : scalar(int), optional(default=100)
        BorutaSHAP percentile to use.  This sets the percentile of the shadow features'
        importances which the algorithm uses as the threshold to determine if a feature
        gets a "hit".  The original Boruta implementation uses the max importance feature
        of all the shadow features, equivalent to the default value of 100 here.

    pvalue : scalar(float), optional(default=0.05)
        P-value to use in BorutaSHAP.

    seed : scalar(int), optional(default=42)
        Seed for BorutaSHAP calculation.

    Note
    ----
    Create a BorutaSHAP instance that is compatible with
    scikit-learn's estimator API and can be used in scikit-learn and
    imblearn's pipelines. It is intended for use with
    `sklearn.feature_selection.SelectFromModel`.  Internally, accepted
    features are given a feature_importance_ of 1, while those rejected
    are assigned zero so that a threshold of, e.g., 0.5 will separate
    them.

    This is essentially a wrapper for
    `BorutaSHAP <https://github.com/Ekeany/Boruta-Shap>`_. See
    documentation therein for additional details. This requires input as a
    Pandas DataFrame so an internal conversion will be performed.  Also,
    you must provide the names of the original columns (in order) at
    instantiation.

    BorutaSHAP works with tree-based models which do not require scaling or
    other preprocessing, therefore this stage can actually be put in the
    pipeline either before or after standard scaling (see example below).

    BorutaSHAP is expensive; default parameters are set to be gentle but it can
    dramatically increase the cost of nested CV or grid searching.

    Leave `column_names` as None in pipelines which have feature engineers that
    can change the number of components. BorutaSHAPFeatureSelector will just label columns
    with integers to handle things consistently internally.

    Example
    -------
    >>> X, y = pd.read_csv(...), pd.read_csv(...)
    >>> pipeline = imblearn.pipeline.Pipeline(steps=[
    ...     ("smote", ScaledSMOTEENN(k_enn=5, kind_sel_enn='mode')),
    ...     ("scaler", StandardScaler()),
    ...     ("boruta", BorutaSHAPFeatureSelector(column_names=X.columns)),
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
    >>> BiasedNestedCV().grid_search(pipeline, param_grid, X.values, y.values)
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
        seed=42,
    ):
        """Instantiate the class."""
        self.set_params(
            **{
                "column_names": column_names,
                "model": model,
                "classification": classification,
                "percentile": percentile,
                "pvalue": pvalue,
                "seed": seed,
            }
        )

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
            "seed": self.seed,
        }

    def fit(self, X, y):
        """
        Fit BorutaSHAP to data.

        Parameters
        ----------
        X : array_like(float, ndim=2) or pandas.DataFrame
            Feature matrix.

        y : array_like(str or int, ndim=1) or array_like(float, ndim=1) or pandas.Series
            Response variable or classes, depending on whether this is is a
            classification or regression problem.

        Returns
        -------
        self : BorutaSHAPFeatureSelector
            Fitted model.
        """
        X_, y_ = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
        )
        self.n_features_in_ = X_.shape[1]

        # Convert X and y to pandas.DataFrame and series
        from BorutaShap import BorutaShap

        self.__boruta_ = BorutaShap(
            model=self.model,
            importance_measure="shap",
            classification=self.classification,
            percentile=self.percentile,
            pvalue=self.pvalue,
        )

        self.column_names_ = self.column_names
        if self.column_names_ is None:
            self.column_names_ = [str(i) for i in range(self.n_features_in_)]
        else:
            assert self.n_features_in_ == len(
                self.column_names_
            ), "X is not compatible \
            with column names provided."

        # BorutaSHAP is expensive so try to keep these to reasonable values.
        # If used in kfold CV the cost goes up very quickly.
        self.__boruta_.fit(
            X=pd.DataFrame(data=X_, columns=self.column_names_),
            y=pd.Series(data=y_),
            sample=False,
            train_or_test="train",
            normalize=True,
            verbose=False,
            random_state=self.seed,
            stratify=(pd.Series(data=y_) if self.classification else None),
        )
        self.is_fitted_ = True

        # For use with sklearn.feature_selection.SelectFromModel
        self.feature_importances_ = np.zeros(
            len(self.column_names_), dtype=np.float64
        )
        self.feature_importances_[self.get_support()] = 1

        return self

    def transform(self, X):
        """
        Select the columns that were deemed important.

        Parameters
        ----------
        X : array_like(float, ndim=2) or pandas.DataFrame
            Feature matrix.

        Returns
        ----------
        X_selected : array_like(float, ndim=2)
            Feature matrix with only the relevant columns.
        """
        check_is_fitted(self, "is_fitted_")
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        df = pd.DataFrame(data=X_, columns=self.column_names_)[
            self.get_feature_names_out()
        ]
        if isinstance(X, pd.DataFrame):
            return df
        else:
            return df.values

    def get_feature_names_out(self):
        """
        Return the selected features.

        Returns
        -------
        support : ndarray(str, ndim=1)
            Names of the columns that were selected.
        """
        check_is_fitted(self, "is_fitted_")
        return np.asarray(self.column_names_)[self.get_support()]

    def get_support(self):
        """
        Return the selected features.

        Returns
        -------
        support : ndarray(bool, ndim=1)
            Boolean array of columns that were selected.
        """
        check_is_fitted(self, "is_fitted_")
        return np.array(
            [
                column in self.__boruta_.accepted
                for column in self.column_names_
            ],
            dtype=bool,
        )

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
                "check_parameters_default_constructible"  # sklearn has problems with model being a RF or other model
            ],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
