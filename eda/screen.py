"""
Screening tools for features of data.

@author: nam
"""

import inspect
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y

import sys
sys.path.append("../")
from pychemauth.preprocessing.feature_selection import JensenShannonDivergence


class RedFlags:
    """
    Check dense, tabular data for any obvious "red flags".

    This check returns warnings for each possible issue that it finds.

    Example
    -------
    >>> r = RedFlags()
    >>> r.all_checks # Get dictionary of all checks possible
    >>> r.get_checks # Get dictionary of all checks that we are going to run
    >>> r.run(X, y)
    """

    def __init__(self, use=None):
        """
        Instantiate the class.

        Parameters
        ----------
        use : list
            Name of checks to use; if None (default), run all.
        """
        self.perform = {}
        self.all_checks = dict(
            [f for f in inspect.getmembers(self) if f[0].startswith("check_")]
        )

        if use:
            assert isinstance(use, list)
            for t in use:
                if t in self.all_checks:
                    self.perform[t] = self.all_checks[t]
        else:
            self.perform = self.all_checks

    @property
    def get_checks(self):
        """Get a list of all checks that will be performed."""
        return self.perform

    def run(self, X, y=None):
        """
        Run all checks.

        Notes
        -----
        X may be an object that contains columns of different types of data (e.g., str, bool, floats)
        while y (if provided) must be either completely composed of floats or strings.
        """
        X_safe, y_safe = check_X_y(
            X,
            y,
            accept_sparse=False,  # Do not allow sparse matrices
            accept_large_sparse=False,
            copy=True,  # Force a copy to made
            dtype=None,  # Do not change the input type
            ensure_2d=True,  # Enforce 2D matrix, raise ValueError if not
            allow_nd=False,
            ensure_min_samples=1,  # Min 1 row
            ensure_min_features=1,  # Min 1 column
            y_numeric=False,  # y can be anything, not just numbers
            force_all_finite=False,  # Check this later - NaN might be intentional for imputation
        )

        # Check that y is either all numbers or all strings
        if all(
            [
                isinstance(
                    item,
                    (
                        float,
                        np.float32,
                        np.float64,
                        int,
                        np.uint,
                        np.int32,
                        np.int64,
                    ),
                )
                for item in np.array(y).flatten()
            ]
        ):
            self.y_type = "float"
        elif all([isinstance(item, str) for item in np.asarray(y).flatten()]):
            self.y_type = "str"
        else:
            raise Exception(
                "y contains mixed types (e.g., some floats some strings)"
            )

        # Leave X open to have bools, floats, etc. in different columns and
        # allow each check to perform the relevant examination.
        for name, test in self.perform.items():
            test(X=X_safe, y=y_safe)

    def check_nan(self, X, y=None):
        """Check if any entries are np.nan."""
        found = False
        if np.any(np.isnan(np.asarray(X).flatten())):
            warnings.warn("X contains NaN values; this will require imputation")
            found = True

        if not (y is None):
            if self.y_type == "float":
                if np.any(np.isnan(np.asarray(y).flatten())):
                    warnings.warn("y contains NaN values")
                    found = True
        return found

    def check_inf(self, X, y=None):
        """Check if any entries are np.inf."""
        found = False
        if np.any(np.isinf(np.asarray(X).flatten())):
            warnings.warn("X contains Inf values; this will require imputation")
            found = True

        if not (y is None):
            if self.y_type == "float":
                if np.any(np.isinf(np.asarray(y).flatten())):
                    warnings.warn("y contains Inf values")
                    found = True
        return found

    def check_zero_variance(self, X, y=None):
        """Check if any columns in X are constant (unsupervised)."""
        tol = 1.0e-12
        std_dev = np.std(np.asarray(X, dtype=np.float64), axis=0)
        if np.any(std_dev < tol):
            warnings.warn(
                "X columns with no variance: {}".format(
                    np.where(std_dev < tol)[0]
                )
            )
            return True
        return False

    def check_min_observations(self, X, y, n=5):
        """Check each class has a minimum number of observations."""
        found = False
        if not (y is None):
            y_ = np.asarray(y)
            X_ = np.asarray(X, dtype=np.float64)
            if y_.ndim == 1:
                for c in np.unique(y_):
                    n_obs = np.sum(y_ == c)
                    if n_obs < n:
                        warnings.warn(
                            "Class {} only contains {} observations".format(
                                c, n_obs
                            )
                        )
                        found = True
        return found

    def check_min_different_values(self, X, y, n=5):
        """
        Check each class has a minimum number of different observations.

        This is important during CV which might split, e.g., a bimodal distribution up so that
        all observations in the train split(s) are the same, leading to a std = 0, causing
        standardization to "explode."  As a result, it is recommended that n be at least k
        in k-fold CV, but should generally be more.
        """
        found = False
        if not (y is None):
            y_ = np.asarray(y)
            X_ = np.asarray(X, dtype=np.float64)
            if y_.ndim == 1:
                for c in np.unique(y_):
                    # NaN and Inf values are considered unique from other things but multiple NaN are considered the same category
                    n_unique = {
                        i: np.unique(X_[y_ == c, i], return_counts=True)
                        for i in range(X_.shape[1])
                    }
                    for column in sorted(n_unique.keys()):
                        if len(n_unique[column][0]) < n:
                            warnings.warn(
                                "Class {} (n={}) only contains only {} different observations for feature (column) index {}".format(
                                    c,
                                    np.sum(y_ == c),
                                    len(n_unique[column][0]),
                                    column,
                                )
                            )
                            found = True

        return found
        
    def check_duplicates(self, X, y=None):
    	"""
    	Check if any rows in X are duplicates (numerically).
    	"""
        tol = 1.0e-12
        if np.any(scipy.spatial.distance.pdist(X, metric='euclidean') < tol):
	    warnings.warn("There are duplicate rows in X")
	    return True
	else:
	    return False

    def check_zero_class_variance(self, X, y=None):
        """
        Check if columns in X are constant for any classes (supervised).

        Notes
        -----
        For models like SIMCA, which break things up based on class, this
        is particularly important.

        This test is skipped if NaN or Inf are found in X.
        """
        tol = 1.0e-12
        found = False
        if not (y is None):
            y_ = np.array(y)
            X_ = np.array(X, dtype=np.float64)
            assert y_.ndim == 1
            for c in np.unique(y_):
                X_sub = X_[y_ == c, :]
                # Cannot compute std() with Inf or NaN correctly, so just skip it
                if not self.check_nan(X_sub, y) and not self.check_inf(
                    X_sub, y
                ):
                    std_dev = np.std(X_sub, axis=0)
                    if np.any(std_dev < tol):
                        warnings.warn(
                            "X[{}] contains columns with no variance: {}".format(
                                c, np.where(std_dev < tol)[0]
                            )
                        )
                        found = True
        return found


class JSBinary:
    """
    Look at pairwise "separability" according to the JensenShannonDivergence.

    For a classification problem, look at the maximum JSD that can exists
    across all features between pairs of classes.  This creates a binary
    comparison between individual classes instead of a OvA comparison done in
    JSScreen.
    """

    def __init__(self, js_bins=25, robust=False):
        """
        Instantiate the class.

        Parameters
        ----------
        js_bins : int
            Number of bins to use when computing the Jensen-Shannon
            divergence.
        robust : bool
            Whether to robust option for JensenShannonDivergence.
        """
        self.set_params(**{"js_bins": js_bins, "robust": robust})
        return

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {"js_bins": self.js_bins, "robust": self.robust}

    def fit(self, X, y):
        """
        Fit the screen to data.

        Parameters
        ----------
        X : array-like
            Features (columns) and observations (rows).
        y : array-like
            Ground truth classes.
        """
        js = JensenShannonDivergence(
            **{
                "per_class": True,  # Sorts by max automatically
                "feature_names": None,  # Index
                "bins": self.js_bins,
                "robust": self.robust,
            }
        )

        self.__enc_ = LabelEncoder()
        self.__enc_.fit(y)
        self.__matrix_ = np.zeros(
            (len(self.__enc_.classes_), len(self.__enc_.classes_))
        )
        self.__top_feature_ = np.empty(
            (len(self.__enc_.classes_), len(self.__enc_.classes_)), dtype=object
        )
        for pairs in itertools.combinations(np.unique(y), r=2):
            # 2. Compute (max) JS divergence
            mask = (y == pairs[0]) | (y == pairs[1])

            # Binary so divergences are the same, just take the first
            div = js.fit(X[mask], y[mask]).divergence
            x = div[pairs[0]][0][1][pairs[0]]
            feature = div[pairs[0]][0][0]
            assert div[pairs[1]][0][1][pairs[1]] == x

            i, j = self.__enc_.transform(pairs)
            self.__matrix_[i][j] = x
            self.__matrix_[j][i] = x
            self.__top_feature_[i][j] = feature
            self.__top_feature_[j][i] = feature

        return self

    @property
    def matrix(self):
        """Return the matrix of maximum JS divergence values."""
        return self.__matrix_.copy()

    def top_features(self, feature_names=None):
        """
        Return which feature was responsible for the max JS divergence.

        Parameters
        ----------
        feature_names : array-like
            List of feature names. Results are internally stored as
            indices so if this is provided, converts indices to names
            based on this array; otherwise a matrix of indices is
            returned.

        Example
        -------
        >>> jsb.top_features(feature_names=X.columns)
        """
        if feature_names is None:
            return self.__top_feature_.copy()
        else:
            names = np.empty_like(self.__top_feature_)
            for i in range(names.shape[0]):
                for j in range(names.shape[1]):
                    if i != j:
                        names[i, j] = feature_names[self.__top_feature_[i, j]]
                    else:
                        names[i, j] = "NONE"
            return names

    def visualize(self, ax=None):
        """Visualize the results with a heatmap."""
        if ax is None:
            ax = plt.figure().gca()

        ax = sns.heatmap(
            self.matrix,
            ax=ax,
            annot=True,
            xticklabels=self.__enc_.classes_,
            yticklabels=self.__enc_.classes_,
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(r"Maximum Pairwise $\nabla \cdot JS$")

        return ax


class JSScreen:
    """
    Use Jensen-Shannon divergences to screen for interesting features.

    For a classification problem, this uses JS divergences to
    combine classes in all possible ways to form "macroclasses."
    The JS divergence is then computed for all features when
    one class is the macroclass and all others are combined to
    form the opposing class (OvA method).

    This allows one to see if sets of classes can be separated from
    the rest of the "pack" according to certain features.  If
    so, this suggests that setting thresholds, or using trees,
    with those features could be an intuitive way to perform
    classification.

    Notes
    -----
    * This is a supervised method.
    * Using too many bins makes individual measurements all start to
    look unique and therefore 2 distributions appear to have a large
    JS divergence.  Be sure to try using a different number of bins
    to check your results qualitatively.  This also means outliers
    can be very problematic because they cause the the (max-min)
    range to be amplified artificially, which might actually make
    divergences look small because the bins are now too coarse.
    * See sklearn_ext.feature_selection.JensenShannonDivergence for
    more discussion on the potential importance/impact of class
    imbalance with respect to bin size.

    Example
    -------
    >>> screen = JSScreen(n=2, feature_names=X.columns)
    >>> screen.fit(X, y)
    >>> screen.visualize_grid(plt.figure(figsize=(20,20)).gca())
    """

    def __init__(self, n=None, feature_names=None, js_bins=25, robust=False):
        """
        Instantiate the class.

        Parameters
        ----------
        n : int or None
            Maximum macroclass size; will return all combinations
            up to the point of containing n atomic classes.  In
            None, goes from 1 to len(atomic_classes).
        feature_names : list(str)
            Names of features (columns of X) in order.
        js_bins : int
            Number of bins to use when computing the Jensen-Shannon
            divergence.
        robust : bool
            Whether or not use the robust option in JensenShannonDivergence.
        """
        self.set_params(
            **{
                "feature_names": np.array(feature_names, dtype=object),
                "n": n,
                "js_bins": js_bins,
                "robust": robust,
            }
        )
        return

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {
            "n": self.n,
            "feature_names": self.feature_names,
            "js_bins": self.js_bins,
            "robust": self.robust,
        }

    @staticmethod
    def macroclasses(atomic_classes, n):
        """
        Create macroclasses from individual, atomic ones.

        Paremeters
        ----------
        atomic_classes : array-like
            List of classes, can strings or integers, for example.
        n : int or None
            Maximum macroclass size; will return all combinations
            up to the point of containing n atomic classes.  In
            None, goes from 1 to len(atomic_classes).

        Returns
        -------
        list(tuple) : macro
            List of combinations of atomic classes in order of n,
            following Pascal's triangle.
        """
        if n is not None:
            assert n >= 1
        macro = {}
        for i in range(1, (len(atomic_classes) if n is None else n) + 1):
            macro[i] = [x for x in itertools.combinations(atomic_classes, i)]

        return macro

    @staticmethod
    def transform(y, macroclass, naming=None):
        """
        Transform classes into a macroclass.

        All entries are turned into strings during this process.

        Parameters
        ----------
        y : array-like
            List of ground-truth classes.
        macroclass : tuple
            Tuple of classes that belong to the macroclass being created.
        naming : callable
            Function to name combinations of atomic classes; None defaults
            to the JSScreen.merge() method.

        Returns
        -------
        macro : ndarray(str)
            Classes after merging atomic ones into the macroclass.
        """
        namer = JSScreen.merge if naming is None else naming
        string_macro = tuple([str(x) for x in macroclass])
        macro_name = namer(macroclass)
        y_macro = []
        for row in y:
            string_row = str(row)
            if string_row in string_macro:
                y_macro.append(macro_name)
            else:
                y_macro.append(string_row)
        y_macro = np.array(y_macro, dtype=str)

        return y_macro

    @staticmethod
    def merge(names, clause="AND", split=False):
        """Naming convention for merging classes."""
        if not clause.startswith(" "):
            clause = " " + clause
        if not clause.endswith(" "):
            clause = clause + " "
        if not split:
            # Merge together
            return clause.join(names)
        else:
            # Split apart
            return names.split(clause)

    def all_sets_(self, y, n):
        """
        Get all transformations of y into sets of [1:n].

        Parameters
        ----------
        y : array-like
            List of ground-truth classes.
        n : int or None
            Maximum macroclass size; will return all combinations
            up to the point of containing n atomic classes.  In
            None, goes from 1 to len(atomic_classes).

        Returns
        -------
        transforms : dict(dict)
            Dictionary of {n:{macroclass:y}}.
        """
        mc = self.macroclasses(np.unique(y), n)
        transforms = {}
        for k, v in mc.items():
            transforms[k] = {}
            for i, macro in enumerate(v):
                transforms[k][self.merge(macro)] = self.transform(y, macro)

        return transforms

    def fit(self, X, y):
        """
        Fit the screen to data.

        y is converted to a numpy array of strings automatically.

        Parameters
        ----------
        X : array-like
            Features (columns) and observations (rows).
        y : array-like
            Ground truth classes.
        """
        self.__X_ = np.array(X)
        self.__y_ = np.array(y, dtype=str)
        assert self.__X_.shape[0] == self.__y_.shape[0]
        self.__transforms_ = self.all_sets_(self.__y_, self.n)

        self.__js_ = JensenShannonDivergence(
            **{
                "per_class": False,
                "feature_names": None,  # Index
                "bins": self.js_bins,
                "robust": self.robust,
            }
        )

        self.__row_labels_ = (
            np.arange(X.shape[1])
            if self.feature_names is None
            else self.feature_names
        )  # Features are rows
        if self.feature_names is not None:
            assert len(self.feature_names) == self.__X_.shape[1]
        self.__column_labels_ = []  # Columns are macro-classes

        grid = []
        for n in tqdm.tqdm(self.__transforms_.keys()):
            for combine in tqdm.tqdm(self.__transforms_[n].keys()):
                self.__column_labels_.append(combine)
                y_ = self.__transforms_[n][combine]
                self.__js_.fit(self.__X_, y_)
                grid.append(
                    [
                        x[1]
                        for x in sorted(
                            {
                                a[0]: a[1][combine]
                                for a in self.__js_.divergence
                            }.items(),
                            key=lambda x: x[0],
                        )
                    ]
                )
        self.__grid_ = np.array(grid).T

        return self

    def visualize_grid(self, ax=None):
        """Visualize the results with a heatmap."""
        if ax is None:
            ax = plt.figure().gca()

        ax = sns.heatmap(
            self.__grid_,
            ax=ax,
            annot=True,
            xticklabels=self.__column_labels_,
            yticklabels=self.__row_labels_,
            cbar=False,
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(r"$\nabla \cdot JS$")

    def visualize_classes(self, method="max", ax=None, display=True):
        """Visualize the classes by summarizing over the features."""
        if display:
            if ax is None:
                ax = plt.figure().gca()

        if method == "mean":
            best = sorted(
                zip(
                    self.__column_labels_,
                    np.mean(self.__grid_, axis=0),
                    np.std(self.__grid_, axis=0),
                ),
                key=lambda x: x[1],
                reverse=True,
            )
        elif method == "max":
            best = sorted(
                zip(
                    self.__column_labels_,
                    np.max(self.__grid_, axis=0),
                    np.std(self.__grid_, axis=0),
                ),
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            raise ValueError("Unrecognized method")

        if display:
            ax.bar(
                x=[x[0] for x in best],
                height=[x[1] for x in best],
                yerr=[x[2] for x in best],
            )
            plt.xticks([x[0] for x in best], rotation=90)
            ax.set_title("Feature {} +/- 1 ".format(method) + r"$\sigma$")
            ax.set_ylabel(r"$\nabla \cdot JS$")

        return best

    def visualize_max(self, top=None, bins=25, ax=None):
        """
        Visualize the distribution of the max feature for classes.

        This will actually provide a visualization for all the top k
        macroclasses, so this is usually best when n=1 so only
        individual atomic classes are visualized.

        Example
        -------
        >>> screen = JSScreen(n=1, feature_names=X.columns, js_bins=25)
        >>> screen.fit(X, y)
        >>> screen.visualize_max()
        """
        import copy

        best = self.visualize_classes(method="max", ax=None, display=False)
        if top is None:
            top = len(best)

        top_feature = list(
            zip(
                self.__column_labels_,
                self.__row_labels_[np.argmax(self.__grid_, axis=0)],
            )
        )

        if ax is None:
            fig, axes = plt.subplots(nrows=top, ncols=1)
            axes = axes.ravel()
        else:
            try:
                iter(ax)
                axes = ax.ravel()
            except:
                axes = [ax]

        best_dict = {a: b for a, b, c in best}
        feat_dict = dict(top_feature)
        for ax_, (class_, _) in list(
            zip(
                axes,
                sorted(best_dict.items(), key=lambda x: x[1], reverse=True),
            )
        )[:top]:
            X_binary = pd.DataFrame(data=self.__X_, columns=self.feature_names)
            y_ = copy.copy(self.__y_).astype(object)
            for c in self.merge(class_, split=True):
                y_[self.__y_ == c] = class_
            y_[y_ != class_] = "OTHER"
            X_binary["class"] = y_
            ax_ = sns.histplot(
                hue="class",
                x=feat_dict[class_],
                data=X_binary,
                # multiple='stack',
                palette="Set1",
                stat="probability",
                bins=bins,
                common_norm=False,
                ax=ax_,
            )
            ax_.set_title(
                class_
                + r"; $\nabla \cdot JS = {}$".format("%.3f" % best_dict[class_])
            )

    @property
    def grid(self):
        """Get the grid of Jensen-Shannon divergences computed."""
        return self.__grid_.copy()

    def interesting(self, threshold=0.7, method="max", min_delta=0.0):
        """
        Try to find the "interesting" macroclasses.

        In this example, we define "interesting merges" as those which cause a
        positive delta of at least `min_delta` and raise the JS divergence to
        above some `threshold` where it was initially below. Moreover, all the
        individual classes must have divergences less than the net of all of
        them less `min_delta` (i.e., merging is exclusively increasing the
        distinguishibility of the macroclass rather than one simply "bringing
        up the average").

        Because the divergences must be low for the atomic classes, it can
        happen that this proposes (B,C,D) as a class (whose complement is A)
        but not (A,) directly; this ie because B,C,D may overlap each other
        and so have low JS divergences, while A may be easily separable to
        begin with so it fails that check.  Ultimately, the result is the same
        but it might seem counterintuitive that this does not always propose
        "symmetric" suggestions.

        Returns
        -------
        incremental, proposed_combinations : list([tuple(macroclass, addition),
        {'delta':change, 'final':JS, 'individuals':{class:JS}}]), dict(set))
            Merges that are considered "interesting", dictionary of unique sets
            formed from these merges.
        """
        interest = []
        for row in self.incremental(method=method):
            if (
                (row[1]["final"] > threshold)
                and (row[1]["final"] - row[1]["delta"]) < threshold
                and (row[1]["delta"] > min_delta)
                and np.all(
                    np.array(list(row[1]["individuals"].values()))
                    < row[1]["final"] - min_delta
                )
            ):
                interest.append(row)

        proposed_combinations = {}
        performances = {}
        idx = 0
        for row in interest:
            union = set(self.merge(row[0][0], split=True)).union({row[0][1]})
            add = True
            for k, v in proposed_combinations.items():
                if v == union:
                    add = False
                    break
            if add:
                proposed_combinations[idx] = union
                performances[idx] = row[1]["final"]
                idx += 1

        return interest, proposed_combinations

    def incremental(self, method="max"):
        """
        Find the changes due to the addition of a single class to a macroclass.

        For each macroclass, the difference of the method (max or mean) of the
        JS divergences is computed when one atomic class is added to the
        macroclass.  Postive values imply the new set is higher than before the
        class has been added; negative means adding the new class decreases the
        JS divergence (max or mean).

        The point at which a large positive occurs, suggests that newly formed
        macroclass represents a cluster that is separate from the other classes
        that is now complete.  A large negative change indicates that you have
        jsut added a class that overlaps the classes the in the macroclass; the
        larger the change, the more significant the overlap (i.e. maybe more of
        the constituent classes.

        Parameters
        ----------
        method : str
            Use the 'max' or the 'mean' of the JS divergences as the metric.

        Returns
        -------
        incremental : list([tuple(macroclass, addition), {'delta':change,
        'final':JS, 'individuals':{class:JS}}])
            The change that results from merging "addition" with macroclass,
            sorted from highest to lowest.  Note that this is a signed change,
            so you may wish to consider the magnitude instead.  The 'final' JS
            is the JS divergence of the new macroclass = macroclass + addition.
            The JS divergence for all the atomic classes is given in
            'individuals' for comparison.
        """
        if method == "max":
            function = np.max
        elif method == "mean":
            function = np.mean

        d = {}
        k = {}
        for j, combination in enumerate(self.__column_labels_):
            k[j] = set(self.merge(combination, split=True))
            d[j] = function(self.__grid_[:, j])

        def find(set_):
            for j, v in k.items():
                if v == set_:
                    return j
            raise ValueError("Could not find the set in question.")

        # Find which single additions resulted in the greatest "jumps"
        incremental = {}
        for j in d.keys():
            if len(k[j]) > 1:
                for x in k[j]:
                    idx = find(k[j].difference({x}))
                    delta = d[j] - d[idx]  # Find the value if x was removed
                    incremental[(self.__column_labels_[idx], x)] = {
                        "delta": delta,
                        "final": d[j],
                        "individuals": {c: d[find({c})] for c in k[j]},
                    }

        return sorted(
            incremental.items(), key=lambda x: x[1]["delta"], reverse=True
        )
