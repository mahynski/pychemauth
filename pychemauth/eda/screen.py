"""
Screening tools for features of data.

author: nam
"""

import inspect
import itertools
import warnings
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y

from pychemauth.preprocessing.feature_selection import JensenShannonDivergence

from typing import Any, Callable, Union, Sequence, Iterable, ClassVar
from numpy.typing import NDArray


class RedFlags:
    """
    Check dense, tabular data for any obvious "red flags".

    Parameters
    ----------
    use : list(str), optional(default=None)
        Name of checks to use; if None then run all.

    tag : str, optional(default="")
        Optional string to prepend to any warnings.

    Note
    ----
    This check returns warnings for each possible issue that it finds. If no
    warnings are issued then the data may be considered "clean".

    Example
    -------
    >>> r = RedFlags(tag="Checks for this class")
    >>> r.all_checks # Get dictionary of all checks possible
    >>> r.get_checks # Get dictionary of all checks that we are going to run
    >>> r.run(X, y)
    """

    def __init__(self, use: Union[list, None] = None, tag: str = "") -> None:
        """Instantiate the class."""
        self.perform = {}
        self.all_checks = dict(
            [f for f in inspect.getmembers(self) if f[0].startswith("check_")]
        )
        self.tag = tag

        if use:
            assert isinstance(use, list)
            for t in use:
                if t in self.all_checks:
                    self.perform[t] = self.all_checks[t]
        else:
            self.perform = self.all_checks

    @property
    def get_checks(self) -> dict[Any, Any]:
        """Get a list of all checks that will be performed."""
        return self.perform

    def run(
        self,
        X: Union[
            NDArray[np.floating],
            NDArray[np.integer],
            Sequence[Sequence[float]],
            Sequence[Sequence[float]],
        ],
        y: Union[Sequence[Any], NDArray[Any]],
    ) -> None:
        """
        Run all checks.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, int, or str, ndim=1)
            Target to predict.

        Raises
        ------
        warnings.warn if any check fails.

        Note
        ----
        `X` may be an object that contains columns of different types of data (e.g., str, bool, floats)
        while `y` (if provided) must be either completely composed of floats or strings.
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
                        np.float_,
                        np.float32,
                        np.float64,
                        np.floating,
                        int,
                        np.int_,
                        np.uint,
                        np.int32,
                        np.int64,
                        np.integer,
                    ),
                )
                for item in y
            ]
        ):
            self.y_type = "float"
        elif all([isinstance(item, str) for item in y]):
            self.y_type = "str"
        else:
            raise Exception(
                "y contains mixed types (e.g., some floats some strings)"
            )

        # Leave X open to have bools, floats, etc. in different columns and
        # allow each check to perform the relevant examination.
        for name, test in self.perform.items():
            test(X=X_safe, y=y_safe)

    def check_nan(
        self,
        X: Union[
            NDArray[np.floating],
            NDArray[np.integer],
            Sequence[Sequence[float]],
            Sequence[Sequence[float]],
        ],
        y: Union[Sequence[Any], NDArray[Any]],
    ) -> bool:
        """
        Check if any entries in X or y are NaN.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, int, or str, ndim=1)
            Target to predict. If these are floating points, also check for NaN values.

        Returns
        -------
        found : bool
            If there are any NaN values in `X` or `y`.
        """
        found = False
        if np.any(np.isnan(np.asarray(X, dtype=np.float64).flatten())):
            warnings.warn(
                "{} : X contains NaN values; this will require imputation".format(
                    self.tag
                )
            )
            found = True

        if not (y is None):
            if self.y_type == "float":
                if np.any(np.isnan(np.asarray(y, dtype=np.float64).flatten())):
                    warnings.warn("{} : y contains NaN values".format(self.tag))
                    found = True
        return found

    def check_inf(
        self,
        X: Union[
            NDArray[np.floating],
            NDArray[np.integer],
            Sequence[Sequence[float]],
            Sequence[Sequence[float]],
        ],
        y: Union[Sequence[Any], NDArray[Any]],
    ) -> bool:
        """
        Check if any entries in X or y are Inf.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, int, or str, ndim=1)
            Target to predict. If these are floating points, also check for Inf values.

        Returns
        -------
        found : bool
            If there are any Inf values in `X` or `y`.
        """
        found = False
        if np.any(np.isinf(np.asarray(X, dtype=np.float64).flatten())):
            warnings.warn(
                "{} : X contains Inf values; this will require imputation".format(
                    self.tag
                )
            )
            found = True

        if not (y is None):
            if self.y_type == "float":
                if np.any(np.isinf(np.asarray(y, dtype=np.float64).flatten())):
                    warnings.warn("{} : y contains Inf values".format(self.tag))
                    found = True
        return found

    def check_zero_variance(
        self,
        X: Union[
            NDArray[np.floating], NDArray[np.integer], Sequence[Sequence[float]]
        ],
        y=None,
    ) -> bool:
        """
        Check if any columns in X are constant (unsupervised).

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, int, or str, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        found : bool
            If any columns of `X` have zero variance.
        """
        tol = 1.0e-12
        std_dev = np.std(np.asarray(X, dtype=np.float64), axis=0)
        if np.any(std_dev < tol):
            warnings.warn(
                "{} : X columns with no variance: {}".format(
                    self.tag, np.where(std_dev < tol)[0]
                )
            )
            return True
        return False

    def check_min_observations(
        self,
        X: Union[
            NDArray[np.floating], NDArray[np.integer], Sequence[Sequence[float]]
        ],
        y=None,
        n: int = 5,
    ) -> bool:
        r"""
        Check each class has a minimum number of observations.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(str or float, ndim=1), optional(default=None)
            Target to predict. If `None` this test is not performed.

        n : scalar(int), optional(default=5)
            Minimum number of observations to expect, else throw warning.

        Returns
        -------
        found : bool
            If any classes have less than `n` observations.
        """
        found = False
        if not (y is None):
            y_ = np.asarray(y)
            X_ = np.asarray(X, dtype=np.float64)
            if y_.ndim == 1:
                for c in np.unique(y_):
                    n_obs = np.sum(y_ == c)
                    if n_obs < n:
                        warnings.warn(
                            "{} : Class {} only contains {} observations".format(
                                self.tag, c, n_obs
                            )
                        )
                        found = True
        return found

    def check_min_different_values(
        self,
        X: Union[
            NDArray[np.floating], NDArray[np.integer], Sequence[Sequence[float]]
        ],
        y=None,
        n: int = 5,
    ) -> bool:
        r"""
        Check each class has a minimum number of unique values in each column of X.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, int, or str, ndim=1), optional(default=None)
            Target to predict. If `None` this test is not performed.

        n : scalar(int), optional(default=5)
            Minimum number of different values to expect, else throw warning.

        Returns
        -------
        found : bool
            If any classes have less than `n` unique values in any column of `X`.

        Note
        ----
        This is important during CV which might split, e.g., a bimodal distribution up so that all observations in the train split(s) are the same, leading to a std = 0, causing standardization to "explode."  As a result, it is recommended that `n` be at least `k` in k-fold CV, but should generally be more.
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
                                "{} : Class {} (n={}) only contains only {} different observations for feature (column) index {}".format(
                                    self.tag,
                                    c,
                                    np.sum(y_ == c),
                                    len(n_unique[column][0]),
                                    column,
                                )
                            )
                            found = True

        return found

    def check_duplicates(
        self,
        X: Union[
            NDArray[np.floating], NDArray[np.integer], Sequence[Sequence[float]]
        ],
        y=None,
        tol: float = 1.0e-12,
    ) -> bool:
        """
        Check if any rows in X are duplicates numerically.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, int, or str, ndim=1), optional(default=None)
            Ignored.

        tol : scalar(float), optional(default=1.0e-12)
            Minimum Euclidean distance between rows to avoid throwing warning.

        Returns
        -------
        found : bool
            If any rows of `X` are duplicates.
        """
        try:
            if np.any(
                scipy.spatial.distance.pdist(
                    np.asarray(X, dtype=np.float64), metric="euclidean"
                )
                < tol
            ):
                warnings.warn(
                    "{} : There are duplicate rows in X".format(self.tag)
                )
                return True
            else:
                return False
        except:
            warnings.warn(
                "{} : Unable to perform duplicate check - there is probably a NaN or Inf value present".format(
                    self.tag
                )
            )
            return False

    def check_zero_class_variance(
        self,
        X: Union[
            NDArray[np.floating], NDArray[np.integer], Sequence[Sequence[float]]
        ],
        y=None,
    ) -> bool:
        """
        Check if columns in X are constant for any classes.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, int, or str, ndim=1), optional(default=None)
            Target to predict. If `None` this test is not performed.

        Returns
        -------
        found : bool
            If any classes have columns of `X` with zero variance.

        Notes
        -----
        For models like SIMCA, which break things up based on class, this is particularly important.

        This test is skipped if NaN or Inf are found in `X`.
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
                            "{} : X[{}] contains columns with no variance: {}".format(
                                self.tag, c, np.where(std_dev < tol)[0]
                            )
                        )
                        found = True
        return found


class JSScreen:
    """
    Use Jensen-Shannon divergences to screen for interesting features.

    Parameters
    ----------
    n : scalar(int), optional(default=None)
        Maximum macroclass size; will return all combinations
        up to the point of containing n atomic classes.  In
        None, goes from 1 to len(atomic_classes).

    feature_names : list(str), optional(default=None)
        Names of features (columns of X) in order.

    js_bins : scalar(int), optional(default=25)
        Number of bins to use when computing the Jensen-Shannon
        divergence.

    robust : bool, optional(default=False)
        Whether or not use the robust option in JensenShannonDivergence.

    Note
    ----
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

    * This is a supervised method.

    * Using too many bins makes individual measurements all start to
    look unique and therefore 2 distributions appear to have a large
    JS divergence.  Be sure to try using a different number of bins
    to check your results qualitatively.  This also means outliers
    can be very problematic because they cause the the (max-min)
    range to be amplified artificially, which might actually make
    divergences look small because the bins are now too coarse.

    * See `pychemauth.preprocessing.feature_selection.JensenShannonDivergence`
    for more discussion on the potential importance/impact of class
    imbalance with respect to bin size.

    Example
    -------
    >>> screen = JSScreen(n=2, feature_names=X.columns)
    >>> screen.fit(X, y)
    >>> screen.visualize_grid(plt.figure(figsize=(20,20)).gca())
    """

    feature_names: ClassVar[Union[list, NDArray[np.str_], None]]
    n: ClassVar[Union[int, None]]
    js_bins: ClassVar[int]
    robust: ClassVar[bool]

    def __init__(
        self,
        n: Union[int, None] = None,
        feature_names: Union[list, NDArray[np.str_], None] = None,
        js_bins: int = 25,
        robust: bool = False,
    ) -> None:
        """Instantiate the class."""
        self.set_params(
            **{
                "feature_names": np.array(feature_names, dtype=object),
                "n": n,
                "js_bins": js_bins,
                "robust": robust,
            }
        )
        return

    def set_params(self, **parameters: Any) -> "JSScreen":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "n": self.n,
            "feature_names": self.feature_names,
            "js_bins": self.js_bins,
            "robust": self.robust,
        }

    @staticmethod
    def macroclasses(
        atomic_classes: Union[
            Sequence[int], Sequence[str], NDArray[np.integer], NDArray[np.str_]
        ],
        n: Union[int, None],
    ) -> dict[int, list[tuple]]:
        """
        Create macroclasses from individual, atomic ones.

        Parameters
        ----------
        atomic_classes : array_like(str or int, ndim=1)
            List of classes, can strings or integers, for example.

        n : scalar(int) or None
            Maximum macroclass size; will return all combinations up to the point of containing n atomic classes.  If `None`, goes from 1 to len(`atomic_classes`).

        Returns
        -------
        macro : dict(int, list(tuple))
            List of combinations of atomic classes in order of `n`, following Pascal's triangle.
        """
        if n is not None:
            assert n >= 1
        macro = {}
        for i in range(1, (len(atomic_classes) if n is None else n) + 1):
            macro[i] = [x for x in itertools.combinations(atomic_classes, i)]

        return macro

    @staticmethod
    def transform(
        y: Union[
            Sequence[int], Sequence[str], NDArray[np.integer], NDArray[np.str_]
        ],
        macroclass: tuple,
        naming: Union[Callable[..., str], None] = None,
    ) -> NDArray[np.str_]:
        """
        Transform classes into a macroclass.

        Parameters
        ----------
        y : array_like(str or int, ndim=1)
            Ground-truth classes.

        macroclass : tuple(str)
            Tuple of classes that belong to the macroclass being created.

        naming : callable, optional(default=None)
            Function to name combinations of atomic classes; None defaults
            to the JSScreen.merge() method.

        Returns
        -------
        macro : ndarray(str, ndim=1)
            Classes after merging atomic ones into the macroclass.

        Note
        ----
        All entries are turned into strings during this process.
        """
        namer = JSScreen.merge if naming is None else naming
        string_macro = tuple([str(x) for x in macroclass])
        macro_name = namer(macroclass)  # type: ignore[operator]
        y_macro_ = []
        for row in y:
            string_row = str(row)
            if string_row in string_macro:
                y_macro_.append(macro_name)
            else:
                y_macro_.append(string_row)
        y_macro = np.array(y_macro_, dtype=np.str_)

        return y_macro

    @staticmethod
    def merge(
        names: Union[str, Iterable[str]],
        clause: str = "AND",
        split: bool = False,
    ) -> Union[str, list[str]]:
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
            return names.split(clause)  # type: ignore[union-attr]

    def _all_sets(
        self,
        y: Union[
            Sequence[int], Sequence[str], NDArray[np.integer], NDArray[np.str_]
        ],
        n: Union[int, None],
    ) -> dict[int, dict]:
        """
        Get all transformations of y into sets of size [1:n].

        Parameters
        ----------
        y : array_like(str or int, ndim=1)
            Ground-truth classes.

        n : scalar(int) or None
            Maximum macroclass size; will return all combinations up to the point of containing `n` atomic classes.  If `None`, goes from 1 to len(`atomic_classes`).

        Returns
        -------
        transforms : dict(int, dict)
            Dictionary of {n:{macroclass:y}}.
        """
        mc = self.macroclasses(np.unique(y), n)
        transforms: dict = {}
        for k, v in mc.items():
            transforms[k] = {}
            for i, macro in enumerate(v):
                transforms[k][self.merge(macro)] = self.transform(y, macro)

        return transforms

    def fit(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.floating]],
        y: Union[
            Sequence[int], Sequence[str], NDArray[np.integer], NDArray[np.str_]
        ],
    ) -> "JSScreen":
        """
        Fit the screen to data.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(str or int, ndim=1)
            Ground truth classes.

        Result
        ------
        self : JSScreen
            Fitted model.

        Note
        ----
        `y` is converted to a numpy array of strings automatically.
        """
        self.__X_ = np.array(X)
        self.__y_ = np.array(y, dtype=str)
        assert self.__X_.shape[0] == self.__y_.shape[0]
        self.__transforms_ = self._all_sets(self.__y_, self.n)

        self.__js_ = JensenShannonDivergence(
            **{
                "per_class": False,
                "feature_names": None,  # Index
                "bins": self.js_bins,
                "robust": self.robust,
            }
        )

        self.__row_labels_ = (
            np.arange(self.__X_.shape[1])
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

    def visualize_grid(
        self, ax: Union[matplotlib.pyplot.Axes, None] = None
    ) -> matplotlib.pyplot.Axes:
        """
        Visualize the results with a heatmap.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes, optional(default=None)
            Axes to plot the result on.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Axes results are plotted on.
        """
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

        return ax

    def visualize_classes(
        self,
        method: str = "max",
        ax: Union[matplotlib.pyplot.Axes, None] = None,
        display: bool = True,
    ) -> list:
        """
        Visualize the classes by summarizing over the features.

        Parameters
        ----------
        method : str, optional(default="max")
            How to determine the "best" results; must be "mean" or "max".

        ax : matplotlib.pyplot.axes, optional(default=None)
            Axes to plot the results on.

        display : bool, optional(default=True)
            Whether to plot the results or not.

        Returns
        -------
        best : list(str, float, float)
            Tuple of columns sorted in descending order based on the "best" score, the score itself, and the standard deviation.
        """
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
            ax.bar(  # type: ignore[union-attr]
                x=[x[0] for x in best],
                height=[x[1] for x in best],
                yerr=[x[2] for x in best],
            )
            plt.xticks([x[0] for x in best], rotation=90)
            ax.set_title("Feature {} +/- 1 ".format(method) + r"$\sigma$")  # type: ignore[union-attr]
            ax.set_ylabel(r"$\nabla \cdot JS$")  # type: ignore[union-attr]

        return best

    def visualize_max(
        self,
        top: Union[int, None] = None,
        bins: int = 25,
        ax: Union[matplotlib.pyplot.Axes, None] = None,
    ) -> NDArray[matplotlib.pyplot.Axes]:
        r"""
        Visualize the distribution of the max feature for classes.

        Parameters
        ----------
        top : scalar(int), optional(default=None)
            The number of top macroclasses to visualize. If None then show all.

        bins : scalar(int), optional(default=25)
            Number of bins to use in the histogram.

        ax : array_like(matplotlib.pyplot.Axes, ndim=1), optional(default=None)
            Axes to plot each macroclasses on. Should have length of :math:`top`.

        Returns
        -------
        ax : ndarray(matplotlib.pyplot.Axes, ndim=1)
            Array of axes results are plotted on.

        Note
        ----
        This will actually provide a visualization for all the top macroclasses, so this is usually best when `n`=1 so only individual atomic classes are visualized.

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
                axes = np.array([ax])

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

        return axes

    @property
    def grid(self):
        """Get the grid of Jensen-Shannon divergences computed."""
        return self.__grid_.copy()

    def interesting(
        self,
        threshold: float = 0.7,
        method: str = "max",
        min_delta: float = 0.0,
    ) -> tuple[list, dict]:
        r"""
        Try to find the "interesting" macroclasses.

        Parameters
        ----------
        threshold : scalar(float), optional(default=0.7)
            The JSD value a set of (macro)classes must be below before merging, but after which they are above is considered "interesting."

        method : str, optional(default="max")
            Use the "max" or the "mean" of the JS divergences as the metric.

        min_delta : scalar(float), optional(default=0.0)
            Minimum amount the JSD must be raised after merging (macro)classes to be considered "interesting."

        Returns
        -------
        interesting : list(tuple(str, str), dict(str:float))
            Summary of incremental changes that meet the "interesting" criteria. See :py:func:`JSScreen.incremental`.

        proposed_combinations : dict(set)
            Merges that are considered "interesting", dictionary of unique sets formed from these merges.

        Note
        ----
        We define "interesting merges" as those which cause a positive change of at least `min_delta` and raise the JS divergence to above some `threshold` where it was initially below. Moreover, all the individual classes must have divergences less than the net of all of them less `min_delta` (i.e., merging is exclusively increasing the distinguishibility of the macroclass rather than one simply "bringing up the average").

        Because the divergences must be low for the atomic classes, it can happen that this proposes (B,C,D) as a class (whose complement is A) but not (A,) directly; this ie because B,C,D may overlap each other and so have low JS divergences, while A may be easily separable to begin with so it fails that check.  Ultimately, the result is the same but it might seem counterintuitive that this does not always propose "symmetric" suggestions.
        """
        interesting = []
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
                interesting.append(row)

        proposed_combinations: dict = {}
        performances: dict = {}
        idx = 0
        for row in interesting:
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

        return interesting, proposed_combinations

    def incremental(self, method: str = "max") -> list:
        """
        Find the changes due to the addition of a single class to a macroclass.

        Parameters
        ----------
        method : str, optional(default="max")
            Use the "max" or the "mean" of the JS divergences as the metric.

        Returns
        -------
        incremental : list(tuple(str, str), dict(str:float))
            Summary of incremental changes given as list([tuple(macroclass, addition),
            {"delta":change, "final":JS, "individuals":{class:JS}}]). This reflects
            the change that results from merging "addition" with macroclass,
            sorted from highest to lowest.  Note that this is a signed change,
            so you may wish to consider the magnitude instead.  The "final" JS
            is the JS divergence of the new macroclass = macroclass + addition.
            The JS divergence for all the atomic classes is given in
            "individuals" for comparison.

        Note
        ----
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
        """
        if method == "max":
            function_ = np.max
        elif method == "mean":
            function_ = np.mean  # type: ignore[assignment]
        else:
            raise ValueError("Unrecognized method.")

        d = {}
        k = {}
        for j, combination in enumerate(self.__column_labels_):
            k[j] = set(self.merge(combination, split=True))
            d[j] = function_(self.__grid_[:, j])

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


class JSBinary:
    """
    Look at pairwise "separability" according to the Jensen-Shannon divergence.

    Parameters
    ----------
    js_bins : scalar(int), optional(default=25)
        Number of bins to use when computing the Jensen-Shannon divergence.

    robust : bool, optional(default=False)
        Whether to robust option for JensenShannonDivergence.

    Note
    ----
    For a classification problem, look at the maximum JSD that can exists across all features between pairs of classes.  This creates a binary comparison between individual classes instead of a one-vs-all comparison done in JSScreen.

    It can be helpful to look for the "elbow" as you plot number of bins vs. max JSD to get a sense for the optimal value.
    """

    js_bins: ClassVar[int]
    robust: ClassVar[bool]

    def __init__(self, js_bins: int = 25, robust: bool = False) -> None:
        """Instantiate the class."""
        self.set_params(**{"js_bins": js_bins, "robust": robust})

    def set_params(self, **parameters: Any) -> "JSBinary":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {"js_bins": self.js_bins, "robust": self.robust}

    def fit(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.floating]],
        y: Union[
            Sequence[int], Sequence[str], NDArray[np.integer], NDArray[np.str_]
        ],
    ) -> "JSBinary":
        """
        Fit the screen to data.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Features matrix.

        y : array_like(str or int, ndim=1)
            Ground truth classes.

        Returns
        -------
        self : JSBinary
            Fitted model.
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
    def matrix(self) -> NDArray[np.floating]:
        """Return the matrix of maximum JS divergence values."""
        return self.__matrix_.copy()

    def top_features(
        self, feature_names: Union[Sequence[Any], NDArray[Any], None] = None
    ) -> NDArray[Any]:
        """
        Return which feature was responsible for the max JS divergence.

        Parameters
        ----------
        feature_names : array_like(str, ndim=1), optional(default=None)
            List of feature names. Results are internally stored as
            indices so if this is provided, converts indices to names
            based on this array; otherwise a matrix of indices is
            returned.

        Returns
        -------
        top_features : ndarray(object, ndim=2)
            Matrix of top feature names (if provided) or indices indicating
            the feature responsible for the maximum JSD between features i and
            j.  Diagonals are set to "NONE".

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

    def visualize(
        self, ax: Union[matplotlib.pyplot.Axes, None] = None
    ) -> matplotlib.pyplot.Axes:
        """
        Visualize the results with a heatmap.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes, optional(default=None)
            Axes to plot the result on.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Axes results are plotted on.
        """
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
