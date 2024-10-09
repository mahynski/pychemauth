"""
Compare ML pipelines.

author: nam
"""
import math
import warnings
import sklearn
import sklearn.model_selection
import imblearn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from baycomp import two_on_single
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    GroupKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)

from typing import Union, Sequence, Any, ClassVar, Generator
from numpy.typing import NDArray, ArrayLike


class _RepeatedGroupKFold:
    """Repeat (Stratified)GroupKFold a number of times."""

    n_splits: ClassVar[int]
    n_repeats: ClassVar[int]
    random_state: ClassVar[Union[int, None]]
    stratified: ClassVar[bool]

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Union[int, None] = None,
        stratified: bool = False,
    ) -> None:
        """
        Perform (Stratified)GroupKFold a number of times.

        Parameters
        ----------
        n_splits : int, optional(default=5)
            Number of splits for GroupKFold.

        n_repeats : int, optional(default=10)
            Number of times to repeat the GroupKFold.

        random_state : int, optional(default=None)
            Random state which controls how data is initially shuffled and split to create different GroupKFold results.

        stratified : bool, optional(default=False)
            Whether to attempt to stratify the results.
        """
        self.set_params(
            **{
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "random_state": random_state,
                "stratified": stratified,
            }
        )

    def set_params(self, **parameters: Any) -> "_RepeatedGroupKFold":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splitting iterations in the cross-validator.
        """
        return self.n_splits * self.n_repeats

    def split(
        self,
        X: ArrayLike,
        y: Union[ArrayLike, None] = None,
        groups: Union[ArrayLike, None] = None,
    ) -> Generator[tuple[NDArray[np.integer], NDArray[np.integer]]]:
        """
        Generate indices to split data into training and test set.

        Step 1:
        The data initially broken up by a simple (outer) KFold split with max(`n_splits`, `n_repeats`) splits.  This creates several datasets with varied group structure.  The train set is selected for Step 2.

        Step 2:
        (Stratified)GroupKFold is performed on each outer training set.  GroupKFold is a deterministic operation which is why Step 1 is required; even if stratified the randomness is somewhat limited.  Instead, this operation breaks up certain groups into the inner test and inner train folds.

        Step 3:
        The inner datasets form the basis of what is returned, but they are only a subset of the total data provided due to the split in Step 1.  To remedy this, the outer test data is added to the inner splits based on their group so that the group structure determined by the inner (Stratified)GroupKFold split is maintained.

        Parameters
        ----------
        X : array-like
            Training data.

        y : array-like
            The target variable for supervised learning problems.

        groups : array-like
            Group labels for the samples used while splitting the dataset into train/test set.

        Yields
        ------
        train : ndarray(int)
            The training set indices for that split.

        test : ndarray(int)
            The testing set indices for that split.
        """
        X_ = np.asarray(X)
        y_ = np.asarray(y)

        if groups is None:
            raise ValueError(
                f"groups must be specified for {self.__class__.__name__}"
            )
        else:
            groups_ = np.asarray(groups)

        """
        Step 1.
        Use CV to randomly split up the dataset initially - this serves as the seed for each of the repeats.
        GroupKFold has no randomness to it so that needs to be introduced via data splitting in the outer loop.
        Using max([n_splits, n_repeats]) avoid issues with small values of n_repeats.
        """
        outer = KFold(
            n_splits=np.max([self.n_splits, self.n_repeats]),
            random_state=self.random_state,
            shuffle=True,
        )

        for n_repeat, (split_index, hold_index) in enumerate(
            outer.split(X_, y_)
        ):
            # Number of outer splits may exceed n_repeats to terminate when appropriate.
            if n_repeat >= self.n_repeats:
                break

            X_split, _ = X_[split_index], X_[hold_index]
            y_split, _ = y_[split_index], y_[hold_index]
            groups_split, groups_hold = (
                groups_[split_index],
                groups_[hold_index],
            )

            """
            Step 2.
            Based on this training dataset make another split that respects the group structure.
            GroupKFold has no randomness to it so that needs to be introduced via data splitting in the outer loop.
            StratifiedGroupKFold has some, but just changing the RNG seed in this does introduce as much noise/variance as this splitting does, which is deemed prefereable.
            Each group will appear exactly once in the test set across all folds (the number of distinct groups has to be at least equal to the number of folds).
            """
            inner = (
                StratifiedGroupKFold(
                    n_splits=self.n_splits, random_state=n_repeat, shuffle=True
                )
                if self.stratified
                else GroupKFold(n_splits=self.n_splits)
            )
            for i, (train_idx, test_idx) in enumerate(
                inner.split(X_split, y_split, groups_split)
            ):
                """
                Step 3.
                Take the inner split as the basis of the datasets.
                Then assign the points from the outer held out fold to test/train based on their group.
                Groups were assigned based on the inner split's decision.
                """
                global_train_idx = split_index[train_idx]
                global_test_idx = split_index[test_idx]
                for idx_, g_ in zip(hold_index, groups_hold):
                    if g_ in set(groups_split[test_idx]):
                        np.append(global_test_idx, idx_)
                    else:
                        np.append(global_train_idx, idx_)

                yield global_train_idx, global_test_idx


class RepeatedGroupKFold(_RepeatedGroupKFold):
    """Repeat GroupKFold a number of times."""

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Union[int, None] = None,
    ) -> None:
        """
        Perform GroupKFold a number of times.

        Parameters
        ----------
        n_splits : int, optional(default=5)
            Number of splits for GroupKFold.

        n_repeats : int, optional(default=10)
            Number of times to repeat the GroupKFold.

        random_state : int, optional(default=None)
            Random state which controls how data is initially shuffled and split to create different GroupKFold results.
        """
        super().__init__(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
            stratified=False,
        )


class RepeatedStratifiedGroupKFold(_RepeatedGroupKFold):
    """Repeat StratifiedGroupKFold a number of times."""

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Union[int, None] = None,
    ) -> None:
        """
        Perform StratifiedGroupKFold a number of times.

        Parameters
        ----------
        n_splits : int, optional(default=5)
            Number of splits for GroupKFold.

        n_repeats : int, optional(default=10)
            Number of times to repeat the GroupKFold.

        random_state : int, optional(default=None)
            Random state which controls how data is initially shuffled and split to create different GroupKFold results.
        """
        super().__init__(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
            stratified=True,
        )


class BiasedNestedCV:
    """
    Perform nested GridSearchCV and get all validation fold scores.

    Parameters
    ----------
    k_inner : scalar(int), optional(default=10)
        K-fold for inner loop.

    k_outer : scalar(int), optional(default=10)
        K-fold for outer loop.

    Note
    ----
    This performs a version of repeated "flat CV" as defined in [1] to statistically compare the performance of two pipelines or models.  This is done by nesting flat CV inside an outer loop, akin to nested CV. This differs from scikit-learn's "built in" method of doing nested CV of cross_val_score(GridSeachCV()) in that cross_val_score() returns the scores from each test fold on the outer loop, obtained by training the best model on the entire inner set of training folds using the best hyperparameters determined by that the inner loop.  This is conventional "Nested CV" which uses the performance on the held-out test fold for an unbiased esimate of the generalization error [2].

    Here, we instead extract the performance estimates from the best model's validation fold (held out part of the inner training folds) used to select the hyperparameters for a given iteration of the outer loop.  These performance estimates are positively biased, but extensive real-world testing suggests that using the scores from the "flat CV" to **rank** different models does not make practical difference [2] to their relative ordering. Therefore, we can use the inner scores directly in concert with paired t-tests to compare different pipelines when it is too computationally expensive to do repeated nested CV.

    Unlike simple repetition of any CV procedure, framing this as a nested CV uses the outer loop as way to "shift" or decorrelate the data chunks more relative to simply reshuffling and reusing the data.  Thus, the Nadeau and Bengio's correction factor [3] for the t-test is less necessary, but further aids in making conservative inference.

    References
    ----------
    [1] Wainer, J. and Cawley G., "Nested cross-validation when selecting classifiers is overzealous for most practical applications," Expert Systems with Applications 182, 115222 (2021).

    [2] Cawley, G.C. and Talbot, N.L.C., "On over-fitting in model selection and subsequent selection bias in performance evaluation," J. Mach. Learn. Res. 11, 2079-2107 (2010).

    [3] Nadeau, C. and Bengio Y., "Inference for the Generalization Error," Machine Learning 52, 239-281 (2003).
    """

    def __init__(self, k_inner: int = 10, k_outer: int = 10) -> None:
        """Instantiate the class."""
        self.__k_inner = k_inner
        self.__k_outer = k_outer

    def _get_test_scores(
        self, gs: sklearn.model_selection.GridSearchCV
    ) -> NDArray[np.float64]:
        """Extract test scores from the GridSearch object."""
        # From the grid, extract the results from the hyperparameter set with
        # the best mean test score (lowest rank = best)
        best_set_idx = gs.best_index_

        # Get scores "in order" for consistency to do paired t-test
        scores = []
        k = len(
            [
                k
                for k in gs.cv_results_.keys()
                if "split" in k and "_test_score" in k
            ]
        )
        for i in range(k):
            scores.append(
                gs.cv_results_["split{}_test_score".format(i)][best_set_idx]
            )

        return np.array(scores)

    def _outer_loop(
        self,
        grid_search: sklearn.model_selection.GridSearchCV,
        X: NDArray[Any],
        y: NDArray[Any],
        cv: sklearn.model_selection.BaseCrossValidator,
        groups: Union[NDArray[np.integer], NDArray[np.str_], None],
    ) -> NDArray[np.floating]:
        """Perform outer loop."""
        scores = np.array([])
        for train_index, test_index in cv.split(X, y):
            X_train, _ = X[train_index], X[test_index]
            y_train, _ = y[train_index], y[test_index]

            if groups is None:
                grid_search.fit(X_train, y_train)
            else:
                grid_search.fit(X_train, y_train, groups=groups[train_index])

            # We don't actually use the test set here!
            # Unlike nested CV where we have k_outer test score estimates, now
            # we are just going to use the inner fold scores.  The outer fold
            # essentially just serves to  slightly "shift" the data which helps
            # decorrelate different repeats of the inner fold.  The "basic"
            # alternative is not to bother with the "outer fold" and just
            # repeat the inner fold procedure k_outer times (on the same data).
            scores = np.concatenate(
                (scores, self._get_test_scores(grid_search))
            )

        return scores

    def random_search(self, *args: Any, **kwargs: Any) -> NDArray[np.floating]:
        """Perform nested random search CV."""
        raise NotImplementedError

    def grid_search(
        self,
        pipeline: Union[sklearn.pipeline.Pipeline, imblearn.pipeline.Pipeline],
        param_grid: list[dict[str, Any]],
        X: NDArray[np.floating],
        y: Union[NDArray[np.floating], NDArray[np.integer], NDArray[np.str_]],
        classification: bool = True,
        error_score: Union[float, int] = np.nan,
        groups: Union[
            Sequence[int],
            Sequence[str],
            NDArray[np.integer],
            NDArray[np.str_],
            None,
        ] = None,
    ) -> NDArray[np.floating]:
        """
        Perform nested grid search CV.

        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline or imblearn.pipeline.Pipeline
            Pipeline to evaluate.

        param_grid : list(dict)
            `GridSearchCV.param_grid` object to perform grid search over.

        X : ndarray(float, ndim=2)
            Dense 2D array of observations (rows) of features (columns).

        y : ndarray(float, int, or str, ndim=1)
            Array of targets.

        classification : bool, optional(default=True)
            Is this a classification task (otherwise assumed to be regression)?

        error_score : scalar(float, int, np.nan), optional(default=np.nan)
            Value to return as the score if a failure occurs during fitting.

        groups : array-like, optional(default=None)
            If specified, these are the groups used to perform splitting in cross-validation.  If `None` it is assumed there is no grouping.  For classification tasks, stratification is also performed.

        Returns
        -------
        scores : ndarray(float, ndim=1)
            Array of length K*R containing scores from all test folds.

        Note
        ----
        For an RxK nested loop, R*K total scores are returned.  For classification tasks, the folds are stratified.
        """
        if groups is not None:
            if len(groups) != len(y):
                raise ValueError("Groups must have same length as y.")

        cv_ = None
        if classification and groups is None:
            cv_ = StratifiedKFold(
                n_splits=self.__k_inner, random_state=1, shuffle=True
            )
        if classification and groups is not None:
            cv_ = StratifiedGroupKFold(
                n_splits=self.__k_inner, random_state=1, shuffle=True
            )
        elif not classification and groups is None:
            cv_ = KFold(n_splits=self.__k_inner, random_state=1, shuffle=True)
        else:  # not classification and groups is not None
            cv_ = GroupKFold(n_splits=self.__k_inner)

        # This is the "inner" loop whose validation folds are going to be used as the "test" results and should use groupings to be less biased.
        self.gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            n_jobs=-1,
            error_score=error_score,
            cv=cv_,
            return_train_score=True,  # Results from the validation folds are stored
        )

        # The "outer" loop is just here to "shuffle" the data in a way that produces disjoint subsets that are removed from the training data pool.  We do NOT want to remove entire groups from the model's awareness; though this choice could be made.  If we use groups here then the "test" folds that are removed on each iteration will contain entire groups, while the model is trained on a fold of data with the "remaining" groups.
        scores = self._outer_loop(
            self.gs,
            X,
            y,
            cv=StratifiedKFold(
                n_splits=self.__k_outer, random_state=1, shuffle=True
            )
            if classification
            else KFold(n_splits=self.__k_outer, random_state=1, shuffle=True),
            groups=np.asarray(groups),
        )

        return scores


class Compare:
    """Compare the performances of different ML pipelines or algorithms."""

    def __init__(self) -> None:
        """Initialize the class."""
        pass

    @staticmethod
    def visualize(
        results: dict[str, list[float]],
        n_repeats: int,
        alpha: float = 0.05,
        ignore: Union[float, int, None] = np.nan,
        cmap: Union[str, matplotlib.colors.LinearSegmentedColormap] = "viridis",
    ) -> matplotlib.pyplot.Axes:
        """
        Plot a radial graph of performances for different pipelines.

        Parameters
        ----------
        results : dict(str, list(float))
            Dictionary of results for different pipelines evaluated on the same
            folds of data.  For example {"pipe1":[0.7, 0.5, 0.8], "pipe2":[0.5,
            0.6, 0.35]}.

        n_repeats : scalar(int)
            Number of times k-fold was repeated; k is inferred from the overall
            length based on this.

        alpha : scalar(float), optional(default=0.05)
            Significance level.

        ignore : scalar(float or int, or None), optional(default=np.nan)
            If any score is equal to this value (in either list of scores) their
            comparison is ignored.  This is to exclude scores from failed fits.
            sklearn's default `error_score` is `np.nan` so this is set as the
            default here, but a numeric value of 0 is also commonly used. A
            warning is raised if any scores are ignored.

        cmap : str or matplotlib.colors.LinearSegmentedColormap, optional(default='viridis')
            Name of matplotlib colormap to use when coloring the dial.
            See https://matplotlib.org/stable/users/explain/colors/colormaps.html.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Axes the radial graph is plotted on.

        Note
        ----
        When nested k-fold or repeated k-fold tests on different pipelines
        has been done, plot the mean and standard deviation of them from best
        (average) to worst.  These should be done on the same splits of data.
        A corrected, paired t-test is then performed and pipelines that we
        fail to reject H0 for (they perform the same as best) are colored via
        a colormap, whereas those that we do reject (performs worse than best)
        are colored in gray.
        """

        def perf(results, n_repeats, alpha):
            if np.isnan(ignore):
                key = lambda k: np.mean(
                    np.asarray(results[k])[~np.isnan(results[k])]
                )
            else:
                key = lambda k: np.mean(
                    np.asarray(results[k])[np.asarray(results[k]) != ignore]
                )

            order = sorted(results, key=key, reverse=True)

            performances = []
            for k in order:
                res_ = np.asarray(results[k])
                if np.isnan(ignore):
                    mask = ~np.isnan(res_)
                else:
                    mask = res_ != ignore
                performances.append(
                    [k, np.mean(res_[mask]), np.std(res_[mask]), False]
                )

            for i in range(1, len(performances)):
                p = Compare.corrected_t(
                    results[order[0]],
                    results[order[i]],
                    n_repeats,
                    ignore=ignore,
                )
                # Do we REJECT H0 (that pipelines perform the same)?
                performances[i][-1] = p < alpha

            return performances

        performances = perf(results, n_repeats=n_repeats, alpha=alpha)

        chart = plt.subplot(projection="polar")
        for i, p in enumerate(performances):
            if not p[-1]:  # REJECT H0, so pipeline 1 DOES outperform this one
                color = matplotlib.colormaps[cmap](
                    (1.0 + i) / len(performances)
                )
                hue = 1.0
            else:
                color = "gray"
                hue = 0.5
            _ = chart.barh(
                len(performances) - 1 - i,
                math.radians(p[1] * 360),
                label=p[0],
                xerr=math.radians(p[2] * 360),
                alpha=hue,
                color=color,
            )
            _ = chart.set_xticks(np.linspace(0, 0.9, 10) * 2.0 * np.pi)
            _ = chart.set_xticklabels(
                ["%.2f" % t for t in np.linspace(0.0, 0.9, 10)]
            )
            _ = chart.set_yticks([i for i in range(len(performances))])
            _ = chart.set_yticklabels([])
            _ = chart.set_title("Score +/- " + r"$\sigma$")
            _ = chart.legend(loc="best", bbox_to_anchor=(1.2, 0.5, 0.5, 0.5))

        return chart

    @staticmethod
    def repeated_kfold(
        estimators: Sequence[
            Union[
                sklearn.pipeline.Pipeline,
                imblearn.pipeline.Pipeline,
                sklearn.base.BaseEstimator,
                sklearn.model_selection.GridSearchCV,
            ]
        ],
        X: ArrayLike,
        y: ArrayLike,
        n_repeats: int = 5,
        k: int = 2,
        random_state: Union[int, np.random.RandomState] = 0,
        stratify: bool = True,
        groups: Union[ArrayLike, None] = None,
        estimators_mask: Union[
            Sequence[Sequence[bool]], Sequence[NDArray[np.bool_]], None
        ] = None,
    ) -> NDArray[np.floating]:
        """
        Perform repeated (stratified) k-fold cross validation to get scores for multiple estimators.

        Parameters
        ----------
        estimators : array-like(sklearn.pipeline.Pipeline, imblearn.pipeline.Pipeline, sklearn.base.BaseEstimator, or sklearn.model_selection.GridSearchCV, ndim=1)
            A list of pipelines or estimators that implements the `.fit()` and `.score()` methods. These can also be a `GridSearchCV` object.

        X : array-like(float, ndim=2)
            Matrix of features.

        y : array-like(float, int, or str, ndim=1)
            Array of outputs to predict.

        n_repeats : scalar(int), optional(default=5)
            Number of times cross-validator needs to be repeated.

        k : scalar(int), optional(default=2)
            K-fold cross-validation to use.

        random_state : scalar(int) or numpy.random.RandomState instance, optional(default=0)
            Controls the randomness of each repeated cross-validation instance.

        stratify : bool, optional(default=True)
            If True, use `RepeatedStratifiedKFold` or `RepeatedStratifiedGroupKFold`, depending on if `groups` is specified - this is only valid for classification tasks.

        groups : array-like(int or str, ndim=1), optional(default=None)
            Groups each observation (row) in `X` belongs to.

        estimators_mask : list(array-like(bool, ndim=1)), optional(default=None)
            Which columns of `X` to use in each estimator; default of `None` uses all columns for all estimators.  If specified, a mask must be given for each estimator.

        Returns
        -------
        scores : ndarray(float, ndim=2)
            List of scores for estimator.  Each row is a different estimator, in order of how they were provided, and each column is the result from a different test fold.

        Note
        ----
        The random state of the CV is the same for each so each pipeline or algorithm is tested on exactly the same dataset.  This enables paired t-test hypothesis testing using these scores.

        When comparing 2 pipelines that use different columns of X, use the estimators_mask variables to specify which columns to use.
        """
        if not hasattr(estimators, "__iter__"):
            raise TypeError("estimators is not iterable")
        else:
            estimators = [clone(est) for est in estimators]

        X_ = np.asarray(X)
        y_ = np.asarray(y)
        if groups is not None:
            groups_ = np.asarray(groups)

        if stratify:
            if groups is None:
                rkf = RepeatedStratifiedKFold
            else:
                rkf = RepeatedStratifiedGroupKFold
        else:
            if groups is None:
                rkf = RepeatedKFold
            else:
                rkf = RepeatedGroupKFold
        split = rkf(
            n_splits=k, n_repeats=n_repeats, random_state=random_state
        ).split(X_, y_, groups=groups_)

        if estimators_mask is not None:
            if len(estimators_mask) != len(estimators):
                raise Exception("A mask must be provided for each estimator.")
            estimators_mask = [
                np.asarray(mask, dtype=bool) for mask in estimators_mask
            ]
            for i, mask in enumerate(estimators_mask):
                if len(mask) != X_.shape[1]:
                    raise ValueError(f"Mask index {i} has the wrong size")
        else:
            estimators_mask = [
                np.array([True] * X_.shape[1]) for _ in estimators
            ]

        scores = []
        for train_index, test_index in split:
            X_train, X_test = X_[train_index], X_[test_index]
            y_train, y_test = y_[train_index], y_[test_index]

            fold_scores = []
            for i, est in enumerate(estimators):
                est.fit(X_train[:, estimators_mask[i]], y_train)
                fold_scores.append(
                    est.score(X_test[:, estimators_mask[i]], y_test)
                )
            scores.append(fold_scores)

        return np.array(scores, dtype=np.float64).T

    @staticmethod
    def corrected_t(
        scores1: Union[NDArray[np.floating], Sequence[float]],
        scores2: Union[NDArray[np.floating], Sequence[float]],
        n_repeats: int,
        ignore: Union[float, int, None] = np.nan,
    ) -> float:
        """
        Perform corrected 1-sided t-test to compare two pipelines.

        Parameters
        ----------
        scores1 : array-like(float, ndim=1)
            List of scores from pipeline 1.

        scores2 : array-like(float, ndim=1)
            List of scores from pipeline 2.

        n_repeats : scalar(int)
            Number of times k-fold was repeated (i.e., k_outer in BiasedNestedCV).

        ignore : scalar(float or int, or None), optional(default=np.nan)
            If any score is equal to this value (in either list of scores) their comparison is ignored.  This is to exclude scores from failed fits. sklearn's default `error_score` is `np.nan` so this is set as the default here, but a numeric value of 0 is also commonly used. A warning is raised if any scores are ignored.

        Returns
        -------
        p : scalar(float)
            p value

        Note
        ----
        Perform 1-sided hypothesis testing to see if any difference in pipelines' performances are statisically significant using a correlated, paired t-test with the Nadeau & Bengio (2003) correction. The test checks if the first pipeline is superior to the second using the alternative hypothesis, H1: mean(scores1-scores2) > 0.

        Reject H0 (that pipelines are equally good) in favor of H1 (pipeline1 is better) if p < alpha, otherwise fail to reject H0 (not enough evidence to suggest they are different). The formulation of this test is that pipeline1 has the best (average) performance or score of the two, and you want to check if that is statistically significant or not.

        Warning
        -------
        It is a good idea to manually check the scores for any `np.nan` or 0 values, etc. which can indicate a failed fit. Use `ignore` to exclude these points from the calculation.
        """
        scores1, scores2, mask = Compare._check_scores(
            scores1=scores1, scores2=scores2, ignore=ignore
        )

        k_fold = len(scores1) // int(n_repeats)
        n = k_fold * n_repeats
        assert n == len(scores1), "scores must be divisible by n_repeats"

        if np.mean(scores1[mask]) < np.mean(scores2[mask]):
            raise ValueError(
                "scores1 should have a higher mean value that scores2; reverse them and try again."
            )

        rho = 1.0 / k_fold
        perf_diffs = scores1[mask] - scores2[mask]  # H1: mu > 0
        corrected_t = (
            np.mean(perf_diffs) - 0.0
        ) / np.sqrt(  # n -> sum(mask) to be consistent
            (1.0 / np.sum(mask) + rho / (1.0 - rho))
            * (np.std(perf_diffs, ddof=1) ** 2)
        )

        return 1.0 - scipy.stats.t.cdf(
            x=corrected_t, df=np.sum(mask) - 1
        )  # 1-sided test

    @staticmethod
    def _check_scores(
        scores1: Union[NDArray[np.floating], Sequence[float]],
        scores2: Union[NDArray[np.floating], Sequence[float]],
        ignore: Union[float, int, None],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]:
        """Perform simple sanity checks for scores."""
        scores1 = np.asarray(scores1).flatten()
        scores2 = np.asarray(scores2).flatten()

        assert len(scores1) == len(
            scores2
        ), "scores must have the same \
        overall length"

        if ignore is not None:
            if np.isnan(ignore):
                mask1 = ~np.isnan(scores1)
                mask2 = ~np.isnan(scores2)
            else:
                mask1 = scores1 != ignore
                mask2 = scores2 != ignore
            mask = mask1 & mask2
        else:
            mask = np.array([True] * len(scores1), dtype=bool)

        if np.sum(mask) < len(scores1):
            warnings.warn(
                "Ignoring {}% of points for corrected_t test".format(
                    "%.2f" % (100.0 * (1 - np.sum(mask) / len(scores1)))
                )
            )

        return scores1, scores2, mask

    @staticmethod
    def bayesian_comparison(
        scores1: Union[NDArray[np.floating], Sequence[float]],
        scores2: Union[NDArray[np.floating], Sequence[float]],
        n_repeats: int,
        alpha: float,
        rope: float = 0.0,
        ignore: Union[float, int, None] = np.nan,
    ) -> tuple[NDArray[np.bool_], NDArray[np.floating]]:
        """
        Bayesian comparison between pipelines to assess relative performance.

        Parameters
        ----------
        scores1 : array-like(float, ndim=1)
            List of scores from each repeat of each CV fold for pipe1.

        scores2 : array-like(float, ndim=1)
            List of scores from each repeat of each CV fold for pipe2.

        n_repeats : scalar(int)
            Number of repetitions of cross validation.

        alpha : scalar(float)
            Statistical significance level.

        rope : scalar(float), optional(default=0.0)
            The width of the region of practical equivalence.

        ignore : scalar(int or float, or None), optional(default=np.nan)
            If any score is equal to this value (in either list of scores) their comparison is ignored.  This is to exclude scores from failed fits. sklearn's default `error_score` is `np.nan` so this is set as the default here, but a numeric value of 0 is also commonly used. A warning is raised if any scores are ignored.

        Returns
        -------
        is_better : ndarray(bool, ndim=1)
            Boolean array which, if True, indicates there is a significant difference.

        probs : ndarray(float, ndim=1)
            Array of (prob_1, p_equiv, prob_2).

        Note
        ----
        Perform Bayesian analysis to predict the probability that pipe(line)1 outperforms pipe(line)2 based on repeated kfold cross validation results using a correlated t-test.

        If prob[X] > 1.0 - alpha, then you make the decision that X is better. If no prob's reach this threshold, make no decision about the super(infer)iority of the pipelines relative to each other.

        References
        ----------
        See https://baycomp.readthedocs.io/en/latest/functions.html.

        Warning
        -------
        It is a good idea to manually check the scores for any `np.nan` or 0 values, etc. which can indicate a failed fit. Use `ignore` to exclude these points from the calculation.
        """
        scores1, scores2, mask = Compare._check_scores(
            scores1=scores1, scores2=scores2, ignore=ignore
        )

        probs = two_on_single(
            scores1[mask],
            scores2[mask],
            rope=rope,
            runs=n_repeats,
            names=None,
            plot=False,
        )

        if rope == 0:
            probs = np.array([probs[0], 0, probs[1]])

        return probs > (1.0 - alpha), probs
