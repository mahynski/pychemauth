"""
Compare ML pipelines.

author: nam
"""
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from baycomp import two_on_single
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
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
    This performs a version of repeated "flat CV" as defined in [1] to statistically
    compare the performance of two pipelines or models.  This is done by
    nesting flat CV inside an outer loop, akin to nested CV.
    This differs from scikit-learn's "built in" method of doing nested CV of
    cross_val_score(GridSeachCV()) in that cross_val_score() returns
    the scores from each test fold on the outer loop, obtained by training the best
    model on the entire inner set of training folds using the best
    hyperparameters determined by that the inner loop.  This is conventional "Nested CV"
    which uses the performance on the held-out test fold for an unbiased esimate of
    the generalization error [2].

    Here, we instead extract the performance estimates from the best model's validation
    fold (held out part of the inner training folds) used to select the hyperparameters
    for a given iteration of the outer loop.  These performance estimates are positively
    biased, but extensive real-world testing suggests that using the scores from the
    "flat CV" to **rank** different models does not make practical difference [2] to
    their relative ordering. Therefore, we can use the inner scores directly in concert
    with paired t-tests to compare different pipelines when it is too computationally
    expensive to do repeated nested CV.

    Unlike simple repetition of any CV procedure, framing this as a nested CV uses
    the outer loop as way to "shift" or decorrelate the data chunks more relative to
    simply reshuffling and reusing the data.  Thus, the Nadeau and Bengio's correction
    factor [3] for the t-test is less necessary, but further aids in making conservative
    inference.

    References
    ----------
    [1] Wainer, J. and Cawley G., "Nested cross-validation when
    selecting classifiers is overzealous for most practical applications."
    Expert Systems with Applications 182, 115222 (2021).

    [2] Cawley, G.C. and Talbot, N.L.C., "On over-fitting in model selection and
    subsequent selection bias in performance evaluation," J. Mach. Learn. Res
    11, 2079-2107 (2010).

    [3] Nadeau, C. and Bengio Y., "Inference for the Generalization Error," Machine
    Learning 52, 239-281 (2003).
    """

    def __init__(self, k_inner=10, k_outer=10):
        """Instantiate the class."""
        self.__k_inner = k_inner
        self.__k_outer = k_outer

    def _get_test_scores(self, gs):
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

    def _outer_loop(self, pipeline, X, y, cv):
        """Perform outer loop."""
        scores = []
        for train_index, test_index in cv.split(X, y):
            X_train, _ = X[train_index], X[test_index]
            y_train, _ = y[train_index], y[test_index]

            pipeline.fit(X_train, y_train)

            # We don't actually use the test set here!
            # Unlike nested CV where we have k_outer test score estimates, now
            # we are just going to use the inner fold scores.  The outer fold
            # essentially just serves to  slightly "shift" the data which helps
            # decorrelate different repeats of the inner fold.  The "basic"
            # alternative is not to bother with the "outer fold" and just
            # repeat the inner fold procedure k_outer times (on the same data).
            scores = np.concatenate((scores, self._get_test_scores(pipeline)))

        return scores

    def random_search(self, *args, **kwargs):
        """Perform nested random search CV."""
        raise NotImplementedError

    def grid_search(self, pipeline, param_grid, X, y, classification=True):
        """
        Perform nested grid search CV.

        Parameters
        ----------
        pipeline : sklearn.pipeline or imblearn.pipeline
            Pipeline to evaluate.

        param_grid : list(dict)
            GridSearchCV.param_grid object to perform grid search over.

        X : ndarray(float, ndim=2)
            Dense 2D array of observations (rows) of features (columns).

        y : ndarray(float, ndim=1)
            Array of targets.

        classification : scalar(bool), optional(default=True)
            Is this a classification task (otherwise assumed to be regression)?

        Returns
        -------
        scores : ndarray(float, ndim=1)
            Array of length K*R containing scores from all test folds.

        Note
        ----
        For an RxK nested loop, R*K total scores are returned.  For
        classification tasks, KFolds are stratified.
        """
        self.gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            n_jobs=-1,
            cv=StratifiedKFold(
                n_splits=self.__k_inner, random_state=1, shuffle=True
            )
            if classification
            else KFold(n_splits=self.__k_inner, random_state=1, shuffle=True),
            return_train_score=True,  # Results from the test folds are stored
        )

        scores = self._outer_loop(
            self.gs,
            X,
            y,
            cv=StratifiedKFold(
                n_splits=self.__k_outer, random_state=1, shuffle=True
            )
            if classification
            else KFold(n_splits=self.__k_outer, random_state=1, shuffle=True),
        )
        return scores


class Compare:
    """Compare the performances of different ML pipelines or algorithms."""

    def __init__(self):
        """Initialize the class."""
        pass

    @staticmethod
    def visualize(results, n_repeats, alpha=0.05, ignore=np.nan):
        """
        Plot a radial graph of performances for different pipelines.

        Parameters
        ----------
        results : dict(list(float))
            Dictionary of results for different pipelines evaluated on the same
            folds of data.  For example {"pipe1":[0.7, 0.5, 0.8], "pipe2":[0.5,
            0.6, 0.35]}.

        n_repeats : scalar(int)
            Number of times k-fold was repeated; k is inferred from the overall
            length based on this.

        alpha : scalar(float), optional(default=0.05)
            Significance level.

        ignore : scalar(int, float, str), optional(default=np.nan)
            If any score is equal to this value (in either list of scores) their 
            comparison is ignored.  This is to exclude scores from failed fits.
            sklearn's default `error_score` is `np.nan` so this is set as the
            default here, but a numeric value of 0 is also commonly used. A
            warning is raised if any scores are ignored.

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
            order = sorted(
                results, key=lambda k: np.mean(results[k]), reverse=True
            )
            performances = [
                [k, np.mean(results[k]), np.std(results[k]), False]
                for k in order
            ]

            for i in range(1, len(performances)):
                p = Compare.corrected_t(
                    results[order[0]], results[order[i]], n_repeats, ignore=ignore
                )
                # Do we REJECT H0 (that pipelines perform the same)?
                performances[i][-1] = p < alpha

            return performances

        performances = perf(results, n_repeats=n_repeats, alpha=alpha)

        chart = plt.subplot(projection="polar")
        for i, p in enumerate(performances):
            if not p[-1]:  # REJECT H0, so pipeline 1 DOES outperform this one
                color = plt.cm.viridis((1.0 + i) / len(performances))
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
        estimators,
        X,
        y,
        n_repeats=5,
        k=2,
        random_state=0,
        stratify=True,
        estimators_mask=None,
    ):
        """
        Perform repeated (stratified) k-fold cross validation to get scores for multiple estimators.

        Parameters
        ----------
        estimators : array_like(sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator, ndim=1)
            A list of pipelines or estimators that implements the fit() and score()
            methods. These can also be a GridSearchCV object.

        X : array_like(float, ndim=2)
            Matrix of features.

        y : array_like(float, ndim=1)
            Array of outputs to predict.

        n_repeats : scalar(int), optional(default=5)
            Number of times cross-validator needs to be repeated.

        k : scalar(int), optional(default=2)
            K-fold cross-validation to use.

        random_state : scalar(int) or numpy.random.RandomState instance, optional(default=0)
            Controls the randomness of each repeated cross-validation instance.

        stratify : scalar(bool), optional(default=True)
            If True, use RepeatedStratifiedKFold - this is only valid for
            classification tasks.

        estimators_mask : list(array_like(bool, ndim=1)), optional(default=None)
            Which columns of X to use in each estimator; default of None uses all columns for
            all estimators.  If specified, a mask must be given for each estimator.

        Returns
        -------
        scores : ndarray(float, ndim=2)
            List of scores for estimator.  Each row is a different estimator, in order
            of how they were provided, and each column is the result from a different
            test fold.

        Note
        ----
        The random state of the CV is the same for each so each pipeline or
        algorithm is tested on exactly the same dataset.  This enables paired
        t-test hypothesis testing using these scores.

        When comparing 2 pipelines that use different columns of X, use the
        estimators_mask variables to specify which columns to use.
        """
        if not hasattr(estimators, "__iter__"):
            raise TypeError("estimators is not iterable")
        else:
            estimators = [clone(est) for est in estimators]

        if stratify:
            rkf = RepeatedStratifiedKFold(
                n_splits=k, n_repeats=n_repeats, random_state=random_state
            )
            split = rkf.split(X, y)
        else:
            rkf = RepeatedKFold(
                n_splits=k, n_repeats=n_repeats, random_state=random_state
            )
            split = rkf.split(X)

        X = np.asarray(X)
        y = np.asarray(y)

        if estimators_mask is not None:
            if len(estimators_mask) != len(estimators):
                raise Exception("A mask must be provided for each estimator.")
            estimators_mask = [
                np.asarray(mask, dtype=bool) for mask in estimators_mask
            ]
            for i, mask in enumerate(estimators_mask):
                if len(mask) != X.shape[1]:
                    raise ValueError(f"Mask index {i} has the wrong size")
        else:
            estimators_mask = [
                np.array([True] * X.shape[1]) for est in estimators
            ]

        scores = []
        for train_index, test_index in split:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            fold_scores = []
            for i, est in enumerate(estimators):
                est.fit(X_train[:, estimators_mask[i]], y_train)
                fold_scores.append(
                    est.score(X_test[:, estimators_mask[i]], y_test)
                )
            scores.append(fold_scores)

        return np.array(scores, dtype=np.float64).T

    @staticmethod
    def corrected_t(scores1, scores2, n_repeats, ignore=np.nan):
        """
        Perform corrected 1-sided t-test to compare two pipelines.

        Parameters
        ----------
        scores1 : array_like(float, ndim=1)
            List of scores from pipeline 1.

        scores2 : array_like(float, ndim=1)
            List of scores from pipeline 2.

        n_repeats : scalar(int)
            Number of times k-fold was repeated (i.e., k_outer in BiasedNestedCV).

        ignore : scalar(int, float, str), optional(default=np.nan)
            If any score is equal to this value (in either list of scores) their 
            comparison is ignored.  This is to exclude scores from failed fits.
            sklearn's default `error_score` is `np.nan` so this is set as the
            default here, but a numeric value of 0 is also commonly used. A
            warning is raised if any scores are ignored.

        Returns
        -------
        p : scalar(float)
            p value

        Note
        ----
        Perform 1-sided hypothesis testing to see if any difference in
        pipelines' performances are statisically significant using a
        correlated, paired t-test with the Nadeau & Bengio (2003)
        correction. The test checks if the first pipeline is superior to the
        second using the alternative hypothesis, H1: mean(scores1-scores2) > 0.

        Reject H0 (that pipelines are equally good) in favor of H1 (pipeline1
        is better) if p < alpha, otherwise fail to reject H0 (not enough
        evidence to suggest they are different). The formulation of this test
        is that pipeline1 has the best (average) performance or score of the
        two, and you want to check if that is statistically significant or not.

        Warning
        -------
        It is a good idea to manually check the scores for any `np.nan` or 0 
        values, etc. which can indicate a failed fit. Use `ignore` to exclude
        these points from the calculation.
        """
        if np.mean(scores1) < np.mean(scores2):
            raise ValueError(
                "scores1 should have a higher mean value that scores2; reverse them and try again."
            )

        assert len(scores1) == len(
            scores2
        ), "scores must have the same \
        overall length"
        k_fold = len(scores1) // int(n_repeats)
        n = k_fold * n_repeats
        assert n == len(scores1), "scores must be divisible by n_repeats"

        if ignore is not None:
            if np.isnan(ignore):
                mask1 = ~np.isnan(scores1)
                mask2 = ~np.isnan(scores2)
            else:
                mask1 = np.asarray(scores1) != ignore
                mask2 = np.asarray(scores2) != ignore
            mask = mask1 & mask2
        else:
            mask = np.array([True]*n, dtype=bool)

        if np.sum(mask) < n:
            warnings.warn("Ignoring {}% of points for corrected_t test".format("%.2f"%(100.0*(1 - np.sum(mask)/n))))

        rho = 1.0 / k_fold
        perf_diffs = np.array(scores1)[mask] - np.array(scores2)[mask]  # H1: mu > 0
        corrected_t = (np.mean(perf_diffs) - 0.0) / np.sqrt(
            (1.0 / n + rho / (1.0 - rho)) * (np.std(perf_diffs, ddof=1) ** 2)
        )

        return 1.0 - scipy.stats.t.cdf(x=corrected_t, df=n - 1)  # 1-sided test

    @staticmethod
    def bayesian_comparison(scores1, scores2, n_repeats, alpha, rope=0.0):
        """
        Bayesian comparison between pipelines to assess relative performance.

        Parameters
        ----------
        scores1 : array_like(float, ndim=1)
            List of scores from each repeat of each CV fold for pipe1.

        scores2 : array_like(float, ndim=1)
            List of scores from each repeat of each CV fold for pipe2.

        n_repeats : scalar(int)
            Number of repetitions of cross validation.

        alpha : scalar(float)
            Statistical significance level.

        rope : scalar(float), optional(default=0.0)
            The width of the region of practical equivalence.

        Returns
        -------
        probs : tuple(float, float , float)
            Tuple of (prob_1, p_equiv, prob_2).

        Note
        ----
        Perform Bayesian analysis to predict the probability that pipe(line)1
        outperforms pipe(line)2 based on repeated kfold cross validation
        results using a correlated t-test.

        If prob[X] > 1.0 - alpha, then you make the decision that X is better.
        If no prob's reach this threshold, make no decision about the
        super(infer)iority of the pipelines relative to each other.

        References
        -----
        See https://baycomp.readthedocs.io/en/latest/functions.html.
        """
        scores1 = np.array(scores1).flatten()
        scores2 = np.array(scores2).flatten()
        probs = two_on_single(
            scores1, scores2, rope=rope, runs=n_repeats, names=None, plot=False
        )

        if rope == 0:
            probs = np.array([probs[0], 0, probs[1]])

        return probs > (1.0 - alpha), probs
