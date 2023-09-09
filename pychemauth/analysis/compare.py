"""
Compare ML pipelines.

author: nam
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from baycomp import two_on_single
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

class BiasedNestedCV:
    """
    Perform nested GridSearchCV and get all validation fold scores.

    Parameters
    ----------
    k_inner : scalar(int), optional(default=2)
        K-fold for inner loop.

    k_outer : scalar(int), optional(default=5)
        K-fold for outer loop.
            
    Note
    ----
    This differs from scikit-learn's "built in" method of doing nested CV of
    cross_val_score(GridSeachCV()) in that cross_val_score() only returns
    the score from the test fold on the outer loop, after the best model is
    (usually) retrained on the entire training set using the best
    hyperparameters determined via the inner loop.  For doing statistical
    tests we want to have the validation scores from all the inner loops
    and are not interested in re-training/scoring on the outer loop.  The
    outer loop is just a way to "shift" or decorrelate the data chunks
    relative to simple repeated CV.

    This is used to assess the generalization error of an entire pipeline,
    which includes the model fitting, and hyperparameter search, for example.
    It may also include resampling, etc. of whatever else is part of the
    pipeline.  Thus, these estimates can be uses to asses the relative
    performances of different pipelines using corrected paired t-tested, etc.

    Typically, it is sufficient to perform relatively coarse grid searching
    when comparing different pipelines.  After pipelines are evaluated, the
    chosen one may be further optimized with more care, but these estimates
    of performance and uncertainty are expected to hold.

    "Nested CV" typically uses the held-out test fold for an unbiased esimate
    of the generalization error, as described above [1].  However, extensive
    real-world tests suggest that using the scores from the "flat CV" (the
    validation set) which is also used to identify optimal hyperparameters,
    does NOT make practical difference [2].  Therefore, we use the inner
    scores directly in concert with the statistical tests developed for cases
    where hyperparameter optimization was not assumed to be occuring
    simultaneously.

    References
    ----------
    [1] Cawley, G.C.; Talbot, N.L.C. On over-fitting in model selection and
    subsequent selection bias in performance evaluation. J. Mach. Learn. Res
    2010, 11, 2079-2107.
    
    [2] Wainer, Jacques, and Gavin Cawley. "Nested cross-validation when
    selecting classifiers is overzealous for most practical applications."
    Expert Systems with Applications 182 (2021): 115222.
    """

    def __init__(self, k_inner=2, k_outer=5):
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
    def visualize(results, n_repeats, alpha=0.05):
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
                    results[order[0]], results[order[i]], n_repeats
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
        estimator1,
        estimator2,
        X,
        y,
        n_repeats=5,
        k=2,
        random_state=0,
        stratify=True,
        pipe1_mask=None,
        pipe2_mask=None,
    ):
        """
        Perform repeated (stratified) k-fold cross validation to get scores.

        Parameters
        ----------
        estimator1 : sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
            Any pipeline or estimator that implements the fit() and score()
            methods. Can also be a GridSearchCV object.

        estimator2 : sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
            Pipeline to compare against. Can also be a GridSearchCV object.
        
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
        
        estimator1_mask : array_like(bool, ndim=1), optional(default=None)
            Which columns of X to use in estimator1; default of None uses all columns.
        
        estimator2_mask : array_like(bool, ndim=1), optional(default=None)
            Which columns of X to use in estimator2; default of None uses all columns.

        Returns
        -------
        scores1 : list(float)
            List of scores for estimator1.

        scores2 : list(float)
            List of scores for estimator2.
            
        Note
        ----
        The random state of the CV is the same for each so each pipeline or
        algorithm is tested on exactly the same dataset.  This enables paired
        t-test hypothesis testing using these scores.

        When comparing 2 pipelines that use different columns of X, use the
        estimatorX_mask variables to specify which columns to use.
        """
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

        if estimator1_mask:
            estimator1_mask = np.asarray(estimator1_mask, dtype=bool)
            assert len(estimator1_mask) == X.shape[1]
        else:
            estimator1_mask = np.array([True] * X.shape[1])

        if estimator2_mask:
            estimator2_mask = np.asarray(estimator2_mask, dtype=bool)
            assert len(estimator2_mask) == X.shape[1]
        else:
            estimator2_mask = np.array([True] * X.shape[1])

        scores1 = []
        scores2 = []
        for train_index, test_index in split:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            estimator1.fit(X_train[:, estimator1_mask], y_train)
            scores1.append(estimator1.score(X_test[:, estimator1_mask], y_test))

            estimator2.fit(X_train[:, estimator2_mask], y_train)
            scores2.append(estimator2.score(X_test[:, estimator2_mask], y_test))

        return scores1, scores2

    @staticmethod
    def corrected_t(scores1, scores2, n_repeats):
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
        """
        if np.mean(scores1) < np.mean(scores2):
            raise ValueError("scores1 should have a higher mean value that scores2; reverse them and try again.")
        
        assert len(scores1) == len(
            scores2
        ), "scores must have the same \
        overall length"
        k_fold = len(scores1) // int(n_repeats)
        n = k_fold * n_repeats
        assert n == len(scores1), "scores must be divisible by n_repeats"

        rho = 1.0 / k_fold
        perf_diffs = np.array(scores1) - np.array(scores2)  # H1: mu > 0
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
