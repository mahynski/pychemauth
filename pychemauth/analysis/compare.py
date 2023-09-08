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
        pipe1,
        pipe2,
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
        pipe1 : sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
            Any pipeline or estimator that implements the fit() and score()
            methods. Can also be a GridSearchCV object.

        pipe2 : sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
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
        
        pipe1_mask : array_like(bool, ndim=1), optional(default=None)
            Which columns of X to use in pipe1; default of None uses all columns.
        
        pipe2_mask : array_like(bool, ndim=1), optional(default=None)
            Which columns of X to use in pipe2; default of None uses all columns.

        Returns
        -------
        scores1 : list(float)
            List of scores for pipeline1.

        scores2 : list(float)
            List of scores for pipeline2.
            
        Note
        ----
        The random state of the CV is the same for each so each pipeline or
        algorithm is tested on exactly the same dataset.  This enables paired
        t-test hypothesis testing using these scores.

        When comparing 2 pipelines that use different columns of X, use the
        pipeX_mask variables to specify which columns to use.
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

        if pipe1_mask:
            pipe1_mask = np.asarray(pipe1_mask, dtype=bool)
            assert len(pipe1_mask) == X.shape[1]
        else:
            pipe1_mask = np.array([True] * X.shape[1])

        if pipe2_mask:
            pipe2_mask = np.asarray(pipe2_mask, dtype=bool)
            assert len(pipe2_mask) == X.shape[1]
        else:
            pipe2_mask = np.array([True] * X.shape[1])

        scores1 = []
        scores2 = []
        for train_index, test_index in split:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            pipe1.fit(X_train[:, pipe1_mask], y_train)
            scores1.append(pipe1.score(X_test[:, pipe1_mask], y_test))

            pipe2.fit(X_train[:, pipe2_mask], y_train)
            scores2.append(pipe2.score(X_test[:, pipe2_mask], y_test))

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
            Number of times k-fold was repeated (i.e., k_outer in NestedCV).

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
