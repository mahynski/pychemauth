"""
Cross validation approaches.

author: nam
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold


class NestedCV:
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
