"""
Open Set Recognition Models.

author: nam
"""
import sys
import copy

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.preprocessing import LabelEncoder

from pychemauth.utils import _multiclass_cm_metrics, _occ_cm_metrics


class OpenSetClassifier(ClassifierMixin, BaseEstimator):
    """
    Train a composite classifier with a reject option to work under open-set conditions.

    Parameters
    ----------
    clf_model : object, optional(default=None)
        Unfitted classification model. Must support `.fit()` and `.predict()` methods.

    outlier_model : object, optional(default=None)
        Unfitted outlier detection model. Must support `.fit()` and `.predict()` methods.
        This should return a value of `inlier_value` for points which are considered inliers.
        If `None` then all points will be passed to the classifier.

    clf_kwargs : dict, optional(default={})
        Keyword arguments to instantiate the classification model with.

    outlier_kwargs : dict, optional(default={})
        Keyword arguments to instantiate the outlier model with.

    known_classes : array_like(int or str, ndim=1), optional(default=None)
        A list of classes which the classifier is responsible for recognizing. If `None`,
        all unique values of `y` are used; otherwise, `y` is filtered to only include these
        instances when training the classifier.

    inlier_value : scalar, optional(default=1)
        The value `outlier_model.predict()` returns for inlier class(es).  Many sklearn routines
        return +1 for inlier vs. -1 for outlier; other routines sometimes use 0 for outlier.
        As a result, we simply check for the inlier value (+1 by default) for greater flexibility.

    unknown_class : scalar(int or str), optional(default="Unknown")
        The name or index to assign to points which are considered unknown according to the
        `outlier_model`.

    score_metric : scalar(str), optional(default="TEFF")
        Default scoring metric to use. See `figures_of_merit` outputs for options.

    clf_style : scalar(str), optional(default="hard")
        Style of classification model; "hard" models assign each point to a single category, while
        "soft" models can make multiple assignments, including to an unknown category.

    score_using : scalar(int or str), optional(default="all")
        Which classes to use for scoring.  The default "all" computes TEFF, etc. using all
        `known_classes`; intead, if a single class name is provided the metrics are computed
        to reflect this model as a one-class classifier (OCC), such as SIMCA.  OCC models
        return a binary yes/no membership decision, but not both, so these are 'hard' models.
        An error will be thrown if this is incorrectly specified.

    Note
    ----
    This is composed of an outlier model, which is called first to determine which points
    are considered inliers and which are outliers, and a classification model, which is
    resposible for classifying the inliers (closed set).

    The type of `unknown_class` should mimic that of the raw data; i.e., if classes in y are
    strings unknown_class should be a string (default="Unknown"). Integers may also be used.

    Warning
    -------
    The TSPS formula changes depending on whether a hard or soft classifier is being used.
    Also see :class:`pychemauth.classifier.plsda.PLSDA`.
    """

    def __init__(
        self,
        clf_model=None,
        outlier_model=None,
        clf_kwargs={},
        outlier_kwargs={},
        known_classes=None,
        inlier_value=1,
        unknown_class="Unknown",
        score_metric="TEFF",
        clf_style="hard",
        score_using="all",
    ):
        """Initialize the class."""
        self.set_params(
            **{
                "clf_model": clf_model,
                "outlier_model": outlier_model,
                "clf_kwargs": clf_kwargs,
                "outlier_kwargs": outlier_kwargs,
                "known_classes": known_classes,
                "inlier_value": inlier_value,
                "unknown_class": unknown_class,
                "score_metric": score_metric,
                "clf_style": clf_style,
                "score_using": score_using,
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
            "clf_model": self.clf_model,
            "outlier_model": self.outlier_model,
            "clf_kwargs": self.clf_kwargs,
            "outlier_kwargs": self.outlier_kwargs,
            "known_classes": self.known_classes,
            "inlier_value": self.inlier_value,
            "unknown_class": self.unknown_class,
            "score_metric": self.score_metric,
            "clf_style": self.clf_style,
            "score_using": self.score_using,
        }

    def _check_category_type(self, y):
        """Check that categories are same type as `unknown_class` variable."""
        t_ = None
        for t_ in [(int, np.int32, np.int64), (str,)]:
            if isinstance(self.unknown_class, t_):
                use_type = t_
                break
        if t_ is None:
            raise TypeError("unknown_class must be an integer or string")
        if not np.all([isinstance(y_, use_type) for y_ in y]):
            raise ValueError(
                "You must set the 'unknown_class' variable type ({}) the same \
                as y, e.g., both must be int or str".format(
                    [type(y_) for y_ in y]
                )
            )

    def fit(self, X, y):
        """
        Fit the composite model.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Input feature matrix.

        y : array_like(str or int, ndim=1)
            Class labels or indices.

        Returns
        -------
        self : OpenWorldClassifier
            Fitted model.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        self._check_category_type(y.ravel())
        assert self.unknown_class not in set(
            y
        ), "unknown_class value is already taken"

        if not (self.clf_style in ["soft", "hard"]):
            raise ValueError("clf_style should be either 'hard' or 'soft'.")

        # Remove any classes the classification model should not be responsible for
        # learning.
        if self.known_classes is None:
            # Consider all training examples.
            self.knowns_ = np.unique(y)
        else:
            self.knowns_ = np.unique(self.known_classes)

        # For sklearn compatibility - not used
        self.classes_ = self.knowns_.tolist() + [self.unknown_class]

        known_mask = np.array([y_ in self.knowns_ for y_ in y], dtype=bool)

        if np.sum(known_mask) == 0:
            raise Exception("There are no known classes in the training set.")

        # Check that self.score_using is valid
        if (self.score_using not in self.knowns_) and (
            self.score_using.lower() != "all"
        ):
            raise ValueError(
                "score_using should be 'all' or one of the classes trained on."
            )

        # Train outlier detection first, since this how it will work at prediction
        # time.  This needs to remember the data that the classifier will use for
        # training and flag anything different (covariate shift).  Thus, this needs
        # to train on the knowns_ only.
        try:
            self.od_ = self.outlier_model(**self.outlier_kwargs)
            self.od_.fit(X[known_mask, :])
        except:
            raise Exception(
                f"Unable to fit outlier model : {sys.exc_info()[0]}"
            )

        # Predict for all X for simplicity. The composite mask will only allow knowns
        # which are not outliers through.
        inlier_mask = self.od_.predict(X) == self.inlier_value

        composite_mask = known_mask & inlier_mask

        if np.sum(composite_mask) == 0:
            raise Exception(
                "There are no inlying known classes in the training set."
            )

        if len(np.unique(y[composite_mask])) < 2:
            raise Exception(
                "There are less than 2 distinct classes available for training."
            )

        try:
            self.clf_ = self.clf_model(**self.clf_kwargs)
            self.clf_.fit(X[composite_mask, :], y[composite_mask])
        except:
            raise Exception(
                f"Unable to fit classification model : {sys.exc_info()[0]}"
            )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Make a prediction.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Input feature matrix.

        Returns
        -------
        predictions : array_like(int or str, ndim=2)
            Class, or classes, assigned to each point.  Points considered outliers are
            assigned the value `unknown_class`.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, accept_sparse=False)
        assert X.shape[1] == self.n_features_in_

        # 1. Check for outliers
        inlier_mask = self.od_.predict(X) == self.inlier_value

        # 2. Predict on points considered inliers
        if np.sum(inlier_mask) > 0:
            pred = self.clf_.predict(X[inlier_mask, :])

        predictions = [[]] * len(X)
        j = 0
        for i in range(X.shape[0]):
            if not inlier_mask[i]:
                predictions[i] = (
                    [self.unknown_class]
                    if self.clf_style == "soft"
                    else self.unknown_class
                )
            else:
                predictions[i] = pred[j]
                j += 1

        return predictions

    def fit_predict(self, X, y):
        """
        Fit then predict.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Input feature matrix.

        y : array_like(str or int, ndim=1)
            Class labels or indices.

        Returns
        -------
        predictions : array_like(int or str, ndim=2)
            Class, or classes, assigned to each point.  Points considered outliers are
            assigned the value `unknown_class`.
        """
        self.fit(X, y)
        return self.predict(X)

    def score(self, X, y):
        """
        Score the prediction.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array_like(str or int, ndim=1)
            Ground truth classes - will be converted to numpy array
            automatically.

        Returns
        -------
        score : scalar(float)
            Score.
        """
        check_is_fitted(self, "is_fitted_")

        X, y = np.asarray(X), np.asarray(y)
        self._check_category_type(y.ravel())
        metrics = self.figures_of_merit(self.predict(X), y)
        if self.score_metric.upper() not in metrics:
            raise ValueError(
                "Unrecognized metric : {}".format(self.score_metric.upper())
            )
        else:
            return metrics[self.score_metric.upper()]

    @property
    def fitted_classification_model(self):
        """Return the fitted classification model."""
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.clf_)

    @property
    def fitted_outlier_model(self):
        """Return the fitted outlier model."""
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.od_)

    def figures_of_merit(self, predictions, actual):
        """
        Compute figures of merit.

        Parameters
        ----------
        predictions : array_like(str or int, ndim=2)
            Array of values containing the predicted class of points (in
            order). Each row may have multiple entries corresponding to
            multiple class predictions.

        actual : array_like(str or int, ndim=1)
            Array of ground truth classes for the predicted points.  Should
            have only one class per point.

        Returns
        -------
        fom : dict
            Dictionary object with the following attributes. Note that CSPS and CEFF
            are absent in the case of a one-class classifier (`score_using` is a single
            class instead of 'all').

            CM : pandas.DataFrame
                Inputs (index) vs. predictions (columns); akin to a confusion matrix.

            I : pandas.Series
                Number of each class asked to classify.

            CSNS : pandas.Series
                Class sensitivity.

            CSPS : pandas.Series
                Class specificity.

            CEFF : pandas.Series
                Class efficiency.

            TSNS : scalar(float)
                Total sensitivity.

            TSPS : scalar(float)
                Total specificity.

            TEFF : scalar(float)
                Total efficiency.

            ACC : scalar(float)
                Accuracy.

        Note
        ----
        When making predictions about purely extraneous classes (not in training set)
        class efficiency (CEFF) is given as simply class specificity (CSPS)
        since class sensitivity (CSNS) cannot be calculated.  For a one-class
        classifier, TSNS = CSNS.

        References
        ----------
        [1] "Multiclass partial least squares discriminant analysis: Taking the
        right way - A critical tutorial," Pomerantsev and Rodionova, Journal of
        Chemometrics (2018). https://doi.org/10.1002/cem.3030.
        """
        check_is_fitted(self, "is_fitted_")

        # Dummy check that not_assigned and y have same data types
        actual = np.asarray(actual).ravel()
        self._check_category_type(actual)
        assert self.unknown_class not in set(
            actual
        ), "unknown_class value is already taken"

        all_classes = [self.unknown_class] + np.unique(
            np.concatenate((np.unique(actual), self.knowns_))
        ).tolist()
        encoder = LabelEncoder()
        encoder.fit(all_classes)
        n_classes = len(all_classes)
        use_classes = encoder.classes_[encoder.classes_ != self.unknown_class]

        n = np.zeros((n_classes, n_classes), dtype=int)
        for row, actual_class in zip(predictions, actual):
            kk = encoder.transform([actual_class])[0]
            if isinstance(row, np.ndarray) or isinstance(row, list):
                if self.clf_style.lower() == "hard":
                    raise Exception(
                        "Found multiple class assignments - perhaps you are using a soft model?"
                    )
                for entry in row:
                    try:
                        ll = encoder.transform([entry])[0]
                    except:
                        # Assume that if the encoder does not recognize the entry it is
                        # from the model returning an "unknown" assignment.  This string/value
                        # won't be in the data and is usually specified when the model
                        # is trained so it is difficult to build consistently into this
                        # workflow; this seems to be the best approach.
                        assert (
                            len(row) == 1
                        )  # If "unknown" then this should be the only assignment made
                        ll = encoder.transform([self.unknown_class])[0]
                    n[kk, ll] += 1
            else:
                if self.clf_style.lower() == "soft":
                    raise Exception(
                        "Class assignments not provided as list - perhaps you are using a hard model or OCC?"
                    )
                ll = encoder.transform([row])[0]
                n[kk, ll] += 1

        df = pd.DataFrame(
            data=n, columns=encoder.classes_, index=encoder.classes_
        )
        df = df[df.index != self.unknown_class]  # Trim off row of "UNKNOWN"
        Itot = pd.Series(
            [np.sum(np.array(actual) == kk) for kk in use_classes],
            index=use_classes,
        )
        assert np.sum(Itot) == len(actual)

        if self.score_using.lower() == "all":
            # correct_ = 0.0
            # for class_ in df.index:  # All input classes
            #     if (
            #         class_ in self.knowns_
            #     ):  # Things to classifier knows about (TP)
            #         correct_ += df[class_][class_]
            #     else:
            #         # Consider an assignment as "unknown" a correct assignment (TN)
            #         correct_ += df[self.unknown_class][class_]
            # ACC = correct_ / df.sum().sum()

            fom = _multiclass_cm_metrics(
                df=df, 
                Itot=Itot, 
                trained_classes=self.knowns_, 
                use_classes=use_classes, 
                style=self.clf_style, 
                not_assigned=self.unknown_class, 
                actual=actual
            )
            # fom['ACC'] = ACC
            
            # # Class-wise FoM
            # # Sensitivity is "true positive" rate and is only defined for
            # # trained/known classes
            # CSNS = pd.Series(
            #     [
            #         df[kk][kk] / Itot[kk] if Itot[kk] > 0 else np.nan
            #         for kk in self.knowns_
            #     ],
            #     index=self.knowns_,
            # )

            # # Specificity is the fraction of points that are NOT a given class that
            # # are correctly predicted to be something besides the class. Thus,
            # # specificity can only be computed for the columns that correspond to
            # # known classes since we have only trained on them. These are "true
            # # negatives". This is always >= 0.
            # CSPS = pd.Series(
            #     [
            #         1.0
            #         - np.sum(df[kk][df.index != kk])  # Column sum
            #         / np.sum(Itot[Itot.index != kk])
            #         for kk in self.knowns_
            #     ],
            #     index=self.knowns_,
            # )

            # # If CSNS can't be calculated, using CSPS as efficiency;
            # # Oliveri & Downey introduced this "efficiency" used in [1]
            # CEFF = pd.Series(
            #     [
            #         np.sqrt(CSNS[c] * CSPS[c])
            #         if not np.isnan(CSNS[c])
            #         else CSPS[c]
            #         for c in self.knowns_
            #     ],
            #     index=self.knowns_,
            # )

            # # Total FoM

            # # Evaluates overall ability to recognize a class is itself.  If you
            # # show the model some class it hasn't trained on, it can't be predicted
            # # so no contribution to the diagonal.  We will normalize by total
            # # number of points shown [1].  If some classes being tested were seen in
            # # training they contribute, otherwise TSNS goes down for a class never
            # # seen before.  This might seem unfair, but TSNS only makes sense if
            # # (1) you are examining what you have trained on or (2) you are
            # # examining extraneous objects so you don't calculate this at all.
            # TSNS = np.sum([df[kk][kk] for kk in self.knowns_]) / np.sum(Itot)

            # # If any untrained class is correctly predicted to be "NOT_ASSIGNED" it
            # # won't contribute to df[use_classes].sum().sum().  Also, unseen
            # # classes can't be assigned to so the diagonal components for those
            # # entries is also 0 (df[k][k]).
            # TSPS = 1.0 - (
            #     df[use_classes].sum().sum()
            #     - np.sum([df[kk][kk] for kk in use_classes])
            # ) / np.sum(Itot) / (
            #     1.0
            #     if self.clf_style.lower() == "hard"
            #     else len(self.knowns_) - 1.0
            # )
            # Soft models can assign a point to all categories which would make this
            # sum > 1, meaning TSPS < 0 would be possible.  By scaling by the total
            # number of classes, TSPS is always positive; TSPS = 0 means all points
            # assigned to all classes (trivial result) vs. TSPS = 1 means no mistakes.

            # Sometimes TEFF is reported as TSPS when TSNS cannot be evaluated (all
            # previously unseen samples).
            # TEFF = np.sqrt(TSPS * TSNS)

            # fom = dict(
            #     zip(
            #         [
            #             "CM",
            #             "I",
            #             "CSNS",
            #             "CSPS",
            #             "CEFF",
            #             "TSNS",
            #             "TSPS",
            #             "TEFF",
            #             "ACC",
            #         ],
            #         (
            #             df[
            #                 [c for c in df.columns if c in self.knowns_]
            #                 + [self.unknown_class]
            #             ][
            #                 [x in np.unique(actual) for x in df.index]
            #             ],  # Re-order for easy visualization
            #             Itot,
            #             CSNS,
            #             CSPS,
            #             CEFF,
            #             TSNS,
            #             TSPS,
            #             TEFF,
            #             ACC,
            #         ),
            #     )
            # )
        else:
            # Evaluate as a OCC where the score_using class is the target class.
            fom = _occ_cm_metrics(df=df, Itot=Itot, target_class=self.score_using, trained_classes=self.knowns_, not_assigned=self.unknown_class, actual=actual)

            # alternatives = [
            #     class_ for class_ in df.index if class_ != self.score_using
            # ]

            # correct_ = df[self.score_using][self.score_using]  # (TP)
            # for class_ in alternatives:  # All "negative" classes
            #     # Number of times an observation NOT from score_using was correctly not assigned to score_using
            #     # Assigning to multiple alternatives does not influence this in the spirit of OCC
            #     correct_ += Itot[class_] - df[self.score_using][class_]  # (TN)
            # ACC = correct_ / float(Itot.sum())

            # CSPS = {}
            # for class_ in alternatives:
            #     if np.sum(Itot[class_]) > 0:
            #         CSPS[class_] = 1.0 - df[class_][self.score_using] / np.sum(
            #             Itot[class_]
            #         )
            #     else:
            #         CSPS[class_] = np.nan

            # if np.all(actual == self.score_using):
            #     # Testing on nothing but the target class, can't evaluate TSPS
            #     TSPS = np.nan
            # else:
            #     TSPS = 1.0 - (
            #         df[self.score_using].sum()
            #         - df[self.score_using][self.score_using]
            #     ) / (Itot.sum() - Itot[self.score_using])

            # # TSNS = CSNS
            # if self.score_using not in set(actual):
            #     # Testing on nothing but alternative classes, can't evaluate TSNS
            #     TSNS = np.nan
            # else:
            #     TSNS = (
            #         df[self.score_using][self.score_using]
            #         / Itot[self.score_using]
            #     )

            # if np.isnan(TSNS):
            #     TEFF = TSPS
            # elif np.isnan(TSPS):
            #     TEFF = TSNS
            # else:
            #     TEFF = np.sqrt(TSNS * TSPS)

            # fom = dict(
            #     zip(
            #         ["CM", "I", "CSPS", "TSNS", "TSPS", "TEFF", "ACC"],
            #         (
            #             df[
            #                 [c for c in df.columns if c in self.knowns_]
            #                 + [self.unknown_class]
            #             ][
            #                 [x in np.unique(actual) for x in df.index]
            #             ],  # Re-order for easy visualization
            #             Itot,
            #             CSPS,
            #             TSNS,
            #             TSPS,
            #             TEFF,
            #             ACC,
            #         ),
            #     )
            # )

        return fom

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
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": True,
            "requires_positive_y": False,
            "_skip_test": ["check_estimators_dtypes"],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
