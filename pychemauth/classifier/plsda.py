"""
Partial-least squares (Projection to Latent Structures) discriminant analysis.

author: nam
"""
import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pychemauth.preprocessing.scaling import CorrectedScaler
from pychemauth.utils import pos_def_mat, CovarianceEllipse

class PLSDA(ClassifierMixin, BaseEstimator):
    """
    PLS-DA for classification.

    Parameters
    ----------
    n_components : scalar(int), optional(default=1)
        Number of dimensions to project into with PLS stage.
        Should be in [1, min(n_samples-1, n_features)].
        See scikit-learn documentation for more details. Sometimes
        K-1 is used as a lower bound instead of 1, where K is
        the number of classes.  This can assist in stability
        issues with the soft version.

    alpha : scalar(float), optional(default=0.05)
        Type I error rate (signficance level).

    gamma : scalar(float), optional(default=0.01)
        Significance level for determining outliers.

    not_assigned : scalar(int) or str, optional(default=-1)
        Category to give a point in soft version if not assigned to any
        known class.

    style : str, optional(default="soft")
        PLS style; can be "soft" or "hard".

    scale_x : scalar(bool), optional(default=True)
        Whether or not to scale the X matrix during the PLS(2) stage.
        This depends on the meaning of X and is up to the user to
        determine if scaling it (by the standard deviation) makes sense.
        Note that X and Y are always centered, Y is never scaled.

    score_metric : str, optional(default="TEFF")
        Which metric to use as the score.  Can be {TEFF, TSNS, TSPS}
        (default=TEFF). TEFF^2 = TSNS*TSPS.

    Note
    ----
    Implements "hard" classification as an "LDA-like" criterion, and a
    "soft" classification using a "QDA-like" criterion as described in [1].
    Soft PLS-DA may assign a point to 0, 1, or >1 classes, while the hard
    PLS-DA always assigns exactly one class to a point.

    This relies on :func:`sklearn.cross_decomposition.PLSRegression` which can
    perform either PLS1 or PLS2; however, here we default to PLS2 and
    always one-hot-encode multiple classes, even in the instance of binary
    classification where PLS1 could be used instead.

    * Note that alpha and gamma are only relevant for the soft version.

    * If y values are going to be passed as strings, "not_assigned" should
    also be a string (e.g., "NOT_ASSIGNED"); if classes are encoded as
    integers passing -1 (default) will signify an unassigned point. This is
    only relevant for the soft version.

    * A rule of thumb for the number of components to use is between K/2(K-1)
    and K/2(K+1) to provide sufficient complexity but avoid overfitting; K is
    the total number of classes. This is not rigorous and may not hold in many
    cases. Cross-validation should be used to evalute this parameter, in general.

    References
    ----------
    [1] "Multiclass partial least squares discriminant analysis: Taking the
    right way - A critical tutorial," Pomerantsev and Rodionova, Journal of
    Chemometrics (2018). https://doi.org/10.1002/cem.3030.
    """

    def __init__(
        self,
        n_components=1,
        alpha=0.05,
        gamma=0.01,
        not_assigned=-1,
        style="soft",
        scale_x=True,
        score_metric="TEFF",
    ):
        """Instantiate the class."""
        self.set_params(
            **{
                "alpha": alpha,
                "gamma": gamma,
                "n_components": n_components,
                "not_assigned": not_assigned,
                "style": style,
                "scale_x": scale_x,
                "score_metric": score_metric,
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
            "alpha": self.alpha,
            "gamma": self.gamma,
            "n_components": self.n_components,
            "not_assigned": self.not_assigned,
            "style": self.style,
            "scale_x": self.scale_x,
            "score_metric": self.score_metric,
        }

    def _check_category_type(self, y):
        """Check that categories are same type as "not_assigned" variable."""
        if self.style.lower() == "soft":
            t_ = None
            for t_ in [(int, np.int32, np.int64), (str,)]:
                if isinstance(self.not_assigned, t_):
                    use_type = t_
                    break
            if t_ is None:
                raise TypeError("not_assigned must be an integer or string")
            if not np.all([isinstance(y_, use_type) for y_ in y]):
                raise ValueError(
                    "You must set the 'not_assigned' variable type ({}) the same \
                    as y, e.g., both must be int or str".format(
                        [type(y_) for y_ in y]
                    )
                )

    def _column_y(self, y):
        """Convert y to column format."""
        y = np.asarray(y)
        if y.ndim != 2:
            y = y[:, np.newaxis]

        return y

    @property
    def categories(self):
        """Return the known categories."""
        check_is_fitted(self, "is_fitted_")
        return copy.copy(self.__ohencoder_.categories_[0])

    def fit(self, X, y):
        """
        Fit the PLS-DA model.

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
        self : PLSDA
        """
        self.n_components = int(
            self.n_components
        )  # scikit-learn PLS does not understand floats
        self.__X_, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
            copy=True,
        )
        self.__y_ = self._column_y(
            y
        )  # scikit-learn expects 1D array, convert to columns

        self.__raw_y_ = copy.copy(self.__y_)
        self.n_features_in_ = self.__X_.shape[1]

        if self.style.lower() not in ["soft", "hard"]:
            raise ValueError("PLSDA style should be either 'soft' or 'hard'.")

        # Dummy check that not_assigned and y have same data types
        self._check_category_type(self.__y_.ravel())

        # 1. Preprocess data (one hot encoding, centering)
        self.__ohencoder_ = OneHotEncoder(
            sparse_output=False, handle_unknown="error"
        )  # Convert integers to OHE
        self.__x_pls_scaler_ = CorrectedScaler(
            with_mean=True, with_std=self.scale_x
        )  # Center and maybe scale X
        self.__y_pls_scaler_ = CorrectedScaler(
            with_mean=True, with_std=False
        )  # Center do not scale Y

        self.__ohencoder_.fit(self.__y_)
        assert self.not_assigned not in set(
            self.__ohencoder_.categories_[0]
        ), "not_assigned value is already taken"
        self.classes_ = np.concatenate(
            (self.__ohencoder_.categories_[0], [self.not_assigned])
        )  # For sklearn compatibility - not used

        self.__class_mask_ = {}
        for i in range(len(self.__ohencoder_.categories_[0])):
            self.__class_mask_[i] = (
                self.__y_ == self.__ohencoder_.categories_[0][i]
            ).ravel()

        self.__y_ = self.__y_pls_scaler_.fit_transform(
            self.__ohencoder_.transform(self.__y_)
        )
        self.__X_ = self.__x_pls_scaler_.fit_transform(self.__X_)

        # 2. PLS2 - bounds based on Rodionova & Pomerantsev but other
        # suggestions exist. scikit-learn suggests an upper bound of
        # "number of classes" but other chemometrics toolkits do not
        # follow this.
        upper_bound = np.min(
            [
                self.__X_.shape[0] - 1,
                self.__X_.shape[1],
            ]
        )

        lower_bound = 1

        # In general, usually K/2(K-1) < N < K/2(K+1), where K is the number of
        # classes, is sometimes heuristically recommended as being optimal.
        # lb = len(self.__ohencoder_.categories_[0])-1 is another rule of thumb
        # which is always less than the first rule (K >=2, which is always true).
        if self.n_components < len(self.__ohencoder_.categories_[0]) - 1:
            warnings.warn(
                "Warning - n_components < number of classes - 1; this may result in instabilities"
            )

        # Note that scikit-learn currently has a typo in its documentation. Only
        # PLSCanonical has an upper bound of min(n_samples, n_features,
        # n_targets) whereas PLSRegression only is bounded by min(n_samples,
        # n_features). We have further lowered the n_samples by 1 for
        # statistical corrections because X is centered, removing 1 DoF.
        # For more discussion see https://scikit-learn.org/stable/\
        # modules/cross_decomposition.html#cross-decomposition
        if self.n_components > upper_bound or self.n_components < lower_bound:
            raise Exception(
                "n_components must [{}, min(n_samples-1 [{}], \
n_features [{}])] = [{}, {}].".format(
                    lower_bound,
                    self.__X_.shape[0] - 1,
                    self.__X_.shape[1],
                    lower_bound,
                    upper_bound,
                )
            )
        self.__pls_ = PLSRegression(
            n_components=self.n_components,
            max_iter=10000,
            tol=1.0e-9,
            scale=False,
        )  # Already scaled as needed, centering is automatic
        _ = self.__pls_.fit(self.__X_, self.__y_)

        y_hat_train = self.__y_pls_scaler_.inverse_transform(
            self.__pls_.predict(self.__X_)
        )

        # 3. Perform PCA on y_hat_train
        self.__pca_ = PCA(
            n_components=len(self.__ohencoder_.categories_[0]) - 1,
            random_state=0,
        )

        # scikit-learn's pca automatically centers
        self.__T_train_ = self.__pca_.fit_transform(y_hat_train)

        # Does centering internally
        self.__class_centers_ = self.__pca_.transform(
            np.eye(len(self.__ohencoder_.categories_[0]))
        )

        # 4. Compute within-class scatter from training set for soft version
        # This is not exactly mean-centered so you cannot use np.cov() to
        # compute it.
        # The class centers are taken as projections of EXACTLY (1,0,0) for
        # example, NOT the mean of class 1.
        # Thus we compute the scatter matrix directly and do not use the
        # covariance of (T-means).T
        if self.style.lower() == "soft":
            self.__S_ = {}
            for i in range(len(self.__ohencoder_.categories_[0])):
                t = (
                    self.__T_train_[self.__class_mask_[i]]
                    - self.__class_centers_[i]
                )
                self.__S_[i] = np.zeros(
                    (self.__T_train_.shape[1], self.__T_train_.shape[1]),
                    dtype=np.float64,
                )
                # Outer product
                for j in range(t.shape[0]):
                    self.__S_[i] += np.dot(
                        t[j, :].reshape(t.shape[1], 1),
                        t[j, :].reshape(t.shape[1], 1).T,
                    )
                self.__S_[i] /= t.shape[
                    0
                ]  # See Ref [1] - centers are known not calculated so do not remove extra DoF
                try:
                    # Check if positive definite.
                    np.linalg.cholesky(self.__S_[i])
                except np.linalg.LinAlgError:
                    try:
                        # If not, try to approximate this matrix.
                        self.__S_[i] = pos_def_mat(self.__S_[i])
                    except Exception as e:
                        raise Exception(
                            "Unable to compute scatter matrix for class {} : \
    {}".format(
                                self.__ohencoder_.categories_[0][i], e
                            )
                        )

        # 4. Continued - compute covariance matrix for hard version
        # Check that covariance of T is diagonal matrix made of eigenvalues
        # from PCA transform. See [1].
        L = np.cov(self.__T_train_.T)
        assert np.allclose(
            L,
            np.eye(len(self.__pca_.explained_variance_))
            * self.__pca_.explained_variance_,
        )
        # Ref [1] seems to not account for this normalization factor, but it only
        # scales all distances the same amount - for hard classification this has no
        # effect since the decision is based on the smallest value, so uniform scaling
        # does not change the result.
        self.__L_ = L  # * (self.__T_train_.shape[0] - 1)
        if self.__L_.ndim == 0:  # When we have a binary problem
            self.__L_ = np.array([[self.__L_]])

        # 5. Compute Mahalanobis critical distance (squared)
        self.__d_crit_ = scipy.stats.chi2.ppf(
            1.0 - self.alpha, len(self.__ohencoder_.categories_[0]) - 1
        )
        self.__d_out_ = [
            scipy.stats.chi2.ppf(
                (1.0 - self.gamma) ** (1.0 / np.sum(self.__class_mask_[i])),
                len(self.__ohencoder_.categories_[0]) - 1,
            )
            for i in range(len(self.__ohencoder_.categories_[0]))
        ]  # Outlier cutoff - these can only be checked for the training set

        self.is_fitted_ = True
        return self

    def check_outliers(self):
        """
        Check if outliers exist in the training data originally fit to.

        Note
        ----
        A point is tested for outlier status only with respect to its class.
        This also, only works for "soft" PLS-DA.

        Returns
        -------
        outliers : ndarray(bool, ndim=1)
            Boolean mask of X_train used in fit() of if each point is
            considered an outlier.
        """
        check_is_fitted(self, "is_fitted_")

        # We can only assess outliers on the training data
        # Others in test set will be "not assigned" and should be assumed
        # correct - just the training stage where we can look at bad data.
        if self.style.lower() != "soft":
            raise Exception("Can only perform outlier check with 'soft' PLSDA")

        outliers = [False] * self.__X_.shape[0]
        for j, t in enumerate(self.__T_train_):
            # Find which class entry j belongs to
            cat = None
            for i in range(len(self.__ohencoder_.categories_[0])):
                if self.__class_mask_[i][j]:
                    cat = i
                    break
            d = np.matmul(
                np.matmul(
                    (t - self.__class_centers_[cat]),
                    np.linalg.inv(self.__S_[cat]),
                ),
                (t - self.__class_centers_[cat]).reshape(-1, 1),
            )[0]
            if d > self.__d_out_[i]:
                outliers[j] = True

        return np.array(outliers)

    def transform(self, X):
        """
        Project X into the feature subspace.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        t-scores : array_like(float, ndim=2)
            Projection of X via PLS, then by PCA into a score space.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        return self.__pca_.transform(
            self.__y_pls_scaler_.inverse_transform(
                self.__pls_.predict(self.__x_pls_scaler_.transform(X))
            )
        )

    def fit_transform(self, X, y):
        """Fit and transform."""
        _ = self.fit(X, y)
        return self.transform(X)

    def mahalanobis(self, X):
        """
        Compute the squared Mahalanobis distance to each class center.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        distance : array_like(float, ndim=1)
            Squared distance to each class for each observation.

        Note
        ----
        Scipy has a built-in function that could replace this in the future.
        Here we compute d^2 whereas scipy evalutes the square root to compute
        d.

        References
        ----------
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=False,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        T_test = self.transform(X)

        distances = []  # Actually squared
        for t in T_test:
            if self.style.lower() == "soft":  # This 'soft' rule is based on QDA
                distances.append(
                    [
                        np.matmul(
                            np.matmul(
                                (t - self.__class_centers_[i]),
                                np.linalg.inv(self.__S_[i]),
                            ),
                            (t - self.__class_centers_[i]).reshape(-1, 1),
                        )[0]
                        for i in range(len(self.__ohencoder_.categories_[0]))
                    ]
                )
            else:  # This 'hard' rule is based on LDA
                distances.append(
                    [
                        np.matmul(
                            np.matmul(
                                (t - self.__class_centers_[i]),
                                np.linalg.inv(self.__L_),
                            ),
                            (t - self.__class_centers_[i]).reshape(-1, 1),
                        )[0]
                        for i in range(len(self.__ohencoder_.categories_[0]))
                    ]
                )
        distances = np.array(distances)
        assert np.all(distances >= 0), "All distances must be >= 0"

        return distances

    def decision_function(self, X, y=None):
        """
        Compute the decision function for each sample.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array_like(str or int, ndim=1), optional(default=None)
            Response. Ignored if it is not used (unsupervised methods).

        Returns
        -------
        decision_function : ndarray
            Shifted, negative distance for each sample.

        Note
        ----
        Following scikit-learn's EllipticEnvelope, this returns the negative
        Mahalanobis distance shifted by the cutoff distance,
        so f < 0 implies an extreme or outlier while f > 0 implies an inlier.

        This is ONLY returned for soft PLSDA; if the hard variant is used a
        NotImplementedError will be raised instead.

        References
        ----------
        See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-decision_function
        """
        check_is_fitted(self, "is_fitted_")
        distances2 = self.mahalanobis(X)

        if self.style.lower() == "soft":
            f = -np.sqrt(distances2) - (-np.sqrt(self.__d_crit_))
        else:
            raise NotImplementedError

        return f

    def predict_proba(self, X, y=None):
        """
        Predict the probability that observations belong each class.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array_like(str or int, ndim=1), optional(default=None)
            Response. Ignored if it is not used (unsupervised methods).

        Returns
        -------
        probabilities : ndarray(float, ndim=2)
            Probability of class membership; columns are ordered according
            to class indices.

        Note
        ----
        Soft PLSDA: assumes each class is normally distributed and uses
        the Mahalanobis distance to compute the (normal) probability
        as a function of this distance from the class' center. The
        cutoff distance was computed using chi-squared statistics such
        that the boundary encompasses 100(1-alpha)% of the true members
        in the final PCA space. These probabilities do NOT sum to 1
        across all categories.

        Hard PLSDA: For a hard model, the softmax function is computed for the
        negative Mahalanobis distances to each class center.
        The column with the highest probability is the prediction, and these
        WILL sum to 1.

        This probability can be used for inspection by SHAP to help explain
        how this makes its decisions, at least with respect to assignment of
        individual class membership.

        This gives the same effective results as predict() except that function
        directly returns the class(es) a point is predicted to belong to and is sorted
        by class likelihood.  No sorting is done here.

        For a soft decision an observation may belong to 1, >1, or 0
        known classes.  The rows will NOT sum to 1 as is convention in scikit-learn.
        Each entry is a simple binary yes/no prediction that the point is an
        inlier for each class.

        The softmax function (hard boundaries) will result in probabilities
        which sum to 1.

        References
        ----------
        See SHAP documentation for a discussion on the utility and impact
        of "squashing functions": https://shap.readthedocs.io/en/latest/\
        example_notebooks/tabular_examples/model_agnostic/Squashing%20Effect.html\
        #Probability-space-explaination

        See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-predict_proba
        """
        check_is_fitted(self, "is_fitted_")
        distances2 = self.mahalanobis(X)
        p = np.exp(-np.clip(distances2 / 2.0, a_max=None, a_min=-500))

        if self.style.lower() == "soft":
            norm = np.zeros(len(self.__S_), dtype=np.float64)
            for i in range(len(norm)):
                norm[i] = np.sqrt(np.linalg.det(2.0 * np.pi * self.__S_[i]))

            prob = np.array(
                [[np.min([1.0, p_]) for p_ in row] for row in (p / norm)],
                dtype=np.float64,
            )
        else:
            # Hard classification predicts one class, so use softmax function on
            # Mahalanobis distances. The Gaussian prefactor is the same for all
            # classes in the "hard" (~det(L)) so it cancels out, if we were
            # computing probabilities as in the soft case.
            prob = (p.T / np.sum(p.T, axis=0)).T

        return prob

    def predict(self, X):
        """
        Predict the class(es) for a given set of features.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        predictions : list() or list(list)
            Predicted classes for each observation.  There may be multiple
            predictions for each entry if `style=soft`, and are listed from left to right in
            order of decreasing likelihood. For Hard PLS-DA only a simple list is
            returned.

        Note
        ----
        If multiple predictions are made, they are ordered according to likelihood,
        from highest to lowest, i.e., by the (lowest) Mahalanobis distance (squared)
        to that class' center.
        """
        check_is_fitted(self, "is_fitted_")
        distances = self.mahalanobis(X)

        # Return all classes within d_crit, sorted from smallest to largest for
        # soft version. "NOT_ASSIGNED" means no assignment.
        predictions = []
        for row in distances:
            d = sorted(
                zip(self.__ohencoder_.categories_[0], row), key=lambda x: x[1]
            )  # The lower d, the higher the certainty of that class
            if self.style.lower() == "soft":
                belongs_to = [x[0] for x in d if x[1] < self.__d_crit_]
                if len(belongs_to) == 0:
                    belongs_to = [self.not_assigned]
            else:
                belongs_to = d[0][
                    0
                ]  # Take the closest class (smallest distance)

            predictions.append(belongs_to)

        return predictions

    def figures_of_merit(self, predictions, actual):
        """
        Compute figures of merit for PLS-DA approaches as in [1].

        Parameters
        ----------
        predictions : array_like(str or int, ndim=2) or array_like(str or int, ndim=1)
            Array of array values containing the predicted class of points (in
            order). Each row may have multiple entries corresponding to
            multiple class predictions in the soft PLS-DA case.

        actual : array_like(str or int, ndim=1)
            Array of ground truth classes for the predicted points.  Should
            have only one class per point.

        Returns
        -------
        fom : dict
            Dictionary object with the following attributes.

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

        Note
        ----
        When making predictions about extraneous classes (not in training set)
        class efficiency (CEFF) is given as simply class specificity (CSPS)
        since class sensitivity (CSNS) cannot be calculated.
        """
        check_is_fitted(self, "is_fitted_")

        # For Hard PLS-DA, internally convert to list(list) so Soft PLS-DA is processed the same way.
        if self.style.lower() == "hard":
            predictions = [[p] for p in predictions]

        trained_classes = np.unique(self.__ohencoder_.categories_)

        # Dummy check that not_assigned and y have same data types
        actual = self._column_y(actual).ravel()
        self._check_category_type(actual)
        assert self.not_assigned not in set(
            actual
        ), "not_assigned value is already taken"

        all_classes = [self.not_assigned] + np.unique(
            np.concatenate((np.unique(actual), trained_classes))
        ).tolist()

        encoder = LabelEncoder()
        encoder.fit(all_classes)
        n_classes = len(all_classes)
        use_classes = encoder.classes_[encoder.classes_ != self.not_assigned]

        n = np.zeros((n_classes, n_classes), dtype=int)
        for row, actual_class in zip(predictions, actual):
            kk = encoder.transform([actual_class])[0]
            for entry in row:
                ll = encoder.transform([entry])[0]
                n[kk, ll] += 1

        df = pd.DataFrame(
            data=n, columns=encoder.classes_, index=encoder.classes_
        )
        df = df[
            df.index != self.not_assigned
        ]  # Trim off final row of "NOT_ASSIGNED" since these are real inputs
        Itot = pd.Series(
            [np.sum(np.array(actual) == kk) for kk in use_classes],
            index=use_classes,
        )
        assert np.sum(Itot) == len(actual)

        # Class-wise FoM
        # Sensitivity is "true positive" rate and is only defined for
        # trained/known classes
        CSNS = pd.Series(
            [
                df[kk][kk] / Itot[kk] if Itot[kk] > 0 else np.nan
                for kk in trained_classes
            ],
            index=trained_classes,
        )

        # Specificity is the fraction of points that are NOT a given class that
        # are correctly predicted to be something besides the class. Thus,
        # specificity can only be computed for the columns that correspond to
        # known classes since we have only trained on them. These are "true
        # negatives". This is always >= 0.
        CSPS = pd.Series(
            [
                1.0
                - np.sum(df[kk][df.index != kk])  # Column sum
                / np.sum(Itot[Itot.index != kk])
                for kk in trained_classes
            ],
            index=trained_classes,
        )

        # If CSNS can't be calculated, using CSPS as efficiency;
        # Oliveri & Downey introduced this "efficiency" used in [1]
        CEFF = pd.Series(
            [
                np.sqrt(CSNS[c] * CSPS[c]) if not np.isnan(CSNS[c]) else CSPS[c]
                for c in trained_classes
            ],
            index=trained_classes,
        )

        # Total FoM

        # Evaluates overall ability to recognize a class is itself.  If you
        # show the model some class it hasn't trained on, it can't be predicted
        # so no contribution to the diagonal.  We will normalize by total
        # number of points shown [1].  If some classes being tested were seen in
        # training they contribute, otherwise TSNS goes down for a class never
        # seen before.  This might seem unfair, but TSNS only makes sense if
        # (1) you are examining what you have trained on or (2) you are
        # examining extraneous objects so you don't calculate this at all.
        TSNS = np.sum([df[kk][kk] for kk in trained_classes]) / np.sum(Itot)

        # If any untrained class is correctly predicted to be "NOT_ASSIGNED" it
        # won't contribute to df[use_classes].sum().sum().  Also, unseen
        # classes can't be assigned to so the diagonal components for those
        # entries is also 0 (df[k][k]).
        TSPS = 1.0 - (
            df[use_classes].sum().sum()
            - np.sum([df[kk][kk] for kk in use_classes])
        ) / np.sum(Itot) / (
            1.0 if self.style.lower() == "hard" else len(trained_classes) - 1.0
        )
        # Soft models can assign a point to all categories which would make this
        # sum > 1, meaning TSPS < 0 would be possible.  By scaling by the total
        # number of classes, TSPS is always positive; TSPS = 0 means all points
        # assigned to all classes (trivial result) vs. TSPS = 1 means no mistakes.

        # Sometimes TEFF is reported as TSPS when TSNS cannot be evaluated (all
        # previously unseen samples).
        TEFF = np.sqrt(TSPS * TSNS)

        return dict(
            zip(
                ["CM", "I", "CSNS", "CSPS", "CEFF", "TSNS", "TSPS", "TEFF"],
                (
                    df[
                        [c for c in df.columns if c in trained_classes]
                        + [self.not_assigned]
                    ][
                        [x in np.unique(actual) for x in df.index]
                    ],  # Re-order for easy visualization
                    Itot,
                    CSNS,
                    CSPS,
                    CEFF,
                    TSNS,
                    TSPS,
                    TEFF,
                ),
            )
        )

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
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        metrics = self.figures_of_merit(self.predict(X), y)
        if self.score_metric.upper() not in metrics:
            raise ValueError(
                "Unrecognized metric : {}".format(self.score_metric.upper())
            )
        else:
            return metrics[self.score_metric.upper()]

    def pls2_coeff(self, classes=None, ax=None, return_coeff=False):
        """
        Plot the coefficients in the PLS2 model to examine variable importance.

        Parameters
        ----------
        classes : list or None, optional(default=None)
            If None, plot coefficients for all categories; otherwise just classes
            specified.

        ax : matplotlib.pyplot.axes, optional(default=None)
            Axes to plot results on.  If None, a new figure is created.

        return_coeff : scalar(bool), optional(default=False)
            Return PLS2 coefficients instead of the figure axis. N x D where D
            is the number of features in X (X.shape[1]) and N is the number of
            categories.

        Returns
        -------
        ax : matplotlib.pyplot.axes or ndarray
            Figure axes being plotted on or the PLS2 coefficients depending on
            the value of `return_coeff`.
        """
        check_is_fitted(self, "is_fitted_")
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        if classes is None:
            plot_classes = self.__ohencoder_.categories_[0]
        else:
            plot_classes = classes

        coeffs = []
        for class_ in plot_classes:
            if class_ in self.__ohencoder_.categories_[0]:
                # https://scikit-learn.org/stable/modules/generated/
                # sklearn.cross_decomposition.PLSRegression.html
                coeffs.append(
                    self.__pls_.coef_[
                        :,
                        np.where(self.__ohencoder_.transform([[class_]])[0])[0][
                            0
                        ],
                    ]
                )
                ax.plot(
                    np.arange(self.__pls_.coef_.shape[0], dtype=int),
                    coeffs[-1],
                    label=class_,
                )

        ax.set_xticks(np.arange(self.__pls_.coef_.shape[0], dtype=int))
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("PLS2 Coefficient")
        ax.legend(loc="best")

        if return_coeff:
            return np.array(coeffs)
        else:
            return ax

    def visualize(self, styles=None, ax=None):
        """
        Plot training results in 1D or 2D automatically.

        Parameters
        ----------
        styles : list, optional(default=None)
            List of styles to plot, e.g., ["hard", "soft"]. This can always
            include ["hard"], but "soft" is only possible if the class was
            instantiated to be use the "soft" style boundaries.

        ax : matplotlib.pyplot.axes, optional(default=None)
            Axes to plot results on.  If None, a new figure is created.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Figure axes being plotted on.
        """
        check_is_fitted(self, "is_fitted_")
        ndim = len(self.__class_centers_) - 1
        if ndim == 1:
            return self.visualize_1d(styles=styles, ax=ax)
        elif ndim == 2:
            return self.visualize_2d(styles=styles, ax=ax)
        else:
            raise Exception(
                "Unable to visualize {} class results ({} dimensions).".format(
                    ndim + 1, ndim
                )
            )

    def visualize_1d(self, styles=None, ax=None):
        """
        Plot 1D training results.

        Parameters
        ----------
        styles : list, optional(default=None)
            List of styles to plot, e.g., ["hard", "soft"]. This can always
            include ["hard"], but "soft" is only possible if the class was
            instantiated to be use the "soft" style boundaries.

        ax : matplotlib.pyplot.axes, optional(default=None)
            Axes to plot results on.  If None, a new figure is created.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Figure axes being plotted on.

        Note
        ----
        This can only be done when we have K=2 training classes because the
        one-hot-encoded classes are projected into K-1=1 dimensions.  This
        can still be a helpful visualization tool if you consider 2 classes
        at a time.

        Also note that the test set can contain other (more) classes, it is
        just that the training stage must rely on only 2 for this to work.

        You can plot test set results on the axes first, then pass that object
        to view these results on the same plot.
        """
        check_is_fitted(self, "is_fitted_")
        if len(self.__class_centers_) != 2:
            raise Exception(
                "Can only do 1D visualization with systems trained on 2 classes."
            )

        def soft_boundaries_1d(rmax=10.0, rbins=1000):
            """
            Compute the bounding ellipse around for "soft" classification.

            Parameters
            ----------
            rmax : scalar(float), optional(default=10.0)
                Radius to go from class center to look for boundary.
                Since these are in normalized score space (projection of OHE
                simplex) one order of magnitude higher (i.e., 10) is usually a
                good bound.

            rbins : scalar(int), optional(default=1000)
                Number of points to seach from class center (r=0 to r=rmax) for
                boundary.

            Returns
            -------
            cutoff : list(ndarray)
                2D array of points for each class (ordered according to
                class_centers) delineating the extremes boundary.

            outlier : list(ndarray)
                2D array of points for each class (ordered according to
                class_centers) delineating the outlier boundary.
            """

            def estimate_boundary(rmax, rbins, style="cutoff"):
                cutoff = []
                for i in range(len(self.__class_centers_)):
                    cutoff.append([])
                    c = self.__class_centers_[i]
                    # For each center, choose a systematic orientation
                    for direction in [+1, -1]:
                        # Walk "outward" until you meet the threshold
                        for r in np.linspace(0, rmax, rbins):
                            sPC = c + r * direction
                            d = np.matmul(
                                np.matmul(
                                    (sPC - c),
                                    np.linalg.inv(self.__S_[i]),
                                ),
                                (sPC - c).reshape(-1, 1),
                            )[0]
                            if d > (
                                self.__d_crit_
                                if style == "cutoff"
                                else self.__d_out_[i]
                            ):
                                cutoff[i].append(sPC)
                                break
                return [np.array(x) for x in cutoff]

            cutoff = estimate_boundary(rmax=rmax, rbins=rbins, style="cutoff")
            outlier = estimate_boundary(rmax=rmax, rbins=rbins, style="outlier")

            return cutoff, outlier

        def hard_boundaries_1d():
            """
            Obtain the hard boundary between the two classes.

            Returns
            -------
            t0 : scalar(float)
                Threshold sPC dividing the two classes.
            """

            def get_v(i):
                """Eq. 9 in [1]."""
                return (
                    np.matmul(
                        np.matmul(
                            self.__class_centers_[i], np.linalg.inv(self.__L_)
                        ),
                        self.__class_centers_.T[:, i],
                    )
                    / 2.0
                )

            # Eq. 10 in [1]
            t0 = np.matmul(
                np.matmul(
                    np.array(
                        [get_v(i) for i in range(len(self.__class_centers_))]
                    ),
                    self.__pca_.components_.T,
                ),
                self.__L_,
            )

            return t0

        if styles is None:
            styles = [self.style.lower()]
        else:
            styles = [a.lower() for a in styles]

        if "soft" in styles and self.style.lower() != "soft":
            raise ValueError(
                "Style must be 'soft' to visualize soft boundaries."
            )

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        for i, c_ in enumerate(self.__ohencoder_.categories_[0]):
            mask = self.__raw_y_.ravel() == c_
            ax.plot(
                self.__T_train_[mask],
                [i] * np.sum(mask),
                "o",
                alpha=0.5,
                color="C{}".format(i),
                label=str(c_) + " (Training)",
            )
            ax.plot(
                self.__class_centers_[i],
                [i],
                "ks",
                alpha=1,
            )
        ax.set_xlabel("sPC1")

        if "soft" in styles:
            cutoff, outlier = soft_boundaries_1d(rmax=10.0, rbins=1000)
            for i in range(len(cutoff)):
                ax.plot(
                    [cutoff[i][0]] * 100,
                    np.linspace(-1 / 3.0 + i, 1 / 3.0 + i, 100),
                    color="C{}".format(i),
                )
                ax.plot(
                    [cutoff[i][1]] * 100,
                    np.linspace(-1 / 3.0 + i, 1 / 3.0 + i, 100),
                    color="C{}".format(i),
                )
                ax.plot(
                    [outlier[i][0]] * 100,
                    np.linspace(-1 / 3.0 + i, 1 / 3.0 + i, 100),
                    linestyle="--",
                    color="C{}".format(i),
                )
                ax.plot(
                    [outlier[i][1]] * 100,
                    np.linspace(-1 / 3.0 + i, 1 / 3.0 + i, 100),
                    linestyle="--",
                    color="C{}".format(i),
                )
        if "hard" in styles:
            t0 = hard_boundaries_1d()
            ax.axvline(t0, color="k")

        ax.legend(loc="best")
        ax.set_ylim(-0.5, 0.5 + len(self.__class_centers_) - 1)
        ax.set_yticks([])

        return ax

    def visualize_2d(self, styles=None, ax=None):
        """
        Plot 2D training data results.

        Parameters
        ----------
        styles : list, optional(default=None)
            List of styles to plot, e.g., ["hard", "soft"]. This can always
            include ["hard"], but "soft" is only possible if the class was
            instantiated to be use the "soft" style boundaries.

        ax : matplotlib.pyplot.axes, optional(default=None)
            Axes to plot results on.  If None, a new figure is created.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Figure axes being plotted on.

        Note
        ----
        This can only be done when we have K=3 training classes because the
        one-hot-encoded classes are projected into K-1=2 dimensions.  This
        can still be a helpful visualization tool if you consider 3 classes
        at a time.

        Also note that the test set can contain other (more) classes, it is
        just that the training stage must rely on only 3 for this to work.

        You can plot test set results on the axes first, then pass that object
        to view these results on the same plot.
        """
        check_is_fitted(self, "is_fitted_")
        if len(self.__class_centers_) != 3:
            raise Exception(
                "Can only do 2D visualization with systems trained on 3 classes."
            )

        def soft_boundaries_2d(rmax=10.0, rbins=1000, tbins=90):
            """
            Compute the bounding ellipse around for "soft" classification.

            Parameters
            ----------
            rmax : scalar(float), optional(default=10.0)
                Radius to g from class center to look for boundary.
                Since these are in normalized score space (projection of OHE
                simplex) one order of magnitude higher (i.e., 10) is usually a
                good bound.

            rbins : scalar(int), optional(default=1000)
                Number of points to seach from class center (r=0 to r=rmax) for
                boundary.

            tbins : scalar(int), optional(default=90)
                Number of bins to split [0, 2*pi) into around the class center.

            Returns
            -------
            cutoff : list(ndarray)
                2D array of points for each class (ordered according to
                class_centers) delineating the extremes boundary.

            outlier : list(ndarray)
                2D array of points for each class (ordered according to
                class_centers) delineating the outlier boundary.
            """

            def estimate_boundary(rmax, rbins, tbins, style="cutoff"):
                cutoff = []
                for i in range(len(self.__class_centers_)):
                    cutoff.append([])
                    c = self.__class_centers_[i]
                    # For each center, choose a systematic orientation
                    for theta in np.linspace(0, 2 * np.pi, tbins):
                        # Walk "outward" until you meet the threshold
                        for r in np.linspace(0, rmax, rbins):
                            sPC = c + r * np.array(
                                [np.cos(theta), np.sin(theta)]
                            )

                            d = np.matmul(
                                np.matmul(
                                    (sPC - c),
                                    np.linalg.inv(self.__S_[i]),
                                ),
                                (sPC - c).reshape(-1, 1),
                            )[0]
                            if d > (
                                self.__d_crit_
                                if style == "cutoff"
                                else self.__d_out_[i]
                            ):
                                cutoff[i].append(sPC)
                                break
                return [np.array(x) for x in cutoff]

            cutoff = estimate_boundary(
                rmax=rmax, rbins=rbins, tbins=tbins, style="cutoff"
            )
            outlier = estimate_boundary(
                rmax=rmax, rbins=rbins, tbins=tbins, style="outlier"
            )

            return cutoff, outlier

        def hard_boundaries_2d(maxp=1000, rmax=2.0, dx=0.05):
            """
            Obtain points along the hard boundaries between classes.

            Parameters
            ----------
            maxp : scalar(int), optional(default=1000)
                Maximum number of points to use along a line.

            rmax : scalar(float), optional(default=2.0)
                Maximum radius from intersection to compute lines.

            dx : scalar(float), optional(default=0.05)
                Delta x along lines.

            Returns
            -------
            lines : dict(tuple, ndarray)
                Dictionary of class index pairs (e.g., (0,1) based on
                class_center ordering) and (x,y) coordinates in sPC space
                which define the discriminating line between classes.
            """

            def get_v(i):
                """Eq. 9 in [1]."""
                return (
                    np.matmul(
                        np.matmul(
                            self.__class_centers_[i], np.linalg.inv(self.__L_)
                        ),
                        self.__class_centers_.T[:, i],
                    )
                    / 2.0
                )

            def get_w(i):
                """Eq. 9 in [1]."""
                return np.matmul(
                    self.__class_centers_[i], np.linalg.inv(self.__L_)
                )

            def get_nebr_pairs(t0):
                """Neighbors are ordered counterclockwise on a circle."""
                angle = {}
                for i in range(len(self.__class_centers_)):
                    dv = self.__class_centers_[i] - t0
                    angle[i] = np.arctan2(dv[1], dv[0]) + 2 * np.pi

                cc_order = sorted(angle, key=lambda x: angle[x])
                unrolled = cc_order + cc_order + cc_order
                lco = len(cc_order)
                pairs = list(
                    zip(
                        unrolled[lco : lco + lco],
                        unrolled[lco + 1 : lco + lco + 1],
                    )
                )

                return pairs

            # Eq. 10 in [1]
            t0 = np.matmul(
                np.matmul(
                    np.array(
                        [get_v(i) for i in range(len(self.__class_centers_))]
                    ),
                    self.__pca_.components_.T,
                ),
                self.__L_,
            )
            pairs = get_nebr_pairs(t0)

            # Determine which direction is "outward" from t0 for each pair
            sign = []
            for i, j in pairs:
                mid = (
                    self.__class_centers_[i] + self.__class_centers_[j]
                ) / 2.0
                sign.append(
                    np.sign(
                        mid[0]
                        - np.mean([p_[0] for p_ in self.__class_centers_])
                    )
                )

            lines = {}
            for sign, (i, j) in list(zip(sign, pairs)):
                dv = get_v(i) - get_v(j)
                dw = get_w(i) - get_w(j)

                pts = [t0.tolist()]
                for k in range(1, maxp):
                    x_ = pts[-1][0] + float(dx) * sign
                    pts.append([x_, (dw[0] * x_ - dv) / -dw[1]])
                    # Stop after (if) some rmax is reached
                    if (
                        np.sqrt(
                            (pts[-1][0] - t0[0]) ** 2
                            + (pts[-1][1] - t0[1]) ** 2
                        )
                        > rmax
                    ):
                        break
                lines[(i, j)] = np.array(pts)

            return lines

        if styles is None:
            styles = [self.style.lower()]
        else:
            styles = [a.lower() for a in styles]

        if "soft" in styles and self.style.lower() != "soft":
            raise ValueError(
                "Style must be 'soft' to visualize soft boundaries."
            )

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        for i, c_ in enumerate(self.__ohencoder_.categories_[0]):
            mask = self.__raw_y_.ravel() == c_
            ax.plot(
                self.__T_train_[mask, 0],
                self.__T_train_[mask, 1],
                "o",
                alpha=0.5,
                color="C{}".format(i),
                label=str(c_) + " (Training)",
            )
        ax.plot(
            self.__class_centers_[:, 0],
            self.__class_centers_[:, 1],
            "ks",
            alpha=1,
            label="Training Class Centers",
        )
        ax.axis("equal")
        ax.set_xlabel("sPC1")
        ax.set_ylabel("sPC2")

        if "soft" in styles:
            cutoff, outlier = soft_boundaries_2d(
                rmax=10.0, rbins=1000, tbins=90
            )
            for i in range(len(cutoff)):
                ax.plot(cutoff[i][:, 0], cutoff[i][:, 1], color="C{}".format(i))
                ax.plot(
                    outlier[i][:, 0],
                    outlier[i][:, 1],
                    linestyle="--",
                    color="C{}".format(i),
                )

            for i, c_ in enumerate(self.__ohencoder_.categories_[0]):
                mask = self.__raw_y_.ravel() == c_
                ellipse = CovarianceEllipse(
                    method='empirical',
                    center=self.__class_centers_[i]
                ).fit(
                    self.__T_train_[mask,:2],
                )

                # Plot the inlier boundary
                _ = ellipse.visualize(ax=ax, alpha=self.alpha, ellipse_kwargs={'alpha':0.3, 'color':f'C{i}'})

                # Plot the outlier boundary
                _ = ellipse.visualize(ax=ax, alpha=1.0-(1.0 - self.gamma) ** (1.0 / np.sum(self.__class_mask_[i])), ellipse_kwargs={'alpha':0.0, 'ls':'--', 'linecolor':f'C{i}'})

        if "hard" in styles:
            lines = hard_boundaries_2d(maxp=1000, rmax=2.0, dx=0.05)
            for k in lines.keys():
                ax.plot(lines[k][:, 0], lines[k][:, 1], "k-")

        ax.legend(loc="best")

        return ax

    def _get_tags(self):
        """For compatibility with scikit-learn >=0.21."""
        return {
            "allow_nan": False,
            "array_api_support": False,
            "binary_only": False,
            "multilabel": True,
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
                "check_dtype_object",  # Causes singular matrix
            ],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
