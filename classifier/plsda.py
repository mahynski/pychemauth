"""
Partial-least squares (Projection to Latent Structures) discriminant analysis.

author: nam
"""
import numpy as np
import pandas as pd
import scipy
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class PLSDA:
    """
    PLS-DA for classification.

    Implements 'hard' classification as an
    'LDA-like' criterion, and a 'soft' classification using a 'QDA-like'
    criterion as described in [1].  Soft PLS-DA may assign a point to 0, 1,
    or >1 classes, while the hard PLS-DA always assigns exactly one class
    to a point.

    Notes
    -----
    * Note that alpha and gamma are only relevant for the soft version.
    * The soft version can become unstable if n_components is too small and
    return negative distances to class centers; this results in an error -
    try increasing n_components if this happens.
    * If y is going to be passed as strings, 'not_assigned' should also be
    set to a string (e.g., "NOT_ASSIGNED"); if classes are encoded as
    integers passing -1 (default) will signify an unassigned point. This is
    only relevant for the soft version.

    [1] "Multiclass partial least squares discriminant analysis: Taking the
    right way - A critical tutorial," Pomerantsev and Rodionova, Journal of
    Chemometrics (2018). https://doi.org/10.1002/cem.3030.
    """

    def __init__(
        self,
        n_components,
        alpha=0.05,
        gamma=0.05,
        not_assigned=-1,
        style="soft",
    ):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of dimensions to project into with PLS stage.
        alpha : float
            Type I error rate (signficance level).
        gamma : float
            Significance level for determining outliers.
        not_assigned : int, str
            Category to give a point in soft version if not assigned to any
            known class.
        style : str
            PLS style; can be "sfot" or "hard".
        """
        self.set_params(
            **{
                "alpha": alpha,
                "gamma": gamma,
                "n_components": n_components,
                "not_assigned": not_assigned,
                "style": style,
            }
        )

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "n_components": self.n_components,
            "not_assigned": self.not_assigned,
            "style": self.style,
        }

    def check_category_type_(self, y):
        """Check that categories are same type as 'not_assigned' variable."""
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
                    use_type
                )
            )

    def column_y_(self, y):
        """Convert y to column format."""
        y = np.array(y)
        if y.ndim != 2:
            y = y.reshape(-1, 1)
        return y

    def fit(self, X, y):
        """
        Fit the PLS-DA model.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : array-like
            Ground truth classes - will be converted to numpy array
            automatically.

        Returns
        -------
        self
        """
        self.__X_ = np.array(X).copy()
        self.__y_ = self.column_y_(y)
        self.n_features_in_ = self.__X_.shape[1]

        if self.__X_.shape[0] != self.__y_.shape[0]:
            raise ValueError("X and y shapes are not compatible")

        # Dummy check that not_assigned and y have same data types
        self.check_category_type_(self.__y_.ravel())

        # 1. Preprocess data (one hot encoding, centering)
        self.__ohencoder_ = OneHotEncoder(
            sparse=False
        )  # Convert integers to OHE
        self.__x_pls_scaler_ = StandardScaler(
            with_mean=True, with_std=True
        )  # Center and scale X
        self.__y_pls_scaler_ = StandardScaler(
            with_mean=True, with_std=False
        )  # Center do not scale Y

        self.__ohencoder_.fit(self.__y_)
        assert self.not_assigned not in set(
            self.__ohencoder_.categories_[0]
        ), "not_assigned value is already taken"

        self.__class_mask_ = {}
        for i in range(len(self.__ohencoder_.categories_[0])):
            self.__class_mask_[i] = (
                self.__y_ == self.__ohencoder_.categories_[0][i]
            ).ravel()

        self.__y_ = self.__y_pls_scaler_.fit_transform(
            self.__ohencoder_.transform(self.__y_)
        )
        self.__X_ = self.__x_pls_scaler_.fit_transform(self.__X_)

        # 2. PLS2
        self.__plsda_ = PLSRegression(
            n_components=self.n_components,
            max_iter=5000,
            tol=1.0e-9,
            scale=False,
        )  # Already scaled, centered as needed
        _ = self.__plsda_.fit(self.__X_, self.__y_)
        y_hat_train = self.__plsda_.predict(self.__X_)

        # 3. Perform PCA on y_hat_train
        self.__y_pca_scaler_ = StandardScaler(with_mean=True, with_std=False)
        self.__pca_ = PCA(
            n_components=len(self.__ohencoder_.categories_[0]) - 1,
            random_state=0,
        )
        self.__T_train_ = self.__pca_.fit_transform(
            self.__y_pca_scaler_.fit_transform(y_hat_train)
        )

        self.__class_centers_ = self.__pca_.transform(
            self.__y_pca_scaler_.transform(
                np.eye(len(self.__ohencoder_.categories_[0]))
            )
        )

        # 4. Compute within-class scatter from training set for soft version
        # This is not exactly mean-centered so you cannot use np.cov() to
        # compute it.
        # The class centers are taken as projections of EXACTLY (1,0,0) for
        # example, NOT the mean of class 1.
        # Thus we compute the scatter matrix directly and do not use the
        # covariance of (T-means).T
        self.__S_ = {}
        for i in range(len(self.__ohencoder_.categories_[0])):
            self.__S_[i] = np.zeros(
                (self.__T_train_.shape[1], self.__T_train_.shape[1]),
                dtype=np.float64,
            )
            for t in self.__T_train_[self.__class_mask_[i]]:
                # Same as an outer product
                self.__S_[i] += (
                    (t - self.__class_centers_[i])
                    .reshape(-1, 1)
                    .dot((t - self.__class_centers_[i]).reshape(-1, 1).T)
                )
            self.__S_[i] /= np.sum(self.__class_mask_[i])

        # 4. continued - compute covariance matrix for hard version
        # Check that covariance of T is diagonal matrix made of eigenvalues
        # from PCA transform. See [1].
        L = np.cov(self.__T_train_.T)
        assert np.allclose(
            L,
            np.eye(len(self.__pca_.explained_variance_))
            * self.__pca_.explained_variance_,
        )
        self.__L_ = L

        # 5. Compute Mahalanobis critical distance
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

        return self

    def check_outliers(self):
        """
        Check if outliers exist in the training data originally fit to.

        Returns
        -------
        outliers : ndarray
            Boolean mask of X_train used in fit() of if each point is
            considered an outlier.
        """
        # We can only assess outliers on the training data
        # Others in test set will be "not assigned" and should be assumed
        # correct - just the training stage where we can look at bad data.
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
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        t-scores : matrix-like
            Projection of X via PLS, then by PCA into a score space.
        """
        X = np.array(X)
        assert X.shape[1] == self.n_features_in_
        X = self.__x_pls_scaler_.transform(X)

        y_hat_test = self.__plsda_.predict(X)

        T_test = self.__pca_.transform(
            self.__y_pca_scaler_.transform(y_hat_test)
        )

        return T_test

    def predict(self, X):
        """
        Predict the class(es) for a given set of features.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        predictions : list(list)
            Predicted classes for each observation.  There may be multiple
            predictions for each entry, and are listed fro left to right in
            order of decreasing likelihood.
        """
        T_test = self.transform(X)

        distances = []
        for t in T_test:
            if self.style == "soft":  # This 'soft' rule is based on QDA
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
        assert np.all(np.array(distances) >= 0), "All distances must be >= 0"

        # Return all classes within d_crit, sorted from smallest to largest for
        # soft version. "NOT_ASSIGNED" means no assignment.
        predictions = []
        for row in distances:
            d = sorted(
                zip(self.__ohencoder_.categories_[0], row), key=lambda x: x[1]
            )  # The lower d, the higher the certainty of that class
            if self.style == "soft":
                belongs_to = [x[0] for x in d if x[1] < self.__d_crit_]
                if len(belongs_to) == 0:
                    belongs_to = [self.not_assigned]
            else:
                belongs_to = [
                    d[0][0]
                ]  # Take the closest class (smallest distance)

            predictions.append(belongs_to)

        return predictions

    def figures_of_merit(self, predictions, actual):
        """
        Compute figures of merit for PLS-DA approaches as in [1].

        Parameters
        ----------
        predictions : list(list)
            Array of array values containing the predicted class of points (in
            order). Each row may have multiple entries corresponding to
            multiple class predictions in the soft PLS-DA case.
        actual : array-like
            Array of ground truth classes for the predicted points.  Should
            have only one class per point.

        Returns
        -------
        df : pandas.DataFrame
            Inputs (index) vs. predictions (columns).
        I : pandas.Series
            Number of each class asked to classify.
        CSNS : pandas.Series
            Class sensitivity.
        CSPS : pandas.Series
            Class specificity.
        CEFF : pandas.Series
            Class efficiency.
        TSNS : float64
            Total sensitivity.
        TSPS : float64
            Total specificity.
        TEFF : float64
            Total efficiency.
        """
        trained_classes = np.unique(self.__ohencoder_.categories_)

        # Dummy check that not_assigned and y have same data types
        actual = self.column_y_(actual).ravel()
        self.check_category_type_(actual)
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

        n = np.zeros((n_classes, n_classes), dtype=np.int)
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
        # negatives".
        CSPS = pd.Series(
            [
                1.0
                - np.sum(df[kk][df.index != kk])
                / np.sum(Itot[Itot.index != kk])
                for kk in trained_classes
            ],
            index=trained_classes,
        )

        # If CSNS can't be calculated, using CSPS as efficiency
        # Oliveri & Downey introduced this "efficiency" used by P&R
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
        # number of points shown.  If a model is trained on classes that
        # correspond to 80% of the training set, and all of those are
        # classified perfectly, then TSNS = 0.8.
        TSNS = np.sum([df[kk][kk] for kk in use_classes]) / np.sum(Itot)

        # If any untrained class is correctly predicted to be "NOT_ASSIGNED" it
        # won't contribute to df[use_classes].sum().sum().  Also, unseen
        # classes can't be assigned to so the diagonal components for those
        # entries is also 0 (df[k][k]).
        TSPS = 1.0 - (
            df[use_classes].sum().sum()
            - np.sum([df[kk][kk] for kk in use_classes])
        ) / np.sum(Itot)

        # A very bad classifier could assign wrong classes often and with soft
        # method TSPS < 0 is possible.
        TSPS = np.max([0, TSPS])
        TEFF = np.sqrt(TSPS * TSNS)

        # Only return FoM for classes seen during training
        return (
            df[
                [c for c in df.columns if c != self.not_assigned]
                + [self.not_assigned]
            ],  # Re-order for easy visualization
            Itot,
            CSNS,
            CSPS,
            CEFF,
            TSNS,
            TSPS,
            TEFF,
        )

    def score(self, X, y, use="TEFF"):
        """
        Score the prediction.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : array-like
            Ground truth classes - will be converted to numpy array
            automatically.
        use : str
            Which metric to use as the score.  Can be {TEFF, TSNS, TSPS}
            (default=TEFF).

        Returns
        -------
        score : float
            Score
        """
        X, y = np.array(X), np.array(y)
        df, I, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = self.figures_of_merit(
            self.predict(X), y
        )
        metrics = {"teff": TEFF, "tsns": TSNS, "tsps": TSPS}
        return metrics[use.lower()]

    def _get_tags(self):
        """For compatibility with sklearn >=0.21."""
        return {
            "allow_nan": False,
            "binary_only": False,
            "multilabel": False,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "preserves_dtype": [np.float64],
            "poor_score": False,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": True,
            "requires_positive_y": False,
            "_skip_test": True,  # Skip since get_tags is unstable anyway
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
