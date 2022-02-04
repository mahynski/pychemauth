"""
Soft independent modeling of class analogies.

author: nam
"""
import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

sys.path.append("../")
from chemometrics.utils import CustomScaler, estimate_dof


class SIMCA_Classifier(ClassifierMixin, BaseEstimator):
    """
    Train a SIMCA model for a target class.

    Essentially, a SIMCA model is trained for one target class. The target is
    set when this class is instantiated and must be one of (but doesn't need
    to be the only) class found in the training set (this is checked
    automatically).  During testing (.score()), points are broken up by
    alternative class and run through the model to see how well the target class
    can be distinguished from each alternative.  This allows you to pass points
    that belong to other classes during training - they are just ignored.  This
    is important for integration with other scikit-learn, etc. workflows.
    """

    def __init__(
        self,
        n_components=1,
        alpha=0.05,
        target_class=None,
        style="dd-simca",
        use="TEFF",
        scale_x=True,
    ):
        """
        Instantiate the classifier.

        Outlier detection may be done on a per-class basis, but is not
        part of the "overall" model.  This ultimately performs K different
        checks for the K classes tested on, to see if the target class can be
        differentiated from them.  This relies on the type I error (alpha) only.
        The final metric used to rate the overall model can be set to TEFF or TSPS,
        for example, if you wish to change how the model is evaluated.

        Parameters
        ----------
        n_components : int
            Number of components to use in the SIMCA model.
        alpha : float
            Significance level for SIMCA model.
        target_class : str or int
            The class used to fit the SIMCA model; the rest are used
            to test specificity.
        style : str
            Type of SIMCA to use ("simca" or "dd-simca")
        use : str
            Which metric to use as the score.  Can be {TEFF, TSNS, TSPS}
            (default=TEFF).
        scale_x : bool
            Whether or not to scale X by its sample standard deviation or not.
            This depends on the meaning of X and is up to the user to
            determine if scaling it (by the standard deviation) makes sense.
            Note that X is always centered.
        """
        self.set_params(
            **{
                "n_components": n_components,
                "alpha": alpha,
                "target_class": target_class,
                "style": style,
                "use": use,
                "scale_x": scale_x,
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
            "n_components": self.n_components,
            "alpha": self.alpha,
            "target_class": self.target_class,
            "style": self.style,
            "use": self.use,
            "scale_x": self.scale_x,
        }

    def fit(self, X, y):
        """
        Fit the SIMCA model.

        Only data of the target class will be used for fitting, though more
        can be provided. This is important in pipelines, for example, when
        SMOTE is used to up-sampled minority classes; in that case, those
        must be part of the pipeline for those steps to work automatically.
        However, a user may manually provide only the data of interest.

        Parameters
        ----------
        X : ndarray
            Inputs
        y : ndarray
            Class labels or indices. Should include some data of
            'target_class'.
        """
        if scipy.sparse.issparse(X) or scipy.sparse.issparse(y):
            raise ValueError("Cannot use sparse data.")
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]

        # Fit model to target data
        if self.style == "simca":
            self.__model_ = SIMCA_Model(
                n_components=self.n_components,
                alpha=self.alpha,
                scale_x=self.scale_x,
            )
        elif self.style == "dd-simca":
            self.__model_ = DDSIMCA_Model(
                n_components=self.n_components,
                alpha=self.alpha,
                scale_x=self.scale_x,
            )
        else:
            raise ValueError("{} is not a recognized style.".format(self.style))

        assert self.target_class in np.unique(
            y
        ), "target_class not in training set"
        self.__model_.fit(X[y == self.target_class], y[y == self.target_class])
        self.is_fitted_ = True

        return self

    def transform(self, X):
        """
        Transform into the SIMCA subspace.

        This is not very relevant, but is necessary for scikit-learn compatibility.
        """
        check_is_fitted(self, "is_fitted_")
        return self.__model_.transform(X)

    def fit_transform(self, X, y):
        """
        Fit and transform.

        This is not very relevant, but is necessary for scikit-learn compatibility.
        """
        _ = self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Make a prediction."""
        check_is_fitted(self, "is_fitted_")
        return self.__model_.predict(X)

    @property
    def CSPS(self):
        """Class specificities."""
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.__CSPS_)

    @property
    def TSNS(self):
        """Total sensitivity of the model."""
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.__TSNS_)

    @property
    def TSPS(self):
        """Total specificity of the model."""
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.__TSPS_)

    @property
    def TEFF(self):
        """Total efficiency of the model."""
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.__TEFF_)

    @property
    def model(self):
        """Trained undelying SIMCA Model."""
        check_is_fitted(self, "is_fitted_")
        return copy.deepcopy(self.__model_)

    def score(self, X, y=None):
        """
        Score the model (uses total efficiency as score).

        If scoring a set with only the target class present, returns
        TSNS.  If only alternatives present, returns TSPS.  Otherwise
        returns TEFF as a geometric mean of the two.

        Parameters
        ----------
        X : ndarray
            Inputs.
        y : ndarray
            Class labels or indices
        """
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(X, y, accept_sparse=False)

        self.__alternatives_ = [
            c for c in sorted(np.unique(y)) if c != self.target_class
        ]

        self.__CSPS_ = {}
        for class_ in self.__alternatives_:
            mask = y == class_
            self.__CSPS_[class_] = 1.0 - np.sum(
                self.__model_.predict(X[mask])
            ) / np.sum(mask)

        mask = y != self.target_class
        if np.sum(mask) == 0:
            # Testing on nothing but the target class, can't evaluate TSPS
            self.__TSPS_ = np.nan
        else:
            self.__TSPS_ = 1.0 - np.sum(
                self.__model_.predict(X[mask])
            ) / np.sum(mask)

        mask = y == self.target_class
        if np.sum(mask) == 0:
            # Testing on nothing but alternative classes, can't evaluate TSNS
            self.__TSNS_ = np.nan
        else:
            self.__TSNS_ = np.sum(self.__model_.predict(X[mask])) / np.sum(
                mask
            )  # TSNS = CSNS for SIMCA

        if np.isnan(self.__TSNS_):
            self.__TEFF_ = self.__TSPS_
        elif np.isnan(self.__TSPS_):
            self.__TEFF_ = self.__TSNS_
        else:
            self.__TEFF_ = np.sqrt(self.__TSNS_ * self.__TSPS_)

        metrics = {
            "teff": self.__TEFF_,
            "tsns": self.__TSNS_,
            "tsps": self.__TSPS_,
        }
        return metrics[self.use.lower()]

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
            "poor_score": False,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": True,
            "requires_positive_y": False,
            "_skip_test": True,  # Skip since get_tags is unstable anyway
            "_xfail_checks": {},
            "stateless": False,
            "X_types": ["2darray"],
        }


class SIMCA_Model(ClassifierMixin, BaseEstimator):
    """
    SIMCA model for a single class.

    In general, you need a separate SIMCA object for each class in the dataset
    you wish to characterize. This code is based on implementation described in
    [1].  An F-test is performed based on the squared orthogonal distance (OD);
    if it is in excess of some critical value a point is not assigned to a
    class, otherwise it is.  Since a different SIMCA object is trained to
    characterize different classes, it is possible that testing a point on a
    different SIMCA class will result in multiple class assignments; however,
    each individual SIMCA class is binary.

    1. "Robust classification in high dimensions based on the SIMCA Method,"
    Vanden Branden and Hubert, Chemometrics and Intelligent Laboratory Systems
    79 (2005) 10-21.
    2. "Pattern recognition by means of disjoint principal components models,"
    S. Wold, Pattern Recognition 8 (1976) 127–139.
    3. De Maesschalk et al., Chemometrics and Intelligent Laboratory Systems
    47 (1999) 65-77.
    """

    def __init__(self, n_components, alpha=0.05, scale_x=True):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of PCA components to use to model this class.
        alpha : float
            Significance level.
        scale_x : bool
            Whether or not to scale X by its sample standard deviation or not.
            This depends on the meaning of X and is up to the user to
            determine if scaling it (by the standard deviation) makes sense.
            Note that X is always centered.
        """
        self.set_params(
            **{"n_components": n_components, "alpha": alpha, "scale_x": scale_x}
        )

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {
            "n_components": self.n_components,
            "alpha": self.alpha,
            "scale_x": self.scale_x,
        }

    def column_y_(self, y):
        """Convert y to column format."""
        y = np.array(y)
        if y.ndim != 2:
            y = y.reshape(-1, 1)

        return y

    def matrix_X_(self, X):
        """Check that observations are rows of X."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert (
            X.shape[1] == self.n_features_in_
        ), "Incorrect number of features given in X."

        return X

    def fit(self, X, y=None):
        """
        Fit the SIMCA model.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        self
        """
        self.__X_ = np.array(X).copy()
        assert self.__X_.ndim == 2, "Expect 2D feature (X) matrix."
        self.n_features_in_ = self.__X_.shape[1]

        if (
            self.n_components
            > np.min([self.n_features_in_, self.__X_.shape[0]]) - 1
        ):
            raise Exception("Reduce the number of PCA components")

        # 1. Standardize X
        self.__ss_ = CustomScaler(with_mean=True, with_std=self.scale_x)

        # 2. Perform PCA on standardized coordinates
        self.__pca_ = PCA(n_components=self.n_components, random_state=0)
        self.__pca_.fit(self.__ss_.fit_transform(self.__X_))

        # 3. Compute critical F value
        II, JJ, KK = self.__X_.shape[0], self.__X_.shape[1], self.n_components
        if II > JJ:  # See De Maesschalk et al. Chem. Intell. Lab. Sys. 47 1999
            self.__a_ = JJ
        else:
            self.__a_ = II - 1
        self.__f_crit_ = scipy.stats.f.ppf(
            1.0 - self.alpha, self.__a_ - KK, (self.__a_ - KK) * (II - KK - 1)
        )

        return self

    def transform(self, X):
        """
        Project X into the feature subspace.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        t-scores : matrix-like
            Projection of X via PCA into a score space.
        """
        return self.__pca_.transform(self.__ss_.transform(self.matrix_X_(X)))

    def fit_transform(self, X, y=None):
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def distance(self, X):
        """
        Compute the F score (distance) for a given set of observations.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        predictions : ndarray
            F value for each observation.
        """
        II, _, KK = self.__X_.shape[0], self.__X_.shape[1], self.n_components

        X = self.matrix_X_(X)

        X_pred = np.matmul(self.transform(X), self.__pca_.components_)
        # See De Maesschalk et al. Chem. Intell. Lab. Sys. 47 1999
        numer = np.sum((self.__ss_.transform(X) - X_pred) ** 2, axis=1) / (
            self.__a_ - KK
        )

        X_pred = np.matmul(self.transform(self.__X_), self.__pca_.components_)
        # See De Maesschalk et al. Chem. Intell. Lab. Sys. 47 1999
        OD2 = np.sum((self.__ss_.transform(self.__X_) - X_pred) ** 2, axis=1)
        denom = np.sum(OD2) / ((self.__a_ - KK) * (II - KK - 1))

        # F-score for each distance
        F = numer / denom

        return F

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
        predictions : ndarray
            Bolean array of whether a point belongs to this class.
        """
        F = self.distance(X)

        # If f < f_crit, it belongs to the class
        return F < self.__f_crit_

    def score(self, X, y):
        """
        Score the prediction.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : array-like
            Boolean array of whether or not each point belongs to the class.

        Returns
        -------
        score : float
            Accuracy
        """
        y = self.column_y_(y)
        if not isinstance(y[0], bool):
            raise ValueError("y must be provided as a Boolean array")
        X_pred = self.predict(X)
        assert (
            y.shape[0] == X_pred.shape[0]
        ), "X and y do not have the same dimensions."

        return np.sum(X_pred == y.ravel()) / X_pred.shape[0]

    def _get_tags(self):
        """For compatibility with sklearn >=0.21."""
        return {
            "allow_nan": False,
            "binary_only": True,
            "multilabel": False,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "poor_score": False,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,  # Usually true for classifiers, but not for SIMCA
            "requires_positive_y": False,
            "_skip_test": True,  # Skip since get_tags is unstable anyway
            "_xfail_checks": {},
            "stateless": False,
            "X_types": ["2darray"],
        }


class DDSIMCA_Model(ClassifierMixin, BaseEstimator):
    """
    Train a Data-driven SIMCA Model.

    DD-SIMCA uses a combination of OD and SD, modeled by a chi-squared
    distribution, to determine the acceptance criteria to belong to a class.
    The degrees of freedom in this model are estimated from a data-driven
    approach. This implementation follows [1].

    As in SIMCA, this is designed to be a binary classification tool (yes/no)
    for a single class.  A separate object must be trained for each class you
    wish to model.

    [1] "Acceptance areas for multivariate classification derived by projection
    methods," Pomerantsev, Journal of Chemometrics 22 (2008) 601-609.
    """

    def __init__(self, n_components, alpha=0.05, gamma=0.01, scale_x=True):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of PCA components to use to model this class.
        alpha : float
            Significance level.
        gamma : float
            Outlier significance level.
        scale_x : bool
            Whether or not to scale X by its sample standard deviation or not.
            This depends on the meaning of X and is up to the user to
            determine if scaling it (by the standard deviation) makes sense.
            Note that X is always centered.
        """
        self.set_params(
            **{
                "n_components": n_components,
                "alpha": alpha,
                "gamma": gamma,
                "scale_x": scale_x,
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
            "n_components": self.n_components,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "scale_x": self.scale_x,
        }

    def column_y_(self, y):
        """Convert y to column format."""
        y = np.array(y)
        if y.ndim != 2:
            y = y.reshape(-1, 1)

        return y

    def matrix_X_(self, X):
        """Check that observations are rows of X."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert (
            X.shape[1] == self.n_features_in_
        ), "Incorrect number of features given in X."

        return X

    def fit(self, X, y=None):
        """
        Fit the SIMCA model.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.
        y : None, array-like
            This option is available to be consistent with scikit-learn's
            estimator API, however, it is ignored.  Only observations for a single
            class at a time should be passed (i.e., all y should be the same).
            If passed, it is checked that they are all identical and this label
            is used; otherwise the name "Training Class" is assigned.

        Returns
        -------
        self
        """
        self.__X_ = np.array(X).copy()
        assert self.__X_.ndim == 2, "Expect 2D feature (X) matrix."
        self.n_features_in_ = self.__X_.shape[1]

        if y is None:
            self.__label_ = "Training Class"
        else:
            label = np.unique(y)
            if len(label) > 1:
                raise Exception("More than one class passed during training.")
            else:
                self.__label_ = str(label[0])

        if (
            self.n_components
            > np.min([self.n_features_in_, self.__X_.shape[0]]) - 1
        ):
            raise Exception("Reduce the number of PCA components")

        # 1. Standardize X
        self.__ss_ = CustomScaler(with_mean=True, with_std=self.scale_x)

        # 2. Perform PCA on standardized coordinates
        self.__pca_ = PCA(n_components=self.n_components, random_state=0)
        self.__pca_.fit(self.__ss_.fit_transform(self.__X_))

        # 3. Compute critical distance
        h_vals, q_vals = self.h_q_(self.__X_)
        self.__h0_, self.__q0_ = np.mean(h_vals), np.mean(q_vals)
        self.__Nh_, self.__Nq_ = estimate_dof(
            h_vals, q_vals, self.n_components, self.n_features_in_
        )

        self.__c_crit_ = scipy.stats.chi2.ppf(
            1.0 - self.alpha, self.__Nh_ + self.__Nq_
        )

        self.__c_out_ = scipy.stats.chi2.ppf(
            (1.0 - self.gamma) ** (1.0 / self.__X_.shape[0]),
            self.__Nh_ + self.__Nq_,
        )

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Project X into the feature subspace.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        t-scores : matrix-like
            Projection of X via PCA into a score space.
        """
        check_is_fitted(self, "is_fitted_")
        return self.__pca_.transform(self.__ss_.transform(self.matrix_X_(X)))

    def fit_transform(self, X, y=None):
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def h_q_(self, X_raw):
        """Compute the h (SD) and q (OD) distances."""
        X_raw_std = self.__ss_.transform(self.matrix_X_(X_raw))
        T = self.__pca_.transform(X_raw_std)
        X_pred = np.matmul(T, self.__pca_.components_)

        # OD
        q_vals = np.sum((X_raw_std - X_pred) ** 2, axis=1)

        # SD
        h_vals = np.sum(T ** 2 / self.__pca_.explained_variance_, axis=1) / (
            self.__X_.shape[0] - 1
        )

        return h_vals, q_vals

    def distance(self, X):
        """
        Compute how far away points are from this class.

        This is computed as a sum of the OD and OD to be used with acceptance
        rule II from [1].

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.

        Returns
        -------
        distance : ndarray
            Distance to class.
        """
        check_is_fitted(self, "is_fitted_")
        h, q = self.h_q_(self.matrix_X_(X))

        return self.__Nh_ * h / self.__h0_ + self.__Nq_ * q / self.__q0_

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
        predictions : ndarray
            Bolean array of whether a point belongs to this class.
        """
        check_is_fitted(self, "is_fitted_")

        # If c < c_crit, it belongs to the class
        return self.distance(self.matrix_X_(X)) < self.__c_crit_

    def check_outliers(self, X):
        """
        Check if extremes and outliers exist in the data.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        extremes, outliers : ndarray, ndarray
            Boolean mask of X if each point falls between acceptance threshold
            (belongs to class) and the outlier threshold (extreme), or beyond
            the outlier (outlier) threshold.
        """
        check_is_fitted(self, "is_fitted_")

        dX_ = self.distance(self.matrix_X_(X))
        extremes = (self.__c_crit_ <= dX_) & (dX_ < self.__c_out_)
        outliers = dX_ >= self.__c_out_
        return extremes, outliers

    def visualize(self, X, y, ax=None):
        """
        Plot the chi-squared acceptance area with observations.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : matrix-like
            Labels for observations in X.
        ax : matplotlib.pyplot.axes
            Axis object to plot on (optional).
        """
        check_is_fitted(self, "is_fitted_")
        h_lim = np.linspace(0, self.__c_crit_ * self.__h0_ / self.__Nh_, 1000)
        h_lim_out = np.linspace(
            0, self.__c_out_ * self.__h0_ / self.__Nh_, 1000
        )
        q_lim = (
            (self.__c_crit_ - self.__Nh_ / self.__h0_ * h_lim)
            * self.__q0_
            / self.__Nq_
        )
        q_lim_out = (
            (self.__c_out_ - self.__Nh_ / self.__h0_ * h_lim_out)
            * self.__q0_
            / self.__Nq_
        )

        if ax is None:
            fig = plt.figure()
            axis = fig.gca()
        else:
            axis = ax

        axis.plot(
            np.log(1.0 + h_lim / self.__h0_),
            np.log(1.0 + q_lim / self.__q0_),
            "g-",
        )
        axis.plot(
            np.log(1.0 + h_lim_out / self.__h0_),
            np.log(1.0 + q_lim_out / self.__q0_),
            "r-",
        )
        xlim, ylim = (
            1.1 * np.max(np.log(1.0 + h_lim_out / self.__h0_)),
            1.1 * np.max(np.log(1.0 + q_lim_out / self.__q0_)),
        )
        X_ = self.matrix_X_(X)
        y_ = np.array(y)
        markers = [
            "o",
            "v",
            "s",
            "*",
            "+",
            "x",
            "^",
            "<",
            ">",
            "1",
            "2",
            "3",
            "4",
        ]
        for i, class_ in enumerate(sorted(np.unique(y_))):
            h_, q_ = self.h_q_(X_[y_ == class_])
            in_mask = self.predict(X_[y_ == class_])
            ext_mask, out_mask = self.check_outliers(X_[y_ == class_])
            for c, mask, label in [
                (
                    "g",
                    in_mask,
                    class_
                    + " = "
                    + self.__label_
                    + " ({})".format(np.sum(in_mask)),
                ),
                (
                    "orange",
                    ext_mask,
                    class_
                    + " = Extreme "
                    + self.__label_
                    + " ({})".format(np.sum(ext_mask)),
                ),
                (
                    "r",
                    out_mask,
                    class_
                    + " = Outlier "
                    + self.__label_
                    + " ({})".format(np.sum(out_mask)),
                ),
            ]:
                axis.plot(
                    np.log(1.0 + h_[mask] / self.__h0_),
                    np.log(1.0 + q_[mask] / self.__q0_),
                    label=label,
                    marker=markers[i % len(markers)],
                    lw=0,
                    color=c,
                    alpha=0.35,
                )
            xlim = np.max([xlim, 1.1 * np.max(np.log(1.0 + h_ / self.__h0_))])
            ylim = np.max([ylim, 1.1 * np.max(np.log(1.0 + q_ / self.__q0_))])
        axis.legend(bbox_to_anchor=(1, 1))
        axis.set_xlim(0, xlim)
        axis.set_ylim(0, ylim)
        axis.set_xlabel(r"${\rm ln(1 + h/h_0)}$")
        axis.set_ylabel(r"${\rm ln(1 + q/q_0)}$")

        return axis

    def _get_tags(self):
        """For compatibility with sklearn >=0.21."""
        return {
            "allow_nan": False,
            "binary_only": True,
            "multilabel": False,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "poor_score": False,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,  # Usually true for classifiers, but not for SIMCA
            "requires_positive_y": False,
            "_skip_test": True,  # Skip since get_tags is unstable anyway
            "_xfail_checks": {},
            "stateless": False,
            "X_types": ["2darray"],
        }
