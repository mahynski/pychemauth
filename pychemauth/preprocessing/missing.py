"""
Fill in missing data.

author: nam
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pychemauth.preprocessing.scaling import CorrectedScaler


class LOD(TransformerMixin, BaseEstimator):
    """
    Fill in "missing" measurement values and those that are below LOD randomly.

    Parameters
    ----------
    lod : array_like(float, ndim=1)
        Numerical limit of detection for each feature.

    missing_values : scalar(float), optional(default=numpy.nan)
        The value in the X matrix that indicates a missing value.

    seed : scalar(int), optional(default=0)
        Random number generator seed.

    ignore : scalar(float), optional(default=None)
        Anything in X with this value is ignored. You can use this to
        mask certain values as needed; e.g., for future processing.
        If this is set to `np.nan` it overrides `missing_values` and the imputer
        will only operate on values explicitly below the LOD.

    skip_columns : array_like(int, ndim=1), optional(default=None)
        Indices of columns to skip (not impute).

    Note
    ----
    By default, all data in the feature matrix (X) is assumed "missing" because it is
    below the limit of detection (LOD).  However, all values explicitly less than
    the LODs provided are also selected for imputation. In both cases, random values
    are chosen between 0 and the LOD (which must be provided by the user). You may wish
    to change this if you are handling both missing data and data below LOD.

    Example
    -------
    >>> itim = LOD(lod=np.array([0.1, 0.2, 0.1]), ignore=np.nan, seed=42)
    >>> X_lod = itim.fit_transform(missing_X) # Will still have NaN's left representing missing values.
    """

    def __init__(
        self,
        lod=None,
        missing_values=np.nan,
        seed=0,
        ignore=None,
        skip_columns=None,
    ):
        """Instantiate the class."""
        self.set_params(
            **{
                "lod": lod,
                "missing_values": missing_values,
                "seed": seed,
                "ignore": ignore,
                "skip_columns": skip_columns,
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
            "lod": self.lod,
            "missing_values": self.missing_values,
            "seed": self.seed,
            "ignore": self.ignore,
            "skip_columns": self.skip_columns,
        }

    def fit(self, X, y=None):
        """
        Compute the "missing" data.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1)
            Ignored.

        Returns
        -------
        self : LOD
            Fitted model.
        """
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        if self.lod is None:
            self.lod_ = np.array([0] * X.shape[1], dtype=np.float64)
        else:
            # Allow NaN at first
            self.lod_ = check_array(
                self.lod,
                accept_sparse=False,
                dtype=np.float64,
                force_all_finite="allow-nan",
                ensure_2d=False,
                copy=True,
            )
            self.lod_ = self.lod_.ravel()

            # Check NaN not in any columns we are not skipping
            if self.skip_columns is not None:
                mask = np.array([True] * self.lod_.shape[0], dtype=bool)
                for index in self.skip_columns:
                    mask[index] = False
                if np.sum(mask) > 0:
                    _ = check_array(
                        self.lod_[mask],
                        accept_sparse=False,
                        dtype=np.float64,
                        force_all_finite=True,
                        ensure_2d=False,
                        copy=False,
                    )

        self.n_features_in_ = X.shape[1]
        if len(self.lod_) != self.n_features_in_:
            raise ValueError("LOD must be specified for each column in X")

        if self.skip_columns is None:
            self.skip_columns = []
        else:
            if not hasattr(self.skip_columns, "__iter__"):
                raise ValueError(
                    "skip_columns should be an interable list containing integers"
                )

            self.skip_columns = np.asarray(self.skip_columns, dtype=int)
            if (np.max(self.skip_columns) > self.n_features_in_ - 1) or (
                np.min(self.skip_columns) < 0
            ):
                raise ValueError(
                    "All skip_columns should be in the range [0, X.shape[1]-1]"
                )

        self.__rng_ = np.random.default_rng(self.seed)
        self.is_fitted_ = True

        return self

    def fit_transform(self, X, y=None):
        """
        Fill in values of X below LOD with a random value between 0 and LOD.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1)
            Ignored.

        Returns
        -------
        X : array_like(float, ndim=2)
            Feature matrix with data below LOD replaced. If X was supplied as a
            pandas.DataFrame a new DataFrame is returned.
        """
        _ = self.fit(X)

        return self.transform(X)

    def transform(self, X):
        """
        Fill in values of X below LOD with a random value between 0 and LOD.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        Returns
        -------
        X : array_like(float, ndim=2)
            Feature matrix with data below LOD replaced. If X was supplied as a
            pandas.DataFrame a new DataFrame is returned.
        """
        X_checked = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        check_is_fitted(self, "is_fitted_")
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        # Take all missing values as below LOD.
        # Convert to new DataFrame, even if already one, so it works for np arrays, too.
        columns_ = np.arange(0, self.n_features_in_)
        index_ = (
            X.index
            if isinstance(X, pd.DataFrame)
            else np.arange(0, X_checked.shape[0])
        )
        X_df = pd.DataFrame(data=X_checked, columns=columns_, index=index_)
        lod_dict = dict(zip(columns_, self.lod_))

        def impute_(x, lod):
            if (
                lod < 0.0
            ):  # Check in the loop so we only look at LODs actually being used
                raise ValueError("LODs must be non-negative.")

            if self.ignore is not None:
                if (np.isnan(self.ignore) and np.isnan(x)) or (
                    x == self.ignore
                ):
                    return x

            if np.isnan(self.missing_values):
                compare = lambda x: np.isnan(x)
            else:
                compare = lambda x: x == self.missing_values

            if (x < lod) or compare(x):
                return self.__rng_.random() * lod
            else:
                # If NaN not used for missing value, this is ok because (np.nan < float) is never true
                # so the value will be left alone by this loop.
                return x

        for column in X_df.columns:
            if not (column in self.skip_columns):
                X_df[column] = X_df[column].apply(
                    lambda x: impute_(x, lod_dict[column])
                )

        if isinstance(X, pd.DataFrame):
            return X_df.rename(dict(zip(columns_, X.columns)), axis="columns")
        else:
            return X_df.values

    def _get_tags(self):
        """For compatibility with scikit-learn >=0.21."""
        return {
            "allow_nan": True,
            "array_api_support": False,
            "binary_only": False,
            "multilabel": False,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "preserves_dtype": [np.float64],  # Only for transformers
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": ["check_fit_score_takes_y"],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }


class PCA_IA(TransformerMixin, BaseEstimator):
    """
    Use iterative PCA to estimate any missing data values.

    Parameters
    ----------
    n_components : scalar(int), optional(default=1)
        Number of dimensions to project into. Should be in the range
        [1, num_features].

    scale_x : scalar(bool), optional(default=True)
        Whether or not to scale X columns by the standard deviation.

    missing_values : scalar(float) or NaN, optional(default=numpy.nan)
        The value in the X matrix that indicates a missing value.

    max_iters : scalar(int), optional(default=5000)
        Maximum number of iterations of PCA to perform. If convergence is not
        achieved in this limit an Exception is thrown.

    tol : scalar(float), optional(default=1.0e-6)
        Maximum amount any imputed X value is allowed to change between iterations.

    Note
    ----
    If no data is missing during training, the model is still trained so it
    can handle missing data during a test phase.

    First, a simple imputation to the (column) mean is performed. Then PCA is performed
    to model the data, from which the missing values can be estimated.  These
    estimates are used to construct a new feature matrix (X) and this process
    is repeated until convergence.

    This is useful for constructing estimates of missing data in an unsupervised
    fashion. If there is missing data in the response (y), one approach is to
    simply append (numpy.hstack(X, y)) the y array/matrix to X and perform this.
    However, other approaches may be more advisable.

    It is advisable to use cross-validation to identify the optimal number of PCA
    components to use. Importantly, you should only choose n_components so that it
    is never larger than the size of any training fold otherwise an exception will
    be thrown.

    The PCA model (loadings) found during training is fixed and used during testing
    to reconstruct test data if that is also missing.  This is necessary during
    cross-validation, for example, because the original data set may have missing data
    throughout and is subsequently split (repeatedly) so test folds will have some
    elements missing.

    Univariate imputation imputes values in the i-th feature dimension using
    only non-missing values in that dimension (i.e., the average of observed
    values in a column).

    Multivariate imputation uses the entire set of available feature dimensions.
    This PCA_IA algorithm performs multivariate imputation by constructing a PCA
    model for X iteratively until the PCA predictions for the missing data
    converge.

    As pointed out in [1], this corresponds to finding the maximum likelihood
    estimates of the PCA model parameters.

    References
    ----------
    [1] Walczak, B. and Massart, D. L., "Dealing with missing data: Part I,"
    Chemometrics and Intelligent Laboratory Systems 58 (2001) 15-27.

    [2] Walczak, B. and Massart, D. L., "Dealing with missing data: Part II,"
    Chemometrics and Intelligent Laboratory Systems 58 (2001) 29-42.

    Example
    -------
    >>> itim = PCA_IA(n_components=1, missing_values=np.nan, tol=1.0e-6,
    ... max_iters=1000)
    >>> X_filled = itim.fit_transform(X_missing)
    """

    def __init__(
        self,
        n_components=1,
        scale_x=True,
        missing_values=np.nan,
        max_iters=5000,
        tol=1.0e-6,
    ):
        """Instantiate the class."""
        self.set_params(
            **{
                "n_components": n_components,
                "scale_x": scale_x,
                "missing_values": missing_values,
                "max_iters": max_iters,
                "tol": tol,
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
            "n_components": self.n_components,
            "scale_x": self.scale_x,
            "missing_values": self.missing_values,
            "max_iters": self.max_iters,
            "tol": self.tol,
        }

    def fit(self, X, y=None):
        """
        Build PCA model to compute missing values in X.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1)
            Ignored.

        Returns
        -------
        self : PCA_IA
            Fitted model.
        """
        self.__Xtrain_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        self.n_features_in_ = self.__Xtrain_.shape[1]

        # Check number of components
        upper_bound = np.min(
            [
                self.__Xtrain_.shape[0] - 1,
                self.__Xtrain_.shape[1],
            ]
        )
        lower_bound = 1
        if self.n_components > upper_bound or self.n_components < lower_bound:
            raise Exception(
                "n_components must be in [{}, min(n_samples-1 [{}], \
n_features [{}])] = [{}, {}].".format(
                    lower_bound,
                    self.__Xtrain_.shape[0] - 1,
                    self.__Xtrain_.shape[1],
                    lower_bound,
                    upper_bound,
                )
            )

        (
            self.__scaler_,
            self.__pca_,
            _,
            _,
            _,
        ) = self._em(self.__Xtrain_, train=True)

        self.is_fitted_ = True

        return self

    def _em(self, X, train=True):
        """Expectation-maximization iteration."""
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
            dtype=np.float64,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        # Identify and record location of missing values
        indicator = MissingIndicator(
            missing_values=self.missing_values, features="all"
        )
        mask = indicator.fit_transform(X)

        # First, impute any missing to mean values
        # Note this is just based on the column values, not rows as in [1].
        si = SimpleImputer(strategy="mean", missing_values=self.missing_values)
        X_old = si.fit_transform(X)
        delta_old = X_old[mask]

        sse = 0.0
        iteration = 0
        while iteration < self.max_iters:
            # Always center before PCA
            if train:
                ss = StandardScaler(with_std=self.scale_x, with_mean=True)
                pca = PCA(n_components=self.n_components)

                # Predict and train
                X_new = ss.inverse_transform(
                    pca.inverse_transform(
                        pca.fit_transform(ss.fit_transform(X_old))
                    )
                )
            else:
                # During a transform, or test set, use previous results
                ss = self.__scaler_
                pca = self.__pca_

                # Just predict
                X_new = ss.inverse_transform(
                    pca.inverse_transform(pca.transform(ss.transform(X_old)))
                )

            # Compute change
            delta_new = X_new[mask] - X_old[mask]
            if np.sum(mask) == 0:
                # No imputation needed
                err = self.tol - 1.0e-12
            else:
                err = np.max(np.abs(delta_new - delta_old))
            sse = np.sum((X[~mask] - X_new[~mask]) ** 2)  # pp. 17 in [1]
            if err < self.tol:
                break
            else:
                X_old[mask] = X_new[mask]
                delta_old = delta_new

            iteration += 1
            if iteration == self.max_iters:
                raise Exception(
                    "Unable to converge imputation in the maximum number of iterations, {} > {}".format(
                        err, self.tol
                    )
                )

        return ss, pca, mask, X_new[mask], sse

    def fit_transform(self, X, y=None):
        """
        Compute and fill in the missing values of X.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1)
            Ignored.

        Returns
        -------
        X_filled : array_like(float, ndim=2)
            Matrix with missing data filled in.
        """
        _ = self.fit(X, y)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Fill in any missing values of X.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1)
            Ignored.

        Returns
        -------
        X_filled : array_like(float, ndim=2)
            Matrix with missing data filled in.
        """
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
            dtype=np.float64,
        )
        check_is_fitted(self, "is_fitted_")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        _, _, mask, imputed_vals, _ = self._em(X, train=False)

        X_filled = X.copy()
        X_filled[mask] = imputed_vals

        return X_filled

    def score(self, X, y=None):
        """
        Score the imputation approach.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1)
            Ignored.

        Returns
        -------
        neg_sse : scalar(float)
            Negative sum of square errors.

        Note
        ----
        This computes the sum squared error on the observed parts of the data (pp. 17
        in [1]). A value of zero implies the PCA model perfectly reconstructs the
        observations. This actually returns the NEGATIVE sum of square error (SSE) on
        the observed data.

        The negative of the SSE is returned so the maximum score is corresponds to the
        best model in cross-validation.
        """
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
            dtype=np.float64,
        )
        check_is_fitted(self, "is_fitted_")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        _, _, _, _, sse = self._em(X, train=False)

        return -sse

    def _get_tags(self):
        """For compatibility with scikit-learn >=0.21."""
        return {
            "allow_nan": True,
            "array_api_support": False,
            "binary_only": False,
            "multilabel": False,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "preserves_dtype": [np.float64],  # Only for transformers
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": [
                "check_fit2d_1sample",  # This is supposed to fail
            ],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }


class PLS_IA(TransformerMixin, BaseEstimator):
    """
    Use iterative PLS to estimate missing data values.

    Parameters
    ----------
    n_components : scalar(int), optional(default=1)
        Number of dimensions to project into. Should be in the range
        [1, num_features].

    scale_x : scalar(bool), optional(default=True)
        Whether or not to scale X columns by the standard deviation.

    missing_values : scalar(float) or NaN, optional(default=numpy.nan)
        The value in the X matrix that indicates a missing value.

    max_iters : scalar(int), optional(default=5000)
        maximum number of iterations of PLS to perform. If convergence is not
        achieved in this limit an Exception is thrown.

    tol : scalar(float), optional(default=1.0e-6)
        Maximum amount any imputed X value is allowed to change between iterations.

    Note
    ----
    If no data is missing during training, the model is still trained so it
    can handle missing data during a test phase.

    First, a simple imputation to the (column) mean is performed. Then PLS is performed
    to model the data, from which the missing values can be estimated.  These
    estimates are used to construct a new feature matrix (X) and this process
    is repeated until convergence.

    This is useful for constructing estimates of missing data in a supervised
    fashion.

    It is advisable to use cross-validation to identify the optimal number of PLS
    components to use. Importantly, you should only choose n_components so that it
    is never larger than the size of any training fold otherwise an exception will
    be thrown.

    The PLS model (loadings) found during training is fixed and used during testing
    to reconstruct test data if that is also missing.  This is necessary during
    cross-validation, for example, because the original data set may have missing data
    throughout and is subsequently split (repeatedly) so test folds will have some
    elements missing.

    Univariate imputation imputes values in the i-th feature dimension using
    only non-missing values in that dimension (i.e., the average of observed
    values in a column).

    Multivariate imputation uses the entire set of available feature dimensions.
    This PLS_IA algorithm performs multivariate imputation by constructing a PLS
    model for X iteratively until the PLS predictions for the missing data
    converge.

    As pointed out in [1], this corresponds to finding the maximum likelihood
    estimates of the PLS model parameters.

    References
    ----------
    [1] Walczak, B. and Massart, D. L., "Dealing with missing data: Part I,"
    Chemometrics and Intelligent Laboratory Systems 58 (2001) 15-27.

    [2] Walczak, B. and Massart, D. L., "Dealing with missing data: Part II,"
    Chemometrics and Intelligent Laboratory Systems 58 (2001) 29-42.

    Example
    -------
    >>> itim = PLS_IA(n_components=1, missing_values=np.nan, tol=1.0e-6,
    ... max_iters=1000)
    >>> X_filled = itim.fit_transform(X_missing, y)
    """

    def __init__(
        self,
        n_components=1,
        scale_x=True,
        missing_values=np.nan,
        max_iters=5000,
        tol=1.0e-6,
    ):
        """Instantiate the class."""
        self.set_params(
            **{
                "n_components": n_components,
                "scale_x": scale_x,
                "missing_values": missing_values,
                "max_iters": max_iters,
                "tol": tol,
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
            "n_components": self.n_components,
            "scale_x": self.scale_x,
            "missing_values": self.missing_values,
            "max_iters": self.max_iters,
            "tol": self.tol,
        }

    def _column_y(self, y):
        """Convert y to column format."""
        y = np.asarray(y)
        if y.ndim != 2:
            y = y[:, np.newaxis]

        return y

    def fit(self, X, y):
        """
        Build PLS model to compute missing values in X.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1)
            Response values. Should only have a single scalar response for each
            observation.

        Returns
        -------
        self : PLS_IA
            Fitted model.
        """
        self.__Xtrain_ = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
            dtype=np.float64,
        )
        self.n_features_in_ = self.__Xtrain_.shape[1]

        self.__ytrain_ = check_array(
            y,
            accept_sparse=False,
            force_all_finite=True,
            copy=True,
            dtype=np.float64,
            ensure_2d=False,  # Will be converted next
        )
        self.__ytrain_ = self._column_y(
            self.__ytrain_
        )  # scikit-learn expects 1D array, convert to columns
        assert self.__ytrain_.shape[1] == 1

        if self.__Xtrain_.shape[0] != self.__ytrain_.shape[0]:
            raise ValueError(
                "X ({}) and y ({}) shapes are not compatible".format(
                    self.__Xtrain_.shape, self.__ytrain_.shape
                )
            )

        # Check number of components
        upper_bound = np.min(
            [
                self.__Xtrain_.shape[0] - 1,
                self.__Xtrain_.shape[1],
            ]
        )
        lower_bound = 1
        if self.n_components > upper_bound or self.n_components < lower_bound:
            raise Exception(
                "n_components must [{}, min(n_samples-1 [{}], \
n_features [{}])] = [{}, {}].".format(
                    lower_bound,
                    self.__Xtrain_.shape[0] - 1,
                    self.__Xtrain_.shape[1],
                    lower_bound,
                    upper_bound,
                )
            )

        self.__x_scaler_, self.__y_scaler_, self.__pls_, _, _, _ = self._em(
            self.__Xtrain_, self.__ytrain_, train=True
        )

        self.is_fitted_ = True

        return self

    def _em(self, X, y=None, train=True):
        """Expectation-maximization iteration."""
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
            dtype=np.float64,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        if train:
            y = check_array(
                y,
                accept_sparse=False,
                force_all_finite=True,
                copy=True,
                dtype=np.float64,
                ensure_2d=False,  # Will be converted next
            )
            y = self._column_y(
                y
            )  # scikit-learn expects 1D array, convert to columns

        # Identify and record location of missing values
        indicator = MissingIndicator(
            missing_values=self.missing_values, features="all"
        )
        mask = indicator.fit_transform(X)

        # First, impute any missing to mean values
        # Note this is just based on the column values, not rows as in [1].
        # It may also be better to sort by y and interpolate as suggested by [1] as an initial guess.
        si = SimpleImputer(strategy="mean", missing_values=self.missing_values)
        X_old = si.fit_transform(X)
        delta_old = X_old[mask]

        sse = 0.0
        iteration = 0
        while iteration < self.max_iters:
            if train:
                # 1. Preprocess X data
                x_scaler = CorrectedScaler(
                    with_mean=True, with_std=self.scale_x
                )  # Always center and maybe scale X

                # 2. Preprocess Y data
                y_scaler = CorrectedScaler(
                    with_mean=True, with_std=False
                )  # Always center and maybe scale Y

                # 3. Fit PLS
                pls = PLSRegression(
                    n_components=self.n_components,
                    scale=self.scale_x,
                    max_iter=10000,
                )

                # 4. Fit and predict
                x_scores_, _ = pls.fit_transform(
                    x_scaler.fit_transform(X_old),
                    y_scaler.fit_transform(y),
                )
            else:
                x_scaler = self.__x_scaler_
                y_scaler = self.__y_scaler_
                pls = self.__pls_

                # Just predict
                x_scores_ = pls.transform(x_scaler.transform(X_old))

            X_new = x_scaler.inverse_transform(pls.inverse_transform(x_scores_))

            # Compute change
            delta_new = X_new[mask] - X_old[mask]
            if np.sum(mask) == 0:
                # No imputation needed
                err = self.tol - 1.0e-12
            else:
                err = np.max(np.abs(delta_new - delta_old))
            sse = np.sum((X[~mask] - X_new[~mask]) ** 2)  # pp. 17 in [1]
            if err < self.tol:
                break
            else:
                X_old[mask] = X_new[mask]
                delta_old = delta_new

            iteration += 1
            if iteration == self.max_iters:
                raise Exception(
                    "Unable to converge imputation in the maximum number of iterations, {} > {}".format(
                        err, self.tol
                    )
                )

        return x_scaler, y_scaler, pls, mask, X_new[mask], sse

    def fit_transform(self, X, y):
        """
        Compute and fill in the missing values of X.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1)
            Response values. Should only have a single scalar response for each
            observation.

        Returns
        -------
        X_filled : array_like(float, ndim=2)
            Matrix with missing data filled in.
        """
        _ = self.fit(X, y)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Fill in any missing values of X.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1)
            Ignored.

        Returns
        -------
        X_filled : array_like(float, ndim=2)
            Matrix with missing data filled in.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
            dtype=np.float64,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        _, _, _, mask, imputed_vals, _ = self._em(X, train=False)

        X_filled = X.copy()
        X_filled[mask] = imputed_vals

        return X_filled

    def score(self, X, y=None):
        """
        Score the imputation approach.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        y : array_like(float, ndim=1)
            Ignored.

        Returns
        -------
        neg_sse : scalar(float)
            Negative sum of squared errors.

        Note
        ----
        This computes the sum squared error on the observed parts of the data (pp. 17
        in [1]). A value of zero implies the PLS model perfectly reconstructs the
        observations. This actually returns the NEGATIVE sum of square error (SSE) on
        the observed data.

        The negative of the SSE is returned so the maximum score is corresponds to the
        best model in cross-validation.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
            dtype=np.float64,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )
        _, _, _, _, _, sse = self._em(X, train=False)

        return -sse

    def _get_tags(self):
        """For compatibility with scikit-learn >=0.21."""
        return {
            "allow_nan": True,
            "array_api_support": False,
            "binary_only": False,
            "multilabel": False,
            "multioutput": False,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "preserves_dtype": [np.float64],  # Only for transformers
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": [
                "check_fit2d_1sample",  # This is supposed to fail
            ],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
