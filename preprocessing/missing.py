"""
Fill in missing data.

author: nam
"""
import sys

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

sys.path.append("../")
from chemometrics.preprocessing.scaling import CorrectedScaler


class LOD:
    """
    Fill in "missing" measurement values that are below LOD.

    Notes
    -----
    Data in the feature matrix (X) is "missing" but is really because it is
    below the limit of detection (LOD).  In this case, random values are
    created that chosen between 0 (some baseline) and the LOD (which must
    be provided by the user).

    Values are divided by LOD's and if less than 1, or are explicitly noted
    as a missing value, are imputed to a random number [0, 1) and multiplied
    by the LOD for that column.  Note that this is done to preserve the sign
    of the measurement.

    This ONLY executes during the training stage of a pipeline; X data during
    the testing phase will be unaffected.

    Example
    -------
    >>> itim = LOD(lod=np.array([0.1, 0.2, 0.1]), missing_values=np.nan, seed=0)
    >>> X_filled = itim.fit_transform(X_missing)
    """

    def __init__(self, lod, missing_values=np.nan, seed=0):
        """
        Instantiate the class.

        Parameters
        ----------
        lod : ndarray
            Numerical limit of detection for each feature.
        missing_values : object
            The value in the X matrix that indicates a missing value.
        seed : int
            Random number generator seed.
        """
        self.set_params(
            **{"lod": lod, "missing_values": missing_values, "seed": seed}
        )
        self.is_fitted_ = False

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {
            "lod": self.lod,
            "missing_values": self.missing_values,
            "seed": self.seed,
        }

    def fit(self, X, y=None):
        """
        Compute the "missing" data.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        self
        """
        self.__X_ = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        self.__indicator_ = MissingIndicator(
            missing_values=self.missing_values, features="all"
        )
        explicit_mask = self.__indicator_.fit_transform(self.__X_)
        self.lod = check_array(
            self.lod, accept_sparse=False, force_all_finite=True, copy=True
        )
        self.lod = self.lod.ravel()
        if len(self.lod) != self.__X_.shape[0]:
            raise ValueError("LOD must be specified for each column in X")

        # We are going to impute anything explicitly missing and anything below LOD
        below_mask = (self.__X_ / self.lod) < 1.0
        self.__mask_ = (explicit_mask) | (below_mask)

        self.__rng_ = np.random.default_rng(self.seed)

        self.is_fitted_ = True
        return self

    def fit_transform(self, X, y=None):
        """
        Fill in values of X below LOD with a random value (between 0 and LOD).

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        X : matrix-like
            Matrix with missing data filled in.
        """
        self.fit(X, y)

        # Replace missing values with the imputed ones
        X_scaled = self.__X_ / self.lod
        X_scaled[self.__mask_] = self.__rng_.random(np.sum(self.__mask_))

        return X_scaled * self.lod

    def transform(self, X):
        """
        Do nothing so that this object can be used in pipelines.

        Notes
        -----
        * Use fit_transform() to fit and transform training data. This should not
        execute on test data later on, so transform() should do nothing.
        * See https://stackoverflow.com/questions/49770851/customized-transformermixin-with-data-labels-in-sklearn/49771602#49771602

        Returns
        -------
        X : matrix-like
            Original input is return unaffected.
        """
        return X


class PCA_IA:
    """
    Use iterative PCA to estimate missing data values.

    Notes
    -----
    First, a simple imputation to the (column) mean is performed. Then PCA is performed
    to model the data, from which the missing values can be estimated.  These
    estimates are used to construct a new feature matrix (X) and this process
    is repeated until convergence.

    This is useful for constructing estimates of missing data in an unsupervised
    fashion. If there is missing data in the response (y), one approach is to
    simply append (numpy.hstack(X, y)) the y array/matrix to X and perform this.
    However, other approaches may be more advisable.

    It is advisable to use cross-validation to identify the optimal number of PCA
    components to use.

    This ONLY executes during the training stage of a pipeline; X data during
    the testing phase will be unaffected.

    Example
    -------
    >>> itim = PCA_IA(n_components=1, missing_values=np.nan, tol=1.0e-6,
    ... max_iters=1000)
    >>> X_filled = itim.fit_transform(X_missing)

    Notes
    -----
    Univariate imputation imputes values in the i-th feature dimension using
    only non-missing values in that dimension (i.e., the average of observed
    values in a column).

    Multivariate imputation uses the entire set of available feature dimensions.
    This PCA_IA algorithm performs multivariate imputation by constructing a PCA
    model for X iteratively until the PCA predictions for the missing data
    converge.

    As pointed out in [1], this corresponds to finding the maximum likelihood
    estimates of the PCA model parameters.

    This implementation follows:

    [1] Walczak, B. and Massart, D. L., "Dealing with missing data: Part I," Chemometrics and Intelligent Laboratory Systems 58 (2001) 15-27.
    [2] Walczak, B. and Massart, D. L., "Dealing with missing data: Part II," Chemometrics and Intelligent Laboratory Systems 58 (2001) 29-42.
    """

    def __init__(
        self,
        n_components=1,
        scale_x=True,
        missing_values=np.nan,
        max_iters=5000,
        tol=1.0e-6,
    ):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of dimensions to project into. Should be in the range
            [1, num_features].
        scale_x : bool
            Whether or not to scale X columns by the standard deviation.
        missing_values : object
            The value in the X matrix that indicates a missing value.
        max_iters : int
            Maximum number of iterations of PCA to perform. If convergence is not
            achieved in this limit an Exception is thrown.
        tol : float
            Maximum amount any imputed X value is allowed to change between iterations.
        """
        self.set_params(
            **{
                "n_components": n_components,
                "scale_x": scale_x,
                "missing_values": missing_values,
                "max_iters": max_iters,
                "tol": tol,
            }
        )
        self.is_fitted_ = False

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {
            "n_components": self.n_components,
            "scale_x": self.scale_x,
            "missing_values": self.missing_values,
            "max_iters": self.max_iters,
            "tol": self.tol,
        }

    def fit(self, X, y=None):
        """
        Compute the missing data.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        self
        """
        self.__X_ = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )

        # Check number of components
        upper_bound = np.min(
            [
                self.__X_.shape[0] - 1,
                self.__X_.shape[1],
            ]
        )
        lower_bound = 1
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

        # Identify and record location of missing values
        self.__indicator_ = MissingIndicator(
            missing_values=self.missing_values, features="all"
        )
        self.__mask_ = self.__indicator_.fit_transform(self.__X_)

        # First, impute any missing to mean values
        # Note this is just based on the column values, not rows as in [1].
        si = SimpleImputer(strategy="mean", missing_values=self.missing_values)
        X = si.fit_transform(self.__X_)
        delta = X[self.__mask_]

        self.__sse_ = 0.0
        iteration = 0
        while iteration < self.max_iters:
            # Always center before PCA
            ss = StandardScaler(with_std=self.scale_x, with_mean=True)
            pca = PCA(n_components=self.n_components)

            # Predict
            X_new = ss.inverse_transform(
                pca.inverse_transform(pca.fit_transform(ss.fit_transform(X)))
            )
            imputed_vals = X_new[self.__mask_]

            # Compute change
            delta_new = imputed_vals - X[self.__mask_]
            err = np.max(np.abs(delta_new - delta))
            self.__sse_ = np.sum(
                (self.__X_[~self.__mask_] - X_new[~self.__mask_]) ** 2
            )  # pp 17 in [1]
            if err < self.tol:
                self.__imputed_vals_ = imputed_vals
                break
            else:
                X[self.__mask_] = imputed_vals
                delta = delta_new

            iteration += 1
            if iteration == self.max_iters:
                raise Exception(
                    "Unable to converge imputation in the maximum number of iterations, {} > {}".format(
                        err, self.tol
                    )
                )

        self.is_fitted_ = True
        return self

    def fit_transform(self, X, y=None):
        """
        Fill in the missing values of X originally fit.

        Parameters
        ----------
        X : matrix-like
            Ignored. The original X matrix is stored internally and used instead.

        Returns
        -------
        X : matrix-like
            Matrix with missing data filled in.
        """
        self.fit(X, y)

        # Replace missing values with the imputed ones
        X_filled = self.__X_.copy()
        X_filled[self.__mask_] = self.__imputed_vals_

        return X_filled

    def transform(self, X):
        """
        Do nothing so that this object can be used in pipelines.

        Notes
        -----
        * Use fit_transform() to fit and transform training data. This should not
        execute on test data later on, so transform() should do nothing.
        * See https://stackoverflow.com/questions/49770851/customized-transformermixin-with-data-labels-in-sklearn/49771602#49771602

        Returns
        -------
        X : matrix-like
            Original input is return unaffected.
        """
        return X

    def score(self, X=None, y=None):
        """
        Score the imputation approach based on fitted data.

        Parameters
        ----------
        X : np.array
            Ignored
        y : np.array
            Ignored

        Notes
        -----
        The negative of the SSE is returned so the maximum score is corresponds to the
        best model in cross-validation.

        This returns the negative sum of square error (SSE) on the observed data.
        A value of zero implies the PCA model perfectly reconstructs the observations.

        Returns
        -------
        sse : np.float
        """
        check_is_fitted(self, "is_fitted_")
        return -self.__sse_


class PLS_IA:
    """
    Use iterative PLS to estimate missing data values.

    Notes
    -----
    First, a simple imputation to the (column) mean is performed. Then PLS is performed
    to model the data, from which the missing values can be estimated.  These
    estimates are used to construct a new feature matrix (X) and this process
    is repeated until convergence.

    This is useful for constructing estimates of missing data in a supervised
    fashion.

    It is advisable to use cross-validation to identify the optimal number of PLS
    components to use.

    This ONLY executes during the training stage of a pipeline; X data during
    the testing phase will be unaffected.

    Example
    -------
    >>> itim = PLS_IA(n_components=1, missing_values=np.nan, tol=1.0e-6,
    ... max_iters=1000)
    >>> X_filled = itim.fit_transform(X_missing, y)

    Notes
    -----
    Univariate imputation imputes values in the i-th feature dimension using
    only non-missing values in that dimension (i.e., the average of observed
    values in a column).

    Multivariate imputation uses the entire set of available feature dimensions.
    This PLS_IA algorithm performs multivariate imputation by constructing a PLS
    model for X iteratively until the PLS predictions for the missing data
    converge.

    As pointed out in [1], this corresponds to finding the maximum likelihood
    estimates of the PLS model parameters.

    This implementation follows:

    [1] Walczak, B. and Massart, D. L., "Dealing with missing data: Part I," Chemometrics and Intelligent Laboratory Systems 58 (2001) 15-27.
    [2] Walczak, B. and Massart, D. L., "Dealing with missing data: Part II," Chemometrics and Intelligent Laboratory Systems 58 (2001) 29-42.
    """

    def __init__(
        self,
        n_components=1,
        scale_x=True,
        missing_values=np.nan,
        max_iters=5000,
        tol=1.0e-6,
    ):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of dimensions to project into. Should be in the range
            [1, num_features].
        scale_x : bool
            Whether or not to scale X columns by the standard deviation.
        missing_values : object
            The value in the X matrix that indicates a missing value.
        max_iters : int
            Maximum number of iterations of PLS to perform. If convergence is not
            achieved in this limit an Exception is thrown.
        tol : float
            Maximum amount any imputed X value is allowed to change between iterations.
        """
        self.set_params(
            **{
                "n_components": n_components,
                "scale_x": scale_x,
                "missing_values": missing_values,
                "max_iters": max_iters,
                "tol": tol,
            }
        )
        self.is_fitted_ = False

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {
            "n_components": self.n_components,
            "scale_x": self.scale_x,
            "missing_values": self.missing_values,
            "max_iters": self.max_iters,
            "tol": self.tol,
        }

    def column_y_(self, y):
        """Convert y to column format."""
        y = np.array(y)
        if y.ndim != 2:
            y = y[:, np.newaxis]

        return y

    def fit(self, X, y):
        """
        Estimate the missing data.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically. [n_samples, n_features]
        y : array-like
            Response values. Should only have a single scalar response for each
            observation. [n_samples, 1]

        Returns
        -------
        self
        """
        self.__X_ = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        self.__y_ = check_array(
            y, accept_sparse=False, force_all_finite=True, copy=True
        )
        self.__y_ = self.column_y_(
            y
        )  # sklearn expects 1D array, convert to columns
        assert self.__y_.shape[1] == 1

        if self.__X_.shape[0] != self.__y_.shape[0]:
            raise ValueError(
                "X ({}) and y ({}) shapes are not compatible".format(
                    self.__X_.shape, self.__y_.shape
                )
            )

        # Check number of components
        upper_bound = np.min(
            [
                self.__X_.shape[0] - 1,
                self.__X_.shape[1],
            ]
        )
        lower_bound = 1
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

        # Identify and record location of missing values
        self.__indicator_ = MissingIndicator(
            missing_values=self.missing_values, features="all"
        )
        self.__mask_ = self.__indicator_.fit_transform(self.__X_)

        # First, impute any missing to mean values
        # Note this is just based on the column values, not rows as in [1].
        # It may also be better to sort by y and interpolate as suggested by [1] as an initial guess.
        si = SimpleImputer(strategy="mean", missing_values=self.missing_values)
        X = si.fit_transform(self.__X_)
        delta = X[self.__mask_]

        self.__sse_ = 0.0
        iteration = 0
        while iteration < self.max_iters:
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

            # 4. Predict
            x_scores_, y_scores_ = pls.fit_transform(
                x_scaler.fit_transform(X),
                y_scaler.fit_transform(self.__y_),
            )
            X_new = x_scaler.inverse_transform(pls.inverse_transform(x_scores_))
            imputed_vals = X_new[self.__mask_]

            # Compute change
            delta_new = imputed_vals - X[self.__mask_]
            err = np.max(np.abs(delta_new - delta))
            self.__sse_ = np.sum(
                (self.__X_[~self.__mask_] - X_new[~self.__mask_]) ** 2
            )  # pp 17 in [1]
            if err < self.tol:
                self.__imputed_vals_ = imputed_vals
                break
            else:
                X[self.__mask_] = imputed_vals
                delta = delta_new

            iteration += 1
            if iteration == self.max_iters:
                raise Exception(
                    "Unable to converge imputation in the maximum number of iterations, {} > {}".format(
                        err, self.tol
                    )
                )

        self.is_fitted_ = True
        return self

    def fit_transform(self, X, y):
        """
        Fill in the missing values of X originally fit.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically. [n_samples, n_features]
        y : array-like
            Response values. Should only have a single scalar response for each
            observation. [n_samples, 1]

        Returns
        -------
        X : matrix-like
            Matrix with missing data filled in.
        """
        _ = self.fit(X, y)

        # Replace missing values with the imputed ones
        X_filled = self.__X_.copy()
        X_filled[self.__mask_] = self.__imputed_vals_

        return X_filled

    def transform(self, X):
        """
        Do nothing so that this object can be used in pipelines.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically. [n_samples, n_features]

        Notes
        -----
        * Use fit_transform() to fit and transform training data. This should not
        execute on test data later on, so transform() should do nothing.
        * See https://stackoverflow.com/questions/49770851/customized-transformermixin-with-data-labels-in-sklearn/49771602#49771602

        Returns
        -------
        X : matrix-like
            Original input is return unaffected.
        """
        return X

    def score(self, X=None, y=None):
        """
        Score the imputation approach based on fitted data.

        Parameters
        ----------
        X : np.array
            Ignored
        y : np.array
            Ignored

        Notes
        -----
        The negative of the SSE is returned so the maximum score is corresponds to the
        best model in cross-validation.

        This returns the negative sum of square error (SSE) on the observed data.
        A value of zero implies the PCA model perfectly reconstructs the observations.

        Returns
        -------
        sse : np.float
        """
        check_is_fitted(self, "is_fitted_")
        return -self.__sse_
