"""
Fill in missing data.

author: nam
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pychemauth.preprocessing.scaling import CorrectedScaler


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

    If there is data that is truly missing, i.e., corrupted or not measured,
    and you wish to impute these values based on PCA_IA, for example, you need
    to indicate values that are "truly missing" vs. those that are missing
    because < LOD with a different indicator. This can be problematic - see
    examples/imputing_examples.ipynb for an example.

    Example
    -------
    >>> itim = LOD(lod=np.array([0.1, 0.2, 0.1]), missing_values=-1, seed=0)
    >>> X_lod = itim.fit_transform(missing_X) # Will still have NaN's for missing
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
        }

    def fit(self, X, y=None):
        """
        Compute the "missing" data.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : np.array
            Ignored

        Returns
        -------
        self
        """
        X, self.lod = check_X_y(
            X.T,
            self.lod,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        X = X.T
        self.n_features_in_ = X.shape[1]

        self.lod = self.lod.ravel()
        if len(self.lod) != X.T.shape[0]:
            raise ValueError("LOD must be specified for each column in X")

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
        y : np.array
            Ignored

        Returns
        -------
        X : matrix-like
            Matrix with missing data filled in.
        """
        _ = self.fit(X, y)

        return self.transform(X)

    def transform(self, X):
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
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        check_is_fitted(self, "is_fitted_")
        assert X.shape[1] == self.n_features_in_

        temp = 0
        mask = np.isnan(X)
        if not np.isnan(self.missing_values) and np.any(mask):
            # Still have NaN but being used for something else, so temporarily
            # modify to a unique random number. This is because MissingIndicator
            # is incompatible with np.nan if it is NOT the missing value.
            while True:
                temp = np.random.random()
                if not np.any(X == temp) and temp != self.missing_values:
                    break
            X[mask] = temp

        # Find any missing values and take as below LOD
        indicator = MissingIndicator(
            missing_values=self.missing_values, features="all"
        )
        explicit_mask = indicator.fit_transform(X)

        if temp > 0:
            # Change back to nan if it was modified earlier
            X[mask] = np.nan

        # We are going to impute anything explicitly missing and anything below LOD
        below_mask = (X / self.lod) < 1.0
        mask = (explicit_mask) | (below_mask)

        # Replace missing values with the imputed ones
        X_scaled = X / self.lod
        X_scaled[mask] = self.__rng_.random(np.sum(mask))

        return X_scaled * self.lod


class PCA_IA:
    """
    Use iterative PCA to estimate any missing data values.

    Notes
    -----
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

    The PCA model (loadings) found during training are fixed and used during testing
    to reconstruct test data if that is also missing.  This is necessary during
    cross-validation, for example, because the original data set may have missing data
    throughout and is subsequently split (repeatedly) so test folds will have some
    elements missing.

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

    [1] Walczak, B. and Massart, D. L., "Dealing with missing data: Part I," 
    Chemometrics and Intelligent Laboratory Systems 58 (2001) 15-27.
    [2] Walczak, B. and Massart, D. L., "Dealing with missing data: Part II," 
    Chemometrics and Intelligent Laboratory Systems 58 (2001) 29-42.
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
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : np.array
            Ignored

        Returns
        -------
        self
        """
        self.__Xtrain_ = check_array(
            X,
            accept_sparse=False,
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
                "n_components must [{}, min(n_samples-1 [{}], \
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
        ) = self.em_(self.__Xtrain_, train=True)

        self.is_fitted_ = True

        return self

    def em_(self, X, train=True):
        """Expectation-maximization iteration."""
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        assert X.shape[1] == self.n_features_in_

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
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : np.array
            Ignored

        Returns
        -------
        X_filled : matrix-like
            Matrix with missing data filled in.
        """
        _ = self.fit(X, y)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Fill in any missing values of X.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : np.array
            Ignored

        Returns
        -------
        X_filled : matrix-like
            Matrix with missing data filled in.
        """
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        check_is_fitted(self, "is_fitted_")
        assert X.shape[1] == self.n_features_in_
        _, _, mask, imputed_vals, _ = self.em_(X, train=False)

        X_filled = X.copy()
        X_filled[mask] = imputed_vals

        return X_filled

    def score(self, X, y=None):
        """
        Score the imputation approach.

        This computes the sum squared error on the observed parts of the data (pp. 17
        in [1]). A value of zero implies the PCA model perfectly reconstructs the
        observations. This actually returns the NEGATIVE sum of square error (SSE) on
        the observed data.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : np.array
            Ignored

        Notes
        -----
        The negative of the SSE is returned so the maximum score is corresponds to the
        best model in cross-validation.

        Returns
        -------
        sse : np.float
        """
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        check_is_fitted(self, "is_fitted_")
        _, _, _, _, sse = self.em_(X, train=False)

        return -sse


class PLS_IA:
    """
    Use iterative PLS to estimate missing data values.

    Notes
    -----
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

    The PLS model (loadings) found during training are fixed and used during testing
    to reconstruct test data if that is also missing.  This is necessary during
    cross-validation, for example, because the original data set may have missing data
    throughout and is subsequently split (repeatedly) so test folds will have some
    elements missing.

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

    [1] Walczak, B. and Massart, D. L., "Dealing with missing data: Part I," 
    Chemometrics and Intelligent Laboratory Systems 58 (2001) 15-27.
    [2] Walczak, B. and Massart, D. L., "Dealing with missing data: Part II," 
    Chemometrics and Intelligent Laboratory Systems 58 (2001) 29-42.
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

    def column_y_(self, y):
        """Convert y to column format."""
        y = np.array(y)
        if y.ndim != 2:
            y = y[:, np.newaxis]

        return y

    def fit(self, X, y):
        """
        Build PLS model to compute missing values in X.

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
        self.__Xtrain_ = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        self.n_features_in_ = self.__Xtrain_.shape[1]

        self.__ytrain_ = check_array(
            y, accept_sparse=False, force_all_finite=True, copy=True
        )
        self.__ytrain_ = self.column_y_(
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

        self.__x_scaler_, self.__y_scaler_, self.__pls_, _, _, _ = self.em_(
            self.__Xtrain_, self.__ytrain_, train=True
        )

        self.is_fitted_ = True

        return self

    def em_(self, X, y=None, train=True):
        """Expectation-maximization iteration."""
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        assert X.shape[1] == self.n_features_in_

        if train:
            y = check_array(
                y, accept_sparse=False, force_all_finite=True, copy=True
            )
            y = self.column_y_(
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
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically. [n_samples, n_features]
        y : array-like
            Response values. Should only have a single scalar response for each
            observation. [n_samples, 1]

        Returns
        -------
        X_filled : matrix-like
            Matrix with missing data filled in.
        """
        _ = self.fit(X, y)

        return self.transform(X)

    def transform(self, X, y=None):
        """
        Fill in any missing values of X.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : array-like
            Ignored.

        Returns
        -------
        X_filled : matrix-like
            Matrix with missing data filled in.
        """
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        check_is_fitted(self, "is_fitted_")
        assert X.shape[1] == self.n_features_in_
        _, _, _, mask, imputed_vals, _ = self.em_(X, train=False)

        X_filled = X.copy()
        X_filled[mask] = imputed_vals

        return X_filled

    def score(self, X, y=None):
        """
        Score the imputation approach.

        This computes the sum squared error on the observed parts of the data (pp. 17
        in [1]). A value of zero implies the PLS model perfectly reconstructs the
        observations. This actually returns the NEGATIVE sum of square error (SSE) on
        the observed data.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : np.array
            Ignored.

        Notes
        -----
        The negative of the SSE is returned so the maximum score is corresponds to the
        best model in cross-validation.

        Returns
        -------
        sse : np.float
        """
        X = check_array(
            X,
            accept_sparse=False,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=True,
        )
        check_is_fitted(self, "is_fitted_")
        _, _, _, _, _, sse = self.em_(X, train=False)

        return -sse
