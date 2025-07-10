"""
Filter data.

author: nam
"""
import copy
import numpy as np
import scipy.signal
from scipy.stats import iqr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from typing import Any, Union, Sequence, ClassVar
from numpy.typing import NDArray


class MSC(TransformerMixin, BaseEstimator):
    """
    Perform multiplicative scatter correction.

    Parameters
    ----------
    Xref : array-like(float, ndim=2), optional(default=None)
        Optional reference to use as background instead of `X` when fitting.

    Note
    ----
    From [1]: Multiplicative scatter correction (MSC) focuses on light scattering and particle size issues. It corrects for both multiplicative and additive effects, and is based on two assumptions:

    * a sample is considered as an addition of two components, a signal term and a noise term (scatter); the latter is to be corrected for.

    * the scatter term is estimated with linear regression and assumes the coefficients will be the same for all samples across all wavelengths (axis=1 for `X`).

    Essentially, a linear regression is performed between the reference and the sample, and the sample is corrected by subtracting the intercept and dividing by the slope. See [1] pp 167.

    Generally the mean of `X` can be taken as the background reference, however, this is not always ideal.

    It is reported that MSC preprocessing tends to yield similar performances as SNV, but MSC requires a reference while SNV does not. [2] Therefore, SNV is generally preferable.

    References
     ----------
    [1] Brown, Steven D., Romà Tauler, and Beata Walczak, eds. Comprehensive Chemometrics: Chemical and Biochemical Data Analysis. Elsevier, 2020.

    [2] https://guifh.github.io/RNIR/MSC.html
    """

    Xref: ClassVar[Union[NDArray[np.floating], None]]

    def __init__(self, Xref: Union[NDArray[np.floating], None] = None) -> None:
        """Instantiate the class."""
        self.set_params(
            **{
                "Xref": Xref,
            }
        )

    def set_params(self, **parameters: Any) -> "MSC":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "Xref": copy.copy(self.Xref),
        }

    def fit(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]], y=None
    ) -> "MSC":
        """
        Compute the reference background, if necessary.

        Parameter
        ---------
        X : array-like(float, ndim=2)
            Feature matrix to use as background. If `Xref` is provided this is ignored.

        y : array-like(float, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        self : MSC
            Fitted model.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
        )
        if y is not None:  # Just so this passes sklearn api checks
            X_, y = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_2d=True,
                force_all_finite=True,
                y_numeric=True,
            )

        if self.Xref is None:  # Use the mean from the training set
            self.Xref_ = np.mean(X_, axis=0, dtype=np.float64)
        else:
            self.Xref_ = np.asarray(self.Xref, dtype=np.float64)

        self.n_features_in_ = len(self.Xref_)

        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        self.is_fitted_ = True

        return self

    def transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """
        Transform the data.

        Parameter
        ---------
        X : array-like(float, ndim=2)
            Feature matrix.

        Returns
        -------
        X_corrected : ndarray(float, ndim=2)
            Corrected feature matrix.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )  # Force a copy
        check_is_fitted(self, "is_fitted_")
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        try:
            for row in range(X_.shape[0]):
                coef = np.polynomial.polynomial.polyfit(
                    self.Xref_, X_[row, :], deg=1
                )
                X_[row, :] = (X_[row, :] - coef[0]) / coef[1]
        except Exception as e:
            raise ValueError("Cannot perform MSC transformation : {}".format(e))

        return X_

    def fit_transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]], y=None
    ) -> NDArray[np.floating]:
        """
        Fit and then transform some data.

        Parameter
        ---------
        X : array-like(float, ndim=2)
            Feature matrix.

        y : array-like(float, ndim=1), optional(default=None)
            Ignored.

        Returns
        -------
        X_corrected : ndarray(float, ndim=2)
            Corrected feature matrix.
        """
        self.fit(X)

        return self.transform(X)

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
            "preserves_dtype": [np.float64],  # Only for transformers
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": [],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }


class SNV(TransformerMixin, BaseEstimator):
    """
    Perform a Standard Normal Variates transformation.

    Parameters
    ----------
    robust : bool, optional(default=False)
        Whether or not to use robust statistics (percentile, IQR) instead of standard descriptive statistics (mean, standard deviation).

    detrend : bool, optional(default=False)
        Whether or not to apply a detrend transformation after the SNV transformation.

    q : scalar(float), optional(default=50)
            Percentile to use for RNV, if RNV is used; for SNV this is ignored.

    Note
    ---
    The so-called Standard Normal Variate (SNV) method performs a normalization of the data by subtracting each row by its own mean then dividing by its own standard deviation. After SNV, each row will have a mean of 0 and a standard deviation of 1. This transformation is equivalent to autoscaling by rows instead of by columns.

    SNV attempts to make rows comparable in terms of magnitude (e.g., intensities for spectra or absorbance level). It can be useful to correct spectra for calibration.

    From [1]:

    * SNV corrects for both baseline shift and global intensity variations.

    * SNV improves the PLS model prediction, especially when used on NIR data with scattering effects.

    * When using SNV, the spectra always have positive and negative values centered on 0, which may make interpretation a little more difficult.

    * SNV assumes that multiplicative effects are uniform over the whole spectral range, which is not always the case, so artifacts could be introduced by this transformation.

    "Guo et al. [4] introduced the robust normal variate (RNV) transformation to tackle certain artifacts created when using SNV transformation by solving the 'closure' problem. Closure is the statistical term indicating that the sum of the data is necessarily equal to a certain amount, so that if one of the variables changes in one direction, the other variables must change in the opposite direction to ensure the constancy of the sum. Guo et al. [4] solved this closure problem by introducing the RNV transformation that modifies SNV by using the percentile instead of the mean."

    Here we implement the version of RNV discussed in [1] which uses the percentile to center the data, and the IQR to scale it (cf. pp 165).

    From [1]: "Absorbance spectra in NIR increase linearly with the wavelength for transparent samples, whereas it increases curvilinearly for spectra of densely packed samples. Therefore, another approach to correct for the baseline shift is the Detrend method, which was introduced along with the standard normal variates transform (SNV) by Barnes et al. [3] Detrending accounts for the variation in baseline shift and curvilinearity, generally found in the reflectance spectra of powdered or densely packed samples. Using a second-degree polynomial regression, the detrend method removes the baseline curvature from each individual spectrum by expressing it as a quadratic function of the wavelengths."

    Detrending is generally useful to help remove any baseline which varies as a function of time, etc. (e.g., axis=1 of X) - cf. [1] pp 147. Using SNV followed by detrending allows you to correct for both shift and drift in your baseline. [1,3]

    References
    ----------
    [1] Brown, Steven D., Romà Tauler, and Beata Walczak, eds. Comprehensive Chemometrics: Chemical and Biochemical Data Analysis. Elsevier, 2020.

    [2] https://guifh.github.io/RNIR/SNV.html

    [3] Barnes, R. J.; Dhanoa, M. S.; Lister, S. J. Correction of the Description of Standard Normal Variate (SNV) and De-Trend Transformations in Practical Spectroscopy with Applications in Food and Beverage Analysis, 2nd ed. J. Near Infrared Spectrosc. 1993, 1, 185–186.

    [4] Guo, Q.; Wu, W.; Massart, D. L. The Robust Normal Variate Transform for Pattern Recognition with Near-Infrared Data. Anal. Chim. Acta 1999, 382 (1), 87–103.
    """

    robust: ClassVar[bool]
    detrend: ClassVar[bool]
    q: ClassVar[np.floating]

    def __init__(
        self, robust: bool = False, detrend: bool = False, q: float = 50
    ) -> None:
        """Instantiate the class."""
        self.set_params(**{"robust": robust, "detrend": detrend, "q": q})

    def set_params(self, **parameters: Any) -> "SNV":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "robust": self.robust,
            "detrend": self.detrend,
            "q": self.q,
        }

    def fit(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]], y=None
    ) -> "SNV":
        """
        Store the size of the data (number of columns) for consistency later.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.  Not stored, but is used to determined the expected size of future matrices.

        y : array-like(float, ndimd=1)
            Ignored.

        Returns
        -------
        self : SNV
            Fitted model.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
        )
        if y is not None:  # Just so this passes sklearn api checks
            X_, y = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_2d=True,
                force_all_finite=True,
                y_numeric=True,
            )

        assert (0.0 < self.q) and (100.0 > self.q)

        self.n_features_in_ = X_.shape[1]
        self.is_fitted_ = True

        return self

    def transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """
        Transform the data.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        Returns
        -------
        X_corrected : array-like(float, ndim=2)
            Corrected feature matrix.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            copy=True,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
        )  # Force a copy
        check_is_fitted(self, "is_fitted_")
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        try:
            if not self.robust:
                scale_ = np.std(X_, axis=1, ddof=1)
                ave_ = np.mean(X_, axis=1)
                X_ = ((X_.T - ave_) / scale_).T
            else:  # Uses algorithm from [1] pp 165.
                for row, per in enumerate(
                    np.percentile(X, q=self.q, axis=1, method="linear")
                ):
                    scale = iqr(
                        X_[row, :],
                        rng=(25, 75),
                        nan_policy="raise",
                        interpolation="linear",
                    )  # Scale is independent of percentile
                    X_[row, :] -= per
                    X_[row, :] /= scale
        except Exception as e:
            raise ValueError("Cannot perform SNV transformation : {}".format(e))

        if self.detrend:  # Fit to second order polynomial and substract
            x_ = np.arange(X_.shape[1])
            for row in range(X_.shape[0]):
                coef = np.polynomial.polynomial.polyfit(x_, X_[row, :], deg=2)
                pred = coef[0] + coef[1] * x_ + coef[2] * x_**2
                X_[row, :] -= pred

        return X_

    def fit_transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]], y=None
    ) -> NDArray[np.floating]:
        """
        Fit and then transform some data.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        y : array-like(float, ndim=1)
            Ignored.

        Returns
        -------
        X_corrected : ndarray(float, ndim=2)
            Corrected feature matrix.
        """
        self.fit(X)

        return self.transform(X)

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
            "preserves_dtype": [np.float64],  # Only for transformers
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": [],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }


class SavGol(TransformerMixin, BaseEstimator):
    """
    Perform a Savitzky-Golay filtering.

    Parameters
    ----------
    window_length : scalar(int)
        The length of the filter window (i.e., the number of coefficients). If mode is "interp", `window_length` must be less than or equal to the size of x.

    polyorder : scalar(int)
        The order of the polynomial used to fit the samples. `polyorder` must be less than `window_length`.

    deriv : scalar(int), optional(default=0)
        The order of the derivative to compute. This must be a nonnegative integer. The default is 0, which means to filter the data without differentiating.

    delta : scalar(float), optional(default=1.0)
        The spacing of the samples to which the filter will be applied. This is only used if `deriv` > 0. Default is 1.0.

    axis : scalar(int), optional(default=-1)
        The axis of the array x along which the filter is to be applied.

    mode : str, optional(default="interp")
        Must be "mirror", "constant", "nearest", "wrap" or "interp". This determines the type of extension to use for the padded signal to which the filter is applied. When mode is "constant", the padding value is given by `cval`. See the Notes for more details on "mirror", "constant", "wrap", and "nearest". When the "interp" mode is selected (the default), no extension is used. Instead, a degree `polyorder` polynomial is fit to the last `window_length` values of the edges, and this polynomial is used to evaluate the last `window_length` // 2 output values.

    cval : scalar(float), optional(default=0.0)
        Value to fill past the edges of the input if mode is "constant". Default is 0.0.

    References
    ----------
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html for description of parameters.
    """

    window_length: ClassVar[int]
    polyorder: ClassVar[int]
    deriv: ClassVar[int]
    delta: ClassVar[float]
    axis: ClassVar[int]
    mode: ClassVar[str]
    cval: ClassVar[float]

    def __init__(
        self,
        window_length: int,
        polyorder: int,
        deriv: int = 0,
        delta: float = 1.0,
        axis: int = -1,
        mode: str = "interp",
        cval: float = 0.0,
    ) -> None:
        """Instantiate the class."""
        self.set_params(
            **{
                "window_length": window_length,
                "polyorder": polyorder,
                "deriv": deriv,
                "delta": delta,
                "axis": axis,
                "mode": mode,
                "cval": cval,
            }
        )

    def set_params(self, **parameters: Any) -> "SavGol":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "window_length": self.window_length,
            "polyorder": self.polyorder,
            "deriv": self.deriv,
            "delta": self.delta,
            "axis": self.axis,
            "mode": self.mode,
            "cval": self.cval,
        }

    def fit(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]], y=None
    ) -> "SavGol":
        """
        Fit the filter using some training data.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        y : array-like(float, ndim=1)
            Ignored.

        Returns
        -------
        self : SavGol
            Fitted model.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
        )

        try:
            _ = scipy.signal.savgol_filter(
                X_,
                window_length=self.window_length,
                polyorder=self.polyorder,
                deriv=self.deriv,
                delta=self.delta,
                axis=self.axis,
                mode=self.mode,
                cval=self.cval,
            )
        except Exception as e:
            raise ValueError(
                "Cannot perform Savitzky-Golay filtering with these parameters : {}".format(
                    e
                )
            )

        self.n_features_in_ = X_.shape[1]
        self.is_fitted_ = True

        return self

    def transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """
        Transform (center and possibly scale) the data after fitting.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        Returns
        -------
        X_filtered : ndarray(float, ndim=2)
            Filtered feature matrix.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
        )
        check_is_fitted(self, "is_fitted_")
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        X_filtered = scipy.signal.savgol_filter(
            X_,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            axis=self.axis,
            mode=self.mode,
            cval=self.cval,
        )

        return X_filtered

    def fit_transform(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]], y=None
    ) -> NDArray[np.floating]:
        """
        Fit and then transform some data.

        Parameters
        ----------
        X : array-like(float, ndim=2)
            Feature matrix.

        y : array-like(float, ndim=1)
            Ignored.

        Returns
        -------
        X_filtered : ndarray(float, ndim=2)
            Filtered feature matrix.
        """
        self.fit(X)

        return self.transform(X)

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
            "preserves_dtype": [np.float64],  # Only for transformers
            "poor_score": True,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": [
                "check_fit2d_1feature"  # Needs to have multiple features to fit to
            ],
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
