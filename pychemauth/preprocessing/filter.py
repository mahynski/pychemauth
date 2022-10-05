"""
Filter data.

@author: nam
"""
import numpy as np
import scipy.signal
from scipy.stats import iqr
from sklearn.utils.validation import check_array, check_is_fitted


class MSC:
    """Perform multiplicative scatter correction."""

    def __init__(self, Xref=None):
        """
        Instantiate the class.

        From [1]: Multiplicative scatter correction (MSC)25,26,
        focuses on light scattering and particle size issues. It corrects for
        both multiplicative and additive effects, and is based on two assumptions:
        * a sample is considered as an addition of two components, a signal
        term and a noise term (scatter); the latter is to be corrected for.
        * the scatter term is estimated with linear regression and assumes
        the coefficients will be the same for all samples across all
        wavelengths (axis=1 for X).

        Essentially, a linear regression is performed between the reference
        and the sample, and the sample is corrected by subtracting the
        intercept and dividing by the slope. See [1] pp 167.

        Generally the mean of X can be taken as the background reference,
        however, this is not always ideal.

        It is reported that MSC preprocessing tends to yield similar
        performances as SNV, but MSC requires a reference while SNV does not. [2]
        Therefore, SNV is generally preferable.

        References
        ----------
        [1] Brown, Steven D., Romà Tauler, and Beata Walczak, eds. Comprehensive
        chemometrics: chemical and biochemical data analysis. Elsevier, 2020.
        [2] https://guifh.github.io/RNIR/MSC.html

        Parameters
        ----------
        start : bool
            Column of X to begin the regression over.
        end : bool
            Column of X to end the regression over.
        """
        self.set_params(
            **{
                "Xref": Xref,
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
            "Xref": self.Xref,
        }

    def fit(self, X, y=None):
        """Compute the reference background, if necessary."""
        X = check_array(X, accept_sparse=False)

        if self.Xref is None:  # Use the mean from the training set
            self.Xref = np.mean(X, axis=0)
        self.n_features_in_ = len(self.Xref)

        assert X.shape[1] == self.n_features_in_

        self.is_fitted_ = True

        return self

    def transform(self, X):
        """Transform the data."""
        X = check_array(X, accept_sparse=False, copy=True)  # Force a copy
        check_is_fitted(self, "is_fitted_")
        assert X.shape[1] == self.n_features_in_

        try:
            for row in range(X.shape[0]):
                coef = np.polynomial.polynomial.polyfit(
                    self.Xref, X[row, :], deg=1
                )
                X[row, :] = (X[row, :] - coef[0]) / coef[1]
        except Exception as e:
            raise ValueError("Cannot perform MSC transformation : {}".format(e))

        return X

    def fit_transform(self, X, y=None):
        """Fit and then transform some data."""
        self.fit(X)

        return self.transform(X)


class SNV:
    """Perform a Standard Normal Variates transformation."""

    def __init__(self, robust=False, detrend=False, q=50):
        """
        Instantiate the class.

        The so-called Standard Normal Variate (SNV) method performs a
        normalization of the data by subtracting each row by its own mean
        then dividing by its own standard deviation. After SNV, each row
        will have a mean of 0 and a standard deviation of 1. This
        transformation is equivalent to autoscaling by rows instead of
        by columns.

        SNV attempts to make rows comparable in terms of magnitude (e.g.,
        intensities for spectra or absorbance level). It can be useful to
        correct spectra for calibration.

        From [1]:
        * SNV corrects for both baseline shift and global intensity variations.
        * SNV improves the PLS model prediction, especially when used on
        NIR data with scattering effects.
        * When using SNV, the spectra always have positive and negative values
        centered on 0, which may make interpretation a little more difficult.
        * SNV assumes that multiplicative effects are uniform over the whole
        spectral range, which is not always the case, so artifacts could be
        introduced by this transformation

        "Guo et al. [4] introduced the robust normal variate (RNV) transformation
        to tackle certain artifacts created when using SNV transformation by
        solving the 'closure' problem. Closure is the statistical term indicating
        that the sum of the data is necessarily equal to a certain amount, so
        that if one of the variables changes in one direction, the other variables
        must change in the opposite direction to ensure the constancy of the sum.
        Guo et al. [4] solved this closure problem by introducing the RNV
        transformation that modifies SNV by using the percentile instead of the
        mean."

        Here we implement the version of RNV discussed in [1] which uses the
        percentile to center the data, and the IQR to scale it (cf. pp 165).

        From [1]: "Absorbance spectra in NIR increase linearly with the wavelength for
        transparent samples, whereas it increases curvilinearly for spectra of
        densely packed samples. Therefore, another approach to correct for the
        baseline shift is the Detrend method, which was introduced along with
        the standard normal variates transform (SNV) by Barnes et al. [3]
        Detrending accounts for the variation in baseline shift and
        curvilinearity, generally found in the reflectance spectra of powdered
        or densely packed samples. Using a second-degree polynomial regression,
        the detrend method removes the baseline curvature from each individual
        spectrum by expressing it as a quadratic function of the wavelengths."

        Detrending is generally useful to help remove any baseline which varies
        as a function of time, etc. (e.g., axis=1 of X) - cf. [1] pp 147.
        Using SNV followed by detrending allows you to correct for both shift
        and drift in your baseline. [1,3]

        References
        ----------
        [1] Brown, Steven D., Romà Tauler, and Beata Walczak, eds. Comprehensive
        chemometrics: chemical and biochemical data analysis. Elsevier, 2020.
        [2] https://guifh.github.io/RNIR/SNV.html
        [3] Barnes, R. J.; Dhanoa, M. S.; Lister, S. J. Correction of the
        Description of Standard Normal Variate (SNV) and De-Trend Transformations
        in Practical Spectroscopy with Applications in Food and Beverage Analysis,
        2nd ed. J. Near Infrared Spectrosc. 1993, 1, 185–186.
        [4] Guo, Q.; Wu, W.; Massart, D. L. The Robust Normal Variate Transform
        for Pattern Recognition with Near-Infrared Data. Anal. Chim. Acta 1999,
        382 (1), 87–103.

        Parameters
        ----------
        robust : bool
            Whether or not to use robust statistics (percentile, IQR) instead
            of standard descriptive statistics (mean, standard deviation).
        detrend : bool
            Whether or not to apply a detrend transformation after the SNV
            transformation.
        q : float
            Percentile to use for RNV, if RNV is used; for SNV this is ignored.
        """
        self.set_params(**{"robust": robust, "detrend": detrend, "q": q})
        self.is_fitted_ = False

    def set_params(self, **parameters):
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "robust": self.robust,
            "detrend": self.detrend,
            "q": self.q,
        }

    def fit(self, X, y=None):
        """Store the size of the data (number of columns) for consistency later."""
        X = check_array(X, accept_sparse=False)

        assert (0.0 < self.q) and (100.0 > self.q)

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def transform(self, X):
        """Transform the data."""
        X = check_array(X, accept_sparse=False, copy=True)  # Force a copy
        check_is_fitted(self, "is_fitted_")
        assert X.shape[1] == self.n_features_in_

        try:
            if not self.robust:
                scale_ = np.std(X, axis=1, ddof=1)
                ave_ = np.mean(X, axis=1)
                X = ((X.T - ave_) / scale_).T
            else:  # Uses algorithm from [1] pp 165.
                for row, per in enumerate(
                    np.percentile(X, q=self.q, axis=1, interpolation="linear")
                ):
                    scale = iqr(
                        X[row, :],
                        rng=(25, 75),
                        nan_policy="raise",
                        interpolation="linear",
                    )  # Scale is independent of percentile
                    X[row, :] -= per
                    X[row, :] /= scale
        except Exception as e:
            raise ValueError("Cannot perform SNV transformation : {}".format(e))

        if self.detrend:  # Fit to second order polynomial and substract
            x_ = np.arange(X.shape[1])
            for row in range(X.shape[0]):
                coef = np.polynomial.polynomial.polyfit(x_, X[row, :], deg=2)
                pred = coef[0] + coef[1] * x_ + coef[2] * x_**2
                X[row, :] -= pred

        return X

    def fit_transform(self, X, y=None):
        """Fit and then transform some data."""
        self.fit(X)

        return self.transform(X)


class SavGol:
    """Perform a Savitzky-Golay filtering."""

    def __init__(
        self,
        window_length,
        polyorder,
        deriv=0,
        delta=1.0,
        axis=-1,
        mode="interp",
        cval=0.0,
    ):
        """
        Instantiate the class.

        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
        for description of parameters.
        """
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
        self.is_fitted_ = False

    def set_params(self, **parameters):
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep=True):
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

    def fit(self, X, y=None):
        """Fit the filter using some training data."""
        X = check_array(X, accept_sparse=False)

        try:
            _ = scipy.signal.savgol_filter(
                X,
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

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def transform(self, X):
        """Transform (center and possibly scale) the data after fitting."""
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, "is_fitted_")
        assert X.shape[1] == self.n_features_in_

        X_filtered = scipy.signal.savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            axis=self.axis,
            mode=self.mode,
            cval=self.cval,
        )

        return X_filtered

    def fit_transform(self, X, y=None):
        """Fit and then transform some data."""
        self.fit(X)

        return self.transform(X)
