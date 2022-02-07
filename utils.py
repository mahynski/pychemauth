"""
General utility functions.

author: nam
"""
import numpy as np
import scipy


def estimate_dof(h_vals, q_vals, n_components, n_features_in):
    """
    Estimate the degrees of freedom for the chi-squared distribution.

    This follows from Ref. 1.

    [1] "Acceptance areas for multivariate classification derived by projection
    methods," Pomerantsev, Journal of Chemometrics 22 (2008) 601-609.
    """

    def err2(N, vals):
        """
        Use a "robust" method for estimating DoF.

        In [1] Eq. 14 suggests the IQR should be divided by the mean (h0),
        however, the citation they provide suggests the median might be
        a better choice; in practice, it seems that is favored since it
        is more robust against outliers, so this is used below in that
        spirit.
        """
        x0 = np.median(vals)  # np.mean(vals)
        a = (scipy.stats.chi2.ppf(0.75, N) - scipy.stats.chi2.ppf(0.25, N)) / N
        b = scipy.stats.iqr(vals, rng=(25, 75)) / x0

        return (a - b) ** 2

    # As in conclusions of [1], Nh ~ n_components is expected
    res = scipy.optimize.minimize(
        err2, n_components, args=(h_vals), method="Nelder-Mead"
    )
    if res.success:
        # Robust method, if possible
        Nh = res.x[0]
    else:
        # Use simple estimate if this fails (Eq. 13 in [1])
        Nh = 2.0 * np.mean(h_vals) ** 2 / np.std(h_vals, ddof=1) ** 2

    # As in conclusions of [1], Nq ~ rank(X)-n_components is expected;
    # assuming near full rank then this is min(I,J)-n_components
    # (n_components<=J)
    res = scipy.optimize.minimize(
        err2,
        np.min([len(q_vals), n_features_in]) - n_components,
        args=(q_vals),
        method="Nelder-Mead",
    )
    if res.success:
        # Robust method, if possible
        Nq = res.x[0]
    else:
        # Use simple estimate if this fails (Eq. 23 in [1])
        Nq = 2.0 * np.mean(q_vals) ** 2 / np.std(q_vals, ddof=1) ** 2

    return Nh, Nq
