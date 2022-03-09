"""
Utility functions for processing PGAA spectra.

author: nam
"""
import numpy as np


def read_spe(
    filename, normalize=True, convert=True, annihilation=True, coarsen=1
):
    """
    Read SPE files specifically provided by Heather Chen-Mayer.

    Parameters
    ----------
    normalize : bool
        Normalize (divide) by the lifetime.
    convert : bool
        Convert bin numbers to energy.
    annihilation : bool
        Remove peak at 511 eV due to positron annihilation. This will only
        work if convert is set to True.
    coarsen : int
        Number of neighboring bins to combine into a single bin. Default of
        1 will not change the input.

    Returns
    -------
    spectra, bins : ndarray, ndarray
        Energy spectrum and (centered) bins (in energy units if converted) they correspond to.
    """
    with open(filename, "r") as f:
        contents = f.read().split("\n")

    first = np.where(["$DATA:" == x for x in contents])[0][0] + 2
    last = np.where(["$ENER_FIT:" == x for x in contents])[0][0]
    lifetime = float(
        contents[
            np.where(["$MEAS_TIM:" == x for x in contents])[0][0] + 1
        ].split(" ")[0]
    )
    intercept = float(contents[last + 1].split(" ")[0])
    slope = float(contents[last + 1].split(" ")[1])

    spectra = np.array(contents[first:last], dtype=np.float64)

    # 1. Convert bins to energy values
    bins = np.arange(len(spectra), dtype=np.float64)
    if convert:
        bins = bins * slope + intercept

    # 2. Normalize before other processing
    spectra = spectra / (lifetime if normalize else 1.0)

    # 3. Remove annihilation peak before coarsening
    positron_peak = 511.0
    if annihilation and convert:
        spectra[
            np.where(
                np.abs(bins - positron_peak)
                == np.min(np.abs(bins - positron_peak))
            )[0][0]
        ] = 0.0

    # 4. Coarsen bins
    spectra = np.array(
        [
            np.sum(spectra[coarsen * start : coarsen * (start + 1)])
            for start in range(len(spectra) // coarsen)
        ],
        dtype=np.float64,
    )
    if convert:
        # Average the energy units
        bins = np.array(
            [
                np.mean(bins[coarsen * start : coarsen * (start + 1)])
                for start in range(len(bins) // coarsen)
            ],
            dtype=np.float64,
        )
    else:
        # Renumber as integers
        bins = np.arange(len(spectra), dtype=np.float64)

    return spectra, bins
