"""
Load datasets.

author: nam
"""
import io

import pandas as pd
import requests
from sklearn.utils import Bunch


def load_pgaa(return_X_y=False, as_frame=False):
    """
    Load prompt gamma ray activation analysis dataset.

    Parameters
    ----------
    return_X_y : scalar(bool), optional(default=False)
        If True, returns (data, target) instead of a Bunch object.
        See below for more information about the data and target object.

    as_frame : scalar(bool), optional(default=False)
        If True, the data is a pandas DataFrame including columns with appropriate dtypes
        (numeric). The target is a pandas DataFrame or Series depending on the number of
        target columns. If return_X_y is True, then (data, target) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with the following attributes.

        data{ndarray, DataFrame}
            The data matrix. If as_frame=True, data will be a pandas DataFrame.

        target: {ndarray, Series}
            The classification target. If as_frame=True, target will be a pandas Series.

        feature_names: list
            The names of the dataset columns.

        target_names: list
            The names of target classes.

        frame: DataFrame
            Only present when as_frame=True. DataFrame with data and target.

        DESCR: str
            The full description of the dataset.

    (data, target) : tuple if return_X_y is True
        A tuple of two ndarrays by default. The first contains a 2D array
        with each row representing one sample and each column representing the features.
        The second array contains the target samples.

    Notes
    -----
    This is intentionally identical to the API in sklearn.datasets.

    The data here has been preprocessed as in described in [1].

    References
    ----------
    [1] Mahynski, N.A., Monroe, J.I., Sheen, D.A. et al. Classification and authentication of
    materials using prompt gamma ray activation analysis. J Radioanal Nucl Chem 332, 3259–3271
    (2023). https://doi.org/10.1007/s10967-023-09024-x
    """
    # Load data directly from the web so this will reflect any future changes.
    url = "https://raw.githubusercontent.com/mahynski/pgaa-material-authentication/master/data/raw/centers.csv"
    centers = pd.read_csv(
        io.StringIO(requests.get(url).content.decode("utf-8")), dtype=float
    )
    feature_names = [name[0] for name in centers.values.tolist()]

    url = "https://raw.githubusercontent.com/mahynski/pgaa-material-authentication/master/data/raw/X.csv"
    X = pd.read_csv(
        io.StringIO(requests.get(url).content.decode("utf-8")), dtype=float
    )
    X.columns = feature_names
    if not as_frame:
        X = X.values

    target_names = ["Material"]
    url = "https://raw.githubusercontent.com/mahynski/pgaa-material-authentication/master/data/raw/y.csv"
    y = pd.read_csv(
        io.StringIO(requests.get(url).content.decode("utf-8")), dtype=str
    ).squeeze()
    y.name = target_names[0]
    if not as_frame:
        y = y.values

    if return_X_y:
        return (X, y)
    else:
        bunch = Bunch(
            data=X,
            target=y,
            feature_names=feature_names,
            target_names=target_names,
            frame=(None if not as_frame else pd.concat((y, X), axis=1)),
            DESCR="Dataset reported in Mahynski, N.A., Monroe, J.I., Sheen, D.A. et al. \
Classification and authentication of materials using prompt gamma ray activation analysis. J \
Radioanal Nucl Chem 332, 3259–3271 (2023). https://doi.org/10.1007/s10967-023-09024-x . \
See this publication for a full description. Briefly, rows of X are PGAA spectra for different \
materials. They have been normalized to sum to 1. The peaks have be binned into histograms whose \
centers (energy in keV) are given as the feature_names. y contains the name of each material.",
        )
        return bunch


def load_stamp2010(return_X_y=False, as_frame=False):
    """
    Load seabird tissue archival and monitoring project (STAMP) 1999-2010 dataset.

    Parameters
    ----------
    return_X_y : scalar(bool), optional(default=False)
        If True, returns (data, target) instead of a Bunch object.
        See below for more information about the data and target object.

    as_frame : scalar(bool), optional(default=False)
        If True, the data is a pandas DataFrame including columns with appropriate dtypes
        (numeric). The target is a pandas DataFrame or Series depending on the number of
        target columns. If return_X_y is True, then (data, target) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with the following attributes.

        data{ndarray, DataFrame}
            The data matrix. If as_frame=True, data will be a pandas DataFrame.

        target: {ndarray, DataFrame}
            The classification target. If as_frame=True, target will be a pandas DataFrame.

        feature_names: list
            The names of the dataset columns.

        target_names: list
            The names of target classes.

        frame: DataFrame
            Only present when as_frame=True. DataFrame with data and target.

        DESCR: str
            The full description of the dataset.

    (data, target) : tuple if return_X_y is True
        A tuple of two ndarrays by default. The first contains a 2D array
        with each row representing one sample and each column representing the features.
        The second array contains the target samples.

    Notes
    -----
    This is intentionally identical to the API in sklearn.datasets.

    References
    ----------
    [1] Schuur, Stacy S., Ragland, Jared M., Mahynski, Nathan A. (2021), Data Supporting
    "Seabird Tissue Archival and Monitoring Project (STAMP) Data from 1999-2010" ,
    National Institute of Standards and Technology, https://doi.org/10.18434/mds2-2431

    [2] Mahynski, Nathan A., et al. "Seabird Tissue Archival and Monitoring Project (STAMP)
    Data from 1999-2010." Journal of Research of the National Institute of Standards and
    Technology 126 (2021): 1-7.

    [3] Mahynski, Nathan A., et al. "Building Interpretable Machine Learning Models to
    Identify Chemometric Trends in Seabirds of the North Pacific Ocean." Environmental
    Science & Technology 56.20 (2022): 14361-14374.
    """
    # Load data directly from the web so this will reflect any future changes.
    url = "https://raw.githubusercontent.com/mahynski/stamp-dataset-1999-2010/master/X.csv"
    X = pd.read_csv(
        io.StringIO(requests.get(url).content.decode("utf-8")), dtype=float
    )
    feature_names = X.columns.tolist()
    if not as_frame:
        X = X.values

    url = "https://raw.githubusercontent.com/mahynski/stamp-dataset-1999-2010/master/y.csv"
    y = pd.read_csv(io.StringIO(requests.get(url).content.decode("utf-8")))
    target_names = y.columns.tolist()
    if not as_frame:
        y = y.values

    if return_X_y:
        return (X, y)
    else:
        bunch = Bunch(
            data=X,
            target=y,
            feature_names=feature_names,
            target_names=target_names,
            frame=(None if not as_frame else pd.concat((y, X), axis=1)),
            DESCR="Dataset reported in Schuur, Stacy S., Ragland, Jared M., Mahynski, \
Nathan A. (2021), Data Supporting 'Seabird Tissue Archival and Monitoring Project \
(STAMP) Data from 1999-2010', National Institute of Standards and Technology, \
https://doi.org/10.18434/mds2-2431 . See this publication for a full description.",
        )
        return bunch
