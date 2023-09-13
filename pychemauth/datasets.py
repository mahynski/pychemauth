"""
Load datasets.

author: nam
"""

from sklearn.utils import Bunch


def load_pgaa(return_X_y=False, as_frame=False):
    """
    Load prompt gamma activation analysis dataset.

    Parameters
    ----------
    return_X_y : scalar(bool), optional(default=None)
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

        data{ndarray, dataframe} of shape (X, X)
            The data matrix. If as_frame=True, data will be a pandas DataFrame.

        target: {ndarray, Series} of shape (X,)
            The classification target. If as_frame=True, target will be a pandas Series.

        feature_names: list
            The names of the dataset columns.

        target_names: list
            The names of target classes.

        frame: DataFrame of shape (X, X)
            Only present when as_frame=True. DataFrame with data and target.

        DESCR: str
            The full description of the dataset.

    (data, target) : tuple if return_X_y is True
        A tuple of two ndarrays by default. The first contains a 2D array of shape (X, X)
        with each row representing one sample and each column representing the features.
        The second array of shape (X,) contains the target samples.

    Notes
    -----
    This is intentionally identical to the API in sklearn.datasets.

    References
    ----------
    [1] Mahynski, N.A., Monroe, J.I., Sheen, D.A. et al. Classification and authentication of
    materials using prompt gamma ray activation analysis. J Radioanal Nucl Chem 332, 3259–3271
    (2023). https://doi.org/10.1007/s10967-023-09024-x
    """
    return NotImplementedError


def load_stamp2010(return_X_y=False, as_frame=False):
    """
    Load seabird tissue archival and monitoring project (STAMP) 1999-2010 dataset.

    Parameters
    ----------
    return_X_y : scalar(bool), optional(default=None)
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

        data{ndarray, dataframe} of shape (X, X)
            The data matrix. If as_frame=True, data will be a pandas DataFrame.

        target: {ndarray, Series} of shape (X,)
            The classification target. If as_frame=True, target will be a pandas Series.

        feature_names: list
            The names of the dataset columns.

        target_names: list
            The names of target classes.

        frame: DataFrame of shape (X, X)
            Only present when as_frame=True. DataFrame with data and target.

        DESCR: str
            The full description of the dataset.

    (data, target) : tuple if return_X_y is True
        A tuple of two ndarrays by default. The first contains a 2D array of shape (X, X)
        with each row representing one sample and each column representing the features.
        The second array of shape (X,) contains the target samples.

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
    return NotImplementedError
