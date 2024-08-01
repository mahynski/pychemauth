"""
Load datasets.

author: nam
"""
import io
import requests
import shutil
import tqdm
import os

import pandas as pd
import numpy as np

from sklearn.utils import Bunch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pychemauth.utils import NNTools, fastnumpyio


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


def make_pgaa_images(
    transformer,
    exclude_classes=None,
    directory=None,
    overwrite=False,
    fmt="npy",
    valid_range=(0, 4056),
    renormalize=True,
    test_size=0.0,
    random_state=42,
    batch_size=10,
    shuffle=True,
):
    """
    Create iteratable dataset of 2D single-channel "images" from the included example dataset of 1D PGAA spectra.

    This can serve as a template for other "imaging" transformations of 1D series data.

    Parameters
    ----------
    transformer : sklearn.base.BaseEstimator
        A transformer which follows sklearn's estimator API. The `.fit_transform` method will be called to fit the transformer to the training data.

    exclude_classes : array-like, optional(default=None)
        Iterable containing classes to exlude as strings.  See `pychemauth.datasets.load_pgaa` for classes in this dataset.

    directory : str, optional(default=None)
        Directory to save transformed images to. If None the images are returned as a numpy array in memory; otherwise an `XLoader` is returned.  Within `directory` both "train" and "test" subdirectories are created with the data split accordingly.

    overwrite : bool, optional(default=False)
        If saving data to disk, whether to delete any `directory` that already exists.

    fmt : str, optional(default='npy')
        Format to save the data to disk in.  Default is to use numpy's native "npy" format.

    valid_range : tuple(int, int), optional(default=(0, 4056))
        A lower (inclusive) and upper (exclusive) bound on the spectra energy indices to use. Default values cover the entire range of the dataset.

    normalization : bool, optional(default=True)
        Whether to renormalize (sum to 1) the spectra after clipping to `valid_range`.

    test_size : float, optional(default=0.0)
        Fraction of data to hold out as test set.  If 0 then no test set is created and all data is returned as part of the training set.  Splitting is always done in a stratified manner.

    random_state : int, optional(default=42)
        Random number generator see for stratified train/test splitting.  If `None` no shuffling will be performed, though by default it is.

    Returns
    -------
    If `directory=None`, data is returned in memory as:
        X_train : ndarray(float, ndim=4)
            Training data with shape (N, R, R, 1) where N is the number of observations in the set and R is the width of the `valid_range`.

        X_test : ndarray(float, ndim=4)
            Testing data with similar shape as X_train.

        y_train : ndarray(int, ndim=1)
            Targets for training data.

        y_test : ndarray(int, ndim=1)
             Targets for test data.

        transformer : sklearn.base.BaseEstimator
            Transformer after being trained on X_train.

        encoder : sklearn.preprocessing.LabelEncoder
            Encoder that transforms y from string classes to integers.

    If `directory` is provided, then the data is transformed and saved to disk so loaders are returned as:
        train_loader : utils.NNTools.XLoader
            Dataset loader for the training set.

        test_loader : utils.NNTools.XLoader
            Dataset loader for the test set.

        encoder : sklearn.preprocessing.LabelEncoder
            Encoder that transforms y from string classes to integers.

    Notes
    -----
    Spectral preprocessing steps include (re)normalization (if desired), then natural logarithm (clipped below 1.0e-7).

    Classes are encoded as integers.

    If `directory` is provided so that data is transformed and written to disk, the `transformer` is fit repeatedly on each individual data point so it does not reflect any average over all X_train.

    Raises
    ------
    ValueError
        Invalid `valid_range` tuple.

    FileExistsError
        If `directory`, or a required subdirectory, already exists.

    Example
    -------
    >>> from pyts.image import GramianAngularField
    >>> res = make_pgaa_images(
    ...     transformer=GramianAngularField(method='difference'),
    ...     exclude_classes=['Carbon Powder', 'Phosphate Rock', 'Zircaloy'],
    ...     directory='./data',
    ...     overwrite=False,
    ...     fmt='npy',
    ...     valid_range=(0, 2631),
    ...     renormalize=True,
    ...     test_size=0.2,
    ...     random_state=42
    ... )
    """
    # Load 1D PGAA dataset
    X, y = load_pgaa(return_X_y=True)

    # Exclude any classes desired
    if hasattr(exclude_classes, "__iter__"):
        mask = np.array([False] * X.shape[0])
        for class_ in exclude_classes:
            mask = mask | (y == class_)
        X = X[~mask]
        y = y[~mask]

    # Possibly clip and normalize
    if valid_range[0] >= valid_range[1]:
        raise ValueError("valid_range should go from low to high.")
    else:
        X = X[:, valid_range[0] : valid_range[1]]
        if renormalize:
            X = (X.T / np.sum(X, axis=1)).T

    # Convert to logscale
    X = np.log(np.clip(X, a_min=1.0e-7, a_max=None))

    # Perform test/train splitting if desired
    if test_size > 0.0:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=False if random_state is None else True,
            stratify=y,
        )
    else:
        X_train, X_test = X, None
        y_train, y_test = y, None

    # Encode y as integers
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test) if y_test is not None else y_test

    def _convert(X):  # Convert 2D array to a "single channeled" image
        return np.expand_dims(X, axis=-1)

    if directory is None:  # Return the imaged dataset in memory
        X_train = _convert(
            transformer.fit_transform(X_train)
        )  # Transform the entire dataset in one step
        X_test = (
            _convert(transformer.transform(X_test))
            if X_test is not None
            else X_test
        )

        return X_train, X_test, y_train, y_test, transformer, encoder
    else:  # Save these images to disk and return an interator to the dataset
        if overwrite and os.path.isdir(directory):
            shutil.rmtree(directory)  # Completely wipe old directory

        loaders = {"train": None, "test": None}
        for dset_, y_, subdir_ in [
            (X_train, y_train, "train"),
            (X_test, y_test, "test"),
        ]:
            if dset_ is not None:
                x_files = []
                path = os.path.join(directory, subdir_)
                os.makedirs(
                    path, exist_ok=False
                )  # Will throw an error if this already exists

                for i in tqdm.tqdm(
                    range(dset_.shape[0]), desc=f"Transforming {subdir_} set"
                ):
                    # Transform one at a time - forced to treat each individual observation as the dataset
                    X_ = _convert(
                        transformer.fit_transform(dset_[i : i + 1])[0]
                    )

                    # Save to disk
                    if fmt == "npy":
                        file = os.path.join(path, f"x_{i}.npy")
                        with open(file, "wb") as f:
                            fastnumpyio.save(
                                f, X_
                            )  # faster than np.save(f, X_)
                            x_files.append(os.path.abspath(file))
                    else:
                        raise NotImplementedError(
                            f"Cannot save data in {fmt} format"
                        )

                # For posterity, also save encoded y, even though loaders will not use.
                # This way, a loader can be recreated from this directory in the future.
                with open(os.path.join(path, "y.npy"), "wb") as f:
                    fastnumpyio.save(f, y_)  # faster than np.save(f, y_)

                # Create Sequence
                loaders[subdir_] = NNTools.XLoader(
                    x_files=x_files,
                    y=y_,
                    batch_size=batch_size,
                    fmt=fmt,
                    shuffle=shuffle,
                )

        return loaders["train"], loaders["test"], encoder


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
