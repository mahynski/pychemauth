"""
General utility functions.

author: nam
"""
import copy
import scipy
import joblib
import sklearn
import pychemauth
import imblearn
import pickle
import datetime
import os
import keras
import json
import visualkeras
import wandb
import struct
import tqdm
import sys
import pathlib
import shutil
import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.cross_decomposition import PLSRegression
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.utils.validation import check_array
from sklearn.utils import shuffle as skshuffle
from matplotlib.patches import Ellipse, Rectangle
from pathlib import Path
from tempfile import TemporaryDirectory
from huggingface_hub import hf_hub_download, HfApi, ModelCard, ModelCardData
from tensorflow.keras import backend as K

from typing import Any, Union, Sequence, ClassVar, Callable
from numpy.typing import NDArray


class fastnumpyio:
    """
    Tools from fastnumpyio to accelerate numpy IO.

    These tools can accelerate I/O operations by a factor of ~25.

    This is a copy-paste from https://github.com/divideconcept/fastnumpyio provided under the MIT license, from commit tag 627bb17 + hotfix to address #4.  If this package is ever released on pypi in the future it will be included in the installation rather than an explicit copy here.
    """

    @staticmethod
    def save(file: Any, array: NDArray) -> None:
        """Save a numpy array to disk."""
        magic_string = b"\x93NUMPY\x01\x00v\x00"
        header = bytes(
            (
                "{'descr': '"
                + array.dtype.descr[0][1]
                + "', 'fortran_order': False, 'shape': "
                + str(array.shape)
                + ", }"
            ).ljust(127 - len(magic_string))
            + "\n",
            "utf-8",
        )
        if type(file) == str:
            file = open(file, "wb")
        file.write(magic_string)
        file.write(header)
        file.write(array.data)

    @staticmethod
    def pack(array: NDArray) -> bytes:
        """Pack a numpy array."""
        size = len(array.shape)
        return (
            bytes(
                array.dtype.byteorder.replace(
                    "=", "<" if sys.byteorder == "little" else ">"
                )
                + array.dtype.kind,
                "utf-8",
            )
            + array.dtype.itemsize.to_bytes(1, byteorder="little")
            + struct.pack(f"<B{size}I", size, *array.shape)
            + array.data
        )

    @staticmethod
    def load(file: Any) -> Union[NDArray, None]:
        """Load a numpy array from disk."""
        if type(file) == str:
            file = open(file, "rb")
        header = file.read(128)
        if not header:
            return None
        descr = str(header[19:25], "utf-8").replace("'", "").replace(" ", "")
        shape = tuple(
            int(num)
            for num in str(header[60:120], "utf-8")
            .strip()
            .replace(", }", "")
            .replace("(", "")
            .replace(")", "")
            .split(",")
            if num != ""
        )
        datasize = np.lib.format.descr_to_dtype(descr).itemsize
        for dimension in shape:
            datasize *= dimension
        return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))

    @staticmethod
    def unpack(data: bytes) -> NDArray:
        """Unpack a numpy array."""
        dtype = str(data[:2], "utf-8")
        dtype += str(data[2])
        size = data[3]
        shape = struct.unpack_from(f"<{size}I", data, 4)
        datasize = data[2]
        for dimension in shape:
            datasize *= dimension
        return np.ndarray(
            shape,
            dtype=dtype,
            buffer=data[4 + size * 4 : 4 + size * 4 + datasize],
        )


def _sort_xdata(directory: str) -> tuple[list, int]:
    """Sort x_i.ext files in a directory by their index, i."""
    path = pathlib.Path(directory).absolute()
    sorted_x = [
        pathlib.Path(os.path.join(path, f)).absolute()
        for f in sorted(
            [f_ for f_ in os.listdir(path) if f_.startswith("x_")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
    ]
    final_idx = int(sorted_x[-1].parts[-1].split("_")[1].split(".")[0])

    return [str(p) for p in sorted_x], final_idx


def write_dataset(
    directory: Union[str, pathlib.Path],
    X: NDArray,
    y: NDArray,
    fmt: str = "npy",
    overwrite: bool = False,
    augment: bool = False,
) -> tuple[list[str], str]:
    """
    Write a dataset from memory to disk.

    Each observation in `X` (row, or first dimension) is saved as a separate file named "x_i.ext" where i is the index and ext is the file extension.  All `y` values are saved in a single file called "y.ext".

    Parameters
    ----------
    directory : str or pathlib.Path, optional(default=None)
        Directory to save dataset to.

    X : ndarray
        Dataset features as a numpy array.  First dimension should corresponds to observations.

    y : ndarray
        Target.

    fmt : str, optional(default='npy')
        Format the X data is saved in. Default is numpy.

    overwrite : bool, optional(default=False)
        Whether to delete any `directory` that already exists.

    augment : bool, optional(default=False)
        If True, assume X is being added to any existing data in `directory`; the data is written
        with "x_i.ext" indices starting immediately after whatever currently exists.  Note that if
        `overwrite=True` any existing directory will be removed so this parameter is irrlevant in
        that case.

    Returns
    -------
    x_files : list(str)
        List of absolute paths to files for each row in X, in order.

    y_file : str
        Filename where y has been stored.

    Raises
    ------
    Exception
        If X and y are not numpy arrays with the same length.
        If `directory` already exists and `overwrite` is False.

    Example
    -------
    >>> write_dataset('./data/train', X_train, y_train)
    >>> write_dataset('./data/test', X_test, y_test)
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise Exception("X and y should have the same length.")

    # Work with absolute paths
    directory_ = str(pathlib.Path(directory).absolute())

    if not augment and not overwrite and os.path.isdir(directory_):
        raise Exception(f"{directory_} already exists.")

    if overwrite and os.path.isdir(directory_):
        shutil.rmtree(directory_)  # Completely wipe old directory

    x_start = 0
    if augment and os.path.isdir(directory_):
        _, x_start = _sort_xdata(directory_)
        x_start += 1

    # Create directory if it doesn't already exist
    os.makedirs(directory_, exist_ok=True)

    x_files, y_file = [], ""
    if fmt == "npy":
        # Save y to disk
        file = str(pathlib.Path(os.path.join(directory_, "y.npy")).absolute())
        if augment:
            try:
                y_prev = fastnumpyio.load(file)
            except IOError:
                if x_start > 0:
                    raise Exception(
                        "Augmentation error: previous x files found but y does not seem to exist."
                    )
            else:
                if y_prev is None:
                    raise IOError(
                        f"Cannot load {file} - it seems to have no header."
                    )
                else:
                    y = np.concatenate((y_prev, y), axis=0)
        fastnumpyio.save(file, y)
        y_file = file

        # Save X to disk
        for i in range(X.shape[0]):
            file = str(
                pathlib.Path(
                    os.path.join(directory_, f"x_{i+x_start:09}.npy")
                ).absolute()
            )
            fastnumpyio.save(file, X[i])
            x_files.append(file)
    else:
        raise NotImplementedError(f"Cannot save data in {fmt} format.")

    return x_files, y_file


class NNTools:
    """Tools for working with neural networks."""

    class XLoader(tf.keras.utils.Sequence):
        """
        Dataset loader that retrieves X from disk and y from memory.

        Example
        -------
        >>> head = os.path.abspath('/path/to/directory')
        >>> loader = NNTools.XLoader(
        ...     x_files = [os.path.abspath(os.path.join(head, f)) for f in os.listdir(head)],
        ...     y = np.load('/path/to/y_data'),
        ...     batch_size = 10
        ... )
        """

        x_files: ClassVar[Union[Sequence[str], NDArray[np.str_]]]
        y: ClassVar[NDArray]
        batch_size: ClassVar[int]
        fmt: ClassVar[str]
        shuffle: ClassVar[bool]
        include: ClassVar[Union[Sequence, NDArray, None]]
        exclude: ClassVar[Union[Sequence, NDArray, None]]
        filter: ClassVar[Union[NDArray[np.bool_], None]]

        def __init__(
            self,
            x_files: Union[Sequence[str], NDArray[np.str_]],
            y: NDArray,
            batch_size: int,
            fmt: str = "npy",
            shuffle: bool = False,
            include: Union[Sequence, NDArray, None] = None,
            exclude: Union[Sequence, NDArray, None] = None,
            filter: Union[NDArray[np.bool_], None] = None,
        ) -> None:
            """
            Instantiate the class.

            Parameters
            ----------
            x_files : array_like(str)
                List of filenames, in order, that correspond to X.

            y : ndarray(object, ndim=1)
                Target.

            batch_size : int
                Batch size used during training.

            fmt : str, optional(default='npy')
                Format the X data is saved in. Default is numpy.

            shuffle : bool, optional(default=False)
                Whether or not to shuffle the order of X between epochs. The seed for this is already set during training so one is not assigned here.

            include : array_like, optional(default=None)
                List of `y` values to include when iterating.  If specified, only instances where `y` is in this list will be returned.  This will create uneven or empty batches. You cannot specify this and `exclude` simultaneously.

            exclude : array_like, optional(default=None)
                List of `y` values to exclude when iterating.  This allows you to filter out certain classes.  This will create uneven or empty batches. You cannot specify this and `include` simultanously.

            filter : ndarray(bool, ndim=1)
                Mask to filter instances. The length of this filter should be equal to that of `x_files`.  All instances for which `filter=False` will not be calculated.  The order corresponds to the ordering of `x_files` when instantiated.  This filter will be shuffled consistently with `x_files` and `y` at the end of an epoch, if desired.  This will create uneven or empty batches.  This can be simultaneously specified with either `include` or `exclude`.  Only observations which satisfy both criteria are calculated and returned.

            Raises
            ------
            NotImplementedError
                If `fmt` is unsupported.
            """
            if len(x_files) != len(y):
                raise Exception("X and y should have the same length.")

            self.__x, self.__y = x_files, y
            self.x_orig, self.y_orig = copy.copy(self.__x), copy.copy(
                self.__y
            )  # Save the original ordering

            # This is only intended to be used when instantiated
            self._set_filter(filter)
            self.filter_orig = copy.copy(self.__filter)

            self.batch_size = int(batch_size)  # type: ignore[misc]
            self.shuffle = shuffle  # type: ignore[misc]

            self.__exclude: Union[Sequence, NDArray, None] = None
            self.__include: Union[Sequence, NDArray, None] = None
            self.set_include(include)
            self.set_exclude(exclude)

            if fmt == "npy":
                self.load = fastnumpyio.load  # faster than np.load
            else:
                raise NotImplementedError(f"Cannot load data in {fmt} format")

        def set_include(self, include: Union[Sequence, NDArray, None]) -> None:
            """Set the include value."""
            if self.__exclude is not None and include is not None:
                raise Exception(
                    "You cannot specify include and exclude simultaneously."
                )
            self.__include = include

        def set_exclude(self, exclude: Union[Sequence, NDArray, None]) -> None:
            """Set the exclude value."""
            if self.__include is not None and exclude is not None:
                raise Exception(
                    "You cannot specify include and exclude simultaneously."
                )
            self.__exclude = exclude

        def _set_filter(self, filter: Union[NDArray[np.bool_], None]) -> None:
            """
            Set the filter value.

            Notes
            -----
            This is only intended to be used when instantiated.
            """
            self.__filter: Union[NDArray[np.bool_], None]
            if filter is not None:
                self.__filter = np.asarray(filter, dtype=bool)
                assert (
                    self.__filter.ndim == 1
                ), "Filter should only have 1 dimension."
                assert self.__filter.shape[0] == len(self.x_orig)
            else:
                self.__filter = None

        def __len__(self) -> int:
            """Return the number of datapoints in a batch."""
            return int(np.ceil(len(self.__x) / self.batch_size))

        def __getitem__(self, idx: int) -> tuple[NDArray, NDArray]:
            """
            Retrieve a batch of data.

            Parameters
            ----------
            idx : int
                Batch index.

            Returns
            -------
            X_batch : ndarray
                Numpy array of X data for this batch.

            y_batch : ndarray(ndim=1)
                Target data for this batch.
            """
            low = idx * self.batch_size
            # Cap upper bound at array length; the last batch may be smaller
            # if the total number of items is not a multiple of batch size.
            high = min(low + self.batch_size, len(self.__x))
            batch_x = self.__x[low:high]
            batch_y = self.__y[low:high]

            # Possibly filter out X
            if self.__filter is None:
                filter_mask = np.array([True] * len(batch_y), dtype=bool)
            else:
                filter_mask = self.__filter[low:high]

            # Possibly filter out based on y values
            if (self.__exclude is not None) and (self.__include is not None):
                raise Exception(
                    "You cannot specify include and exclude simultaneously."
                )

            if self.__include is not None:
                mask = np.array(
                    [y_ in self.__include for y_ in batch_y], dtype=bool
                )
            elif self.__exclude is not None:
                mask = ~np.array(
                    [y_ in self.__exclude for y_ in batch_y], dtype=bool
                )
            else:
                mask = np.array([True] * len(batch_y), dtype=bool)

            # Net filter
            mask = mask & filter_mask

            return (
                np.array([self.load(file_name) for file_name in batch_x])[mask],
                np.array(batch_y)[mask],
            )

        def on_epoch_end(self) -> None:
            """Execute changes at the end of a training epoch."""
            if self.shuffle:  # Shuffle if desired
                if self.__filter is None:
                    self.__x, self.__y = skshuffle(self.__x, self.__y)
                else:
                    self.__x, self.__y, self.__filter = skshuffle(
                        self.__x, self.__y, self.__filter
                    )

    @staticmethod
    def build_loader(
        directory: Union[str, pathlib.Path],
        loader: str = "x",
        batch_size: int = 1,
        fmt: str = "npy",
        shuffle: bool = False,
        include: Union[Sequence, NDArray, None] = None,
        exclude: Union[Sequence, NDArray, None] = None,
        filter: Union[NDArray[np.bool_], None] = None,
    ) -> tf.keras.utils.Sequence:
        """
        Build a dataset loader from a directory.

        Parameters
        ----------
        directory : str or pathlib.Path, optional(default=None)
            Directory to read dataset from.  Should be the directory used in `write_dataset`.

        loader : str, optional(default='x')
            Type of Loader to create; default is XLoader.

        batch_size : int
            Batch size used during training.

        fmt : str, optional(default='npy')
            Format the X data is saved in. Default is numpy.

        shuffle : bool, optional(default=False)
            Whether or not to shuffle the order of X between epochs. The seed for this is already set during training so one is not assigned here.

        include : array_like, optional(default=None)
            List of `y` values to include when iterating. If specified, only instances where `y` is in this list will be returned. This will create uneven or empty batch sizes. You cannot specify this and `exclude` simultaneously.

        exclude : array_like, optional(default=None)
            List of `y` values to exclude when iterating. This allows you to filter out certain classes. This will create uneven or empty batch sizes. You cannot specify this and `include` simultaneously.

        filter : numpy.ndarray(bool, ndim=1)
            Mask to filter instances. The length of this filter should be equal to that of `y`.  All instances for which `filter=False` will not be calculated.  The order corresponds to the ordering when instantiated.  This filter will be shuffled consistently with `x` and `y` at the end of an epoch, if desired.  This will create uneven or empty batches.  This can be simultaneously specified with either `include` or `exclude`.  Only observations which satisfy both criteria are calculated and returned.

        Notes
        -----
        The directory should have the correct structure, i.e., that which is created by `write_dataset`.

        Example
        -------
        >>> Train_Loader = NNTools.build_loader('./data/train', batch_size=10)
        >>> Test_Loader = NNTools.build_loader('./data/test', batch_size=10)
        """
        directory_ = str(pathlib.Path(directory).absolute())

        def _read_y(y_file):
            if y_file.endswith("npy"):
                return fastnumpyio.load(y_file)
            else:
                raise NotImplementedError(f"Cannot save data in {fmt} format.")

        if loader.lower() == "x":
            if not os.path.isdir(directory_):
                raise Exception(f"{directory_} does not exist.")

            y_file = os.path.join(directory_, f"y.{fmt}")
            if not os.path.isfile(y_file):
                raise Exception(f"Could not find {y_file} in {directory_}.")

            loader = NNTools.XLoader(
                x_files=_sort_xdata(directory_)[0],
                y=_read_y(y_file),
                batch_size=batch_size,
                fmt=fmt,
                shuffle=shuffle,
                include=include,
                exclude=exclude,
                filter=filter,
            )

        else:
            raise NotImplementedError(
                f"Cannot create an '{loader}' style Loader."
            )

        return loader

    class LearningRateFinder(keras.callbacks.Callback):
        """
        Keras Callback for finding bounds on optimal learning rates for neural networks.

        See `NNTools.find_learning_rate` for a more user-friendly interface to this function.

        Essentially, at the beginning of training the learning rate is set to a lower bound. Then it is increased exponentially after each batch. Initially, the loss function should not change since the learning rate is too low; however, a minimum in the loss tends to appear at intermediate learning rates, before "exploding" at high rates.

        The band between roughly the point where the loss starts to change and the minimum is a reasonably optimal learning rate.  Cyclical learning rates can also be used to dynamically move between them.

        Notes
        -----
        Code inspired by: https://pyimagesearch.com/2019/08/05/keras-learning-rate-finder/

        References
        ----------
        1. "Cyclical Learning Rates for Training Neural Networks," L. Smith, arXiv:1506.01186v6 (2017) (https://arxiv.org/pdf/1506.01186).

        Example
        -------
        >>> finder = NNTools.find_learning_rate(
        ...     model=CNNFactory(...),
        ...     (X_train, y_train),
        ...     compiler_kwargs=compiler_kwargs
        ... )
        >>> ax = finder.plot()
        >>> ax.set_yscale('log')
        >>> for l_ in finder.estimate_clr():
        ...     ax.axvline(l_, color='red')
        """

        start_lr: ClassVar[float]
        end_lr: ClassVar[float]
        n_updates: ClassVar[int]
        stop_factor: ClassVar[float]
        beta: ClassVar[float]

        def __init__(
            self,
            start_lr: float = 1.0e-8,
            end_lr: float = 10.0,
            n_updates: int = 100,
            stop_factor: float = 4.0,
            beta: float = 0.98,
        ) -> None:
            """
            Instantiate the class.

            Parameters
            ----------
            start_lr : float, optional(default=1.0e-8)
                Initial learning rate.  Should be a value expected to be too small.

            end_lr : float, optional(default=10.0)
                Final learning rate.  Should be a value expected to be too large.

            n_updates : int, optional(default=100)
                Number of times to update learning rate; this is done after each batch so this determines
                the total number of batches (and therefore, epochs) run.

            stop_factor : float, optional(default=4.0)
                The factor multiplied by the smallest loss found to determine the limit at which the run should stop.  This stops the training from continuing after the loss goes through a minimum.

            beta : float, optional(default=0.98)
                Smoothing factor used to smooth the loss value over time.
            """
            super(NNTools.LearningRateFinder, self).__init__()
            self.start_lr = start_lr  # type: ignore[misc]
            self.end_lr = end_lr  # type: ignore[misc]
            self.n_updates = n_updates  # type: ignore[misc]
            self.stop_factor = stop_factor  # type: ignore[misc]
            self.beta = beta  # type: ignore[misc]

            self._smoothed_loss: list = []
            self._loss: list = []
            self._lr: list = []
            self.batch_num = 0
            self.avg_loss = 0
            self.best_loss = np.inf
            self.lr_mult = (self.end_lr / self.start_lr) ** (1.0 / n_updates)

        @property
        def smoothed_loss(self) -> list:
            """Return the smoothed loss."""
            return copy.copy(self._smoothed_loss)

        @property
        def loss(self) -> list:
            """Return the (unsoothed) loss."""
            return copy.copy(self._loss)

        @property
        def learning_rate(self) -> list:
            """Return the learning rates."""
            return copy.copy(self._lr)

        def on_train_begin(self, logs=None) -> None:
            """Configure the model parameters when the training starts."""
            K.set_value(
                self.model.optimizer.lr, self.start_lr
            )  # Set optimizer's initial learning rate

        def on_train_batch_end(self, batch, logs=None) -> None:
            """Update the learning rate and save history."""
            current_lr = K.get_value(self.model.optimizer.lr)  # Current state
            current_loss = logs["loss"]

            self._lr.append(current_lr)
            self._loss.append(current_loss)

            # Smooth loss and save
            self.batch_num += 1
            self.avg_loss = (self.beta * self.avg_loss) + (
                (1.0 - self.beta) * current_loss
            )
            smoothed_loss = self.avg_loss / (1 - (self.beta**self.batch_num))
            self._smoothed_loss.append(smoothed_loss)

            # Check if we should stop - i.e., loss started to grow at high LR
            stop_loss = self.stop_factor * self.best_loss
            if self.batch_num > 1 and smoothed_loss > stop_loss:
                self.model.stop_training = True
                return

            # Update best loss
            if self.batch_num == 1 or smoothed_loss < self.best_loss:
                self.best_loss = smoothed_loss

            # Update lr for next batch
            K.set_value(self.model.optimizer.lr, current_lr * self.lr_mult)

        def plot(
            self, ax: Union[matplotlib.pyplot.Axes, None] = None
        ) -> matplotlib.pyplot.Axes:
            """
            Plot the smoothed loss vs. the learning rate.

            This enables the user to manually determine their ideal learning rate visually.

            Parameters
            ----------
            ax : matplotlib.pyplot.Axes, optional(default=None)
                Axes object to plot the results on.

            Returns
            -------
            ax : matplotlib.pyplot.Axes
                Axes object the results are plotted on.
            """
            if ax is None:
                fig, ax = plt.subplots()

            ax.plot(self.learning_rate, self.smoothed_loss, "-")
            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Loss")

            return ax

        def estimate_clr(
            self, frac: float = 0.75, skip: int = 0
        ) -> tuple[float, float]:
            """
            Automatically estimate the upper and lower cyclical learning rate bounds.

            The upper bound is taken as the order of magnitude just below the minima; e.g., if `lr_opt` = 0.0123 then the `max_lr` = 0.01.  The lower bound is taken as the learning rate where the loss is 25% of the way from the value at a learning rate of 0 to the minimum.

            Parameters
            ----------
            frac : float, optional(default=0.75)
                Determines the lower bound on the learning rate which is set when the loss is 100*(1-`frac`)% of the way from the value at a learning rate of 0 to the minimum.

            skip : int, optional(default=0)
                Number of points to skip from the beginning of the sweep when performing analysis.  This can be helpful to trim off any initial dips or other unusual behavior from the warmup stage.

            Returns
            -------
            base_lr : float
                Minimum learning rate to use.

            max_lr : float
                Maximum learning rate to use.

            Raises
            ------
            Exception
                Minimum in smoothed loss vs. learning rate occurs in the first 1/5 of the rates.
                The estimate of the loss when the learning rate goes to 0 is higher than the minimum.
            """
            _smoothed_loss = copy.copy(self._smoothed_loss[skip:])
            _lr = copy.copy(self._lr[skip:])

            min_idx_ = np.argmin(_smoothed_loss)
            limit_ = 5  # The mean of the first 1/limit_ "chunk" of losses will be taken as an estimate of LR -> 0
            if min_idx_ < len(_lr) // limit_:
                raise Exception(
                    f"Minimum in loss occurs in the first 1/{limit_} of the learning rates; recommend reducing the lower bound and re-running."
                )

            # Round down to nearest order of magnitude for maximum LR at best loss
            max_lr = 10 ** (np.floor(np.log10(_lr[min_idx_])))

            # Find the location that corresponds to just below the max_lr
            idx_ = np.argmin((np.array(_lr) - max_lr) ** 2)
            while _lr[idx_] > max_lr and idx_ >= 0:
                idx_ -= 1

            # Take lower bound of LR as the point where 90% of the gap from LR -> 0 and best LR
            baseline_loss_ = np.mean(_smoothed_loss[: (len(_lr) // limit_)])
            if baseline_loss_ < _smoothed_loss[idx_]:
                raise Exception(
                    "Cannot estimate cyclical learning rate bounds automatically; inspect visually instead."
                )
            threshold_ = _smoothed_loss[idx_] + frac * (
                baseline_loss_ - _smoothed_loss[idx_]
            )

            # Walk toward LR = 0 to find this point
            while _smoothed_loss[idx_] < threshold_ and idx_ >= 0:
                idx_ -= 1
            base_lr = _lr[idx_ + 1]

            return base_lr, max_lr

    class CyclicalLearningRate(keras.callbacks.Callback):
        """
        Create a cyclical learning rate policy for a Keras model during training.

        Notes
        -----
        Code inspired by: https://github.com/bckenstler/CLR/blob/master/clr_callback.py

        See `NNTools.find_learning_rate` for a user-friendly way to estimate the `base_lr` and `max_lr`.

        References
        ----------
        1. "Cyclical Learning Rates for Training Neural Networks," L. Smith, arXiv:1506.01186v6 (2017) (https://arxiv.org/pdf/1506.01186).

        Example
        -------
        >>> finder = NNTools.find_learning_rate(
        ...     model=CNNFactory(...),
        ...     (X_train, y_train),
        ...     compiler_kwargs=compiler_kwargs
        ... )
        >>> clr = NNTools.CyclicalLearningRate(
        ...     base_lr=finder.estimate_clr()[0],
        ...     max_lr=finder.estimate_clr()[1],
        ...     step_size=20,
        ... )
        >>> model = NNTools.train(
        ...     model=CNNFactory(...),
        ...     data=(X_train, y_train),
        ...     fit_kwargs={
        ...         'batch_size': 50,
        ...         'epochs': 100,
        ...         'validation_split': 0.2,
        ...         'shuffle': True,
        ...         'callbacks': [clr]
        ...     },
        ...     compiler_kwargs=compiler_kwargs, # Should be the same as finder
        ...     model_filename=None,
        ...     history_filename=None
        ... )
        """

        base_lr: ClassVar[float]
        max_lr: ClassVar[float]
        step_size: ClassVar[int]
        mode: ClassVar[str]
        gamma: ClassVar[float]
        scale_fn: ClassVar[Union[Callable[[int], Any], None]]
        scale_mode: ClassVar[str]

        def __init__(
            self,
            base_lr: float = 0.001,
            max_lr: float = 0.01,
            step_size: int = 20,
            mode: str = "triangular2",
            gamma: float = 1.0,
            scale_fn: Union[Callable[[int], Any], None] = None,
            scale_mode: str = "cycle",
        ) -> None:
            """
            Instantiate the class.

            Parameters
            ----------
            base_lr : float, optional(default=0.001)
                Minimum learning rate to use.

            max_lr : float, optional(default=0.01)
                Maximum learning rate to use.

            step_size : int, optional(default=2000)
                Number of iterations in half a cycle.

            mode : str, optional(default='triangular2')
                Mode defined in Ref. [1].  Determines the shape of the learning function over time.  Should be one of {'triangular', 'triangular2', 'exp_range'}.
                If a `scale_fn` is provided then this argument is ignored.

            gamma : float, optional(default=1.0)
                Constant in 'exp_range' scaling function `gamma`**(cycle iterations); ignored if not using this policy.

            scale_fn : callable, optional(default=None)
                Custom scaling policy defined by a single argument lambda function, where 0 <= `scale_fn`(x) <= 1 for all x >= 0. The `mode` paramater is ignored if this is provided.

            scale_mode : str, optional(default='cycle')
                Defines whether `scale_fn` is evaluated on cycle number or cycle iterations (training iterations since start of cycle). Should be in {'cycle', 'iterations'}.
            """
            super(NNTools.CyclicalLearningRate, self).__init__()
            self.base_lr = base_lr  # type: ignore[misc]
            self.max_lr = max_lr  # type: ignore[misc]
            self.step_size = step_size  # type: ignore[misc]
            self.mode = mode  # type: ignore[misc]
            self.gamma = gamma  # type: ignore[misc]
            self.scale_fn = scale_fn  # type: ignore[misc]
            self.scale_mode = scale_mode  # type: ignore[misc]

            self._clr_iterations: float = 0.0
            self._trn_iterations: float = 0.0
            self._history: dict = {}

        @property
        def history(self) -> dict:
            """Return the learning rates."""
            return copy.copy(self._history)

        def _reset(self) -> None:
            """Reset the cycle iterations."""
            self._clr_iterations = 0.0

        def _clr(self) -> float:
            """Compute instantaneous learning rate."""
            if self.scale_fn is None:
                if self.mode == "triangular":
                    scale_fn = lambda x: 1.0
                    scale_mode = "cycle"
                elif self.mode == "triangular2":
                    scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                    scale_mode = "cycle"
                elif self.mode == "exp_range":
                    scale_fn = lambda x: self.gamma ** (x)
                    scale_mode = "iterations"
                else:
                    raise ValueError(
                        f'Unrecognized mode {self.mode}; should be in {"triangular", "triangular2", "exp_range"}'
                    )
            else:
                scale_fn = self.scale_fn
                scale_mode = self.scale_mode

            cycle = np.floor(1 + self._clr_iterations / (2 * self.step_size))
            x = np.abs(self._clr_iterations / self.step_size - 2 * cycle + 1)
            if scale_mode == "cycle":
                return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(
                    0, (1 - x)
                ) * scale_fn(cycle)
            else:
                return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(
                    0, (1 - x)
                ) * scale_fn(self._clr_iterations)

        def on_train_begin(self, logs=None) -> None:
            """Initialize the learning rate when training starts."""
            if self._clr_iterations == 0:
                K.set_value(self.model.optimizer.lr, self.base_lr)  # First run
            else:
                K.set_value(self.model.optimizer.lr, self._clr())  # Restarting

        def on_epoch_begin(self, epoch, logs=None) -> None:
            """Update the learning rate at the end of an epoch."""
            self._trn_iterations += 1
            self._clr_iterations += 1

            self._history.setdefault("lr", []).append(
                K.get_value(self.model.optimizer.lr)
            )
            self._history.setdefault("iterations", []).append(
                self._trn_iterations
            )

            for k, v in logs.items():
                self._history.setdefault(k, []).append(v)

            K.set_value(self.model.optimizer.lr, self._clr())

    @staticmethod
    def visualkeras(
        model: keras.Model,
        legend: bool = False,
        draw_volume: bool = True,
        scale_xy: float = 0.25,
        scale_z: float = 10.0,
        max_z: int = 200,
        padding: int = 50,
        **kwargs: Any,
    ) -> None:
        """
        Use visualkeras to visualize a Keras model as a PIL image.

        This should work for an arbitrary model, but the defaults are tuned for convolutional neural networks.

        Parameters
        ----------
        model : keras.Model
            A Keras model to visualize.

        legend : bool, optional(default=False)
            Add a legend of the layers to the image.

        draw_volume : bool, optional(default=True)
            Flag to switch between 3D volumetric view and 2D box view. True is generally better for CNN architectures.

        scale_xy : float, optional(default=0.25)
            Scalar multiplier for the x and y size of each layer.

        scale_z : float, optional(default=10.0)
            Scalar multiplier for the z size of each layer.

        max_z : int, optional(default=200)
            Maximum z size in pixel a layer will have.

        padding : int, optional(default=50)
            Distance in pixel before the first and after the last layer.

        kwargs : dict
            Optional other keyword arguments to `visualkeras.layered_view`.

        References
        ----------
        1. Gavrikov, Paul, "visualkeras" https://github.com/paulgavrikov/visualkeras (2020).

        Example
        -------
        >>> model = CNNFactory(...)
        >>> img = NNTools.visualkeras(model)
        >>> img.save('my_model.png')
        """

        def _text_callable(
            layer_index, layer
        ):  # Based on visualkeras documentation example
            # Every other piece of text is drawn above the layer, the first one above
            above = not bool(layer_index % 2)

            # If the output shape is a list of tuples, we only take the first one
            output_shape = [
                x for x in list(layer.output_shape) if x is not None
            ]
            if isinstance(output_shape[0], tuple):
                output_shape = list(output_shape[0])
                output_shape = [x for x in output_shape if x is not None]

            # Create a string representation of the output shape
            output_shape_txt = ""
            for ii in range(len(output_shape)):
                output_shape_txt += str(output_shape[ii])
                if (
                    ii < len(output_shape) - 1
                ):  # Add an x between dimensions, e.g. 3x3
                    output_shape_txt += "x"

            # Remove the trailing index of layers for better visualization
            segments = layer.name.split("_")
            try:
                int(segments[-1])  # Last segment is an integer index
                name = "_".join(segments[:-1])
            except:
                name = layer.name  # Otherwise use the whole name

            # Add the name of the layer to the text, as a new line
            output_shape_txt += f"\n{name}"

            # Return the text value and if it should be drawn above the layer
            return output_shape_txt, above

        img = visualkeras.layered_view(
            model=model,
            legend=legend,
            draw_volume=draw_volume,
            scale_xy=scale_xy,
            scale_z=scale_z,
            max_z=max_z,
            text_callable=_text_callable,
            padding=padding,
        ).convert(
            "RGB"
        )  # default RGBA has some display issues in Jupyter Notebooks

        return img

    @staticmethod
    def _is_data_iter(data: Any) -> bool:
        """Check if data is an iterator that returns batches."""
        classes = [
            tf.keras.preprocessing.image.NumpyArrayIterator,
            tf.keras.preprocessing.image.DirectoryIterator,
            tf.keras.utils.Sequence,
            tf.keras.utils.experimental.DatasetCreator,
            tf.data.Iterator,
            tf.data.NumpyIterator,
            tf.data.Dataset,
        ]
        for class_ in classes:
            if isinstance(data, class_):
                return True
        return False

    @staticmethod
    def find_learning_rate(
        model: keras.Model,
        data: Any,
        start_lr: float = 1.0e-8,
        end_lr: float = 1.0e1,
        n_updates: int = 100,
        stop_factor: float = 4.0,
        beta: float = 0.98,
        batch_size: int = 100,
        compiler_kwargs: dict = {
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "weighted_metrics": ["accuracy", "sparse_categorical_accuracy"],
        },
        seed: int = 42,
        class_weight: Union[dict, None] = None,
    ) -> "NNTools.LearningRateFinder":
        """
        Scan through different learning rates to understand how the model trains.

        Default parameters correspond to classification or authentication applications.

        Parameters
        ----------
        model : keras.Model
            Uncompiled model to use.

        data : tuple or data iterator
            If the data is in memory this can be provided as (`X_train`, `y_train`). Otherwise, provide an iterable, like `NNTools.XLoader` that can provide data in batches, consistent with Keras' API.  If the latter is provided, `batch_size` is ignored.

        start_lr : float, optional(default=1.0e-8)
            Initial learning rate.  Should be a value expected to be too small.

        end_lr : float, optional(default=10.0)
            Final learning rate.  Should be a value expected to be too large.

        n_updates : int, optional(default=100)
            Number of times to update learning rate; this is done after each bactch so this determines the total number of batches / epochs run.

        stop_factor : float, optional(default=4.0)
            The factor multiplied by the smallest loss found to determine the limit at which the run should stop.  This stops the training from continuing after the loss goes through a minimum.

        beta : float, optional(default=0.98)
           Smoothing factor used to smooth the loss value over time.

        batch_size : int, optional(default=100)
            Number of instances per batch. A large `batch_size` can lead to memory overflow if you have large images or other data.  Consider reducing this if you run in issues.  If `data` is provided as an iterable data loader, then this is ignored.

        compiler_kwargs : dict(str, object), optional(default={'optimizer': 'adam', 'loss': 'sparse_categorical_crossentropy', 'weighted_metrics': ['accuracy', 'sparse_categorical_accuracy'],})
            Arguments to use when compiling the `model`. See https://keras.io/api/models/model_training_apis/#compile-method for details.

        seed : int, optional(default=42)
            Seed for keras random weight initialization when the model is compiled.

        class_weight : dict(int, float), optional(default=None)
            Dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). By default, use class weighting inversely proportional to observation frequency as in sklearn's RandomForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html. To turn off weighting, specify `class_weight={c:1 for c in range(N)}` where `N` is the number of classes in the model. Turning off weighting means the 'weighted_metrics' simply become equivalent to 'metrics'.  Normally, this weighting is part of `fit_kwargs` but we have kept it separate here because the default behavior used is different from Keras.  This will be added to `fit_kwargs` before the model is fit, and will overwrite any class_weight previously specified in `fit_kwargs`.

        Returns
        -------
        finder : NNTools.LearningRateFinder
            Object containing the history of the loss as a function of learning rate.  This can be used to plot the results directly.

        Notes
        -----
        This is intended to be fast so should not require logging tools.

        Example
        -------
        >>> finder = NNTools.find_learning_rate(
        ...     model=CNNFactory(...),
        ...     (X_train, y_train),
        ...     compiler_kwargs=compiler_kwargs
        ... )
        """
        finder = NNTools.LearningRateFinder(
            start_lr=start_lr,
            end_lr=end_lr,
            n_updates=n_updates,
            stop_factor=stop_factor,
            beta=beta,
        )

        if not NNTools._is_data_iter(data):
            N = data[0].shape[0]
        else:
            batch_size = data[0][0].shape[0]  # Take size of X from first batch
            N = (len(data) - 1) * batch_size + len(
                data[len(data) - 1][0]
            )  # Last batch may be smaller so account for last one explicitly
        epochs = int(np.ceil(n_updates * batch_size / float(N)))

        # Do a "fast" training updated LR with every batch
        NNTools.train(
            model=model,
            data=data,
            compiler_kwargs=compiler_kwargs,
            fit_kwargs={
                "batch_size": batch_size,
                "epochs": epochs,
                "validation_split": 0.0,  # No validation for this search
                "validation_data": None,
                "shuffle": True,  # Automatically ignored if data is an iterator
                "callbacks": [
                    finder,  # Update LR after each batch
                ],
            },
            class_weight=class_weight,
            seed=seed,
            wandb_project=None,  # No tracking or saving - this should be a 'fast' run
            wandb_kwargs=None,
            model_filename=None,
            history_filename=None,
        )

        return finder

    @staticmethod
    def load(
        filepath: Union[str, pathlib.Path],
        weights_only: bool = False,
        model: Union[keras.Model, None] = None,
    ) -> keras.Model:
        """
        Load a Keras model from disk.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to saved information (weights or a complete `model`).

        weights_only : bool, optional(default=False)
            If True, the `filepath` should contain the weights of the `model` to load.  The `model` provided should have the same architecture. and will not be compiled after loading the weights. If False, assumes the `filepath` points to a complete keras.Model which will be compiled automatically after loading.

        model : keras.Model, optional(default=None)
            Model to use if `weights_only=True`; it does not need to be compiled so you can use `NNFactory()` to produce a new `model` that will inherit weights from an old `model` of the exact same architecture.

        Returns
        -------
        model : keras.Model
            Model loaded from disk.
        """
        if weights_only:
            model.load_weights(filepath, skip_mismatch=False)  # type: ignore[union-attr]
        else:
            model = keras.saving.load_model(
                filepath, custom_objects=None, compile=True
            )

        return model

    @staticmethod
    def _json_serializable(arg_dict: dict) -> dict:
        """Check what can be serialized from a dictionary so WandB can save these parts as its config."""

        def _safe(x):
            try:
                json.dumps(x)
                return True
            except:
                return False

        new_args = copy.copy(arg_dict)
        for key, value in arg_dict.items():
            if not _safe(value):
                new_args.pop(key)

        return new_args

    @staticmethod
    def _summarize_batches(
        data: Any,
    ) -> tuple[int, int, dict, NDArray, NDArray]:
        """
        Perform 1 iteration over the dataset to summarize its contents for various consistency checks.

        Parameters
        ----------
        data : data iterator
            A data iterator of some kind for a Keras model.

        Returns
        -------
        N_data : int
            Number of datapoints total.

        N_batches : int
            Total number of batches.

        unique_targets : dict(int or str, int)
            Unique classes found in the data and their counts in the dataset.  If not a classification problem returns an empty dictionary.

        X_batch : ndarray
            First batch of `X` data.

        y_batch : ndarray
            First batch of `y` data.
        """
        if len(data) < 1:
            raise Exception("data iterator appears to be empty")

        N_data, N_batches = 0, 0
        X_, y_ = data[N_batches]
        X_batch, y_batch = copy.copy(X_), copy.copy(
            y_
        )  # Save these to return as an example

        classification = False
        if y_.ndim == 1:  # Single target
            if (y_.dtype.type == np.str_) or (y_.dtype == int):
                classification = True
        elif (
            y_.ndim == 2
        ):  # If classification then this must be OHE matrix so only ints valid
            if y_.dtype == int:
                classification = True
        else:
            raise Exception("Target should have no more than 2 dimensions.")

        unique_targets: dict = {}

        def _update(y_batch):  # Update class count for classification
            if y_batch.ndim == 1:  # Single integers or strings for each class
                d = dict(
                    map(
                        lambda i, j: (i, j),
                        *np.unique(y_batch, return_counts=True),
                    )
                )
            elif y_batch.ndim == 2:  # OHE integers for classes
                d = dict(
                    map(
                        lambda i, j: (i, j),
                        np.arange(y_batch.shape[1]),
                        np.sum(y_batch, axis=0),
                    )
                )
            else:
                raise Exception("Target should have no more than 2 dimensions.")

            for k, v in d.items():
                try:
                    unique_targets[k] += v
                except:
                    unique_targets[k] = v

        # This is slow, but doesn't depend on internal variables, etc. that might vary between different types of iterators
        # len(data) will given N_batches, but won't give us info on what is in y, etc.
        while tqdm.tqdm(
            len(X_) > 0,
            desc="Iterating through all batches to summarize, be patient...",
        ):
            if classification:
                _update(y_)
            N_data += len(X_)

            N_batches += 1
            X_, y_ = data[N_batches]

        return N_data, N_batches, unique_targets, X_batch, y_batch

    @staticmethod
    def train(
        model: keras.Model,
        data: Any,
        compiler_kwargs: dict = {
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "weighted_metrics": ["accuracy", "sparse_categorical_accuracy"],
        },
        fit_kwargs: dict = {
            "batch_size": 100,
            "epochs": 100,
            "validation_split": 0.0,
            "validation_data": None,
            "shuffle": True,  # Automatically ignored if data is an iterator
            "callbacks": [
                keras.callbacks.ModelCheckpoint(
                    filepath="./checkpoints/model.{epoch:05d}",
                    save_weights_only=True,
                    monitor="val_sparse_categorical_accuracy",
                    save_freq="epoch",
                    mode="max",
                    save_best_only=False,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.1,
                    patience=10,
                    verbose=0,
                    mode="auto",
                    min_delta=0.0001,
                    cooldown=0,
                    min_lr=1.0e-6,
                ),
            ],
        },
        class_weight: Union[dict, None] = None,
        seed: int = 42,
        wandb_project: Union[str, None] = None,
        wandb_kwargs: Union[dict, None] = None,
        model_filename: Union[str, None] = "trained_model.keras",
        history_filename: Union[str, None] = "training_history.pkl",
        restart: Union[dict, None] = None,
    ) -> keras.Model:
        """
        Train a Keras model.

        Parameters
        ----------
        model : keras.Model
            Uncompiled model to compile and train.

        data : tuple(X, y) or iterable
            If the entire dataset can fit in memory you can provide it as a tuple.

                X : numpy.ndarray(float, ndim=3 or 4)
                    Channels-last formatted training instances.  For 2D images, this should have a shape of (N, D1, D2, C) where N is the number of instances and each image is D1xD2 with C channels.  For 1D series, the expected format is (N, D1, C).

                y : numpy.ndarray(int, ndim=1 or 2)
                    Integer encoded class for each training instance. If a 1 dimensional input is provided, it is assumed the classes are uniquely encoded as integers and you should use a 'sparse_categorical_crossentropy' compiler loss and 'sparse_categorical_accuracy' metric. This works for binary as well as multiclass datasets. If a 2 dimensional input is provided, it is assumed each class is one-hot encoded and you should use a 'categorical_crossentropy' compiler loss and 'categorical_accuracy' metric.

            Otherwise a data iterator may be provided consistent with Keras' API.  For example, a `NNTools.XLoader`.

        compiler_kwargs : dict(str, object), optional(default={'optimizer': 'adam', 'loss': 'sparse_categorical_crossentropy', 'weighted_metrics': ['accuracy', 'sparse_categorical_accuracy'],})
            Arguments to use when compiling the `model`. See https://keras.io/api/models/model_training_apis/#compile-method for details.  If None, will assume the model is already compiled.

        fit_kwargs : dict(str, object), optional(default={'batch_size': 100, 'epochs': 100, 'validation_split': 0.0, 'validation_data': None, 'shuffle': True, 'callbacks': [keras.callbacks.ModelCheckpoint(filepath='./checkpoints/model.{epoch:05d}', save_weights_only=True, monitor='val_sparse_categorical_accuracy', save_freq='epoch', mode='max', save_best_only=False), keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=0, mode="auto", min_delta=0.0001, cooldown=0, min_lr=1.0e-6)]})
            Arguments to use when fitting the `model`. See https://keras.io/api/models/model_training_apis/#fit-method for details. A large `batch_size` can lead to memory overflow if you have large images or other data.  Consider reducing this if you run in issues, or using a data iterator instead; in this case `batch_size` will be ignored.  Also, `validation_split` is invalid when using data iterators; in that case, specify `validation_data` with a test or validation set iterator of its own.

        class_weight : dict(int, float), optional(default=None)
            Dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). By default, use class weighting inversely proportional to observation frequency as in sklearn's RandomForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html. To turn off weighting, specify `class_weight={c:1 for c in range(N)}` where `N` is the number of classes in the model. Turning off weighting means the 'weighted_metrics' simply become equivalent to 'metrics'.  Normally, this weighting is part of `fit_kwargs` but we have kept it separate here because the default behavior used is different from Keras.  This will be added to `fit_kwargs` before the model is fit, and will overwrite any class_weight previously specified in `fit_kwargs`.

        seed : int, optional(default=42)
            Seed for keras random weight initialization when the model is compiled.

        wandb_project : str, optional(default=None)
            Project name for WandB; if None, do not use WandB to track the run.  Visit https://docs.wandb.ai/ to learn about this and set up a free account.  If you wish to store things locally and monitor them with tensorboard instead, provide a callback if `fit_kwargs` as illustrated below.

        wandb_kwargs : dict(str, object), optional(default=None)
            Any additional parameters and their values you may want to record in WandB; for example, parameters used to create the model.

        model_filename : str, optional(default='trained_model.keras')
            Name of the file to save the model to when finished training. If None the model will not be saved.

        history_filename : str, optional(default='training_history.pkl')
            Name of the file to save the training history to when finished training. If None the history will not be saved.

        restart : None or dict, optional(default=None)
            If None, start training from scratch. Otherwise should be a dictionary of the following form: {"from_wandb": bool, "filepath": str, "weights_only": bool}.  "from_wandb" refers to whether or not to load from a W&B project online, "filepath" is either the local checkpoint / model or the path to the W&B artifact, and "weights_only" specifies if the saved checkpoint / model contains only weights or not. By default, checkpoints only save weights to W&B while the complete model is saved once at the end of the run.  For example, to restart from an old W&B checkpoint, use {"from_wandb": True, "filepath": "user-name/project-name/artifact-name:latest", "weights_only": True}; you can also specify the model name saved at the end of a run (`model_filename`) and set `weights_only=False`.  To load a local restart file set the `filepath` to the local filename instead, e.g., "/path/to/my/model_filename".

        Returns
        -------
        model : keras.Model
            The fitted model at the end of training.

        Notes
        -----
        The Keras seed and other factors are set at the begnning of this function in the interest of reproducibility.  The produces identical initial layer weights, etc. between different runs with the same seed.

        With the default parameters above (1) model checkpoints are written locally to 'checkpoints/model.{epoch:05d}', (2) the final model is saved locally to 'model_filename', and (3) the model history is saved locally to 'history_filename'.  The `save_best_only=False` option in the keras.callbacks.ModelCheckpoint will save all checkpoints, which could take up disk space; consider `save_best_only=True` if that is an issue. Also, the `save_weights_only=True` option helps save disk space by only saving weights; to load a new model from these weights use `NNTools.load(filepath=filepath, weights_only=True, model=same_model)` where `filepath` refers to weights you wish to load.

        If you use wandb to track the run a "wandb" folder will also be created locally.

        Examples
        --------
        To train a model use cyclical learning rates:
        >>> finder = NNTools.find_learning_rate(
        ...     model=CNNFactory(...), # Instantiate a new model
        ...     (X_train, y_train),
        ...     compiler_kwargs=compiler_kwargs
        ... )
        >>> clr = NNTools.CyclicalLearningRate(
        ...     base_lr=finder.estimate_clr()[0],
        ...     max_lr=finder.estimate_clr()[1],
        ...     step_size=20,
        ... )
        >>> model = NNTools.train(
        ...     model=CNNFactory(...), # Instantiate a new model, do not recycle the old one!
        ...     data=(X_train, y_train),
        ...     fit_kwargs={
        ...         'batch_size': 50,
        ...         'epochs': 100,
        ...         'validation_split': 0.2,
        ...         'shuffle': True,
        ...         'callbacks': [clr]
        ...     },
        ...     compiler_kwargs=compiler_kwargs, # Should be the same as finder
        ...     model_filename=None,
        ...     history_filename=None
        ... )

        To use tensorboard to view the history of a run instead of WandB, you can use a callback like this:
        >>> tb_callback = keras.callbacks.TensorBoard(
        ...                log_dir='path/to/my/logs/',
        ...                histogram_freq=1,
        ...                write_graph=True,
        ...                write_steps_per_second=False,
        ...                update_freq="batch",
        ...                profile_batch=5
        ...            )
        >>> fit_kwargs = {'callbacks': [tb_callback, ...], ...}
        >>> NNTools.train(model, X, y, fit_kwargs=fit_kwargs, ....)
        Then, from the command line, execute:
        $ tensorboard --logdir=path/to/my/logs/ --bind_all
        """
        # To ensure as much reproducibility as possible: https://keras.io/examples/keras_recipes/reproducibility_recipes/
        keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()

        if NNTools._is_data_iter(data):
            _, _, unique_targets, X, y = NNTools._summarize_batches(data)
            N_data = len(unique_targets)
            vals = np.array(sorted(unique_targets.keys()))
            counts = np.array([unique_targets[k] for k in vals])

            # Overwrite this to ensure batch size is determined by the data iterator.  Keras knows to ignore this, but this is good to alert the user.
            fit_kwargs["batch_size"] = None
        else:
            X, y = data
            N_data = len(np.unique(y))

            # Check length consistency between X and y
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "X and y should have the same first dimension corresponding to the number of datapoints"
                )

        # Check n_classes is the same as in model
        N_model = model.predict(X[:1]).shape[1]
        if N_data != N_model:
            raise Exception(
                f"Model is configured to predict {N_model} classes, while the training set contains {N_data} unique classes"
            )

        # Check y is encoded as integers and other consistencies
        if compiler_kwargs is not None:
            if compiler_kwargs.get("loss") == "sparse_categorical_crossentropy":
                # Classes are unique integer values
                if type(y[0].item()) != int:
                    raise ValueError(
                        "A sparse_categorical_crossentropy loss is used when y is not one-hot encoded, but y should be provided as integers"
                    )
                if not NNTools._is_data_iter(data):
                    vals, counts = np.unique(y, return_counts=True)
            elif compiler_kwargs.get("loss") == "categorical_crossentropy":
                # OHE of y
                if y.shape[1] != N_model:
                    raise Exception(
                        f"A categorical_crossentropy loss is used for one-hot-encoded y; y contains {y.shape[1]} classes but model is expecting {N_model}"
                    )
                if type(y[0][0].item()) != int:
                    raise ValueError("y should be encoded as integers")
                if not NNTools._is_data_iter(data):
                    vals, counts = np.arange(y.shape[1]), np.sum(y, axis=0)
            else:
                raise ValueError(
                    'compiler loss should be "sparse_categorical_crossentropy" or "categorical_crossentropy"'
                )

            # Bad compiler kwargs should cause this to fail
            model.compile(**compiler_kwargs)

        # Class weights following: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        if class_weight is None:
            class_weight = dict(
                zip(vals, (1.0 / counts) / (len(vals) / np.sum(counts)))
            )

        # Add class_weight to fit_kwargs
        fit_kwargs["class_weight"] = class_weight

        # Update callbacks with W&B logger, if used
        if wandb_project is not None:
            from wandb.integration.keras import (
                WandbMetricsLogger,
                WandbModelCheckpoint,
            )

            wandb.login()
            _ = wandb.init(
                project=str(wandb_project),
                config={
                    "compiler_kwargs": {}
                    if compiler_kwargs is None
                    else NNTools._json_serializable(compiler_kwargs),
                    "fit_kwargs": NNTools._json_serializable(fit_kwargs),
                    "wandb_kwargs": {}
                    if wandb_kwargs is None
                    else NNTools._json_serializable(wandb_kwargs),
                    "seed": seed,
                    "model_filename": ""
                    if model_filename is None
                    else model_filename,
                    "history_filename": ""
                    if history_filename is None
                    else history_filename,
                    "restart": ""
                    if restart is None
                    else NNTools._json_serializable(restart),
                },
            )

            logger = WandbMetricsLogger(log_freq="batch")  # vs. 'epoch' default
            chkpt = WandbModelCheckpoint(  # Opt to save as much detail as possible to W&B
                filepath="checkpoints/",
                save_weights_only=True,
                save_freq="epoch",
                save_best_only=False,
            )
            if "callbacks" in fit_kwargs:
                fit_kwargs["callbacks"].append([logger, chkpt])
            else:
                fit_kwargs["callbacks"] = [logger, chkpt]

        # Restarting
        if restart is not None:
            try:
                if restart["from_wandb"]:
                    api = wandb.Api()
                    artifact = api.artifact(restart["filepath"])
                    name_ = str(datetime.datetime.now()).replace(" ", "-")
                    checkpoint = str(
                        os.path.join(
                            os.path.abspath(os.getcwd()),
                            f"restart-{name_}/",  # The trailing "/" is critical here for keras to understand
                        )
                    )
                    if os.path.isdir(
                        checkpoint
                    ):  # Remove any existing directory
                        shutil.rmtree(checkpoint)
                    artifact.download(checkpoint)

                    model = NNTools.load(
                        filepath=checkpoint,
                        weights_only=restart["weights_only"],
                        model=model,
                    )
                    # Do not cleanup the restart file - it is necessary
                else:
                    model = NNTools.load(
                        filepath=restart["filepath"],
                        weights_only=restart["weights_only"],
                        model=model,
                    )
            except Exception as e:
                raise Exception(f"Unable to restart : {e}")

        # Fit model with incremental saving / checkpointing
        if NNTools._is_data_iter(data):
            _ = model.fit(x=data, **fit_kwargs)
        else:
            _ = model.fit(x=X, y=y, **fit_kwargs)

        # Write the final model and full training history
        if model_filename is not None:
            model.save(model_filename, overwrite=True)
            if wandb_project is not None:
                wandb.save(model_filename)  # Also save to W&B
        if history_filename is not None:
            with open(history_filename, "wb") as f:
                pickle.dump(model.history.history, f)
            if wandb_project is not None:
                wandb.save(history_filename)  # Also save to W&B

        return model


class HuggingFace:
    """Tools to help store and load models on Hugging Face Hub."""

    @staticmethod
    def from_pretrained(
        model_id: str,
        filename: str = "model.pkl",
        revision: Union[str, None] = None,
        token: Union[str, None] = None,
        library_version: Union[str, None] = None,
    ) -> Any:
        """
        Load a pre-trained model from Hugging Face.

        Parameters
        ----------
        model_id : str
            Model ID, for example "hf-user/my-awesome-model"

        filename : str, optional(default="model.pkl")
            The name of the model file in the repo, e.g., "model.pkl". This is the default name
            used when pushing to Hugging Face hub (`push_to_hub`), but if you change it or
            use another repo with a different name, change it to match here.

        revision : str, optional(default=None)
            Model revision; if None, the latest version is retrieved.

        token : str, optional(default=None)
            Your Hugging Face access token. Refer to https://huggingface.co/settings/tokens.
            Ungated, public models do not require this to be specified.

        library_version : str, optional(default=None)
            The version of the PyChemAuth library to use; if None, the latest version is used.

        Returns
        -------
        model : estimator
            Model, or pipeline, from PyChemAuth that is compatible with sklearn's estimator API.
        """
        filename = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            token=token,
            library_version=library_version,
            library_name="PyChemAuth",
        )
        return joblib.load(filename)

    @staticmethod
    def push_to_hub(
        model: Any,
        namespace: str,
        repo_name: str,
        token: str,
        revision: Union[str, None] = None,
        private: Union[bool, None] = True,
    ) -> None:
        """
        Push a PyChemAuth model, or pipeline, to Hugging Face.

        If no repo (namespace/repo_name) exists on Hugging Face this creates a minimal model card and repo to hold a PyChemAuth model or pipeline. By default, all new repos are set to private.

        It is strongly recommended that you visit the link below for instructions on how to fill out an effective model card that accurately and completely describes your model.

        https://huggingface.co/docs/hub/model-card-annotated

        Parameters
        ----------
        model : estimator
            Model, or pipeline, from PyChemAuth that is compatible with sklearn's estimator API.

        namespace : str
            User or organization name on Hugging Face.

        repo_name : str
            Name of Hugging Face repository, e.g., "my-awesome-model".

        token : str
            Your Hugging Face access token.  Be sure it has write access. Refer to https://huggingface.co/settings/tokens.

        revision : str, optional(default=None)
            The git revision to commit from. Defaults to the head of the `"main"` branch.

        private : bool, optional(default=True)
            If a new repo is created, this indicates if it should be private.

        Notes
        -----
        All models are serialized using pickle(protocol=4).
        """

        def _check_model_type(model):
            """Determine if the model is a regressor or classifier."""
            from pychemauth.classifier import osr, plsda, simca
            from pychemauth.manifold import elliptic

            # First check if this is a NN model
            if isinstance(model, keras.Model):

                def _is_classifier(model):
                    # If final layer is explicitly as activation then assume only linear is compatible with regression.
                    # This logic is valid if last layer is explicity a keras.layer.Activation or keras.layer.Dense
                    if model.layers[-1].activation == keras.activations.linear:
                        return False
                    else:
                        return True

                classifier = _is_classifier(model)
                if len(model.input_shape) == 3:  # (index, D1, 1)
                    # Effectively tabular
                    return f"tabular-{'classification' if _is_classifier(model) else 'regression'}"
                elif (
                    len(model.input_shape) == 4
                ):  # (index, D1, D2, C) where C is channels
                    # Effectively working with images (2D) data
                    if _is_classifier(model):
                        return "image-classification"
                    else:
                        # Could be be image detection, segmentation, etc. and there is not a simple way to automatically tag this for now
                        return "other"
            else:
                _type = type(model)
                if (_type is sklearn.pipeline.Pipeline) or (
                    _type is imblearn.pipeline.Pipeline
                ):
                    _type = type(model.steps[-1][1])

                if _type in [
                    pychemauth.classifier.osr.OpenSetClassifier,
                    pychemauth.classifier.plsda.PLSDA,
                    pychemauth.classifier.simca.SIMCA_Authenticator,
                    pychemauth.classifier.simca.SIMCA_Model,
                    pychemauth.classifier.simca.DDSIMCA_Model,
                    pychemauth.manifold.elliptic.EllipticManifold_Authenticator,
                    pychemauth.manifold.elliptic.EllipticManifold_Model,
                ]:
                    # Tag as classifier
                    return "tabular-classification"
                elif _type in [
                    pychemauth.regressor.pcr.PCR,
                    pychemauth.regressor.pls.PLS,
                ]:
                    # Tag as regressor
                    return "tabular-regression"
                else:
                    # No tags - e.g., PCA.
                    return "other"

        # Save all files in a temporary directory and push them in a single commit
        try:
            repo_id = f"{namespace}/{repo_name}"

            # Create repo
            api = HfApi()

            def _create_repo(exist_ok=False):
                return api.create_repo(
                    repo_id=repo_id,
                    token=token,
                    private=private,
                    repo_type="model",
                    exist_ok=exist_ok,
                )

            try:
                _ = _create_repo(exist_ok=False)
                _new_repo = True
            except:
                _ = _create_repo(exist_ok=True)
                _new_repo = False

            with TemporaryDirectory() as tmpdir_:
                tmpdir = Path(tmpdir_)

                # Serialize the model
                with open(os.path.join(tmpdir, "model.pkl"), mode="bw") as f:
                    pickle.dump(model, file=f, protocol=4)

                # Create the model card for new repos only - otherwise this will overwrite
                if _new_repo:
                    card_data = ModelCardData(
                        library_name="PyChemAuth",
                        license="other",
                        license_name="nist",
                        license_link="https://github.com/mahynski/pychemauth/blob/main/LICENSE.md",
                        pipeline_tag=_check_model_type(model),
                        tags=["PyChemAuth"],
                    )
                    content = f"""
---
{ card_data.to_yaml() }
---

# Model Card

This is a default card created by PyChemAuth.

Refer to [this link](https://huggingface.co/docs/hub/model-card-annotated) for best practices on filling this out.
"""
                    card = ModelCard(content)
                    card.validate()
                    (tmpdir / "README.md").write_text(card.content)

                return api.upload_folder(
                    repo_id=repo_id,
                    folder_path=tmpdir,
                    token=token,
                    commit_message="Pushing model on {}".format(
                        datetime.datetime.now()
                    ),
                    revision=revision,
                    repo_type="model",
                )

        except Exception as e:
            raise Exception(
                "Unable to create temporary directory and save model information : {}".format(
                    e
                )
            )

        return


class ControlBoundary:
    """Base class for plotting statistical control boundaries."""

    def __init__(self):
        """Initialize class."""
        self.boundary_ = None

    def set_params(self, **parameters: Any) -> "ControlBoundary":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        """Plot the control boundary."""
        raise NotImplementedError

    @property
    def boundary(self):
        """Return the boundary."""
        return copy.deepcopy(self.boundary_)


def _adjusted_covariance(
    X: NDArray, method: str, center: Union[Sequence, NDArray, None], dim: int
) -> tuple[NDArray, NDArray]:
    """Compute the covariance of data around a fixed center."""
    if center is None:
        # Not forcing the center, leave
        adjust = np.array([0.0 for i in range(dim)])
    else:
        adjust = check_array(
            center,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=False,
            force_all_finite=True,
            copy=True,
        )
    if adjust.shape != (dim,):
        raise Exception("Invalid center.")

    X = X[:, :dim] - adjust
    if method.lower() == "empirical":
        cov = EmpiricalCovariance(
            assume_centered=False if center is None else True
        ).fit(X)
    elif method.lower() == "mcd":
        cov = MinCovDet(
            assume_centered=False if center is None else True, random_state=42
        ).fit(X)
    else:
        raise ValueError("Unrecognized method for determining the covariance.")

    return cov.covariance_, cov.location_ + adjust


class CovarianceEllipse(ControlBoundary):
    """Draw chi-squared limits of a two dimensional distribution as an ellipse."""

    method: ClassVar[str]
    center: ClassVar[str]

    def __init__(
        self,
        method: str = "empirical",
        center: Union[Sequence, NDArray, None] = None,
    ):
        """
        Instantiate the class.

        Parameters
        ----------
        method : str, optional(default='empirical')
            How to compute the covariance matrix.  The default 'empirical' uses the empirical covariance, if 'mcd' the minimum covariance determinant is computed.

        center : array_like(float, ndim=1), optional(default=None)
            Shifts the training data to make this the center.  If `None`, no shifting is done, and the data is not assumed to be centered when the ellipse is calculated.
        """
        super(CovarianceEllipse, self).__init__()
        self.set_params(**{"method": method, "center": center})

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {"method": self.method, "center": self.center}

    @property
    def S(self) -> NDArray:
        """Return the covariance matrix."""
        return self.__S_.copy()

    def fit(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> "CovarianceEllipse":
        """
        Fit the covariance ellipse to the data.

        Only the first 2 dimensions, or columns, of the data will be used.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix with at least 2 features (columns).

        Returns
        -------
        self

        Raises
        ------
        Exception if X has less than 2 columns.
        ValueError if the covariance method is unrecognized.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        if X_.shape[1] < 2:
            raise Exception(
                "Can only draw 2D covariance ellipse if there are at least 2 features."
            )

        self.__S_, self.__class_center_ = _adjusted_covariance(
            X_, self.method, self.center, dim=2
        )

        evals, evecs = np.linalg.eig(self.__S_)
        ordered = sorted(zip(evals, evecs.T), key=lambda x: x[0], reverse=True)
        self.__l1_, self.__l2_ = ordered[0][0], ordered[1][0]
        largest_evec = ordered[0][1]
        self.__angle_ = (
            np.arctan2(largest_evec[1], largest_evec[0]) * 180.0 / np.pi
        )

        return self

    def visualize(
        self,
        ax: matplotlib.axes.Axes,
        alpha: float = 0.05,
        ellipse_kwargs: dict = {"alpha": 0.3},
    ) -> matplotlib.axes.Axes:
        """
        Draw a covariance ellipse boundary at a certain threshold.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot the ellipse on.

        alpha : float, optional(default=0.05)
            Significance level (Type I error rate).

        ellipse_kwargs: dict, optional(default={'alpha':0.3})
            Dictionary of formatting arguments for the ellipse.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Ellipse.html.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object with ellipse plotted on it.
        """
        k = np.sqrt(
            -2 * np.log(alpha)
        )  # https://www.kalmanfilter.net/background2.html
        self.boundary_ = Ellipse(
            xy=self.__class_center_,
            width=np.sqrt(self.__l1_) * k * 2,
            height=np.sqrt(self.__l2_) * k * 2,
            angle=self.__angle_,
            **ellipse_kwargs,
        )
        ax.add_artist(self.boundary_)

        return ax


class OneDimLimits(ControlBoundary):
    """Draw chi-squared limits of a one dimensional distribution as a rectangle."""

    method: ClassVar[str]
    center: ClassVar[str]

    def __init__(
        self,
        method: str = "empirical",
        center: Union[Sequence, NDArray, None] = None,
    ):
        """
        Instantiate the class.

        Parameters
        ----------
        method : str, optional(default='empirical')
            How to compute the covariance matrix.  The default 'empirical' uses the empirical covariance, if 'mcd' the minimum covariance determinant is computed.

        center : array_like(float, ndim=1), optional(default=None)
            Shifts the training data to make this the center.  If None, no shifting is done, and the data is not assumed to be centered when the ellipse is calculated.
        """
        super(OneDimLimits, self).__init__()
        self.set_params(**{"method": method, "center": center})

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {"method": self.method, "center": self.center}

    @property
    def S(self) -> NDArray:
        """Return the covariance matrix."""
        return self.__S_.copy()

    def fit(
        self, X: Union[Sequence[Sequence[float]], NDArray[np.floating]]
    ) -> "OneDimLimits":
        """
        Fit the covariance to the data.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix with a single feature (column).

        Returns
        -------
        self

        Raises
        ------
        Exception if X has more than 1 column.
        ValueError if the covariance method is unrecognized.
        """
        X_ = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        if X_.shape[1] != 1:
            raise Exception(
                "Can only draw one dimensional boundary if there is a single feature."
            )

        self.__S_, self.__class_center_ = _adjusted_covariance(
            X_, self.method, self.center, dim=1
        )

        return self

    def visualize(
        self,
        ax: matplotlib.axes.Axes,
        x: float,
        alpha: float = 0.05,
        rectangle_kwargs: dict = {"alpha": 0.3},
        vertical: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Draw a covariance boundary as a rectangle at a certain threshold.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot the ellipse on.

        x : float
            `X` coordinate to center the covariance "bar" on. If `vertical` is True, this is a `y` coordinate instead.

        alpha : float, optional(default=0.05)
            Significance level (Type I error rate).

        rectangle_kwargs: dict, optional(default={'alpha':0.3})
            Dictionary of formatting arguments for the rectangle_kwargs.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html.

        vertical : bool, optional(default=True)
            Whether or not to plot the boundary vertically (True) or horizontally (False).

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object with rectangle plotted on it.
        """
        d_crit = scipy.stats.chi2.ppf(1.0 - alpha, 1)

        if vertical:
            self.boundary_ = Rectangle(
                xy=[
                    x,
                    self.__class_center_[0] - np.sqrt(d_crit * self.__S_[0][0]),
                ],
                width=0.6,
                height=2 * np.sqrt(d_crit * self.__S_[0][0]),
                **rectangle_kwargs,
            )
        else:
            dy = 2.0 / 3.0
            self.boundary_ = Rectangle(
                xy=[
                    self.__class_center_[0] - np.sqrt(d_crit * self.__S_[0][0]),
                    x - 0.5 * dy,
                ],
                width=2 * np.sqrt(d_crit * self.__S_[0][0]),
                height=dy,
                **rectangle_kwargs,
            )
        ax.add_artist(self.boundary_)

        return ax


def estimate_dof(
    u_vals: Union[Sequence[float], NDArray[np.floating]],
    robust: bool = True,
    initial_guess: Union[float, None] = None,
) -> tuple[int, Any]:
    """
    Estimate the degrees of freedom for projection-based modeling.

    Parameters
    ----------
    u_vals : array_like(float, ndim=1)
        Observation values.

    robust : bool, optional(default=True)
        Whether to use a statistically robust approach or not.

    initial_guess : scalar(float), optional(default=None)
        Initial guess for the degrees of freedom.

    Returns
    -------
    Nu : scalar(int)
        Number of degrees of freedom.

    u0 : scalar(float)
        Associated scaling factor.

    References
    ----------
    [1] "Acceptance areas for multivariate classification derived by projection methods," Pomerantsev, Journal of Chemometrics 22 (2008) 601-609.

    [2] "Concept and role of extreme objects in PCA/SIMCA," Pomerantsev A., Rodionova, O., Journal of Chemometrics 28 (2014) 429-438.

    [3] "Detection of outliers in projection-based modeling," Rodionova, O., and Pomerantsev, A., Anal. Chem. 92 (2020) 2656-2664.
    """
    if not robust:
        # Eq. 12 in [2]
        u0 = np.mean(u_vals)
        Nu = int(
            np.max([round(2.0 * u0**2 / np.std(u_vals, ddof=1) ** 2, 0), 1])
        )
    else:

        def err2(N, vals):
            # Use a "robust" method for estimating DoF - solve Eq. 14 in [2].
            if N < 1:
                N = 1
            a = (
                (scipy.stats.chi2.ppf(0.75, N) - scipy.stats.chi2.ppf(0.25, N))
                * np.median(vals)
                / scipy.stats.chi2.ppf(0.5, N)
            )
            b = scipy.stats.iqr(vals, rng=(25, 75))

            return (a - b) ** 2

        def approximate(vals):
            # Eq. 16 in [2]
            a = 0.72414
            b = 2.68631
            c = 0.84332
            M = np.median(vals)
            S = scipy.stats.iqr(vals, rng=(25, 75))

            arg = b * M / S
            if arg < 1:
                return 1
            else:
                return int(
                    round(np.exp(((1.0 / a) * np.log(arg)) ** (1.0 / c)), 0)
                )

        def averaged_estimator(N, vals):
            # Eq. 17 in [2]
            M = np.median(vals)
            S = scipy.stats.iqr(vals, rng=(25, 75))

            return (
                0.5
                * N
                * (
                    M / scipy.stats.chi2.ppf(0.5, N)
                    + S
                    / (
                        scipy.stats.chi2.ppf(0.75, N)
                        - scipy.stats.chi2.ppf(0.25, N)
                    )
                )
            )

        res = scipy.optimize.minimize(
            err2,
            (1 if initial_guess is None else initial_guess),
            args=(u_vals),
            method="Nelder-Mead",
        )
        if res.success:
            # Direct method, if possible
            Nu = int(np.max([round(res.x[0], 0), 1]))
        else:
            # Else, use analytical approximation
            Nu = approximate(u_vals)

        u0 = averaged_estimator(Nu, u_vals)

    return Nu, u0


def pos_def_mat(
    S: Union[Sequence[Sequence[float]], NDArray[np.floating]],
    inner_max: int = 10,
    outer_max: int = 100,
) -> NDArray[np.floating]:
    """
    Create a positive definite approximation of a square, symmetric matrix.

    Parameters
    ----------
    S : array_like(float, ndim=2)
        2D square, symmetric matrix to make positive definite.

    inner_max : scalar(int), optional(default=10)
        Number of iterations at a fixed tolerance to try.

    outer_max : scalar(int), optional(default=100)
        Number of different tolerances to try.

    Returns
    -------
    recon : ndarray(float, ndim=2)
        Symmetric, positive definite matrix approximation of S.
    """
    S = np.asarray(S, dtype=np.float64)
    assert S.shape[0] == S.shape[1]  # Check square
    assert np.allclose(S, (S + S.T) / 2.0)  # Check symmetric

    for j in range(outer_max):
        min_ = np.min(np.abs(S)) / 1000.0  # Drop down by 3 orders of magnitude
        max_ = np.min(np.abs(S)) * 10.0  # Within one order of magnitude of min
        tol = min_ + j * (max_ - min_) / float(outer_max)

        recon = copy.copy(S)

        # Compute evecs, evals, set all evals to tol, reconstruct
        for i in range(inner_max):
            evals, evecs = np.linalg.eig(recon)
            if np.any(np.abs(evals) < tol):
                evals[np.abs(evals) < tol] = tol
                recon = np.matmul(
                    evecs, np.matmul(np.diag(evals), np.linalg.inv(evecs))
                )
            else:
                break

        safe = True
        try:
            # Try even if inner loop reached its limit
            np.linalg.cholesky(recon)
        except np.linalg.LinAlgError:
            safe = False

        if np.max(np.abs(S - recon)) > tol:
            # If the maximum difference is more than the eigenvalue
            # tolerance, reject this.
            safe = False

        if safe:
            return recon

    raise Exception("Unable to create a symmetric, positive definite matrix")


def pls_vip(pls: PLSRegression, mode: str = "weights") -> NDArray[np.floating]:
    """
    Compute the variable importance in projection (VIP) in a PLS(1) model.

    Parameters
    ----------
    pls : sklearn.cross_decomposition.PLSRegression
        Trained PLS model.

    mode : str, optional(default='weights')
        Whether to use the weights or the rotations to compute the VIP.

    Returns
    -------
    vip : ndarray(float, ndim=1)
        Variable importances.

    Note
    ----
    Often, both VIP and the PLS coefficients are used to remove features. [1]

    References
    ----------
    [1] Wold, S., Sjoestroem, M., & Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics. Chemometrics and Intelligent Laboratory Systems, 58(2), 109-130.

    [2] Chong, I.-G., Jun, C.-H. (2005). Performance of some variable selection methods when multicollinearity is present. Chemometrics and Intelligent Laboratory Systems, 78(1), 103-112.
    """
    t = pls.x_scores_
    q = pls.y_loadings_

    if mode == "weights":
        w = pls.x_weights_
    else:
        w = pls.x_rotations_
    w /= np.linalg.norm(w, axis=0)

    n, _ = w.shape
    s = np.diag(t.T @ t @ q.T @ q)

    return np.sqrt(n * (w**2 @ s) / np.sum(s))


def _logistic_proba(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the logistic function of a given input.

    This is designed to work on margin space "distances" from classifiers or authenticators to predict probabilities. See scikit-learn convention: https://scikit-learn.org/stable/glossary.html#term-predict_proba

    Parameters
    ----------
    x : ndarray(float, ndim=1)
        Array of distances.

    Returns
    -------
    probabilities : ndarray(float, ndim=2)
        2D array as logistic function of the the input, x. First column is NOT inlier, 1-p(x), second column is inlier probability, p(x).
    """
    p_inlier = p_inlier = 1.0 / (
        1.0 + np.exp(-np.clip(x, a_max=None, a_min=-500))
    )
    prob = np.zeros((p_inlier.shape[0], 2), dtype=np.float64)
    prob[:, 1] = p_inlier
    prob[:, 0] = 1.0 - p_inlier

    return prob


def _multi_cm_metrics(
    df: pd.DataFrame,
    Itot: pd.Series,
    trained_classes: Union[NDArray[np.integer], NDArray[np.str_]],
    use_classes: Union[NDArray[np.integer], NDArray[np.str_]],
    style: str,
    not_assigned: Union[int, str],
    actual: Union[NDArray[np.integer], NDArray[np.str_]],
) -> dict:
    """
    Compute metrics for a (possibly) multiclass, multilabel classifier / authenticator using the confusion matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Inputs (index) vs. predictions (columns); akin to a confusion matrix.  Should have a row for all `use_classes` and a column for all of these plus one for `not_assigned`; thus, the shape is (N, N+1).

    Itot : pandas.Series
        Number of each class asked to classify.  Should have a row for each class in `use_classes`; e.g., 0 for classes seen during training but not testing.

    trained_classes : ndarray(int or str)
        Classes seen during training.

    use_classes : ndarray(int or str)
        Classes to use when computing metrics; this includes all classes seen during testing and training excluding the "unknown" class.

    style : str
        Either "hard" or "soft' denoting whether a point can be assigned to one or multiple classes, respectively.
        This determines whether the metrics are multilabel (soft) or not (hard).

    not_assigned : int or str
        The designation for an "unknown" or unrecognized class.

    actual : ndarray(int or str)
        True target (y) values.

    Returns
    -------
    fom : dict
        Dictionary object with the following attributes.

        CM : pandas.DataFrame
            Inputs (index) vs. predictions (columns); akin to a confusion matrix.

        I : pandas.Series
            Number of each class asked to classify.  This is the same as the `Itot` input.

        CSNS : pandas.Series
            Class sensitivity.

        CSPS : pandas.Series
            Class specificity.

        CEFF : pandas.Series
            Class efficiency.

        TSNS : scalar(float)
            Total sensitivity.

        TSPS : scalar(float)
            Total specificity.

        TEFF : scalar(float)
            Total efficiency.

        ACC : scalar(float)
            Accuracy.

    Note
    ----
    When making predictions about extraneous classes (not in training set) class efficiency (CEFF) is given as simply class specificity (CSPS) since class sensitivity (CSNS) cannot be calculated.

    References
    ----------
    [1] "Multiclass partial least squares discriminant analysis: Taking the right way - A critical tutorial," Pomerantsev and Rodionova, Journal of Chemometrics (2018). https://doi.org/10.1002/cem.3030.
    """
    correct_ = 0.0
    for class_ in df.index:  # All input classes
        if class_ in trained_classes:  # Things to classifier knows about (TP)
            correct_ += df[class_][class_]
        else:
            # Consider an assignment as "unknown" a correct assignment (TN)
            correct_ += df[not_assigned][class_]
    ACC = correct_ / Itot.sum()

    # Class-wise FoM
    # Sensitivity is "true positive" rate and is only defined for trained/known classes.
    CSNS = pd.Series(
        [
            df[kk][kk] / Itot[kk] if Itot[kk] > 0 else np.nan
            for kk in trained_classes
        ],
        index=trained_classes,
    )

    # Specificity is the fraction of points that are NOT a given class that
    # are correctly predicted to be something besides the class. Thus,
    # specificity can only be computed for the columns that correspond to
    # known classes since we have only trained on them. These are "true
    # negatives". This is always >= 0.
    CSPS = pd.Series(
        [
            1.0
            - np.sum(df[kk][df.index != kk])  # Column sum
            / np.sum(Itot[Itot.index != kk])
            for kk in trained_classes
        ],
        index=trained_classes,
    )

    # If CSNS can't be calculated, using CSPS as efficiency;
    # Oliveri & Downey introduced this "efficiency" used in [1]
    CEFF = pd.Series(
        [
            np.sqrt(CSNS[c] * CSPS[c]) if not np.isnan(CSNS[c]) else CSPS[c]
            for c in trained_classes
        ],
        index=trained_classes,
    )

    # Total FoM
    if len(set(actual).intersection(set(trained_classes))) > 0:
        TSNS = np.sum([df[kk][kk] for kk in trained_classes]) / float(
            np.sum([Itot[kk] for kk in trained_classes])
        )  # Itot.sum()
    else:
        # No trained classes are being provided for testing
        TSNS = np.nan

    # If any untrained class is correctly predicted to be "NOT_ASSIGNED" it
    # won't contribute to df[use_classes].sum().sum().  Also, unseen
    # classes can't be assigned to so the diagonal components for those
    # entries is also 0 (df[k][k]).
    TSPS = 1.0 - (
        df[use_classes].sum().sum()
        - np.sum([df[kk][kk] for kk in trained_classes])
    ) / Itot.sum() / (
        1.0 if style.lower() == "hard" else len(trained_classes) - 1.0
    )
    # Soft models can assign a point to all categories which would make this
    # sum > 1, meaning TSPS < 0 would be possible.  By scaling by the total
    # number of classes, TSPS is always positive; TSPS = 0 means all points
    # assigned to all classes (trivial result) vs. TSPS = 1 means no mistakes.

    # Sometimes TEFF is reported as TSPS when TSNS cannot be evaluated (all
    # previously unseen samples).
    if not np.isnan(TSNS):
        TEFF = np.sqrt(TSPS * TSNS)
    else:
        TEFF = TSPS

    return dict(
        zip(
            ["CM", "I", "CSNS", "CSPS", "CEFF", "TSNS", "TSPS", "TEFF", "ACC"],
            (
                df[
                    [c for c in df.columns if c in trained_classes]
                    + [not_assigned]
                ][
                    [x in np.unique(actual) for x in df.index]
                ],  # Re-order for easy visualization
                Itot,
                CSNS,
                CSPS,
                CEFF,
                TSNS,
                TSPS,
                TEFF,
                ACC,
            ),
        )
    )


def _occ_cm_metrics(
    df: pd.DataFrame,
    Itot: pd.Series,
    target_class: Union[NDArray[np.integer], NDArray[np.str_]],
    trained_classes: Union[NDArray[np.integer], NDArray[np.str_]],
    not_assigned: Union[int, str],
    actual: Union[NDArray[np.integer], NDArray[np.str_]],
) -> dict:
    """
    Compute one-class classifier (OCC) metrics from the confusion matrix.

    OCCs are "hard" by definition and assign a point to one class ("inlier" vs. "outlier") and only one class since they are mutually exclusive.

    Parameters
    ----------
    df : pandas.DataFrame
        Inputs (index) vs. predictions (columns); akin to a confusion matrix.

    Itot : pandas.Series
        Number of each class asked to classify.

    target_class : scalar(int or str), optional(default=None)
        The class used to fit the model; the rest are used to test specificity.

    trained_classes : ndarray(int or str)
        Classes seen during training.

    not_assigned : int or str
        The designation for an "unknown" or unrecognized class.

    actual : ndarray(int or str)
        True target (y) values.

    Returns
    -------
    fom : dict
        Dictionary object with the following attributes.

        CM : pandas.DataFrame
            Inputs (index) vs. predictions (columns); akin to a confusion matrix.

        I : pandas.Series
            Number of each class asked to classify.  This is the same as the `Itot` input.

        CSPS : pandas.Series
            Class specificity.

        TSNS : scalar(float)
            Total sensitivity.  For OCC this is also the CSNS.

        TSPS : scalar(float)
            Total specificity.

        TEFF : scalar(float)
            Total efficiency.

        ACC : scalar(float)
            Accuracy.
    """
    alternatives = [class_ for class_ in df.index if class_ != target_class]

    correct_ = df[target_class][target_class]  # (TP)
    for class_ in alternatives:  # All "negative" classes
        # Number of times an observation NOT from target_class was correctly not assigned to target_class
        # Assigning to multiple alternatives does not influence this in the spirit of OCC
        correct_ += Itot[class_] - df[target_class][class_]  # (TN)
    ACC = correct_ / float(Itot.sum())

    CSPS = {}
    for class_ in alternatives:
        if np.sum(Itot[class_]) > 0:
            CSPS[class_] = 1.0 - df[class_][target_class] / np.sum(Itot[class_])
        else:
            CSPS[class_] = np.nan

    if np.all(actual == target_class):
        # Testing on nothing but the target class, can't evaluate TSPS
        TSPS = np.nan
    else:
        TSPS = 1.0 - (
            df[target_class].sum() - df[target_class][target_class]
        ) / (Itot.sum() - Itot[target_class])

    # TSNS = CSNS
    if target_class not in set(actual):
        # Testing on nothing but alternative classes, can't evaluate TSNS
        TSNS = np.nan
    else:
        TSNS = df[target_class][target_class] / Itot[target_class]

    if np.isnan(TSNS):
        TEFF = TSPS
    elif np.isnan(TSPS):
        TEFF = TSNS
    else:
        TEFF = np.sqrt(TSNS * TSPS)

    fom = dict(
        zip(
            ["CM", "I", "CSPS", "TSNS", "TSPS", "TEFF", "ACC"],
            (
                df[
                    [c for c in df.columns if c in trained_classes]
                    + [not_assigned]
                ][
                    [x in np.unique(actual) for x in df.index]
                ],  # Re-order for easy visualization
                Itot,
                CSPS,
                TSNS,
                TSPS,
                TEFF,
                ACC,
            ),
        )
    )

    return fom


def _occ_metrics(
    X: NDArray[np.floating],
    y: Union[NDArray[np.integer], NDArray[np.str_]],
    target_class: Union[int, str],
    predict_function: Callable[..., NDArray[np.bool_]],
) -> tuple[dict, list]:
    """
    Compute one-class classifier (OCC) metrics directly from data.

    OCCs are "hard" by definition and assign a point to one class ("inlier" vs. "outlier") and only one class since they are mutually exclusive.

    Parameters
    ----------
    X : ndarray(float, ndim=2)
        Input feature matrix.

    y : ndarray(str or int, ndim=1)
        Class labels or indices.

    target_class : int or str
        Target class being modeled by the OCC; should have the same type as `y`.

    predict_function : callable
        Should return a 1D numpy array of booleans corresponding to whether a point is an inlier.

    Returns
    -------
    fom : dict
        Dictionary object with the following attributes.

        CSPS : pandas.Series
            Class specificity.

        TSNS : scalar(float)
            Total sensitivity.  For OCC this is also the CSNS.

        TSPS : scalar(float)
            Total specificity.

        TEFF : scalar(float)
            Total efficiency.

        ACC : scalar(float)
            Accuracy.

    alternatives : list(str or int)
        Classes besides the target class present in `y`.
    """
    alternatives = [c for c in sorted(np.unique(y)) if c != target_class]

    CSPS = {}
    for class_ in alternatives:
        mask = y == class_
        CSPS[class_] = 1.0 - np.sum(predict_function(X[mask])) / np.sum(mask)

    mask = y != target_class
    if np.sum(mask) == 0:
        # Testing on nothing but the target class, can't evaluate TSPS
        TSPS = np.nan
    else:
        TSPS = 1.0 - np.sum(predict_function(X[mask])) / np.sum(mask)

    mask = y == target_class
    if np.sum(mask) == 0:
        # Testing on nothing but alternative classes, can't evaluate TSNS
        TSNS = np.nan
    else:
        TSNS = np.sum(predict_function(X[mask])) / np.sum(mask)  # TSNS = CSNS

    if np.isnan(TSNS):
        TEFF = TSPS
    elif np.isnan(TSPS):
        TEFF = TSNS
    else:
        TEFF = np.sqrt(TSNS * TSPS)

    # Compute accuracy
    y_in = y == target_class
    ACC = np.sum(predict_function(X) == y_in) / X.shape[0]

    return {
        "CSPS": CSPS,
        "TSNS": TSNS,
        "TSPS": TSPS,
        "TEFF": TEFF,
        "ACC": ACC,
    }, alternatives
