"""
Unittests for sklearn estimator consistency.

author: nam
"""
import unittest
import tempfile
import os
import pytest

import numpy as np

from pychemauth.datasets import load_pgaa
from pychemauth.utils import write_dataset, _sort_xdata, fastnumpyio


class CheckDatasetWriting(unittest.TestCase):
    """Check we can write datasets to disk correctly."""

    @classmethod
    def setUpClass(self):
        """Load the test data."""
        self.X, self.y = load_pgaa(return_X_y=True)

    def test_fresh_npy(self):
        """Test a fresh write to disk."""
        dir_ = tempfile.TemporaryDirectory()

        try:
            # Check we can write
            x_f_, y_ = write_dataset(
                dir_.name + "/dummy",
                self.X,
                self.y,
                fmt="npy",
                overwrite=False,
                augment=False,
            )
        except Exception as e:
            raise Exception(f"Failed to write dataset to disk : {e}")
        else:
            # Check what we wrote was the original data
            self.assertEqual(len(x_f_), 325)
            self.assertEqual(y_.split("/")[-1], "y.npy")

            for i in range(325):
                read_ = fastnumpyio.load(x_f_[i])
                np.testing.assert_allclose(read_, self.X[i])

            read_ = fastnumpyio.load(y_)
            np.testing.assert_equal(read_, self.y)

        dir_.cleanup()

    def test_overwrite_catch_npy(self):
        """Test overwrite caught."""
        dir_ = tempfile.TemporaryDirectory()

        with pytest.raises(
            Exception, match=f"{dir_.name}/dummy already exists."
        ):
            _ = write_dataset(
                dir_.name + "/dummy",
                self.X,
                self.y,
                fmt="npy",
                overwrite=False,
                augment=False,
            )
            _ = write_dataset(
                dir_.name + "/dummy",
                self.X,
                self.y,
                fmt="npy",
                overwrite=False,
                augment=False,
            )

        dir_.cleanup()

    def test_overwrite_success_npy(self):
        """Test overwrite allowed."""
        dir_ = tempfile.TemporaryDirectory()

        try:
            _ = write_dataset(
                dir_.name + "/dummy",
                self.X,
                self.y,
                fmt="npy",
                overwrite=False,
                augment=False,
            )
            _ = write_dataset(
                dir_.name + "/dummy",
                self.X,
                self.y,
                fmt="npy",
                overwrite=True,
                augment=False,
            )
        except Exception as e:
            raise Exception(f"Failed to overwrite successfully : {e}")

        dir_.cleanup()

    def test_augment_npy(self):
        """Test augmenting existing dataset."""
        dir_ = tempfile.TemporaryDirectory()

        try:
            x_f1_, y1_ = write_dataset(
                dir_.name + "/dummy",
                self.X,
                self.y,
                fmt="npy",
                overwrite=False,
                augment=False,
            )
            _, idx1 = _sort_xdata(dir_.name + "/dummy")
            x_f2_, y2_ = write_dataset(
                dir_.name + "/dummy",
                self.X,
                self.y,
                fmt="npy",
                overwrite=False,
                augment=True,
            )
            _, idx2 = _sort_xdata(dir_.name + "/dummy")
        except Exception as e:
            raise Exception(f"Dataset augmentation failed : {e}")

        self.assertEqual(idx1, 324)  # 0-324
        self.assertEqual(idx2, 649)  # 325-649
        self.assertEqual(len(x_f1_), 325)
        self.assertEqual(len(x_f2_), 325)
        self.assertEqual(y1_.split("/")[-1], "y.npy")
        self.assertEqual(y2_.split("/")[-1], "y.npy")

        # Check that x and y are copied in order
        for a_, b_ in zip(x_f1_, x_f2_):
            self.assertEqual(
                int(a_.split("x_")[1].split(".npy")[0]),
                int(b_.split("x_")[1].split(".npy")[0])
                - 325,  # Check that these are offset by 325
            )

        for i in range(325):
            np.testing.assert_allclose(np.load(x_f1_[i]), np.load(x_f2_[i]))
        y_ = fastnumpyio.load(y2_)
        np.testing.assert_equal(y_[:325], y_[325:])

        dir_.cleanup()
