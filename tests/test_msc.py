"""
Unittests for MSC transformation.

author: nam
"""
import copy
import os
import unittest

import numpy as np

from pychemauth.preprocessing.filter import MSC


class TestMSC(unittest.TestCase):
    """Test MSC transformation."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        self.X = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

    def test_fit(self):
        """Fit the class and set up default Xref."""
        X = np.zeros((2, 3), dtype=np.float64)
        msc = MSC()
        try:
            _ = msc.fit(X)
        except:
            self.assertTrue(False)

        self.assertTrue(np.allclose(X, msc.Xref))

    def test_Xref(self):
        """Test Xref is set properly."""
        X = np.zeros((2, 3), dtype=np.float64)
        msc = MSC(Xref=X)
        self.assertTrue(np.allclose(X, msc.Xref))

    def test_transform(self):
        """Test MSC transformation."""
        msc = MSC(self.X[0, :])
        X_t = msc.fit_transform(self.X)
        self.assertTrue(np.allclose(X_t[0, :], msc.Xref))
        self.assertTrue(np.allclose(X_t[1, :], msc.Xref))

        msc = MSC(Xref=self.X[1, :])
        X_t = msc.fit_transform(self.X)
        self.assertTrue(np.allclose(X_t[0, :], msc.Xref))
        self.assertTrue(np.allclose(X_t[1, :], msc.Xref))

        msc = MSC()
        X_t = msc.fit_transform(self.X)
        self.assertTrue(np.allclose(np.mean(self.X, axis=0), msc.Xref))
        self.assertTrue(np.allclose(np.mean(self.X, axis=0), msc.Xref))
