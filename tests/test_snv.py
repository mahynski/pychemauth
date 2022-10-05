"""
Unittests for SNV transformation.

author: nam
"""
import copy
import os
import unittest

import numpy as np

from pychemauth.preprocessing.filter import SNV


class TestSNV(unittest.TestCase):
    """Test SNV transformation."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        self.X = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

        self.Xq = np.array([[1.0, 2.0, 3.0], [1.0, 0.0, 1.0]])

    def test_fit(self):
        """Fit the class and set up default Xref."""
        X = np.zeros((2, 3), dtype=np.float64)
        snv = SNV(q=50)
        try:
            _ = snv.fit(X)
        except:
            self.assertTrue(False)

    def test_transform(self):
        """Test SNV transformation."""
        snv = SNV(q=50, robust=False, detrend=False)
        X_t = snv.fit_transform(self.X)
        self.assertTrue(np.allclose(X_t[0, :], [-1, 0, 1]))
        self.assertTrue(np.allclose(X_t[1, :], [-1, 0, 1]))

    def test_detrend_transform(self):
        """Test SNV transformation with detrend."""
        snv = SNV(q=50, robust=False, detrend=True)
        X_t = snv.fit_transform(self.X)
        self.assertTrue(np.allclose(X_t, 0.0))

    def test_robust_transform(self):
        """Test Robust SNV transformation."""
        snv = SNV(q=50, robust=True, detrend=False)
        X_t = snv.fit_transform(self.X)
        self.assertTrue(np.allclose(X_t[0, :], [-1, 0, 1]))
        self.assertTrue(np.allclose(X_t[1, :], [-1, 0, 1]))

        snv = SNV(q=51, robust=True, detrend=False)
        X_t = snv.fit_transform(self.X)
        self.assertTrue(np.allclose(X_t[0, :], [-1.02, -0.02, 0.98]))
        self.assertTrue(np.allclose(X_t[1, :], [-1.02, -0.02, 0.98]))

        snv = SNV(q=49, robust=True, detrend=False)
        X_t = snv.fit_transform(self.X)
        self.assertTrue(np.allclose(X_t[0, :], [-0.98, 0.02, 1.02]))
        self.assertTrue(np.allclose(X_t[1, :], [-0.98, 0.02, 1.02]))

    def test_robust_detrend_transform(self):
        """Test Robust SNV transformation with detrend."""
        snv = SNV(q=50, robust=True, detrend=True)
        X_t = snv.fit_transform(self.X)
        self.assertTrue(np.allclose(X_t, 0.0))

        snv = SNV(q=51, robust=True, detrend=True)
        X_t = snv.fit_transform(self.X)
        self.assertTrue(np.allclose(X_t, 0.0))

        snv = SNV(q=49, robust=True, detrend=True)
        X_t = snv.fit_transform(self.X)
        self.assertTrue(np.allclose(X_t, 0.0))

        snv = SNV(q=50, robust=True, detrend=True)
        X_t = snv.fit_transform(self.Xq)
        self.assertTrue(np.allclose(X_t, 0.0))
