"""
Unittests for SNV transformation.

author: nam
"""
import copy
import os
import unittest

import numpy as np

from pychemauth.preprocessing.filter import SNV


class BenchmarkMdatools(unittest.TestCase):
    """Compare to calculations in mdatools 0.14.1."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        self.X = np.loadtxt(
            os.path.dirname(os.path.abspath(__file__)) + "/data/spectra.txt",
            skiprows=1,
            dtype=object,
        )
        self.X = np.array(self.X[:, 1:], dtype=np.float64)

    def test_spectra(self):
        """Test the transformation against mdatools 0.14.1."""
        correct = np.array(
            [
                -2.09954806e-02,
                2.90609704e-02,
                6.72692702e-02,
                1.07120938e-01,
                1.59224972e-01,
                1.81642363e-01,
                2.34442178e-01,
                2.45196960e-01,
                2.69920367e-01,
                2.52552356e-01,
                1.92085700e-01,
                7.60427334e-02,
                -1.78810337e-02,
                -9.58349773e-02,
                -1.79567214e-01,
                -2.39172427e-01,
                -2.96862586e-01,
                -3.38808222e-01,
                -3.91236954e-01,
                -3.93006225e-01,
                -3.82874333e-01,
                -4.24409127e-01,
                -4.42757858e-01,
                -4.19823601e-01,
                -4.22977806e-01,
                -3.93582729e-01,
                -3.88056242e-01,
                -3.88321301e-01,
                -3.85862876e-01,
                -3.90249608e-01,
                -3.95259229e-01,
                -3.49397341e-01,
                -2.93171635e-01,
                -2.29716436e-01,
                -1.53644413e-01,
                -8.11440649e-02,
                -2.06361927e-03,
                1.05656485e-01,
                2.13369963e-01,
                4.72021466e-01,
                6.15690239e-01,
                7.48809652e-01,
                8.13954604e-01,
                9.57338439e-01,
                1.15953894e00,
                1.34880454e00,
                1.55648513e00,
                1.72591105e00,
                1.86429851e00,
                1.98865109e00,
                2.09899528e00,
                2.23988756e00,
                2.36267628e00,
                2.54964250e00,
                2.73864304e00,
                2.84941132e00,
                2.91812132e00,
                2.90535872e00,
                2.78021759e00,
                2.52138717e00,
                2.26820252e00,
                1.97325114e00,
                1.67588773e00,
                1.43282171e00,
                1.16346181e00,
                9.88463029e-01,
                8.05691380e-01,
                6.73744855e-01,
                5.48219391e-01,
                4.73022065e-01,
                3.69728451e-01,
                2.91516074e-01,
                2.05921796e-01,
                1.32354584e-01,
                8.61017343e-02,
                4.92982489e-02,
                5.81247240e-02,
                7.82692316e-02,
                1.05722750e-01,
                1.57389436e-01,
                1.96121228e-01,
                2.15775376e-01,
                2.36350605e-01,
                1.87321259e-01,
                1.09016112e-01,
                1.28658465e-02,
                -1.02733146e-01,
                -1.91899098e-01,
                -2.73212669e-01,
                -3.20777562e-01,
                -3.10195069e-01,
                -3.24355863e-01,
                -3.27119106e-01,
                -3.27158865e-01,
                -3.30803431e-01,
                -3.63319582e-01,
                -3.76539415e-01,
                -4.01580893e-01,
                -4.11765797e-01,
                -4.54857815e-01,
                -5.06458237e-01,
                -5.22792517e-01,
                -5.79952557e-01,
                -6.19519286e-01,
                -6.62776966e-01,
                -6.95173840e-01,
                -7.56621215e-01,
                -7.73618143e-01,
                -8.08831273e-01,
                -8.38339001e-01,
                -8.50750403e-01,
                -8.64540114e-01,
                -8.75188872e-01,
                -8.86533410e-01,
                -8.97082771e-01,
                -9.14384517e-01,
                -9.12840547e-01,
                -9.30261570e-01,
                -9.24039303e-01,
                -9.29950126e-01,
                -9.30559762e-01,
                -9.28644708e-01,
                -9.29446513e-01,
                -9.32461563e-01,
                -9.32375418e-01,
                -9.34104930e-01,
                -9.31129639e-01,
                -9.30831448e-01,
                -9.40810931e-01,
                -9.34913361e-01,
                -9.41844662e-01,
                -9.41075990e-01,
                -9.42606708e-01,
                -9.31666385e-01,
                -9.42242251e-01,
                -9.36510344e-01,
                -9.39863344e-01,
                -9.25556767e-01,
                -9.40592257e-01,
                -9.34635049e-01,
                -9.35098903e-01,
                -9.34840470e-01,
                -9.31374819e-01,
                -9.23091716e-01,
                -9.34601916e-01,
                -9.28936274e-01,
                -9.35867575e-01,
                -9.53566910e-01,
                -9.48020544e-01,
                -9.34151316e-01,
            ]
        )
        np.testing.assert_almost_equal(
            SNV().fit_transform(self.X)[0], correct, decimal=6
        )

        correct = np.array(
            [
                -0.01783243,
                0.03634031,
                0.07247045,
                0.09904597,
                0.14100319,
                0.17317001,
                0.22767959,
                0.27206194,
                0.27339802,
                0.2623726,
                0.17700982,
                0.0923656,
                -0.02062808,
                -0.12214725,
                -0.20822859,
                -0.2485915,
                -0.2816453,
                -0.35444449,
                -0.38792493,
                -0.40842637,
                -0.42338142,
                -0.4308814,
                -0.42921973,
                -0.42779383,
                -0.41860973,
                -0.41143535,
                -0.41063819,
                -0.40870706,
                -0.38135684,
                -0.39910754,
                -0.41319807,
                -0.35817203,
                -0.3196392,
                -0.25043281,
                -0.16457602,
                -0.10704625,
                -0.02352478,
                0.07459245,
                0.17586462,
                0.45033239,
                0.59526835,
                0.71664899,
                0.76246847,
                0.92576142,
                1.13531171,
                1.33135531,
                1.52556883,
                1.68563948,
                1.82923936,
                1.97072848,
                2.07224765,
                2.22315665,
                2.38486651,
                2.55496333,
                2.76275091,
                2.87693471,
                2.95309093,
                2.94525413,
                2.82231286,
                2.5443421,
                2.28847831,
                1.97391709,
                1.68121584,
                1.42751896,
                1.17944707,
                1.00218709,
                0.80511056,
                0.68825461,
                0.53321389,
                0.47197904,
                0.3599846,
                0.3029376,
                0.21335328,
                0.14926664,
                0.07700637,
                0.07659095,
                0.05844729,
                0.08716726,
                0.10732065,
                0.16669174,
                0.20035182,
                0.21776569,
                0.25628729,
                0.1850824,
                0.12502644,
                0.02342868,
                -0.08977834,
                -0.17879006,
                -0.29469167,
                -0.3008668,
                -0.32843035,
                -0.33509949,
                -0.32212048,
                -0.33845651,
                -0.32924995,
                -0.33353886,
                -0.36758068,
                -0.42041736,
                -0.41883428,
                -0.45246068,
                -0.48394262,
                -0.54336985,
                -0.56643116,
                -0.62305151,
                -0.66237026,
                -0.70457448,
                -0.7440055,
                -0.75702941,
                -0.79169997,
                -0.83207411,
                -0.86426339,
                -0.86137792,
                -0.8904684,
                -0.86561069,
                -0.90675952,
                -0.8806331,
                -0.90465998,
                -0.91194664,
                -0.90293094,
                -0.92460004,
                -0.92419585,
                -0.93419957,
                -0.92162475,
                -0.93765764,
                -0.93970105,
                -0.92772129,
                -0.9324593,
                -0.92188298,
                -0.9315611,
                -0.94654983,
                -0.93687171,
                -0.93814042,
                -0.92346606,
                -0.93733204,
                -0.92069287,
                -0.93299822,
                -0.92310678,
                -0.91578644,
                -0.927115,
                -0.91812177,
                -0.93622052,
                -0.90766895,
                -0.92778866,
                -0.92591366,
                -0.91906488,
                -0.94048697,
                -0.901404,
                -0.92069287,
                -0.95342107,
                -0.92009781,
            ]
        )
        np.testing.assert_almost_equal(
            SNV().fit_transform(self.X)[20], correct, decimal=6
        )


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
