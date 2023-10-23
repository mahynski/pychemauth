"""
Unittests for PLS.

author: nam
"""
import copy
import os
import unittest

import numpy as np
import pandas as pd

from pychemauth.regressor.pls import PLS


class BenchmarkMdatools_ScaledRobust(unittest.TestCase):
    """Compare to calculations in mdatools 0.14.1."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        raw = np.loadtxt(
            os.path.dirname(os.path.abspath(__file__)) + "/data/people.txt",
            skiprows=2,
            dtype=object,
        )
        self.X = np.array([row[1:] for row in raw], dtype=np.float64)
        self.y = np.array([row[3] for row in self.X], dtype=np.float64)
        self.X = np.hstack((self.X[:, :3], self.X[:, 4:]))

        mask = np.array([True] * len(self.X))
        for i in [4, 8, 12, 16, 20, 24, 28, 32]:
            mask[i - 1] = False

        self.Xc = self.X[mask]
        self.yc = self.y[mask]
        self.Xt = self.X[~mask]
        self.yt = self.y[~mask]

        self.model = PLS(
            n_components=7, alpha=0.05, gamma=0.01, scale_x=True, robust=True
        )
        self.model.fit(self.Xc, self.yc)

    def test_distances(self):
        """Test h, q, z distances."""
        h = np.array(
            [
                10.29014234,
                5.68396294,
                8.45943246,
                4.65207861,
                4.67722056,
                15.79446821,
                6.62819327,
                3.12982965,
                7.07121626,
                4.38433553,
                4.84622103,
                6.59608612,
                6.53752951,
                7.9270221,
                8.63861616,
                6.15785337,
                7.54084761,
                4.02231233,
                5.05095549,
                5.21724246,
                10.4166253,
                5.26796247,
                6.27498745,
                5.73485876,
            ]
        )
        q = np.array(
            [
                0.43389152,
                0.14470249,
                0.26963732,
                0.00874239,
                0.04608073,
                0.02633357,
                0.03248984,
                0.03455517,
                0.16067683,
                0.26912052,
                0.22612725,
                0.04602259,
                0.02272952,
                0.05832545,
                0.37249989,
                0.05211344,
                0.19858228,
                0.0941601,
                0.10476625,
                0.01081812,
                0.000926,
                0.22716434,
                0.10918501,
                0.65167524,
            ]
        )

        # mdatools 0.41.1 returns a different z and z0 values when scale_x = True; this could be a bug
        # in their code, but since this is a constant factor so it cancels out when normalized by z0.
        # see https://github.com/svkucheryavski/mdatools/blob/master/R/pls.R line 585
        z = np.array(
            [
                1.58228921e-01,
                3.94751369e-02,
                3.77222896e-02,
                3.30816568e-01,
                1.85407827e-02,
                5.24226344e-02,
                1.27932484e-01,
                4.12414812e-02,
                6.06064957e-01,
                3.88376416e-01,
                1.39654903e-01,
                8.83245922e-02,
                1.39894300e-01,
                5.92985925e-03,
                3.24497189e-03,
                5.64916449e-02,
                4.13179574e-02,
                5.40697478e-04,
                1.47805513e-01,
                5.04271503e-01,
                2.14004278e-04,
                5.00918846e-02,
                1.52019455e00,
                4.31411532e-02,
            ]
        )

        np.testing.assert_almost_equal(
            self.model._h_q(self.Xc)[0], h, decimal=6
        )
        np.testing.assert_almost_equal(
            self.model._h_q(self.Xc)[1], q, decimal=6
        )
        np.testing.assert_almost_equal(
            self.model._z(self.Xc, self.yc), z, decimal=6
        )

    def test_limits(self):
        """Test h, q, z, f limits."""
        np.testing.assert_almost_equal(
            6.433454627069881, self.model._PLS__h0_, decimal=6
        )

        # mdatools uses the approximate calculation and gets 0.1880321, but here we are able
        # to be more accurate with the numerical optimization which leads to differences
        np.testing.assert_almost_equal(
            0.15928871670572453, self.model._PLS__q0_, decimal=6
        )
        np.testing.assert_almost_equal(
            0.10543040976824493, self.model._PLS__z0_, decimal=6
        )

        np.testing.assert_equal(21, self.model._PLS__Nh_)
        np.testing.assert_equal(2, self.model._PLS__Nq_)
        np.testing.assert_equal(1, self.model._PLS__Nz_)

        np.testing.assert_equal(self.model._PLS__Nf_, self.model._PLS__f0_)
        np.testing.assert_equal(
            self.model._PLS__Nf_, self.model._PLS__Nh_ + self.model._PLS__Nq_
        )

    def test_predictions(self):
        """Test predicted y values."""
        preds = np.array(
            [
                [35.63128259],
                [42.2737451],
                [33.52252497],
                [43.01107763],
                [37.08153933],
                [42.34507716],
                [41.66132062],
                [35.8585947],
            ]
        ).ravel()
        np.testing.assert_almost_equal(
            preds, self.model.predict(self.Xt), decimal=6
        )

    def test_score(self):
        """Test R^2 calculation."""
        np.testing.assert_almost_equal(
            0.9873776585483119, self.model.score(self.Xc, self.yc), decimal=6
        )
        np.testing.assert_almost_equal(
            0.9051699790584818, self.model.score(self.Xt, self.yt), decimal=6
        )

    def test_categorize(self):
        """Test extreme/outlier detection."""
        ext = np.array(
            [
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )
        out = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )
        np.testing.assert_equal(
            ext, self.model.check_xy_outliers(self.Xc, self.yc)[0]
        )
        np.testing.assert_equal(
            out, self.model.check_xy_outliers(self.Xc, self.yc)[1]
        )

        ext = np.array([False, True, False, False, False, False, False, False])
        out = np.array([True, False, True, False, True, False, True, False])

        np.testing.assert_equal(
            ext, self.model.check_xy_outliers(self.Xt, self.yt)[0]
        )
        np.testing.assert_equal(
            out, self.model.check_xy_outliers(self.Xt, self.yt)[1]
        )


class BenchmarkMdatools_Scaled(unittest.TestCase):
    """Compare to calculations in mdatools 0.14.1."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        raw = np.loadtxt(
            os.path.dirname(os.path.abspath(__file__)) + "/data/people.txt",
            skiprows=2,
            dtype=object,
        )
        self.X = np.array([row[1:] for row in raw], dtype=np.float64)
        self.y = np.array([row[3] for row in self.X], dtype=np.float64)
        self.X = np.hstack((self.X[:, :3], self.X[:, 4:]))

        mask = np.array([True] * len(self.X))
        for i in [4, 8, 12, 16, 20, 24, 28, 32]:
            mask[i - 1] = False

        self.Xc = self.X[mask]
        self.yc = self.y[mask]
        self.Xt = self.X[~mask]
        self.yt = self.y[~mask]

        self.model = PLS(
            n_components=7, alpha=0.05, gamma=0.01, scale_x=True, robust=False
        )
        self.model.fit(self.Xc, self.yc)

    def test_distances(self):
        """Test h, q, z distances."""
        h = np.array(
            [
                10.29014234,
                5.68396294,
                8.45943246,
                4.65207861,
                4.67722056,
                15.79446821,
                6.62819327,
                3.12982965,
                7.07121626,
                4.38433553,
                4.84622103,
                6.59608612,
                6.53752951,
                7.9270221,
                8.63861616,
                6.15785337,
                7.54084761,
                4.02231233,
                5.05095549,
                5.21724246,
                10.4166253,
                5.26796247,
                6.27498745,
                5.73485876,
            ]
        )
        q = np.array(
            [
                0.43389152,
                0.14470249,
                0.26963732,
                0.00874239,
                0.04608073,
                0.02633357,
                0.03248984,
                0.03455517,
                0.16067683,
                0.26912052,
                0.22612725,
                0.04602259,
                0.02272952,
                0.05832545,
                0.37249989,
                0.05211344,
                0.19858228,
                0.0941601,
                0.10476625,
                0.01081812,
                0.000926,
                0.22716434,
                0.10918501,
                0.65167524,
            ]
        )

        # mdatools 0.41.1 returns a different z and z0 values when scale_x = True; this could be a bug
        # in their code, but since this is a constant factor so it cancels out when normalized by z0.
        # see https://github.com/svkucheryavski/mdatools/blob/master/R/pls.R line 585
        z = np.array(
            [
                1.58228921e-01,
                3.94751369e-02,
                3.77222896e-02,
                3.30816568e-01,
                1.85407827e-02,
                5.24226344e-02,
                1.27932484e-01,
                4.12414812e-02,
                6.06064957e-01,
                3.88376416e-01,
                1.39654903e-01,
                8.83245922e-02,
                1.39894300e-01,
                5.92985925e-03,
                3.24497189e-03,
                5.64916449e-02,
                4.13179574e-02,
                5.40697478e-04,
                1.47805513e-01,
                5.04271503e-01,
                2.14004278e-04,
                5.00918846e-02,
                1.52019455e00,
                4.31411532e-02,
            ]
        )

        np.testing.assert_almost_equal(
            self.model._h_q(self.Xc)[0], h, decimal=6
        )
        np.testing.assert_almost_equal(
            self.model._h_q(self.Xc)[1], q, decimal=6
        )
        np.testing.assert_almost_equal(
            self.model._z(self.Xc, self.yc), z, decimal=6
        )

    def test_limits(self):
        """Test h, q, z, f limits."""
        np.testing.assert_almost_equal(
            6.708333333333333, self.model._PLS__h0_, decimal=6
        )
        np.testing.assert_almost_equal(
            0.15005524333227455, self.model._PLS__q0_, decimal=6
        )
        np.testing.assert_almost_equal(
            0.1892474666263504, self.model._PLS__z0_, decimal=6
        )

        np.testing.assert_equal(13, self.model._PLS__Nh_)
        np.testing.assert_equal(2, self.model._PLS__Nq_)
        np.testing.assert_equal(1, self.model._PLS__Nz_)

        np.testing.assert_equal(self.model._PLS__Nf_, self.model._PLS__f0_)
        np.testing.assert_equal(
            self.model._PLS__Nf_, self.model._PLS__Nh_ + self.model._PLS__Nq_
        )

    def test_predictions(self):
        """Test predicted y values."""
        preds = np.array(
            [
                [35.63128259],
                [42.2737451],
                [33.52252497],
                [43.01107763],
                [37.08153933],
                [42.34507716],
                [41.66132062],
                [35.8585947],
            ]
        ).ravel()
        np.testing.assert_almost_equal(
            preds, self.model.predict(self.Xt), decimal=6
        )

    def test_score(self):
        """Test R^2 calculation."""
        np.testing.assert_almost_equal(
            0.9873776585483119, self.model.score(self.Xc, self.yc), decimal=6
        )
        np.testing.assert_almost_equal(
            0.9051699790584818, self.model.score(self.Xt, self.yt), decimal=6
        )

    def test_categorize(self):
        """Test extreme/outlier detection."""
        ext = np.array(
            [
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )
        out = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )
        np.testing.assert_equal(
            ext, self.model.check_xy_outliers(self.Xc, self.yc)[0]
        )
        np.testing.assert_equal(
            out, self.model.check_xy_outliers(self.Xc, self.yc)[1]
        )

        ext = np.array([False, True, False, False, False, False, False, False])
        out = np.array([True, False, True, False, True, False, True, False])

        np.testing.assert_equal(
            ext, self.model.check_xy_outliers(self.Xt, self.yt)[0]
        )
        np.testing.assert_equal(
            out, self.model.check_xy_outliers(self.Xt, self.yt)[1]
        )


class BenchmarkMdatools_UnscaledRobust(unittest.TestCase):
    """Compare to calculations in mdatools 0.14.1."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        raw = np.loadtxt(
            os.path.dirname(os.path.abspath(__file__)) + "/data/people.txt",
            skiprows=2,
            dtype=object,
        )
        self.X = np.array([row[1:] for row in raw], dtype=np.float64)
        self.y = np.array([row[3] for row in self.X], dtype=np.float64)
        self.X = np.hstack((self.X[:, :3], self.X[:, 4:]))

        mask = np.array([True] * len(self.X))
        for i in [4, 8, 12, 16, 20, 24, 28, 32]:
            mask[i - 1] = False

        self.Xc = self.X[mask]
        self.yc = self.y[mask]
        self.Xt = self.X[~mask]
        self.yt = self.y[~mask]

        self.model = PLS(
            n_components=7, alpha=0.05, gamma=0.01, scale_x=False, robust=True
        )
        self.model.fit(self.Xc, self.yc)

    def test_distances(self):
        """Test h, q, z distances."""
        h = np.array(
            [
                10.16012747,
                7.47244224,
                5.53044946,
                4.13920174,
                2.8048815,
                12.86596107,
                5.87311824,
                3.31835631,
                7.94855802,
                6.07635506,
                7.19903601,
                4.86277004,
                6.50457825,
                7.86213481,
                9.3495848,
                5.49094806,
                1.81682582,
                5.37668895,
                5.3674368,
                5.02186922,
                8.85073553,
                5.38273693,
                8.87643819,
                12.84876547,
            ]
        )
        q = np.array(
            [
                14.04170827,
                0.17045717,
                0.17624006,
                0.15934482,
                1.94461148,
                6.33353444,
                9.10658786,
                3.79674233,
                0.14996447,
                0.37547292,
                1.41237012,
                15.97249387,
                0.10225634,
                4.99075633,
                12.69703894,
                7.36851262,
                18.80177517,
                0.31126861,
                0.75446964,
                0.16304307,
                4.5955245,
                4.32964194,
                0.11064471,
                27.26946415,
            ]
        )
        z = np.array(
            [
                0.03041174,
                0.07500194,
                0.26865468,
                0.41675539,
                0.07220708,
                0.14245054,
                0.07646303,
                0.10221563,
                0.59254802,
                0.56562692,
                0.14034235,
                0.12690661,
                0.22535524,
                0.00598724,
                0.03662903,
                0.071516,
                0.14186018,
                0.05203803,
                0.05441322,
                0.50914342,
                0.00750257,
                0.03739828,
                1.72671652,
                0.0038618,
            ]
        )

        np.testing.assert_almost_equal(
            self.model._h_q(self.Xc)[0], h, decimal=6
        )
        np.testing.assert_almost_equal(
            self.model._h_q(self.Xc)[1], q, decimal=6
        )
        np.testing.assert_almost_equal(
            self.model._z(self.Xc, self.yc), z, decimal=6
        )

    def test_limits(self):
        """Test h, q, z, f limits."""
        np.testing.assert_almost_equal(
            6.220076403682505, self.model._PLS__h0_, decimal=6
        )
        np.testing.assert_almost_equal(
            6.276821593745483, self.model._PLS__q0_, decimal=6
        )
        np.testing.assert_almost_equal(
            0.17504517218907578, self.model._PLS__z0_, decimal=6
        )

        np.testing.assert_equal(16, self.model._PLS__Nh_)
        np.testing.assert_equal(1, self.model._PLS__Nq_)
        np.testing.assert_equal(1, self.model._PLS__Nz_)

        np.testing.assert_equal(self.model._PLS__Nf_, self.model._PLS__f0_)
        np.testing.assert_equal(
            self.model._PLS__Nf_, self.model._PLS__Nh_ + self.model._PLS__Nq_
        )

    def test_predictions(self):
        """Test predicted y values."""
        preds = np.array(
            [
                [35.46612406],
                [42.38736392],
                [33.73733706],
                [43.19013661],
                [37.14740648],
                [42.34026293],
                [41.65760507],
                [35.97989294],
            ]
        ).ravel()
        np.testing.assert_almost_equal(
            preds, self.model.predict(self.Xt), decimal=6
        )

    def test_score(self):
        """Test R^2 calculation."""
        np.testing.assert_almost_equal(
            0.9847652, self.model.score(self.Xc, self.yc), decimal=6
        )
        np.testing.assert_almost_equal(
            0.9141127, self.model.score(self.Xt, self.yt), decimal=6
        )

    def test_categorize(self):
        """Test extreme/outlier detection."""
        ext = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
            ]
        )
        out = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )
        np.testing.assert_equal(
            ext, self.model.check_xy_outliers(self.Xc, self.yc)[0]
        )
        np.testing.assert_equal(
            out, self.model.check_xy_outliers(self.Xc, self.yc)[1]
        )

        ext = np.array([False, False, False, False, False, True, True, True])
        out = np.array([False, False, True, False, True, False, False, False])

        np.testing.assert_equal(
            ext, self.model.check_xy_outliers(self.Xt, self.yt)[0]
        )
        np.testing.assert_equal(
            out, self.model.check_xy_outliers(self.Xt, self.yt)[1]
        )


class BenchmarkMdatools_Unscaled(unittest.TestCase):
    """Compare to calculations in mdatools 0.14.1."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        raw = np.loadtxt(
            os.path.dirname(os.path.abspath(__file__)) + "/data/people.txt",
            skiprows=2,
            dtype=object,
        )
        self.X = np.array([row[1:] for row in raw], dtype=np.float64)
        self.y = np.array([row[3] for row in self.X], dtype=np.float64)
        self.X = np.hstack((self.X[:, :3], self.X[:, 4:]))

        mask = np.array([True] * len(self.X))
        for i in [4, 8, 12, 16, 20, 24, 28, 32]:
            mask[i - 1] = False

        self.Xc = self.X[mask]
        self.yc = self.y[mask]
        self.Xt = self.X[~mask]
        self.yt = self.y[~mask]

        self.model = PLS(
            n_components=7, alpha=0.05, gamma=0.01, scale_x=False, robust=False
        )
        self.model.fit(self.Xc, self.yc)

    def test_distances(self):
        """Test h, q, z distances."""
        h = np.array(
            [
                10.16012747,
                7.47244224,
                5.53044946,
                4.13920174,
                2.8048815,
                12.86596107,
                5.87311824,
                3.31835631,
                7.94855802,
                6.07635506,
                7.19903601,
                4.86277004,
                6.50457825,
                7.86213481,
                9.3495848,
                5.49094806,
                1.81682582,
                5.37668895,
                5.3674368,
                5.02186922,
                8.85073553,
                5.38273693,
                8.87643819,
                12.84876547,
            ]
        )
        q = np.array(
            [
                14.04170827,
                0.17045717,
                0.17624006,
                0.15934482,
                1.94461148,
                6.33353444,
                9.10658786,
                3.79674233,
                0.14996447,
                0.37547292,
                1.41237012,
                15.97249387,
                0.10225634,
                4.99075633,
                12.69703894,
                7.36851262,
                18.80177517,
                0.31126861,
                0.75446964,
                0.16304307,
                4.5955245,
                4.32964194,
                0.11064471,
                27.26946415,
            ]
        )
        z = np.array(
            [
                0.03041174,
                0.07500194,
                0.26865468,
                0.41675539,
                0.07220708,
                0.14245054,
                0.07646303,
                0.10221563,
                0.59254802,
                0.56562692,
                0.14034235,
                0.12690661,
                0.22535524,
                0.00598724,
                0.03662903,
                0.071516,
                0.14186018,
                0.05203803,
                0.05441322,
                0.50914342,
                0.00750257,
                0.03739828,
                1.72671652,
                0.0038618,
            ]
        )

        np.testing.assert_almost_equal(
            self.model._h_q(self.Xc)[0], h, decimal=6
        )
        np.testing.assert_almost_equal(
            self.model._h_q(self.Xc)[1], q, decimal=6
        )
        np.testing.assert_almost_equal(
            self.model._z(self.Xc, self.yc), z, decimal=6
        )

    def test_limits(self):
        """Test h, q, z, f limits."""
        np.testing.assert_almost_equal(
            6.708333333333333, self.model._PLS__h0_, decimal=6
        )
        np.testing.assert_almost_equal(
            5.630580159752494, self.model._PLS__q0_, decimal=6
        )
        np.testing.assert_almost_equal(
            0.22841689421337583, self.model._PLS__z0_, decimal=6
        )

        np.testing.assert_equal(11, self.model._PLS__Nh_)
        np.testing.assert_equal(1, self.model._PLS__Nq_)
        np.testing.assert_equal(1, self.model._PLS__Nz_)

        np.testing.assert_equal(self.model._PLS__Nf_, self.model._PLS__f0_)
        np.testing.assert_equal(
            self.model._PLS__Nf_, self.model._PLS__Nh_ + self.model._PLS__Nq_
        )

    def test_predictions(self):
        """Test predicted y values."""
        preds = np.array(
            [
                [35.46612406],
                [42.38736392],
                [33.73733706],
                [43.19013661],
                [37.14740648],
                [42.34026293],
                [41.65760507],
                [35.97989294],
            ]
        ).ravel()
        np.testing.assert_almost_equal(
            preds, self.model.predict(self.Xt), decimal=6
        )

    def test_score(self):
        """Test R^2 calculation."""
        np.testing.assert_almost_equal(
            0.9847652, self.model.score(self.Xc, self.yc), decimal=6
        )
        np.testing.assert_almost_equal(
            0.9141127, self.model.score(self.Xt, self.yt), decimal=6
        )

    def test_categorize(self):
        """Test extreme/outlier detection."""
        ext = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ]
        )
        out = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )
        np.testing.assert_equal(
            ext, self.model.check_xy_outliers(self.Xc, self.yc)[0]
        )
        np.testing.assert_equal(
            out, self.model.check_xy_outliers(self.Xc, self.yc)[1]
        )

        ext = np.array([False, False, False, False, False, False, False, False])
        out = np.array([False, False, True, False, True, False, False, False])

        np.testing.assert_equal(
            ext, self.model.check_xy_outliers(self.Xt, self.yt)[0]
        )
        np.testing.assert_equal(
            out, self.model.check_xy_outliers(self.Xt, self.yt)[1]
        )


class TestPLS_Scaled(unittest.TestCase):
    """Test PLS class with scaling used."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        df = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + "/data/pls_train.csv"
        )
        self.X = np.array(df.values[:, 3:], dtype=float)
        self.y = np.array(df["Water"].values, dtype=float)
        self.model = PLS(
            n_components=6, alpha=0.05, gamma=0.01, scale_x=True, robust=True
        )
        self.model.fit(self.X, self.y)

    """def test_sklearn_compatibility(self):
        #Check compatible with sklearn's estimator API.
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(PLS(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)"""

    def test_transform(self):
        """Check a few x-scores."""
        res = self.model.transform(self.X).ravel()[:3]
        ans = np.array([-11.77911937, -7.40108219, -0.98177486])
        np.testing.assert_almost_equal(res, ans, decimal=6)
        self.assertEqual(self.model._PLS__Nh_, 5)
        self.assertEqual(self.model._PLS__Nq_, 1)
        self.assertEqual(self.model._PLS__Nz_, 1)
        np.testing.assert_almost_equal(
            self.model._PLS__h0_, 0.050062304729048504 * (self.X.shape[0] - 1)
        )
        np.testing.assert_almost_equal(
            self.model._PLS__q0_, 0.09231118088499507
        )
        np.testing.assert_almost_equal(
            self.model._PLS__z0_, 0.040841606976563964
        )

    def test_h_q(self):
        """Check some h and q values."""
        h, q = self.model._h_q(self.X)
        ans_h = np.array([0.08272128, 0.12718815, 0.0304968]) * (
            self.X.shape[0] - 1
        )
        ans_q = np.array([0.44912905, 0.31931337, 0.10831947])
        np.testing.assert_almost_equal(h[:3], ans_h, decimal=6)
        np.testing.assert_almost_equal(q[:3], ans_q, decimal=6)

    def test_f(self):
        """Check some f values."""
        res = self.model._f(*self.model._h_q(self.X))[:3]
        ans = np.array([13.12721385, 16.16208378, 4.21930139])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_z(self):
        """Check some z values."""
        res = self.model._z(self.X, self.y)[:3]
        ans = np.array([1.48329627e-03, 3.39669025e-03, 1.06630496e-01])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_g(self):
        """Check some g values."""
        res = self.model._g(self.X, self.y)[:3]
        ans = np.array([13.16353211, 16.24525118, 6.83013148])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_predict(self):
        """Check some predictions."""
        res = self.model.predict(self.X)[:3]
        ans = np.array([13.13851359, 13.15828113, 12.82654325])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_x_out(self):
        """Check critical distances for X."""
        res = np.array([self.model._PLS__x_crit_, self.model._PLS__x_out_])
        ans = np.array([12.59158724, 27.93536325])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_xy_out(self):
        """Check critical distances for XY."""
        res = np.array([self.model._PLS__xy_crit_, self.model._PLS__xy_out_])
        ans = np.array([14.06714045, 29.95863241])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_score(self):
        """Check score."""
        np.testing.assert_almost_equal(
            self.model.score(self.X, self.y), 0.924088794825304
        )


class TestPLS_Unscaled(unittest.TestCase):
    """Test PLS class without scaling used."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        df = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + "/data/pls_train.csv"
        )
        self.X = np.array(df.values[:, 3:], dtype=float)
        self.y = np.array(df["Water"].values, dtype=float)
        self.model = PLS(
            n_components=6, alpha=0.05, gamma=0.01, scale_x=False, robust=True
        )
        self.model.fit(self.X, self.y)

    """def test_sklearn_compatibility(self):
        #Check compatible with sklearn's estimator API.
        from sklearn.utils.estimator_checks import check_estimator

        try:
            check_estimator(PLS(n_components=1))
        except Exception as e:
            error = str(e)
        else:
            error = None
        self.assertIsNone(error, msg=error)"""

    def test_transform(self):
        """Check a few x-scores."""
        res = self.model.transform(self.X).ravel()[:3]
        ans = np.array([0.55028919, -0.1011258, -0.03846212])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_h_q(self):
        """Check some h and q values."""
        h, q = self.model._h_q(self.X)
        ans_h = np.array([0.13355759, 0.13201587, 0.03812691]) * (
            self.X.shape[0] - 1
        )
        ans_q = np.array([5.08900958e-06, 3.49069667e-04, 2.25974209e-05])
        np.testing.assert_almost_equal(h[:3], ans_h, decimal=6)
        np.testing.assert_almost_equal(q[:3], ans_q, decimal=6)

    def test_f(self):
        """Check some f values."""
        res = self.model._f(*self.model._h_q(self.X))[:3]
        ans = np.array([18.13690428, 21.96959773, 5.42599725])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_z(self):
        """Check some z values."""
        res = self.model._z(self.X, self.y)[:3]
        ans = np.array([3.25158186e-04, 4.34807900e-09, 9.94443229e-02])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_g(self):
        """Check some g values."""
        res = self.model._g(self.X, self.y)[:3]
        ans = np.array([18.14605433, 21.96959786, 8.22439253])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_predict(self):
        """Check some predictions."""
        res = self.model.predict(self.X)[:3]
        ans = np.array([13.11803214, 13.09993406, 12.81534794])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_x_out(self):
        """Check critical distances for X."""
        res = np.array([self.model._PLS__x_crit_, self.model._PLS__x_out_])
        ans = np.array([15.50731306, 31.91074006])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_xy_out(self):
        """Check critical distances for XY."""
        res = np.array([self.model._PLS__xy_crit_, self.model._PLS__xy_out_])
        ans = np.array([16.9189776, 33.80494136])
        np.testing.assert_almost_equal(res, ans, decimal=6)

    def test_score(self):
        """Check score."""
        np.testing.assert_almost_equal(
            self.model.score(self.X, self.y), 0.9243675391888061
        )
