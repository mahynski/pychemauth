"""
Unittests for LOD preprocessing.

author: nam
"""
import copy
import os
import unittest

import numpy as np
import pandas as pd

from PyChemAuth.preprocessing.missing import LOD


class TestLOD(unittest.TestCase):
    """Test LOD preprocessing."""

    @classmethod
    def setUpClass(self):
        """Set up class with a baseline model."""
        self.X = np.array(
            [
                [1.0, np.nan, 3.0, 4.0],
                [-1, 3.0, 2.0, -1],
                [5.0, 1.0, -1, 5.0],
                [2.0, 3.0, np.nan, 5.0],
            ]
        )

        self.lod = np.array([0.15, 0.15, 0.25, 0.15])

    def test_nan(self):
        """Test nan as missing_value."""
        imputer = LOD(self.lod, missing_values=np.nan, seed=0)
        X_lod = imputer.fit_transform(self.X)
        np.testing.assert_almost_equal(
            X_lod,
            np.array(
                [
                    [
                        1.00000000e00,
                        9.55442531e-02,
                        3.00000000e00,
                        4.00000000e00,
                    ],
                    [
                        4.04680071e-02,
                        3.00000000e00,
                        2.00000000e00,
                        6.14602859e-03,
                    ],
                    [
                        5.00000000e00,
                        1.00000000e00,
                        4.13190888e-03,
                        5.00000000e00,
                    ],
                    [
                        2.00000000e00,
                        3.00000000e00,
                        2.03317560e-01,
                        5.00000000e00,
                    ],
                ]
            ),
            decimal=6,
        )

    def test_negative(self):
        """Test -1 as missing_value."""
        imputer = LOD(self.lod, missing_values=-1, seed=0)
        X_lod = imputer.fit_transform(self.X)
        np.testing.assert_almost_equal(
            X_lod[~np.isnan(X_lod)],
            np.array(
                [
                    1.0,
                    3.0,
                    4.0,
                    0.09554425,
                    3.0,
                    2.0,
                    0.04046801,
                    5.0,
                    1.0,
                    0.01024338,
                    5.0,
                    2.0,
                    3.0,
                    5.0,
                ]
            ),
            decimal=6,
        )
        self.assertTrue(
            np.all(
                np.array(
                    [
                        [False, True, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, True, False],
                    ]
                )
                == np.isnan(X_lod)
            )
        )
