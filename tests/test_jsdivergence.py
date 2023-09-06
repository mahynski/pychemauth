"""
Unittests for JensenShannonDivergence.

author: nam
"""
import unittest

import numpy as np

from pychemauth.preprocessing.feature_selection import JensenShannonDivergence


class TestJensenShannonDivergence(unittest.TestCase):
    """Test JensenShannonDivergence class."""

    def setUp(self):
        """Set up the class."""
        self.js = JensenShannonDivergence(top_k=1, per_class=True)

        # Feature 0, div = 0, feature 1 = 0.66
        self.X = np.array([[1, 1], [2, 2], [3, 3], [0, 0], [4, 2], [5, 5]])
        self.y = np.array(["A", "A", "A", "B", "B", "B"])

    def test_set_params(self):
        """Check compatible with sklearn's estimator API."""
        given = {
            "epsilon": 1.23,
            "threshold": 2.34,
            "top_k": 1,
            "per_class": True,
            "feature_names": ["a", "b", "c"],
            "bins": 7,
            "robust": False,
        }
        js = JensenShannonDivergence(**given)
        returned = js.get_params()
        self.assertEqual(given, returned)

        given["robust"] = True
        js = JensenShannonDivergence(**given)
        returned = js.get_params()
        self.assertEqual(given, returned)

    def test_make_prob(self):
        """Test making probabilities out of histograms."""
        # Make a simple distribution
        p = [1, 1, 2, 3]
        normed_prob = self.js._make_prob(p, ranges=(1, 3), bins=3)
        self.assertTrue(np.allclose(normed_prob, [0.5, 0.25, 0.25]))

        # Leave a hole in the middle - epsilon should not affect
        p = [1, 1, 2, 4]
        normed_prob = self.js._make_prob(p, ranges=(1, 4), bins=4)
        self.assertTrue(np.allclose(normed_prob, [0.5, 0.25, 0.0, 0.25]))

        # Leave holes on the edges - epsilon should not affect
        p = [1, 1, 2, 3]
        normed_prob = self.js._make_prob(p, ranges=(0, 5), bins=5)
        self.assertTrue(np.allclose(normed_prob, [0.0, 0.5, 0.25, 0.25, 0.0]))

    def test_jensen_shannon(self):
        """Test computation of Jensen-Shannon divergence."""
        # Same distribution has div = 0
        p = np.array([1, 2, 3])
        normed_p = self.js._make_prob(p, ranges=(0, 5), bins=6)
        self.assertAlmostEqual(self.js._jensen_shannon(normed_p, normed_p), 0.0)

        # Disjoint have div = 1
        q = np.array([0, 4, 5])
        normed_q = self.js._make_prob(q, ranges=(0, 5), bins=6)
        self.assertAlmostEqual(self.js._jensen_shannon(normed_p, normed_q), 1.0)

        # A manual calculation for some intermediate
        r = np.array([0, 2, 5])
        normed_r = self.js._make_prob(r, ranges=(0, 5), bins=6)
        a = np.array([0, 1 / 3.0, 1 / 3.0, 1 / 3.0, 0, 0]) + 1.0e-12
        b = np.array([1 / 3.0, 0, 1 / 3.0, 0, 0, 1 / 3.0]) + 1.0e-12
        c = 0.5 * a + 0.5 * b

        def kl_div(p, q):
            return np.sum(p * np.log2(p / q))

        div = 0.5 * kl_div(a, c) + 0.5 * kl_div(b, c)
        self.assertAlmostEqual(div, self.js._jensen_shannon(normed_p, normed_r))

    def test_fit_per_class(self):
        """Test fitting on a per_class basis."""
        self.js.fit(self.X, self.y)
        div = self.js.divergence

        # Feature 0, A and B disjoint (same for both classes) and
        # feature 0 is the leading divergence.
        self.assertAlmostEqual(div["A"][0][0], 0)
        self.assertAlmostEqual(div["B"][0][0], 0)
        self.assertAlmostEqual(div["A"][0][1]["A"], 1.0)
        self.assertAlmostEqual(div["A"][0][1]["B"], 1.0)
        self.assertAlmostEqual(div["B"][0][1]["A"], 1.0)
        self.assertAlmostEqual(div["B"][0][1]["B"], 1.0)

        # Feature 1 has the second highest divergence.
        self.assertAlmostEqual(div["A"][1][0], 1)
        self.assertAlmostEqual(div["B"][1][0], 1)
        self.assertAlmostEqual(div["A"][1][1]["A"], 2 / 3.0)
        self.assertAlmostEqual(div["A"][1][1]["B"], 2 / 3.0)
        self.assertAlmostEqual(div["B"][1][1]["A"], 2 / 3.0)
        self.assertAlmostEqual(div["B"][1][1]["B"], 2 / 3.0)

    def test_outlier(self):
        """Test outlier removal."""
        # Outlier should be removed and so results shouldn't change
        X = np.vstack((self.X, [[100, 100]]))
        y = np.concatenate((self.y, ["A"]))
        js = JensenShannonDivergence(top_k=1, per_class=True, robust=True)
        js.fit(X, y)
        div = js.divergence

        # Feature 0, A and B disjoint (same for both classes) and
        # feature 0 is the leading divergence.
        self.assertAlmostEqual(div["A"][0][0], 0)
        self.assertAlmostEqual(div["B"][0][0], 0)
        self.assertAlmostEqual(div["A"][0][1]["A"], 1.0)
        self.assertAlmostEqual(div["A"][0][1]["B"], 1.0)
        self.assertAlmostEqual(div["B"][0][1]["A"], 1.0)
        self.assertAlmostEqual(div["B"][0][1]["B"], 1.0)

        # Feature 1 has the second highest divergence.
        self.assertAlmostEqual(div["A"][1][0], 1)
        self.assertAlmostEqual(div["B"][1][0], 1)
        self.assertAlmostEqual(div["A"][1][1]["A"], 2 / 3.0)
        self.assertAlmostEqual(div["A"][1][1]["B"], 2 / 3.0)
        self.assertAlmostEqual(div["B"][1][1]["A"], 2 / 3.0)
        self.assertAlmostEqual(div["B"][1][1]["B"], 2 / 3.0)

    def test_fit_overall(self):
        """Test fitting on an overall basis."""
        js = JensenShannonDivergence(top_k=1, per_class=False)
        js.fit(self.X, self.y)
        div = js.divergence

        # Feature 0 is the most important
        self.assertAlmostEqual(div[0][0], 0)
        self.assertAlmostEqual(div[1][0], 1)

        # Feature 0 has div = 1, feature 1 has div = 2/3
        self.assertAlmostEqual(div[0][1]["A"], 1.0)
        self.assertAlmostEqual(div[0][1]["B"], 1.0)
        self.assertAlmostEqual(div[1][1]["A"], 2 / 3.0)
        self.assertAlmostEqual(div[1][1]["B"], 2 / 3.0)

    def test_get_feature_names_out(self):
        """Test get_feature_names_out returns the correct feature."""
        # Without names
        self.js.fit(self.X, self.y)
        self.assertTrue(np.all(self.js.get_feature_names_out() == np.array([True, False])))

        # With names
        for per_class in [True, False]:
            js = JensenShannonDivergence(
                top_k=1, per_class=per_class, feature_names=["FEAT_A", "FEAT_B"]
            )
            js.fit(self.X, self.y)
            self.assertTrue(np.all(js.get_feature_names_out() == np.array(["FEAT_A"])))

    def test_transform(self):
        """Test transformation selects right column."""
        # X_train
        self.js.fit(self.X, self.y)
        self.assertTrue(
            np.all(self.X[:, 0].reshape(-1, 1) == self.js.transform(self.X))
        )

        # Some X_test
        X_test = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])
        self.assertTrue(
            np.all(X_test[:, 0].reshape(-1, 1) == self.js.transform(X_test))
        )

        # X needs to have the same shape as before
        X_bad = X_test = np.array([[10, 20, 30], [30, 40, 50]])
        try:
            self.js.transform(X_bad)
        except ValueError:
            pass
        else:
            self.assertTrue(False)
