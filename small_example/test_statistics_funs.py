import unittest
import numpy as np
from utilities import compute_pval, acceptance_ratio, compute_critical_val, ecdf

class TestStatisticsFunctions(unittest.TestCase):

    def test_compute_pval(self):
        d_stat_values = np.array([1, 2, 3, 4, 5])
        x_dval = 3
        expected_pval = 0.6  # 3/5 values are >= 3
        self.assertAlmostEqual(compute_pval(x_dval, d_stat_values), expected_pval, places=5)

    def test_acceptance_ratio(self):
        p_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        alpha = 0.25
        expected_ratio = 0.6  # 3/5 values are >= 0.25
        self.assertAlmostEqual(acceptance_ratio(p_vals, alpha), expected_ratio, places=5)

    def test_compute_critical_val(self):
        d_stat_values = np.array([1, 2, 3, 4, 5])
        alpha = 0.2
        # In a right-tailed test with these values and alpha, 
        # the critical value would typically be near the upper end of the distribution.
        # This specific expected value would depend on the precise behavior and implementation details
        # of compute_critical_val, and might need adjustment.
        expected_critical_val = 4  # Example placeholder, adjust based on actual implementation
        self.assertAlmostEqual(compute_critical_val(d_stat_values, alpha), expected_critical_val, places=5)

    def test_ecdf(self):
        data = np.array([1, 2, 3, 4, 5])
        expected_x = np.array([1, 2, 3, 4, 5])
        expected_y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        x, y = ecdf(data)
        np.testing.assert_array_almost_equal(x, expected_x)
        np.testing.assert_array_almost_equal(y, expected_y)

if __name__ == '__main__':
    unittest.main()
