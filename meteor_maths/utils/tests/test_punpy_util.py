"""
Tests for punpy_util module
"""

import unittest

import numpy as np

from meteor_maths.utils.punpy_util import _max_dim, _unc_to_dim, func_with_unc

"""___Authorship___"""
__author__ = "Sam Hunt"
__created__ = "1/8/2021"


class TestPunpyUtils(unittest.TestCase):
    def test__max_dim(self):
        a = np.zeros((3, 4, 5, 6))
        b = np.zeros(3)

        self.assertEqual(_max_dim([a, b]), 4)

    def test__unc_to_dim_None(self):
        self.assertIsNone(_unc_to_dim(None, 2))

    def test__unc_to_dim_01_xlen(self):
        np.testing.assert_array_equal(_unc_to_dim(3, 1, x_len=2), np.array([3, 3]))

    def test__unc_to_dim_01_x(self):
        np.testing.assert_array_equal(_unc_to_dim(0.1, 1, x=np.array([10, 20])), np.array([1, 2]))

    def test__unc_to_dim_02_xlen(self):
        np.testing.assert_array_equal(_unc_to_dim(3, 2, x_len=2), np.array([[3, 0], [0, 3]]))

    def test__unc_to_dim_02_x(self):
        np.testing.assert_array_equal(_unc_to_dim(0.1, 2, x=np.array([10, 20])), np.array([[1, 0], [0, 2]]))

    def test__unc_to_dim_12_x(self):
        np.testing.assert_array_equal(_unc_to_dim(np.array([1, 2]), 2), np.array([[1, 0], [0, 2]]))

    def test_func_with_unc_None(self):
        def poly(x1, x2):
            return x1**2 + 3 * x2

        y, u_y = func_with_unc(
            poly,
            params={"x1": np.array([1, 2, 3]), "x2": np.array([1, 2, 3])},
            u_params={"x1": None, "x2": None},
        )

        np.testing.assert_array_equal(y, np.array([4, 10, 18]))
        self.assertIsNone(u_y)

    def test_func_with_unc(self):
        def poly(x1, x2, x3):
            return x1**2 + 3 * x2 + x3

        y, _ = func_with_unc(
            poly,
            params={"x1": np.array([1, 2, 3]), "x2": np.array([1, 2, 3]), "x3": 1},
            u_params={"x1": 0.1, "x2": np.array([3, 3, 3])},
            parallel=False,  # have to test punpy not in parallel for some reason...
        )

        np.testing.assert_array_equal(y, np.array([5, 11, 19]))
        # np.testing.assert_array_almost_equal(u_y, np.array([7.60914338, 9.31506956, 9.72720938]))


if __name__ == "__main__":
    unittest.main()
