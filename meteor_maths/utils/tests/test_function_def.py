"""Tests for function_def module"""

import unittest

import numpy as np

from meteor_maths.utils import function_def as fd

__author__ = "Sam Hunt"
__created__ = "01/08/2021"
__maintainer__ = "Sam Hunt"
__email__ = "sam.hunt@npl.co.uk"
__status__ = "Development"


class TestFunctionDef(unittest.TestCase):
    def test_f_tophat(self):

        x = np.arange(0, 11, 1)
        y = fd.f_tophat(x, 5, 4)

        np.testing.assert_array_equal(y, np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

    def test_f_triangle(self):

        x = np.arange(0, 11, 1)
        y = fd.f_triangle(x, 5, 2)

        np.testing.assert_array_equal(y, np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]))

    def test_f_gaussian(self):

        x = np.arange(0, 11, 1)
        y = fd.f_gaussian(x, 5, 2 * np.sqrt(2 * np.log(2)) * 2 / 2)

        np.testing.assert_array_almost_equal(
            y,
            np.array(
                [
                    3.726653e-06,
                    3.354626e-04,
                    1.110900e-02,
                    1.353353e-01,
                    6.065307e-01,
                    1.000000e00,
                    6.065307e-01,
                    1.353353e-01,
                    1.110900e-02,
                    3.354626e-04,
                    3.726653e-06,
                ]
            ),
        )

    def test_f_normalised_tophat(self):
        x = np.arange(0, 11, 1)
        y = fd.f_normalised(fd.f_tophat, x, 5, 4)

        np.testing.assert_array_almost_equal(
            y,
            np.array([0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0]),
            decimal=2,
        )

    def test_repeat_f(self):
        y, x = fd.repeat_f(
            f=fd.f_tophat,
            centres=np.array([5.0, 6.0, 7.0]),
            widths=np.array([2.0, 4.0, 8.0]),
            x_sampling=1.0,
            xlim_width=1.5 / 2,
        )

        x_expected = np.arange(1, 14, 1)
        y_expected = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ]
        )

        np.testing.assert_array_equal(x, x_expected)
        np.testing.assert_array_equal(y, y_expected)

    def test_repeat_f_normalise(self):
        y, x = fd.repeat_f(
            f=fd.f_tophat,
            centres=np.array([5.0, 6.0, 7.0]),
            widths=np.array([2.0, 4.0, 8.0]),
            x_sampling=1.0,
            xlim_width=1.5 / 2,
            normalise=True,
        )

        x_expected = np.arange(1, 14, 1)
        y_expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.0,
                    0.0,
                ],
            ]
        )

        np.testing.assert_array_equal(x, x_expected)
        np.testing.assert_array_almost_equal(y, y_expected, decimal=2)

    def test_iter_f(self):
        f_iter = fd.iter_f(
            f=fd.f_tophat,
            centres=np.array([5.0, 6.0, 7.0]),
            widths=np.array([2.0, 4.0, 8.0]),
            x_sampling=1.0,
            xlim_width=1,
        )

        x_expected = [np.arange(3, 8, 1), np.arange(2, 11, 1), np.arange(-1, 16, 1)]
        y_expected = [
            np.array([0.0, 1.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ]

        for (y, x), x_exp, y_exp in zip(f_iter, x_expected, y_expected):
            np.testing.assert_array_equal(x, x_exp)
            np.testing.assert_array_equal(y, y_exp)

    def test_iter_f_normalise(self):
        f_iter = fd.iter_f(
            f=fd.f_tophat,
            centres=np.array([5.0, 6.0, 7.0]),
            widths=np.array([2.0, 4.0, 8.0]),
            x_sampling=1.0,
            xlim_width=1,
            normalise=True,
        )

        x_expected = [np.arange(3, 8, 1), np.arange(2, 11, 1), np.arange(-1, 16, 1)]
        y_expected = [
            np.array([0.0, 0.5, 0.5, 0.5, 0.0]),
            np.array([0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0]),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ]

        for (y, x), x_exp, y_exp in zip(f_iter, x_expected, y_expected):
            np.testing.assert_array_equal(x, x_exp)
            np.testing.assert_array_almost_equal(y, y_exp, decimal=2)


if __name__ == "__main__":
    unittest.main()
