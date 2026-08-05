"""Tests for matheo.sampling.utils module"""

import unittest

import numpy as np

import matheo.sampling.utils as util


class TestUtils(unittest.TestCase):
    def test_add_geolocation_error(self):
        lat_grid_deg = 0.0001 * np.linspace(0, 1, 9).reshape(3, 3)
        lon_grid_deg = 0.0001 * np.linspace(0, -1, 9).reshape(3, 3)

        lat_grid_err, lon_grid_err = util.add_geolocation_error(
            111, 11.1, lat_grid_deg, lon_grid_deg
        )

        np.testing.assert_array_almost_equal(
            lat_grid_err, (lat_grid_deg + 0.001) * np.ones((3, 3)), 5
        )
        np.testing.assert_array_almost_equal(
            lon_grid_err, (lon_grid_deg + 0.0001) * np.ones((3, 3)), 5
        )

        lat_grid_err, lon_grid_err = util.add_geolocation_error(
            111, 0, lat_grid_deg, lon_grid_deg
        )

        np.testing.assert_array_almost_equal(
            lat_grid_err, (lat_grid_deg + 0.001) * np.ones((3, 3)), 5
        )
        np.testing.assert_array_almost_equal(lon_grid_err, lon_grid_deg)

        lat_grid_err, lon_grid_err = util.add_geolocation_error(
            0, 11.1, lat_grid_deg, lon_grid_deg
        )

        np.testing.assert_array_almost_equal(
            lon_grid_err, (lon_grid_deg + 0.0001) * np.ones((3, 3))
        )
        np.testing.assert_array_almost_equal(lat_grid_err, lat_grid_deg)


if __name__ == "__main__":
    unittest.main()
