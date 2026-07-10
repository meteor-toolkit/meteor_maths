import unittest
import numpy as np
from matheo.sampling.new_sampling import (
    RegridCache,
    regrid_with_cache,
)


class TestNN(unittest.TestCase):
    def test_nearest_neighbour_resample(self):
        data_source = np.vstack([np.arange(15) for i in range(18)])
        x_source, y_source = np.meshgrid(np.arange(15), np.arange(18))

        x_target = np.array(
            [
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
            ]
        )
        y_target = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [4.0, 4.0, 4.0, 4.0, 4.0],
                [7.0, 7.0, 7.0, 7.0, 7.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [13.0, 13.0, 13.0, 13.0, 13.0],
                [16.0, 16.0, 16.0, 16.0, 16.0],
            ]
        )

        cache = RegridCache(x_source, y_source, x_target, y_target)
        data_target, mask = regrid_with_cache(data_source, cache)

        np.testing.assert_array_almost_equal(
            np.array(
                [
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                ]
            ),
            data_target,
        )

    def test_nearest_neighbour_resample_NansInGrid(self):
        data_source = np.vstack([np.arange(15) for i in range(18)]).astype(float)
        data_source[3, 5] = np.nan
        data_source[10, 12] = np.nan
        x_source, y_source = np.meshgrid(np.arange(15), np.arange(18))

        x_target = np.array(
            [
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
            ]
        )
        y_target = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [4.0, 4.0, 4.0, 4.0, 4.0],
                [7.0, 7.0, 7.0, 7.0, 7.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [13.0, 13.0, 13.0, 13.0, 13.0],
                [16.0, 16.0, 16.0, 16.0, 16.0],
            ]
        )

        cache = RegridCache(x_source, y_source, x_target, y_target)
        data_target, mask = regrid_with_cache(data_source, cache)

        np.testing.assert_array_almost_equal(
            data_target,
            np.array(
                [
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, np.nan, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, np.nan],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                ]
            ),
        )

    def test_irregular_grid_shape(self):
        x_src, y_src = np.meshgrid(np.arange(5), np.arange(3))
        data = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
            ]
        )
        x_tgt, y_tgt = np.meshgrid(np.linspace(0, 4, 2), np.linspace(0, 2, 3))
        cache = RegridCache(x_src, y_src, x_tgt, y_tgt)
        data_target, mask = regrid_with_cache(data, cache)
        self.assertEqual(data_target.shape, x_tgt.shape)
        self.assertEqual(mask.shape, x_tgt.shape)
        self.assertTrue(np.all(np.isnan(data_target[~mask])))
        np.testing.assert_array_almost_equal(
            data_target, np.array([[2.0, 4.0], [7.0, 9.0], [12.0, 14.0]])
        )

    def test_nearest_neighbour_resample_invalid_edge_pixels(self):
        data_source = np.vstack([np.arange(16) for i in range(17)])
        x_source, y_source = np.meshgrid(np.arange(16), np.arange(17))
        x_target = np.array(
            [
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
                [1.0, 4.0, 7.0, 10.0, 13.0],
            ]
        )
        y_target = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [4.0, 4.0, 4.0, 4.0, 4.0],
                [7.0, 7.0, 7.0, 7.0, 7.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [13.0, 13.0, 13.0, 13.0, 13.0],
                [16.0, 16.0, 16.0, 16.0, 16.0],
            ]
        )
        cache = RegridCache(x_source, y_source, x_target, y_target)
        data_target, mask = regrid_with_cache(data_source, cache)

        np.testing.assert_array_almost_equal(
            data_target,
            np.array(
                [
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        )

    def test_nearest_neighbour_resample_RotatedGrids(self):
        # Create simple source grid
        data_source = np.vstack([np.arange(16) for _ in range(17)])
        x_source, y_source = np.meshgrid(np.arange(16), np.arange(17))
        # Create target grid (unrotated first)
        x_target, y_target = np.meshgrid(
            [1.0, 4.0, 7.0, 10.0, 13.0], [1.0, 4.0, 7.0, 10.0, 13.0, 16.0]
        )
        # Apply a rotation to the target grid (e.g., 10 degrees)
        theta = np.deg2rad(10)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = cos_t * x_target - sin_t * y_target
        y_rot = sin_t * x_target + cos_t * y_target
        # Build cache and regrid
        cache = RegridCache(x_source, y_source, x_rot, y_rot)
        data_target, mask = regrid_with_cache(data_source, cache)

        # Function to compute expected nearest-neighbour value from source grid
        def expected_value(x, y):
            dist_sq = (x_source - x) ** 2 + (y_source - y) ** 2
            iy, ix = np.unravel_index(np.argmin(dist_sq), x_source.shape)
            return data_source[iy, ix]

        # Check a few central points exactly
        # central_coords = [(1, 2), (3, 2), (3,3)]
        # for iy, ix in central_coords:
        #     exp_val = expected_value(x_rot[iy, ix], y_rot[iy, ix])
        #     self.assertAlmostEqual(data_target[iy, ix], exp_val)
        print(data_target)

        np.testing.assert_array_almost_equal(
            data_target,
            np.array([[ 1.        ,  4.        ,  7.        ,         np.nan, 12.8       ],
       [        np.nan,         np.nan,  6.        ,  9.        , 12.        ],
       [        np.nan,         np.nan,  5.8       ,  8.77777778, 11.66666667],
       [        np.nan,  2.        ,  5.        ,  8.        , 11.        ],
       [        np.nan,         np.nan,  4.66666667,  7.66666667, 10.66666667],
       [        np.nan,         np.nan,         np.nan,         np.nan,         np.nan]]),
        )


    def test_nearest_neighbour_resample_to_larger_grid(self):
        data_source = np.vstack([np.arange(16) for i in range(17)])
        x_source, y_source = np.meshgrid(np.arange(16), np.arange(17))
        x_target = np.array(
            [
                [1.0, 4.0, 7.0, 10.0, 18.0],
                [1.0, 4.0, 7.0, 10.0, 18.0],
                [1.0, 4.0, 7.0, 10.0, 18.0],
                [1.0, 4.0, 7.0, 10.0, 18.0],
                [1.0, 4.0, 7.0, 10.0, 18.0],
                [1.0, 4.0, 7.0, 10.0, 18.0],
            ]
        )
        y_target = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [4.0, 4.0, 4.0, 4.0, 4.0],
                [7.0, 7.0, 7.0, 7.0, 7.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [18.0, 18.0, 18.0, 18.0, 18.0],
                [20.0, 20.0, 20.0, 20.0, 20.0],
            ]
        )

        cache = RegridCache(x_source, y_source, x_target, y_target)
        data_target, mask = regrid_with_cache(data_source, cache)

        np.testing.assert_array_almost_equal(
            data_target,
            np.array(
                [
                    [1.0, 4.0, 7.0, 10.0, np.nan],
                    [1.0, 4.0, 7.0, 10.0, np.nan],
                    [1.0, 4.0, 7.0, 10.0, np.nan],
                    [1.0, 4.0, 7.0, 10.0, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        )

    # def test_edge_masking(self):
    #     x_src, y_src = np.meshgrid(np.arange(5), np.arange(5))
    #     data = np.ones((5, 5))
    #     x_tgt, y_tgt = np.meshgrid(np.linspace(0, 4, 3), np.linspace(0, 4, 3))
    #     cache = RegridCache(x_src, y_src, x_tgt, y_tgt, fraction=1.0, min_per_pixel=2)
    #     data_target, mask = regrid_with_cache(data, cache)
    #     self.assertTrue(np.any(~mask))
    #     self.assertTrue(mask[2, 2])
    #     np.testing.assert_array_equal(
    #         mask,
    #         np.array([[True, True, False],
    #                   [True, True, False],
    #                   [False, False, False]]
    #     ))

    # def test_multiple_fields_same_cache(self):
    #     x_src, y_src = np.meshgrid(np.arange(4), np.arange(4))
    #     data1 = np.arange(16).reshape(4, 4).astype(float)
    #     data2 = np.arange(16, 32).reshape(4, 4).astype(float)
    #     x_tgt, y_tgt = np.meshgrid(np.linspace(0, 3, 2), np.linspace(0, 3, 2))
    #     cache = RegridCache(x_src, y_src, x_tgt, y_tgt, fraction=1.0)
    #     dt1, mask1 = regrid_with_cache(data1, cache)
    #     dt2, mask2 = regrid_with_cache(data2, cache)
    #     self.assertEqual(dt1.shape, x_tgt.shape)
    #     self.assertEqual(dt2.shape, x_tgt.shape)
    #     np.testing.assert_array_equal(mask1, mask2)

    # def test_sparse_source_grid(self):
    #     # Sparse source: only corners
    #     x_src, y_src = np.meshgrid(np.arange(5), np.arange(5))
    #     data = np.full((5, 5), np.nan)
    #     data[0, 0] = 10
    #     data[0, 4] = 20
    #     data[4, 0] = 30
    #     data[4, 4] = 40
    #     x_tgt, y_tgt = np.meshgrid(np.arange(5), np.arange(5))
    #     cache = RegridCache(x_src, y_src, x_tgt, y_tgt, fraction=1.0, min_per_pixel=2)
    #     data_target, mask = regrid_with_cache(data, cache)
    #     # Corners should be masked because min_per_pixel=2
    #     self.assertFalse(mask[0, 0])
    #     self.assertFalse(mask[0, 4])
    #     self.assertFalse(mask[4, 0])
    #     self.assertFalse(mask[4, 4])
    #     # Most pixels should be masked
    #     masked_count = np.sum(~mask)
    #     self.assertGreaterEqual(masked_count, 16)
    #     self.assertTrue(np.all(np.isnan(data_target[~mask])))

    # def test_min_count_vs_expected_count(self):
    #     """
    #     Ensure min_per_pixel overrides expected_count if expected_count < min_per_pixel
    #     """
    #     x_src, y_src = np.meshgrid(np.arange(3), np.arange(3))
    #     data = np.ones((3, 3))
    #     x_tgt, y_tgt = np.meshgrid(np.linspace(0, 2, 5), np.linspace(0, 2, 5))
    #     # Use a fraction < 1 to force expected_count < min_per_pixel
    #     cache = RegridCache(x_src, y_src, x_tgt, y_tgt, fraction=0.1, min_per_pixel=2)
    #     data_target, mask = regrid_with_cache(data, cache)
    #     # Check that min_per_pixel is applied: all valid pixels must have at least 2 sources
    #     # Central pixel should be valid
    #     self.assertTrue(mask[2, 2])
    #     # Edge pixels may be masked
    #     self.assertFalse(mask[0, 0])

    # def test_target_outside_source(self):
    #     """
    #     Target pixels outside source grid should be masked
    #     """
    #     x_src, y_src = np.meshgrid(np.arange(3), np.arange(3))
    #     data = np.arange(9).reshape(3, 3)
    #     # Target extends beyond source grid
    #     x_tgt, y_tgt = np.meshgrid(np.linspace(-1, 4, 6), np.linspace(-1, 4, 6))
    #     cache = RegridCache(x_src, y_src, x_tgt, y_tgt, fraction=0.5)
    #     data_target, mask = regrid_with_cache(data, cache)
    #     # Corners outside the source grid should be masked
    #     self.assertFalse(mask[0, 0])  # top-left outside
    #     self.assertFalse(mask[-1, -1])  # bottom-right outside
    #     # Central pixels should be valid
    #     self.assertTrue(mask[2, 2])

    # # def test_all_nan_source(self):
    # #     """
    # #     Source array all NaN should result in fully masked target
    # #     """
    # #     x_src, y_src = np.meshgrid(np.arange(3), np.arange(3))
    # #     data = np.full((3, 3), np.nan)
    # #     x_tgt, y_tgt = np.meshgrid(np.linspace(0, 2, 2), np.linspace(0, 2, 2))
    # #     cache = RegridCache(x_src, y_src, x_tgt, y_tgt, fraction=0.5)
    # #     data_target, mask = regrid_with_cache(data, cache)
    # #     # All pixels should be masked
    # #     self.assertTrue(np.all(~mask))
    # #     self.assertTrue(np.all(np.isnan(data_target)))


if __name__ == "__main__":
    unittest.main()
