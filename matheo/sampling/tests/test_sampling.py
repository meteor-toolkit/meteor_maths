"""
Tests for sampling module
"""

import numpy.testing

import unittest
from unittest.mock import patch, call
import numpy as np
import xarray as xr

from matheo.sampling import resample, nearest_neighbour_resample


__author__ = "Maddie Stedman"


class TestSampling(unittest.TestCase):
    def setUp(self) -> None:
        size_dict = {
            "bands": 4,
            "x_source": 15,
            "y_source": 16,
            "x_target": 8,
            "y_target": 8,
        }
        fv = 9.969209968386869e36

        test_ds = xr.Dataset(
            data_vars=dict(
                data_3d_source=(
                    ["bands", "x_source", "y_source"],
                    np.ones(
                        (
                            size_dict["bands"],
                            size_dict["x_source"],
                            size_dict["y_source"],
                        )
                    ),
                ),
                data_3d_target=(
                    ["bands", "x_target", "y_target"],
                    np.ones(
                        (
                            size_dict["bands"],
                            size_dict["x_target"],
                            size_dict["y_target"],
                        )
                    ),
                ),
                data_2d_source=(
                    ["x_source", "y_source"],
                    np.ones(
                        (
                            size_dict["x_source"],
                            size_dict["y_source"],
                        )
                    ),
                ),
                coord1_source=(
                    ["x_source", "y_source"],
                    np.ones((size_dict["x_source"], size_dict["y_source"])),
                ),
                coord2_source=(
                    ["x_source", "y_source"],
                    2 * np.ones((size_dict["x_source"], size_dict["y_source"])),
                ),
                coord1_target=(
                    ["x_target", "y_target"],
                    np.ones((size_dict["x_target"], size_dict["y_target"])),
                ),
                coord2_target=(
                    ["x_target", "y_target"],
                    2 * np.ones((size_dict["x_target"], size_dict["y_target"])),
                ),
            ),
        )

        self.test_ds = test_ds

    @patch("matheo.sampling.sampling.nearest_neighbour_resample")
    def test_resample_2d(self, mock_nearest_neighbour_resample):
        mock_nearest_neighbour_resample.return_value = np.ones(
            self.test_ds.coord1_target.shape
        )

        test_proc_data = resample(
            "data_2d_source",
            self.test_ds,
            x_source=self.test_ds.coord1_source.values,
            y_source=self.test_ds.coord2_source.values,
            x_target=self.test_ds.coord1_target.values,
            y_target=self.test_ds.coord2_target.values,
            k=10,
            mask_invalid=False,
        )

        mock_nearest_neighbour_resample.assert_called_once_with(
            self.test_ds.data_2d_source.values,
            self.test_ds.coord1_source.values,
            self.test_ds.coord2_source.values,
            self.test_ds.coord1_target.values,
            self.test_ds.coord2_target.values,
            k=10,
            mask_invalid=False,
        )

        np.testing.assert_array_equal(test_proc_data.shape, (8))

    @patch("matheo.sampling.sampling.nearest_neighbour_resample")
    def test_resample_2d_same_grid(self, mock_nearest_neighbour_resample):
        mock_nearest_neighbour_resample.return_value = np.ones(
            self.test_ds.coord1_source.shape
        )

        test_proc_data = resample(
            "data_2d_source",
            self.test_ds,
            x_source=self.test_ds.coord1_source.values,
            y_source=self.test_ds.coord2_source.values,
            x_target=self.test_ds.coord1_source.values,
            y_target=self.test_ds.coord2_source.values,
            n_min_source=10,
            mask_invalid=False,
        )

        mock_nearest_neighbour_resample.assert_not_called()

        np.testing.assert_array_equal(test_proc_data, self.test_ds["data_2d_source"])

    @patch("matheo.sampling.sampling.nearest_neighbour_resample")
    def test_resample_3d(self, mock_nearest_neighbour_resample):
        mock_nearest_neighbour_resample.return_value = np.ones((8, 8))

        test_proc_data = resample(
            "data_3d_source",
            self.test_ds,
            x_source=self.test_ds.coord1_source.values,
            y_source=self.test_ds.coord2_source.values,
            x_target=self.test_ds.coord1_target.values,
            y_target=self.test_ds.coord2_target.values,
            n_min_source=10,
            mask_invalid=False,
        )

        # test call args for mock_nearest_neighbour_resample
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[0][0],
            self.test_ds["data_3d_source"].values[0],
        )
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[0][1],
            self.test_ds["coord1_source"].values,
        )
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[0][2],
            self.test_ds["coord2_source"].values,
        )
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[0][3],
            self.test_ds["coord1_target"].values,
        )
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[0][4],
            self.test_ds["coord2_target"].values,
        )
        self.assertEqual(
            mock_nearest_neighbour_resample.call_args[1]["n_min_source"], 10
        )
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[1]["mask_invalid"],
            False,
        )

        self.assertEqual(mock_nearest_neighbour_resample.call_count, 4)

        np.testing.assert_array_equal(test_proc_data.shape, (4, 8, 8))

    @patch("matheo.sampling.sampling.nearest_neighbour_resample")
    def test_resample_3d_mask_invalidTrue(self, mock_nearest_neighbour_resample):
        mock_nearest_neighbour_resample.return_value = np.ones(
            self.test_ds.coord1_target.shape
        )

        test_proc_data = resample(
            "data_3d_source",
            self.test_ds,
            x_source=self.test_ds.coord1_source.values,
            y_source=self.test_ds.coord2_source.values,
            x_target=self.test_ds.coord1_target.values,
            y_target=self.test_ds.coord2_target.values,
            n_min_source=49,
            mask_invalid=True,
        )

        # test call args for mock_nearest_neighbour_resample
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[0][0],
            self.test_ds["data_3d_source"].values[0],
        )
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[0][1],
            self.test_ds["coord1_source"].values,
        )
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[0][2],
            self.test_ds["coord2_source"].values,
        )
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[0][3],
            self.test_ds["coord1_target"].values,
        )
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[0][4],
            self.test_ds["coord2_target"].values,
        )
        self.assertEqual(
            mock_nearest_neighbour_resample.call_args[1]["n_min_source"], 49
        )
        np.testing.assert_array_equal(
            mock_nearest_neighbour_resample.call_args[1]["mask_invalid"],
            True,
        )

        self.assertEqual(mock_nearest_neighbour_resample.call_count, 4)

        np.testing.assert_array_equal(test_proc_data.shape, (4, 8, 8))

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

        data_target = nearest_neighbour_resample(
            data_source,
            x_source,
            y_source,
            x_target,
            y_target,
            n_min_source=9,
        )[0]

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

        data_target = nearest_neighbour_resample(
            data_source,
            x_source,
            y_source,
            x_target,
            y_target,
            # n_min_source=9,
            mask_invalid=True,
        )[0]

        np.testing.assert_array_almost_equal(
            np.array(
                [
                    [1.0, 4.0, 7.0, 10.0, np.nan],
                    [1.0, 4.0, 7.0, 10.0, np.nan],
                    [1.0, 4.0, 7.0, 10.0, np.nan],
                    [1.0, 4.0, 7.0, 10.0, np.nan],
                    [1.0, 4.0, 7.0, 10.0, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
            data_target,
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

        data_target = nearest_neighbour_resample(
            data_source,
            x_source,
            y_source,
            x_target,
            y_target,
            n_min_source=9,
            mask_invalid=True,
        )[0]

        np.testing.assert_array_almost_equal(
            np.array(
                [
                    [1.0, 4.0, 7.0, np.nan, np.nan],
                    [1.0, 4.0, 7.0, np.nan, np.nan],
                    [1.0, 4.0, 7.0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
            data_target,
        )

    def test_nearest_neighbour_resample_std(self):
        data_source = np.vstack([np.arange(15) for i in range(18)])
        data_source[3, 3] = 15
        data_source[15, 12] = 11
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

        data_target, std_target = nearest_neighbour_resample(
            data_source,
            x_source,
            y_source,
            x_target,
            y_target,
            n_min_source=9,
        )

        np.testing.assert_array_almost_equal(
            np.array(
                [
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 5.333333333, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 13.0],
                    [1.0, 4.0, 7.0, 10.0, 12.8888888888],
                ]
            ),
            data_target,
        )

        np.testing.assert_array_almost_equal(
            np.array(
                [
                    [0.81649658, 0.81649658, 0.81649658, 0.81649658, 0.81649658],
                    [0.81649658, 3.49602949, 0.81649658, 0.81649658, 0.81649658],
                    [0.81649658, 0.81649658, 0.81649658, 0.81649658, 0.81649658],
                    [0.81649658, 0.81649658, 0.81649658, 0.81649658, 0.81649658],
                    [0.81649658, 0.81649658, 0.81649658, 0.81649658, 0.81649658],
                    [0.81649658, 0.81649658, 0.81649658, 0.81649658, 0.99380798],
                ]
            ),
            std_target,
        )


if __name__ == "__main__":
    unittest.main()
