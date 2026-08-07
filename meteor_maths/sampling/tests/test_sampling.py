"""
Tests for sampling module
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing
import xarray as xr

from meteor_maths.sampling import (
    RegridCache,
    Resampler,
    nearest_neighbour_resample,
    resample,
)

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
        _ = 9.969209968386869e36

        test_ds = xr.Dataset(
            data_vars={
                "data_3d_source": (
                    ["bands", "x_source", "y_source"],
                    np.ones(
                        (
                            size_dict["bands"],
                            size_dict["x_source"],
                            size_dict["y_source"],
                        )
                    ),
                ),
                "data_3d_target": (
                    ["bands", "x_target", "y_target"],
                    np.ones(
                        (
                            size_dict["bands"],
                            size_dict["x_target"],
                            size_dict["y_target"],
                        )
                    ),
                ),
                "data_2d_source": (
                    ["x_source", "y_source"],
                    np.ones(
                        (
                            size_dict["x_source"],
                            size_dict["y_source"],
                        )
                    ),
                ),
                "coord1_source": (
                    ["x_source", "y_source"],
                    np.ones((size_dict["x_source"], size_dict["y_source"])),
                ),
                "coord2_source": (
                    ["x_source", "y_source"],
                    2 * np.ones((size_dict["x_source"], size_dict["y_source"])),
                ),
                "coord1_target": (
                    ["x_target", "y_target"],
                    np.ones((size_dict["x_target"], size_dict["y_target"])),
                ),
                "coord2_target": (
                    ["x_target", "y_target"],
                    2 * np.ones((size_dict["x_target"], size_dict["y_target"])),
                ),
            },
        )

        self.test_ds = test_ds

    def test_resample_2d(self):
        mock_resampler_cls = MagicMock(name="ResamplerCls")
        mock_resampler = mock_resampler_cls.return_value
        mock_resampler.regrid.return_value = (
            np.ones(self.test_ds.coord1_target.shape),
            np.ones(self.test_ds.coord1_target.shape, dtype=bool),
        )

        # resample() has no knowledge of RegridCache specifically - it looks
        # up a resampler class by method name in _RESAMPLERS, builds one, and
        # only ever calls its .regrid() method. Swapping that registry entry
        # for a mock is how we verify that dispatch without depending on any
        # particular algorithm's internals.
        with patch.dict(
            "meteor_maths.sampling.sampling._RESAMPLERS",
            {"nearest_neighbour": mock_resampler_cls},
        ):
            test_proc_data = resample(
                "data_2d_source",
                self.test_ds,
                x_source=self.test_ds.coord1_source.values,
                y_source=self.test_ds.coord2_source.values,
                x_target=self.test_ds.coord1_target.values,
                y_target=self.test_ds.coord2_target.values,
                mask_invalid=False,
            )

        mock_resampler_cls.assert_called_once_with(
            self.test_ds.coord1_source.values,
            self.test_ds.coord2_source.values,
            self.test_ds.coord1_target.values,
            self.test_ds.coord2_target.values,
        )
        # data_2d_source's dims are ordered [x_source, y_source], so its 2D
        # values are transposed before being handed to the resampler (see
        # the "transpose_xy" comment in resample()). Compared via call_args
        # directly (rather than assert_called_once_with) since mock's array
        # equality check relies on object identity as a fast path, which a
        # freshly-computed .T view won't match even when equal.
        mock_resampler.regrid.assert_called_once()
        call_args, call_kwargs = mock_resampler.regrid.call_args
        np.testing.assert_array_equal(call_args[0], self.test_ds["data_2d_source"].values.T)
        self.assertEqual(call_kwargs, {"mask_invalid": False})

        np.testing.assert_array_equal(test_proc_data.shape, self.test_ds.coord1_target.shape)

    def test_resample_2d_same_grid(self):
        mock_resampler_cls = MagicMock(name="ResamplerCls")

        with patch.dict(
            "meteor_maths.sampling.sampling._RESAMPLERS",
            {"nearest_neighbour": mock_resampler_cls},
        ):
            test_proc_data = resample(
                "data_2d_source",
                self.test_ds,
                x_source=self.test_ds.coord1_source.values,
                y_source=self.test_ds.coord2_source.values,
                x_target=self.test_ds.coord1_source.values,
                y_target=self.test_ds.coord2_source.values,
                mask_invalid=False,
            )

        # source and target grid are identical, so no regridding should happen
        mock_resampler_cls.assert_not_called()

        np.testing.assert_array_equal(test_proc_data, self.test_ds["data_2d_source"])

    def test_resample_3d(self):
        mock_resampler_cls = MagicMock(name="ResamplerCls")
        mock_resampler = mock_resampler_cls.return_value
        mock_resampler.regrid.return_value = (
            np.ones((8, 8)),
            np.ones((8, 8), dtype=bool),
        )

        with patch.dict(
            "meteor_maths.sampling.sampling._RESAMPLERS",
            {"nearest_neighbour": mock_resampler_cls},
        ):
            test_proc_data = resample(
                "data_3d_source",
                self.test_ds,
                x_source=self.test_ds.coord1_source.values,
                y_source=self.test_ds.coord2_source.values,
                x_target=self.test_ds.coord1_target.values,
                y_target=self.test_ds.coord2_target.values,
                mask_invalid=False,
            )

        # the resampler should be built exactly once and reused across all bands
        mock_resampler_cls.assert_called_once_with(
            self.test_ds.coord1_source.values,
            self.test_ds.coord2_source.values,
            self.test_ds.coord1_target.values,
            self.test_ds.coord2_target.values,
        )
        self.assertEqual(mock_resampler.regrid.call_count, 4)
        for band_call in mock_resampler.regrid.call_args_list:
            # data_3d_source's dims are ordered [bands, x_source, y_source],
            # so each band's 2D slice is transposed before being handed to
            # the resampler (see the "transpose_xy" comment in resample())
            np.testing.assert_array_equal(band_call[0][0], self.test_ds["data_3d_source"].values[0].T)
            self.assertEqual(band_call[1]["mask_invalid"], False)

        np.testing.assert_array_equal(test_proc_data.shape, (4, 8, 8))

    def test_resample_3d_mask_invalidTrue(self):
        mock_resampler_cls = MagicMock(name="ResamplerCls")
        mock_resampler = mock_resampler_cls.return_value
        mock_resampler.regrid.return_value = (
            np.ones(self.test_ds.coord1_target.shape),
            np.ones(self.test_ds.coord1_target.shape, dtype=bool),
        )

        with patch.dict(
            "meteor_maths.sampling.sampling._RESAMPLERS",
            {"nearest_neighbour": mock_resampler_cls},
        ):
            test_proc_data = resample(
                "data_3d_source",
                self.test_ds,
                x_source=self.test_ds.coord1_source.values,
                y_source=self.test_ds.coord2_source.values,
                x_target=self.test_ds.coord1_target.values,
                y_target=self.test_ds.coord2_target.values,
                mask_invalid=True,
            )

        mock_resampler_cls.assert_called_once_with(
            self.test_ds.coord1_source.values,
            self.test_ds.coord2_source.values,
            self.test_ds.coord1_target.values,
            self.test_ds.coord2_target.values,
        )
        self.assertEqual(mock_resampler.regrid.call_count, 4)
        for band_call in mock_resampler.regrid.call_args_list:
            self.assertEqual(band_call[1]["mask_invalid"], True)

        np.testing.assert_array_equal(test_proc_data.shape, (4, 8, 8))

    def test_resample_unknown_method_raises(self):
        with self.assertRaises(NotImplementedError):
            resample(
                "data_2d_source",
                self.test_ds,
                x_source=self.test_ds.coord1_source.values,
                y_source=self.test_ds.coord2_source.values,
                x_target=self.test_ds.coord1_target.values,
                y_target=self.test_ds.coord2_target.values,
                method="bilinear",
            )

    def test_resample_unknown_method_raises_even_when_grids_match(self):
        """Regression test: method validation must happen even when the
        'source and target grid are identical, skip resampling' shortcut
        would otherwise return early - an invalid/misspelled method name
        must never be silently accepted just because it happened not to be
        needed for this particular call."""
        with self.assertRaises(NotImplementedError):
            resample(
                "data_2d_source",
                self.test_ds,
                x_source=self.test_ds.coord1_source.values,
                y_source=self.test_ds.coord2_source.values,
                x_target=self.test_ds.coord1_source.values,
                y_target=self.test_ds.coord2_source.values,
                method="bilinear",
            )

    def test_resample_dispatches_new_methods_via_registry(self):
        """The whole point of the registry: registering a new resampling
        backend (any class satisfying the Resampler protocol, i.e. having a
        matching .regrid() method) makes it available to resample() by name,
        with no changes to resample() itself - it never needs to know
        RegridCache exists, only that _RESAMPLERS[method] gives it something
        with a .regrid() method."""
        mock_resampler_cls = MagicMock(name="FakeInterpolationResamplerCls")
        mock_resampler = mock_resampler_cls.return_value
        mock_resampler.regrid.return_value = (
            np.ones(self.test_ds.coord1_target.shape),
            np.ones(self.test_ds.coord1_target.shape, dtype=bool),
        )

        with patch.dict(
            "meteor_maths.sampling.sampling._RESAMPLERS",
            {"fake_interpolation": mock_resampler_cls},
        ):
            resample(
                "data_2d_source",
                self.test_ds,
                x_source=self.test_ds.coord1_source.values,
                y_source=self.test_ds.coord2_source.values,
                x_target=self.test_ds.coord1_target.values,
                y_target=self.test_ds.coord2_target.values,
                method="fake_interpolation",
            )

        mock_resampler_cls.assert_called_once_with(
            self.test_ds.coord1_source.values,
            self.test_ds.coord2_source.values,
            self.test_ds.coord1_target.values,
            self.test_ds.coord2_target.values,
        )
        mock_resampler.regrid.assert_called_once()

    def test_resample_reuses_precomputed_resampler_across_calls(self):
        """A resampler (e.g. a RegridCache) built once for a pair of grids
        can be passed into resample() via ``resampler=`` for several
        variables without rebuilding the KDTree each time. This is also how
        to use non-default settings for a given algorithm (e.g.
        RegridCache(..., regular_grid=True)) - those aren't exposed as
        resample() arguments, precisely so resample() doesn't need to know
        about every algorithm's own knobs."""
        x_source, y_source = np.meshgrid(np.arange(15), np.arange(18))
        x_target = np.array([[1.0, 4.0, 7.0, 10.0, 13.0]] * 6)
        y_target = np.array([[1.0] * 5, [4.0] * 5, [7.0] * 5, [10.0] * 5, [13.0] * 5, [16.0] * 5])
        ds = xr.Dataset(
            data_vars={
                "var_a": (["y_source", "x_source"], np.vstack([np.arange(15)] * 18)),
                "var_b": (["y_source", "x_source"], np.vstack([np.arange(15)] * 18) * 2),
            }
        )

        cache = RegridCache(x_source, y_source, x_target, y_target)
        # guard against resample() falling back to building its own resampler
        # via the registry instead of using the one we passed in. Patching
        # RegridCache by name would be inert here (resample() looks classes
        # up through _RESAMPLERS, which already holds a direct reference to
        # the real class captured at import time - renaming the module-level
        # name afterwards doesn't change what's in the dict), so patch the
        # registry entry itself instead.
        with patch.dict(
            "meteor_maths.sampling.sampling._RESAMPLERS",
            {"nearest_neighbour": MagicMock(side_effect=AssertionError)},
        ):
            result_a = resample("var_a", ds, x_source, y_source, x_target, y_target, resampler=cache)
            result_b = resample("var_b", ds, x_source, y_source, x_target, y_target, resampler=cache)

        expected_a, _ = cache.regrid(ds["var_a"].values)
        expected_b, _ = cache.regrid(ds["var_b"].values)
        np.testing.assert_array_almost_equal(result_a, expected_a)
        np.testing.assert_array_almost_equal(result_b, expected_b)

    def test_resample_3d_real_data_with_x_before_y_dims(self):
        """End-to-end (no mocking) regression test for a variable whose dims
        are ordered [x, y] rather than [y, x]: x_source/y_source (like any
        numpy.meshgrid output) are shaped (y, x), so a variable's own 2D
        slices need transposing on the way in and back out to line up with
        that when its dims list x before y. Previously this either silently
        mislabelled the output axes (2D case, no error) or crashed with a
        broadcast shape mismatch (3D/4D case, since the pre-allocated output
        array follows the dims' own [x, y] order)."""
        x_source, y_source = np.meshgrid(np.arange(15), np.arange(18))
        x_target = np.array([[1.0, 4.0, 7.0, 10.0, 13.0]] * 6)
        y_target = np.array([[1.0] * 5, [4.0] * 5, [7.0] * 5, [10.0] * 5, [13.0] * 5, [16.0] * 5])

        # (18, 15), matching x_source/y_source's own (y, x) shape
        base = np.vstack([np.arange(15)] * 18).astype(float)
        # deliberately transposed to (15, 18), to match dims=[x_source, y_source]
        data_2d = base.T
        ds = xr.Dataset(
            data_vars={
                "var3d": (
                    ["bands", "x_source", "y_source"],
                    np.stack([data_2d, data_2d * 2]),
                ),
            }
        )

        result = resample("var3d", ds, x_source, y_source, x_target, y_target)

        # dims are [bands, x_source, y_source], so the output is (2, 5, 6):
        # x_target has 5 columns, y_target 6 rows
        self.assertEqual(result.shape, (2, 5, 6))
        expected_band0 = np.array([[1.0, 4.0, 7.0, 10.0, 13.0]] * 6).T
        np.testing.assert_array_almost_equal(result[0], expected_band0)
        np.testing.assert_array_almost_equal(result[1], expected_band0 * 2)

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
        # source grid (17 x 16) is not an exact multiple of the target grid
        # (6 x 5): the bottom row of the target grid only has 2 rows of source
        # pixels within its footprint (vs. 3 for every other row), so it falls
        # below the dynamically estimated n_min_source and is masked invalid.
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
            mask_invalid=True,
        )[0]

        np.testing.assert_array_almost_equal(
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
            data_target,
        )

    def test_nearest_neighbour_resample_to_larger_grid(self):
        # target grid extends beyond the source grid in both x (18, 20) and y
        # (18, 20): pixels there have no (or too few) source pixels within
        # their footprint and are masked invalid, rather than absorbing
        # whatever nearby source pixels happen to be nearest.
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
            mask_invalid=True,
        )[0]

        np.testing.assert_array_almost_equal(
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


class TestRegridCache(unittest.TestCase):
    """
    Tests for RegridCache and its regrid() method: the cached counterpart to
    nearest_neighbour_resample, for resampling several data arrays on the
    same source/target grids without rebuilding the KDTree each time.
    """

    def test_regrid_cache_satisfies_resampler_protocol(self):
        """RegridCache must structurally match the Resampler protocol (i.e.
        have a matching regrid() method) for resample()'s registry-based
        dispatch to be able to use it like any other resampling backend."""
        x_source, y_source = np.meshgrid(np.arange(15), np.arange(18))
        x_target = np.array([[1.0, 4.0, 7.0, 10.0, 13.0]] * 6)
        y_target = np.array([[1.0] * 5, [4.0] * 5, [7.0] * 5, [10.0] * 5, [13.0] * 5, [16.0] * 5])
        cache = RegridCache(x_source, y_source, x_target, y_target)
        self.assertIsInstance(cache, Resampler)

    def test_regrid_with_cache_matches_nearest_neighbour_resample(self):
        data_source = np.vstack([np.arange(15) for i in range(18)])
        x_source, y_source = np.meshgrid(np.arange(15), np.arange(18))
        x_target = np.array([[1.0, 4.0, 7.0, 10.0, 13.0]] * 6)
        y_target = np.array([[1.0] * 5, [4.0] * 5, [7.0] * 5, [10.0] * 5, [13.0] * 5, [16.0] * 5])

        cache = RegridCache(x_source, y_source, x_target, y_target)
        data_target, mask = cache.regrid(data_source)

        expected, _ = nearest_neighbour_resample(data_source, x_source, y_source, x_target, y_target)
        np.testing.assert_array_almost_equal(data_target, expected)
        self.assertTrue(np.all(mask))

    def test_regrid_with_cache_reused_across_multiple_fields(self):
        """Building one RegridCache and reusing it for several data arrays
        gives the same result as resampling each independently, without
        rebuilding the KDTree for every field."""
        x_source, y_source = np.meshgrid(np.arange(15), np.arange(18))
        x_target = np.array([[1.0, 4.0, 7.0, 10.0, 13.0]] * 6)
        y_target = np.array([[1.0] * 5, [4.0] * 5, [7.0] * 5, [10.0] * 5, [13.0] * 5, [16.0] * 5])
        rng = np.random.default_rng(0)
        fields = [rng.random(x_source.shape) for _ in range(3)]

        cache = RegridCache(x_source, y_source, x_target, y_target)
        # n_min_source is computed lazily on first access (it needs its own
        # KDTree query); force that here so the patch below only guards
        # against *further* KDTree construction while reusing the cache
        _ = cache.n_min_source

        # the cache's trees were already built above; regridding the three
        # fields below must not need to construct any new ones
        with patch("scipy.spatial.cKDTree", side_effect=AssertionError):
            results = [cache.regrid(field)[0] for field in fields]

        for field, result in zip(fields, results):
            expected, _ = nearest_neighbour_resample(field, x_source, y_source, x_target, y_target)
            np.testing.assert_array_almost_equal(result, expected)

    def test_regrid_with_cache_nans_excluded_from_average_not_propagated(self):
        """A NaN source pixel is excluded from the average of the bin it
        falls in (rather than poisoning the whole bin's arithmetic mean via
        NaN propagation), and doesn't count towards that bin's contributing-
        pixel count. Uses the same locally-dense source grid as
        test_regrid_with_cache_locally_varying_source_density, where the
        left-hand bins have a comfortable natural margin (~25 contributing
        pixels against a threshold of 16) - removing just one of them still
        leaves the bin valid, averaged over the remaining ones."""
        x_dense = np.arange(0, 10, 0.5)
        x_sparse = np.arange(10, 20, 1.0)
        x_row = np.concatenate([x_dense, x_sparse])
        y_col = np.arange(0, 10, 0.5)
        x_source, y_source = np.meshgrid(x_row, y_col)
        # source value equal to its own x-coordinate, so the effect of
        # removing a specific pixel on the bin average is easy to compute
        data_source = x_source.copy()

        x_target, y_target = np.meshgrid(np.arange(1, 19, 2.0), np.arange(1, 9, 2.0))
        cache = RegridCache(x_source, y_source, x_target, y_target)

        removed = np.zeros_like(x_source, dtype=bool)
        removed[0, 0] = True  # a single pixel (x=0.0, y=0.0) within bin 0
        data_source_with_nan = np.where(removed, np.nan, data_source)

        data_target, mask = cache.regrid(data_source_with_nan)

        data_in_range = data_source.ravel()[cache.source_in_range]
        removed_in_range = removed.ravel()[cache.source_in_range]
        bin0_values = data_in_range[cache.idx == 0]
        bin0_removed = removed_in_range[cache.idx == 0]
        self.assertGreater(bin0_removed.sum(), 0)  # sanity: the pixel is in range
        expected_bin0 = bin0_values[~bin0_removed].mean()

        self.assertTrue(mask.ravel()[0])
        self.assertAlmostEqual(data_target.ravel()[0], expected_bin0, places=6)

    def test_regrid_with_cache_random_nans_discard_undersampled_bins(self):
        """With NaNs scattered randomly throughout the source data (rather
        than a hand-placed one or two), some target pixels - including
        interior ones, not just edge pixels - end up with too few non-NaN
        contributing source pixels and should be discarded, while
        sufficiently-sampled pixels remain valid and are averaged only from
        their non-NaN contributors. Checked as a property against the cache's
        own source -> target mapping, rather than hand-computed expected
        values, so it isn't sensitive to the exact estimator used."""
        rng = np.random.default_rng(42)
        x_source, y_source = np.meshgrid(np.arange(30), np.arange(30))
        data_source = rng.random(x_source.shape)
        data_source[rng.random(x_source.shape) < 0.3] = np.nan

        x_target, y_target = np.meshgrid(np.arange(1, 29, 3.0), np.arange(1, 29, 3.0))
        cache = RegridCache(x_source, y_source, x_target, y_target)
        data_target, mask = cache.regrid(data_source)

        data_in_range = data_source.ravel()[cache.source_in_range]
        n_min_source = np.broadcast_to(cache.n_min_source, mask.size)
        data_target_flat = data_target.ravel()
        mask_flat = mask.ravel()
        for bin_idx in range(mask_flat.size):
            values = data_in_range[cache.idx == bin_idx]
            valid_values = values[~np.isnan(values)]
            expect_valid = valid_values.size >= n_min_source[bin_idx]
            self.assertEqual(mask_flat[bin_idx], expect_valid, f"bin {bin_idx}")
            if expect_valid:
                self.assertAlmostEqual(data_target_flat[bin_idx], valid_values.mean(), places=6)
            else:
                self.assertTrue(np.isnan(data_target_flat[bin_idx]))

        # with 30% of source pixels randomly missing, this random seed should
        # produce a genuine mix of surviving and discarded target pixels,
        # including some away from the grid edges (not just edge effects)
        self.assertTrue(np.any(mask))
        self.assertTrue(np.any(~mask))
        interior = mask[1:-1, 1:-1]
        self.assertTrue(np.any(~interior))

    def test_regrid_with_cache_irregular_grid_shape(self):
        # a non-square target shape (4 rows x 6 columns) resampled from a
        # non-square source (12 x 24), each axis with its own genuinely
        # independent spacing - not just a small hand-written grid, so the
        # automatic local estimate has enough neighbours on each axis to work
        # with (a target axis with only 2-3 points doesn't give the local
        # estimator enough information to distinguish "coarse but complete
        # coverage" from "genuinely under-sampled", so is deliberately
        # avoided here)
        x_src, y_src = np.meshgrid(np.arange(24), np.arange(12))
        data = (x_src + y_src * 100).astype(float)
        x_tgt, y_tgt = np.meshgrid(np.linspace(1, 22, 6), np.linspace(1, 10, 4))

        cache = RegridCache(x_src, y_src, x_tgt, y_tgt)
        data_target, mask = cache.regrid(data)

        self.assertEqual(data_target.shape, x_tgt.shape)
        self.assertEqual(mask.shape, x_tgt.shape)
        self.assertTrue(np.all(mask))
        np.testing.assert_array_almost_equal(
            data_target,
            np.array(
                [
                    [101.5, 105.5, 109.5, 113.5, 117.5, 121.5],
                    [401.5, 405.5, 409.5, 413.5, 417.5, 421.5],
                    [701.5, 705.5, 709.5, 713.5, 717.5, 721.5],
                    [1001.5, 1005.5, 1009.5, 1013.5, 1017.5, 1021.5],
                ]
            ),
        )

    def test_regrid_with_cache_to_larger_grid_dynamic_n_min_source(self):
        # target grid extends beyond the source grid in x only; the dynamic
        # n_min_source estimate should still correctly flag the out-of-range
        # column as invalid without it being passed explicitly
        data_source = np.vstack([np.arange(16) for i in range(17)])
        x_source, y_source = np.meshgrid(np.arange(16), np.arange(17))
        x_target = np.array([[1.0, 4.0, 7.0, 10.0, 18.0]] * 6)
        y_target = np.array([[1.0] * 5, [4.0] * 5, [7.0] * 5, [10.0] * 5, [13.0] * 5, [16.0] * 5])

        cache = RegridCache(x_source, y_source, x_target, y_target)
        data_target, mask = cache.regrid(data_source)

        np.testing.assert_array_almost_equal(
            data_target,
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
        )
        np.testing.assert_array_equal(mask, ~np.isnan(data_target))

    @staticmethod
    def _rotate(x, y, degrees):
        theta = np.deg2rad(degrees)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        return cos_t * x - sin_t * y, sin_t * x + cos_t * y

    def _assert_mask_matches_source_in_range_counts(self, cache, data, mask):
        """Property check used throughout the rotation tests below: a target
        pixel's validity must always agree with the cache's own record of how
        many non-NaN source pixels actually fall in its footprint (idx /
        source_in_range / n_min_source) - independent of geometry, rotation,
        or which estimator produced n_min_source. This is what "the invalid
        sample masking is robust" cashes out to concretely: not that any
        particular rotation happens to look reasonable, but that the mask
        always, exactly, matches this definition."""
        data_in_range = data.ravel()[cache.source_in_range]
        n_min_source = np.broadcast_to(cache.n_min_source, mask.size)
        mask_flat = mask.ravel()
        for bin_idx in range(mask_flat.size):
            values = data_in_range[cache.idx == bin_idx]
            valid_values = values[~np.isnan(values)]
            expect_valid = valid_values.size >= n_min_source[bin_idx]
            self.assertEqual(bool(mask_flat[bin_idx]), bool(expect_valid), f"bin {bin_idx}")

    def test_regrid_with_cache_co_rotated_grids_match_unrotated_baseline(self):
        """Rotating source and target together is just a coordinate change -
        it doesn't affect which source pixels are nearest to which target
        pixels, or how far away they are. The footprint/validity estimate is
        built from each target pixel's own local (rotation-following)
        neighbour directions rather than the global x/y axes, so this should
        give exactly the same result - including which pixels are masked
        invalid - as not rotating at all, at any rotation angle.

        Uses a source/target pair (17 x 16 source, 6 x 5 target) that isn't
        an exact multiple, so the *unrotated* baseline already has a genuinely
        under-sampled row that gets masked invalid (see
        test_nearest_neighbour_resample_invalid_edge_pixels) - checking that
        this masking survives rotation intact is the actual point here, not
        just that a trivially all-valid grid doesn't get scrambled."""
        data_source = np.vstack([np.arange(16) for i in range(17)])
        x_source, y_source = np.meshgrid(np.arange(16), np.arange(17))
        x_target = np.array([[1.0, 4.0, 7.0, 10.0, 13.0]] * 6)
        y_target = np.array([[1.0] * 5, [4.0] * 5, [7.0] * 5, [10.0] * 5, [13.0] * 5, [16.0] * 5])

        cache_base = RegridCache(x_source, y_source, x_target, y_target)
        data_base, mask_base = cache_base.regrid(data_source)
        # sanity: the baseline actually has a mix of valid and invalid pixels
        # (the last row) - otherwise this test wouldn't be exercising masking
        self.assertTrue(np.any(mask_base))
        self.assertTrue(np.any(~mask_base))

        for degrees in [0, 1e-6, 15, 45, 89.9999, 90, 123.456, -37, 180, 270, 359]:
            with self.subTest(degrees=degrees):
                x_source_rot, y_source_rot = self._rotate(x_source, y_source, degrees)
                x_target_rot, y_target_rot = self._rotate(x_target, y_target, degrees)

                cache_rot = RegridCache(x_source_rot, y_source_rot, x_target_rot, y_target_rot)
                data_rot, mask_rot = cache_rot.regrid(data_source)

                np.testing.assert_array_equal(mask_rot, mask_base)
                np.testing.assert_array_almost_equal(data_rot, data_base)

    def test_regrid_with_cache_co_rotated_grids_with_random_nans_match_baseline(self):
        """As above, but with a data field that has randomly scattered NaNs
        and a source/target grid with zero natural margin (n_min_source
        exactly equal to the true, noise-free expected count everywhere) -
        the scenario most likely to expose any rotation-induced floating
        point drift in the expected-count estimate tipping a pixel across
        its threshold. (This is a regression test: earlier versions of the
        local estimator could compute e.g. 8.999999999999996 instead of 9.0
        for a rotated grid whose true ratio is exactly 9, and floor() would
        silently round that down to 8, disagreeing with the unrotated case.)
        """
        rng = np.random.default_rng(7)
        x_source, y_source = np.meshgrid(np.arange(30), np.arange(30))
        data_source = rng.random(x_source.shape)
        data_source[rng.random(x_source.shape) < 0.3] = np.nan
        x_target, y_target = np.meshgrid(np.arange(1, 29, 3.0), np.arange(1, 29, 3.0))

        cache_base = RegridCache(x_source, y_source, x_target, y_target)
        data_base, mask_base = cache_base.regrid(data_source)
        self.assertTrue(np.any(mask_base))
        self.assertTrue(np.any(~mask_base))

        for degrees in [15, 40, 90, 123.456, -37, 200]:
            with self.subTest(degrees=degrees):
                x_source_rot, y_source_rot = self._rotate(x_source, y_source, degrees)
                x_target_rot, y_target_rot = self._rotate(x_target, y_target, degrees)

                cache_rot = RegridCache(x_source_rot, y_source_rot, x_target_rot, y_target_rot)
                data_rot, mask_rot = cache_rot.regrid(data_source)

                np.testing.assert_array_equal(mask_rot, mask_base)
                np.testing.assert_array_almost_equal(data_rot, data_base)

    def _assert_independently_rotated_swaths_are_frame_invariant(
        self,
        x_source,
        y_source,
        data_source,
        x_target,
        y_target,
        relative_angle,
        source_angles,
    ):
        """Shared property check for the realistic version of the
        misalignment scenario: two source images each with their own,
        independent, non-zero rotation (e.g. two satellite swaths on
        different overpasses, neither aligned to lat/lon), rather than one
        grid conveniently fixed at 0 degrees. What actually determines the
        masking is only the *relative* angle between the two grids, not
        their shared absolute orientation - rotating both by some extra
        common amount is just a change of reference frame (see the
        co_rotated_grids tests above) - so resampling should give identical
        results (same masking, same values) regardless of which absolute
        angle pair happens to produce that relative angle. Checked directly:
        several unrelated absolute-angle pairs sharing one relative angle
        must all agree, including with realistic NaN dropout, and each must
        independently satisfy the usual mask/count consistency property.
        Also cross-checks against the fixed-source-at-0 case (as tested in
        test_regrid_with_cache_target_rotated_relative_to_source), which this
        reduces to when the source's own angle happens to be 0. Returns the
        reference mask, for scenario-specific assertions by the caller."""
        x_target_ref, y_target_ref = self._rotate(x_target, y_target, relative_angle)
        cache_ref = RegridCache(x_source, y_source, x_target_ref, y_target_ref)
        data_ref, mask_ref = cache_ref.regrid(data_source)
        # sanity: this scenario has a genuine mix of valid/invalid pixels,
        # so the comparisons below are actually exercising the masking
        self.assertTrue(np.any(mask_ref))
        self.assertTrue(np.any(~mask_ref))

        for source_angle in source_angles:
            with self.subTest(source_angle=source_angle):
                # "two satellites": source swath at its own absolute angle,
                # target swath at a *different* absolute angle - only their
                # difference (relative_angle) is held fixed across cases
                x_source_i, y_source_i = self._rotate(x_source, y_source, source_angle)
                x_target_i, y_target_i = self._rotate(x_target, y_target, source_angle + relative_angle)

                cache = RegridCache(x_source_i, y_source_i, x_target_i, y_target_i)
                data_target, mask = cache.regrid(data_source)

                self._assert_mask_matches_source_in_range_counts(cache, data_source, mask)
                np.testing.assert_array_equal(mask, mask_ref)
                np.testing.assert_array_almost_equal(data_target, data_ref)

        return mask_ref

    def test_regrid_with_cache_two_independently_rotated_swaths_target_coarser(self):
        """Downsampling case: target grid coarser than source (spacing 3 vs
        1) - the common "average many source pixels into one target pixel"
        scenario used throughout this file, here combined with two
        independently, differently rotated swaths rather than a convenient
        shared or zero rotation."""
        rng = np.random.default_rng(3)
        x_source, y_source = np.meshgrid(np.arange(30), np.arange(30))
        data_source = rng.random(x_source.shape)
        data_source[rng.random(x_source.shape) < 0.2] = np.nan
        x_target, y_target = np.meshgrid(np.arange(1, 29, 3.0), np.arange(1, 29, 3.0))

        self._assert_independently_rotated_swaths_are_frame_invariant(
            x_source,
            y_source,
            data_source,
            x_target,
            y_target,
            relative_angle=22.0,
            source_angles=[0.0, 17.0, -63.0, 310.0],
        )

    def test_regrid_with_cache_two_independently_rotated_swaths_source_coarser(self):
        """Upsampling case: source grid coarser than target (spacing 3 vs 1)
        - the reverse of the usual scenario. Each target pixel's footprint is
        then much smaller than the gap between source pixels, so only a
        small, sparse fraction of target pixels (those that happen to land
        close to an actual source point) get a value at all; the rest
        legitimately have no nearby source data and should be masked
        invalid, rather than something being interpolated/guessed at -
        nearest-neighbour resampling is not an interpolation method. Checked
        together with two independently, differently rotated swaths, as
        above."""
        rng = np.random.default_rng(11)
        x_source, y_source = np.meshgrid(np.arange(0, 18, 3), np.arange(0, 18, 3))
        data_source = rng.random(x_source.shape)
        data_source[rng.random(x_source.shape) < 0.3] = np.nan
        x_target, y_target = np.meshgrid(np.arange(0, 17, 1.0), np.arange(0, 17, 1.0))

        mask_ref = self._assert_independently_rotated_swaths_are_frame_invariant(
            x_source,
            y_source,
            data_source,
            x_target,
            y_target,
            relative_angle=22.0,
            source_angles=[0.0, 15.0, -40.0, 200.0],
        )
        # upsampling: only a small fraction of the much finer target grid
        # should end up with any nearby source data at all
        self.assertLess(mask_ref.mean(), 0.2)

    def test_regrid_with_cache_target_rotated_relative_to_source(self):
        """When the target grid is rotated relative to an axis-aligned source
        grid (rather than both sharing the same rotation), the two grids are
        genuinely misaligned: a rotated rectangular footprint doesn't tile
        as cleanly against a fixed axis-aligned point lattice, so more edge
        pixels are under-sampled and discarded than in the unrotated case.
        That's expected, not a bug - what must still hold, and is checked
        here at several angles, is that the mask stays exactly consistent
        with the cache's own counting rule (see
        _assert_mask_matches_source_in_range_counts), and that
        resampling degrades gracefully (more masking at larger misalignment)
        rather than e.g. masking everything or nothing regardless of angle."""
        data_source = np.vstack([np.arange(16) for _ in range(17)])
        x_source, y_source = np.meshgrid(np.arange(16), np.arange(17))
        x_target, y_target = np.meshgrid([1.0, 4.0, 7.0, 10.0, 13.0], [1.0, 4.0, 7.0, 10.0, 13.0, 16.0])

        valid_fractions = {}
        for degrees in [0, 5, 10, 20, 45, 60]:
            with self.subTest(degrees=degrees):
                x_rot, y_rot = self._rotate(x_target, y_target, degrees)

                cache = RegridCache(x_source, y_source, x_rot, y_rot)
                data_target, mask = cache.regrid(data_source)

                self.assertEqual(data_target.shape, x_rot.shape)
                self.assertEqual(mask.shape, x_rot.shape)
                np.testing.assert_array_equal(np.isnan(data_target), ~mask)
                if np.any(mask):
                    valid = data_target[mask]
                    self.assertTrue(np.all(valid >= data_source.min()))
                    self.assertTrue(np.all(valid <= data_source.max()))

                self._assert_mask_matches_source_in_range_counts(cache, data_source, mask)

                valid_fractions[degrees] = mask.mean()

        # increasing misalignment should degrade coverage monotonically (or
        # at least not get better) - a real symptom of broken masking would
        # be validity that doesn't track the actual geometric misalignment
        degrees_sorted = sorted(valid_fractions)
        fractions_in_order = [valid_fractions[d] for d in degrees_sorted]
        self.assertEqual(fractions_in_order, sorted(fractions_in_order, reverse=True))

    def test_regrid_with_cache_locally_varying_source_density(self):
        """The source grid is twice as dense in its left half as its right
        half (e.g. a swath whose resolution degrades off-nadir), resampled
        onto a uniform target grid. A single grid-wide expected-count
        estimate is calibrated to one density and systematically masks out
        the other (verified below against regular_grid=True, which does
        exactly that); estimating locally per target pixel instead should
        treat both halves as comparably, adequately sampled - some pixels
        near bin boundaries will still legitimately fall just short by
        chance (the local estimate isn't a perfect predictor, just a locally
        appropriate one), so this checks the two halves are treated
        even-handedly rather than requiring every single pixel to pass."""
        x_dense = np.arange(0, 10, 0.5)
        x_sparse = np.arange(10, 20, 1.0)
        x_row = np.concatenate([x_dense, x_sparse])
        y_col = np.arange(0, 10, 0.5)
        x_source, y_source = np.meshgrid(x_row, y_col)
        data_source = np.ones_like(x_source)

        x_target, y_target = np.meshgrid(np.arange(1, 19, 2.0), np.arange(1, 9, 2.0))
        cache = RegridCache(x_source, y_source, x_target, y_target)
        data_target, mask = cache.regrid(data_source)

        n_cols = x_target.shape[1]
        dense_half, sparse_half = mask[:, : n_cols // 2], mask[:, n_cols // 2 :]
        self.assertGreater(dense_half.mean(), 0.5)
        self.assertGreater(sparse_half.mean(), 0.5)
        # every valid target pixel overlaps real, uniform-valued source data
        np.testing.assert_array_almost_equal(data_target[mask], np.ones(np.sum(mask)))

        # the cheap regular_grid=True path uses one grid-wide density and so
        # is calibrated to whichever half dominates the grid-wide estimate -
        # here that miscalibrates it badly enough to reject an entire half
        cache_regular = RegridCache(x_source, y_source, x_target, y_target, regular_grid=True)
        _, mask_regular = cache_regular.regrid(data_source)
        sparse_half_regular = mask_regular[:, n_cols // 2 :]
        self.assertLess(sparse_half_regular.mean(), 0.5)

    def test_regrid_with_cache_regular_grid_fast_path_matches_default(self):
        """On a genuinely regular grid, the cheap regular_grid=True path
        (one grid-wide spacing estimate) should agree with the default local
        per-pixel estimate - they're only expected to diverge on grids that
        aren't actually regular."""
        data_source = np.vstack([np.arange(15) for i in range(18)])
        x_source, y_source = np.meshgrid(np.arange(15), np.arange(18))
        x_target = np.array([[1.0, 4.0, 7.0, 10.0, 13.0]] * 6)
        y_target = np.array([[1.0] * 5, [4.0] * 5, [7.0] * 5, [10.0] * 5, [13.0] * 5, [16.0] * 5])

        cache_local = RegridCache(x_source, y_source, x_target, y_target)
        data_local, mask_local = cache_local.regrid(data_source)

        cache_regular = RegridCache(x_source, y_source, x_target, y_target, regular_grid=True)
        data_regular, mask_regular = cache_regular.regrid(data_source)

        np.testing.assert_array_almost_equal(data_local, data_regular)
        np.testing.assert_array_equal(mask_local, mask_regular)

    def test_regrid_with_cache_degenerate_source_grid(self):
        """Regression test: a source grid with only one distinct coordinate
        along an axis (e.g. a single-column north-south transect) used to
        either silently discard all data (regular_grid=False: local spacing
        along the degenerate axis was treated as exactly zero, which floored
        the local source pixel area to ~0 and made the expected-count ratio
        explode to billions, so n_min_source could never be met - see
        test_regrid_with_cache_locally_varying_source_density's sibling bug
        report) or crash outright (regular_grid=True: _grid_spacing's
        np.median(np.diff(np.unique(...))) is NaN for a single-valued axis,
        and np.floor(nan) raises ValueError when cast to int). Both modes
        should instead treat the degenerate axis as carrying no density
        information, and still resample the one target column that
        genuinely overlaps the source transect."""
        rng = np.random.default_rng(0)
        x_source, y_source = np.meshgrid(np.array([5.0]), np.arange(10))
        data_source = rng.random(x_source.shape)
        x_target, y_target = np.meshgrid(np.array([4.0, 5.0, 6.0]), np.arange(0, 10, 2.0))

        for regular_grid in [False, True]:
            with self.subTest(regular_grid=regular_grid):
                cache = RegridCache(x_source, y_source, x_target, y_target, regular_grid=regular_grid)
                # a sane (small, finite) threshold - not the ~2e12 the
                # unfixed regular_grid=False estimator produced here
                self.assertTrue(np.all(np.asarray(cache.n_min_source) <= 10))

                data_target, mask = cache.regrid(data_source)
                # only the middle target column (x=5) actually overlaps the
                # source transect (also at x=5) - the outer columns (x=4,
                # x=6) genuinely have no nearby source data and should stay
                # masked invalid, but the middle one must now be valid
                self.assertTrue(np.all(mask[:, 1]))
                self.assertFalse(np.any(mask[:, [0, 2]]))
                self.assertFalse(np.any(np.isnan(data_target[:, 1])))

    def test_regrid_with_cache_degenerate_target_grid(self):
        """Companion to the degenerate-source test above: a target grid with
        only one distinct coordinate along an axis (e.g. resampling onto a
        transect) must also not crash or spuriously discard data, for both
        regular_grid modes."""
        rng = np.random.default_rng(1)
        x_source, y_source = np.meshgrid(np.arange(0, 10, 1.0), np.arange(0, 10, 1.0))
        data_source = rng.random(x_source.shape)
        x_target, y_target = np.meshgrid(np.array([5.0]), np.arange(0, 10, 2.0))

        for regular_grid in [False, True]:
            with self.subTest(regular_grid=regular_grid):
                cache = RegridCache(x_source, y_source, x_target, y_target, regular_grid=regular_grid)
                data_target, mask = cache.regrid(data_source)
                self.assertTrue(np.all(mask))
                self.assertFalse(np.any(np.isnan(data_target)))

    def test_regrid_with_cache_both_grids_degenerate_on_same_axis(self):
        """Edge case: source and target both single-column (e.g. two
        transects being compared) - neither grid has any information about
        spacing along that axis, so it should be treated as fully neutral
        rather than raising or dividing by zero."""
        x_source, y_source = np.meshgrid(np.array([5.0]), np.arange(10))
        data_source = np.random.default_rng(2).random(x_source.shape)
        x_target, y_target = np.meshgrid(np.array([5.0]), np.arange(0, 10, 2.0))

        for regular_grid in [False, True]:
            with self.subTest(regular_grid=regular_grid):
                cache = RegridCache(x_source, y_source, x_target, y_target, regular_grid=regular_grid)
                data_target, mask = cache.regrid(data_source)
                self.assertTrue(np.all(mask))
                self.assertFalse(np.any(np.isnan(data_target)))

    def test_regrid_with_cache_duplicated_source_row_does_not_corrupt_threshold(self):
        """Regression test: two array-adjacent source points that merely
        happen to coincide (e.g. a duplicated scanline in real satellite
        geolocation data) is not the same as a *fully* degenerate axis - the
        rest of the axis has perfectly normal spacing - but it used to slip
        past the degenerate-axis guard: _local_axis_geometry returned an
        exact 0.0 local spacing for the duplicated pixel, which
        _expected_source_count then divided by, producing inf (or, if both
        sides of the ratio were affected, nan). Casting that to int is
        undefined in numpy: it silently wrapped to either 9223372036854775807
        (permanently invalid, no count could ever satisfy it) or 0
        (permanently "valid" regardless of data quality) - with only a
        suppressed RuntimeWarning, no visible error. Both directions should
        now be impossible: n_min_source must stay a small, sane integer
        everywhere, and a target region with genuinely uniform, complete
        coverage must resample cleanly despite the nearby duplicate."""
        x_row = np.arange(20).astype(float)
        y_col = np.arange(20).astype(float)
        y_col[10] = y_col[9]  # rows 9 and 10 now coincide
        x_source, y_source = np.meshgrid(x_row, y_col)
        data_source = np.ones_like(x_source)
        x_target, y_target = np.meshgrid(np.arange(1, 19, 3.0), np.arange(1, 19, 3.0))

        cache = RegridCache(x_source, y_source, x_target, y_target)
        self.assertTrue(np.all(np.asarray(cache.n_min_source) < 1000))

        data_target, mask = cache.regrid(data_source)
        self.assertTrue(np.all(mask))
        np.testing.assert_array_almost_equal(data_target, np.ones_like(data_target))


if __name__ == "__main__":
    unittest.main()
