"""Module of resampling functions"""

import re
from typing import Tuple

import numpy as np
import xarray as xr
from scipy import spatial


__author__ = ["Maddie Stedman"]
__all__ = ["nearest_neighbour_resample", "resample"]


def resample(
    var: str,
    ds: xr.Dataset,
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    mask_invalid: bool = True,
    method: str = "nearest_neighbour",
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Resample variable data

    :param var: variable to resample
    :param ds: dataset containing data to resample
    :param x_source: x coordinates for source data
    :param y_source: y coordinates for source data
    :param x_target: x coordinates for target data
    :param y_target: y coordinates for target data
    :param mask_invalid: boolean for setting invalid edge pixels to nan
    :param method: resampling method to use
    :return: array of resampled variable data
    """

    # if source and target grid are the same, no resampling is necessary
    if x_source.shape == x_target.shape:
        if np.all(x_source == x_target) & np.all(y_source == y_target):
            return ds[var].values
    else:
        # create empty array for the processed data
        shape = list(ds[var].values.shape)
        shape[
            ds[var].dims.index(
                [dim for dim in ds[var].dims if re.search("^x_", dim)][0]
            )
        ] = x_target.shape[-1]
        shape[
            ds[var].dims.index(
                [dim for dim in ds[var].dims if re.search("^y_", dim)][0]
            )
        ] = x_target.shape[-2]
        data_intxy = np.zeros(shape)
        std_intxy = np.zeros(shape)

        # Following section is commented out as it's not used in the current resampling method but will be used in the required update of the resampling method.
        # to optimise the calculation, find the smallest rectangle in the source grid indices, that completely covers the target area.
        # min_id0, max_id0, min_id1, max_id1 = self.find_bounding_indices(
        #     lats_source,
        #     lons_source,
        #     lats_target,
        #     lons_target,
        # )
        #
        # lons_source = lons_source[min_id0:max_id0, min_id1:max_id1]
        # lats_source = lats_source[min_id0:max_id0, min_id1:max_id1]
        if method.lower() == "nearest_neighbour":
            f_resample = nearest_neighbour_resample
        else:
            raise NotImplementedError(f"Method {method} not implemented.")

        if data_intxy.ndim == 2:
            data_intxy = f_resample(
                ds[var].values,
                x_source,
                y_source,
                x_target,
                y_target,
                mask_invalid=mask_invalid,
                *args,
                **kwargs,
            )[0]
        elif data_intxy.ndim == 3:
            for i, val in enumerate(ds[ds[var].dims[0]].values):
                data_intxy[i] = f_resample(
                    ds[var].values[i],
                    x_source,
                    y_source,
                    x_target,
                    y_target,
                    mask_invalid=mask_invalid,
                    *args,
                    **kwargs,
                )[0]
        elif data_intxy.ndim == 4:
            for i, val in enumerate(ds[ds[var].dims[0]].values):
                for j, val in enumerate(ds[ds[var].dims[1]].values):
                    data_intxy[i, j] = f_resample(
                        ds[var].values[i, j],
                        x_source,
                        y_source,
                        x_target,
                        y_target,
                        mask_invalid=mask_invalid,
                        *args,
                        **kwargs,
                    )[0]
        else:
            raise NotImplementedError(
                f"Resampling not implemented for {data_intxy.ndim} dims. Data must have 2, 3 or 4 dims."
            )

    return data_intxy


def nearest_neighbour_resample(
    data: np.ndarray,
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    n_min_source: int = None,
    percentile_threshold: float = 50.0,
    mask_invalid: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample 2D data by averaging nearest neighbour values. Invalid pixels set to nan if mask=True - invalid pixels defined as those with < n_min_source pixels being binned to form the sample.

    :param data: data as 2D array
    :param x_source: x coordinates of source grid
    :param y_source: y coordinates of source grid
    :param x_target: x coordinates of target grid
    :param y_target: y coordinates of target grid
    :param n_min_source: minimum number of source pixels to be binned to target pixel to qualify as valid target pixel
    :param mask_invalid: boolean for setting invalid edge pixels to nan
    :return: resampled 2D data, standard deviation of samples
    """
    # Create grid of source and target coordinates
    grid_source = np.c_[x_source.ravel(), y_source.ravel()]
    grid_target = np.c_[x_target.ravel(), y_target.ravel()]

    if ((x_source.shape[0] % x_target.shape[0]) > 0) and (
        x_source.shape[0] % x_target.shape[0]
    ) <= n_min_source / 2:
        data = data[: -(x_source.shape[0] % x_target.shape[0]), :]
        x_source = x_source[: -(x_source.shape[0] % x_target.shape[0]), :]
        y_source = y_source[: -(x_source.shape[0] % x_target.shape[0]), :]
    if ((x_source.shape[1] % x_target.shape[1]) > 0) and (
        x_source.shape[1] % x_target.shape[1]
    ) <= n_min_source / 2:
        data = data[:, : -(x_source.shape[1] % x_target.shape[1])]
        x_source = x_source[:, : -(x_source.shape[1] % x_target.shape[1])]
        y_source = y_source[:, : -(x_source.shape[1] % x_target.shape[1])]
    # Create KDTree for target grid
    tree = spatial.cKDTree(grid_target)

    # Find nearest neighbour for each source grid point
    dist, idx = tree.query(grid_source, k=1)

    # Create empty array of target grid values
    sum_target = np.zeros(x_target.ravel().shape)
    std_target = np.zeros(x_target.ravel().shape)

    n_data_target = np.zeros(x_target.ravel().shape)

    np.add.at(n_data_target, idx, 1)
    np.add.at(sum_target, idx, data.ravel())
    np.add.at(std_target, idx, data.ravel() ** 2)

    # Calculate average target grid values
    data_target = sum_target / n_data_target
    std_target = (std_target / n_data_target - data_target**2.0) ** 0.5

    # # Evaluate mask for invalid comparison sample values
    # if mask_invalid is True:
    #     data_target.reshape(x_target.shape)[
    #         np.where(n_data_target.reshape(x_target.shape) != n_min_source)
    #     ] = np.nan

    # Dynamically estimate n_min_source if not provided
    if n_min_source is None:
        dx_source = np.mean(np.diff(np.unique(x_source)))
        dy_source = np.mean(np.diff(np.unique(y_source)))
        dx_target = np.mean(np.diff(np.unique(x_target)))
        dy_target = np.mean(np.diff(np.unique(y_target)))
        source_pixel_area = dx_source * dy_source
        target_pixel_area = dx_target * dy_target
        expected_n_source = target_pixel_area / source_pixel_area
        n_min_source = int(np.floor(expected_n_source))

    # Evaluate mask for invalid comparison sample values
    if mask_invalid:
        mask = n_data_target.reshape(x_target.shape) < n_min_source
        data_target.reshape(x_target.shape)[mask] = np.nan
        std_target.reshape(x_target.shape)[mask] = np.nan

    # if mask_invalid:
    #     threshold = np.percentile(n_data_target[n_data_target > 0], percentile_threshold)
    #     mask = n_data_target.reshape(x_target.shape) < threshold
    #     data_target.reshape(x_target.shape)[mask] = np.nan

    return data_target.reshape(x_target.shape), std_target.reshape(x_target.shape)


if __name__ == "__main__":
    pass
