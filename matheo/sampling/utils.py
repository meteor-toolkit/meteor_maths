"""Module of utility functions for algorithms"""

from typing import Tuple

from pyproj import Transformer, CRS
import numpy as np


def add_geolocation_error(
    lat_error: float,
    lon_error: float,
    lat_grid_deg: np.ndarray,
    lon_grid_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add latitude & longitude errors (input in m) to latitude & longitudes grids given in degrees.

    :param lat_error: latitude error to add (in m)
    :param lon_error: longitude error to add (in m)
    :param lat_grid_deg: latitude grid (in deg)
    :param lon_grid_deg: longitude grid (in deg)
    :return: latitude grid with added error (in deg), longitude grid with added error (in deg)
    """
    metreProj = CRS.from_epsg(3857)
    latlonProj = CRS.from_epsg(4326)

    # Create transformer objects to/from latlon and metres
    transformer_to_m = Transformer.from_crs(latlonProj, metreProj)
    transformer_to_deg = Transformer.from_crs(metreProj, latlonProj)

    # Convert lat/lon grids to metres and add error (in m)
    lat_grid_m, lon_grid_m = transformer_to_m.transform(lon_grid_deg, lat_grid_deg)
    lat_grid_m_error = lat_grid_m + lat_error
    lon_grid_m_error = lon_grid_m + lon_error

    # Convert back to lat/lon grids (in deg)
    lat_grid_deg_error, lon_grid_deg_error = transformer_to_deg.transform(
        lon_grid_m_error, lat_grid_m_error
    )

    return lat_grid_deg_error, lon_grid_deg_error


if __name__ == "__main__":
    pass
