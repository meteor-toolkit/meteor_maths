"""
Functions to read spectral response function data with pyspectral
"""

from collections.abc import Iterator

import numpy as np
from pyspectral.rsr_reader import RelativeSpectralResponse

"""___Authorship___"""
__author__ = "Sam Hunt"
__created__ = "5/11/2020"


def return_band_names(
    platform_name: str,
    sensor_name: str,
    band_names: list[str] | None = None,
    min_wl: float | None = None,
    max_wl: float | None = None,
) -> list[str]:
    """
    Returns band names for specified sensor from `pyspectral <https://pyspectral.readthedocs.io/en/master/installation.html#static-data>`_ library.

    :param platform_name: satellite name
    :param sensor_name: name of instrument on satellite
    :param band_names: (optional) if omitted all sensor band names are returned, otherwise submitted band names validated and returned
    :param min_wl: minimum wavelength to include in range
    :param max_wl: maximum wavelength to include in range

    :return: band names
    """

    srf_util = SensorSRFUtil(platform_name, sensor_name)
    return srf_util.return_band_names(band_names=band_names, min_wl=min_wl, max_wl=max_wl)


def return_band_centres(
    platform_name: str,
    sensor_name: str,
    band_names: list[str] | None = None,
    detector_name: str | None = None,
    min_wl: float | None = None,
    max_wl: float | None = None,
) -> np.ndarray:
    """
    Returns band centres for specified sensor from `pyspectral <https://pyspectral.readthedocs.io/en/master/installation.html#static-data>`_ library.

    :param platform_name: satellite name
    :param sensor_name: name of instrument on satellite
    :param band_names: name of bands to return band centres of, if omitted all band returned
    :param detector_name: name of sensor detector. Can be used in sensor has SRF data for for different
    detectors separately - if not specified in this case different
    :param min_wl: minimum wavelength to include in range
    :param max_wl: maximum wavelength to include in range

    :return: band centres in nm
    """

    srf_util = SensorSRFUtil(platform_name, sensor_name, detector_name, band_names=band_names)
    return srf_util.return_band_centres(band_names=band_names, min_wl=min_wl, max_wl=max_wl)


def return_srf(
    platform_name: str,
    sensor_name: str,
    band_name: str | None = None,
    detector_name: None | str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """

    Returns srf data for named band of for specified sensor from `pyspectral <https://pyspectral.readthedocs.io/en/master/installation.html#static-data>`_ library.

    :param platform_name: satellite name
    :param sensor_name: name of instrument on satellite
    :param band_name: name of sensor band
    :param detector_name: (optional) name of sensor detector. Can be used in sensor has SRF data for for different
    detectors separately - if not specified in this case different

    :return: band srf
    :return: band srf wavelength coordinates
    """

    srf_util = SensorSRFUtil(platform_name, sensor_name, detector_name)
    return srf_util.return_srf(band_name)


def return_iter_srf(
    platform_name: str,
    sensor_name: str,
    band_names: list[str] | None = None,
    detector_name: str | None = None,
) -> Iterator:
    """
    Returns iterable of band srfs for specified sensor from `pyspectral <https://pyspectral.readthedocs.io/en/master/installation.html#static-data>`_ library.

    :param platform_name: satellite name
    :param sensor_name: name of instrument on satellite
    :param band_names: name of bands to iterate through, if omitted all bands included
    :param detector_name: name of sensor detector. Can be used in sensor has SRF data for for different
    detectors separately - if not specified in this case different

    :return: iterable that returns band srf and srf wavelength coordinates at each iteration
    """

    srf_util = SensorSRFUtil(platform_name, sensor_name, detector_name, band_names=band_names)
    return iter(srf_util)


class SensorSRFUtil:
    """
    Helper class to define repeating functions along a coordinate axis

    from `pyspectral <https://pyspectral.readthedocs.io/en/master/installation.html#static-data>`_ library.

    :param platform_name: satellite name
    :param sensor_name: name of instrument on satellite
    :param detector_name: (optional) name of sensor detector. Can be used in sensor has SRF data for for different
    detectors separately - if not specified in this case different
    :param band_names: (optional) sensor bands to evaluate band integral for, if omitted band integral evaluated for
    all bands within spectral range of datar
    """

    def __init__(
        self,
        platform_name,
        sensor_name,
        detector_name: None | str = "det-1",
        band_names: None | list[str] = None,
    ):

        # Set attributes from arguments
        self.sensor = RelativeSpectralResponse(platform_name, sensor_name)
        self.detector_name = "det-1" if detector_name is None else detector_name

        # Unpack and validate selected bands
        self.band_names = self.return_band_names(band_names)
        self.band_centres = self.return_band_centres(band_names=self.band_names)

    def _band_info(self) -> tuple[list[str], np.ndarray]:
        """Return all sensor band names and centres.

        :return: band names
        :return: band centres (nm)
        """
        band_names = self.return_sensor_band_names()
        band_centres = np.array(
            [
                self.sensor.rsr[band_name][self.detector_name]["central_wavelength"] * self.sensor.si_scale / 1e-9
                for band_name in band_names
            ]
        )
        return band_names, band_centres

    def _filter_bands(
        self,
        band_names=None,
        min_wl=None,
        max_wl=None,
    ):
        names, centres = self._band_info()
        centre_lookup = dict(zip(names, centres))
        # Start from requested order, otherwise sensor order
        if band_names is None:
            selected_names = names
        else:
            selected_names = list(band_names)

        filtered_names = []
        filtered_centres = []
        for name in selected_names:
            centre = centre_lookup[name]

            if min_wl is not None and centre <= min_wl:
                continue
            if max_wl is not None and centre >= max_wl:
                continue
            filtered_names.append(name)
            filtered_centres.append(centre)

        return filtered_names, np.array(filtered_centres)

    def return_band_names(
        self,
        band_names: list[str] | str | None = None,
        min_wl: float | None = None,
        max_wl: float | None = None,
    ) -> list[str] | str:
        """
        Returns band names for specified sensor bands

        :param band_names: if omitted all sensor band names are returned,
         otherwise submitted band names validated and returned
        :param min_wl: minimum wavelength to include in range
        :param max_wl: maximum wavelength to include in range

        :return: band names
        """

        names, _ = self._filter_bands(band_names=band_names, min_wl=min_wl, max_wl=max_wl)
        return names

    def return_band_centres(
        self,
        band_names: list[str] | str | None = None,
        min_wl: float | None = None,
        max_wl: float | None = None,
    ) -> np.ndarray:
        """
        Returns band centres for specified sensor bands

        :param min_wl: minimum wavelength to include in range
        :param max_wl: maximum wavelength to include in range

        :return: band centres in nm
        """
        _, centres = self._filter_bands(band_names=band_names, min_wl=min_wl, max_wl=max_wl)
        return centres

    def return_sensor_band_names(self) -> list[str]:
        """
        Returns list of all sensor band names

        :return: sensor band names
        """

        return list(self.sensor.rsr.keys())

    def return_srf(self, band_name: str | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns srf data for specified sensor band

        :param band_name: sensor band name

        :return: band srf
        :return: band srf wavelength coordinates
        """

        srf = self.sensor.rsr[band_name][self.detector_name]["response"]  # gets rsr for given band
        wl_srf = 1000 * self.sensor.rsr[band_name][self.detector_name]["wavelength"]
        return srf, wl_srf

    def __iter__(self):

        # Define counter
        self.i = 0
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns ith function

        :return: band srf
        :return: band srf wavelength coordinates
        """

        # Iterate through bands
        if self.i < len(self.band_names):
            # Update counter
            self.i += 1

            return self.return_srf(self.band_names[self.i - 1])

        else:
            raise StopIteration


if __name__ == "__main__":
    pass
