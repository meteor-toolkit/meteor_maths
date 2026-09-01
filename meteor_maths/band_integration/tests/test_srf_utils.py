"""Tests for the SensorSRFUtil SRF helper class."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing

from meteor_maths.band_integration import srf_utils
from meteor_maths.band_integration.srf_utils import SensorSRFUtil


class FakeSensorSRF:
    def __init__(self):
        self.si_scale = 1
        self.rsr = {
            "B01": {
                "det-1": {
                    "central_wavelength": 0.55e-6,
                    "wavelength": np.array([0.5, 0.6]),
                    "response": np.array([0.1, 0.9]),
                },
                "det-2": {
                    "central_wavelength": 0.56e-6,
                    "wavelength": np.array([0.51, 0.61]),
                    "response": np.array([0.2, 0.8]),
                },
            },
            "B02": {
                "det-1": {
                    "central_wavelength": 0.70e-6,
                    "wavelength": np.array([0.65, 0.75]),
                    "response": np.array([0.3, 0.7]),
                },
                "det-2": {
                    "central_wavelength": 0.71e-6,
                    "wavelength": np.array([0.66, 0.76]),
                    "response": np.array([0.4, 0.6]),
                },
            },
            "B03": {
                "det-1": {
                    "central_wavelength": 0.90e-6,
                    "wavelength": np.array([0.85, 0.95]),
                    "response": np.array([0.5, 0.5]),
                },
                "det-2": {
                    "central_wavelength": 0.91e-6,
                    "wavelength": np.array([0.86, 0.96]),
                    "response": np.array([0.7, 0.3]),
                },
            },
        }


class TestSensorSRFUtil(unittest.TestCase):
    @patch("meteor_maths.band_integration.srf_utils.RelativeSpectralResponse")
    def test_init_sets_default_detector_and_band_centres(self, mock_rsr):
        mock_rsr.return_value = FakeSensorSRF()

        util = SensorSRFUtil("platform", "sensor")

        self.assertEqual(util.detector_name, "det-1")
        self.assertEqual(util.band_names, ["B01", "B02", "B03"])
        np.testing.assert_array_almost_equal(util.band_centres, np.array([550.0, 700.0, 900.0]))

    @patch("meteor_maths.band_integration.srf_utils.RelativeSpectralResponse")
    def test_return_band_names_filters_by_explicit_band_list(self, mock_rsr):
        mock_rsr.return_value = FakeSensorSRF()
        util = SensorSRFUtil("platform", "sensor")

        result = util.return_band_names(["B03", "B01"])

        self.assertEqual(result, ["B03", "B01"])

    @patch("meteor_maths.band_integration.srf_utils.RelativeSpectralResponse")
    def test_return_band_names_filters_by_wavelength_range(self, mock_rsr):
        mock_rsr.return_value = FakeSensorSRF()
        util = SensorSRFUtil("platform", "sensor")

        result = util.return_band_names(min_wl=600.0, max_wl=850.0)

        self.assertEqual(result, ["B02"])

    @patch("meteor_maths.band_integration.srf_utils.RelativeSpectralResponse")
    def test_return_band_centres_filters_by_explicit_band_list(self, mock_rsr):
        mock_rsr.return_value = FakeSensorSRF()
        util = SensorSRFUtil("platform", "sensor")

        result = util.return_band_centres(["B03", "B01"])

        np.testing.assert_array_almost_equal(result, np.array([900.0, 550.0]))

    @patch("meteor_maths.band_integration.srf_utils.RelativeSpectralResponse")
    def test_return_band_centres_filters_by_wavelength_range(self, mock_rsr):
        mock_rsr.return_value = FakeSensorSRF()
        util = SensorSRFUtil("platform", "sensor")

        result = util.return_band_centres(min_wl=600.0, max_wl=850.0)

        np.testing.assert_array_almost_equal(result, np.array([700.0]))

    @patch("meteor_maths.band_integration.srf_utils.RelativeSpectralResponse")
    def test_return_sensor_band_names_returns_canonical_sensor_names(self, mock_rsr):
        mock_rsr.return_value = FakeSensorSRF()
        util = SensorSRFUtil("platform", "sensor")

        result = util.return_sensor_band_names()

        self.assertEqual(result, ["B01", "B02", "B03"])

    @patch("meteor_maths.band_integration.srf_utils.RelativeSpectralResponse")
    def test_return_srf_uses_requested_detector(self, mock_rsr):
        mock_rsr.return_value = FakeSensorSRF()
        util = SensorSRFUtil("platform", "sensor", detector_name="det-2")

        srf, wavelength = util.return_srf("B02")

        np.testing.assert_array_equal(srf, np.array([0.4, 0.6]))
        np.testing.assert_array_equal(wavelength, np.array([660.0, 760.0]))

    @patch("meteor_maths.band_integration.srf_utils.RelativeSpectralResponse")
    def test_return_srf_uses_default_detector_when_none_is_passed(self, mock_rsr):
        mock_rsr.return_value = FakeSensorSRF()
        util = SensorSRFUtil("platform", "sensor", detector_name=None)

        srf, wavelength = util.return_srf("B01")

        np.testing.assert_array_equal(srf, np.array([0.1, 0.9]))
        np.testing.assert_array_equal(wavelength, np.array([500.0, 600.0]))

    @patch("meteor_maths.band_integration.srf_utils.RelativeSpectralResponse")
    def test_iterator_returns_selected_band_data_in_order(self, mock_rsr):
        mock_rsr.return_value = FakeSensorSRF()
        util = SensorSRFUtil("platform", "sensor", band_names=["B03", "B01"])

        iterator = iter(util)
        first = next(iterator)
        second = next(iterator)

        np.testing.assert_array_equal(first[0], np.array([0.5, 0.5]))
        np.testing.assert_array_equal(first[1], np.array([850.0, 950.0]))
        np.testing.assert_array_equal(second[0], np.array([0.1, 0.9]))
        np.testing.assert_array_equal(second[1], np.array([500.0, 600.0]))

    @patch("meteor_maths.band_integration.srf_utils.RelativeSpectralResponse")
    def test_iterator_raises_stop_iteration_after_last_band(self, mock_rsr):
        mock_rsr.return_value = FakeSensorSRF()
        util = SensorSRFUtil("platform", "sensor", band_names=["B01"])

        iterator = iter(util)
        next(iterator)

        with self.assertRaises(StopIteration):
            next(iterator)

    @patch("meteor_maths.band_integration.srf_utils.SensorSRFUtil")
    def test_return_band_names_wrapper_uses_sensor_util(self, mock_util):
        mock_util.return_value.return_band_names.return_value = ["B01", "B02"]

        result = srf_utils.return_band_names("platform", "sensor", ["B01"], min_wl=500.0, max_wl=800.0)

        self.assertEqual(result, ["B01", "B02"])
        mock_util.assert_called_once_with("platform", "sensor")
        mock_util.return_value.return_band_names.assert_called_once_with(band_names=["B01"], min_wl=500.0, max_wl=800.0)

    @patch("meteor_maths.band_integration.srf_utils.SensorSRFUtil")
    def test_return_band_centres_wrapper_uses_sensor_util(self, mock_util):
        mock_util.return_value.return_band_centres.return_value = np.array([550.0, 700.0])

        result = srf_utils.return_band_centres("platform", "sensor", ["B01"], "det-1", 500.0, 800.0)

        np.testing.assert_array_equal(result, np.array([550.0, 700.0]))
        mock_util.assert_called_once_with("platform", "sensor", "det-1", band_names=["B01"])
        mock_util.return_value.return_band_centres.assert_called_once_with(
            band_names=["B01"], min_wl=500.0, max_wl=800.0
        )

    @patch("meteor_maths.band_integration.srf_utils.SensorSRFUtil")
    def test_return_srf_wrapper_uses_sensor_util(self, mock_util):
        mock_util.return_value.return_srf.return_value = (np.array([0.2, 0.8]), np.array([500.0, 600.0]))

        result = srf_utils.return_srf("platform", "sensor", "B01", "det-1")

        np.testing.assert_array_equal(result[0], np.array([0.2, 0.8]))
        np.testing.assert_array_equal(result[1], np.array([500.0, 600.0]))
        mock_util.assert_called_once_with("platform", "sensor", "det-1")
        mock_util.return_value.return_srf.assert_called_once_with("B01")

    @patch("meteor_maths.band_integration.srf_utils.SensorSRFUtil")
    def test_return_iter_srf_wrapper_uses_sensor_util(self, mock_util):
        iterator = iter([("band", np.array([1.0]))])
        mock_util.return_value.__iter__ = MagicMock(return_value=iterator)

        result = srf_utils.return_iter_srf("platform", "sensor", ["B01"], "det-1")

        self.assertIs(result, iterator)
        mock_util.assert_called_once_with("platform", "sensor", "det-1", band_names=["B01"])
        mock_util.return_value.__iter__.assert_called_once_with()
