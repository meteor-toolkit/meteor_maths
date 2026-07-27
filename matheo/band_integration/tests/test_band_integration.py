"""
Tests for band_integration module
"""

import numpy.testing

from matheo.band_integration import band_integration as bi
from matheo.utils import function_def as fd
import unittest
from unittest.mock import patch, call
import numpy as np


"""___Authorship___"""
__author__ = "Sam Hunt"
__created__ = "1/8/2021"


TEST_WL_SPECTRA = np.arange(500, 952, 2)
TEST_SPECTRA = np.array(
    [
        86.40240984,
        87.0470624,
        87.93084099,
        89.34519073,
        88.9919428,
        88.60131038,
        87.64776277,
        84.56509229,
        80.20699595,
        77.57993473,
        79.70188401,
        82.92151658,
        82.80886134,
        80.83847235,
        81.07832761,
        82.60714817,
        81.57882444,
        80.39565515,
        81.47071281,
        81.22992248,
        79.38628691,
        79.1960746,
        80.10704642,
        79.99129846,
        78.99152007,
        78.53258221,
        78.73681595,
        78.59562385,
        77.4069349,
        75.88195983,
        75.29629964,
        75.51243919,
        74.94571446,
        73.74180549,
        73.11458274,
        72.54168798,
        72.24234318,
        71.92226982,
        70.94521004,
        70.45248178,
        70.99364808,
        72.00680476,
        72.3550522,
        71.58434072,
        69.98142462,
        68.92419954,
        70.26385016,
        75.19944433,
        81.83504616,
        84.86159094,
        84.26705631,
        83.90527171,
        84.47465617,
        84.59068069,
        83.82628876,
        82.86556563,
        81.64814889,
        80.20934677,
        79.8305731,
        80.94084382,
        81.73141581,
        80.8522182,
        79.26928993,
        78.39712344,
        78.15093325,
        77.97080866,
        78.13166684,
        78.69433567,
        79.47464896,
        80.23475687,
        80.25169924,
        79.98833192,
        79.82274719,
        79.39371379,
        78.7166021,
        78.48135269,
        79.04004412,
        77.08470515,
        73.92763009,
        75.20355205,
        78.27530866,
        79.20028329,
        78.9060233,
        78.68683697,
        78.68743231,
        78.4101011,
        78.00164631,
        77.85004546,
        77.71481071,
        77.36596503,
        77.02105272,
        76.76404584,
        74.76152893,
        68.14158018,
        64.48681771,
        72.32108895,
        82.55587109,
        87.13105185,
        88.6025905,
        88.77327593,
        87.87524042,
        87.17447792,
        87.54344993,
        87.89064515,
        87.78310067,
        87.4372012,
        86.75809996,
        85.59221129,
        82.45337676,
        77.89214634,
        76.88395498,
        78.8209346,
        79.04027269,
        78.6144584,
        78.56530008,
        78.76360561,
        80.1262072,
        81.5695274,
        81.80366939,
        81.31974949,
        80.66255438,
        80.46491483,
        81.03730472,
        81.40129017,
        81.10318427,
        80.59216682,
        80.49183472,
        80.52269386,
        79.50098152,
        66.05451868,
        40.17258215,
        26.60884751,
        33.40940982,
        50.73591094,
        66.91599287,
        74.31876894,
        76.12143969,
        76.35720283,
        76.37629724,
        76.32542992,
        76.03904999,
        75.60167261,
        75.06019506,
        74.59689624,
        74.14847808,
        73.3689048,
        72.31972623,
        71.67775051,
        71.8884457,
        72.08096512,
        71.6581701,
        71.10508221,
        70.76328245,
        70.59985507,
        70.07171878,
        69.1986223,
        67.97425178,
        65.63276356,
        63.08357537,
        62.156023,
        62.44737329,
        62.65974306,
        63.49386315,
        64.65446038,
        64.57515343,
        63.91961441,
        63.65660967,
        64.05091569,
        64.72807323,
        64.9748043,
        64.97701333,
        64.81060187,
        64.56459521,
        64.51001273,
        63.49677515,
        62.0992306,
        60.46544467,
        58.73146736,
        60.21829619,
        62.54268154,
        62.72533738,
        62.59816901,
        60.95152855,
        58.15937556,
        58.38529592,
        60.43190302,
        61.09648317,
        60.69503418,
        60.39087873,
        60.25081674,
        59.67189389,
        59.29399198,
        59.14432292,
        58.75771598,
        58.81311952,
        58.96741475,
        58.12570394,
        56.19088757,
        53.69796175,
        50.63340289,
        47.91125029,
        48.85782742,
        52.11158853,
        51.94845901,
        49.06416768,
        48.18201855,
        48.92499353,
        48.68125047,
        48.02615653,
        48.55940995,
        49.89487678,
        50.23265632,
        49.75751011,
        49.00514577,
        46.06337159,
        39.22109244,
        31.28268297,
        26.6232523,
        25.62742936,
        28.6281889,
        32.56591759,
        32.48297369,
        31.30941894,
        31.62610506,
        32.32666082,
        32.81016886,
    ]
)


def fake__band_int(d, x, r, x_r, rint_norm=True):
    return 1


def fake_band_int(d, x, r, x_r, d_axis_x, rint_norm=True):
    sli = [slice(None)] * d.ndim
    sli[d_axis_x] = 0
    sli = tuple(sli)

    return np.ones(d[sli].shape)


class FakeBandGen:
    def __init__(self, max_iter=3):
        self.max_iter = max_iter
        self.len_out = 5

    def __iter__(self):

        # Define counter
        self.i = 0
        return self

    def __next__(self):
        """
        Returns ith function

        :return: fake band srf
        :return: band srf wavelength coordinates
        """

        # Iterate through bands
        if self.i < self.max_iter:
            # Update counter
            self.i += 1

            return np.full(self.len_out, self.i), np.arange(self.len_out)

        else:
            raise StopIteration


class TestBandIntegrate(unittest.TestCase):
    def test_cutout_nonzero_buffer(self):
        x = np.arange(20, 80, 0.1)
        y = fd.f_tophat(x, 50, 10)

        x_test = np.arange(43, 57, 0.1)
        y_test = fd.f_tophat(x_test, 50, 10)

        y_eval, x_eval, idx = bi.cutout_nonzero(y, x, buffer=0.2)

        np.testing.assert_array_almost_equal(x_test, x_eval)
        np.testing.assert_array_almost_equal(y_test, y_eval)

    def test_cutout_nonzero_nobuffer(self):
        x = np.arange(20, 80, 0.1)
        y = fd.f_tophat(x, 50, 10)

        x_test = np.arange(45, 55, 0.1)
        y_test = np.ones(x_test.shape)

        y_eval, x_eval, idx = bi.cutout_nonzero(y, x, buffer=0.0)

        np.testing.assert_array_almost_equal(x_test, x_eval)
        np.testing.assert_array_almost_equal(y_test, y_eval)

    def test_get_x_offset(self):
        x = np.arange(0, 40, 1)
        y = fd.f_tophat(x, 20, 10)
        y[23] = 1.1

        x_off = bi.get_x_offset(y, x, 50)

        self.assertEqual(x_off, 27)

    def test__band_int_highressrf(self):

        x = np.arange(0, 100, 0.01)
        d = (0.02 * x) ** 3 + (-0.2 * x) ** 2 + (-3 * x) + 100

        x_r = np.arange(30, 70, 0.001)
        r = fd.f_triangle(x_r, 50, 5)

        x_band = bi._band_int(d, x, r, x_r)

        self.assertAlmostEqual(x_band, 51.1717, places=3)

    def test__band_int_highresspec(self):

        x = np.arange(0, 100, 0.0001)
        d = (0.02 * x) ** 3 + (-0.2 * x) ** 2 + (-3 * x) + 100

        x_r = np.arange(30, 70, 0.001)
        r = fd.f_triangle(x_r, 50, 5)

        x_band = bi._band_int(d, x, r, x_r)

        self.assertAlmostEqual(x_band, 51.1717, places=3)

    def test__band_int_highresspec_nonorm(self):

        x = np.arange(0, 100, 0.0001)
        d = (0.02 * x) ** 3 + (-0.2 * x) ** 2 + (-3 * x) + 100

        x_r = np.arange(30, 70, 0.001)
        r = fd.f_triangle(x_r, 50, 5)

        x_band = bi._band_int(d, x, r, x_r, rint_norm=False)

        self.assertAlmostEqual(x_band, 51.1717 * 5, places=3)

    def test__band_int_regular_grid_r1d_d1d(self):
        d = np.array([4, 4, 4, 4, 4])
        x = np.arange(4)
        r = np.array([1, 1, 1, 1, 1])

        self.assertEqual(bi._band_int_regular_grid(d, x, r), 4)

    def test__band_int_regular_grid_r1d_d1d_nonorm(self):
        d = np.array([4, 4, 4, 4, 4])
        x = np.arange(4)
        r = np.array([1, 1, 1, 1, 1])

        self.assertEqual(bi._band_int_regular_grid(d, x, r, rint_norm=False), 20.0)

    def test__band_int_regular_grid_r2d_d2d(self):
        d = np.array(
            [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
        )

        x = np.arange(4)

        r = np.array(
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        )

        d_int = bi._band_int_regular_grid(d, x, r, d_axis_x=1)

        d_int_expected = np.array(
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        )

        np.testing.assert_array_equal(d_int, d_int_expected)

    def test__band_int_regular_grid_r2d_d3d(self):
        d = np.array(
            [
                [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
                [[4, 4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6]],
            ]
        )

        x = np.arange(4)

        r = np.array(
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        )

        d_int = bi._band_int_regular_grid(d, x, r, d_axis_x=2)

        d_int_expected = np.array(
            [
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
                [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]],
            ]
        )

        np.testing.assert_array_equal(d_int, d_int_expected)

    @patch("matheo.band_integration.band_integration._band_int_regular_grid")
    def test__band_int_arr_regular_grid(self, mock):
        x = np.arange(4)
        x_r = np.arange(4)
        d_band = bi._band_int_arr("d", x, "r", x_r, d_axis_x=2)

        mock.assert_called_once_with("d", x, "r", d_axis_x=2)

    @patch("matheo.band_integration.band_integration._band_int", wraps=fake__band_int)
    def test_band_int_arr_r1d_d1d(self, mock):
        d = np.zeros(5)
        x = np.arange(5)
        x_r = np.arange(10)
        r = fd.f_triangle(x_r, 5, 5)

        d_band = bi._band_int_arr(d, x, r, x_r)

        np.testing.assert_array_equal(np.ones(1), d_band)

        self.assertEqual(1, mock.call_count)

        np.testing.assert_array_equal(mock.call_args_list[0][0][0], d)
        np.testing.assert_array_equal(mock.call_args_list[0][1]["x"], x)
        np.testing.assert_array_equal(mock.call_args_list[0][1]["r"], r)
        np.testing.assert_array_equal(mock.call_args_list[0][1]["x_r"], x_r)

    @patch("matheo.band_integration.band_integration._band_int", wraps=fake__band_int)
    def test_band_int_arr_r1d_d3d(self, mock):

        d = np.zeros((3, 4, 5))
        x = np.arange(5)
        x_r = np.arange(10)
        r = fd.f_triangle(x_r, 5, 5)

        d_band = bi._band_int_arr(d, x, r, x_r, d_axis_x=2)

        np.testing.assert_array_equal(np.ones((3, 4)), d_band)

        self.assertEqual(12, mock.call_count)

        expected_calls = []
        for i in range(12):
            expected_calls.append(call(np.zeros(5), x=x, r=r, x_r=x_r))

        for expected_call, real_call in zip(expected_calls, mock.call_args_list):
            np.testing.assert_array_equal(real_call[0][0], expected_call[1][0])

            real_kwargs = real_call[1]
            expected_kwargs = expected_call[2]
            np.testing.assert_array_equal(real_kwargs["x"], expected_kwargs["x"])
            np.testing.assert_array_equal(real_kwargs["r"], expected_kwargs["r"])
            np.testing.assert_array_equal(real_kwargs["x_r"], expected_kwargs["x_r"])

    @patch("matheo.band_integration.band_integration._band_int", wraps=fake__band_int)
    def test_band_int_arr_r2d_d1d(self, mock):
        d = np.zeros(5)
        x = np.arange(5)
        x_r = np.arange(10)
        r = np.zeros((5, 10))
        r[0, :] = 0.0
        r[1, :] = 1.0
        r[2, :] = 2.0
        r[3, :] = 3.0
        r[4, :] = 4.0

        d_band = bi._band_int_arr(d, x, r, x_r, d_axis_x=2)

        np.testing.assert_array_equal(np.ones(5), d_band)

        self.assertEqual(5, mock.call_count)

        expected_calls = []
        for i in range(5):
            expected_calls.append(call(np.zeros(5), x=x, r=np.full(10, i), x_r=x_r))

        for expected_call, real_call in zip(expected_calls, mock.call_args_list):
            np.testing.assert_array_equal(real_call[0][0], expected_call[1][0])

            real_kwargs = real_call[1]
            expected_kwargs = expected_call[2]
            np.testing.assert_array_equal(real_kwargs["x"], expected_kwargs["x"])
            np.testing.assert_array_equal(real_kwargs["r"], expected_kwargs["r"])
            np.testing.assert_array_equal(real_kwargs["x_r"], expected_kwargs["x_r"])

    @patch("matheo.band_integration.band_integration._band_int", wraps=fake__band_int)
    def test_band_int_arr_r2d_d3d(self, mock):
        d = np.zeros((3, 4, 5))
        x = np.arange(5)
        x_r = np.arange(10)
        r = np.zeros((6, 10))
        r[0, :] = 0.0
        r[1, :] = 1.0
        r[2, :] = 2.0
        r[3, :] = 3.0
        r[4, :] = 4.0
        r[5, :] = 5.0

        d_band = bi._band_int_arr(d, x, r, x_r, d_axis_x=2)

        np.testing.assert_array_equal(np.ones((3, 4, 6)), d_band)

        self.assertEqual(72, mock.call_count)

        expected_calls = []
        for i in range(5):
            for j in range(12):
                expected_calls.append(call(np.zeros(5), x=x, r=np.full(10, i), x_r=x_r))

        for expected_call, real_call in zip(expected_calls, mock.call_args_list):
            np.testing.assert_array_equal(real_call[0][0], expected_call[1][0])

            real_kwargs = real_call[1]
            expected_kwargs = expected_call[2]
            np.testing.assert_array_equal(real_kwargs["x"], expected_kwargs["x"])
            np.testing.assert_array_equal(real_kwargs["r"], expected_kwargs["r"])
            np.testing.assert_array_equal(real_kwargs["x_r"], expected_kwargs["x_r"])

    @patch("matheo.band_integration.band_integration._band_int", wraps=fake__band_int)
    def test_band_int2ax_arr(self, mock):

        d = np.zeros((3, 4, 5))
        x = np.arange(5)
        y = np.arange(3)
        x_rx = np.arange(10)
        rx = fd.f_triangle(x_rx, 5, 5)
        y_ry = np.arange(5)
        ry = fd.f_triangle(y_ry, 5, 3)

        d_band = bi._band_int2ax_arr(
            d, x, y, rx, x_rx, ry, y_ry, d_axis_x=2, d_axis_y=0
        )

        np.testing.assert_array_equal(np.ones(4), d_band)

        self.assertEqual(16, mock.call_count)

        expected_calls = []
        for i in range(12):
            expected_calls.append(call(np.zeros(5), x=x, r=rx, x_r=x_rx))

        for i in range(4):
            expected_calls.append(call(np.ones(3), x=y, r=ry, x_r=y_ry))

        for expected_call, real_call in zip(expected_calls, mock.call_args_list):
            np.testing.assert_array_equal(real_call[0][0], expected_call[1][0])

            real_kwargs = real_call[1]
            expected_kwargs = expected_call[2]
            np.testing.assert_array_equal(real_kwargs["x"], expected_kwargs["x"])
            np.testing.assert_array_equal(real_kwargs["r"], expected_kwargs["r"])
            np.testing.assert_array_equal(real_kwargs["x_r"], expected_kwargs["x_r"])

    @patch("matheo.band_integration.band_integration._band_int", wraps=fake__band_int)
    def test_band_int3ax_arr(self, mock):

        d = np.zeros((3, 4, 5))
        x = np.arange(5)
        y = np.arange(3)
        z = np.arange(4)
        x_rx = np.arange(10)
        rx = fd.f_triangle(x_rx, 5, 5)
        y_ry = np.arange(5)
        ry = fd.f_triangle(y_ry, 5, 3)
        z_rz = np.arange(7)
        rz = fd.f_triangle(y_ry, 5, 3)

        d_band = bi._band_int3ax_arr(
            d, x, y, z, rx, x_rx, ry, y_ry, rz, z_rz, d_axis_x=2, d_axis_y=0, d_axis_z=1
        )

        np.testing.assert_array_equal(np.array([1]), d_band)

        self.assertEqual(17, mock.call_count)

        expected_calls = []
        for i in range(12):
            expected_calls.append(call(np.zeros(5), x=x, r=rx, x_r=x_rx))

        for i in range(4):
            expected_calls.append(call(np.ones(3), x=y, r=ry, x_r=y_ry))

        expected_calls.append(call(np.ones(4), x=z, r=rz, x_r=z_rz))

        for expected_call, real_call in zip(expected_calls, mock.call_args_list):
            np.testing.assert_array_equal(real_call[0][0], expected_call[1][0])

            real_kwargs = real_call[1]
            expected_kwargs = expected_call[2]
            np.testing.assert_array_equal(real_kwargs["x"], expected_kwargs["x"])
            np.testing.assert_array_equal(real_kwargs["r"], expected_kwargs["r"])
            np.testing.assert_array_equal(real_kwargs["x_r"], expected_kwargs["x_r"])

    @patch("matheo.band_integration.band_integration.band_int", wraps=fake_band_int)
    def test_iter_band_int(self, mock):
        d = np.zeros((3, 4, 11))
        x = np.arange(11)
        fakebandgen = FakeBandGen(2)
        fakebanditer = iter(fakebandgen)

        d_band = bi.iter_band_int(d, x, fakebanditer, d_axis_x=2)

        self.assertEqual(d_band.shape, (3, 4, 2))
        np.testing.assert_array_equal(d_band, np.ones(d_band.shape))

        expected_calls = [
            call(d, x, np.full(5, 1), np.arange(5), 2),
            call(d, x, np.full(5, 2), np.arange(5), 2),
        ]

        for expected_call, real_call in zip(expected_calls, mock.call_args_list):
            np.testing.assert_array_equal(real_call[0][0], expected_call[1][0])
            np.testing.assert_array_equal(real_call[0][1], expected_call[1][1])
            np.testing.assert_array_equal(real_call[0][2], expected_call[1][2])
            np.testing.assert_array_equal(real_call[0][3], expected_call[1][3])
            self.assertEqual(real_call[0][4], expected_call[1][4])

    @patch("matheo.band_integration.band_integration.band_int", wraps=fake_band_int)
    def test_spectral_band_int_sensor(self, mock):
        d = np.zeros((3, 4, 11))
        wl = np.arange(400, 510, 10)

        d_band, wl_band = bi.spectral_band_int_sensor(
            d,
            wl,
            d_axis_wl=2,
            platform_name="Sentinel-2A",
            sensor_name="msi",
        )

        self.assertEqual(d_band.shape, (3, 4, 2))
        np.testing.assert_array_equal(d_band, np.ones(d_band.shape))

        expected_calls = [
            call(
                d,
                wl,
                np.array(
                    [
                        0.00000000e00,
                        0.00000000e00,
                        1.77574169e-03,
                        4.07306058e-03,
                        3.62614286e-03,
                        3.51519883e-03,
                        5.72916260e-03,
                        3.78029188e-03,
                        2.63673207e-03,
                        1.26211275e-03,
                        1.98758300e-03,
                        1.36891310e-03,
                        1.25044398e-03,
                        4.63454402e-04,
                        8.14292987e-04,
                        1.37643155e-03,
                        1.48508593e-03,
                        1.82373472e-03,
                        1.62681751e-03,
                        4.39206185e-03,
                        2.90080979e-02,
                        1.18745930e-01,
                        3.23875070e-01,
                        5.72819233e-01,
                        7.14727521e-01,
                        7.61967778e-01,
                        7.89297044e-01,
                        8.08623850e-01,
                        8.10893834e-01,
                        8.24198782e-01,
                        8.54158103e-01,
                        8.70790899e-01,
                        8.87310982e-01,
                        9.26199257e-01,
                        9.82281506e-01,
                        1.00000000e00,
                        9.75238204e-01,
                        9.35963392e-01,
                        8.89971495e-01,
                        8.50210488e-01,
                        8.25694501e-01,
                        7.83902407e-01,
                        6.14174187e-01,
                        3.30071092e-01,
                        1.24108315e-01,
                        4.36569415e-02,
                        1.47495950e-02,
                        0.00000000e00,
                        0.00000000e00,
                    ]
                ),
                np.arange(410, 459, 1),
                2,
            ),
            call(
                d,
                wl,
                np.array(
                    [
                        0.0000000e00,
                        0.0000000e00,
                        4.2555310e-02,
                        7.2298303e-02,
                        1.5374322e-01,
                        3.2799226e-01,
                        5.5336785e-01,
                        7.1011168e-01,
                        7.5285178e-01,
                        7.5232691e-01,
                        7.5668079e-01,
                        7.6326948e-01,
                        7.6239425e-01,
                        7.8525150e-01,
                        8.1546670e-01,
                        8.6179179e-01,
                        8.9282596e-01,
                        9.1952211e-01,
                        9.1900647e-01,
                        9.1315752e-01,
                        9.0035367e-01,
                        8.8989693e-01,
                        8.8232458e-01,
                        8.7606120e-01,
                        8.8429987e-01,
                        9.0695542e-01,
                        9.3232083e-01,
                        9.3947250e-01,
                        9.4383544e-01,
                        9.2204088e-01,
                        8.8602310e-01,
                        8.4743607e-01,
                        8.1251687e-01,
                        7.8239709e-01,
                        7.7310872e-01,
                        7.7209055e-01,
                        7.8742653e-01,
                        8.1217176e-01,
                        8.4605050e-01,
                        8.8767993e-01,
                        9.2793995e-01,
                        9.5069236e-01,
                        9.6573311e-01,
                        9.6938252e-01,
                        9.6570295e-01,
                        9.5832002e-01,
                        9.5405066e-01,
                        9.5178270e-01,
                        9.5699722e-01,
                        9.6556515e-01,
                        9.7705138e-01,
                        9.7709572e-01,
                        9.7436607e-01,
                        9.5903182e-01,
                        9.3506318e-01,
                        9.0190136e-01,
                        8.7165791e-01,
                        8.4402442e-01,
                        8.2280850e-01,
                        8.1536043e-01,
                        8.2057637e-01,
                        8.3951491e-01,
                        8.6992168e-01,
                        9.1526204e-01,
                        9.6067029e-01,
                        9.9163699e-01,
                        1.0000000e00,
                        9.8356098e-01,
                        9.1130763e-01,
                        7.4018258e-01,
                        5.0395858e-01,
                        3.0501550e-01,
                        1.8004605e-01,
                        1.0738342e-01,
                        6.5935917e-02,
                        4.2077459e-02,
                        2.6621290e-02,
                        1.4339600e-02,
                        2.6577900e-03,
                        8.1822003e-04,
                        0.0000000e00,
                        0.0000000e00,
                    ]
                ),
                np.arange(454, 536, 1),
                2,
            ),
        ]

        for expected_call, real_call in zip(expected_calls, mock.call_args_list):
            np.testing.assert_array_equal(real_call[0][0], expected_call[1][0])
            np.testing.assert_array_equal(real_call[0][1], expected_call[1][1])
            np.testing.assert_array_almost_equal(real_call[0][2], expected_call[1][2])
            np.testing.assert_array_almost_equal(
                real_call[0][3], expected_call[1][3], decimal=4
            )
            self.assertEqual(real_call[0][4], expected_call[1][4])

    def test_spectral_band_int_sensor_nomock(
        self,
    ):
        d = np.zeros((3, 4, 16))
        wl = np.arange(400, 560, 10)

        d_band, wl_band = bi.spectral_band_int_sensor(
            d,
            wl,
            d_axis_wl=2,
            platform_name="Sentinel-2A",
            sensor_name="msi",
        )

        self.assertEqual(d_band.shape, (3, 4, 2))
        np.testing.assert_array_equal(d_band, np.zeros(d_band.shape))

        expected_wl_band = np.array([442.69, 492.7])

        np.testing.assert_array_almost_equal(wl_band, expected_wl_band, decimal=1)

        d = np.zeros(16)
        wl = np.arange(400, 560, 10)

        d_band, wl_band = bi.spectral_band_int_sensor(
            d,
            wl,
            platform_name="Sentinel-2A",
            sensor_name="msi",
        )

        self.assertEqual(d_band.shape, (2,))
        np.testing.assert_array_equal(d_band, np.zeros(d_band.shape))

        expected_wl_band = np.array([442.69, 492.7])

        np.testing.assert_array_almost_equal(wl_band, expected_wl_band, decimal=1)

    @patch("matheo.band_integration.band_integration.return_r_pixel")
    @patch("matheo.band_integration.band_integration.band_int")
    def test_pixel_int(self, mock_bi, mock_rrp):
        d = np.zeros(12)
        x = np.arange(12)
        x_pixel = np.array([5, 10])
        width_pixel = np.array([2, 4])
        d_axis_x = 0

        d_band = bi.pixel_int(
            d=d, x=x, x_pixel=x_pixel, width_pixel=width_pixel, d_axis_x=d_axis_x
        )

        mock_bi.assert_called_once_with(
            d=d, x=x, r=mock_rrp.return_value, x_r=x, d_axis_x=d_axis_x
        )

    @patch("matheo.band_integration.band_integration.return_r_pixel")
    @patch("matheo.band_integration.band_integration.band_int")
    def test_pixel_int_r_sampling(self, mock_bi, mock_rrp):
        d = np.zeros(12)
        x = np.arange(12)
        x_pixel = np.array([5, 10])
        width_pixel = np.array([2, 4])
        d_axis_x = 0

        d_band = bi.pixel_int(
            d=d, x=x, x_pixel=x_pixel, width_pixel=width_pixel, d_axis_x=d_axis_x
        )

        mock_bi.assert_called_once_with(
            d=d, x=x, r=mock_rrp.return_value, x_r=x, d_axis_x=d_axis_x
        )

    @patch("matheo.band_integration.band_integration.return_r_pixel")
    @patch("matheo.band_integration.band_integration.band_int")
    def test_pixel_int_r_sampling(self, mock_bi, mock_rrp):
        d = np.zeros(12)
        x = np.arange(12)
        x_pixel = np.array([5, 10])
        width_pixel = np.array([2, 4])
        d_axis_x = 0

        d_band = bi.pixel_int(
            d=d,
            x=x,
            x_pixel=x_pixel,
            width_pixel=width_pixel,
            r_sampling=1.0,
            band_shape="tophat",
            d_axis_x=d_axis_x,
        )

        x_r = np.arange(3.0, 15.0)

        np.testing.assert_array_almost_equal(mock_bi.call_args[1]["d"], d)
        np.testing.assert_array_almost_equal(mock_bi.call_args[1]["x"], x)
        self.assertEqual(mock_bi.call_args[1]["r"], mock_rrp.return_value)
        np.testing.assert_array_almost_equal(mock_bi.call_args[1]["x_r"], x_r)
        self.assertEqual(mock_bi.call_args[1]["d_axis_x"], d_axis_x)

    @patch("matheo.band_integration.band_integration.band_int", wraps=fake_band_int)
    def test_pixel_int_eval_iter(self, mock):
        d = np.zeros((3, 4, 11))
        x = np.arange(11)
        x_pixel = np.array([5, 10])
        width_pixel = np.array([2, 4])

        d_band = bi.pixel_int(
            d=d,
            x=x,
            x_pixel=x_pixel,
            width_pixel=width_pixel,
            d_axis_x=2,
            eval_iter=True,
        )

        self.assertEqual(d_band.shape, (3, 4, 2))
        np.testing.assert_array_equal(d_band, np.ones(d_band.shape))

        x_r_0 = np.arange(
            5 - 2, 5 + 2 + 1, 0.01
        )  # (centre-width, centre+width+1, 0.01)
        # r_0 = f_triangle(x_r_0, centre, width):
        r_0 = np.zeros(x_r_0.shape)
        first_half = np.logical_and(5 - 2 < x_r_0, x_r_0 <= 5)
        r_0[first_half] = (x_r_0[first_half] - (5 - 2)) / (x_pixel[0] - (5 - 2))

        second_half = np.logical_and(x_pixel[0] < x_r_0, x_r_0 < (5 + 2))
        r_0[second_half] = ((5 + 2) - x_r_0[second_half]) / ((5 + 2) - 5)

        x_r_1 = np.arange(
            10 - 4, 10 + 4 + 1, 0.01
        )  # (centre-width, centre+width+1, 0.01)
        # r_1 = f_triangle(x_r_1, centre, width):
        r_1 = np.zeros(x_r_1.shape)
        first_half = np.logical_and((10 - 4) < x_r_1, x_r_1 <= 10)
        r_1[first_half] = (x_r_1[first_half] - (10 - 4)) / (10 - (10 - 4))

        second_half = np.logical_and(x_pixel[1] < x_r_1, x_r_1 < (10 + 4))
        r_1[second_half] = ((10 + 4) - x_r_1[second_half]) / ((10 + 4) - 10)

        expected_calls = [call(d, x, r_0, x_r_0, 2), call(d, x, r_1, x_r_1, 2)]

        for expected_call, real_call in zip(expected_calls, mock.call_args_list):
            np.testing.assert_array_equal(real_call[0][0], expected_call[1][0])
            np.testing.assert_array_equal(real_call[0][1], expected_call[1][1])
            np.testing.assert_array_equal(real_call[0][2], expected_call[1][2])
            np.testing.assert_array_equal(real_call[0][3], expected_call[1][3])
            self.assertEqual(real_call[0][4], expected_call[1][4])

    def test_return_r_pixel(self):

        x = np.arange(20)
        x_pixel = np.array([4.0, 8.0, 12.0])
        width_pixel = np.array([2.0, 2.0, 4.0])
        r_pixel = bi.return_r_pixel(x_pixel, width_pixel, x, fd.f_tophat)

        expected_r_pixel = np.zeros((3, 20))

        expected_r_pixel[0, 3:6] = 1.0
        expected_r_pixel[1, 7:10] = 1.0
        expected_r_pixel[2, 10:15] = 1.0

        numpy.testing.assert_almost_equal(r_pixel, expected_r_pixel)


if __name__ == "__main__":
    unittest.main()
