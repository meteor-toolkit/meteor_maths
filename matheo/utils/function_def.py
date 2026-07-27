"""
Functions to define commonly used function forms
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Callable, Iterator, Tuple


__author__ = "Sam Hunt"
__created__ = "27/10/2020"


def f_tophat(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    """
    Evaluate a top hat function along x

    :param x: coordinate array
    :param centre: x position of centre top hat
    :param width: FWHM width of top hat

    :return: top hat function
    """

    y = np.ones(x.shape)
    y[x < centre - width / 2] = 0
    y[x > centre + width / 2] = 0

    return y


def f_gaussian(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    """
    Evaluate a gaussian function along x

    :param x: coordinate array
    :param centre: mean of distribution
    :param width: FWHM of distribution

    :return: gaussian function
    """
    std = width / (2 * np.sqrt(2 * np.log(2)))
    return norm.pdf(x, centre, std) * np.sqrt(2 * np.pi) * std


def f_triangle(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    """
    Evaluate a top hat function along x

    :param x: coordinate array
    :param centre: x position of start of triangle
    :param width: half width of base of triangle (i.e. FWHM)

    :return: top hat function
    :rtype: numpy.ndarray
    """

    y = np.zeros(x.shape)

    start = centre - width
    stop = centre + width

    first_half = np.logical_and(start < x, x <= centre)
    y[first_half] = (x[first_half] - start) / (centre - start)

    second_half = np.logical_and(centre < x, x < stop)
    y[second_half] = (stop - x[second_half]) / (stop - centre)

    return y


def f_normalised(f: Callable, x: np.ndarray, *args, high_res_sampling: float = 0.001):
    """
    Return normalised f along x

    :param f: defining function
    :param x: coordinate array
    :param args: f parameters
    :param high_res_sampling: (default 0.001) sampling to evaluate area of function to normalise by

    :return: normalised function
    """

    y = f(x, *args)

    x_highres = np.arange(min(x), max(x), high_res_sampling)
    y_highres = f(x_highres, *args)

    y /= np.trapezoid(y_highres, x_highres)

    return y


def repeat_f(
    f: Callable,
    centres: np.ndarray,
    widths: np.ndarray,
    normalise: bool = False,
    x_sampling: float = 0.01,
    xlim_width: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates repeating functions along a coordinate axis

    .. note::
        Defines all function on common x coordinates, so may use a lot of memory if many bands defined with x sampling.
        For a lower memory solution try **matheo.utils.function_def.iter_fs()**..

    :param f: function to repeat
    :param centres: function centres
    :param widths: function widths
    :param normalise: (default True) switch to define if area of return SRFs should be normalised to 1
    :param x_sampling: sampling along function coordinates
    :param xlim_width: (default 3) multiple function widths to define function over

    :return: evaluated functions
    :return: evaluated function coordinates
    """

    fs = RepeatingFuncUtil(
        f=f,
        centres=centres,
        widths=widths,
        normalise=normalise,
        x_sampling=x_sampling,
        xlim_width=xlim_width,
    )

    return fs.return_fs()


def iter_f(
    f: Callable,
    centres: np.ndarray,
    widths: np.ndarray,
    normalise: bool = False,
    x_sampling: float = 0.01,
    xlim_width: float = 3.0,
) -> Iterator:
    """
    Returns iterator to evaluate repeating functions along a coordinate axis

    .. note::
        Offers a lower memory solution to **matheo.utils.function_def.return_fs()**.

    :param f: function to repeat
    :param centres: distribution centres
    :param widths: distribution widths
    :param normalise: (default True) switch to define if area of return SRFs should be normalised to 1
    :param x_sampling: sampling along function coordinates
    :param xlim_width: (default 3) multiple function widths to define function over

    :return: repeating function iterator
    """

    fs = RepeatingFuncUtil(
        f=f,
        centres=centres,
        widths=widths,
        normalise=normalise,
        x_sampling=x_sampling,
        xlim_width=xlim_width,
    )

    return iter(fs)


class RepeatingFuncUtil:
    """
    Helper class to define repeating functions along a coordinate axis

    :param f: function to repeat
    :param centres: distribution centres
    :param widths: distribution widths
    :param normalise: (default True) switch to define if area of return SRFs should be normalised to 1
    :param x_sampling: sampling along function coordinates
    :param xlim_width: (default 3) multiple function widths to define function over
    """

    def __init__(
        self,
        f: Callable,
        centres: np.ndarray,
        widths: np.ndarray,
        normalise: bool = False,
        x_sampling: float = 0.01,
        xlim_width: float = 3.0,
    ):

        # Set attributes from arguments
        self.f = f
        self.centres = centres
        self.widths = widths
        self.normalise = normalise
        self.x_sampling = x_sampling
        self.xlim_width = xlim_width

    def return_fs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates all repeating functions along a coordinate axis

        :return: evaluated functions
        :return: evaluated function coordinates
        """

        x = np.arange(
            min(self.centres - self.xlim_width * self.widths),
            max(self.centres + self.xlim_width * self.widths + 1),
            self.x_sampling,
        )

        # Evaluate function
        ys = np.zeros((len(self.centres), len(x)))
        for i_band, (centre, width) in enumerate(zip(self.centres, self.widths)):

            if self.normalise:
                ys[i_band, :] = f_normalised(self.f, x, centre, width)
            else:
                ys[i_band, :] = self.f(x, centre, width)

        return ys, x

    def __iter__(self):

        # Define counter
        self.i = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns ith function

        :return: evaluated function
        :return: evaluated function coordinates
        """

        # Iterate through bands
        if self.i < len(self.centres):

            centre = self.centres[self.i]
            width = self.widths[self.i]

            x = np.arange(
                centre - self.xlim_width * width,
                centre + self.xlim_width * width + 1,
                self.x_sampling,
            )

            if self.normalise:
                y = f_normalised(self.f, x, centre, width)
            else:
                y = self.f(x, centre, width)

            # Update counter
            self.i += 1

            return y, x

        else:
            raise StopIteration


if __name__ == "__main__":
    pass
