.. atbd - algorithm theoretical basis
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/12/22

.. _atbd:

Algorithm Theoretical Basis
===========================

Band integration
###################

Earth Observation satellites typically have one or more sensors with multiple bands.
The spectral response function (SRF) of each band of the sensor describes its relative sensitivity
to energy of different wavelengths and is normally determined in the laboratory using
a tunable laser or a scanning monochromator.

The band data observed by the sensor are given by:

.. math::
   d_{\mathrm{int}} = \frac{\int r(x_r) d(x) \!\mathrm{d} x}{\int r(x_r) \!\mathrm{d} x}

where:

* :math:`d` - data
* :math:`x` - data coordinates
* :math:`r` - band response function
* :math:`x_r` - band response function coordinates

In the case that :math:`x = x_r` and :math:`x` is evenly-sampled (i.e. a common spacing between every consecutive :math:`x` element), this reduces to:


.. math:: d_{\mathrm{int}} = \frac{r \cdot d}{\sum r}

This formulation also applies for the case where :math:`r` defines multiple band response functions as an N x M array, where N is number of response bands and M is the length of :math:`x_r`.

Within meteor_maths, the meteor_maths.band_integration.band_int() function implements this spectral integration ove the SRF in an efficient way (automatically using either the above sum and dot product, or trapezium rule depending on whether the coordinates are evenly-sampled).
Alternatively, the meteor_maths.band_integration.spectral_band_int_sensor() function can be used to automatically look up the values of the SRF using pyspectral, rather than manually providing :math:`r` and :math:`x_r`.

Uncertainties can also be provided on the data, coordinates and SRF, in which case these will be propagated to the band data observed by the sensor using the `punpy <https://punpy.readthedocs.io/en/latest/>`_ Monte Carlo approach.

Utilities
############

There are also a number of functions defined in meteor_maths.utils.function_def, which can be used to build SRF of a specific shape, or for other general uses.
These functions include tophat, Gaussian and triangular functions, as well as utilities to make normalised and repeating functions.
