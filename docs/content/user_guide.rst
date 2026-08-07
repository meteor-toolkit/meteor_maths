.. user guide
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/12/22

.. _userguide:

Using meteor_maths
=============

Band integration
###################
Band integration can be done in meteor_maths in a number of ways, depending on the input data one wants to use.
The most general approach is to use the band_int() function, in which the user supplies the band response function(s) as numpy arrays.
In the meteor_maths band_integration arguments:

* :math:`d` - the data to be band integrated
* :math:`x` - the data coordinates
* :math:`r` - band response function(s). For a single band, a 1D length-M array is required, where M is the length of ``x_r``. Multiple bands may be defined in an N x M array, where N is number of response bands.
* :math:`x_r` - band response function coordinates

In the band_int() function, these are passed as numpy arrays::

   import numpy as np
   import meteor_maths.band_integration as bi
   from meteor_maths.utils import function_def as fd

   x = np.arange(0, 100, 0.01)
   d = (0.02 * x) ** 3 + (-0.2 * x) ** 2 + (-3 * x) + 100

   x_r = np.arange(30, 70, 0.001)
   r = fd.f_triangle(x_r, 50, 5)

   d_band = bi.band_int(d, x, r, x_r)

The band integrated data `d_band` will here contain a numpy array with a single value.
If a multidimensional response function is provided, d_band will have the
band integrated data for all bands. Note that the f_triangle() function was used to define
the response function. See the Utilities Section below for further information on these helper functions.

Another way of performing band integration using meteor_maths is with the pixel_int() function.
For this function, instead of providing the spectral response functions as numpy arrays,
it is possible to just provide the centres, widths and shapes of the response functions, where:

* `x_pixel` - centre of band response per pixel
* `width_pixel` - width of band response per pixel
* `band_shape` - functional shape of response band - must be either a defined name, one of 'triangle', 'tophat', or 'gaussian', or a python function with the interface `f(x, centre, width)`, where `x` is a numpy array of the x coordinates to define the function along, `centre` is the response band centre, and `width` is the response band width. The default is `triangle`.

For example::

   d = np.zeros(12)
   x = np.arange(12)
   x_pixel = np.array([5, 10])
   width_pixel = np.array([2, 4])

   d_band = bi.pixel_int(
      d=d,
      x=x,
      x_pixel=x_pixel,
      width_pixel=width_pixel,
   )

Finally, when doing spectral band integration for an earth observation sensor which is included in pyspectral (see list `here <https://pyspectral.readthedocs.io/en/master/platforms_supported.html>`_),
it is possible to just specify the platform and sensor as a string using the spectral_band_int_sensor() function::

   d = np.random.random((3, 4, 11))
   wl = np.arange(400, 510, 10)

   d_band, wl_band = bi.spectral_band_int_sensor(
      d,
      wl,
      d_axis_wl=2,
      platform_name="Sentinel-2A",
      sensor_name="msi",
   )

Note that here we specified the wavelength dimension in d using the d_axis_wl keyword.

It is also possible to propagate uncertainties through all these functions. There are optional keywords such as:

* `u_d` - uncertainty on the data to be band integrated
* `u_x` - uncertainty on the data coordinates
* `u_r` - uncertainty on band response function(s). For a single band, a 1D length-M array is required, where M is the length of ``x_r``. Multiple bands may be defined in an N x M array, where N is number of response bands.
* `u_x_r` - uncertainty on band response function coordinates

When any of these optional keyword are set, uncertainties are propagated using a Monte Carlo approach with 10000 iterations using `punpy <https://punpy.readthedocs.io/en/latest/>`_, which is part of the `CoMet toolkit <https://www.comet-toolkit.org/>`_.

There are also a number of additional functions (such as higher dimension version of the above, e.g. the band_int2ax() function) and other optional keywords.
For explanations on these we refer to the meteor_maths API.

Resampling
###################
meteor_maths also provides tools for resampling gridded data from one 2D grid onto another, e.g. when pairing up two Earth observation products defined on different grids.

The simplest way to do this is with the nearest_neighbour_resample() function, which averages the source pixels nearest to each target pixel::

   import numpy as np
   from meteor_maths.sampling import nearest_neighbour_resample

   x_source, y_source = np.meshgrid(np.arange(20), np.arange(20))
   data_source = np.random.random(x_source.shape)
   x_target, y_target = np.meshgrid(np.arange(1, 19, 2.0), np.arange(1, 19, 2.0))

   data_target, std_target = nearest_neighbour_resample(
      data_source, x_source, y_source, x_target, y_target
   )

Target pixels for which too few source pixels were available (estimated automatically from the relative density of the two grids) are set to nan by default - pass ``mask_invalid=False`` to disable this.

For xarray Datasets with 2, 3 or 4 dimensional variables, use the resample() function instead, which resamples a variable's x/y dimensions while looping over any other (e.g. band or time) dimensions::

   import xarray as xr
   from meteor_maths.sampling import resample

   ds = xr.Dataset(data_vars=dict(var=(["y_source", "x_source"], data_source)))
   data_target = resample("var", ds, x_source, y_source, x_target, y_target)

When resampling several variables that share the same source/target grids (e.g. in a loop), build a RegridCache once and reuse it, to avoid recomputing the underlying KDTree on every call::

   from meteor_maths.sampling import RegridCache

   cache = RegridCache(x_source, y_source, x_target, y_target)
   for var in ["var_a", "var_b"]:
      data_target = resample(
         var, ds, x_source, y_source, x_target, y_target, resampler=cache
      )

Utilities
############

There are also a number of functions defined in meteor_maths.utils.function_def, which can be used to build response functions of a specific shape, or for other general uses.
These functions include tophat, Gaussian and triangular functions, as well as utilities to make normalised and repeating functions::

   from meteor_maths.utils import function_def as fd
   import numpy as np

   x = np.arange(0, 11, 1)
   y = fd.f_tophat(x, 5, 4)
   y = fd.f_triangle(x, 5, 2)
   y = fd.f_gaussian(x, 5, 2*np.sqrt(2*np.log(2))*2/2)
   y = fd.f_normalised(fd.f_tophat, x, 5, 4)
   y, x = fd.repeat_f(
      f=fd.f_tophat,
      centres=np.array([5.0, 6.0, 7.0]),
      widths=np.array([2.0, 4.0, 8.0]),
      x_sampling=1.0,
      xlim_width=1.5 / 2
   )
