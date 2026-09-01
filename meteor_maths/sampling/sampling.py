"""Module of resampling functions"""

import re
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr
from scipy import spatial

__author__ = ["Maddie Stedman"]
__all__ = [
    "RegridCache",
    "Resampler",
    "nearest_neighbour_resample",
    "resample",
]


def _grid_spacing(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
) -> tuple[float, float, float, float]:
    """Typical pixel spacing of the source and target grids along each axis.

    Uses the median (rather than mean) grid step so that a single irregular
    gap in an otherwise regular grid doesn't skew the estimate for every
    other pixel. A grid with fewer than two unique coordinates along an axis
    (e.g. a single-column/single-row "degenerate" grid) has no defined pixel
    width there - returned as infinite spacing (rather than NaN), so that
    axis is later treated as carrying no density information instead of
    corrupting the estimate."""

    def spacing(coord: np.ndarray) -> float:
        diffs = np.diff(np.unique(coord))
        return float(np.median(diffs)) if diffs.size > 0 else np.inf

    return (
        spacing(x_source),
        spacing(y_source),
        spacing(x_target),
        spacing(y_target),
    )


#: When flooring an expected-count ratio to get an integer threshold, nudge
#: it up by this much first. Ratios that are mathematically exact integers
#: (e.g. a target pixel exactly 9x the area of a source pixel) can come out
#: as e.g. 8.999999999999996 after going through a coordinate rotation, due
#: to ordinary floating-point rounding in the sin/cos/hypot chain - without
#: this, floor() would round such a ratio down to 8, silently tightening the
#: threshold and making otherwise-identical rotated and unrotated grids
#: disagree on which pixels are valid.
_FLOOR_EPSILON = 1e-9

#: Below this, two array-adjacent grid points are treated as coincident
#: (see _local_axis_geometry) rather than as a genuine, very small spacing.
_DEGENERATE_SPACING_TOLERANCE = 1e-9


def _expected_source_count(dx_target, dy_target, dx_source, dy_source):
    """Expected number of source pixels within a target pixel's footprint,
    from the ratio of target to source spacing along each axis.

    Computed per-axis (rather than as a single area ratio) so that an axis
    along which either grid is degenerate (a single, constant coordinate -
    no defined pixel width, represented as infinite spacing by
    :func:`_grid_spacing`/:func:`_local_axis_geometry`) doesn't scale the
    estimate up or down: neither grid having real extent there means that
    axis simply carries no density information, rather than the undefined
    width being treated as either zero (collapsing the ratio to nothing,
    which forces every pixel invalid) or infinite (blowing the ratio up to
    an enormous threshold, which has the same effect).

    Shared by :func:`_estimate_n_min_source` (a single grid-wide spacing per
    axis) and :class:`RegridCache`'s local per-pixel estimate - both pass in
    either plain floats or same-shaped arrays, and numpy broadcasting/
    ``np.where`` handles either uniformly.
    """

    def axis_ratio(target_spacing, source_spacing):
        degenerate = np.isinf(target_spacing) | np.isinf(source_spacing)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = target_spacing / source_spacing
        return np.where(degenerate, 1.0, ratio)

    expected_n_source = axis_ratio(dx_target, dx_source) * axis_ratio(dy_target, dy_source)
    return np.maximum(1, np.floor(expected_n_source + _FLOOR_EPSILON))


def _estimate_n_min_source(dx_source: float, dy_source: float, dx_target: float, dy_target: float) -> int:
    """Estimate the number of source pixels expected to fall within a target
    pixel footprint, from the relative pixel spacing of the two grids."""
    return int(_expected_source_count(dx_target, dy_target, dx_source, dy_source))


def _local_axis_geometry(x: np.ndarray, y: np.ndarray, axis: int, direction: bool = True):
    """Per-pixel local grid spacing (and, unless ``direction=False``, unit
    direction vector) along one array axis of a structured 2D grid (as from
    ``numpy.meshgrid``), estimated from the distance to a pixel's two
    array-adjacent neighbours along that axis.

    Uses the *smaller* of the (Euclidean) distances to a pixel's two
    neighbours (rather than e.g. their average), so that a pixel sitting next
    to an unusually large gap doesn't inflate its footprint out into that gap
    - better to conservatively exclude a source pixel near the edge of its
    true footprint than to fold in source pixels from well outside it.

    A grid with only one point along this axis (e.g. a single-column source)
    has no defined spacing here - returned as infinite, so callers can treat
    this axis as carrying no width/density information, rather than as
    exactly zero (which would otherwise make a footprint check impossibly
    strict, or an expected-count ratio explode - see
    :func:`_expected_source_count`).

    The same applies, pixel by pixel, if two array-adjacent points merely
    happen to coincide (e.g. a duplicated scanline in real satellite
    geolocation data) even though the axis as a whole isn't degenerate: a
    literal zero spacing there isn't a meaningful "very high density"
    measurement, just a local data artifact, so it's promoted to infinite
    too rather than left at zero. Left as zero, it would either make that
    one pixel's footprint impossibly strict, or - since the underlying
    coordinate arrays are typically float64 while dividing a real number by
    it in :func:`_expected_source_count` produces ``inf``/``nan`` - silently
    corrupt the final ``int`` cast there (numpy wraps ``inf``/``nan`` cast to
    ``int`` to an arbitrary sentinel rather than raising).

    Returns the local spacing magnitude, and - unless ``direction=False``,
    which skips this (more expensive) part when only the magnitude is
    needed - the (x, y) unit vector pointing along this axis at each pixel.
    The direction is needed (rather than just treating this axis as aligned
    with the global x or y axis) so that the footprint it feeds into is
    rotation-invariant: if both grids are rotated together, this direction
    rotates with them, and the footprint check gives the same result as if
    neither had been rotated at all.
    """
    dx = np.diff(x, axis=axis)
    dy = np.diff(y, axis=axis)
    mag = np.hypot(dx, dy)

    if mag.shape[axis] == 0:
        spacing = np.full(x.shape, np.inf)
        if not direction:
            return spacing
        zeros = np.zeros(x.shape)
        return spacing, zeros, zeros

    def pad_before_after(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (1, 1)
        padded = np.pad(a, pad_width, mode="edge")
        before = np.take(padded, range(padded.shape[axis] - 1), axis=axis)
        after = np.take(padded, range(1, padded.shape[axis]), axis=axis)
        return before, after

    mag_before, mag_after = pad_before_after(mag)
    use_before = mag_before <= mag_after
    spacing = np.where(use_before, mag_before, mag_after)
    spacing = np.where(spacing <= _DEGENERATE_SPACING_TOLERANCE, np.inf, spacing)
    if not direction:
        return spacing

    mag_safe = np.maximum(mag, 1e-12)
    ux, uy = dx / mag_safe, dy / mag_safe
    ux_before, ux_after = pad_before_after(ux)
    uy_before, uy_after = pad_before_after(uy)
    direction_x = np.where(use_before, ux_before, ux_after)
    direction_y = np.where(use_before, uy_before, uy_after)
    return spacing, direction_x, direction_y


def _aggregate_valid(
    data_in_range: np.ndarray,
    idx: np.ndarray,
    n_target: int,
    include_sumsq: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Sum, (optionally) sum-of-squares, and count of the non-NaN values of
    ``data_in_range``, binned by ``idx``. NaNs are excluded from the
    aggregation entirely (rather than being summed in and propagating to
    poison the whole bin), and don't count towards a bin's contributing-pixel
    count either - so a target pixel fed by a mix of good and NaN source
    pixels is still averaged from the good ones, and is only masked invalid
    if too few good pixels remain, same as if it had too few source pixels
    in range to begin with.

    :param include_sumsq: whether to also compute the sum of squares (needed
        for a standard deviation). Skipped by default, since it's wasted
        work for callers - like RegridCache.regrid - that don't use it.
    """
    valid = ~np.isnan(data_in_range)
    valid_idx = idx[valid]
    valid_data = data_in_range[valid]
    sum_target = np.bincount(valid_idx, weights=valid_data, minlength=n_target)
    n_valid_target = np.bincount(valid_idx, minlength=n_target).astype(float)

    sumsq_target = None
    if include_sumsq:
        sumsq_target = np.bincount(valid_idx, weights=valid_data**2, minlength=n_target)

    return sum_target, sumsq_target, n_valid_target


@runtime_checkable
class Resampler(Protocol):
    """Common interface for resampling backends usable by :func:`resample`.

    A resampler is built once for a fixed pair of source/target grids (via
    whatever constructor arguments that particular algorithm needs), and can
    then be reused to regrid several 2D data arrays sharing those grids -
    e.g. different variables, or repeated calls in a loop - without redoing
    the expensive part (KDTree construction, triangulation, or whatever else
    a given algorithm needs to set up) each time.

    :func:`resample` only ever calls :meth:`regrid`. It has no knowledge of
    which concrete algorithm is in play, or of that algorithm's own
    constructor arguments (e.g. :class:`RegridCache`'s ``regular_grid``) - so
    adding a new resampling method (e.g. a scipy-based interpolator) means
    writing a class that satisfies this interface and registering it in
    ``_RESAMPLERS``, with no changes to :func:`resample` itself. Structural
    (``Protocol``) typing is used rather than a base class, so a resampler
    doesn't need to inherit from anything - it just needs a matching
    ``regrid`` method.
    """

    def regrid(self, data: np.ndarray, mask_invalid: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """Resample 2D source data (matching the source grid this resampler
        was built for) onto its target grid.

        :param data: source data as a 2D array
        :param mask_invalid: whether to compute and apply a validity mask, setting invalid target pixels to nan. Implementations should skip computing validity entirely when this is False, if that's the expensive part of the algorithm.
        :return: resampled 2D data, and a boolean mask that is True where a target pixel is valid (or None if mask_invalid is False)
        """
        ...


class RegridCache:
    """
    Nearest-neighbour :class:`Resampler`: precomputed regridding state for a
    fixed pair of source/target grids - the KDTree over the target grid, the
    mapping of each source pixel to its nearest target pixel, and the
    expected-source-count threshold (:attr:`n_min_source`) used to flag
    under-sampled target pixels. Note that which pixels actually end up
    flagged invalid also depends on the data being regridded (a NaN source
    pixel doesn't count towards a bin's contributing-pixel count), so that
    final mask is computed per data array in :meth:`regrid`, not cached here.

    Building a RegridCache is the expensive part of nearest-neighbour
    resampling. Build one instance for a given (x_source, y_source, x_target,
    y_target) and reuse it via :meth:`regrid` to resample multiple data
    arrays (e.g. different variables, or repeated calls in a loop) on the
    same grids, instead of recomputing the KDTree and index mapping each
    time - including across separate calls to :func:`resample`, by building
    a RegridCache upfront and passing it as ``resampler=``.

    Two ways of deciding a target pixel's own footprint (used both to decide
    which source pixels contribute to it, and how many are required for it to
    be considered valid) are available, via ``regular_grid``:

    - ``regular_grid=True``: a single grid-wide spacing estimate (median pixel
      step along each axis) is used for every target pixel. Cheap, but only
      accurate if both grids really do have uniform, axis-aligned spacing
      throughout.
    - ``regular_grid=False`` (default): each target pixel's footprint is
      estimated from its own array-adjacent neighbours instead of a single
      grid-wide value, and its expected source count from the local source
      point density around it. Correct for grids whose resolution varies
      across their extent (e.g. finer in one region than another, or
      anisotropic - different spacing in x than in y), at the cost of an
      extra KDTree query. Note this still assumes both grids are laid out as
      a structured 2D grid (as from ``numpy.meshgrid``) - a genuinely rotated
      or scattered grid is not fully accounted for, since the footprint
      check is still axis-aligned.

    Both modes fall back gracefully on a grid that's degenerate (has only one
    distinct coordinate) along one axis - e.g. a single-column/single-row
    source or target - by treating that axis as carrying no width/density
    information rather than as zero width (see :func:`_expected_source_count`).

    The (more expensive, source-density) part of this estimate is only
    computed lazily, the first time validity is actually needed - so it's
    never paid for if you only ever call :meth:`regrid` with
    ``mask_invalid=False``.
    """

    def __init__(
        self,
        x_source: np.ndarray,
        y_source: np.ndarray,
        x_target: np.ndarray,
        y_target: np.ndarray,
        regular_grid: bool = False,
    ):
        self.x_source = x_source
        self.y_source = y_source
        self.x_target = x_target
        self.y_target = y_target
        self.regular_grid = regular_grid
        self._n_min_source: int | np.ndarray | None = None

        self._grid_source = np.c_[x_source.ravel(), y_source.ravel()]
        grid_target = np.c_[x_target.ravel(), y_target.ravel()]

        # Create KDTree for target grid and find the nearest target pixel for
        # each source grid point. A source pixel is only kept if it falls
        # within its assigned target pixel's own footprint - otherwise, if the
        # source grid extends further than the target grid, that extra area
        # of extent would be folded into the edge pixels of the target grid
        # rather than simply being discarded as outside of it.
        tree = spatial.cKDTree(grid_target)
        _, idx = tree.query(self._grid_source, k=1)
        offset_x = self._grid_source[:, 0] - grid_target[idx, 0]
        offset_y = self._grid_source[:, 1] - grid_target[idx, 1]

        if regular_grid:
            self._dx_source, self._dy_source, self._dx_target, self._dy_target = _grid_spacing(
                x_source, y_source, x_target, y_target
            )
            dx_target_local = np.full(x_target.size, self._dx_target)
            dy_target_local = np.full(x_target.size, self._dy_target)
            # axis-aligned: the "column"/"row" directions are just the global x/y axes
            col_ux, col_uy = np.ones(x_target.size), np.zeros(x_target.size)
            row_ux, row_uy = np.zeros(x_target.size), np.ones(x_target.size)
        else:
            # per-target-pixel spacing *and direction* along each of the
            # target grid's two array axes, from array-adjacent neighbours.
            # Projecting onto the local direction (rather than assuming it's
            # aligned with the global x/y axes) makes this rotation-invariant:
            # if both grids are rotated together, the footprint check gives
            # the same result as if neither had been rotated at all. It also
            # varies correctly across grids whose resolution changes across
            # their extent, or that are anisotropic (different spacing along
            # one array axis than the other).
            dx_target_local, col_ux, col_uy = _local_axis_geometry(x_target, y_target, axis=1)
            dy_target_local, row_ux, row_uy = _local_axis_geometry(x_target, y_target, axis=0)
            dx_target_local = dx_target_local.ravel()
            dy_target_local = dy_target_local.ravel()
            col_ux, col_uy = col_ux.ravel(), col_uy.ravel()
            row_ux, row_uy = row_ux.ravel(), row_uy.ravel()

        # one shared footprint check (a source pixel is "in range" of its
        # assigned target pixel if its offset, projected onto that target
        # pixel's own local column/row directions, is within half the local
        # spacing along each) for both the regular and locally-varying case -
        # they differ only in what spacing/direction values feed into it
        local_col_coord = offset_x * col_ux[idx] + offset_y * col_uy[idx]
        local_row_coord = offset_x * row_ux[idx] + offset_y * row_uy[idx]
        within_x = np.abs(local_col_coord) <= dx_target_local[idx] / 2
        within_y = np.abs(local_row_coord) <= dy_target_local[idx] / 2

        self._dx_target_local = dx_target_local
        self._dy_target_local = dy_target_local
        self.source_in_range = within_x & within_y
        self.idx = idx[self.source_in_range]

    @property
    def n_min_source(self):
        self._ensure_n_min_source_computed()
        return self._n_min_source

    def _ensure_n_min_source_computed(self) -> None:
        if self._n_min_source is not None:
            return
        n_min_source: int | np.ndarray
        if self.regular_grid:
            n_min_source = _estimate_n_min_source(self._dx_source, self._dy_source, self._dx_target, self._dy_target)
        else:
            n_min_source = self._estimate_local_n_min_source()
        self._n_min_source = n_min_source

    def _estimate_local_n_min_source(self) -> np.ndarray:
        """Per-target-pixel expected source count, from that pixel's own
        local footprint spacing and the local source pixel spacing around it
        (that of whichever source pixel ends up nearest to it).

        Note this deliberately isn't a k-nearest-neighbour density estimate:
        that estimator is biased on a regular lattice (its k-th neighbour
        distance systematically over- or under-shoots vs. a Poisson process,
        e.g. by a factor of 4/pi for k=4), which is exactly the common case
        here - a fairly regular, if locally-varying, source grid."""
        dx_source_2d = _local_axis_geometry(self.x_source, self.y_source, axis=1, direction=False)
        dy_source_2d = _local_axis_geometry(self.x_source, self.y_source, axis=0, direction=False)
        source_tree = spatial.cKDTree(self._grid_source)
        _, nearest_source_idx = source_tree.query(np.c_[self.x_target.ravel(), self.y_target.ravel()], k=1)
        local_source_dx = dx_source_2d.ravel()[nearest_source_idx]
        local_source_dy = dy_source_2d.ravel()[nearest_source_idx]
        expected_n_source = _expected_source_count(
            self._dx_target_local,
            self._dy_target_local,
            local_source_dx,
            local_source_dy,
        )
        return expected_n_source.astype(int)

    def _regrid(
        self, data: np.ndarray, mask_invalid: bool = True, include_std: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Shared implementation behind :meth:`regrid` (the public,
        Resampler-protocol-facing entry point) and
        :func:`nearest_neighbour_resample` (which additionally wants the
        standard deviation of each bin's samples) - so the aggregation and
        masking logic lives in exactly one place regardless of which of
        those a caller needs."""
        data = data.ravel()[self.source_in_range]
        sum_target, sumsq_target, n_valid_target = _aggregate_valid(
            data, self.idx, self.x_target.size, include_sumsq=include_std
        )
        with np.errstate(invalid="ignore"):
            data_target = sum_target / n_valid_target
            std_target: np.ndarray | None = None
            if include_std:
                std_target = (sumsq_target / n_valid_target - data_target**2.0) ** 0.5

        data_target = data_target.reshape(self.x_target.shape)
        if std_target is not None:
            std_target = std_target.reshape(self.x_target.shape)

        if not mask_invalid:
            return data_target, std_target, None

        mask_valid = (n_valid_target >= self.n_min_source).reshape(self.x_target.shape)
        data_target = np.where(mask_valid, data_target, np.nan)
        if std_target is not None:
            std_target = np.where(mask_valid, std_target, np.nan)
        return data_target, std_target, mask_valid

    def regrid(self, data: np.ndarray, mask_invalid: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Resample 2D data onto this cache's target grid, by averaging the
        source pixels nearest to each target pixel.

        :param data: source data as a 2D array, matching the source grid shape this cache was built from
        :param mask_invalid: boolean for setting invalid target pixels (fewer than n_min_source non-NaN contributing source pixels) to nan. If False, validity is not computed at all (it can be the expensive part of the estimate for irregular grids), and None is returned in its place.
        :return: resampled 2D data, boolean mask that is True where a target pixel is valid (or None if mask_invalid is False)
        """
        data_target, _, mask_valid = self._regrid(data, mask_invalid=mask_invalid, include_std=False)
        return data_target, mask_valid


#: Resampling backends usable by resample()'s ``method`` argument, when no
#: pre-built ``resampler`` is passed in. Register a new algorithm here (a
#: class satisfying the Resampler protocol) to make it available by name -
#: resample() itself needs no changes. For any algorithm-specific
#: configuration (e.g. RegridCache's regular_grid), build the resampler
#: directly and pass it as resample()'s ``resampler=`` argument instead of
#: going through this registry, which only ever uses each class's defaults.
_RESAMPLERS: dict[str, type[Resampler]] = {
    "nearest_neighbour": RegridCache,
}


def nearest_neighbour_resample(
    data: np.ndarray,
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    mask_invalid: bool = True,
    regular_grid: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Resample 2D data by averaging nearest neighbour values. Invalid pixels set to nan if mask=True - invalid pixels defined as those with fewer than the automatically estimated expected number of source pixels binned to form the sample.

    For repeated resampling of different data on the same source/target grids
    (e.g. multiple variables, or in a loop), build a :class:`RegridCache` once
    and use its :meth:`~RegridCache.regrid` method instead, to avoid
    rebuilding the KDTree and nearest-neighbour mapping on every call.

    :param data: data as 2D array
    :param x_source: x coordinates of source grid
    :param y_source: y coordinates of source grid
    :param x_target: x coordinates of target grid
    :param y_target: y coordinates of target grid
    :param mask_invalid: boolean for setting invalid edge pixels to nan. If False, validity is never computed (skipping the more expensive part of the estimate for irregular grids)
    :param regular_grid: whether both grids have uniform, axis-aligned pixel spacing - see :class:`RegridCache`
    :return: resampled 2D data, standard deviation of samples
    """
    cache = RegridCache(
        x_source,
        y_source,
        x_target,
        y_target,
        regular_grid=regular_grid,
    )
    data_target, std_target, _ = cache._regrid(data, mask_invalid=mask_invalid, include_std=True)
    return data_target, std_target


def resample(
    var: str,
    ds: xr.Dataset,
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    mask_invalid: bool = True,
    method: str = "nearest_neighbour",
    resampler: Resampler | None = None,
) -> np.ndarray:
    """
    Resample variable data.

    This function itself is resampling-algorithm-agnostic: it only handles
    picking out the right 2D slices of ``var`` (for 2D/3D/4D data), lining
    their axes up with x_source/y_source's own layout, and building the
    output array - the actual resampling for each 2D slice is delegated to
    a :class:`Resampler` (see ``method``/``resampler`` below). To add a new
    resampling algorithm, write a class satisfying the ``Resampler``
    interface and register it in ``_RESAMPLERS``; this function does not
    need to change.

    :param var: variable to resample
    :param ds: dataset containing data to resample
    :param x_source: x coordinates for source data
    :param y_source: y coordinates for source data
    :param x_target: x coordinates for target data
    :param y_target: y coordinates for target data
    :param mask_invalid: boolean for setting invalid edge pixels to nan
    :param method: name of a registered resampling algorithm to use (see ``_RESAMPLERS``), built with its default settings. Ignored if ``resampler`` is given.
    :param resampler: an optional, already-built :class:`Resampler` (e.g. a :class:`RegridCache`) for the given source/target grids. Passing one in is how to use non-default settings for a given algorithm (e.g. ``RegridCache(..., regular_grid=True)``) - those settings are specific to each algorithm, so aren't exposed as arguments here. Building one once and reusing it also avoids recomputing e.g. a KDTree on every call, when resampling several variables that share the same grids.
    :return: array of resampled variable data
    """
    # an unknown method should raise regardless of whether resampling turns
    # out to be unnecessary below - checked first so a typo'd/stale method
    # name is never silently ignored just because the grids happen to match
    if resampler is None:
        try:
            resampler_cls: Callable[..., Resampler] = _RESAMPLERS[method.lower()]
        except KeyError:
            raise NotImplementedError(f"Method {method} not implemented.") from None

    # if source and target grid are the same, no resampling is necessary
    if x_source.shape == x_target.shape and np.all(x_source == x_target) and np.all(y_source == y_target):
        return ds[var].values

    if resampler is None:
        resampler = resampler_cls(x_source, y_source, x_target, y_target)

    # create empty array for the processed data
    # matches a dim literally named "x"/"y", or one with an "x_"/"y_" prefix (e.g.
    # "x_10m", to disambiguate multiple grids in the same dataset) -- previously
    # only the prefixed form matched, forcing callers with plain "x"/"y" dims to
    # rename them just to satisfy this lookup.
    x_dim = next(dim for dim in ds[var].dims if re.search("^x(_|$)", str(dim)))
    y_dim = next(dim for dim in ds[var].dims if re.search("^y(_|$)", str(dim)))
    x_dim_idx = ds[var].dims.index(x_dim)
    y_dim_idx = ds[var].dims.index(y_dim)
    shape = list(ds[var].values.shape)
    shape[x_dim_idx] = x_target.shape[-1]
    shape[y_dim_idx] = x_target.shape[-2]
    data_intxy = np.zeros(shape)

    # a Resampler always works on 2D arrays laid out the same way as
    # x_source/x_target (last axis = x, second-to-last = y, following
    # numpy.meshgrid's convention). If the variable's own x/y dims are
    # instead ordered [x, y], its 2D slices need transposing on the way in
    # and out to line up with that convention.
    transpose_xy = x_dim_idx < y_dim_idx

    def _regrid_2d(data_2d: np.ndarray) -> np.ndarray:
        if transpose_xy:
            data_2d = data_2d.T
        result = resampler.regrid(data_2d, mask_invalid=mask_invalid)[0]
        return result.T if transpose_xy else result

    values = ds[var].values
    if data_intxy.ndim == 2:
        data_intxy = _regrid_2d(values)
    elif data_intxy.ndim == 3:
        n_leading = len(ds[ds[var].dims[0]].values)
        for i in range(n_leading):
            data_intxy[i] = _regrid_2d(values[i])
    elif data_intxy.ndim == 4:
        n_leading0 = len(ds[ds[var].dims[0]].values)
        n_leading1 = len(ds[ds[var].dims[1]].values)
        for i in range(n_leading0):
            for j in range(n_leading1):
                data_intxy[i, j] = _regrid_2d(values[i, j])
    else:
        raise NotImplementedError(
            f"Resampling not implemented for {data_intxy.ndim} dims. Data must have 2, 3 or 4 dims."
        )

    return data_intxy


if __name__ == "__main__":
    pass
