import numpy as np
from numba import njit, prange, typed
from scipy.ndimage import map_coordinates

from HARK.core import MetricObject

AVAILABLE_TARGETS = ["cpu", "parallel"]

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates

    CUPY_AVAILABLE = True
    AVAILABLE_TARGETS.append("gpu")
except ImportError:
    CUPY_AVAILABLE = False


MC_KWARGS = {
    "order": 1,  # order of interpolation
    "mode": "nearest",  # how to handle extrapolation
    "cval": 0.0,  # value to use for extrapolation
    "prefilter": False,  # whether to prefilter input
}


class _RegularGridInterp(MetricObject):
    """
    Abstract class for interpolating on a regular grid. Sets up
    structure for using different targets (cpu, parallel, gpu).
    Takes in arguments to be used by `map_coordinates`.
    """

    distance_criteria = ["values"]

    def __init__(self, values, target="cpu", **kwargs):
        """
        Initialize a regular grid interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        target : str, optional
            Determines which target to use for interpolation.
            Options are "cpu", "parallel", and "gpu".
            If "cpu", uses numpy and scipy.
            If "parallel", uses numba and scipy.
            If "gpu", uses cupy.

        Raises
        ------
        ValueError
            Target is invalid.
        """
        if target not in AVAILABLE_TARGETS:
            raise ValueError("Invalid target.")
        self.target = target

        self.mc_kwargs = MC_KWARGS.copy()
        # update mc_kwargs with any kwargs that are in MC_KWARGS
        self.mc_kwargs.update((k, v) for k, v in kwargs.items() if k in MC_KWARGS)

        if target in ["cpu", "parallel"]:
            self.values = np.asarray(values)
        elif target == "gpu":
            self.values = cp.asarray(values)

        self.ndim = self.values.ndim  # should match number of grids
        self.shape = self.values.shape  # should match points in each grid

    def __call__(self, *args):
        """
        Interpolates arguments on the regular grid.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each argument.

        Raises
        ------
        ValueError
            Number of argumets does not match number of dimensions.
        """
        if self.target in ["cpu", "parallel"]:
            args = np.asarray(args)
        elif self.target == "gpu":
            args = cp.asarray(args)

        if args.shape[0] != self.ndim:
            raise ValueError("Number of arguments must match number of dimensions.")

        coordinates = self._get_coordinates(args)
        return self._map_coordinates(coordinates)

    def _get_coordinates(self, args):
        """
        Abstract method for getting coordinates for interpolation.

        Parameters
        ----------
        args : np.ndarray
            Arguments to be interpolated.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def _map_coordinates(self, coordinates):
        """
        Uses coordinates to interpolate on the regular grid with
        `map_coordinates` from scipy or cupy, depending on target.

        Parameters
        ----------
        coordinates : np.ndarray
            Index coordinates for interpolation.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each coordinate.
        """
        if self.target in ["cpu", "parallel"]:
            # there is no parallelization for scipy map_coordinates
            output = map_coordinates(
                self.values, coordinates.reshape(self.ndim, -1), **self.mc_kwargs
            )

        elif self.target == "gpu":
            output = cupy_map_coordinates(
                self.values, coordinates.reshape(self.ndim, -1), **self.mc_kwargs
            )

        return output.reshape(coordinates[0].shape)


class MultivariateInterp(_RegularGridInterp):
    """
    Multivariate Interpolator on a regular grid. Maps functional coordinates
    to index coordinates and uses `map_coordinates` from scipy or cupy.
    """

    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids, target="cpu", **kwargs):
        """
        Initialize a multivariate interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        grids : _type_
            1D grids for each dimension.
        target : str, optional
            One of "cpu", "parallel", or "gpu". Determines
            hardware to use for interpolation.
        """

        super().__init__(values, target=target, **kwargs)

        if target == "cpu":
            self.grids = [np.asarray(grid) for grid in grids]
        elif target == "parallel":
            self.grids = typed.List(grids)
        elif target == "gpu":
            self.grids = [cp.asarray(grid) for grid in grids]

        if not (self.ndim == len(self.grids)):
            raise ValueError("Number of grids must match number of dimensions.")

        if not all(self.shape[i] == grid.size for i, grid in enumerate(self.grids)):
            raise ValueError("Values shape must match points in each grid.")

    def _get_coordinates(self, args):
        """
        For each argument, finds the index coordinates for interpolation.

        Parameters
        ----------
        args : np.ndarray
            Arguments to be interpolated.

        Returns
        -------
        np.ndarray
            Index coordinates for interpolation.
        """

        if self.target == "cpu":
            coordinates = np.empty_like(args)
            for dim, grid in enumerate(self.grids):  # for each dimension
                coordinates[dim] = np.interp(  # x, xp, fp (new x, x points, y values)
                    args[dim], grid, np.arange(self.shape[dim])
                )
        elif self.target == "parallel":
            coordinates = _nb_interp(self.grids, args)
        elif self.target == "gpu":
            coordinates = cp.empty_like(args)
            for dim, grid in enumerate(self.grids):  # for each dimension
                coordinates[dim] = cp.interp(  # x, xp, fp (new x, x points, y values)
                    args[dim], grid, cp.arange(self.shape[dim])
                )

        return coordinates


@njit(parallel=True, cache=True, fastmath=True)
def _nb_interp(grids, args):
    """
    Just-in-time compiled function for interpolating on a regular grid.

    Parameters
    ----------
    grids : np.ndarray
        1D grids for each dimension.
    args : np.ndarray
        Arguments to be interpolated.

    Returns
    -------
    np.ndarray
        Index coordinates for each argument.
    """

    coordinates = np.empty_like(args)
    for dim in prange(args.shape[0]):
        coordinates[dim] = np.interp(args[dim], grids[dim], np.arange(grids[dim].size))

    return coordinates


class _CurvilinearGridInterp(_RegularGridInterp):
    """
    Abstract class for interpolating on a curvilinear grid.
    """

    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids, target="cpu"):
        """
        Initialize a curvilinear grid interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a curvilinear grid.
        grids : np.ndarray
            ND curvilinear grids for each dimension
        target : str, optional
            One of "cpu", "parallel", or "gpu".
        """

        super().__init__(values, target=target)

        if target in ["cpu", "parallel"]:
            self.grids = np.asarray(grids)
        elif target == "gpu":
            self.grids = cp.asarray(grids)

        assert (
            self.ndim == self.grids[0].ndim
        ), "Number of grids must match number of dimensions."
        assert (
            self.shape == self.grids[0].shape
        ), "Values shape must match points in each grid."


class WarpedInterpOnInterp2D(_CurvilinearGridInterp):
    """
    Warped Grid Interpolation on a 2D grid.
    """

    def __call__(self, *args, axis=1):
        """
        Interpolate on a warped grid using the Warped Grid Interpolation
        method described in `EGM$^n$`.

        Parameters
        ----------
        axis : int, 0 or 1
            Determines which axis to use for linear interpolators.
            Setting to 0 may fix some issues where interpolation fails.

        Returns
        -------
        np.ndarray
            Interpolated values on a warped grid.

        Raises
        ------
        ValueError
            Number of arguments doesn't match number of dimensions.
        """

        if self.target in ["cpu", "parallel"]:
            args = np.asarray(args)
        elif self.target == "gpu":
            args = cp.asarray(args)

        if args.shape[0] != self.ndim:
            raise ValueError("Number of arguments must match number of dimensions.")

        if self.target == "cpu":
            output = self._target_cpu(args, axis)
        elif self.target == "parallel":
            output = self._target_parallel(args, axis)
        elif self.target == "gpu":
            output = self._target_gpu(args, axis)

        return output

    def _target_cpu(self, args, axis):
        """
        Uses numpy to interpolate on a warped grid.

        Parameters
        ----------
        args : np.ndarray
            Coordinates to be interpolated.
        axis : int, 0 or 1
            See `WarpedInterpOnInterp2D.__call__`.

        Returns
        -------
        np.ndarray
            Interpolated values on arguments.
        """

        shape = args[0].shape  # original shape of arguments
        size = args[0].size  # number of points in arguments
        shape_axis = self.shape[axis]  # number of points in axis

        # flatten arguments by dimension
        args = args.reshape((self.ndim, -1))

        y_intermed = np.empty((shape_axis, size))
        z_intermed = np.empty((shape_axis, size))

        for i in range(shape_axis):
            # for each dimension, interpolate the first argument
            grids0 = np.take(self.grids[0], i, axis=axis)
            grids1 = np.take(self.grids[1], i, axis=axis)
            values = np.take(self.values, i, axis=axis)
            y_intermed[i] = np.interp(args[0], grids0, grids1)
            z_intermed[i] = np.interp(args[0], grids0, values)

        output = np.empty_like(args[0])

        for j in range(size):
            y_temp = y_intermed[:, j]
            z_temp = z_intermed[:, j]

            if y_temp[0] > y_temp[-1]:
                # reverse
                y_temp = y_temp[::-1]
                z_temp = z_temp[::-1]

            output[j] = np.interp(args[1][j], y_temp, z_temp)

        return output.reshape(shape)

    def _target_parallel(self, args, axis):
        """
        Uses numba to interpolate on a warped grid.

        Parameters
        ----------
        args : np.ndarray
            Coordinates to be interpolated.
        axis : int, 0 or 1
            See `WarpedInterpOnInterp2D.__call__`.

        Returns
        -------
        np.ndarray
            Interpolated values on arguments.
        """

        return nb_interp_piecewise(args, self.grids, self.values, axis)

    def _target_gpu(self, args, axis):
        """
        Uses cupy to interpolate on a warped grid.

        Parameters
        ----------
        args : np.ndarray
            Coordinates to be interpolated.
        axis : int, 0 or 1
            See `WarpedInterpOnInterp2D.__call__`.

        Returns
        -------
        np.ndarray
            Interpolated values on arguments.
        """

        shape = args[0].shape  # original shape of arguments
        size = args[0].size  # number of points in arguments
        shape_axis = self.shape[axis]  # number of points in axis

        args = args.reshape((self.ndim, -1))

        y_intermed = cp.empty((shape_axis, size))
        z_intermed = cp.empty((shape_axis, size))

        for i in range(shape_axis):
            # for each dimension, interpolate the first argument
            grids0 = cp.take(self.grids[0], i, axis=axis)
            grids1 = cp.take(self.grids[1], i, axis=axis)
            values = cp.take(self.values, i, axis=axis)
            y_intermed[i] = cp.interp(args[0], grids0, grids1)
            z_intermed[i] = cp.interp(args[0], grids0, values)

        output = cp.empty_like(args[0])

        for j in range(size):
            y_temp = y_intermed[:, j]
            z_temp = z_intermed[:, j]

            if y_temp[0] > y_temp[-1]:
                # reverse
                y_temp = y_temp[::-1]
                z_temp = z_temp[::-1]

            output[j] = cp.interp(args[1][j], y_temp, z_temp)

        return output.reshape(shape)

    def warmup(self):
        self(*self.grids)

        return None


@njit(parallel=True, cache=True, fastmath=True)
def nb_interp_piecewise(args, grids, values, axis):
    """
    Just-in-time compiled function to interpolate on a warped grid.

    Parameters
    ----------
    args : np.ndarray
        Arguments to be interpolated.
    grids : np.ndarray
        Curvilinear grids for each dimension.
    values : np.ndarray
        Functional values on a curvilinear grid.
    axis : int, 0 or 1
        See `WarpedInterpOnInterp2D.__call__`.


    Returns
    -------
    np.ndarray
        Interpolated values on arguments.
    """

    shape = args[0].shape  # original shape of arguments
    size = args[0].size  # number of points in arguments
    shape_axis = values.shape[axis]  # number of points in axis

    # flatten arguments by dimension
    args = args.reshape((values.ndim, -1))

    y_intermed = np.empty((shape_axis, size))
    z_intermed = np.empty((shape_axis, size))

    for i in prange(shape_axis):
        # for each dimension, interpolate the first argument
        grids0 = grids[0][i] if axis == 0 else grids[0][:, i]
        grids1 = grids[1][i] if axis == 0 else grids[1][:, i]
        vals = values[i] if axis == 0 else values[:, i]
        y_intermed[i] = np.interp(args[0], grids0, grids1)
        z_intermed[i] = np.interp(args[0], grids0, vals)

    output = np.empty_like(args[0])

    for j in prange(size):
        y_temp = y_intermed[:, j]
        z_temp = z_intermed[:, j]

        if y_temp[0] > y_temp[-1]:
            # reverse
            y_temp = y_temp[::-1]
            z_temp = z_temp[::-1]

        output[j] = np.interp(args[1][j], y_temp, z_temp)

    return output.reshape(shape)


class _UnstructuredGridInterp(_CurvilinearGridInterp):
    """
    Abstract class for interpolation on unstructured grids.
    """

    def __init__(self, values, grids, target="cpu"):
        """
        Initialize interpolation on unstructured grids.

        Parameters
        ----------
        values : np.ndarray
            Functional values on an unstructured grid.
        grids : np.ndarray
            ND unstructured grids for each dimension.
        target : str, optional
            One of "cpu", "parallel", or "gpu".
        """

        super().__init__(values, grids, target=target)
        # remove non finite values that might result from
        # sequential endogenous grid method
        condition = np.logical_and.reduce([np.isfinite(grid) for grid in self.grids])
        condition = np.logical_and(condition, np.isfinite(self.values))
        self.values = self.values[condition]
        self.grids = self.grids[:, condition]
        self.ndim = self.grids.shape[0]
