import cupy as cp
import jax.numpy as jnp
import numpy as np
from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates
from jax import jit as jax_jit
from jax.scipy.ndimage import map_coordinates as jax_map_coordinates
from scipy.ndimage import map_coordinates

from HARK.core import MetricObject
from numba import njit, prange

DIM_MESSAGE = "Dimension mismatch."


class MultInterp(MetricObject):

    distance_criteria = ["input", "grids", "order", "mode", "cval", "prefilter"]

    def __init__(
        self,
        input,
        grids,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
        target="cpu",
    ):

        assert target in ["cpu", "numba", "cupy", "jax"], "Invalid target."

        if target == "cpu" or target == "numba":
            import numpy as xp
        elif target == "cupy":
            import cupy as xp
        elif target == "jax":
            import jax.numpy as xp

        self.input = xp.asarray(input)
        self.grids = xp.asarray(grids)
        self.order = order
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.target = target

        self.ndim = input.ndim  # should match number of grids
        self.shape = input.shape  # should match points in each grid

        assert self.ndim == self.grids.shape[0], DIM_MESSAGE

    def __call__(self, *args):

        if self.target == "cpu" or self.target == "numba":
            import numpy as xp
        elif self.target == "cupy":
            import cupy as xp
        elif self.target == "jax":
            import jax.numpy as xp

        args = xp.asarray(args)
        assert args.shape[0] == self.ndim, DIM_MESSAGE

        coordinates = xp.empty_like(args)

        if self.target == "cpu":
            output = self._target_cpu(args, coordinates)
        elif self.target == "numba":
            output = self._target_numba(args, coordinates)
        elif self.target == "cupy":
            output = self._target_cupy(args, coordinates)
        elif self.target == "jax":
            output = self._target_jax(args, coordinates)

        return output

    def _map_coordinates(self, args, coordinates):

        output = map_coordinates(
            self.input,
            coordinates.reshape(args.shape[0], -1),
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            prefilter=self.prefilter,
        ).reshape(args[0].shape)

        return output

    def _target_cpu(self, args, coordinates):

        for dim in range(args.shape[0]):
            arg_grid = self.grids[dim]
            new_args = args[dim]
            coordinates[dim] = np.interp(new_args, arg_grid, np.arange(arg_grid.size))

        output = self._map_coordinates(args, coordinates)

        return output

    def _target_numba(self, args, coordinates):

        nb_interp(self.grids, args, coordinates)
        output = self._map_coordinates(args, coordinates)

        return output

    def _target_cupy(self, args, coordinates):

        ndim = args.shape[0]

        for dim in range(ndim):
            arg_grid = self.grids[dim]
            new_args = args[dim]
            coordinates[dim] = cp.interp(new_args, arg_grid, cp.arange(arg_grid.size))

        output = cupy_map_coordinates(
            self.input,
            coordinates.reshape(ndim, -1),
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            prefilter=self.prefilter,
        ).reshape(args[0].shape)

        return output

    def _target_jax(self, args, coordinates):
        return jax_interp(self.input, self.grids, args, coordinates)

    def interp(self, *args):

        if self.target == "cupy":
            import cupy as xp
            from cupyx.scipy.ndimage import map_coordinates
        else:
            import numpy as xp
            from scipy.ndimage import map_coordinates

        # last index is the element

        args = xp.asarray(args)
        ndim = args.shape[0]
        out_shape = args[0].shape

        assert ndim == self.ndim, DIM_MESSAGE

        coordinates = xp.empty_like(args)

        #####

        if self.target == "parallel":
            nb_interp(self.grids, args, coordinates)
        else:
            for dim in range(ndim):
                arg_grid = self.grids[dim]
                new_args = args[dim]
                coordinates[dim] = xp.interp(
                    new_args, arg_grid, xp.arange(arg_grid.size)
                )

        output = map_coordinates(
            self.input,
            coordinates.reshape(ndim, -1),
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            prefilter=self.prefilter,
        ).reshape(out_shape)

        return output


@njit(parallel=True, cache=True, fastmath=True)
def nb_interp(grids, args, coordinates):

    for dim in prange(args.shape[0]):
        arg_grid = grids[dim]
        new_args = args[dim]
        coordinates[dim] = np.interp(new_args, arg_grid, np.arange(arg_grid.size))

    return coordinates


class JaxInterp(MetricObject):
    distance_criteria = ["input", "grids", "order", "mode", "cval", "prefilter"]

    def __init__(
        self,
        input,
        grids,
        order=1,
        mode="constant",
        cval=0.0,
        target="cpu",
    ):

        self.input = jnp.asarray(input)
        self.grids = jnp.asarray(grids)
        self.order = order
        self.mode = mode
        self.cval = cval
        self.target = target

        self.ndim = input.ndim  # should match number of grids
        self.shape = input.shape  # should match points in each grid

        assert self.ndim == self.grids.shape[0], DIM_MESSAGE

    def __call__(self, *args):

        # last index is the element

        args = jnp.asarray(args)
        ndim = args.shape[0]

        assert ndim == self.ndim, DIM_MESSAGE

        coordinates = jnp.empty_like(args)

        output = jax_interp(self.input, self.grids, args, coordinates)

        return output


@jax_jit
def jax_interp(input, grids, args, coordinates):

    ndim = args.shape[0]

    for dim in range(ndim):
        arg_grid = grids[dim]
        new_args = args[dim]
        coordinates = coordinates.at[dim].set(
            jnp.interp(new_args, arg_grid, jnp.arange(arg_grid.size))
        )

    output = jax_map_coordinates(input, coordinates.reshape(ndim, -1), order=1).reshape(
        args[0].shape
    )

    return output
