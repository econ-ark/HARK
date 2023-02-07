# this file assumes that jax is installed; do not import this file if jax is not installed

import jax.numpy as jnp
from HARK.core import MetricObject
from jax import jit as jax_jit
from jax.scipy.ndimage import map_coordinates as jax_map_coordinates

DIM_MESSAGE = "Dimension mismatch."


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
