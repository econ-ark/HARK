class UnstructuredInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(
        self,
        values,
        grids,
        method="linear",
        rescale=False,
        fill_value=np.nan,
        # CloughTocher2DInterpolator options
        tol=1e-6,
        maxiter=400,
        # NearestNDInterpolator options
        tree_options=None,
    ):

        values = np.asarray(values)
        grids = np.asarray(grids)

        # remove non finite values that might result from
        # sequential endogenous grid method
        condition = np.logical_and.reduce([np.isfinite(grid) for grid in grids])
        condition = np.logical_and(condition, np.isfinite(values))

        self.values = values[condition]
        self.grids = grids[:, condition].T

        self.method = method
        self.rescale = rescale
        self.fill_value = fill_value
        self.tol = tol
        self.maxiter = maxiter
        self.tree_options = tree_options

        self.ndim = self.grids.shape[-1]

        assert self.ndim == values.ndim, "Dimension mismatch."

        if method == "nearest":
            interpolator = NearestNDInterpolator(
                self.grids, self.values, rescale=rescale, tree_options=tree_options
            )
        elif method == "linear":
            interpolator = LinearNDInterpolator(
                self.grids, self.values, fill_value=fill_value, rescale=rescale
            )
        elif method == "cubic" and self.ndim == 2:
            interpolator = CloughTocher2DInterpolator(
                self.grids,
                self.values,
                fill_value=fill_value,
                tol=tol,
                maxiter=maxiter,
                rescale=rescale,
            )
        else:
            raise ValueError(
                "Unknown interpolation method %r for "
                "%d dimensional data" % (method, self.ndim)
            )

        self.interpolator = interpolator

    def __call__(self, *args):

        return self.interpolator(*args)

