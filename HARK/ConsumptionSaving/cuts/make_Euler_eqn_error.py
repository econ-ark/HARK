 def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):
      """
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncShkDstn
        or to use a (temporary) very dense approximation.

        Only works on (one period) infinite horizon models at this time, will
        be generalized later.

        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncShkDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncShkDstn; when False, makes and uses a very dense approximation.

        Returns
        -------
        None

        Notes
        -----
        This method is not used by any other code in the library. Rather, it is here
        for expository and benchmarking purposes.
        """
       # Get the income distribution (or make a very dense one)
       if approx_inc_dstn:
            IncShkDstn = self.IncShkDstn[0]
        else:
            tranShkDstn = MeanOneLogNormal(sigma=self.tranShkStd[0]).approx(
                N=200, tail_N=50, tail_order=1.3, tail_bound=[0.05, 0.95]
            )
            tranShkDstn = add_discrete_outcome_constant_mean(
                tranShkDstn, self.UnempPrb, self.IncUnemp
            )
            permShkDstn = MeanOneLogNormal(sigma=self.permShkStd[0]).approx(
                N=200, tail_N=50, tail_order=1.3, tail_bound=[0.05, 0.95]
            )
            IncShkDstn = combine_indep_dstns(permShkDstn, tranShkDstn)

        # Make a grid of market resources
        mMin = self.solution[0].mNrmMin + 10 ** (
            -15
        )  # add tiny bit to get around 0/0 problem
        mMax = mMax
        mGrid = np.linspace(mMin, mMax, 1000)

        # Get the consumption function this period and the marginal value function
        # for next period.  Note that this part assumes a one period cycle.
        cFunc = self.solution[0].cFunc
        vPfuncNext = self.solution[0].vPfunc

        # Calculate consumption this period at each gridpoint (and assets)
        cGrid = cFunc(mGrid)
        aGrid = mGrid - cGrid

        # Tile the grids for fast computation
        ShkCount = IncShkDstn[0].size
        aCount = aGrid.size
        aGrid_tiled = np.tile(aGrid, (ShkCount, 1))
        permShkValsNxt_tiled = (np.tile(IncShkDstn[1], (aCount, 1))).transpose()
        tranShkVals_tiled = (np.tile(IncShkDstn[2], (aCount, 1))).transpose()
        ShkPrbs_tiled = (np.tile(IncShkDstn[0], (aCount, 1))).transpose()

        # Calculate marginal value next period for each gridpoint and each shock
        mNextArray = (
            self.Rfree / (self.permGroFac[0] * permShkValsNxt_tiled) * aGrid_tiled
            + tranShkVals_tiled
        )
        vPnextArray = vPfuncNext(mNextArray)

        # Calculate expected marginal value and implied optimal consumption
        ExvPnextGrid = (
            self.DiscFac
            * self.Rfree
            * self.LivPrb[0]
            * self.permGroFac[0] ** (-self.CRRA)
            * np.sum(
                permShkValsNxt_tiled ** (-self.CRRA) * vPnextArray * ShkPrbs_tiled, axis=0
            )
        )
        cOptGrid = ExvPnextGrid ** (
            -1.0 / self.CRRA
        )  # This is the 'Endogenous Gridpoints' step

        # Calculate Euler error and store an interpolated function
        EulerErrorNrmGrid = (cGrid - cOptGrid) / cOptGrid
        eulerErrorFunc = LinearInterp(mGrid, EulerErrorNrmGrid)
        self.eulerErrorFunc = eulerErrorFunc
