def calc_bounding_values(self):
    """
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality (because your income matters to you only if you are still alive).
        The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
    # Unpack the income distribution and get average and worst outcomes
    permShkValsNxt = self.IncShkDstn[0][1]
    tranShkValsNxt = self.IncShkDstn[0][2]
    ShkPrbs = self.IncShkDstn[0][0]
    Ex_IncNrmNxt = 𝔼_dot(ShkPrbs, permShkValsNxt * tranShkValsNxt)
    permShkMinNext = np.min(permShkValsNxt)
    tranShkMinNext = np.min(tranShkValsNxt)
    WorstIncNext = permShkMinNext * tranShkMinNext
    WorstIncPrb = np.sum(
        ShkPrbs[(permShkValsNxt * tranShkValsNxt) == WorstIncNext]
    )
    permGro = self.permGroFac[0]  # AgentType gets list of growth rates
    LivNxt = self.LivPrb[0]  # and survival rates

    # Calculate human wealth and the infinite horizon natural borrowing constraint
    hNrm = (Ex_IncNrmNxt * permGro / self.Rfree) / (
        1.0 - permGro / self.Rfree
        )
    temp = permGro * permShkMinNext / self.Rfree
     BoroCnstNat = -tranShkMinNext * temp / (1.0 - temp)

      RPF = (self.DiscFac * LivNxt * self.Rfree) ** (
           1.0 / self.CRRA
           ) / self.Rfree
       if BoroCnstNat < self.BoroCnstArt:
            MPCmax = 1.0  # if natural borrowing constraint is overridden by artificial one, MPCmax is 1
        else:
            MPCmax = 1.0 - WorstIncPrb ** (1.0 / self.CRRA) * RPF
            MPCmin = 1.0 - RPF

        # Store the results as attributes of self
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
