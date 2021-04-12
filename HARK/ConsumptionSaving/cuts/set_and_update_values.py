#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 11:04:50 2021

@author: ccarroll
"""

    def set_and_update_values(self, solution_next, IncShkDstn_0_, LivPrb_0_, DiscFac_0_):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, next period's marginal value function
        (etc), the probability of getting the worst income shock next period,
        the patience factor, human wealth, and the bounding MPCs.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncShkDstn : distribution.DiscreteApproximationToContinuousDistribution
            A DiscreteApproximationToContinuousDistribution with a pmf
            and two point value arrays in X, order:
            permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac_0_ : float
            Intertemporal discount factor for future utility.

        Returns
        -------
        None
        """
        self.soln_stge.DiscFac_0_Eff = DiscFac_0_ * LivPrb_0_  # "effective" discount factor
        self.soln_stge.IncShkDstn = IncShkDstn_0_
        self.soln_stge.ShkPrbsNext = IncShkDstn_0_.pmf
        self.PermShkValsNext = IncShkDstn_0_.X[0]
        self.TranShkValsNext = IncShkDstn_0_.X[1]
        self.PermShkMinNext = np.min(self.PermShkValsNext)
        self.TranShkMinNext = np.min(self.TranShkValsNext)
        self.vPfuncNext = solution_next.vPfunc
        self.WorstIncPrb = np.sum(
            self.ShkPrbsNext[
                (self.PermShkValsNext * self.TranShkValsNext)
                == (self.PermShkMinNext * self.TranShkMinNext)
            ]
        )

        if self.CubicBool:
            self.vPPfuncNext = solution_next.vPPfunc

        if self.vFuncBool:
            self.vFuncNext = solution_next.vFunc

        # Update the bounding MPCs and PDV of human wealth:
        self.RPF = ((self.Rfree * self.DiscFac_0_Eff) ** (1.0 / self.CRRA)) / self.Rfree
        self.MPCminNow = 1.0 / (1.0 + self.RPF / solution_next.MPCminNow)
        self.Ex_IncNext = np.dot(
            self.ShkPrbsNext, self.TranShkValsNext * self.PermShkValsNext
        )
        self.hNrmNow = (
            self.PermGroFac_0_ / self.Rfree * (self.Ex_IncNext + solution_next.hNrmNow)
        )
        self.MPCmaxNow = 1.0 / (
            1.0
            + (self.WorstIncPrb ** (1.0 / self.CRRA))
            * self.RPF
            / solution_next.MPCmaxNow
        )

        self.cFuncLimitIntercept = self.MPCminNow * self.hNrmNow
        self.cFuncLimitSlope = self.MPCminNow
