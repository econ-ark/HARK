#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 11:03:34 2021

@author: ccarroll
"""

    def add_stable_points(self, soln_stge):
        """
        Checks necessary conditions for the existence of the individual pseudo
        steady state StE and target Trg levels of market resources.
        If the conditions are satisfied, computes and adds the stable points
        to the soln_stge.

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was provided, augmented with attributes mNrmStE and
            mNrmTrg, if they exist.

        """
        # This is the version for perfect foresight model; models that
        # inherit from the PF model will replace it with suitable alternatives
        # For the PF model:
        # 0. There is no non-degenerate steady state without constraints
        # 1. There is a non-degenerate SS for constrained PF model if GICRaw holds.
        # Therefore
        # Check if  (GICRaw and BoroCnstArt) and if so compute them both
        APF = (self.Rfree*self.DiscFacEff)**(1/self.CRRA)
        GICRaw = 1 > APF/self.PermGroFac_0_
        if self.BoroCnstArt is not None and GICRaw:
            # Result of borrowing max allowed
            bNrmNxt = -self.BoroCnstArt*self.Rfree/self.PermGroFac_0_
            soln_stge.mNrmStE = self.Ex_IncNextNrm-bNrmNxt
            soln_stge.mNrmTrg = self.Ex_IncNextNrm-bNrmNxt
        else:
            _log.warning("The unconstrained PF model solution is degenerate")
            if GICRaw:
                if self.Rfree > self.PermGroFac_0_:  # impatience drives wealth to minimum
                    soln_stge.mNrmStE = -(1/(1-self.PermGroFac_0_/self.Rfree))
                    soln_stge.mNrmTrg = -(1/(1-self.PermGroFac_0_/self.Rfree))
                else:  # patience drives wealth to infinity
                    _log.warning(
                        "Pathological patience plus infinite human wealth: solution undefined")
                    soln_stge.mNrmStE = float('NaN')
                    soln_stge.mNrmTrg = float('NaN')
            else:
                soln_stge.mNrmStE = float('inf')
                soln_stge.mNrmTrg = float('inf')
        return soln_stge

 