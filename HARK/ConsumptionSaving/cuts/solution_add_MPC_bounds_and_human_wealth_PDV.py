#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:54:02 2021

@author: ccarroll
"""


    def solution_add_MPC_bounds_and_human_wealth_PDV(self, soln_stge):
        """
        Take a solution and add human wealth and the bounding MPCs to it.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this period's consumption-saving problem.

        Returns:
        ----------
        solution : ConsumerSolution
            The solution to this period's consumption-saving problem, but now
            with human wealth and the bounding MPCs.
        """
        soln_stge.hNrmNow = self.hNrmNow
        soln_stge.MPCminNow = self.MPCminNow
        soln_stge.MPCmaxNow = self.MPCmaxNowEff
#        _log.warning(
#            "add_MPC_bounds_and_human_wealth_PDV is deprecated; its functions have been absorbed by add_results")
        return soln_stge
