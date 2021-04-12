#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 11:27:31 2021

@author: ccarroll
"""


def add_stable_points(self, soln_stge):
    """
        TODO:
        Placeholder method for a possible future implementation of stable
        points in the kinked R model. For now it simply serves to override
        ConsIndShock's method, which does not apply here given the multiple
        interest rates.

        Discusson:
        - The target and steady state should exist under the same conditions
          as in ConsIndShock.
        - The ConsIndShock code as it stands can not be directly applied
          because it assumes that R is a constant, and in this model R depends
          on the level of wealth.
        - After allowing for wealth-depending interest rates, the existing
         code might work without modification to add the stable points. If not,
         it should be possible to find these values by checking within three
         distinct intervals:
             - From h_min to the lower kink.
             - From the lower kink to the upper kink
             - From the upper kink to infinity.
        the stable points must be in one of these regions.

        """
    return soln_stge

    def add_mNrmStE():
        """
        Finds value of (normalized) market resources m at which individual consumer
        expects m not to change.

        This will exist if the GICNrm holds.

        https://econ-ark.github.io/BufferStockTheory#UniqueStablePoints

        Parameters
        ----------
        self : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        self : ConsumerSolution
            Same solution that was passed, but now with the attribute mNrmStE.
        """

        # Minimum market resources plus next income is okay starting guess
        m_init_guess = self.soln_stge.mNrmMin + self.soln_stge.Ex_IncNextNrm
        try:
            m_t = newton(
                Ex_m_tp1_minus_m_t,
                m_init_guess)
        except:
            m_t = None

        # Add mNrmTrg to the solution and return it
        mNrmTrg = m_t
