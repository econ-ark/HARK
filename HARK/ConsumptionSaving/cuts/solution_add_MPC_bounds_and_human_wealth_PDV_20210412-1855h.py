#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 18:53:44 2021

@author: ccarroll
"""

    def solution_add_MPC_bounds_and_human_wealth_PDV_20210410(self, crnt):
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
        hNrmNow = (
            (crnt.Nxt.PermGro / crnt.Nxt.Rfree) * (1.0 + self.solution_next.hNrmNow)
        )
        hNrmNowfcts = {'about': 'Human Wealth Now'}
        hNrmNowfcts.update({'latexexpr': r'\hNrmNow'})
        hNrmNowfcts.update({'_unicode_': r'R/Γ'})
        hNrmNowfcts.update({'urlhandle': urlroot+'hNrmNow'})
        hNrmNowfcts.update({'py___code': r'crnt.Nxt.PermGroInf/Rfree'})
        hNrmNowfcts.update({'value_now': hNrmNow})
        crnt.hNrmNowfcts = hNrmNowfcts
        crnt.fcts.update({'hNrmNow': hNrmNowfcts})
        self.hNrmNow = crnt.hNrmNow = hNrmNow

        # Calculate the minimum allowable value of money resources in this period

        crnt.BoroCnstNat = (
            (self.solution_next.mNrmMin - min(self.solution_next.TranShkValsNxt))
            * (self.crnt.Nxt.PermGro * min(self.solution_next.PermShkValsNxt))
            / self.solution_next.Rfree
        )

        if crnt.BoroCnstArt is None:
            crnt.mNrmMin = crnt.BoroCnstNat
        else:  # Artificial is only relevant if tighter than natural
            crnt.mNrmMin = np.max([crnt.BoroCnstNat, crnt.BoroCnstArt])
            # Liquidity constrained consumption function: c(mMin+x) = x
            crnt.cFuncNowCnst = LinearInterp(
                np.array([crnt.mNrmMin, crnt.mNrmMin + 1]
                         ), np.array([0.0, 1.0])
            )

        mNrmMin = crnt.mNrmMin
        mNrmMinfcts = {'about': 'Minimum mNrm'}
        mNrmMinfcts.update({'latexexpr': r'\mNrmMin'})
        crnt.fcts.update({'mNrmMin': mNrmMinfcts})
        crnt.mNrmMinfcts = mNrmMinfcts
        crnt.mNrmMin = mNrmMin

        MPCminNow = 1.0 / (1.0 + crnt.RPF / self.solution_next.MPCminNow)
        MPCminNowfcts = {
            'about': 'Minimal MPC as m -> infty'}
        MPCminNowfcts.update({'latexexpr': r''})
        MPCminNowfcts.update({'urlhandle': urlroot+'MPCminNow'})
        MPCminNowfcts.update({'value_now': MPCminNow})
        crnt.fcts.update({'MPCminNow': MPCminNowfcts})
        crnt.MPCminNowfcts = MPCminNowfcts
        crnt.MPCminNow = crnt.MPCminNow = MPCminNow

        MPCmaxNow = 1.0 / \
            (1.0 + (self.solution_next.WorstIncPrbNxt ** (1.0 / crnt.CRRA))
             * self.solution_next.RPF
             / self.solution_next.MPCmaxNow)
        MPCmaxNowfcts = {
            'about': 'Maximal MPC in current period as m -> minimum'}
        MPCmaxNowfcts.update({'latexexpr': r''})
        MPCmaxNowfcts.update({'urlhandle': urlroot+'MPCmaxNow'})
        MPCmaxNowfcts.update({'value_now': MPCmaxNow})
        crnt.fcts.update({'MPCmaxNow': MPCmaxNowfcts})
        crnt.MPCmaxNowfcts = MPCmaxNowfcts
        crnt.MPCmaxNow = MPCmaxNow

        # Lower bound of aggregate wealth growth if all inheritances squandered
        cFuncLimitIntercept = MPCminNow * crnt.hNrmNow
        cFuncLimitInterceptfcts = {
            'about': 'Vertical intercept of perfect foresight consumption function'}
        cFuncLimitInterceptfcts.update({'latexexpr': '\MPC '})
        cFuncLimitInterceptfcts.update({'urlhandle': ''})
        cFuncLimitInterceptfcts.update({'value_now': cFuncLimitIntercept})
        cFuncLimitInterceptfcts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        crnt.fcts.update({'cFuncLimitIntercept': cFuncLimitInterceptfcts})
        crnt.cFuncLimitInterceptfcts = cFuncLimitInterceptfcts
        crnt.cFuncLimitIntercept = cFuncLimitIntercept

        cFuncLimitSlope = MPCminNow
        cFuncLimitSlopefcts = {
            'about': 'Slope of limiting consumption function'}
        cFuncLimitSlopefcts = dict({'latexexpr': '\MPC \hNrmNow'})
        cFuncLimitSlopefcts.update({'urlhandle': ''})
        cFuncLimitSlopefcts.update({'value_now': cFuncLimitSlope})
        cFuncLimitSlopefcts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        crnt.fcts.update({'cFuncLimitSlope': cFuncLimitSlopefcts})
        crnt.cFuncLimitSlopefcts = cFuncLimitSlopefcts
        crnt.cFuncLimitSlope = cFuncLimitSlope

        # Fcts that apply in the perfect foresight case should already have been added

        if crnt.Inv_PF_RNrm < 1:        # Finite if Rfree > crnt.Nxt.PermGro
            crnt.hNrmNowInf = 1/(1-crnt.Inv_PF_RNrm)

        # Given m, value of c where E[mLev_{t+1}/mLev_{t}]=crnt.Nxt.PermGro
        # Solves for c in equation at url/#balgrostable

        crnt.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - crnt.Inv_PF_RNrm) + crnt.Inv_PF_RNrm
        )

        crnt.Ex_cLev_tp1_Over_cLev_t_from_mt = (
            lambda m_t:
            crnt.Ex_cLev_tp1_Over_pLev_t_from_mt(crnt,
                                                 m_t - crnt.cFunc(m_t))
            / crnt.cFunc(m_t)
        )

#        # E[m_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        crnt.Ex_mLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
                crnt.Nxt.PermGro *
            (crnt.PF_RNrm * a_t + crnt.Ex_IncNextNrm)
        )

        # E[m_{t+1} pLev_{t+1}/(m_{t}pLev_{t})] as a fn of m_{t}
        crnt.Ex_mLev_tp1_Over_mLev_t_from_at = (
            lambda m_t:
                crnt.Ex_mLev_tp1_Over_pLev_t_from_at(crnt,
                                                     m_t-crnt.cFunc(m_t)
                                                     )/m_t
        )

        return crnt

        RfreeNow=self.Rfree * np.ones(self.mcrlo_AgentCount)
