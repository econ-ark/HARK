#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:52:47 2021

@author: ccarroll
"""

    def soln_stge_add_further_info(self, soln_stge):
        # self here is the solver, which knows info about the problem from the agent
        """
        For versions with uncertainty in transitory and/or permanent shocks,
        adds to the solution a set of results useful for calculating
        and various diagnostic conditions about the problem, and stable
        points (if they exist).

        Parameters
        ----------
        solution: ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.

        Returns
        -------
        solution : ConsumerSolution
            Same solution that was provided, augmented with the factors

        """
#        if hasattr(self, 'soln_stge_add_further_info_ConsPerfForesightSolver'):
#            self.soln_stge_add_further_info_ConsPerfForesightSolver(soln_stge)
        # The Xref versions are full rank, for direct multiplication by probs
        soln_stge.PermShkValsXref = PermShkValsXref = self.soln_stge.IncShkDstn.X[0]
        soln_stge.TranShkValsXref = TranShkValsXref = self.soln_stge.IncShkDstn.X[1]
        soln_stge.ShkPrbsNext = ShkPrbsNext = self.soln_stge.IncShkDstn.pmf

        soln_stge.PermShkVals = PermShkVals = self.soln_stge.PermShkDstn.X
        soln_stge.TranShkVals = TranShkVals = self.soln_stge.TranShkDstn.X
        PermShkMin = np.min(PermShkVals)
        TranShkMin = np.min(TranShkVals)

        self.soln_stge.WorstIncPrb = np.sum(
            ShkPrbsNext[
                (PermShkValsXref * TranShkValsXref)
                == (PermShkMin * TranShkMin)
            ]
        )

        self.soln_stge_add_further_info_ConsPerfForesightSolver_20210410(soln_stge)
#        soln_stge.parameters_model = self.parameters_model

        if not hasattr(soln_stge, 'facts'):
            soln_stge.facts = {}

        soln_stge.Rfree = Rfree = self.soln_stge.Rfree
        soln_stge.DiscFac_0_ = DiscFac_0_ = self.soln_stge.DiscFac
        soln_stge.PermGroFac_0_ = PermGroFac_0_ = self.soln_stge.PermGroFac
        soln_stge.LivPrb_0_ = LivPrb_0_ = self.soln_stge.LivPrb
        soln_stge.DiscFac_0_Eff = DiscFac_0_Eff = DiscFac_0_ * LivPrb_0_
        soln_stge.CRRA = CRRA = self.soln_stge.CRRA
        soln_stge.UnempPrb = UnempPrb = self.soln_stge.IncShkDstn.parameters['UnempPrb']
        soln_stge.UnempPrbRet = UnempPrbRet = self.soln_stge.IncShkDstn.parameters[
            'UnempPrbRet']

        # Many other facts will have been inherited from the perfect foresight
        # model of which this model, as a descendant, has already inherited
        # Here we need compute only those objects whose value changes when
        # the shock distributions are nondegenerate.
        IncShkDstn_facts = {
            'about': 'Income Shock Distribution: .X[0] and .X[1] retrieve shocks, .pmf retrieves probabilities'}
#       IncShkDstn_facts.update({'latexexpr': r'\IncShkDstn'})
        IncShkDstn_facts.update(
            {'py___code': r'construct_lognormal_income_process_unemployment'})
        soln_stge.facts.update({'IncShkDstn': IncShkDstn_facts})
        soln_stge.IncShkDstn_facts = IncShkDstn_facts

        Ex_IncNextNrm_facts = {
            'about': 'Expected income next period'}
        soln_stge.Ex_IncNextNrm = Ex_IncNextNrm = np.dot(
            ShkPrbsNext, TranShkValsXref * PermShkValsXref).item()
        Ex_IncNextNrm_facts.update({'latexexpr': r'\Ex_IncNextNrm'})
        Ex_IncNextNrm_facts.update({'_unicode_': r'R/Γ'})
        Ex_IncNextNrm_facts.update({'urlhandle': self.soln_stge.urlroot+'ExIncNextNrm'})
        Ex_IncNextNrm_facts.update(
            {'py___code': r'np.dot(ShkPrbsNext,TranShkValsXref*PermShkValsXref)'})
        Ex_IncNextNrm_facts.update({'value_now': Ex_IncNextNrm})
        soln_stge.facts.update({'Ex_IncNextNrm': Ex_IncNextNrm_facts})
        soln_stge.Ex_IncNextNrm_facts = Ex_IncNextNrm_facts

#        Ex_Inv_PermShk = calc_expectation(            PermShkDstn_0_[0], lambda x: 1 / x        )
        soln_stge.Ex_Inv_PermShk = self.soln_stge.Ex_Inv_PermShk  # Precalc
        Ex_Inv_PermShk_facts = {
            'about': 'Expectation of Inverse of Permanent Shock'}
        Ex_Inv_PermShk_facts.update({'latexexpr': r'\Ex_Inv_PermShk'})
#        Ex_Inv_PermShk_facts.update({'_unicode_': r'R/Γ'})
        Ex_Inv_PermShk_facts.update({'urlhandle': self.soln_stge.urlroot+'ExInvPermShk'})
        Ex_Inv_PermShk_facts.update({'py___code': r'Rfree/PermGroFacAdj'})
        Ex_Inv_PermShk_facts.update({'value_now': Ex_Inv_PermShk})
        soln_stge.facts.update({'Ex_Inv_PermShk': Ex_Inv_PermShk_facts})
        soln_stge.Ex_Inv_PermShk_facts = Ex_Inv_PermShk_facts
        # soln_stge.Ex_Inv_PermShk = Ex_Inv_PermShk

        Inv_Ex_Inv_PermShk_facts = {
            'about': 'Inverse of Expectation of Inverse of Permanent Shock'}
        soln_stge.Inv_Ex_Inv_PermShk = Inv_Ex_Inv_PermShk = 1/Ex_Inv_PermShk
        Inv_Ex_Inv_PermShk_facts.update(
            {'latexexpr': '\InvExInvPermShk=\left(\Ex[\PermShk^{-1}]\right)^{-1}'})
#        Inv_Ex_Inv_PermShk_facts.update({'_unicode_': r'R/Γ'})
        Inv_Ex_Inv_PermShk_facts.update({'urlhandle': self.soln_stge.urlroot+'InvExInvPermShk'})
        Inv_Ex_Inv_PermShk_facts.update({'py___code': r'1/Ex_Inv_PermShk'})
        Inv_Ex_Inv_PermShk_facts.update({'value_now': Inv_Ex_Inv_PermShk})
        soln_stge.facts.update({'Inv_Ex_Inv_PermShk': Inv_Ex_Inv_PermShk_facts})
        soln_stge.Inv_Ex_Inv_PermShk_facts = Inv_Ex_Inv_PermShk_facts
        # soln_stge.Inv_Ex_Inv_PermShk = Inv_Ex_Inv_PermShk

        Ex_RNrm_facts = {
            'about': 'Expectation of Stochastic-Growth-Normalized Return'}
        Ex_RNrm = PF_RNrm * Ex_Inv_PermShk
        Ex_RNrm_facts.update({'latexexpr': r'\Ex_RNrm'})
#        Ex_RNrm_facts.update({'_unicode_': r'R/Γ'})
        Ex_RNrm_facts.update({'urlhandle': self.soln_stge.urlroot+'ExRNrm'})
        Ex_RNrm_facts.update({'py___code': r'Rfree/PermGroFacAdj'})
        Ex_RNrm_facts.update({'value_now': Ex_RNrm})
        soln_stge.facts.update({'Ex_RNrm': Ex_RNrm_facts})
        soln_stge.Ex_RNrm_facts = Ex_RNrm_facts
        soln_stge.Ex_RNrm = Ex_RNrm

        Inv_Ex_RNrm_facts = {
            'about': 'Inverse of Expectation of Stochastic-Growth-Normalized Return'}
        Inv_Ex_RNrm = 1/Ex_RNrm
        Inv_Ex_RNrm_facts.update(
            {'latexexpr': '\InvExInvPermShk=\left(\Ex[\PermShk^{-1}]\right)^{-1}'})
#        Inv_Ex_RNrm_facts.update({'_unicode_': r'R/Γ'})
        Inv_Ex_RNrm_facts.update({'urlhandle': self.soln_stge.urlroot+'InvExRNrm'})
        Inv_Ex_RNrm_facts.update({'py___code': r'1/Ex_RNrm'})
        Inv_Ex_RNrm_facts.update({'value_now': Inv_Ex_RNrm})
        soln_stge.facts.update({'Inv_Ex_RNrm': Inv_Ex_RNrm_facts})
        soln_stge.Inv_Ex_RNrm_facts = Inv_Ex_RNrm_facts
        soln_stge.Inv_Ex_RNrm = Inv_Ex_RNrm

        Ex_uInv_PermShk_facts = {
            'about': 'Expected Utility for Consuming Permanent Shock'}

        Ex_uInv_PermShk_facts.update({'latexexpr': r'\Ex_uInv_PermShk'})
        Ex_uInv_PermShk_facts.update({'urlhandle': r'ExuInvPermShk'})
        Ex_uInv_PermShk_facts.update(
            {'py___code': r'np.dot(PermShkValsXref**(1-CRRA),ShkPrbsNext)'})
        Ex_uInv_PermShk_facts.update({'value_now': Ex_uInv_PermShk})
        soln_stge.facts.update({'Ex_uInv_PermShk': Ex_uInv_PermShk_facts})
        soln_stge.Ex_uInv_PermShk_facts = Ex_uInv_PermShk_facts
        soln_stge.Ex_uInv_PermShk = Ex_uInv_PermShk = self.soln_stge.Ex_uInv_PermShk

        uInv_Ex_uInv_PermShk = Ex_uInv_PermShk ** (1 / (1 - CRRA))
        uInv_Ex_uInv_PermShk_facts = {
            'about': 'Inverted Expected Utility for Consuming Permanent Shock'}
        uInv_Ex_uInv_PermShk_facts.update({'latexexpr': r'\uInvExuInvPermShk'})
        uInv_Ex_uInv_PermShk_facts.update({'urlhandle': self.soln_stge.urlroot+'uInvExuInvPermShk'})
        uInv_Ex_uInv_PermShk_facts.update({'py___code': r'Ex_uInv_PermShk**(1/(1-CRRA))'})
        uInv_Ex_uInv_PermShk_facts.update({'value_now': uInv_Ex_uInv_PermShk})
        soln_stge.facts.update({'uInv_Ex_uInv_PermShk': uInv_Ex_uInv_PermShk_facts})
        soln_stge.uInv_Ex_uInv_PermShk_facts = uInv_Ex_uInv_PermShk_facts
        self.soln_stge.uInv_Ex_uInv_PermShk = soln_stge.uInv_Ex_uInv_PermShk = uInv_Ex_uInv_PermShk

        PermGroFacAdj_facts = {
            'about': 'Uncertainty-Adjusted Permanent Income Growth Factor'}
        PermGroFacAdj = PermGroFac_0_ * Inv_Ex_Inv_PermShk
        PermGroFacAdj_facts.update({'latexexpr': r'\mathcal{R}\underline{\permShk}'})
        PermGroFacAdj_facts.update({'urlhandle': self.soln_stge.urlroot+'PermGroFacAdj'})
        PermGroFacAdj_facts.update({'value_now': PermGroFacAdj})
        soln_stge.facts.update({'PermGroFacAdj': PermGroFacAdj_facts})
        soln_stge.PermGroFacAdj_facts = PermGroFacAdj_facts
        soln_stge.PermGroFacAdj = PermGroFacAdj

        GPFNrm_facts = {
            'about': 'Normalized Expected Growth Patience Factor'}
        soln_stge.GPFNrm = GPFNrm = soln_stge.GPFRaw * Ex_Inv_PermShk
        GPFNrm_facts.update({'latexexpr': r'\GPFNrm'})
        GPFNrm_facts.update({'_unicode_': r'Þ_Γ'})
        GPFNrm_facts.update({'urlhandle': self.soln_stge.urlroot+'GPFNrm'})
        GPFNrm_facts.update({'py___code': 'test: GPFNrm < 1'})
        soln_stge.facts.update({'GPFNrm': GPFNrm_facts})
        soln_stge.GPFNrm_facts = GPFNrm_facts

        GICNrm_facts = {'about': 'Growth Impatience Condition'}
        GICNrm_facts.update({'latexexpr': r'\GICNrm'})
        GICNrm_facts.update({'urlhandle': self.soln_stge.urlroot+'GICNrm'})
        GICNrm_facts.update({'py___code': 'test: agent.GPFNrm < 1'})
        soln_stge.facts.update({'GICNrm': GICNrm_facts})
        soln_stge.GICNrm_facts = GICNrm_facts

        FVAC_facts = {'about': 'Finite Value of Autarky Condition'}
        FVAC_facts.update({'latexexpr': r'\FVAC'})
        FVAC_facts.update({'urlhandle': self.soln_stge.urlroot+'FVAC'})
        FVAC_facts.update({'py___code': 'test: FVAF < 1'})
        soln_stge.facts.update({'FVAC': FVAC_facts})
        soln_stge.FVAC_facts = FVAC_facts

        DiscGPFNrmCusp_facts = {'about':
                                'DiscFac s.t. GPFNrm = 1'}
        soln_stge.DiscGPFNrmCusp = DiscGPFNrmCusp = (
            (PermGroFac_0_*Inv_Ex_Inv_PermShk)**(CRRA))/Rfree
        DiscGPFNrmCusp_facts.update({'latexexpr': ''})
        DiscGPFNrmCusp_facts.update({'value_now': DiscGPFNrmCusp})
        DiscGPFNrmCusp_facts.update({
            'py___code': '((PermGro * Inv_Ex_Inv_PermShk) ** CRRA)/(Rfree)'})
        soln_stge.facts.update({'DiscGPFNrmCusp': DiscGPFNrmCusp_facts})
        soln_stge.DiscGPFNrmCusp_facts = DiscGPFNrmCusp_facts

        # # Merge all the parameters
        # # In python 3.9, the syntax is new_dict = dict_a | dict_b
        # soln_stge.params_all = {**self.params_cons_ind_shock_setup_init,
        #                    **params_cons_ind_shock_setup_set_and_update_values}

        # Now that the calculations are done, store results in self.
        # self, here, is the solver
        # goal: agent,  solver, and solution should be standalone
        # this requires the solution to get some info from the solver

        if soln_stge.Inv_PF_RNrm < 1:        # Finite if Rfree > PermGroFac_0_
            soln_stge.hNrmNowInf = 1/(1-soln_stge.Inv_PF_RNrm)

        # Given m, value of c where E[m_{t+1}]=m_{t}
        # url/#
        soln_stge.c_where_Ex_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - soln_stge.Inv_Ex_RNrm) + (soln_stge.Inv_Ex_RNrm)
        )

        # Given m, value of c where E[mLev_{t+1}/mLev_{t}]=PermGroFac_0_
        # Solves for c in equation at url/#balgrostable

        soln_stge.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - soln_stge.Inv_PF_RNrm) + soln_stge.Inv_PF_RNrm
        )

        # E[c_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        soln_stge.Ex_cLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
            np.dot(soln_stge.PermGroFac_0_ *
                   soln_stge.PermShkValsXref *
                   soln_stge.cFunc(
                       (soln_stge.PF_RNrm/soln_stge.PermShkValsXref) * a_t
                       + soln_stge.TranShkValsXref
                   ),
                   soln_stge.ShkPrbsNext)
        )

        soln_stge.c_where_Ex_mtp1_minus_mt_eq_0 = c_where_Ex_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - 1/soln_stge.Ex_RNrm) + (1/soln_stge.Ex_RNrm)
        )

        # Solve the equation at url/#balgrostable
        soln_stge.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = \
            c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
                lambda m_t:
                (m_t * (1 - 1/soln_stge.PF_RNrm)) + (1/soln_stge.PF_RNrm)
            )

        # mNrmTrg solves Ex_RNrm*(m - c(m)) + E[inc_next] - m = 0
        Ex_m_tp1_minus_m_t = (
            lambda m_t:
            soln_stge.Ex_RNrm * (m_t - soln_stge.cFunc(m_t)) +
            soln_stge.Ex_IncNextNrm - m_t
        )
        soln_stge.Ex_m_tp1_minus_m_t = Ex_m_tp1_minus_m_t

        soln_stge.Ex_cLev_tp1_Over_pLev_t_from_at = Ex_cLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
            np.dot(
                soln_stge.PermShkValsXref * soln_stge.PermGroFac_0_ * soln_stge.cFunc(
                    (soln_stge.PF_RNrm/soln_stge.PermShkValsXref) *
                    a_t + soln_stge.TranShkValsXref
                ),
                soln_stge.ShkPrbsNext)
        )

        soln_stge.Ex_PermShk_tp1_times_m_tp1_minus_m_t = \
            Ex_PermShk_tp1_times_m_tp1_minus_m_t = (
                lambda m_t: self.soln_stge.PF_RNrm *
                (m_t - soln_stge.cFunc(m_t)) + 1.0 - m_t
            )

        return soln_stge
