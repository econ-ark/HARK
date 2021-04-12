
def add_facts_to_PerfForesightConsumerType_solution(self, soln_stge):
    # self here is the agent, whose self must have attached to it
    # solution_now and solution_next objects.
    # solution_now will be updated.
    """
        Adds to the solution a set of results useful for calculating
        and various diagnostic conditions about the problem, and stable
        points (if they exist).

        Parameters
        ----------
        solution: ConsumerSolution
            Solution a consumer's problem, which must have attribute cFunc.

        Returns
        -------
        solution : ConsumerSolution
            Same solution that was provided, augmented with the factors

        """
    self.soln_stge.urlroot = self.url_ref+'/#'
    soln_stge.parameters_model = self.parameters

    BoroCnstArt = soln_stge.BoroCnstArt = self.BoroCnstArt
    CRRA = soln_stge.CRRA = self.CRRA
    DiscFac = soln_stge.DiscFac = self.DiscFac
    LivPrb_0_ = soln_stge.LivPrb_0_ = self.LivPrb[0]
    PermGroFac_0_ = soln_stge.PermGroFac_0_ = self.PermGroFac[0]
    Rfree = soln_stge.Rfree = self.Rfree
    DiscFacEff = soln_stge.DiscFacEff = DiscFac * LivPrb_0_
    WorstIncPrbNxt = self.soln_stge.WorstIncPrbNxt

    soln_stge.facts = {}
    # First calculate a bunch of things that do not required
    # info about the income shocks

    uInv_Ex_uInv_PermShk = 1.0
    soln_stge.conditions = {}

    APF_facts = {
        'about': 'Absolute Patience Factor'}
    soln_stge.APF = APF = \
        ((Rfree * DiscFacEff) ** (1.0 / CRRA))
    APF_facts.update({'latexexpr': r'\APF'})
    APF_facts.update({'_unicode_': r'Þ'})
    APF_facts.update({'urlhandle': self.soln_stge.urlroot+'APF'})
    APF_facts.update({'py___code': '(Rfree*DiscFacEff)**(1/CRRA)'})
    APF_facts.update({'value_now': APF})
    soln_stge.facts.update({'APF': APF_facts})
    soln_stge.APF_facts = APF_facts

    AIC_facts = {'about': 'Absolute Impatience Condition'}
    AIC_facts.update({'latexexpr': r'\AIC'})
    AIC_facts.update({'urlhandle': self.soln_stge.urlroot+'AIC'})
    AIC_facts.update({'py___code': 'test: APF < 1'})
    soln_stge.facts.update({'AIC': AIC_facts})
    soln_stge.AIC_facts = AIC_facts

    RPF_facts = {
        'about': 'Return Patience Factor'}
    RPF = APF / Rfree
    RPF_facts.update({'latexexpr': r'\RPF'})
    RPF_facts.update({'_unicode_': r'Þ_R'})
    RPF_facts.update({'urlhandle': self.soln_stge.urlroot+'RPF'})
    RPF_facts.update({'py___code': r'APF/Rfree'})
    RPF_facts.update({'value_now': RPF})
    soln_stge.facts.update({'RPF': RPF_facts})
    soln_stge.RPF_facts = RPF_facts
    soln_stge.RPF = RPF

    RIC_facts = {'about': 'Growth Impatience Condition'}
    RIC_facts.update({'latexexpr': r'\RIC'})
    RIC_facts.update({'urlhandle': self.soln_stge.urlroot+'RIC'})
    RIC_facts.update({'py___code': 'test: agent.RPF < 1'})
    soln_stge.facts.update({'RIC': RIC_facts})
    soln_stge.RIC_facts = RIC_facts

    GPFRaw_facts = {
        'about': 'Growth Patience Factor'}
    GPFRaw = APF / PermGroFac_0_
    GPFRaw_facts.update({'latexexpr': '\GPFRaw'})
    GPFRaw_facts.update({'urlhandle': self.soln_stge.urlroot+'GPFRaw'})
    GPFRaw_facts.update({'_unicode_': r'Þ_Γ'})
    GPFRaw_facts.update({'value_now': GPFRaw})
    soln_stge.facts.update({'GPFRaw': GPFRaw_facts})
    soln_stge.GPFRaw_facts = GPFRaw_facts
    soln_stge.GPFRaw = GPFRaw

    GICRaw_facts = {'about': 'Growth Impatience Condition'}
    GICRaw_facts.update({'latexexpr': r'\GICRaw'})
    GICRaw_facts.update({'urlhandle': self.soln_stge.urlroot+'GICRaw'})
    GICRaw_facts.update({'py___code': 'test: agent.GPFRaw < 1'})
    soln_stge.facts.update({'GICRaw': GICRaw_facts})
    soln_stge.GICRaw_facts = GICRaw_facts

    GPFLiv_facts = {
        'about': 'Mortality-Risk-Adjusted Growth Patience Factor'}
    GPFLiv = APF * LivPrb_0_ / PermGroFac_0_
    GPFLiv_facts.update({'latexexpr': '\GPFLiv'})
    GPFLiv_facts.update({'urlhandle': self.soln_stge.urlroot+'GPFLiv'})
    GPFLiv_facts.update({'py___code': 'APF*Liv/PermGroFac_0_'})
    GPFLiv_facts.update({'value_now': GPFLiv})
    soln_stge.facts.update({'GPFLiv': GPFLiv_facts})
    soln_stge.GPFLiv_facts = GPFLiv_facts
    soln_stge.GPFLiv = GPFLiv

    GICLiv_facts = {'about': 'Growth Impatience Condition'}
    GICLiv_facts.update({'latexexpr': r'\GICLiv'})
    GICLiv_facts.update({'urlhandle': self.soln_stge.urlroot+'GICLiv'})
    GICLiv_facts.update({'py___code': 'test: GPFLiv < 1'})
    soln_stge.facts.update({'GICLiv': GICLiv_facts})
    soln_stge.GICLiv_facts = GICLiv_facts

    PF_RNrm_facts = {
        'about': 'Growth-Normalized Perfect Foresight Return Factor'}
    PF_RNrm = Rfree/PermGroFac_0_
    PF_RNrm_facts.update({'latexexpr': r'\PF_RNrm'})
    PF_RNrm_facts.update({'_unicode_': r'R/Γ'})
    PF_RNrm_facts.update({'py___code': r'Rfree/PermGroFac_0_'})
    PF_RNrm_facts.update({'value_now': PF_RNrm})
    soln_stge.facts.update({'PF_RNrm': PF_RNrm_facts})
    soln_stge.PF_RNrm_facts = PF_RNrm_facts
    soln_stge.PF_RNrm = PF_RNrm

    Inv_PF_RNrm_facts = {
        'about': 'Inverse of Growth-Normalized Perfect Foresight Return Factor'}
    Inv_PF_RNrm = 1/PF_RNrm
    Inv_PF_RNrm_facts.update({'latexexpr': r'\Inv_PF_RNrm'})
    Inv_PF_RNrm_facts.update({'_unicode_': r'Γ/R'})
    Inv_PF_RNrm_facts.update({'py___code': r'PermGroFac_0_Ind/Rfree'})
    Inv_PF_RNrm_facts.update({'value_now': Inv_PF_RNrm})
    soln_stge.facts.update({'Inv_PF_RNrm': Inv_PF_RNrm_facts})
    soln_stge.Inv_PF_RNrm_facts = Inv_PF_RNrm_facts
    soln_stge.Inv_PF_RNrm = Inv_PF_RNrm

    FHWF_facts = {
        'about': 'Finite Human Wealth Factor'}
    FHWF = PermGroFac_0_/Rfree
    FHWF_facts.update({'latexexpr': r'\FHWF'})
    FHWF_facts.update({'_unicode_': r'R/Γ'})
    FHWF_facts.update({'urlhandle': self.soln_stge.urlroot+'FHWF'})
    FHWF_facts.update({'py___code': r'PermGroFac_0_Inf/Rfree'})
    FHWF_facts.update({'value_now': FHWF})
    soln_stge.facts.update({'FHWF': FHWF_facts})
    soln_stge.FHWF_facts = FHWF_facts
    soln_stge.FHWF = FHWF

    FHWC_facts = {'about': 'Finite Human Wealth Condition'}
    FHWC_facts.update({'latexexpr': r'\FHWC'})
    FHWC_facts.update({'urlhandle': self.soln_stge.urlroot+'FHWC'})
    FHWC_facts.update({'py___code': 'test: agent.FHWF < 1'})
    soln_stge.facts.update({'FHWC': FHWC_facts})
    soln_stge.FHWC_facts = FHWC_facts

     hNrmNowInf_facts = {'about':
                          'Human wealth for infinite horizon consumer'}
      hNrmNowInf = float('inf')  # default to infinity
       if FHWF < 1:  # If it is finite, set it to its value
            hNrmNowInf = 1/(1-FHWF)

        soln_stge.hNrmNowInf = hNrmNowInf
        hNrmNowInf_facts = dict({'latexexpr': '1/(1-\FHWF)'})
        hNrmNowInf_facts.update({'value_now': hNrmNowInf})
        hNrmNowInf_facts.update({
            'py___code': '1/(1-FHWF)'})
        soln_stge.facts.update({'hNrmNowInf': hNrmNowInf_facts})
        soln_stge.hNrmNowInf_facts = hNrmNowInf_facts
        # soln_stge.hNrmNowInf = hNrmNowInf

        DiscGPFRawCusp_facts = {
            'about': 'DiscFac s.t. GPFRaw = 1'}
        soln_stge.DiscGPFRawCusp = DiscGPFRawCusp = \
            ((PermGroFac_0_) ** (CRRA)) / (Rfree)
        DiscGPFRawCusp_facts.update({'latexexpr': ''})
        DiscGPFRawCusp_facts.update({'value_now': DiscGPFRawCusp})
        DiscGPFRawCusp_facts.update({
            'py___code': '( PermGroFac_0_                       ** CRRA)/(Rfree)'})
        soln_stge.facts.update({'DiscGPFRawCusp': DiscGPFRawCusp_facts})
        soln_stge.DiscGPFRawCusp_facts = DiscGPFRawCusp_facts

        DiscGPFLivCusp_facts = {
            'about': 'DiscFac s.t. GPFLiv = 1'}
        soln_stge.DiscGPFLivCusp = DiscGPFLivCusp = ((PermGroFac_0_) ** (CRRA)) \
            / (Rfree * LivPrb_0_)
        DiscGPFLivCusp_facts.update({'latexexpr': ''})
        DiscGPFLivCusp_facts.update({'value_now': DiscGPFLivCusp})
        DiscGPFLivCusp_facts.update({
            'py___code': '( PermGroFac_0_                       ** CRRA)/(Rfree*LivPrb_0_)'})
        soln_stge.facts.update({'DiscGPFLivCusp': DiscGPFLivCusp_facts})
        soln_stge.DiscGPFLivCusp_facts = DiscGPFLivCusp_facts

        FVAF_facts = {'about': 'Finite Value of Autarky Factor'}
        soln_stge.FVAF = FVAF = LivPrb_0_ * DiscFacEff * uInv_Ex_uInv_PermShk
        FVAF_facts.update({'latexexpr': r'\FVAFPF'})
        FVAF_facts.update({'urlhandle': self.soln_stge.urlroot+'FVAFPF'})
        soln_stge.facts.update({'FVAF': FVAF_facts})
        soln_stge.FVAF_facts = FVAF_facts

        FVAC_facts = {'about': 'Finite Value of Autarky Condition - Perfect Foresight'}
        FVAC_facts.update({'latexexpr': r'\FVACPF'})
        FVAC_facts.update({'urlhandle': self.soln_stge.urlroot+'FVACPF'})
        FVAC_facts.update({'py___code': 'test: FVAFPF < 1'})
        soln_stge.facts.update({'FVAC': FVAC_facts})
        soln_stge.FVAC_facts = FVAC_facts

        # To reduce "self." clutter in formulae, retrieve local
        # values of useful variables

        # These formulae do not require "live" computation of expectations
        # from a distribution that is on hand.  So, having constructed
        # expected values above, we can use those below.

        # This allows sharing these formulae between the perfect foresight
        # and the non-perfect-foresight models.  They are constructed here
        # and inherited by the descendant model, which augments them with
        # the objects that require live calculation.

        if soln_stge.Inv_PF_RNrm < 1:        # Finite if Rfree > PermGroFac_0_
            soln_stge.hNrmNowInf = 1/(1-soln_stge.Inv_PF_RNrm)

        # Given m, value of c where E[mLev_{t+1}/mLev_{t}]=PermGroFac_0_
        # Solves for c in equation at url/#balgrostable

        soln_stge.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - soln_stge.Inv_PF_RNrm) + soln_stge.Inv_PF_RNrm
        )

        soln_stge.Ex_cLev_tp1_Over_cLev_t_from_mt = (
            lambda m_t:
            soln_stge.Ex_cLev_tp1_Over_pLev_t_from_mt(soln_stge,
                                                      m_t - soln_stge.cFunc(m_t))
            / soln_stge.cFunc(m_t)
        )

#        # E[m_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        soln_stge.Ex_mLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
                soln_stge.PermGroFac_0_ *
            (soln_stge.PF_RNrm * a_t + soln_stge.Ex_IncNextNrm)
        )

        # E[m_{t+1} pLev_{t+1}/(m_{t}pLev_{t})] as a fn of m_{t}
        soln_stge.Ex_mLev_tp1_Over_mLev_t_from_at = (
            lambda m_t:
                soln_stge.Ex_mLev_tp1_Over_pLev_t_from_at(soln_stge,
                                                          m_t-soln_stge.cFunc(m_t)
                                                          )/m_t
        )

        return soln_stge
