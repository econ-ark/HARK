class ConsPerfForesightSolverOld(MetricObject):
    """
    A class for solving a one period perfect foresight
    CRRA utility consumption-saving problem.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one-period problem.
    DiscFac : float
        Intertemporal discount factor for future utility.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the next period.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt : float or None
        Artificial borrowing constraint, as a multiple of permanent income.
        Can be None, indicating no artificial constraint.
    MaxKinks : int, optional
        Maximum number of kink points to allow in the consumption function;
        additional points will be thrown out.  Only relevant in infinite
        horizon model with artificial borrowing constraint.
    """
    # CDC 20200426: MaxKinks adds a good bit of complexity to little purpose
    # because everything it accomplishes could be done using a finite horizon
    # model (including tests of convergence conditions, which can be invoked
    # manually if a user wants them).

    def __init__(
            self, solution_next, DiscFac=1.0, LivPrb=1.0, CRRA=2.0, Rfree=1.0, PermGroFac=1.0, BoroCnstArt=None, MaxKinks=None, **kwds
    ):
        self.soln_futr = soln_futr = solution_next
        # list objects whose _tp1 value is neeeded to solve problem of t
        self.recursive = {'cFunc', 'vFunc', 'vPfunc', 'vPPfunc',
                          'u', 'uP', 'uPP', 'uPinv', 'uPinvP', 'uinvP', 'uinv',
                          'hNrm', 'mNrmMin', 'MPCmin', 'MPCmax', 'BoroCnstNat'}

        self.soln_crnt = ConsumerSolutionOneStateCRRA()

        # Get solver parameters and store for later use
        parameters_solver = \
            {k: v for k, v in {**kwds, **locals()}.items()
             if k not in {'self', 'solution_next', 'kwds', 'soln_futr',
                          'bilt_futr', 'soln_crnt', 'bilt'}}
        # omitting things that would cause recursion

        if hasattr(self.soln_futr.bilt, 'stge_kind') \
                and (soln_futr.bilt.stge_kind['iter_status'] == 'terminal_pseudo'):
            self.soln_crnt.bilt = deepcopy(self.soln_futr.bilt)
        # links for docs; urls are used when "fcts" are added
        self.url_doc_for_solver_get()

        self.soln_crnt.bilt.parameters_solver = deepcopy(parameters_solver)
        # Store the exact params with which solver was called
        # except for solution_next and self (to prevent inf recursion)
        for key in parameters_solver:
            setattr(self.soln_crnt.bilt, key, parameters_solver[key])
            setattr(self.soln_crnt.bilt, key, parameters_solver[key])
        return

    # Methods
    def url_doc_for_solver_get(self):
        # Generate a url that will locate the documentation
        self.class_name = self.__class__.__name__
        self.soln_crnt.bilt.url_ref = self.url_ref =\
            "https://econ-ark.github.io/BufferStockTheory"
        self.soln_crnt.bilt.urlroot = self.urlroot = \
            self.url_ref+'/#'
        self.soln_crnt.bilt.url_doc = self.url_doc = \
            "https://hark.readthedocs.io/en/latest/search.html?q=" +\
            self.class_name+"&check_keywords=yes&area=default#"

    # Arguably def_utility_funcs should be in definition of agent_type
    # But solvers always use characteristics of utility function to solve
    # and therefore must have them.  Single source of truth says they
    # should be in solver since they must be there but only CAN be elsewhere
    def def_utility_funcs(self, stge, CRRA):
        """
        Defines CRRA utility function for this period (and its derivatives,
        and their inverses), saving them as attributes of self for other methods
        to use.

        Parameters
        ----------
        solution_stage

        Returns
        -------
        none
        """
        breakpoint()
        bilt = stge.bilt
        # utility function
        bilt.u = stge.u = lambda c: utility(c, CRRA)
        # marginal utility function
        bilt.uP = stge.uP = lambda c: utilityP(c, CRRA)
        # marginal marginal utility function
        bilt.uPP = stge.uPP = lambda c: utilityPP(c, CRRA)

        # Inverses thereof
        bilt.uPinv = stge.uPinv = lambda uP: utilityP_inv(uP, CRRA)
        bilt.uPinvP = stge.uPinvP = lambda uP: utilityP_invP(uP, CRRA)
        bilt.uinvP = stge.uinvP = lambda uinvP: utility_invP(uinvP, CRRA)
        bilt.uinv = stge.uinv = lambda uinv: utility_inv(uinv, CRRA)  # in case vFuncBool

        return stge

    def def_value_funcs(self, stge):
        """
        Defines the value and marginal value functions for this period.
        See PerfForesightConsumerType.ipynb for a brief explanation
        and the links below for a fuller treatment.

        https://github.com/llorracc/SolvingMicroDSOPs/#vFuncPF

        Parameters
        ----------
        stge

        Returns
        -------
        None

        Notes
        -------
        Uses the fact that for a perfect foresight CRRA utility problem,
        if the MPC in period t is :math:`\kappa_{t}`, and relative risk
        aversion :math:`\\rho`, then the inverse value vFuncNvrs has a
        constant slope of :math:`\\kappa_{t}^{-\\rho/(1-\\rho)}` and
        vFuncNvrs has value of zero at the lower bound of market resources
        """

#        prms = stge.parameters_solver
        breakpoint()
        bilt = stge.bilt
        CRRA = stge.bilt.CRRA

        # See PerfForesightConsumerType.ipynb docs for derivations
        vFuncNvrsSlope = bilt.MPCmin ** (-CRRA / (1.0 - CRRA))
        vFuncNvrs = LinearInterp(
            np.array([bilt.mNrmMin, bilt.mNrmMin + 1.0]),
            np.array([0.0, vFuncNvrsSlope]),
        )
        bilt.vFunc = ValueFuncCRRA(vFuncNvrs, CRRA)
        bilt.vPfunc = MargValueFuncCRRA(bilt.cFunc, CRRA)
        bilt.vPPfunc = MargMargValueFuncCRRA(bilt.cFunc, CRRA)

        return stge

    def make_cFunc_PF(self):
        """
        Makes the (linear) consumption function for this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Reduce cluttered formulae with local aliases

        bild = self.soln_crnt.bilt
        folw = self.soln_crnt.folw
#        futr = self.soln_futr.bilt

        Rfree = bild.Rfree
        PermGroFac = bild.PermGroFac
        MPCmin = bild.MPCmin
        MaxKinks = bild.MaxKinks
        BoroCnstArt = bild.BoroCnstArt
        DiscLiv = bild.DiscLiv
        Ex_IncNrmNxt = bild.Ex_IncNrmNxt
        DiscLiv = bild.DiscLiv

        # Use local value of BoroCnstArt to prevent comparing None and float
        if BoroCnstArt is None:
            BoroCnstArt = -np.inf
        else:
            BoroCnstArt = BoroCnstArt

        # Extract kink points in next period's consumption function;
        # don't take the last one; it only defines extrapolation, is not kink.

        mNrmGrid_tp1 = folw.cFunc_tp1.x_list[:-1]
        cNrmGrid_tp1 = folw.cFunc_tp1.y_list[:-1]

        # Calculate end-of-this-period asset vals that would reach those kink points
        # next period, then invert first order condition to get c. Then find
        # endogenous gridpoint (kink point) today corresponding to each kink

        bNrmGrid_tp1 = mNrmGrid_tp1 - Ex_IncNrmNxt  # = (R/Γψ) aNrmGrid
        aNrmGrid = bNrmGrid_tp1 * (PermGroFac / Rfree)
        EvP_tp1 = (DiscLiv * Rfree) * \
            folw.uP_tp1(PermGroFac * cNrmGrid_tp1)  # Rβ E[u'(Γ c_tp1)]
        cNrmGrid = bild.uPinv(EvP_tp1)  # EGM step 1
        mNrmGrid = aNrmGrid + cNrmGrid  # EGM step 2: DBC inverted

        # Add additional point to the list of gridpoints for extrapolation,
        # using this period's new value of the lower bound of the MPC, which
        # defines the PF unconstrained problem through the end of the horizon
        mNrmGrid = np.append(mNrmGrid, mNrmGrid[-1] + 1.0)
        cNrmGrid = np.append(cNrmGrid, cNrmGrid[-1] + MPCmin)
        # If artificial borrowing constraint binds, combine constrained and
        # unconstrained consumption functions.
        if BoroCnstArt > mNrmGrid[0]:
            # Find the highest index where constraint binds
            cNrmGridCnst = mNrmGrid - BoroCnstArt
            CnstBinds = cNrmGridCnst < cNrmGrid
            idx = np.where(CnstBinds)[0][-1]
            if idx < (mNrmGrid.size - 1):
                # If not the *very last* index, find the the critical level
                # of mNrmGrid where artificial borrowing contraint begins to bind.
                d0 = cNrmGrid[idx] - cNrmGridCnst[idx]
                d1 = cNrmGridCnst[idx + 1] - cNrmGrid[idx + 1]
                m0 = mNrmGrid[idx]
                m1 = mNrmGrid[idx + 1]
                alpha = d0 / (d0 + d1)
                mCrit = m0 + alpha * (m1 - m0)
                # Adjust grids of mNrmGrid and cNrmGrid to account for constraint.
                cCrit = mCrit - BoroCnstArt
                mNrmGrid = np.concatenate(([BoroCnstArt, mCrit], mNrmGrid[(idx + 1):]))
                cNrmGrid = np.concatenate(([0.0, cCrit], cNrmGrid[(idx + 1):]))
            else:
                # If it *is* the last index, then there are only three points
                # that characterize the c function: the artificial borrowing
                # constraint, the constraint kink, and the extrapolation point
                mXtra = (cNrmGrid[-1] - cNrmGridCnst[-1]) / (1.0 - MPCmin)
                mCrit = mNrmGrid[-1] + mXtra
                cCrit = mCrit - BoroCnstArt
                mNrmGrid = np.array([BoroCnstArt, mCrit, mCrit + 1.0])
                cNrmGrid = np.array([0.0, cCrit, cCrit + MPCmin])
                # If mNrmGrid, cNrmGrid grids have become too large, throw out last
                # kink point, being sure to adjust the extrapolation.

        if mNrmGrid.size > MaxKinks:
            mNrmGrid = np.concatenate((mNrmGrid[:-2], [cNrmGrid[-3] + 1.0]))
            cNrmGrid = np.concatenate((cNrmGrid[:-2], [cNrmGrid[-3] + MPCmin]))
            # Construct the consumption function as a linear interpolation.
        self.cFunc = self.soln_crnt.cFunc = bild.cFunc = LinearInterp(mNrmGrid, cNrmGrid)

        # Calculate the upper bound of the MPC as the slope of bottom segment.
        bild.MPCmax = (cNrmGrid[1] - cNrmGrid[0]) / (mNrmGrid[1] - mNrmGrid[0])

        # Lower bound of mNrm is lowest gridpoint -- usually 0
        bild.mNrmMin = mNrmGrid[0]

        # Add the calculated grids to self.bild
        bild.aNrmGrid = aNrmGrid
        bild.EvP_tp1 = EvP_tp1
        bild.cNrmGrid = cNrmGrid
        bild.mNrmGrid = mNrmGrid


#    def build_infhor_facts_from_params_ConsPerfForesightSolver(self):


    def build_infhor_facts_from_params(self):
        """
            Adds to the solution extensive information and references about
            its elements.

            Parameters
            ----------
            solution: ConsumerSolution
                A solution that already has minimal requirements (vPfunc, cFunc)

            Returns
            -------
            solution : ConsumerSolution
                Same solution that was provided, augmented with facts
        """

        # Using local variables makes formulae below more readable

        soln_crnt = self.soln_crnt
#        scsr = self.soln_crnt.scsr
#     breakpoint()
        bilt = self.soln_crnt.bilt
        folw = self.soln_crnt.folw
        urlroot = bilt.urlroot
        bilt.DiscLiv = bilt.DiscFac * bilt.LivPrb

        APF_fcts = {
            'about': 'Absolute Patience Factor'
        }
        py___code = '((Rfree * DiscLiv) ** (1.0 / CRRA))'
#        soln_crnt.APF = \
        bilt.APF = APF = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        APF_fcts.update({'latexexpr': r'\APF'})
        APF_fcts.update({'_unicode_': r'Þ'})
        APF_fcts.update({'urlhandle': urlroot+'APF'})
        APF_fcts.update({'py___code': py___code})
        APF_fcts.update({'value_now': APF})
        # soln_crnt.fcts.update({'APF': APF_fcts})
#        soln_crnt.APF_fcts = \
        bilt.APF_fcts = APF_fcts

        AIC_fcts = {
            'about': 'Absolute Impatience Condition'
        }
        AIC_fcts.update({'latexexpr': r'\AIC'})
        AIC_fcts.update({'urlhandle': urlroot+'AIC'})
        AIC_fcts.update({'py___code': 'test: APF < 1'})
        # soln_crnt.fcts.update({'AIC': AIC_fcts})
#        soln_crnt.AIC_fcts =
        bilt.AIC_fcts = AIC_fcts

        RPF_fcts = {
            'about': 'Return Patience Factor'
        }
        py___code = 'APF / Rfree'
#        soln_crnt.RPF = \
        bilt.RPF = RPF = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        RPF_fcts.update({'latexexpr': r'\RPF'})
        RPF_fcts.update({'_unicode_': r'Þ_R'})
        RPF_fcts.update({'urlhandle': urlroot+'RPF'})
        RPF_fcts.update({'py___code': py___code})
        RPF_fcts.update({'value_now': RPF})
        # soln_crnt.fcts.update({'RPF': RPF_fcts})
#        soln_crnt.RPF_fcts = \
        bilt.RPF_fcts = RPF_fcts

        RIC_fcts = {
            'about': 'Growth Impatience Condition'
        }
        RIC_fcts.update({'latexexpr': r'\RIC'})
        RIC_fcts.update({'urlhandle': urlroot+'RIC'})
        RIC_fcts.update({'py___code': 'test: RPF < 1'})
        # soln_crnt.fcts.update({'RIC': RIC_fcts})
#        soln_crnt.RIC_fcts = \
        bilt.RIC_fcts = RIC_fcts

        GPFRaw_fcts = {
            'about': 'Growth Patience Factor'
        }
        py___code = 'APF / PermGroFac'
#        soln_crnt.GPFRaw = \
        bilt.GPFRaw = GPFRaw = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        GPFRaw_fcts.update({'latexexpr': '\GPFRaw'})
        GPFRaw_fcts.update({'_unicode_': r'Þ_Γ'})
        GPFRaw_fcts.update({'urlhandle': urlroot+'GPFRaw'})
        GPFRaw_fcts.update({'py___code': py___code})
        GPFRaw_fcts.update({'value_now': GPFRaw})
        # soln_crnt.fcts.update({'GPFRaw': GPFRaw_fcts})
#        soln_crnt.GPFRaw_fcts = \
        bilt.GPFRaw_fcts = GPFRaw_fcts

        GICRaw_fcts = {
            'about': 'Growth Impatience Condition'
        }
        GICRaw_fcts.update({'latexexpr': r'\GICRaw'})
        GICRaw_fcts.update({'urlhandle': urlroot+'GICRaw'})
        GICRaw_fcts.update({'py___code': 'test: GPFRaw < 1'})
        # soln_crnt.fcts.update({'GICRaw': GICRaw_fcts})
#        soln_crnt.GICRaw_fcts = \
        bilt.GICRaw_fcts = GICRaw_fcts

        GPFLiv_fcts = {
            'about': 'Mortality-Adjusted Growth Patience Factor'
        }
        py___code = 'APF * LivPrb / PermGroFac'
#        soln_crnt.GPFLiv = \
        bilt.GPFLiv = GPFLiv = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        GPFLiv_fcts.update({'latexexpr': '\GPFLiv'})
        GPFLiv_fcts.update({'urlhandle': urlroot+'GPFLiv'})
        GPFLiv_fcts.update({'py___code': py___code})
        GPFLiv_fcts.update({'value_now': GPFLiv})
        # soln_crnt.fcts.update({'GPFLiv': GPFLiv_fcts})
#        soln_crnt.GPFLiv_fcts = \
        bilt.GPFLiv_fcts = GPFLiv_fcts

        GICLiv_fcts = {
            'about': 'Growth Impatience Condition'
        }
        GICLiv_fcts.update({'latexexpr': r'\GICLiv'})
        GICLiv_fcts.update({'urlhandle': urlroot+'GICLiv'})
        GICLiv_fcts.update({'py___code': 'test: GPFLiv < 1'})
        # soln_crnt.fcts.update({'GICLiv': GICLiv_fcts})
#        soln_crnt.GICLiv_fcts = \
        bilt.GICLiv_fcts = GICLiv_fcts

        PF_RNrm_fcts = {
            'about': 'Growth-Normalized PF Return Factor'
        }
        py___code = 'Rfree/PermGroFac'
#        soln_crnt.PF_RNrm = \
        bilt.PF_RNrm = PF_RNrm = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        PF_RNrm_fcts.update({'latexexpr': r'\PFRNrm'})
        PF_RNrm_fcts.update({'_unicode_': r'R/Γ'})
        PF_RNrm_fcts.update({'py___code': py___code})
        PF_RNrm_fcts.update({'value_now': PF_RNrm})
        # soln_crnt.fcts.update({'PF_RNrm': PF_RNrm_fcts})
#        soln_crnt.PF_RNrm_fcts = \
        bilt.PF_RNrm_fcts = PF_RNrm_fcts
#        soln_crnt.PF_RNrm = PF_RNrm

        Inv_PF_RNrm_fcts = {
            'about': 'Inv of Growth-Normalized PF Return Factor'
        }
        py___code = '1 / PF_RNrm'
#        soln_crnt.Inv_PF_RNrm = \
        bilt.Inv_PF_RNrm = Inv_PF_RNrm = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        Inv_PF_RNrm_fcts.update({'latexexpr': r'\InvPFRNrm'})
        Inv_PF_RNrm_fcts.update({'_unicode_': r'Γ/R'})
        Inv_PF_RNrm_fcts.update({'py___code': py___code})
        Inv_PF_RNrm_fcts.update({'value_now': Inv_PF_RNrm})
        # soln_crnt.fcts.update({'Inv_PF_RNrm': Inv_PF_RNrm_fcts})
#        soln_crnt.Inv_PF_RNrm_fcts = \
        bilt.Inv_PF_RNrm_fcts = \
            Inv_PF_RNrm_fcts

        FHWF_fcts = {
            'about': 'Finite Human Wealth Factor'
        }
        py___code = 'PermGroFac / Rfree'
#        soln_crnt.FHWF = \
        bilt.FHWF = FHWF = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        FHWF_fcts.update({'latexexpr': r'\FHWF'})
        FHWF_fcts.update({'_unicode_': r'R/Γ'})
        FHWF_fcts.update({'urlhandle': urlroot+'FHWF'})
        FHWF_fcts.update({'py___code': py___code})
        FHWF_fcts.update({'value_now': FHWF})
        # soln_crnt.fcts.update({'FHWF': FHWF_fcts})
#        soln_crnt.FHWF_fcts = \
        bilt.FHWF_fcts = \
            FHWF_fcts

        FHWC_fcts = {
            'about': 'Finite Human Wealth Condition'
        }
        FHWC_fcts.update({'latexexpr': r'\FHWC'})
        FHWC_fcts.update({'urlhandle': urlroot+'FHWC'})
        FHWC_fcts.update({'py___code': 'test: FHWF < 1'})
        # soln_crnt.fcts.update({'FHWC': FHWC_fcts})
#        soln_crnt.FHWC_fcts = \
        bilt.FHWC_fcts = FHWC_fcts

        hNrmInf_fcts = {
            'about': 'Human wealth for inf hor'
        }
        py___code = '1/(1-FHWF) if (FHWF < 1) else float("inf")'
#        soln_crnt.hNrmInf = \
        bilt.hNrmInf = hNrmInf = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        hNrmInf_fcts = dict({'latexexpr': '1/(1-\FHWF)'})
        hNrmInf_fcts.update({'value_now': hNrmInf})
        hNrmInf_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'hNrmInf': hNrmInf_fcts})
#        soln_crnt.hNrmInf_fcts = \
        bilt.hNrmInf_fcts = hNrmInf_fcts

        DiscGPFRawCusp_fcts = {
            'about': 'DiscFac s.t. GPFRaw = 1'
        }
        py___code = '( PermGroFac                       ** CRRA)/(Rfree)'
#        soln_crnt.DiscGPFRawCusp = \
        bilt.DiscGPFRawCusp = DiscGPFRawCusp = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        DiscGPFRawCusp_fcts.update({'latexexpr': '\PermGroFac^{\CRRA}/\Rfree'})
        DiscGPFRawCusp_fcts.update({'value_now': DiscGPFRawCusp})
        DiscGPFRawCusp_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'DiscGPFRawCusp': DiscGPFRawCusp_fcts})
#        soln_crnt.DiscGPFRawCusp_fcts = \
        bilt.DiscGPFRawCusp_fcts = \
            DiscGPFRawCusp_fcts

        DiscGPFLivCusp_fcts = {
            'about': 'DiscFac s.t. GPFLiv = 1'
        }
        py___code = '( PermGroFac                       ** CRRA)/(Rfree*LivPrb)'
#        soln_crnt.DiscGPFLivCusp = \
        bilt.DiscGPFLivCusp = DiscGPFLivCusp = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        DiscGPFLivCusp_fcts.update({'latexexpr': '\PermGroFac^{\CRRA}/(\Rfree\LivPrb)'})
        DiscGPFLivCusp_fcts.update({'value_now': DiscGPFLivCusp})
        DiscGPFLivCusp_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'DiscGPFLivCusp': DiscGPFLivCusp_fcts})
#        soln_crnt.DiscGPFLivCusp_fcts = \
        bilt.DiscGPFLivCusp_fcts = DiscGPFLivCusp_fcts

        FVAF_fcts = {  # overwritten by version with uncertainty
            'about': 'Finite Value of Autarky Factor'
        }
        py___code = 'LivPrb * DiscLiv'
#        soln_crnt.FVAF = \
        bilt.FVAF = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        FVAF_fcts.update({'latexexpr': r'\FVAFPF'})
        FVAF_fcts.update({'urlhandle': urlroot+'FVAFPF'})
        FVAF_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'FVAF': FVAF_fcts})
#        soln_crnt.FVAF_fcts = \
        bilt.FVAF_fcts = FVAF_fcts

        FVAC_fcts = {  # overwritten by version with uncertainty
            'about': 'Finite Value of Autarky Condition - Perfect Foresight'
        }
        FVAC_fcts.update({'latexexpr': r'\FVACPF'})
        FVAC_fcts.update({'urlhandle': urlroot+'FVACPF'})
        FVAC_fcts.update({'py___code': 'test: FVAFPF < 1'})
        # soln_crnt.fcts.update({'FVAC': FVAC_fcts})
#        soln_crnt.FVAC_fcts = \
        bilt.FVAC_fcts = FVAC_fcts

        Ex_IncNrmNxt_fcts = {  # Overwritten by version with uncertainty
            'about': 'Expected income next period'
        }
        py___code = '1.0'
#        soln_crnt.Ex_IncNrmNxt = \
        bilt.Ex_IncNrmNxt = Ex_IncNrmNxt = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
#        Ex_IncNrmNxt_fcts.update({'latexexpr': r'\Ex_IncNrmNxt'})
#        Ex_IncNrmNxt_fcts.update({'_unicode_': r'R/Γ'})
#        Ex_IncNrmNxt_fcts.update({'urlhandle': urlroot+'ExIncNrmNxt'})
        Ex_IncNrmNxt_fcts.update({'py___code': py___code})
        Ex_IncNrmNxt_fcts.update({'value_now': Ex_IncNrmNxt})
        # soln_crnt.fcts.update({'Ex_IncNrmNxt': Ex_IncNrmNxt_fcts})
        soln_crnt.Ex_IncNrmNxt_fcts = soln_crnt.bilt.Ex_IncNrmNxt_fcts = Ex_IncNrmNxt_fcts

        PF_RNrm_fcts = {
            'about': 'Expected Growth-Normalized Return'
        }
        py___code = 'Rfree / PermGroFac'
#        soln_crnt.PF_RNrm = \
        bilt.PF_RNrm = PF_RNrm = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        PF_RNrm_fcts.update({'latexexpr': r'\PFRNrm'})
        PF_RNrm_fcts.update({'_unicode_': r'R/Γ'})
        PF_RNrm_fcts.update({'urlhandle': urlroot+'PFRNrm'})
        PF_RNrm_fcts.update({'py___code': py___code})
        PF_RNrm_fcts.update({'value_now': PF_RNrm})
        # soln_crnt.fcts.update({'PF_RNrm': PF_RNrm_fcts})
#        soln_crnt.PF_RNrm_fcts = \
        bilt.PF_RNrm_fcts = PF_RNrm_fcts

        PF_RNrm_fcts = {
            'about': 'Expected Growth-Normalized Return'
        }
        py___code = 'Rfree / PermGroFac'
#        soln_crnt.PF_RNrm = \
        bilt.PF_RNrm = PF_RNrm = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        PF_RNrm_fcts.update({'latexexpr': r'\PFRNrm'})
        PF_RNrm_fcts.update({'_unicode_': r'R/Γ'})
        PF_RNrm_fcts.update({'urlhandle': urlroot+'PFRNrm'})
        PF_RNrm_fcts.update({'py___code': py___code})
        PF_RNrm_fcts.update({'value_now': PF_RNrm})
        # soln_crnt.fcts.update({'PF_RNrm': PF_RNrm_fcts})
#        soln_crnt.PF_RNrm_fcts = \
        bilt.PF_RNrm_fcts = PF_RNrm_fcts

        DiscLiv_fcts = {
            'about': 'Mortality-Inclusive Discounting'
        }
        py___code = 'DiscFac * LivPrb'
#        soln_crnt.DiscLiv = \
        bilt.DiscLiv = DiscLiv = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        DiscLiv_fcts.update({'latexexpr': r'\PFRNrm'})
        DiscLiv_fcts.update({'_unicode_': r'R/Γ'})
        DiscLiv_fcts.update({'urlhandle': urlroot+'PFRNrm'})
        DiscLiv_fcts.update({'py___code': py___code})
        DiscLiv_fcts.update({'value_now': DiscLiv})
        # soln_crnt.fcts.update({'DiscLiv': DiscLiv_fcts})
#        soln_crnt.DiscLiv_fcts = \
        bilt.DiscLiv_fcts = DiscLiv_fcts

#    def build_recursive_facts_ConsPerfForesightSolver(self):
    def build_recursive_facts(self):

        soln_crnt = self.soln_crnt
        bilt = self.soln_crnt.bilt
        folw = soln_crnt.folw
        urlroot = bilt.urlroot
        bilt.DiscLiv = bilt.DiscFac * bilt.LivPrb

#        breakpoint()
        hNrm_fcts = {
            'about': 'Human Wealth '
        }
        py___code = '((PermGroFac / Rfree) * (1.0 + hNrm_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            #        if soln_crnt.bilt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            soln_crnt.hNrm_tp1 = -1.0  # causes hNrm = 0 for final period
#        soln_crnt.hNrm = \
        bilt.hNrm = hNrm = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        hNrm_fcts.update({'latexexpr': r'\hNrm'})
        hNrm_fcts.update({'_unicode_': r'R/Γ'})
        hNrm_fcts.update({'urlhandle': urlroot+'hNrm'})
        hNrm_fcts.update({'py___code': py___code})
        hNrm_fcts.update({'value_now': hNrm})
        # soln_crnt.fcts.update({'hNrm': hNrm_fcts})
#        soln_crnt.hNrm_fcts = \
        bilt.hNrm_fcts = hNrm_fcts

        BoroCnstNat_fcts = {
            'about': 'Natural Borrowing Constraint'
        }
        py___code = '(mNrmMin_tp1 - tranShkMin)*(PermGroFac/Rfree)*permShkMin'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge
            #        if soln_crnt.bilt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge
            py___code = 'hNrm'  # Presumably zero
#        soln_crnt.BoroCnstNat = \
        bilt.BoroCnstNat = BoroCnstNat = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        BoroCnstNat_fcts.update({'latexexpr': r'\BoroCnstNat'})
        BoroCnstNat_fcts.update({'_unicode_': r''})
        BoroCnstNat_fcts.update({'urlhandle': urlroot+'BoroCnstNat'})
        BoroCnstNat_fcts.update({'py___code': py___code})
        BoroCnstNat_fcts.update({'value_now': BoroCnstNat})
        # soln_crnt.fcts.update({'BoroCnstNat': BoroCnstNat_fcts})
#        soln_crnt.BoroCnstNat_fcts = \
        bilt.BoroCnstNat_fcts = BoroCnstNat_fcts

        BoroCnst_fcts = {
            'about': 'Effective Borrowing Constraint'
        }
        py___code = 'BoroCnstNat if (BoroCnstArt == None) else (BoroCnstArt if BoroCnstNat < BoroCnstArt else BoroCnstNat)'
#        soln_crnt.BoroCnst = \
        bilt.BoroCnst = BoroCnst = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        BoroCnst_fcts.update({'latexexpr': r'\BoroCnst'})
        BoroCnst_fcts.update({'_unicode_': r''})
        BoroCnst_fcts.update({'urlhandle': urlroot+'BoroCnst'})
        BoroCnst_fcts.update({'py___code': py___code})
        BoroCnst_fcts.update({'value_now': BoroCnst})
        # soln_crnt.fcts.update({'BoroCnst': BoroCnst_fcts})
#        soln_crnt.BoroCnst_fcts = \
        bilt.BoroCnst_fcts = BoroCnst_fcts

        # MPCmax is not a meaningful object in the PF model so is not created there
        # so create it here
        MPCmax_fcts = {
            'about': 'Maximal MPC in current period as m -> mNrmMin'
        }
        py___code = '1.0 / (1.0 + (RPF / MPCmax_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            soln_crnt.MPCmax_tp1 = float('inf')  # causes MPCmax = 1 for final period
#        soln_crnt.MPCmax = \
        bilt.MPCmax = MPCmax = eval(
            py___code, {}, {**bilt.__dict__, **folw.__dict__})
        MPCmax_fcts.update({'latexexpr': r''})
        MPCmax_fcts.update({'urlhandle': urlroot+'MPCmax'})
        MPCmax_fcts.update({'py___code': py___code})
        MPCmax_fcts.update({'value_now': MPCmax})
        # soln_crnt.fcts.update({'MPCmax': MPCmax_fcts})
#        soln_crnt.bilt.MPCmax_fcts = \
        bilt.MPCmax_fcts = MPCmax_fcts

        mNrmMin_fcts = {
            'about': 'Min m is the max you can borrow'
        }
        py___code = 'BoroCnst'
#        soln_crnt.mNrmMin = \
        bilt.mNrmMin =  \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        mNrmMin_fcts.update({'latexexpr': r'\mNrmMin'})
        mNrmMin_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'mNrmMin': mNrmMin_fcts})
#        soln_crnt.mNrmMin_fcts = \
        bilt.mNrmMin_fcts = mNrmMin_fcts

        MPCmin_fcts = {
            'about': 'Minimal MPC in current period as m -> infty'
        }
        py___code = '1.0 / (1.0 + (RPF /MPCmin_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            #        if soln_crnt.bilt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            bilt.MPCmin_tp1 = float('inf')  # causes MPCmin = 1 for final period
#        soln_crnt.MPCmin = \
        bilt.MPCmin = MPCmin = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        MPCmin_fcts.update({'latexexpr': r''})
        MPCmin_fcts.update({'urlhandle': urlroot+'MPCmin'})
        MPCmin_fcts.update({'py___code': py___code})
        MPCmin_fcts.update({'value_now': MPCmin})
        # soln_crnt.fcts.update({'MPCmin': MPCmin_fcts})
#        soln_crnt.MPCmin_fcts = \
        bilt.MPCmin_fcts = MPCmin_fcts

        MPCmax_fcts = {
            'about': 'Maximal MPC in current period as m -> mNrmMin'
        }
        py___code = '1.0 / (1.0 + (RPF / MPCmax_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            #        if soln_crnt.bilt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            bilt.MPCmax_tp1 = float('inf')  # causes MPCmax = 1 for final period
#        soln_crnt.MPCmax = \
        bilt.MPCmax = MPCmax = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        MPCmax_fcts.update({'latexexpr': r''})
        MPCmax_fcts.update({'urlhandle': urlroot+'MPCmax'})
        MPCmax_fcts.update({'py___code': py___code})
        MPCmax_fcts.update({'value_now': MPCmax})
        # soln_crnt.fcts.update({'MPCmax': MPCmax_fcts})
#        soln_crnt.MPCmax_fcts = \
        bilt.MPCmax_fcts = MPCmax_fcts

        cFuncLimitIntercept_fcts = {
            'about': 'Vertical intercept of perfect foresight consumption function'}
        py___code = 'MPCmin * hNrm'
#        soln_crnt.cFuncLimitIntercept = \
        bilt.cFuncLimitIntercept = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        cFuncLimitIntercept_fcts.update({'py___code': py___code})
        cFuncLimitIntercept_fcts.update({'latexexpr': '\MPC \hNrm'})
#        cFuncLimitIntercept_fcts.update({'urlhandle': ''})
#        cFuncLimitIntercept_fcts.update({'value_now': cFuncLimitIntercept})
#        cFuncLimitIntercept_fcts.update({'cFuncLimitIntercept': cFuncLimitIntercept_fcts})
        soln_crnt.cFuncLimitIntercept_fcts = cFuncLimitIntercept_fcts

        cFuncLimitSlope_fcts = {
            'about': 'Slope of limiting consumption function'}
        py___code = 'MPCmin'
        cFuncLimitSlope_fcts.update({'py___code': 'MPCmin'})
        bilt.cFuncLimitSlope = soln_crnt.cFuncLimitSlope = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        cFuncLimitSlope_fcts.update({'py___code': py___code})
        cFuncLimitSlope_fcts = dict({'latexexpr': '\MPCmin'})
        cFuncLimitSlope_fcts.update({'urlhandle': '\MPC'})
#        cFuncLimitSlope_fcts.update({'value_now': cFuncLimitSlope})
#        stg_crt.fcts.update({'cFuncLimitSlope': cFuncLimitSlope_fcts})
        soln_crnt.cFuncLimitSlope_fcts = cFuncLimitSlope_fcts
        # That's the end of things that are identical for PF and non-PF models

        return soln_crnt

    def solve_prepared_stage(self):  # ConsPerfForesightSolver
        """
        Solves the one-period/stage perfect foresight consumption-saving problem.

        Parameters
        ----------
        None (all should be in self)

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period/stage's problem.
        """
        soln_futr = self.soln_futr
        soln_crnt = self.soln_crnt
        soln_futr_bilt = soln_futr.bilt
        soln_crnt_bilt = soln_crnt.bilt
        CRRA = soln_crnt.bilt.CRRA

        if not hasattr(soln_futr.bilt, 'stge_kind'):
            print('no stge_kind')
            breakpoint()

        if soln_futr.bilt.stge_kind['iter_status'] == 'finished':
            breakpoint()
            # Should not have gotten here
            # because core.py tests whehter solution_last is 'finished'

        if soln_futr_bilt.stge_kind['iter_status'] == 'terminal_pseudo':
            # bare-bones default terminal solution does not have all the facts
            # we want, so add them
            #            breakpoint()
            soln_futr_bilt = soln_crnt_bilt = def_utility(soln_crnt, CRRA)
#            self.build_infhor_facts_from_params_ConsPerfForesightSolver()
            self.build_infhor_facts_from_params()
#            breakpoint()
            soln_futr = soln_crnt = def_value_funcs(soln_crnt, CRRA)
            # Now that they've been added, it's good to go as a source for iteration
            if not hasattr(soln_crnt.bilt, 'stge_kind'):
                print('No stge_kind')
                breakpoint()
            soln_crnt.bilt.stge_kind['iter_status'] = 'iterator'
            soln_crnt.stge_kind = soln_crnt.bilt.stge_kind
            self.soln_crnt.vPfunc = self.soln_crnt.bilt.vPfunc  # Need for distance
            self.soln_crnt.cFunc = self.soln_crnt.bilt.cFunc  # Need for distance
            if hasattr(self.soln_crnt.bilt, 'IncShkDstn'):
                self.soln_crnt.IncShkDstn = self.soln_crnt.bilt.IncShkDstn

#            breakpoint()
            return soln_crnt  # if pseudo_terminal = True, enhanced replaces original

        # self.soln_crnt.bilt.stge_kind = \
        #     self.soln_crnt.stge_kind = {'iter_status': 'iterator',
        #                                 'slvr_type': self.__class.__name}

#        breakpoint()
        CRRA = self.soln_crnt.bilt.CRRA
        self.soln_crnt = def_utility(soln_crnt, CRRA)
#        breakpoint()  # Need to build evPfut here, but previously had it building current
#        self.build_infhor_facts_from_params_ConsPerfForesightSolver()
        self.build_infhor_facts_from_params()
        self.build_recursive_facts()
        self.make_cFunc_PF()
#        breakpoint()
        CRRA = soln_crnt.bilt.CRRA
        soln_crnt = def_value_funcs(soln_crnt, CRRA)
#        breakpoint()

        return soln_crnt

    solve = solve_prepared_stage

    def solver_prep_solution_for_an_iteration(self):  # self: solver for this stage
        """
        Prepare the current stage for processing by the one-stage solver.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
#        breakpoint()
        soln_crnt = self.soln_crnt
        soln_futr = self.soln_futr
#        breakpoint()
        bilt = soln_crnt.bilt

        # Organizing principle: folw should have a deepcopy of everything
        # needed to re-solve crnt problem; and everything needed to construct
        # the "fcts" about current stage of the problem, so that the stge could
        # be deepcopied as a standalone object and solved without soln_futr
        # or soln_crnt

        folw = soln_crnt.folw = SuccessorInfo()
        # for avoid_recursion in {'solution__tp1', 'scsr'}:
        #     if hasattr(self.soln_crnt._tp1, avoid_recursion):
        #         delattr(self.soln_crnt._tp1, avoid_recursion)

        # Add '_tp1' to names of variables in self.soln_futr.bilt
        # and put in soln_crnt.folw

        # Catch degenerate case of zero-variance income distributions
        if hasattr(bilt, "tranShkVals") and hasattr(bilt, "permShkVals"):
            if ((bilt.tranShkMin == 1.0) and (bilt.permShkMin == 1.0)):
                bilt.Ex_Inv_permShk = 1.0
                bilt.Ex_uInv_permShk = 1.0
        else:
            bilt.tranShkMin = bilt.permShkMin = 1.0

        if hasattr(bilt, 'stge_kind'):
            if 'iter_status' in bilt.stge_kind:
                if (bilt.stge_kind['iter_status'] == 'terminal_pseudo'):
                    # No work needed in terminal period, which replaces itself
                    return

        if not ('MPCmin' in soln_futr.bilt.__dict__):
            print('Breaking because no MPCmin')
            breakpoint()

        for key in (k for k in self.recursive
                    if k not in
                    {'solution_next', 'bilt', 'stge_kind', 'folw'}):
            setattr(folw, key+'_tp1',
                    soln_futr.bilt.__dict__[key])

        self.soln_crnt.stge_kind = self.soln_crnt.bilt.stge_kind = {'iter_status': 'iterator',
                                                                    'slvr_type': self.__class__.__name__}

        return

    # Disambiguate confusing "prepare_to_solve" from similar method names elsewhere
    # (preserve "prepare_to_solve" as alias because core.py calls prepare_to_solve)
    prepare_to_solve = solver_prep_solution_for_an_iteration
