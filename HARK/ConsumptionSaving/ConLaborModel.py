import numpy as np
from scipy.optimize import brentq
from copy import deepcopy
from HARK.core import AgentType
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import ConsumerSolution
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import ValueFunc, MargValueFunc
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks
from HARK.utilities import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityP_inv
from HARK.interpolation import LinearInterp, LowerEnvelope
from HARK.distribution import *

# pre-set the parameters
testParam = deepcopy(init_idiosyncratic_shocks)
testParam.update(
    {'PermGroFacEmp': [0.0, 0.0],
     'PermGroFacUnemp': [0.0, 0.0],
     'IncUnemp': [0.0, 0.0],
     'LifeSpan': 2.0,
     'N_bar': 2
     }
)


class LaborSearchConsumerType(MarkovConsumerType):
    '''
    A consumer type with idiosyncratic shocks to permanent and transitory income. His utility function
    is non-separable. And he is subject to an exogenous probability of being fired if he is currently
    employed. If he is unemployed, he need to decide how much effort he will exert to be re-employed.
    His problem is defined by a sequence of income distributions, survival probabilities, and permanent
    income growth rates, as well as time invariant values for risk aversion, discount factor, the interest
    rate, the grid of end-of-period assets, the exogenous unemployment probability and an artificial
    borrowing constraint.
    '''

    def __init__(self, cycles=1, **kwds):
        '''
        Instantiate a new consumer type with given data.

        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.

        Returns
        -------
        None
        '''

        params = testParam.copy()
        params.update(kwds)
        super().__init__(cycles=cycles, **params)
        self.solveOnePeriod = self.solveConsLaborSearch

    def checkMarkovInputs(self):
        '''
        This overwrites a method of the parent class that would not work as intended in this model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        pass

    def preSolve(self):
        '''
        Check to make sure that the inputs that are specific to LaborSearchConsumerType
        are of the right shape (if arrays) or length (if lists). If they are of right shape,
        then update the parameters used for solver function.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        AgentType.preSolve(self)
        self.update()

    def update(self):
        '''
        Update the income process, the assets grid, the terminal solution, and growth factor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        super().update()
        self.updatePermGroFac()

    def updatePermGroFac(self):
        '''
        Construct the attribute PermGroFac as a time-varying. It should have the length of unemployment state + 1.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        PermGroFac = []
        for j in range(0, self.T_cycle):
            PermGroFac_by_age = np.asarray(self.PermGroFacUnemp)
            PermGroFac_by_age = np.insert(PermGroFac_by_age, 0, self.PermGroFacEmp[j])
            PermGroFac.append(PermGroFac_by_age)

        self.PermGroFac = deepcopy(PermGroFac)
        self.addToTimeVary('PermGroFac')

    def updateIncomeProcess(self):
        '''
        constructs the attributes IncomeDstn, PermShkDstn, and TranShkDstn using the primitive attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        IncUnemp_all = deepcopy(self.IncUnemp)
        UnempPrb = self.UnempPrb

        self.IncUnemp = [0.0 for _ in range(0, len(IncUnemp_all))]
        self.UnempPrb = 0.0

        IndShockConsumerType.updateIncomeProcess(self)
        IncomeDstn_emp = deepcopy(self.IncomeDstn)
        PermShkDstn_emp = deepcopy(self.PermShkDstn)
        TranShkDstn_emp = deepcopy(self.TranShkDstn)

        PermShkDstn_unemp = DiscreteDistribution(np.array([1.0]), np.array([1.0]))
        TranShkDstn_unemp = []
        IncomeDstn_by_unemp = []
        for n in range(0, self.Statecount):
            TranDiscDstn_tempobj = DiscreteDistribution(np.array([1.0]), np.array(IncUnemp_all[n]))
            TranShkDstn_unemp.append(TranDiscDstn_tempobj)
            IncomeDstn_tempobj = combineIndepDstns(PermShkDstn_unemp, TranDiscDstn_tempobj)
            IncomeDstn_by_unemp.append(IncomeDstn_tempobj)

        IncomeDstn = []
        TranShkDstn = []
        PermShkDstn = []

        for j in range(0, self.T_cycle):
            IncomeDstn_temp = [IncomeDstn_emp[j]]
            TranShkDstn_temp = [TranShkDstn_emp[j]]
            PermShkDstn_temp = [PermShkDstn_emp[j]]
            for incdstn, transdstn in zip(IncomeDstn_by_unemp, TranShkDstn_unemp):
                IncomeDstn_temp.append(incdstn)
                TranShkDstn_temp.append(transdstn)
                PermShkDstn_temp.append(PermShkDstn_unemp)

            IncomeDstn.append(IncomeDstn_temp)
            TranShkDstn.append(TranShkDstn_temp)
            PermShkDstn.append(PermShkDstn_temp)

        self.TranShkDstn = deepcopy(TranShkDstn)
        self.IncomeDstn = deepcopy(IncomeDstn)
        self.PermShkDstn = deepcopy(PermShkDstn)

        self.addToTimeVary('TranShkDstn', 'PermShkDstn', 'IncomeDstn')

        self.IncUnemp = IncUnemp_all
        self.UnempPrb = UnempPrb

    def solveConsLaborSearch(self, solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rfree, PermGroFac, BoroCnstArt,
                             SearchCost, aXtraGrid, SepaRte):
        '''
        Solver function that solves for one period.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one-period problem.
        IncomeDstn: [[np.array]]
            A length N list of income distributions in each succeeding unemployment state.
            Each income distribution contains three arrays of floats,representing a discrete
            approximation to the income process at the beginning of the succeeding period.
            Order: event probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of the next period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : list
            Expected permanent income growth factor in each unemployment state.
        BoroCnstArt : float or None
            Artificial borrowing constraint, as a multiple of permanent income.
            Can be None, indicating no artificial constraint.
        SearchCost: float
            Cost of employment parameter at this age.
        SepaRte: float
            The job separation probability at this age.
        aXtraGrid: np.array
            Gridpoints represents “normalized assets above minimum”, at which
            the model will be solved by the endogenous grid method.

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period's problem.
        '''

        # 1
        utilitySepa = lambda c: CRRAutility(c, CRRA)
        utility = lambda c, s: utilitySepa(c) * (1.0 - s ** SearchCost) ** (1.0 - CRRA)
        utilityPc = lambda c, s: CRRAutilityP(c, CRRA) * (1 - s ** SearchCost) ** (1.0 - CRRA)
        utility_inv = lambda u: CRRAutility_inv(u, CRRA)
        utilityP_inv = lambda uP: CRRAutilityP_inv(uP, CRRA)

        # 2
        EndOfPrdvFunc_by_state_next = []
        EndOfPrdvPfunc_by_state_next = []

        # 3
        BoroCnstNat = np.zeros(self.StateCount) + np.nan

        # 4
        for n in range(self.StateCount):
            # (a)
            # unpack "shock distributions" into different arrays for later use
            ShkProb = IncomeDstn[n].pmf
            ShkValPerm = IncomeDstn[n].X[0]
            ShkValTran = IncomeDstn[n].X[1]

            # (b)
            # make tiled version of shocks
            ShkPrbs_rep = np.tile(ShkProb, (aXtraGrid.size, 1))
            ShkValPerm_rep = np.tile(ShkValPerm, (aXtraGrid.size, 1))
            ShkValTran_rep = np.tile(ShkValTran, (aXtraGrid.size, 1))

            # (c)
            # calculate the minimum value of a_t given next period unemployment state and minimum shock realizations,
            # this serves as the natural borrowing constraint for this state.
            PermShkMinNext = np.min(ShkValPerm)
            TranShkMinNext = np.min(ShkValTran)
            BoroCnstNat[n] = (solution_next.mNrmMin[n] - TranShkMinNext) * (PermGroFac[n] * PermShkMinNext) / Rfree

            # (d)
            aNrmNowGrid = np.asarray(aXtraGrid) + BoroCnstNat[n]
            ShkCount = ShkValPerm.size
            aNrmNowGrid_rep = np.tile(aNrmNowGrid, (ShkCount, 1))
            aNrmNowGrid_rep = aNrmNowGrid_rep.transpose()

            # (e)
            # using equation on the top of page2 to calculate the next period m values.
            mNrmNext = Rfree / (PermGroFac[n] * ShkValPerm_rep) * aNrmNowGrid_rep + ShkValTran_rep

            # (f)
            # evaluate future realizations of value and marginal value at above m values.
            vFunc_next = solution_next.vFunc[n](mNrmNext)
            vPfunc_next = solution_next.vPfunc[n](mNrmNext)

            # (g)
            # calculate the pseudo (marginal) value function value
            EndOfPrdv = DiscFac * LivPrb * PermGroFac[n] ** (1.0 - CRRA) * \
                        np.sum(ShkValPerm_rep ** (1.0 - CRRA) * vFunc_next * ShkPrbs_rep, axis=1)
            EndOfPrdvP = DiscFac * LivPrb * PermGroFac[n] ** (-CRRA) * Rfree * \
                         np.sum(ShkValPerm_rep ** (-CRRA) * vPfunc_next * ShkPrbs_rep, axis=1)

            # (h)
            # evaluate the inverse pseudo (marginal) utility function
            EndOfPrdvNvrs_cond = utility_inv(EndOfPrdv)
            EndOfPrdvNvrs_cond = np.insert(EndOfPrdvNvrs_cond, 0, 0.0)
            EndOfPrdvPnvrs_cond = utilityP_inv(EndOfPrdvP)
            EndOfPrdvPnvrs_cond = np.insert(EndOfPrdvPnvrs_cond, 0, 0.0)

            # (i)
            aNrmNowGrid = np.insert(aNrmNowGrid, 0, BoroCnstNat[n])
            EndOfPrdvNvrsFunc_cond = LinearInterp(aNrmNowGrid, EndOfPrdvNvrs_cond)
            EndOfPrdvPnvrsFunc_cond = LinearInterp(aNrmNowGrid, EndOfPrdvPnvrs_cond)
            EndOfPrdvFunc_cond = ValueFunc(EndOfPrdvNvrsFunc_cond, CRRA)
            EndOfPrdvPfunc_cond = MargValueFunc(EndOfPrdvPnvrsFunc_cond, CRRA)

            # (j)
            EndOfPrdvFunc_by_state_next.append(EndOfPrdvFunc_cond)
            EndOfPrdvPfunc_by_state_next.append(EndOfPrdvPfunc_cond)

        # 5
        cFunc_by_state = []
        sFunc_by_state = []
        vFunc_by_state = []
        vPfunc_by_state = []

        # 6
        # (a)
        # calculate the minimum m value in this state
        mNrmMinNow = (BoroCnstArt if BoroCnstArt is not None and BoroCnstArt > BoroCnstNat[1] else BoroCnstNat[1])

        # (b)
        aNrmGrid_temp = aXtraGrid + BoroCnstNat[1]
        if BoroCnstArt is not None:
            sortInd = np.searchsorted(aNrmGrid_temp, BoroCnstArt)
            aNrmGrid_temp = np.insert(aNrmGrid_temp, sortInd, BoroCnstArt)

        EndOfPrdv_unemp = EndOfPrdvFunc_by_state_next[1](aNrmGrid_temp)
        EndOfPrdvP_unemp = EndOfPrdvPfunc_by_state_next[1](aNrmGrid_temp)

        # (c)
        EndOfPrdv_emp = EndOfPrdvFunc_by_state_next[0](aNrmGrid_temp)
        EndOfPrdvP_emp = EndOfPrdvPfunc_by_state_next[0](aNrmGrid_temp)

        # (e)
        cNrmGrid_temp = []
        sGrid_temp = []

        for vVal_n, vPval_n, vVal_0, vPval_0 in zip(EndOfPrdv_unemp, EndOfPrdvP_unemp, EndOfPrdv_emp, EndOfPrdvP_emp):
            cVal_FOCs = lambda s_t: ((vVal_0 - vVal_n) / (SearchCost * s_t ** (SearchCost - 1.0) *
                                                          (1.0 - s_t ** SearchCost) ** (-CRRA))) ** (1 / (1.0 - CRRA))
            cVal_FOCc = lambda s_t: ((s_t * vPval_0 + (1.0 - s_t) * vPval_n) /
                                     (1.0 - s_t ** SearchCost) ** (1.0 - CRRA)) ** (-1 / CRRA)
            cDiff_temp = lambda s_t: cVal_FOCc(s_t) - cVal_FOCs(s_t)
            sVal = brentq(cDiff_temp, 1e-6, 1.0 - 1e-6)
            cVal = ((sVal * vPval_0 + (1.0 - sVal[0]) * vPval_n) / (1.0 - sVal[0] ** SearchCost) ** (1.0 - CRRA)) ** \
                   (-1 / CRRA)
            cNrmGrid_temp.append(cVal)
            sGrid_temp.append(sVal)

        cNrmGrid_temp = np.asarray(cNrmGrid_temp)
        sGrid_temp = np.asarray(sGrid_temp)

        # (f)
        mNrmGrid_temp = aNrmGrid_temp + cNrmGrid_temp
        mNrmGrid_temp = np.insert(mNrmGrid_temp, 0, BoroCnstNat)
        cNrmGrid_temp = np.insert(cNrmGrid_temp, 0, 0.0)
        sGrid_temp = np.insert(sGrid_temp, 0, 0.0)

        cFuncUnc_at_this_state = LinearInterp(mNrmGrid_temp, cNrmGrid_temp)
        sFuncUnc_at_this_state = LinearInterp(mNrmGrid_temp, sGrid_temp)

        # (g)
        mCnst = np.array([mNrmMinNow[1], mNrmMinNow[1] + 1.0])
        cCnst = np.array([0.0, 1.0])
        cFuncCnst = LinearInterp(mCnst, cCnst)

        # (h)
        cFuncNow = LowerEnvelope(cFuncUnc_at_this_state, cFuncCnst)
        cFunc_by_state.append(cFuncNow)

        # (i)
        # the following steps are calculating the s-function when liquidity is constrained:
        # 1) Make a grid of m_t values on the open interval bounded below by BoroCnstArt and bounded above by the m_t
        #    value associated with \underline{a}; 10-20 is fine. Call it mNrmGrid_cnst.
        # 2) At each of those m_t values, the agent is liquidity constrained and will end the period with BoroCnstArt
        #    in assets, so calculate c_t = m_t – BoroCnstArt for this grid.
        # 3) Loop through the m_t values and solve (10) for s_t at each one, assuming a_t = \underline{a} and c_t
        #    is the value just calculated. Store these in sGrid_cnst. This is solving the first order condition
        #    for optimal s_t when knowing both c_t and a_t because the agent is liquidity constrained at this m_t.
        # 4) Make mNrmGrid_adj as a truncated version of mNrmGrid_temp, cutting off its lower values that are
        #    associated with a_t < BoroCnstArt. Do the same for sGrid_temp to make sGrid_adj.
        # 5) Concatenate mNrmGrid_cnst and mNrmGrid_adj, and concatenate sGrid_cnst with sGrid_adj.
        #    In this step, sticking together the constrained and unconstrained search functions.
        # 6) Insert a zero at the beginning of both those arrays, then make a LinearInterp; that’s sFunc and should
        #    be appended to sFunc_by_state.

        cEndPoint = cFunc_by_state[1](BoroCnstArt)
        mEndPoint = cEndPoint + BoroCnstArt
        mStartPoint = BoroCnstArt
        mNrmGrid_cnst = np.linspace(mStartPoint, mEndPoint, 20)

        cVal_cnst = mNrmGrid_cnst - BoroCnstArt
        vEmp_cnst = EndOfPrdvFunc_by_state_next[0](BoroCnstArt)
        vUnemp_cnst = EndOfPrdvFunc_by_state_next[1](BoroCnstArt)

        sGrid_cnst = []
        for c, vEmp, vUnemp in zip(cVal_cnst, vEmp_cnst, vUnemp_cnst):
            sFunc = lambda s_t: SearchCost * s_t ** (SearchCost - 1.0) * (1.0 - s_t ** SearchCost) ** (-CRRA) * c ** \
                                (1.0 - CRRA) - (vEmp - vUnemp)
            sGrid_cnst.append(brentq(sFunc, 1e-6, 1.0 - 1e-6))
        sGrid_cnst = np.asarray(sGrid_cnst)

        sortInd = np.searchsorted(aNrmGrid_temp, BoroCnstArt)
        mNrmGrid_adj = mNrmGrid_temp[sortInd:]
        sGrid_adj = sGrid_temp[sortInd:]

        mNrmGrid_cct = np.concatenate((mNrmGrid_cnst, mNrmGrid_adj), axis=None)
        sGrid_cct = np.concatenate((sGrid_cnst, sGrid_adj), axis=None)
        mNrmGrid_cct = np.insert(mNrmGrid_cct, 0, 0.0)
        sGrid_cct = np.insert(sGrid_cct, 0, 0.0)

        # (f)
        sFuncNow = LinearInterp(mNrmGrid_cct, sGrid_cct)
        sFunc_by_state.append(sFuncNow)

        # (k)
        # the following steps are calculating the marginal value function:
        # 1) Make a new grid of m_t values by adding mNrmMinNow to aXtraGrid,
        #    This starts *just above* the minimum permissible value of m_t this period.
        # 2) Evaluate the consumption and search functions at that grid of m_t values.
        # 3) Compute marginal utility of consumption (partial derivative of utility wrt c_t) at these values.
        #    That’s a marginal value grid.
        # 4) Run that through the inverse marginal utility function; that’s a pseudo-inverse marginal value grid.
        # 5) Prepend mNrmMinNow onto the beginning of the m_t grid, and prepend a zero onto
        #    the pseudo-inverse marginal value grid.
        # 6) Make a LinearInterp over m_t and pseudo-inverse marginal value;
        #    that’s the pseudo inverse marginal value function.
        # 7) Make a MargValueFunction using the pseudo-inverse marginal value function and CRRA.
        # 8) Stick that into the list of state-conditional marginal value function.s

        mNrmGrid_new = mNrmMinNow + aXtraGrid
        cValOnThisNewmGrid = cFunc_by_state[1](mNrmGrid_new)
        sValOnThisNewmGrid = sFunc_by_state[1](mNrmGrid_new)
        vPvalOnThisNewmGrid = utilityPc(cValOnThisNewmGrid, sValOnThisNewmGrid)

        vPvalNrvsOnThisNewmGrid = utility_inv(vPvalOnThisNewmGrid)
        mNrmGrid_new = np.insert(mNrmGrid_new, 0, mNrmMinNow)
        vPvalNrvsOnThisNewmGrid = np.insert(vPvalNrvsOnThisNewmGrid, 0, 0.0)
        vPfuncNow = LinearInterp(mNrmGrid_new, vPvalNrvsOnThisNewmGrid)
        vPfunc_by_state.append(MargValueFunc(vPfuncNow, CRRA))

        # (l)
        __mGrid = aXtraGrid + mNrmMinNow[1]
        cGridOnThisGrid = cFunc_by_state[1](__mGrid)
        sGridOnThisGrid = sFunc_by_state[1](__mGrid)
        aGridOnThisGrid = __mGrid - cGridOnThisGrid
        EndOfPrdvFuncUnempOnThisGrid = EndOfPrdvFunc_by_state_next[1](aGridOnThisGrid)
        EndOfPrdvFuncEmpOnThisGrid = EndOfPrdvFunc_by_state_next[0](aGridOnThisGrid)

        vFuncOnThisGrid = utility(cGridOnThisGrid, sGridOnThisGrid) + \
                          sGridOnThisGrid * EndOfPrdvFuncEmpOnThisGrid + \
                          (1.0 - sGridOnThisGrid) * EndOfPrdvFuncUnempOnThisGrid

        # (m)
        vFuncNvrsOnThisGrid = utility_inv(vFuncOnThisGrid)
        vFuncNvrsOnThisGrid = np.insert(vFuncNvrsOnThisGrid, 0, 0.0)
        __mGrid = np.insert(__mGrid, 0, mNrmMinNow[1])
        vFuncNrvs = LinearInterp(__mGrid, vFuncNvrsOnThisGrid)

        # (n)
        vFuncNow = ValueFunc(vFuncNrvs, CRRA)
        vFunc_by_state.append(vFuncNow)

        # 7
        # agent is employed
        mNrmMinNowEmp = (BoroCnstArt if BoroCnstArt is not None and BoroCnstArt > BoroCnstNat[0] else BoroCnstNat[0])

        __aNrmGrid = aXtraGrid + BoroCnstNat[0]
        EndOfPrdvPEmp = EndOfPrdvFunc_by_state_next[0](__aNrmGrid)
        EndOfPrdvPUnemp = EndOfPrdvPfunc_by_state_next[1](__aNrmGrid)

        __cNrmGrid = ((1.0 - SepaRte) * EndOfPrdvPEmp + SepaRte * EndOfPrdvPUnemp) ** (-1.0 / CRRA)
        mNrmGridEmp = __aNrmGrid + __cNrmGrid
        cFuncEmpUncnst = LinearInterp(mNrmGridEmp, __cNrmGrid)
        cFuncEmpCnst = LinearInterp(np.array([mNrmMinNowEmp, mNrmMinNowEmp + 1.0]), np.array([0.0, 1.0]))
        __cFunc = LowerEnvelope(cFuncEmpUncnst, cFuncEmpCnst)
        cFunc_by_state = np.insert(cFunc_by_state, 0, __cFunc)

        sGridEmp = np.zeros(mNrmGridEmp.size)
        __sFunc = LinearInterp(mNrmGridEmp, sGridEmp)
        sFunc_by_state = np.insert(sFunc_by_state, 0, __sFunc)

        __vPfunc = MargValueFunc(cFunc_by_state[0], CRRA)
        vPfunc_by_state = np.insert(vPfunc_by_state, 0, __vPfunc)

        __mGridEmp = aXtraGrid + mNrmMinNowEmp
        __cValEmpOnThisGrid = cFunc_by_state[0](__mGridEmp)
        __aValEmpOnThisGrid = __mGridEmp - __cValEmpOnThisGrid
        __EndOfPrdvFuncUnempOnThisGrid = EndOfPrdvFunc_by_state_next[1](__aValEmpOnThisGrid)
        __EndOfPrdvFuncEmpOnThisGrid = EndOfPrdvFunc_by_state_next[0](__aValEmpOnThisGrid)
        __vFuncOnThisGrid = utilitySepa(__cValEmpOnThisGrid) + (1.0 - SepaRte) * __EndOfPrdvFuncEmpOnThisGrid + \
                            SepaRte * __EndOfPrdvFuncUnempOnThisGrid
        __vFuncNrvsOnThisGrid = utility_inv(__vFuncOnThisGrid)
        __vFuncNrvsOnThisGrid = np.insert(__vFuncNrvsOnThisGrid, 0, 0.0)
        __mGridEmp = np.insert(__mGridEmp, 0, mNrmMinNowEmp)
        __vFuncNrvs = LinearInterp(__mGridEmp, __vFuncNrvsOnThisGrid)
        __vFuncEmp = ValueFunc(__vFuncNrvs, CRRA)
        vFunc_by_state = np.insert(vFunc_by_state, 0, __vFuncEmp)

        mNrmMin_list = np.array([mNrmMinNowEmp, mNrmMinNow])

        # 8
        # construct an object called solution now
        solution_now = ConsumerSolution()
        solution_now.cFunc = cFunc_by_state
        solution_now.sFunc = sFunc_by_state
        solution_now.vFunc = vFunc_by_state
        solution_now.vPfunc = vPfunc_by_state
        solution_now.mNrmMin = mNrmMin_list

        return solution_now
