import numpy as np
from scipy.optimize import brentq, fsolve
from HARK.ConsumptionSaving.ConsIndShockModel import ValueFunc, MargValueFunc
from copy import deepcopy
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks, ConsumerSolution
from HARK.interpolation import LinearInterp, LowerEnvelope
from HARK.core import AgentType
from HARK.utilities import *
from HARK.distribution import *

params = init_idiosyncratic_shocks


class EducChoiceConsumerType(MarkovConsumerType):
    '''
    A consumer type with idiosyncratic discount factor, age-path of education effort cost, direct monetary cost
    of education, with non-separable utility function (derive utility from consumption and experience disutility
    from periods spent receiving education). In each discrete time period, an agent makes a continuous decision over
    how much of their market resources to consume and how much to retain in a risk free asset; If they received
    education in the prior period, they also choose how much costly effort to exert to remain in school.
    '''

    def __init__(self, cycles=1, **kwds):
        '''new consumer type'''

        params.update(kwds)
        MarkovConsumerType.__init__(cycles=cycles, **params)
        self.solveOnePeriod = self.solveConsEducChoice

    def checkMarkovInputs(self):
        '''if parent class not true, overwrite'''

        pass

    def preSolve(self):
        '''check if consumer type is the right shape'''

        AgentType.preSolve(self)
        self.update()
        if self.cycles != 1:
            raise ValueError

    def update(self):
        MarkovConsumerType.update(self)
        self.updateEducInputs()

    def updateEducInputs(self):
        '''time varying attribute EducInputs (EducMax+1)'''
        Coeffs = self.EducCostCoeffs
        N = len(Coeffs)
        age_vec = np.arange(self.T_cycle)
        EducCostBase = np.zero(self.T_cycle)
        for n in range(N):
            EducCostBase += Coeffs[n] * age_vec ** n
        EffortParam = 1 + np.exp(EducCostBase)
        EffortParam_temp = EffortParam.tolist()
        for i in range(0, self.T_cycle - self.EducMax):
            EffortParam_temp[self.EducMax + i] = 0
        self.EffortParam = EffortParam_temp

        self.EducIdx = []
        for i in range(1, self.EducMax + 1):
            self.EducIdx.append(i)
        for i in range(1, self.T_cycle - self.EducMax + 1):
            self.EducIdx.append(0)

        self.StateCount = self.EducMax + 1
        self.addToTimeVary('EffortParam', 'EducIdx')

    def updateIncomeProcess(self):
        '''update IncomeProcess'''
        PermShkStd_all = deepcopy(self.PermShkStd)
        TranShkStd_all = deepcopy(self.TranShkStd)

        IncomeDstn_by_educ = []
        TranShkDstn_by_educ = []
        PermShkDstn_by_educ = []

        for n in range(0, self.EducMax):
            self.PermShkStd = PermShkStd_all[n, :]
            self.TranShkStd = TranShkStd_all[n, :]
            IncomeDstn, PermShkDstn, TranShkDstn = self.constructLognormalIncomeProcessUnemployment()
            IncomeDstn_by_educ.append(IncomeDstn)
            TranShkDstn_by_educ.append(PermShkDstn)
            PermShkDstn_by_educ.append(TranShkDstn)

        self.IncomeDstn = []
        self.TranShkDstn = []
        self.PermShkDstn = []

        for j in range(0, self.T_cycle):
            self.IncomeDstn.append([])
            self.TranShkDstn.append([])
            self.PermShkDstn.append([])
            for n in range(0, self.EducMax):
                self.IncomeDstn[j].append(IncomeDstn_by_educ[n][j])
                self.TranShkDstn[j].append(TranShkDstn_by_educ[n][j])
                self.PermShkDstn[j].append(PermShkDstn_by_educ[n][j])

        self.addToTimeVary('IncomeDstn', 'TranShkDstn', 'PermShkDstn')

    def solveConsEducChoice(self, solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rboro, Rsave, PermGroFac,
                            BoroCnstArt, EffortParam, EducCostNow, EducCostNext, EducIdx, aXtraGrid):
        '''
        one period solver function (B9)

        parameters:
        • solution next, the representation of next period’s solution
        • IncomeDstn, a list of DiscreteDistributions for each education level for this age.
        • LivPrb, an array of survival probabilities at this age for each education level.
        • DiscFac, the intertemporal discount factor.
        • CRRA, the coefficient of relative risk aversion.
        • Rboro, an array of interest factors on borrowing for each education level. In most calibrations, these won’t actually differ by education, but it’s trivial to allow it.
        • Rsave, an array of interest factors on saving for each education level. 4
        • PermGroFac, an array of permanent income growth factors between this period and the next one.
        • BoroCnstArt, a float or None representing the (normalized) artifical borrowing constraint.
        • EducCost, a float representing the cost of education parameter at this age.
        • EducIdx, an int indicating which education index would be achieved if the agent did not stay in school next period; it is zero if it is impossible to remain in school after this period.
        • aXtraGrid, an array of numbers representing “normalized assets above minimum” gridpoints at which the model will be solved by the endogenous grid method.

        return: ConsumerSolution
        '''

        # 1
        u = lambda c, s: CRRAutility(c * (1 - s ** EffortParam), CRRA)
        u_prime = lambda c: CRRAutilityP(c, CRRA)
        u_prime_c = lambda c, s: u_prime(c) * (1 - s ** EffortParam) ** (1 - CRRA)
        u_prime_inv = lambda uP: CRRAutilityP_inv(uP, CRRA)
        u_inv = lambda u: CRRAutility_inv(u, CRRA)

        # 2
        EndOfPrdvFunc_by_educ_next = []
        EndOfPrdvPfunc_by_educ_next = []

        # 3
        BoroCnstNat = np.zeros(self.StateCount) + np.nan

        # 4
        for n in range(self.StateCount):
            # 4a
            prob = IncomeDstn[n].pmf
            permX = IncomeDstn[n].X[0]
            tranX = IncomeDstn[n].X[1]

            # 4b
            gridNmbs = aXtraGrid.size
            tiled_prob = np.tile(prob, (gridNmbs, 1))
            tiled_permX = np.tile(permX, (gridNmbs, 1))
            tiled_tranX = np.tile(tranX, (gridNmbs, 1))

            # 4c
            perm_min = np.min(permX)
            tran_min = np.min(tranX)
            if n != 0:
                BoroCnstNat[n] = (solution_next.mNrmMin[n] - tran_min) * (PermGroFac[n] * perm_min) / Rboro
            else:
                BoroCnstNat[n] = (solution_next.mNrmMin[n] + EducCostNext - tran_min) * \
                                 (PermGroFac[n] * perm_min) / Rboro

            # 4d
            aNrmNowGrid = np.array(aXtraGrid) + BoroCnstNat[n]
            shockNmbs = prob.size
            tiled_aNrmNowGrid = np.tile(aNrmNowGrid, (shockNmbs, 1))
            tiled_aNrmNowGrid = tiled_aNrmNowGrid.transpose()

            # 4e
            if n != 0:
                mNrmNext = Rboro / (PermGroFac[n] * tiled_tranX) * tiled_aNrmNowGrid + tiled_permX
            else:
                mNrmNext = Rboro / (PermGroFac[n] * tiled_tranX) * tiled_aNrmNowGrid + tiled_permX - EducCostNext

            # 4f
            next_vf = solution_next.vFunc[n](mNrmNext)
            next_vp = solution_next.vPfunc[n](mNrmNext)

            # 4g
            EndOfPrdv = DiscFac * LivPrb * PermGroFac[n] ** (1 - CRRA) * \
                        np.sum(tiled_permX ** (1 - CRRA) * next_vf * tiled_prob, axis=1)
            EndOfPrdvP = DiscFac * LivPrb * PermGroFac[n] ** (-CRRA) * Rboro * \
                         np.sum(tiled_permX ** (-CRRA) * next_vp * tiled_prob, axis=1)

            # 4h
            EndOfPrdvNvrs_cond = u_inv(EndOfPrdv)
            EndOfPrdvNvrs_cond = np.insert(EndOfPrdvNvrs_cond, 0, 0.0)
            EndOfPrdvPnvrs_cond = u_prime_inv(EndOfPrdvP)
            EndOfPrdvPnvrs_cond = np.insert(EndOfPrdvPnvrs_cond, 0, 0.0)

            # 4i
            aNrmNowGrid_prepend = np.insert(aNrmNowGrid, 0, BoroCnstNat[n])
            EndOfPrdvNvrsFunc_cond = LinearInterp(aNrmNowGrid_prepend, EndOfPrdvNvrs_cond)
            EndOfPrdvPfuncFunc_cond = LinearInterp(aNrmNowGrid_prepend, EndOfPrdvPnvrs_cond)
            EndOfPrdvFunc_cond = ValueFunc(EndOfPrdvNvrsFunc_cond, CRRA)
            EndOfPrdvPfunc_cond = MargValueFunc(EndOfPrdvPfuncFunc_cond, CRRA)

            # 4j
            EndOfPrdvFunc_by_educ_next.append(EndOfPrdvFunc_cond)
            EndOfPrdvPfunc_by_educ_next.append(EndOfPrdvPfunc_cond)

        # 5.
        mNrmMinNow = np.zeros(self.StateCount)

        # 6.
        cFunc_by_educ = []
        vFunc_by_educ = []
        vPfunc_by_educ = []

        # 7.
        for n in range(1, self.StateCount):
            # 7a
            aNrmGrid_temp = aXtraGrid + BoroCnstNat[n]
            EndOfPrdvFunc = EndOfPrdvFunc_by_educ_next[n](aNrmGrid_temp)
            EndOfPrdvPfunc = EndOfPrdvPfunc_by_educ_next[n](aNrmGrid_temp)

            # 7b
            cNrmGrid_temp = []
            for v_prime in range(EndOfPrdvPfunc):
                c = (LivPrb[n] * v_prime) ** (-1 / CRRA)
                cNrmGrid_temp.append(c)
            cNrmGrid_temp = np.array(cNrmGrid_temp)

            mNrmGrid_temp = aNrmGrid_temp + cNrmGrid_temp

            # 7c
            cNrmGrid_temp = np.insert(cNrmGrid_temp, 0, 0.0)
            mNrmGrid_temp = np.insert(mNrmGrid_temp, 0, BoroCnstNat[n])
            cFuncUnc_at_this_educ = LinearInterp(mNrmGrid_temp, cNrmGrid_temp)

            # 7d
            if BoroCnstArt is not None:
                if BoroCnstArt > BoroCnstNat[n]:
                    mNrmMinNow[n] = BoroCnstArt
                else:
                    mNrmMinNow[n] = BoroCnstNat
            else:
                mNrmMinNow[n] = BoroCnstNat

            # 7e
            cFuncCnst = LinearInterp(np.array([mNrmMinNow[n], mNrmMinNow[n] + 1]), np.array([0.0, 1.0]))

            # 7f
            cFunc = LowerEnvelope(cFuncCnst, cFuncUnc_at_this_educ)
            cFunc_by_educ.append(cFunc)

            # 7g
            vPFunc = MargValueFunc(cFunc_by_educ[n], CRRA)
            vPfunc_by_educ.append(vPFunc)

            # 7h marginal value function
            mTempGrid = aXtraGrid + mNrmMinNow[n]
            cOnThisGrid = cFunc_by_educ[n](mTempGrid)
            aOnThisGrid = mTempGrid - cOnThisGrid
            pseudovFuncOnThisGrid = EndOfPrdvFunc_by_educ_next[n](aOnThisGrid)
            vFuncOnThisGrid = cOnThisGrid ** (1 - CRRA) / (1 - CRRA) + LivPrb[n] * pseudovFuncOnThisGrid

            # 7i
            pseudoNvrsvFunc = u_inv(vFuncOnThisGrid)
            pseudoNvrsvFunc = np.insert(pseudoNvrsvFunc, 0, 0.0)
            mTempGrid = np.insert(mTempGrid, 0, mNrmMinNow[n])
            NvrsvFuncOnThisGrid = LinearInterp(mTempGrid, pseudoNvrsvFunc)
            vFunc = ValueFunc(NvrsvFuncOnThisGrid, CRRA)
            vFunc_by_educ.append(vFunc)

        # 8 EducIdx:number of periods of education an agent would have if they exited the education process after this period
        if EducIdx == 0:
            aNrmGrid_case1 = aXtraGrid + BoroCnstNat[0]
            cNrmGrid_case1 = np.linspace(0.0, 1.0, aXtraGrid.size)  # cFunc is a identity function
            mNrmGrid_case1 = aNrmGrid_case1 + cNrmGrid_case1
            cFunc_case1 = LinearInterp(mNrmGrid_case1, mNrmGrid_case1)
            cFunc_by_educ = np.insert(cFunc_by_educ, 0, cFunc_case1)
            vFunc_case1 = LinearInterp(mNrmGrid_case1, np.zeros(mNrmGrid_case1.size))
            vFunc_by_educ = np.insert(vFunc_by_educ, 0, vFunc_case1)
            vPFunc_case1 = MargValueFunc(cFunc_by_educ[0], CRRA)  # search function is zero function, so same case
            vPfunc_by_educ = np.isert(vPfunc_by_educ, 0, vPFunc_case1)
        elif EffortParam == 0:
            cFunc_by_educ = np.insert(cFunc_by_educ, 0, cFunc_by_educ[EducIdx + 1])
            vFunc_by_educ = np.insert(vFunc_by_educ, 0, vFunc_by_educ[EducIdx + 1])
            vPfunc_by_educ = np.insert(vPfunc_by_educ, 0, vPfunc_by_educ[EducIdx + 1])
        else:
            # 9a
            func_temp = lambda a_t: EndOfPrdvFunc_by_educ_next[0](a_t) - EndOfPrdvFunc_by_educ_next[EducIdx](a_t)
            aCrtlVal = fsolve(func_temp, BoroCnstArt)

            # 9b
            aTempGrid = aXtraGrid + BoroCnstNat[0]
            aCutOffVal = np.searchsorted(aXtraGrid, aCrtlVal)
            cNrm_above_crit = []
            sNrm_above_crit = []

            for a in range(aTempGrid[aCutOffVal:]):
                v_0 = EndOfPrdvFunc_by_educ_next[0](a)
                vP_0 = EndOfPrdvPfunc_by_educ_next[0](a)
                v_n = EndOfPrdvFunc_by_educ_next[EducIdx](a)
                vP_n = EndOfPrdvPfunc_by_educ_next[EducIdx](a)
                FOC_c = lambda s_t: (LivPrb * (s_t * vP_0 + (1 - s_t) * vP_n) / (1 - s_t ** EffortParam) **
                                     (1 - CRRA)) ** (-1 / CRRA)
                FOC_s = lambda s_t: (LivPrb * (v_0 - v_n) / EffortParam * s_t ** (EffortParam - 1) *
                                     (1 - s_t ** EffortParam) ** (-CRRA)) ** (1 / (1 - CRRA))
                DSPF = lambda s_t: FOC_c(s_t) - FOC_s(s_t)
                s = brentq(DSPF, 1e-6, 1 - 1e-6)
                c = (LivPrb * (s * vP_0 + (1 - s[0]) * vP_n) / (1 - s[0] ** EffortParam) ** (1 - CRRA)) ** (-1 / CRRA)
                cNrm_above_crit.append(c)
                sNrm_above_crit.append(s)

            cNrm_above_crit = np.array(cNrm_above_crit)
            sNrm_above_crit = np.array(sNrm_above_crit)
            cNrm_below_crit = (EndOfPrdvPfunc_by_educ_next[EducIdx](aTempGrid[0:aCutOffVal])) ** (-1.0 / CRRA)
            cNrm = np.concatenate((cNrm_below_crit, cNrm_above_crit), axis=None)
            sNrm_below_crit = np.zeros(aCutOffVal)
            sNrm = np.concatenate((sNrm_below_crit, sNrm_above_crit), axis=None)

            # 9c
            mTempGridNtrvl = aTempGrid + cNrm + EducCostNow

            # 9d
            cFuncNtrvl_ = LinearInterp(mTempGridNtrvl, cNrm)
            sFuncNtrvl = LinearInterp(mTempGridNtrvl, sNrm)

            # 9e
            cFuncCnst = LinearInterp(np.array([mNrmMinNow[0], mNrmMinNow[0] + 1]), np.array([0.0, 1.0]))
            cFuncNtrvl = LowerEnvelope(cFuncNtrvl_, cFuncCnst)

            cFunc_by_educ = np.insert(cFunc_by_educ, 0, cFuncNtrvl)

            # 9f
            vFuncNtrvl = u(cFuncNtrvl(mTempGridNtrvl), sFuncNtrvl(mTempGridNtrvl)) + LivPrb * \
                         (sFuncNtrvl(mTempGridNtrvl) * EndOfPrdvFunc_by_educ_next[0](aTempGrid) +
                          (1 - sFuncNtrvl(aTempGrid)) ** EndOfPrdvFunc_by_educ_next[EducIdx](aTempGrid))
            vFuncNrvsNtrvl = u_inv(vFuncNtrvl)
            temp = LinearInterp(mTempGridNtrvl, vFuncNrvsNtrvl)
            vFuncNtrvl = ValueFunc(temp, CRRA)
            vFunc_by_educ = np.insert(vFunc_by_educ, 0, vFuncNtrvl)

            # 9g
            mgrid = mNrmMinNow + aXtraGrid
            cfuncval = cFunc_by_educ[0](mgrid)
            sfuncval = sFuncNtrvl(mgrid)
            upfuncval = u_prime_c(cfuncval, sfuncval)
            upnvrsval = u_inv(upfuncval)
            mgrid = np.insert(mgrid, 0, mNrmMinNow)
            upnvrsval = np.insert(upnvrsval, 0, 0.0)
            temp = LinearInterp(mgrid, upnvrsval)
            vPfuncNtrvl = MargValueFunc(temp, CRRA)
            vPfunc_by_educ = np.insert(vPfunc_by_educ, 0, vPfuncNtrvl)

            solution_now = ConsumerSolution()
            solution_now.cFunc = cFunc_by_educ
            solution_now.vFunc = vFunc_by_educ
            solution_now.vPfunc = vPfunc_by_educ
            solution_now.mNrmMin = mNrmMinNow

            return solution_now
