# -*- coding: utf-8 -*-
from HARK.core import (_log, core_check_condition)
from HARK.utilities import CRRAutility as utility
from HARK.utilities import CRRAutilityP as utilityP
from HARK.utilities import CRRAutilityPP as utilityPP
from HARK.utilities import CRRAutilityP_inv as utilityP_inv
from HARK.utilities import CRRAutility_invP as utility_invP
from HARK.utilities import CRRAutility_inv as utility_inv
from HARK.utilities import CRRAutilityP_invP as utilityP_invP

from HARK.interpolation import (CubicInterp, LowerEnvelope, LinearInterp,
                                ValueFuncCRRA, MargValueFuncCRRA,
                                MargMargValueFuncCRRA)
from HARK import NullFunc, MetricObject
from scipy.optimize import newton as find_zero_newton
from numpy import dot as E_dot  # easier to type
from numpy.testing import assert_approx_equal as assert_approx_equal
import numpy as np
from copy import deepcopy
from builtins import (str, breakpoint)
from types import SimpleNamespace
from IPython.lib.pretty import pprint
from HARK.ConsumptionSaving.ConsIndShockModel_Both import (
    def_utility, def_value_funcs)
from HARK.distribution import (calc_expectation)

__all__ = [
    "ConsumerSolutionOneStateCRRA",
    "ConsPerfForesightSolverEOP",
    "ConsIndShockSetupEOP",
    "ConsIndShockSolverBasicEOP",
    "ConsIndShockSolverEOP",
    #    "ConsKinkedRsolverEOP",
]


class ConsIndShockSolverBasicEOP(ConsIndShockSetupEOP):
    """
    This class solves a single period of a standard consumption-saving problem,
    using linear interpolation and without the ability to calculate the value
    function.  ConsIndShockSolver inherits from this class and adds the ability
    to perform cubic interpolation and to calculate the value function.

    Note that this class does not have its own initializing method.  It initial-
    izes the same problem in the same way as ConsIndShockSetup, from which it
    inherits.
    """

    def prepare_to_calc_EndOfPrdvP(self):
        """
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, given the
        current grid of end-of-period assets and the distribution of shocks
        they might experience next period.

        Parameters
        ----------
        none

        Returns
        -------
        aNrmGrid : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.soln_crnt.bilt.
        """

        # We define aNrmGrid all the way from BoroCnstNat up to max(self.aXtraGrid)
        # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
        # function as the lower envelope of the (by the artificial borrowing con-
        # straint) unconstrained consumption function, and the artificially con-
        # strained consumption function.
        self.soln_crnt.bilt.aNrmGrid = np.asarray(
            self.soln_crnt.bilt.aXtraGrid) + self.soln_crnt.bilt.BoroCnstNat

        return self.soln_crnt.bilt.aNrmGrid

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets at each point in aNrm.
        Does so by taking a weighted sum of next period marginal values across
        income shocks (in a preconstructed grid self.soln_crnt.bilt.mNrmNext).

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets
        """

        bilt = self.soln_crnt.bilt
        folw = self.soln_crnt.folw

        def vP_tp1(shk_vector, a_number):
            return shk_vector[0] ** (-bilt.CRRA) \
                * folw.vPfunc_tp1(self.m_Nrm_tp1(shk_vector, a_number))

        EndOfPrdvP = (
            bilt.DiscFac * bilt.LivPrb
            * bilt.Rfree
            * bilt.PermGroFac ** (-bilt.CRRA)
            * calc_expectation(
                bilt.IncShkDstn,
                vP_tp1,
                bilt.aNrmGrid
            )
        )
        return EndOfPrdvP

    def get_source_points_via_EGM(self, EndOfPrdvP, aNrm):
        """
        Finds interpolation points (c,m) for the consumption function.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrm : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.

        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation.
        m_for_interpolation : np.array
            Corresponding market resource points for interpolation.
        """
#        breakpoint()
#        CRRA = self.soln_crnt.bilt.CRRA
#        self.soln_crnt.bilt = def_utility(self.soln_crnt.bilt, CRRA)
        cNrm = self.soln_crnt.bilt.uPinv(EndOfPrdvP)
        mNrm = cNrm + aNrm

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.insert(cNrm, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(mNrm, 0, self.soln_crnt.bilt.BoroCnstNat, axis=-1)

        # Store these for calcvFunc
        self.soln_crnt.bilt.cNrm = cNrm
        self.soln_crnt.bilt.mNrm = mNrm

        return c_for_interpolation, m_for_interpolation

    # def use_points_for_interpolation(self, cNrm, mNrm, interpolator):
    #     """
    #     Constructs a basic solution for this period, including the consumption
    #     function and marginal value function.

    #     Parameters
    #     ----------
    #     cNrm : np.array
    #         (Normalized) consumption points for interpolation.
    #     mNrm : np.array
    #         (Normalized) corresponding market resource points for interpolation.
    #     interpolator : function
    #         A function that constructs and returns a consumption function.

    #     Returns
    #     -------
    #     solution_interpolating : ConsumerSolution
    #         The solution to this period's consumption-saving problem, with a
    #         minimum m, a consumption function, and marginal value function.
    #     """
    #     bilt = self.soln_crnt.bilt
    #     # Use the given interpolator to construct the consumption function
    #     cFuncUnc = interpolator(mNrm, cNrm)  # Unc=Unconstrained

    #     # Combine the constrained and unconstrained functions into the true consumption function
    #     # by choosing the lower of the constrained and unconstrained functions
    #     # LowerEnvelope should only be used when BoroCnstArt is true
    #     if bilt.BoroCnstArt is None:
    #         cFunc = cFuncUnc
    #     else:
    #         bilt.cFuncCnst = LinearInterp(
    #             np.array([bilt.mNrmMin, bilt.mNrmMin + 1]
    #                      ), np.array([0.0, 1.0]))
    #         cFunc = LowerEnvelope(cFuncUnc, bilt.cFuncCnst, nan_bool=False)

    #     # Make the marginal value function and the marginal marginal value function
    #     vPfunc = MargValueFuncCRRA(cFunc, bilt.CRRA)

    #     # Pack up the solution and return it
    #     solution_interpolating = ConsumerSolutionOneStateCRRA(
    #         cFunc=cFunc,
    #         vPfunc=vPfunc,
    #         mNrmMin=bilt.mNrmMin
    #     )

    #     return solution_interpolating

    # def interpolating_EGM_solution(self, EndOfPrdvP, aNrmGrid, interpolator):
    #     """
    #     Given end of period assets and end of period marginal value, construct
    #     the basic solution for this period.

    #     Parameters
    #     ----------
    #     EndOfPrdvP : np.array
    #         Array of end-of-period marginal values.
    #     aNrmGrid : np.array
    #         Array of end-of-period asset values that yield the marginal values
    #         in EndOfPrdvP.

    #     interpolator : function
    #         A function that constructs and returns a consumption function.

    #     Returns
    #     -------
    #     sol_EGM : ConsumerSolution
    #         The EGM solution to this period's consumption-saving problem, with a
    #         consumption function, marginal value function, and minimum m.
    #     """
    #     cNrm, mNrm = self.get_source_points_via_EGM(EndOfPrdvP, aNrmGrid)
    #     sol_EGM = self.use_points_for_interpolation(cNrm, mNrm, interpolator)

    #     return sol_EGM

    # def make_sol_using_EGM(self):  # Endogenous Gridpts Method
    #     """
    #     Given a grid of end-of-period values of assets a, use the endogenous
    #     gridpoints method to obtain the corresponding values of consumption,
    #     and use the dynamic budget constraint to obtain the corresponding value
    #     of market resources m.

    #     Parameters
    #     ----------
    #     none (relies upon self.soln_crnt.aNrm existing before invocation)

    #     Returns
    #     -------
    #     solution : ConsumerSolution
    #         The solution to the single period consumption-saving problem.
    #     """
    #     bilt = self.soln_crnt.bilt
    #     bilt.aNrmGrid = self.prepare_to_calc_EndOfPrdvP()
    #     bilt.EndOfPrdvP = self.calc_EndOfPrdvP()

    #     # Construct a solution for this period
    #     if bilt.CubicBool:
    #         soln_crnt = self.interpolating_EGM_solution(
    #             bilt.EndOfPrdvP, bilt.aNrmGrid,
    #             interpolator=self.make_cubic_cFunc
    #         )
    #     else:
    #         soln_crnt = self.interpolating_EGM_solution(
    #             bilt.EndOfPrdvP, bilt.aNrmGrid,
    #             interpolator=self.make_linear_cFunc
    #         )
    #     return soln_crnt

    def make_linear_cFunc(self, mNrm, cNrm):
        """
        Makes a linear interpolation to represent the (unconstrained) consumption function.

        Parameters
        ----------
        mNrm : np.array
            Corresponding market resource points for interpolation.
        cNrm : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFunc_unconstrained : LinearInterp
            The unconstrained consumption function for this period.
        """
        cFunc_unconstrained = LinearInterp(
            mNrm, cNrm, self.soln_crnt.bilt.cFuncLimitIntercept, self.soln_crnt.bilt.cFuncLimitSlope
        )
        return cFunc_unconstrained

#    def solve_prepared_stage(self):  # solves ONE stage of ConsIndShockSolverBasic
    def solve_prepared_stageEOP(self):  # solves ONE stage of ConsIndShockSolverBasic
        """
        Solves one stage (period, in this model) of the consumption-saving problem.  

        Solution includes a decision rule (consumption function), cFunc,
        value and marginal value functions vFunc and vPfunc, 
        a minimum possible level of normalized market resources mNrmMin, 
        normalized human wealth hNrm, and bounding MPCs MPCmin and MPCmax.  

        If the user chooses sets `CubicBool` to True, cFunc
        have a value function vFunc and marginal marginal value function vPPfunc.

        Parameters
        ----------
        none (all should be on self)

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period/stage's problem.
        """
        soln_futr_bilt = self.soln_futr.bilt
        soln_crnt_bilt = self.soln_crnt.bilt
#        soln_futr = self.soln_futr
        soln_crnt = self.soln_crnt
        CRRA = soln_crnt.bilt.CRRA

        if soln_futr_bilt.stge_kind['iter_status'] == 'finished':
            breakpoint()
            # Should not have gotten here
            # core.py tests whehter solution_last is 'finished'

        # If this is the first invocation of solve, just flesh out the
        # terminal_pseudo solution so it is a proper starting point for iteration
        # given the further info that has been added since generic
        # solution_terminal was constructed.  This involves copying its
        # contents into the bilt attribute, then invoking the
        # build_infhor_facts_from_params() method
#        breakpoint()
        if soln_futr_bilt.stge_kind['iter_status'] == 'terminal_pseudo':
            # generic AgentType solution_terminal does not have utility or value
            #            breakpoint()

            soln_futr = soln_crnt = def_utility(soln_crnt, CRRA)
#            print('Test whether value funcs are already defined; they are in PF case ...')
#            breakpoint()
#            soln_futr_bilt = soln_crnt_bilt = def_value_funcs(soln_crnt_bilt, CRRA)
            self.build_infhor_facts_from_paramsEOP()
            # Now it is good to go as a starting point for backward induction:
            soln_crnt_bilt.stge_kind['iter_status'] = 'iterator'
#            breakpoint()
            self.soln_crnt.vPfunc = self.soln_crnt.bilt.vPfunc  # Need for distance
            self.soln_crnt.cFunc = self.soln_crnt.bilt.cFunc  # Need for distance
            self.soln_crnt.IncShkDstn = self.soln_crnt.bilt.IncShkDstn
            return self.soln_crnt  # Replaces original "terminal" solution; next soln_futr

        # Add a bunch of useful stuff
        # CDC 20200428: This stuff is "useful" only for a candidate converged solution
        # in an infinite horizon model.  It's not costly to compute but there's not
        # much point in computing most of it for a non-final infhor stage or a finhor model
        # TODO: Distinguish between those things that need to be computed for the
        # "useful" computations in the final stage, and those that are merely info,
        # and make mandatory only the computations of the former category
        self.build_infhor_facts_from_params()
#        if self.soln_futr.bilt.completed_cycles == 1:
#            print('about to call recursive on soln_futr.completed_cycles==1')
#            breakpoint()
        self.build_recursive_facts()

        # Current utility functions colud be different from future
        soln_crnt_bilt = def_utility(soln_crnt, CRRA)
#        breakpoint()
        sol_EGM = self.make_sol_using_EGM()  # Need to add test for finished, change stge_kind if so

#        breakpoint()
        soln_crnt.bilt.cFunc = soln_crnt.cFunc = sol_EGM.bilt.cFunc
        # soln_crnt.bilt.vPfunc = soln_crnt.vPfunc = sol_EGM.vPfunc
        # # Adding vPPfunc does no harm if non-cubic solution is being used
        # soln_crnt.bilt.vPPfunc = MargMargValueFuncCRRA(soln_crnt.bilt.cFunc, soln_crnt.bilt.CRRA)
        # Can't build current value function until current consumption function exists
#        CRRA = soln_crnt.bilt.CRRA
#        soln_crnt.bilt = def_value_funcs(soln_crnt.bilt, CRRA)

        soln_crnt = def_value_funcs(soln_crnt, CRRA)
        soln_crnt.vPfunc = soln_crnt.bilt.vPfunc
        soln_crnt.cFunc = soln_crnt.bilt.cFunc
        if not hasattr(soln_crnt.bilt, 'IncShkDstn'):
            print('not hasattr(soln_crnt.bilt, "IncShkDstn")')
            breakpoint()

        soln_crnt.IncShkDstn = soln_crnt.bilt.IncShkDstn
        # Add the value function if requested, as well as the marginal marginal
        # value function if cubic splines were used for interpolation
        # CDC 20210428: We should just always make the value function.  The cost
        # is trivial and making it optional is not worth the maintainence and
        # mindspace time the option takes in the codebase
        # if soln_crnt.bilt.vFuncBool:
        #     soln_crnt.bilt.vFunc = self.vFunc = self.add_vFunc(soln_crnt, self.EndOfPrdvP)
        # if soln_crnt.bilt.CubicBool:
        #     soln_crnt.bilt.vPPfunc = self.add_vPPfunc(soln_crnt)

        # EndOfPrdvP=self.soln_crnt.bilt.EndOfPrdvP
        # aNrmGrid=self.soln_crnt.bilt.aNrmGrid

        # solnow=self.mbs(EndOfPrdvP, aNrmGrid, self.make_cubic_cFunc)

        return soln_crnt

    solve = solve_prepared_stageEOP

    def m_Nrm_tp1(self, shk_vector, a_number):
        """
        Computes normalized market resources of the next period
        from income shocks and current normalized market resources.

        Parameters
        ----------
        shk_vector: [float]
            Permanent and transitory income shock levels.

        a_number: float
            Normalized market assets this period

        Returns
        -------
        float
           normalized market resources in the next period
        """
        return self.soln_crnt.bilt.Rfree / (self.soln_crnt.bilt.PermGroFac * shk_vector[0]) \
            * a_number + shk_vector[1]


###############################################################################
# ##############################################################################


class ConsIndShockSolverEOP(ConsIndShockSolverBasicEOP):
    """
    This class solves a single period of a standard consumption-saving problem.
    It inherits from ConsIndShockSolverBasic, and adds the ability to perform cubic
    interpolation and to calculate the value function.
    """

#     def make_cubic_cFunc(self, mNrm_Vec, cNrm_Vec):
#         """
#         Makes a cubic spline interpolation of the unconstrained consumption
#         function for this period.

#         Requires self.soln_crnt.bilt.aNrm to have been computed already.

#         Parameters
#         ----------
#         mNrm_Vec : np.array
#             Corresponding market resource points for interpolation.
#         cNrm_Vec : np.array
#             Consumption points for interpolation.

#         Returns
#         -------
#         cFunc_unconstrained : CubicInterp
#             The unconstrained consumption function for this period.
#         """

# #        scsr = self.soln_crnt.scsr
#         bilt = self.soln_crnt.bilt
#         folw = self.soln_crnt.folw

#         def vPP_tp1(shk_vector, a_number):
#             return shk_vector[0] ** (- bilt.CRRA - 1.0) \
#                 * folw.vPPfunc_tp1(self.m_Nrm_tp1(shk_vector, a_number))

#         EndOfPrdvPP = (
#             bilt.DiscFac * bilt.LivPrb
#             * bilt.Rfree
#             * bilt.Rfree
#             * bilt.PermGroFac ** (-bilt.CRRA - 1.0)
#             * calc_expectation(
#                 bilt.IncShkDstn,
#                 vPP_tp1,
#                 bilt.aNrmGrid
#             )
#         )
#         dcda = EndOfPrdvPP / bilt.uPP(np.array(cNrm_Vec[1:]))
#         MPC = dcda / (dcda + 1.0)
#         MPC = np.insert(MPC, 0, bilt.MPCmax)

#         cFuncUnc = CubicInterp(
#             mNrm_Vec, cNrm_Vec, MPC, bilt.MPCmin *
#             bilt.hNrm, bilt.MPCmin
#         )
#         return cFuncUnc

    def make_EndOfPrdvFunc(self, EndOfPrdvP):
        """
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.soln_crnt.aNrm.

        Returns
        -------
        none
        """

        breakpoint()
        bilt = self.soln_crnt.bilt

        def v_Lvl_tp1(shk_vector, a_number):
            return (
                shk_vector[0] ** (1.0 - bilt.CRRA)
                * bilt.PermGroFac ** (1.0 - bilt.CRRA)
            ) * bilt.vFuncNxt(self.soln_crnt.m_Nrm_tp1(shk_vector, a_number))
        EndOfPrdv = bilt.DiscLiv * calc_expectation(
            bilt.IncShkDstn, v_Lvl_tp1, self.soln_crnt.aNrm
        )
        EndOfPrdvNvrs = self.soln_crnt.uinv(
            EndOfPrdv
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * self.soln_crnt.uinvP(EndOfPrdv)
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(
            EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0]
        )  # This is a very good approximation, vNvrsPP = 0 at the asset minimum
        aNrm_temp = np.insert(self.soln_crnt.aNrm, 0, self.soln_crnt.BoroCnstNat)
        EndOfPrdvNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        self.soln_crnt.EndOfPrdvFunc = ValueFuncCRRA(
            EndOfPrdvNvrsFunc, bilt.CRRA)

    def add_vFunc(self, soln_crnt, EndOfPrdvP):
        """
        Creates the value function for this period and adds it to the soln_crnt.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, likely including the
            consumption function, marginal value function, etc.
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.soln_crnt.aNrm.

        Returns
        -------
        solution : ConsumerSolution
            The single period solution passed as an input, but now with the
            value function (defined over market resources m) as an attribute.
        """
        self.make_EndOfPrdvFunc(EndOfPrdvP)
        self.vFunc = soln_crnt.vFunc = self.make_vFunc(soln_crnt)
        return soln_crnt.vFunc

    # def make_vFunc(self, soln_crnt):
    #     """
    #     Creates the value function for this period, defined over market resources m.
    #     self must have the attribute EndOfPrdvFunc in order to execute.

    #     Parameters
    #     ----------
    #     solution : ConsumerSolution
    #         The solution to this single period problem, which must include the
    #         consumption function.

    #     Returns
    #     -------
    #     vFunc : ValueFuncCRRA
    #         A representation of the value function for this period, defined over
    #         normalized market resources m: v = vFunc(m).
    #     """
    #     # Compute expected value and marginal value on a grid of market resources
    #     bilt = self.soln_crnt.bilt

    #     mNrm_temp = bilt.mNrmMin + bilt.aXtraGrid
    #     cNrm = soln_crnt.cFunc(mNrm_temp)
    #     aNrm = mNrm_temp - cNrm
    #     vNrm = bilt.u(cNrm) + self.EndOfPrdvFunc(aNrm)
    #     vPnow = self.uP(cNrm)

    #     # Construct the beginning value function
    #     vNvrs = bilt.uinv(vNrm)  # value transformed through inverse utility
    #     vNvrsP = vPnow * bilt.uinvP(vNrm)
    #     mNrm_temp = np.insert(mNrm_temp, 0, bilt.mNrmMin)
    #     vNvrs = np.insert(vNvrs, 0, 0.0)
    #     vNvrsP = np.insert(
    #         vNvrsP, 0, bilt.MPCmaxEff ** (-bilt.CRRA /
    #                                       (1.0 - bilt.CRRA))
    #     )
    #     MPCminNvrs = bilt.MPCmin ** (-bilt.CRRA /
    #                                  (1.0 - bilt.CRRA))
    #     vNvrsFunc = CubicInterp(
    #         mNrm_temp, vNvrs, vNvrsP, MPCminNvrs * bilt.hNrm, MPCminNvrs
    #     )
    #     vFunc = ValueFuncCRRA(vNvrsFunc, bilt.CRRA)
    #     return vFunc

    # def add_vPPfunc(self, soln_crnt):
    #     """
    #     Adds the marginal marginal value function to an existing solution, so
    #     that the next solver can evaluate vPP and thus use cubic interpolation.

    #     Parameters
    #     ----------
    #     solution : ConsumerSolution
    #         The solution to this single period problem, which must include the
    #         consumption function.

    #     Returns
    #     -------
    #     solution : ConsumerSolution
    #         The same solution passed as input, but with the marginal marginal
    #         value function for this period added as the attribute vPPfunc.
    #     """
    #     self.vPPfunc = MargMargValueFuncCRRA(soln_crnt.bilt.cFunc, soln_crnt.bilt.CRRA)
    #     soln_crnt.bilt.vPPfunc = self.vPPfunc
    #     return soln_crnt.bilt.vPPfunc


####################################################################################################
####################################################################################################
