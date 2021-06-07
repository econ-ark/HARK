import numpy as np
from copy import copy, deepcopy
from builtins import (range, str, breakpoint)
from types import SimpleNamespace

from HARK.core import (_log, set_verbosity_level)
from HARK.distribution \
    import (add_discrete_outcome_constant_mean,
            combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform)
from HARK.interpolation import (LinearInterp)
from HARK import AgentType, make_one_period_oo_solver
from HARK.ConsumptionSaving.ConsIndShockModel_CommonDefs \
    import (def_utility, def_value_funcs,
            construct_assets_grid)
from HARK.ConsumptionSaving.ConsIndShockModel_AgentSolve \
    import (ConsumerSolutionOneStateCRRA,
            ConsPerfForesightSolver,
            ConsIndShockSolverBasic, ConsIndShockSolver,
            ConsKinkedRsolver
            )
from HARK.ConsumptionSaving.ConsIndShockModel_AgentDicts \
    import (init_perfect_foresight, init_idiosyncratic_shocks)

from HARK.ConsumptionSaving.ConsIndShockModel_AgentTypes \
    import AgentTypePlus


class OneStateCRRA_Bellman_EndOfPeriod(AgentTypePlus):

    """

    Parameters
    ----------

    solution_startfrom : stge, optional
        A user-specified terminal period/stage solution for the iteration,
        to be used in place of the hardwired solution_terminal.  One use
        might be to set a loose tolerance to get a quick `solution_rough`
        using the default hardwired solution (nobequest), then
        set the tolerance tighter, or change some approximation parameter,
        and resume iteration using `solution_startfrom = solution_rough` until
        the new tolerance is met with the (presumably better but slower)
        approximation parameter.
    """

    # Time invariant parameters
    time_inv_ = [
        "CubicBool"
    ]

    def __init__(self,
                 cycles=1, T_cycle=1, verbose=1,  quiet=True, solution_startfrom=None,
                 **kwds):
        params = init_idiosyncratic_shocks.copy()  # Get default params
        params.update(kwds)  # Update/overwrite defaults with user-specified

        # Inherit characteristics of a PF model with the same parameters
        PerfForesightConsumerType.__init__(self, cycles=cycles,
                                           verbose=verbose, quiet=quiet,
                                           **params)

        self.cycles = 1

        # If precooked terminal stage not provided by user ...
        if not hasattr(self, 'solution_startfrom'):  # .. then init the default
            self.agent_force_prepare_info_needed_to_begin_solving()

        # - Default interpolation method is piecewise linear
        # - Cubic is smoother, works if problem has no constraints
        # - User may or may not want to create the value function
        # TODO: CDC 20210428: Basic solver is not worth preserving
        # - 1. We might as well always compute vFunc
        # - 2. Cubic vs linear interpolation is not worth two different solvers
        # -    * Cubic should be preserved as an option
        if self.CubicBool or self.vFuncBool:
            solverName = ConsIndShockSolver

        # Attach the corresponding one-stage solver to the agent
        # This is what gets called when the user invokes [instance].solve()
        if (solverType == 'HARK') or (solverType == 'DARKolo'):
            self.solve_one_period = make_one_period_oo_solver(solverName)

        if (solverType == 'dolo') or (solverType == 'DARKolo'):
            # If we want to solve with dolo, set up the model
            self.dolo_model()

        # Store setup parameters so later we can check for changes
        # that necessitate restarting solution process

        self.agent_store_model_params(params['prmtv_par'], params['aprox_lim'])

        # Put the (enhanced) solution_terminal in self.solution[0]
        self.make_solution_for_final_period()

    def dolo_model(self):
        # Create a dolo version of the model
        from dolo import yaml_import
        self.dolo_modl = yaml_import(
            '/Volumes/Data/Code/ARK/DARKolo/chimeras/BufferStock/bufferstock.yaml'
        )
        if self.verbose >= 2:
            _log.info(self.dolo_modl)

    def agent_force_prepare_info_needed_to_begin_solving(self):
        """
        Update any characteristics of the agent that need to be recomputed
        as a result of changes in parameters since the last time the solver was invoked.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        solve_par_vals_now = {}
        if not hasattr(self, 'solve_par_vals'):  # We haven't set it up yet
            self.update_income_process()
            self.update_assets_grid()
        else:  # it has been set up, so see if anything changed
            for par in self.solve_par_vals:
                solve_par_vals_now[par] = getattr(self, par)
            if not solve_par_vals_now == self.solve_par_vals:
                _log.info('Some model parameter has changed since last update.')
                _log.info('Storing new parameters and updating shocks and grid.')
                self.update_income_process()
                self.update_assets_grid()

    pre_solve = agent_force_prepare_info_needed_to_begin_solving

    # The former "[AgentType].update_pre_solve()" was poor nomenclature --
    #  easy to confuse with the also-existing "[AgentType].pre_solve()" and with
    # "[SolverType].prepare_to_solve()".  The new name,
    #
    # agent_force_prepare_info_needed_to_begin_solving()
    #
    # is better.  The old one
    # is preserved as an alias, below, to prevent breakage of existing code:

    def update_income_process(self):
        """
        Updates agent's income shock specs based on its current attributes.

        Parameters
        ----------
        none

        Returns:
        --------
        none
        """

        (self.IncShkDstn,
            self.permShkDstn,
            self.tranShkDstn,
         ) = self.construct_lognormal_income_process_unemployment()
        self.add_to_time_vary("IncShkDstn", "permShkDstn", "tranShkDstn")
        self.parameters.update({'IncShkDstn': self.IncShkDstn,
                                'permShkDstn': self.permShkDstn,
                                'tranShkDstn': self.tranShkDstn})

    def update_assets_grid(self):
        """
        Updates this agent's end-of-period assets grid by constructing a multi-
        exponentially spaced grid of aXtra values.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.aXtraGrid = construct_assets_grid(self)
        self.add_to_time_inv("aXtraGrid")
        self.parameters.update({'aXtraGrid': self.aXtraGrid})

    def reset_rng(self):
        """
        Reset the RNG behavior of this type.  This method is called automatically
        by initialize_sim(), ensuring that each simulation run uses the same sequence
        of random shocks; this is necessary for structural estimation to work.
        This method extends AgentType.reset_rng() to also reset elements of IncShkDstn.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PerfForesightConsumerType.reset_rng(self)

        # Reset IncShkDstn if it exists (it might not because reset_rng is called at init)
        if hasattr(self, "IncShkDstn"):
            for dstn in self.IncShkDstn:
                dstn.reset()

    mcrlo_reset_rng = reset_rng_mcrlo = reset_rng

    def construct_lognormal_income_process_unemployment(self):
        """
        Generates a sequence of discrete approximations to the income process for each
        life period, from end of life to beginning of life.  Permanent shocks are mean
        one lognormally distributed with standard deviation permShkStd[t] during the
        working life, and degenerate at 1 in the retirement period.  transitory shocks
        are mean one lognormally distributed with a point mass at IncUnemp with
        probability UnempPrb while working; they are mean one with a point mass at
        IncUnempRet with probability UnempPrbRet.  Retirement occurs
        after t=T_retire periods of working.

        Note 1: All time in this function runs forward, from t=0 to t=T

        Note 2: All parameters are passed as attributes of the input parameters.

        Parameters (passed as attributes of the input parameters)
        ----------
        permShkStd : [float]
            List of standard deviations in log permanent income uncertainty during
            the agent's life.
        permShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        tranShkStd : [float]
            List of standard deviations in log transitory income uncertainty during
            the agent's life.
        tranShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        UnempPrb : float
            The probability of becoming unemployed during the working period.
        UnempPrbRet : float
            The probability of not receiving typical retirement income when retired.
        T_retire : int
            The index value for the final working period in the agent's life.
            If T_retire <= 0 then there is no retirement.
        IncUnemp : float
            transitory income received when unemployed.
        IncUnempRet : float
            transitory income received while "unemployed" when retired.
        T_cycle :  int
            Total number of non-terminal periods in the consumer's sequence of periods.

        Returns
        -------
        IncShkDstn :  [distribution.Distribution]
            A list with elements from t = 0 to T_cycle, each of which is a
            discrete approximation to the joint income distribution at at [t]
        permShkDstn : [[distribution.Distribution]]
            A list with elements from t = 0 to T_cycle, each of which is a
            discrete approximation to the permanent shock distribution at [t]
        tranShkDstn : [[distribution.Distribution]]
            A list with elements from t = 0 to T_cycle, each of which is a
            discrete approximation to the transitory shock distribution at [t]
        """
        # Unpack the parameters from the input

        permShkStd = self.permShkStd
        permShkCount = self.permShkCount
        tranShkStd = self.tranShkStd
        tranShkCount = self.tranShkCount
        UnempPrb = self.UnempPrb
        UnempPrbRet = self.UnempPrbRet
        T_retire = self.T_retire
        IncUnemp = self.IncUnemp
        IncUnempRet = self.IncUnempRet
        T_cycle = self.T_cycle

        # make a dictionary of the parameters
        # Created so, later, we can determine whether any parameters have changed
        parameters = {
            'permShkStd':  self.permShkStd,
            'permShkCount':  self.permShkCount,
            'tranShkStd':  self.tranShkStd,
            'tranShkCount':  self.tranShkCount,
            'UnempPrb':  self.UnempPrb,
            'UnempPrbRet':  self.UnempPrbRet,
            'T_retire':  self.T_retire,
            'IncUnemp':  self.IncUnemp,
            'IncUnempRet':  self.IncUnempRet,
            'T_cycle':  self.T_cycle,
            'ShkPosn': {'perm': 0, 'tran': 1}
        }

        # constructed_by: later, we can determine whether another distribution
        # object was constructed using the same method or a different method
        constructed_by = {'method': 'construct_lognormal_income_process_unemployment'}

        IncShkDstn = []  # Discrete approximations to income process in each period
        permShkDstn = []  # Discrete approximations to permanent income shocks
        tranShkDstn = []  # Discrete approximations to transitory income shocks

        # Fill out a simple discrete RV for retirement, with value 1.0 (mean of shocks)
        # in normal times; value 0.0 in "unemployment" times with small prob.
        if T_retire > 0:
            if UnempPrbRet > 0:
                #                permShkValsNxtRet = np.array([1.0, 1.0])  # Permanent income is deterministic in retirement (2 states for temp income shocks)
                tranShkValsRet = np.array(
                    [
                        IncUnempRet,
                        (1.0 - UnempPrbRet * IncUnempRet) / (1.0 - UnempPrbRet),
                    ]
                )
                ShkPrbsRet = np.array([UnempPrbRet, 1.0 - UnempPrbRet])
            else:
                (IncShkDstnRet,
                 permShkDstnRet,
                 tranShkDstnRet,
                 ) = self.construct_lognormal_income_process_unemployment()
                ShkPrbsRet = IncShkDstnRet.pmf

        # Loop to fill in the list of IncShkDstn random variables.
        for t in range(T_cycle):  # Iterate over all periods, counting forward
            if T_retire > 0 and t >= T_retire:
                # Then we are in the "retirement period" and add a retirement income object.
                IncShkDstn.append(deepcopy(IncShkDstnRet))
                permShkDstn.append([np.array([1.0]), np.array([1.0])])
                tranShkDstn.append([ShkPrbsRet, tranShkValsRet])
            else:
                # We are in the "working life" periods.
                tranShkDstn_t = MeanOneLogNormal(sigma=tranShkStd[t]).approx(
                    tranShkCount, tail_N=0
                )
                if UnempPrb > 0:
                    tranShkDstn_t = add_discrete_outcome_constant_mean(
                        tranShkDstn_t, p=UnempPrb, x=IncUnemp
                    )
                permShkDstn_t = MeanOneLogNormal(sigma=permShkStd[t]).approx(
                    permShkCount, tail_N=0
                )
                IncShkDstn.append(
                    combine_indep_dstns(
                        permShkDstn_t,
                        tranShkDstn_t,
                        seed=self.RNG.randint(0, 2 ** 31 - 1),
                    )
                )  # mix the independent distributions
                permShkDstn.append(permShkDstn_t)
                tranShkDstn.append(tranShkDstn_t)

        IncShkDstn[-1].parameters = parameters
        IncShkDstn[-1].constructed_by = constructed_by

        return IncShkDstn, permShkDstn, tranShkDstn

    def get_shocks(self):  # mcrlo simulation tool
        """
        Gets permanent and transitory income shocks for this period.  Samples from IncShkDstn for
        each period in the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        permShk = np.zeros(self.AgentCount)  # Initialize shock arrays
        tranShk = np.zeros(self.AgentCount)
        newborn = self.t_age == 0
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                IncShkDstn = self.IncShkDstn[
                    t - 1
                ]  # set current income distribution
                PermGroFac = self.PermGroFac[t - 1]  # and permanent growth factor
                # Get random draws of income shocks from the discrete distribution
                IncShks = IncShkDstn.draw(N)

                permShk[these] = (
                    IncShks[0, :] * PermGroFac
                )  # permanent "shock" includes expected growth
                tranShk[these] = IncShks[1, :]

        # That procedure used the *last* period in the sequence for newborns, but that's not right
        # Redraw shocks for newborns, using the *first* period in the sequence.  Approximation.
        N = np.sum(newborn)
        if N > 0:
            these = newborn
            IncShkDstn = self.IncShkDstn[0]  # set current income distribution
            PermGroFac = self.PermGroFac[0]  # and permanent growth factor

            # Get random draws of income shocks from the discrete distribution
            EventDraws = IncShkDstn.draw_events(N)
            permShk[these] = (
                IncShkDstn.X[0][EventDraws] * PermGroFac
            )  # permanent "shock" includes expected growth
            tranShk[these] = IncShkDstn.X[1][EventDraws]
            #        permShk[newborn] = 1.0
        tranShk[newborn] = 1.0

        # Store the shocks in self
        self.Emp = np.ones(self.AgentCount, dtype=bool)
        self.Emp[tranShk == self.IncUnemp] = False
        self.shocks['permShk'] = permShk
        self.shocks['tranShk'] = tranShk

    get_shocks_mcrlo = mcrlo_get_shocks = get_shocks


# Make a dictionary to specify a "kinked R" idiosyncratic shock consumer
init_kinked_R = dict(
    init_idiosyncratic_shocks,
    **{
        "Rboro": 1.20,  # Interest factor on assets when borrowing, a < 0
        "Rsave": 1.02,  # Interest factor on assets when saving, a > 0
        "BoroCnstArt": None,  # kinked R is a bit silly if borrowing not allowed
        "CubicBool": True,  # kinked R is now compatible with linear cFunc and cubic cFunc
        "aXtraCount": 48,  # ...so need lots of extra gridpoints to make up for it
    }
)
del init_kinked_R["Rfree"]  # get rid of constant interest factor
