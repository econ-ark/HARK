"""
This file has a slightly modified and extended version of ConsIndShock that is
meant to be used in heterogeneous agents new Keynesian (HANK) models. The micro-
economic model is identical, but additional primitive parameters have been added
to the specification of the income process. These parameters would have no inde-
pendent meaning in a "micro only" setting, but with dynamic equilibrium elements
(as in HANK models), they can have meaning.
"""

from copy import deepcopy
import numpy as np
from scipy import sparse as sp

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    make_basic_CRRA_solution_terminal,
    solve_one_period_ConsIndShock,
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
)

from HARK.Calibration.Income.IncomeProcesses import (
    construct_HANK_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)

from HARK.utilities import (
    gen_tran_matrix_1D,
    gen_tran_matrix_2D,
    jump_to_grid_1D,
    jump_to_grid_2D,
    make_grid_exp_mult,
    make_assets_grid,
)

# Make a dictionary of constructors for the idiosyncratic income shocks model
newkeynesian_constructor_dict = {
    "IncShkDstn": construct_HANK_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
    "solution_terminal": make_basic_CRRA_solution_terminal,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
default_kNrmInitDstn_params = {
    "kLogInitMean": 0.0,  # Mean of log initial capital
    "kLogInitStd": 1.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
default_pLvlInitDstn_params = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
default_IncShkDstn_params = {
    "PermShkStd": [0.1],  # Standard deviation of log permanent income shocks
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.1],  # Standard deviation of log transitory income shocks
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate while working
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
    "tax_rate": [0.0],  # Flat tax rate on labor income NEW FOR HANK
    "labor": [1.0],  # Intensive margin labor supply NEW FOR HANK
    "wage": [1.0],  # Wage rate scaling factor NEW FOR HANK
}

# Default parameters to make aXtraGrid using make_assets_grid
default_aXtraGrid_params = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 50,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 100,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Make a dictionary to specify an idiosyncratic income shocks consumer type
init_newkeynesian = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 0,  # Infinite horizon model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": newkeynesian_constructor_dict,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.0],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    # (The portion of PermGroFac attributable to aggregate productivity growth)
    "NewbornTransShk": False,  # Whether Newborns have transitory shock
    # ADDITIONAL OPTIONAL PARAMETERS
    "PerfMITShk": False,  # Do Perfect Foresight MIT Shock
    # (Forces Newborns to follow solution path of the agent they replaced if True)
    "neutral_measure": False,  # Whether to use permanent income neutral measure (see Harmenberg 2021)
    # ADDITIONAL PARAMETERS FOR GRID-BASED TRANSITION SIMULATION
    "mMin": 0.001,
    "mMax": 50,
    "mCount": 200,
    "mFac": 3,
}
init_newkeynesian.update(default_kNrmInitDstn_params)
init_newkeynesian.update(default_pLvlInitDstn_params)
init_newkeynesian.update(default_IncShkDstn_params)
init_newkeynesian.update(default_aXtraGrid_params)


class NewKeynesianConsumerType(IndShockConsumerType):
    """
    A slight extension of IndShockConsumerType that permits individual labor supply,
    the wage rate, and the labor income tax rate to enter the income shock process.
    """

    default_ = {
        "params": init_newkeynesian,
        "solver": solve_one_period_ConsIndShock,
    }

    def define_distribution_grid(
        self,
        dist_mGrid=None,
        dist_pGrid=None,
        m_density=0,
        num_pointsM=None,
        timestonest=None,
        num_pointsP=55,
        max_p_fac=30.0,
    ):
        """
        Defines the grid on which the distribution is defined. Stores the grid of market resources and permanent income as attributes of self.
        Grid for normalized market resources and permanent income may be prespecified
        as dist_mGrid and dist_pGrid, respectively. If not then default grid is computed based off given parameters.

        Parameters
        ----------
        dist_mGrid : np.array
                Prespecified grid for distribution over normalized market resources

        dist_pGrid : np.array
                Prespecified grid for distribution over permanent income.

        m_density: float
                Density of normalized market resources grid. Default value is mdensity = 0.
                Only affects grid of market resources if dist_mGrid=None.

        num_pointsM: float
                Number of gridpoints for market resources grid.

        num_pointsP: float
                 Number of gridpoints for permanent income.
                 This grid will be exponentiated by the function make_grid_exp_mult.

        max_p_fac : float
                Factor that scales the maximum value of permanent income grid.
                Larger values increases the maximum value of permanent income grid.

        Returns
        -------
        None
        """

        # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
        if not hasattr(self, "neutral_measure"):
            self.neutral_measure = False

        if num_pointsM is None:
            m_points = self.mCount
        else:
            m_points = num_pointsM

        if timestonest is None:
            timestonest = self.mFac
        elif not isinstance(timestonest, (int, float)):
            raise TypeError("timestonest must be a numeric value (int or float).")

        if self.cycles == 0:
            if not hasattr(dist_mGrid, "__len__"):
                mGrid = make_grid_exp_mult(
                    ming=self.mMin,
                    maxg=self.mMax,
                    ng=m_points,
                    timestonest=timestonest,
                )  # Generate Market resources grid given density and number of points

                for i in range(m_density):
                    m_shifted = np.delete(mGrid, -1)
                    m_shifted = np.insert(m_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = mGrid - m_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_A_grid = m_shifted + dist_betw_pts_half
                    mGrid = np.concatenate((mGrid, new_A_grid))
                    mGrid = np.sort(mGrid)

                self.dist_mGrid = mGrid

            else:
                # If grid of market resources prespecified then use as mgrid
                self.dist_mGrid = dist_mGrid

            if not hasattr(dist_pGrid, "__len__"):
                num_points = num_pointsP  # Number of permanent income gridpoints
                # Dist_pGrid is taken to cover most of the ergodic distribution
                # set variance of permanent income shocks
                p_variance = self.PermShkStd[0] ** 2
                # Maximum Permanent income value
                max_p = max_p_fac * (p_variance / (1 - self.LivPrb[0])) ** 0.5
                one_sided_grid = make_grid_exp_mult(
                    1.05 + 1e-3, np.exp(max_p), num_points, 3
                )
                self.dist_pGrid = np.append(
                    np.append(1.0 / np.fliplr([one_sided_grid])[0], np.ones(1)),
                    one_sided_grid,
                )  # Compute permanent income grid
            else:
                # If grid of permanent income prespecified then use it as pgrid
                self.dist_pGrid = dist_pGrid

            if (
                self.neutral_measure is True
            ):  # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
                self.dist_pGrid = np.array([1])

        elif self.cycles > 1:
            raise Exception(
                "define_distribution_grid requires cycles = 0 or cycles = 1"
            )

        elif self.T_cycle != 0:
            if num_pointsM is None:
                m_points = self.mCount
            else:
                m_points = num_pointsM

            if not hasattr(dist_mGrid, "__len__"):
                mGrid = make_grid_exp_mult(
                    ming=self.mMin,
                    maxg=self.mMax,
                    ng=m_points,
                    timestonest=timestonest,
                )  # Generate Market resources grid given density and number of points

                for i in range(m_density):
                    m_shifted = np.delete(mGrid, -1)
                    m_shifted = np.insert(m_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = mGrid - m_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_A_grid = m_shifted + dist_betw_pts_half
                    mGrid = np.concatenate((mGrid, new_A_grid))
                    mGrid = np.sort(mGrid)

                self.dist_mGrid = mGrid

            else:
                # If grid of market resources prespecified then use as mgrid
                self.dist_mGrid = dist_mGrid

            if not hasattr(dist_pGrid, "__len__"):
                self.dist_pGrid = []  # list of grids of permanent income

                for i in range(self.T_cycle):
                    num_points = num_pointsP
                    # Dist_pGrid is taken to cover most of the ergodic distribution
                    # set variance of permanent income shocks this period
                    p_variance = self.PermShkStd[i] ** 2
                    # Consider probability of staying alive this period
                    max_p = max_p_fac * (p_variance / (1 - self.LivPrb[i])) ** 0.5
                    one_sided_grid = make_grid_exp_mult(
                        1.05 + 1e-3, np.exp(max_p), num_points, 2
                    )

                    # Compute permanent income grid this period. Grid of permanent income may differ dependent on PermShkStd
                    dist_pGrid = np.append(
                        np.append(1.0 / np.fliplr([one_sided_grid])[0], np.ones(1)),
                        one_sided_grid,
                    )
                    self.dist_pGrid.append(dist_pGrid)

            else:
                # If grid of permanent income prespecified then use as pgrid
                self.dist_pGrid = dist_pGrid

            if (
                self.neutral_measure is True
            ):  # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
                self.dist_pGrid = self.T_cycle * [np.array([1])]

    def calc_transition_matrix(self, shk_dstn=None):
        """
        Calculates how the distribution of agents across market resources
        transitions from one period to the next. If finite horizon problem, then calculates
        a list of transition matrices, consumption and asset policy grids for each period of the problem.
        The transition matrix/matrices and consumption and asset policy grid(s) are stored as attributes of self.


        Parameters
        ----------
            shk_dstn: list
                list of income shock distributions. Each Income Shock Distribution should be a DiscreteDistribution Object (see Distribution.py)
        Returns
        -------
        None

        """

        if self.cycles == 0:  # Infinite Horizon Problem
            if not hasattr(shk_dstn, "pmv"):
                shk_dstn = self.IncShkDstn

            dist_mGrid = self.dist_mGrid  # Grid of market resources
            dist_pGrid = self.dist_pGrid  # Grid of permanent incomes
            # assets next period
            aNext = dist_mGrid - self.solution[0].cFunc(dist_mGrid)

            self.aPol_Grid = aNext  # Steady State Asset Policy Grid
            # Steady State Consumption Policy Grid
            self.cPol_Grid = self.solution[0].cFunc(dist_mGrid)

            # Obtain shock values and shock probabilities from income distribution
            # Bank Balances next period (Interest rate * assets)
            bNext = self.Rfree[0] * aNext
            shk_prbs = shk_dstn[0].pmv  # Probability of shocks
            tran_shks = shk_dstn[0].atoms[1]  # Transitory shocks
            perm_shks = shk_dstn[0].atoms[0]  # Permanent shocks
            LivPrb = self.LivPrb[0]  # Update probability of staying alive

            # New borns have this distribution (assumes start with no assets and permanent income=1)
            NewBornDist = jump_to_grid_2D(
                tran_shks, np.ones_like(tran_shks), shk_prbs, dist_mGrid, dist_pGrid
            )

            if len(dist_pGrid) == 1:
                NewBornDist = jump_to_grid_1D(
                    np.ones_like(tran_shks), shk_prbs, dist_mGrid
                )
                # Compute Transition Matrix given shocks and grids.
                self.tran_matrix = gen_tran_matrix_1D(
                    dist_mGrid,
                    bNext,
                    shk_prbs,
                    perm_shks,
                    tran_shks,
                    LivPrb,
                    NewBornDist,
                )

            else:
                NewBornDist = jump_to_grid_2D(
                    np.ones_like(tran_shks),
                    np.ones_like(tran_shks),
                    shk_prbs,
                    dist_mGrid,
                    dist_pGrid,
                )

                # Generate Transition Matrix
                # Compute Transition Matrix given shocks and grids.
                self.tran_matrix = gen_tran_matrix_2D(
                    dist_mGrid,
                    dist_pGrid,
                    bNext,
                    shk_prbs,
                    perm_shks,
                    tran_shks,
                    LivPrb,
                    NewBornDist,
                )

        elif self.cycles > 1:
            raise Exception("calc_transition_matrix requires cycles = 0 or cycles = 1")

        elif self.T_cycle != 0:  # finite horizon problem
            if not hasattr(shk_dstn, "pmv"):
                shk_dstn = self.IncShkDstn

            self.cPol_Grid = []
            # List of consumption policy grids for each period in T_cycle
            self.aPol_Grid = []
            # List of asset policy grids for each period in T_cycle
            self.tran_matrix = []  # List of transition matrices

            dist_mGrid = self.dist_mGrid

            for k in range(self.T_cycle):
                if type(self.dist_pGrid) == list:
                    # Permanent income grid this period
                    dist_pGrid = self.dist_pGrid[k]
                else:
                    dist_pGrid = (
                        self.dist_pGrid
                    )  # If here then use prespecified permanent income grid

                # Consumption policy grid in period k
                Cnow = self.solution[k].cFunc(dist_mGrid)
                self.cPol_Grid.append(Cnow)  # Add to list

                aNext = dist_mGrid - Cnow  # Asset policy grid in period k
                self.aPol_Grid.append(aNext)  # Add to list

                bNext = self.Rfree[k] * aNext

                # Obtain shocks and shock probabilities from income distribution this period
                shk_prbs = shk_dstn[k].pmv  # Probability of shocks this period
                # Transitory shocks this period
                tran_shks = shk_dstn[k].atoms[1]
                # Permanent shocks this period
                perm_shks = shk_dstn[k].atoms[0]
                # Update probability of staying alive this period
                LivPrb = self.LivPrb[k]

                if len(dist_pGrid) == 1:
                    # New borns have this distribution (assumes start with no assets and permanent income=1)
                    NewBornDist = jump_to_grid_1D(
                        np.ones_like(tran_shks), shk_prbs, dist_mGrid
                    )
                    # Compute Transition Matrix given shocks and grids.
                    TranMatrix_M = gen_tran_matrix_1D(
                        dist_mGrid,
                        bNext,
                        shk_prbs,
                        perm_shks,
                        tran_shks,
                        LivPrb,
                        NewBornDist,
                    )
                    self.tran_matrix.append(TranMatrix_M)

                else:
                    NewBornDist = jump_to_grid_2D(
                        np.ones_like(tran_shks),
                        np.ones_like(tran_shks),
                        shk_prbs,
                        dist_mGrid,
                        dist_pGrid,
                    )
                    # Compute Transition Matrix given shocks and grids.
                    TranMatrix = gen_tran_matrix_2D(
                        dist_mGrid,
                        dist_pGrid,
                        bNext,
                        shk_prbs,
                        perm_shks,
                        tran_shks,
                        LivPrb,
                        NewBornDist,
                    )
                    self.tran_matrix.append(TranMatrix)

    def calc_ergodic_dist(self, transition_matrix=None):
        """
        Calculates the ergodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is stored as attributes of self both as a vector and as a reshaped array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.

        Parameters
        ----------
        transition_matrix: List
                    list with one transition matrix whose ergordic distribution is to be solved
        Returns
        -------
        None
        """

        if not isinstance(transition_matrix, list):
            transition_matrix = [self.tran_matrix]

        eigen, ergodic_distr = sp.linalg.eigs(
            transition_matrix[0], v0=np.ones(len(transition_matrix[0])), k=1, which="LM"
        )  # Solve for ergodic distribution
        ergodic_distr = ergodic_distr.real / np.sum(ergodic_distr.real)

        self.vec_erg_dstn = ergodic_distr  # distribution as a vector
        # distribution reshaped into len(mgrid) by len(pgrid) array
        self.erg_dstn = ergodic_distr.reshape(
            (len(self.dist_mGrid), len(self.dist_pGrid))
        )

    def compute_steady_state(self):
        # Compute steady state to perturb around
        self.cycles = 0
        self.solve()

        # Use Harmenberg Measure
        self.neutral_measure = True
        self.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")

        # Non stochastic simuation
        self.define_distribution_grid()
        self.calc_transition_matrix()

        self.c_ss = self.cPol_Grid  # Normalized Consumption Policy grid
        self.a_ss = self.aPol_Grid  # Normalized Asset Policy grid

        self.calc_ergodic_dist()  # Calculate ergodic distribution
        # Steady State Distribution as a vector (m*p x 1) where m is the number of gridpoints on the market resources grid
        ss_dstn = self.vec_erg_dstn

        self.A_ss = np.dot(self.a_ss, ss_dstn)[0]
        self.C_ss = np.dot(self.c_ss, ss_dstn)[0]

        return self.A_ss, self.C_ss

    def calc_jacobian(self, shk_param, T):
        """
        Calculates the Jacobians of aggregate consumption and aggregate assets.
        Parameters that can be shocked are LivPrb, PermShkStd,TranShkStd, DiscFac,
        UnempPrb, Rfree, IncUnemp, and DiscFac.

        Parameters:
        -----------

        shk_param: string
            name of variable to be shocked

        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        ----------
        CJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Consumption with respect to shk_param

        AJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        """

        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix

        # Specify a dictionary of lists because problem we are solving is
        # technically finite horizon so variables can be time varying (see
        # section on fake news algorithm in
        # https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp]
        params["wage"] = params["T_cycle"] * [self.wage[0]]
        params["labor"] = params["T_cycle"] * [self.labor[0]]
        params["tax_rate"] = params["T_cycle"] * [self.tax_rate[0]]
        params["cycles"] = 1  # "finite horizon", sort of

        # Create instance of a finite horizon agent
        FinHorizonAgent = NewKeynesianConsumerType(**params)

        dx = 0.0001  # Size of perturbation
        # Period in which the change in the interest rate occurs (second to last period)
        i = params["T_cycle"] - 1

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        FinHorizonAgent.del_from_time_inv(shk_param)
        FinHorizonAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            perturbed_list = (
                (i) * [getattr(self, shk_param)[0]]
                + [getattr(self, shk_param)[0] + dx]
                + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            perturbed_list = (
                (i) * [getattr(self, shk_param)]
                + [getattr(self, shk_param) + dx]
                + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent faces
        setattr(FinHorizonAgent, shk_param, perturbed_list)
        self.parameters[shk_param] = perturbed_list

        # Update income process if perturbed parameter enters the income shock distribution
        FinHorizonAgent.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")

        # Solve the "finite horizon" model assuming that it ends back in steady state
        FinHorizonAgent.solve(presolve=False, from_solution=self.solution[0])

        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        FinHorizonAgent.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid()
        FinHorizonAgent.calc_transition_matrix()

        # Normalized consumption Policy Grids across time
        c_t = FinHorizonAgent.cPol_Grid
        a_t = FinHorizonAgent.aPol_Grid

        # Append steady state policy grid into list of policy grids as HARK does not provide the initial policy
        c_t.append(self.c_ss)
        a_t.append(self.a_ss)

        # Fake News Algorithm begins below ( To find fake news algorithm See page 2388 of https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434  )

        ##########
        # STEP 1 # of fake news algorithm, As in the paper for Curly Y and Curly D. Here the policies are over assets and consumption so we denote them as curly C and curly D.
        ##########
        a_ss = self.aPol_Grid  # steady state Asset Policy
        c_ss = self.cPol_Grid  # steady state Consumption Policy
        tranmat_ss = self.tran_matrix  # Steady State Transition Matrix

        # List of asset policies grids where households expect the shock to occur in the second to last Period
        a_t = FinHorizonAgent.aPol_Grid
        # add steady state assets to list as it does not get appended in calc_transition_matrix method
        a_t.append(self.a_ss)

        # List of consumption policies grids where households expect the shock to occur in the second to last Period
        c_t = FinHorizonAgent.cPol_Grid
        # add steady state consumption to list as it does not get appended in calc_transition_matrix method
        c_t.append(self.c_ss)

        da0_s = []  # Deviation of asset policy from steady state policy
        dc0_s = []  # Deviation of Consumption policy from steady state policy
        for i in range(T):
            da0_s.append(a_t[T - i] - a_ss)
            dc0_s.append(c_t[T - i] - c_ss)

        da0_s = np.array(da0_s)
        dc0_s = np.array(dc0_s)

        # Steady state distribution of market resources (permanent income weighted distribution)
        D_ss = self.vec_erg_dstn.T[0]
        dA0_s = []
        dC0_s = []
        for i in range(T):
            dA0_s.append(np.dot(da0_s[i], D_ss))
            dC0_s.append(np.dot(dc0_s[i], D_ss))

        dA0_s = np.array(dA0_s)
        # This is equivalent to the curly Y scalar detailed in the first step of the algorithm
        A_curl_s = dA0_s / dx

        dC0_s = np.array(dC0_s)
        C_curl_s = dC0_s / dx

        # List of computed transition matrices for each period
        tranmat_t = FinHorizonAgent.tran_matrix
        tranmat_t.append(tranmat_ss)

        # List of change in transition matrix relative to the steady state transition matrix
        dlambda0_s = []
        for i in range(T):
            dlambda0_s.append(tranmat_t[T - i] - tranmat_ss)

        dlambda0_s = np.array(dlambda0_s)

        dD0_s = []
        for i in range(T):
            dD0_s.append(np.dot(dlambda0_s[i], D_ss))

        dD0_s = np.array(dD0_s)
        D_curl_s = dD0_s / dx  # Curly D in the sequence space jacobian

        ########
        # STEP2 # of fake news algorithm
        ########

        # Expectation Vectors
        exp_vecs_a = []
        exp_vecs_c = []

        # First expectation vector is the steady state policy
        exp_vec_a = a_ss
        exp_vec_c = c_ss
        for i in range(T):
            exp_vecs_a.append(exp_vec_a)
            exp_vec_a = np.dot(tranmat_ss.T, exp_vec_a)

            exp_vecs_c.append(exp_vec_c)
            exp_vec_c = np.dot(tranmat_ss.T, exp_vec_c)

        # Turn expectation vectors into arrays
        exp_vecs_a = np.array(exp_vecs_a)
        exp_vecs_c = np.array(exp_vecs_c)

        #########
        # STEP3 # of the algorithm. In particular equation 26 of the published paper.
        #########
        # Fake news matrices
        Curl_F_A = np.zeros((T, T))  # Fake news matrix for assets
        Curl_F_C = np.zeros((T, T))  # Fake news matrix for consumption

        # First row of Fake News Matrix
        Curl_F_A[0] = A_curl_s
        Curl_F_C[0] = C_curl_s

        for i in range(T - 1):
            for j in range(T):
                Curl_F_A[i + 1][j] = np.dot(exp_vecs_a[i], D_curl_s[j])
                Curl_F_C[i + 1][j] = np.dot(exp_vecs_c[i], D_curl_s[j])

        ########
        # STEP4 #  of the algorithm
        ########

        # Function to compute jacobian matrix from fake news matrix
        def J_from_F(F):
            J = F.copy()
            for t in range(1, F.shape[0]):
                J[1:, t] += J[:-1, t - 1]
            return J

        J_A = J_from_F(Curl_F_A)
        J_C = J_from_F(Curl_F_C)

        ########
        # Additional step due to compute Zeroth Column of the Jacobian
        ########

        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = 2  # Just need one transition matrix
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp]
        params["IncShkDstn"] = params["T_cycle"] * [self.IncShkDstn[0]]
        params["wage"] = params["T_cycle"] * [self.wage[0]]
        params["labor"] = params["T_cycle"] * [self.labor[0]]
        params["tax_rate"] = params["T_cycle"] * [self.tax_rate[0]]
        params["cycles"] = 1  # Now it's "finite" horizon while things are changing

        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = NewKeynesianConsumerType(**params)

        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        # Update income process if perturbed parameter enters the income shock distribution
        ZerothColAgent.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")

        # Solve the "finite horizon" problem, again assuming that steady state comes
        # after the shocks
        ZerothColAgent.solve(presolve=False, from_solution=self.solution[0])

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            perturbed_list = [getattr(self, shk_param)[0] + dx] + (
                params["T_cycle"] - 1
            ) * [
                getattr(self, shk_param)[0]
            ]  # Sequence of interest rates the agent faces
        else:
            perturbed_list = [getattr(self, shk_param) + dx] + (
                params["T_cycle"] - 1
            ) * [getattr(self, shk_param)]
            # Sequence of interest rates the agent

        setattr(ZerothColAgent, shk_param, perturbed_list)  # Set attribute to agent
        self.parameters[shk_param] = perturbed_list

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")

        # Calculate Transition Matrices
        ZerothColAgent.define_distribution_grid()
        ZerothColAgent.calc_transition_matrix()

        tranmat_t_zeroth_col = ZerothColAgent.tran_matrix
        dstn_t_zeroth_col = self.vec_erg_dstn.T[0]

        C_t_no_sim = np.zeros(T)
        A_t_no_sim = np.zeros(T)

        for i in range(T):
            if i == 0:
                dstn_t_zeroth_col = np.dot(tranmat_t_zeroth_col[i], dstn_t_zeroth_col)
            else:
                dstn_t_zeroth_col = np.dot(tranmat_ss, dstn_t_zeroth_col)

            C_t_no_sim[i] = np.dot(self.cPol_Grid, dstn_t_zeroth_col)
            A_t_no_sim[i] = np.dot(self.aPol_Grid, dstn_t_zeroth_col)

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx

        return J_C, J_A
