from copy import deepcopy

import numpy as np
from scipy import sparse as sp

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    LognormPermIncShk,
)
from HARK.distribution import (
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    IndexDistribution,
    MeanOneLogNormal,
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
)
from HARK.utilities import (
    gen_tran_matrix_1D,
    gen_tran_matrix_2D,
    jump_to_grid_1D,
    jump_to_grid_2D,
    make_grid_exp_mult,
)


class SequenceSpaceJacobianType(IndShockConsumerType):
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
        """Defines the grid on which the distribution is defined. Stores the grid of market resources and permanent income as attributes of self.
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

        if not isinstance(timestonest, int):
            timestonest = self.mFac
        else:
            timestonest = timestonest

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
                    1.05 + 1e-3,
                    np.exp(max_p),
                    num_points,
                    3,
                )
                self.dist_pGrid = np.append(
                    np.append(1.0 / np.fliplr([one_sided_grid])[0], np.ones(1)),
                    one_sided_grid,
                )  # Compute permanent income grid

            else:
                # If grid of permanent income prespecified then use it as pgrid
                self.dist_pGrid = dist_pGrid

            if self.neutral_measure:  # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
                self.dist_pGrid = np.array([1])

        elif self.cycles > 1:
            raise Exception(
                "define_distribution_grid requires cycles = 0 or cycles = 1",
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
                        1.05 + 1e-3,
                        np.exp(max_p),
                        num_points,
                        2,
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

            if self.neutral_measure:  # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
                self.dist_pGrid = self.T_cycle * [np.array([1])]

    def calc_transition_matrix(self, shk_dstn=None):
        """Calculates how the distribution of agents across market resources
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
            bNext = self.Rfree * aNext
            shk_prbs = shk_dstn[0].pmv  # Probability of shocks
            tran_shks = shk_dstn[0].atoms[1]  # Transitory shocks
            perm_shks = shk_dstn[0].atoms[0]  # Permanent shocks
            LivPrb = self.LivPrb[0]  # Update probability of staying alive

            # New borns have this distribution (assumes start with no assets and permanent income=1)
            # NewBornDist = jump_to_grid_2D(
            #     tran_shks, np.ones_like(tran_shks), shk_prbs, dist_mGrid, dist_pGrid
            # )

            if len(dist_pGrid) == 1:
                NewBornDist = jump_to_grid_1D(
                    1.1 * np.ones_like(tran_shks),
                    shk_prbs,
                    dist_mGrid,
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

            self.cPol_Grid = []  # List of consumption policy grids for each period in T_cycle
            self.aPol_Grid = []  # List of asset policy grids for each period in T_cycle
            self.tran_matrix = []  # List of transition matrices

            dist_mGrid = self.dist_mGrid
            # dist_aGrid = deepcopy(self.dist_mGrid)

            # dist_mGrid_temp  = dist_aGrid*self.Rfree[k] +
            for k in range(self.T_cycle):
                if type(self.dist_pGrid) == list:
                    # Permanent income grid this period
                    dist_pGrid = self.dist_pGrid[k]
                else:
                    dist_pGrid = (
                        self.dist_pGrid
                    )  # If here then use prespecified permanent income grid

                # Consumption policy grid in period k
                # Cnow = self.solution[k].cFunc(dist_mGrid)
                Cnow = self.solution[k].cFunc(dist_mGrid)

                self.cPol_Grid.append(Cnow)  # Add to list

                aNext = dist_mGrid - Cnow  # Asset policy grid in period k
                self.aPol_Grid.append(aNext)  # Add to list

                if type(self.Rfree) == list:
                    bNext = self.Rfree[k] * aNext
                else:
                    bNext = self.Rfree * aNext

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
                        np.ones_like(tran_shks),
                        shk_prbs,
                        dist_mGrid,
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
        """Calculates the ergodic distribution across normalized market resources and
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
            transition_matrix[0],
            v0=np.ones(len(transition_matrix[0])),
            k=1,
            which="LM",
        )  # Solve for ergodic distribution
        ergodic_distr = ergodic_distr.real / np.sum(ergodic_distr.real)

        self.vec_erg_dstn = ergodic_distr  # distribution as a vector
        # distribution reshaped into len(mgrid) by len(pgrid) array
        self.erg_dstn = ergodic_distr.reshape(
            (len(self.dist_mGrid), len(self.dist_pGrid)),
        )

    def compute_steady_state(self):
        # Compute steady state to perturb around
        self.cycles = 0
        self.solve()

        # Use Harmenberg Measure
        self.neutral_measure = True
        self.update_income_process()

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
        """Calculates the Jacobians of aggregate consumption and aggregate assets. Parameters that can be shocked are
        LivPrb, PermShkStd,TranShkStd, DiscFac, UnempPrb, Rfree, IncUnemp, DiscFac .

        Parameters
        ----------
        shk_param: string
            name of variable to be shocked

        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        -------
        CJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Consumption with respect to shk_param

        AJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        """
        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["Rfree"] = params["T_cycle"] * [self.Rfree]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp]

        params["wage"] = params["T_cycle"] * [self.wage[0]]
        params["taxrate"] = params["T_cycle"] * [self.taxrate[0]]
        params["labor"] = params["T_cycle"] * [self.labor[0]]
        params["TranShkMean_Func"] = params["T_cycle"] * [self.TranShkMean_Func[0]]

        # Create instance of a finite horizon agent
        FinHorizonAgent = SequenceSpaceJacobianType(**params)
        FinHorizonAgent.cycles = 1  # required

        # delete Rfree from time invariant list since it varies overtime
        FinHorizonAgent.del_from_time_inv("Rfree")
        # Add Rfree to time varying list to be able to introduce time varying interest rates
        FinHorizonAgent.add_to_time_vary("Rfree")

        # Set Terminal Solution as Steady State Consumption Function
        FinHorizonAgent.cFunc_terminal_ = deepcopy(self.solution[0].cFunc)

        dx = 0.0001  # Size of perturbation
        # Period in which the change in the interest rate occurs (second to last period)
        i = params["T_cycle"] - 1

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        FinHorizonAgent.del_from_time_inv(shk_param)
        FinHorizonAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                (i) * [getattr(self, shk_param)[0]]
                + [getattr(self, shk_param)[0] + dx]
                + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                (i) * [getattr(self, shk_param)]
                + [getattr(self, shk_param) + dx]
                + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent

        setattr(FinHorizonAgent, shk_param, peturbed_list)

        # Update income process if perturbed parameter enters the income shock distribution
        FinHorizonAgent.update_income_process()

        # Solve
        FinHorizonAgent.solve()

        # FinHorizonAgent.Rfree = params["T_cycle"] * [self.Rfree]
        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        FinHorizonAgent.update_income_process()

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid()
        FinHorizonAgent.calc_transition_matrix()

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
        params["T_cycle"] = 2  # Dimension of Jacobian Matrix

        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["Rfree"] = params["T_cycle"] * [self.Rfree]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp]
        params["wage"] = params["T_cycle"] * [self.wage[0]]
        params["taxrate"] = params["T_cycle"] * [self.taxrate[0]]
        params["labor"] = params["T_cycle"] * [self.labor[0]]
        params["TranShkMean_Func"] = params["T_cycle"] * [self.TranShkMean_Func[0]]
        params["IncShkDstn"] = params["T_cycle"] * [self.IncShkDstn[0]]
        params["cFunc_terminal_"] = deepcopy(self.solution[0].cFunc)

        if shk_param == "DiscFac":
            params["DiscFac"] = params["T_cycle"] * [self.DiscFac]

        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = SequenceSpaceJacobianType(**params)
        ZerothColAgent.cycles = 1  # required

        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        if type(getattr(self, shk_param)) == list:
            ZerothColAgent.shk_param = params["T_cycle"] * [getattr(self, shk_param)[0]]
        else:
            ZerothColAgent.shk_param = params["T_cycle"] * [getattr(self, shk_param)]

        # Update income process if perturbed parameter enters the income shock distribution
        ZerothColAgent.update_income_process()

        # Solve
        ZerothColAgent.solve()

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = [getattr(self, shk_param)[0] + dx] + (
                params["T_cycle"] - 1
            ) * [
                getattr(self, shk_param)[0],
            ]  # Sequence of interest rates the agent faces
        else:
            peturbed_list = [getattr(self, shk_param) + dx] + (
                params["T_cycle"] - 1
            ) * [
                getattr(self, shk_param),
            ]  # Sequence of interest rates the agent

        setattr(ZerothColAgent, shk_param, peturbed_list)  # Set attribute to agent

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process()

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

    def calc_agg_path(self, Z, T):
        """Parameters
        ---------
        Z: numpy.array
            sequence of labor values

        Returns
        -------
         CJAC: numpy.array
             TxT Jacobian Matrix of Aggregate Consumption with respect to shk_param

         AJAC: numpy.array
             TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        """
        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["Rfree"] = params["T_cycle"] * [self.Rfree]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp]

        params["wage"] = params["T_cycle"] * [self.wage[0]]
        params["taxrate"] = params["T_cycle"] * [self.taxrate[0]]
        params["labor"] = params["T_cycle"] * [self.labor[0]]
        params["TranShkMean_Func"] = params["T_cycle"] * [self.TranShkMean_Func[0]]

        # Create instance of a finite horizon agent
        FinHorizonAgent = SequenceSpaceJacobianType(**params)
        FinHorizonAgent.cycles = 1  # required

        # delete Rfree from time invariant list since it varies overtime
        FinHorizonAgent.del_from_time_inv("Rfree")
        # Add Rfree to time varying list to be able to introduce time varying interest rates
        FinHorizonAgent.add_to_time_vary("Rfree")

        # Set Terminal Solution as Steady State Consumption Function
        FinHorizonAgent.cFunc_terminal_ = deepcopy(self.solution[0].cFunc)

        # Period in which the change in the interest rate occurs (second to last period)

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        # FinHorizonAgent.del_from_time_inv(shk_param)
        # FinHorizonAgent.add_to_time_vary(shk_param)

        peturbed_list = []
        for z in Z:
            peturbed_list += [z]

            # (
            #     (i) * [getattr(self, shk_param)[0]]
            #     + [getattr(self, shk_param)[0] + dx]
            #     + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
            # )

        # # this condition is because some attributes are specified as lists while other as floats
        # if type(getattr(self, shk_param)) == list:
        #     peturbed_list = (
        #         (i) * [getattr(self, shk_param)[0]]
        #         + [getattr(self, shk_param)[0] + dx]
        #         + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
        #     )  # Sequence of interest rates the agent faces
        # else:
        #     peturbed_list = (
        #         (i) * [getattr(self, shk_param)]
        #         + [getattr(self, shk_param) + dx]
        #         + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
        #     )  # Sequence of interest rates the agent

        FinHorizonAgent.labor = peturbed_list

        # Update income process if perturbed parameter enters the income shock distribution
        FinHorizonAgent.update_income_process()

        # Solve
        FinHorizonAgent.solve()

        # FinHorizonAgent.Rfree = params["T_cycle"] * [self.Rfree]
        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        FinHorizonAgent.update_income_process()

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid()
        FinHorizonAgent.calc_transition_matrix()

        tranmats = FinHorizonAgent.tran_matrix
        cGrids = FinHorizonAgent.cPol_Grid
        aGrids = FinHorizonAgent.aPol_Grid

        C_t = np.zeros(FinHorizonAgent.T_cycle)
        A_t = np.zeros(FinHorizonAgent.T_cycle)

        dstn_t = self.vec_erg_dstn
        for t in range(FinHorizonAgent.T_cycle):
            dstn_t = np.dot(tranmats[t], dstn_t)

            C_t[t] = np.dot(cGrids[t], dstn_t)
            A_t[t] = np.dot(aGrids[t], dstn_t)

        return C_t, A_t

    # = Functions for generating discrete income processes and
    #   simulated income shocks =
    # ========================================================

    def construct_lognormal_income_process_unemployment(self):
        """Generates a list of discrete approximations to the income process for each
        life period, from end of life to beginning of life.  Permanent shocks are mean
        one lognormally distributed with standard deviation PermShkStd[t] during the
        working life, and degenerate at 1 in the retirement period.  Transitory shocks
        are mean one lognormally distributed with a point mass at IncUnemp with
        probability UnempPrb while working; they are mean one with a point mass at
        IncUnempRet with probability UnempPrbRet.  Retirement occurs
        after t=T_retire periods of working.

        Note 1: All time in this function runs forward, from t=0 to t=T

        Note 2: All parameters are passed as attributes of the input parameters.

        Parameters (passed as attributes of the input parameters)
        ----------
        PermShkStd : [float]
            List of standard deviations in log permanent income uncertainty during
            the agent's life.
        PermShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        TranShkStd : [float]
            List of standard deviations in log transitory income uncertainty during
            the agent's life.
        TranShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        UnempPrb : float or [float]
            The probability of becoming unemployed during the working period.
        UnempPrbRet : float or None
            The probability of not receiving typical retirement income when retired.
        T_retire : int
            The index value for the final working period in the agent's life.
            If T_retire <= 0 then there is no retirement.
        IncUnemp : float or [float]
            Transitory income received when unemployed.
        IncUnempRet : float or None
            Transitory income received while "unemployed" when retired.
        T_cycle :  int
            Total number of non-terminal periods in the consumer's sequence of periods.

        Returns
        -------
        IncShkDstn :  [distribution.Distribution]
            A list with T_cycle elements, each of which is a
            discrete approximation to the income process in a period.
        PermShkDstn : [[distribution.Distributiony]]
            A list with T_cycle elements, each of which
            a discrete approximation to the permanent income shocks.
        TranShkDstn : [[distribution.Distribution]]
            A list with T_cycle elements, each of which
            a discrete approximation to the transitory income shocks.

        """
        # Unpack the parameters from the input
        T_cycle = self.T_cycle
        PermShkStd = self.PermShkStd
        PermShkCount = self.PermShkCount
        TranShkStd = self.TranShkStd
        TranShkCount = self.TranShkCount
        T_retire = self.T_retire
        UnempPrb = self.UnempPrb
        IncUnemp = self.IncUnemp
        UnempPrbRet = self.UnempPrbRet
        IncUnempRet = self.IncUnempRet

        taxrate = self.taxrate
        TranShkMean_Func = self.TranShkMean_Func
        labor = self.labor
        wage = self.wage

        if T_retire > 0:
            normal_length = T_retire
            retire_length = T_cycle - T_retire
        else:
            normal_length = T_cycle
            retire_length = 0

        if all(
            [
                isinstance(x, (float, int)) or (x is None)
                for x in [UnempPrb, IncUnemp, UnempPrbRet, IncUnempRet]
            ],
        ):
            UnempPrb_list = [UnempPrb] * normal_length + [UnempPrbRet] * retire_length
            IncUnemp_list = [IncUnemp] * normal_length + [IncUnempRet] * retire_length

        elif all([isinstance(x, list) for x in [UnempPrb, IncUnemp]]):
            UnempPrb_list = UnempPrb
            IncUnemp_list = IncUnemp

        else:
            raise Exception(
                "Unemployment must be specified either using floats for UnempPrb,"
                + "IncUnemp, UnempPrbRet, and IncUnempRet, in which case the "
                + "unemployment probability and income change only with retirement, or "
                + "using lists of length T_cycle for UnempPrb and IncUnemp, specifying "
                + "each feature at every age.",
            )

        PermShkCount_list = [PermShkCount] * normal_length + [1] * retire_length
        TranShkCount_list = [TranShkCount] * normal_length + [1] * retire_length

        if not hasattr(self, "neutral_measure"):
            self.neutral_measure = False

        neutral_measure_list = [self.neutral_measure] * len(PermShkCount_list)
        """
        IncShkDstn = IndexDistribution(
            engine=BufferStockIncShkDstn,
            conditional={
                "sigma_Perm": PermShkStd,
                "sigma_Tran": TranShkStd,
                "n_approx_Perm": PermShkCount_list,
                "n_approx_Tran": TranShkCount_list,
                "neutral_measure": neutral_measure_list,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
            },
            RNG=self.RNG,
        )
        """
        PermShkDstn = IndexDistribution(
            engine=LognormPermIncShk,
            conditional={
                "sigma": PermShkStd,
                "n_approx": PermShkCount_list,
                "neutral_measure": neutral_measure_list,
            },
        )
        """
        TranShkDstn = IndexDistribution(
            engine=MixtureTranIncShk,
            conditional={
                "sigma": TranShkStd,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
                "n_approx": TranShkCount_list,
            },
        )
        """

        IncShkDstn = IndexDistribution(
            engine=HANKIncShkDstn,
            conditional={
                "sigma_Perm": PermShkStd,
                "sigma_Tran": TranShkStd,
                "n_approx_Perm": PermShkCount_list,
                "n_approx_Tran": TranShkCount_list,
                "neutral_measure": neutral_measure_list,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
                "wage": wage,
                "taxrate": taxrate,
                "labor": labor,
                "TranShkMean_Func": TranShkMean_Func,
            },
            RNG=self.RNG,
        )

        TranShkDstn = IndexDistribution(
            engine=MixtureTranIncShk_HANK,
            conditional={
                "sigma": TranShkStd,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
                "n_approx": TranShkCount_list,
                "wage": wage,
                "taxrate": taxrate,
                "labor": labor,
                "TranShkMean_Func": TranShkMean_Func,
            },
        )

        return IncShkDstn, PermShkDstn, TranShkDstn


class MixtureTranIncShk_HANK(DiscreteDistribution):
    """A one-period distribution for transitory income shocks that are a mixture
    between a log-normal and a single-value unemployment shock.

    Parameters
    ----------
    sigma : float
        Standard deviation of the log-shock.
    UnempPrb : float
        Probability of the "unemployment" shock.
    IncUnemp : float
        Income shock in the "unemployment" state.
    n_approx : int
        Number of points to use in the discrete approximation.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    TranShkDstn : DiscreteDistribution
        Transitory income shock distribution.

    """

    def __init__(
        self,
        sigma,
        UnempPrb,
        IncUnemp,
        n_approx,
        wage,
        labor,
        taxrate,
        TranShkMean_Func,
        seed=0,
    ):
        dstn_approx = MeanOneLogNormal(sigma).discretize(
            n_approx if sigma > 0.0 else 1,
            method="equiprobable",
            tail_N=0,
        )

        if UnempPrb > 0.0:
            dstn_approx = add_discrete_outcome_constant_mean(
                dstn_approx,
                p=UnempPrb,
                x=IncUnemp,
            )

        dstn_approx.atoms = dstn_approx.atoms * TranShkMean_Func(taxrate, labor, wage)

        super().__init__(pmv=dstn_approx.pmv, atoms=dstn_approx.atoms, seed=seed)


class HANKIncShkDstn(DiscreteDistributionLabeled):
    """A one-period distribution object for the joint distribution of income
    shocks (permanent and transitory), as modeled in the Buffer Stock Theory
    paper:
        - Lognormal, discretized permanent income shocks.
        - Transitory shocks that are a mixture of:
            - A lognormal distribution in normal times.
            - An "unemployment" shock.

    Parameters
    ----------
    sigma_Perm : float
        Standard deviation of the log- permanent shock.
    sigma_Tran : float
        Standard deviation of the log- transitory shock.
    n_approx_Perm : int
        Number of points to use in the discrete approximation of the permanent shock.
    n_approx_Tran : int
        Number of points to use in the discrete approximation of the transitory shock.
    UnempPrb : float
        Probability of the "unemployment" shock.
    IncUnemp : float
        Income shock in the "unemployment" state.
    neutral_measure : Bool, optional
        Whether to use Harmenberg's permanent-income-neutral measure. The default is False.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    IncShkDstn : DiscreteDistribution
        Income shock distribution.

    """

    def __init__(
        self,
        sigma_Perm,
        sigma_Tran,
        n_approx_Perm,
        n_approx_Tran,
        UnempPrb,
        IncUnemp,
        taxrate,
        TranShkMean_Func,
        labor,
        wage,
        neutral_measure=False,
        seed=0,
    ):
        perm_dstn = LognormPermIncShk(
            sigma=sigma_Perm,
            n_approx=n_approx_Perm,
            neutral_measure=neutral_measure,
        )
        tran_dstn = MixtureTranIncShk_HANK(
            sigma=sigma_Tran,
            UnempPrb=UnempPrb,
            IncUnemp=IncUnemp,
            n_approx=n_approx_Tran,
            wage=wage,
            labor=labor,
            taxrate=taxrate,
            TranShkMean_Func=TranShkMean_Func,
        )

        joint_dstn = combine_indep_dstns(perm_dstn, tran_dstn)

        super().__init__(
            name="HANK",
            var_names=["PermShk", "TranShk"],
            pmv=joint_dstn.pmv,
            atoms=joint_dstn.atoms,
            seed=seed,
        )
