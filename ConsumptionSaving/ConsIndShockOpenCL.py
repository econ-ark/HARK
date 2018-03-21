'''
This module provides functions and classes for implementing ConsIndShockModel
on OpenCL, enabling the use of GPU computing.
'''

import sys
import os
import numpy as np
import opencl4py as cl
os.environ["PYOPENCL_CTX"] = "0:1" # This is where you choose a device number
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

f = open('ConsIndShockModel.cl')
program_code = f.read()
f.close()

platforms = cl.Platforms()
ctx = platforms.create_some_context()
queue = ctx.create_queue(ctx.devices[0])

class IndShockConsumerTypesOpenCL():
    '''
    A class for representing one or more instances of IndShockConsumerType using
    OpenCL, possibly on a Graphics Processing Unit (GPU).
    '''
    def __init__(self,agents):
        '''
        Make a new instance by passing a list of agent types.
        
        Parameters
        ----------
        agents : [IndShockConsumerType]
            List of agent types to be represented in OpenCL.
            
        Returns
        -------
        None
        '''
        self.agents = agents
        self.TypeCount = len(agents)
        self.IntegerInputs = np.zeros(8,dtype=int)
        self.IntegerInputs[1] = self.TypeCount
        self.IntegerInputs_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,self.IntegerInputs)
        
        self.makeParameterBuffers()  # Make buffers with primitive and constructed parameters
        self.makeSimulationBuffers() # Make buffers for holding current simulated variables
        self.program = ctx.create_program(program_code)
        
        
        
    def loadSimulationKernels(self):
        '''
        Loads simulation kernels into memory, with buffers slotted into each
        input as appropriate.  Should only be run after running makeSolutionBuffers.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        self.getMortalityKrn = self.program.get_kernel("getMortality")
        self.getMortalityKrn.set_args(self.IntegerInputs_buf,
                                      self.TypeNow_buf,
                                      self.t_cycle_buf,
                                      self.t_age_buf,
                                      self.TypeAddress_buf,
                                      self.NormDraws_buf,
                                      self.UniDraws_buf,
                                      self.LivPrb_buf,
                                      self.aNrmInitMean_buf,
                                      self.aNrmInitStd_buf,
                                      self.pLvlInitMean_buf,
                                      self.pLvlInitStd_buf,
                                      self.aNrmNow_buf,
                                      self.pLvlNow_buf)
        
        self.getShocksKrn = self.program.get_kernel("getShocks")
        self.getShocksKrn.set_args(self.IntegerInputs_buf,
                                   self.TypeNow_buf,
                                   self.t_cycle_buf,
                                   self.TypeAddress_buf,
                                   self.NormDraws_buf,
                                   self.UniDraws_buf,
                                   self.PermStd_buf,
                                   self.TranStd_buf,
                                   self.UnempPrb_buf,
                                   self.IncUnemp_buf,
                                   self.PermShkNow_buf,
                                   self.TranShkNow_buf)
        
        self.getStatesKrn = self.program.get_kernel("getStates")
        self.getStatesKrn.set_args(self.IntegerInputs_buf,
                                   self.TypeNow_buf,
                                   self.t_cycle_buf,
                                   self.TypeAddress_buf,
                                   self.PermGroFac_buf,
                                   self.Rfree_buf,
                                   self.aNrmNow_buf,
                                   self.PermShkNow_buf,
                                   self.TranShkNow_buf,
                                   self.mNrmNow_buf,
                                   self.pLvlNow_buf)
        
        self.getControlsKrn = self.program.get_kernel("getControls")
        self.getControlsKrn.set_args(self.IntegerInputs_buf,
                                     self.TypeNow_buf,
                                     self.t_cycle_buf,
                                     self.T_total_buf,
                                     self.TypeAddress_buf,
                                     self.CoeffsAddress_buf,
                                     self.mGrid_buf,
                                     self.mLowerBound_buf,
                                     self.Coeffs0_buf,
                                     self.Coeffs1_buf,
                                     self.Coeffs2_buf,
                                     self.Coeffs3_buf,
                                     self.mNrmNow_buf,
                                     self.cNrmNow_buf,
                                     self.MPCnow_buf)
        
        self.getPostStatesKrn = self.program.get_kernel("getPostStates")
        self.getPostStatesKrn.set_args(self.IntegerInputs_buf,
                                       self.TypeNow_buf,
                                       self.t_cycle_buf,
                                       self.t_age_buf,
                                       self.T_total_buf,
                                       self.mNrmNow_buf,
                                       self.cNrmNow_buf,
                                       self.pLvlNow_buf,
                                       self.aNrmNow_buf,
                                       self.aLvlNow_buf)
        
        self.simOnePeriodKrn = self.program.get_kernel("simOnePeriod")
        self.simOnePeriodKrn.set_args(self.IntegerInputs_buf,
                                      self.TypeNow_buf,
                                      self.t_cycle_buf,
                                      self.t_age_buf,
                                      self.T_total_buf,
                                      self.TypeAddress_buf,
                                      self.CoeffsAddress_buf,
                                      self.NormDraws_buf,
                                      self.UniDraws_buf,
                                      self.LivPrb_buf,
                                      self.aNrmInitMean_buf,
                                      self.aNrmInitStd_buf,
                                      self.pLvlInitMean_buf,
                                      self.pLvlInitStd_buf,
                                      self.PermStd_buf,
                                      self.TranStd_buf,
                                      self.UnempPrb_buf,
                                      self.IncUnemp_buf,
                                      self.PermGroFac_buf,
                                      self.Rfree_buf,
                                      self.mGrid_buf,
                                      self.mLowerBound_buf,
                                      self.Coeffs0_buf,
                                      self.Coeffs1_buf,
                                      self.Coeffs2_buf,
                                      self.Coeffs3_buf,
                                      self.PermShkNow_buf,
                                      self.TranShkNow_buf,
                                      self.aNrmNow_buf,
                                      self.pLvlNow_buf,
                                      self.mNrmNow_buf,
                                      self.cNrmNow_buf,
                                      self.MPCnow_buf,
                                      self.aLvlNow_buf,
                                      self.TestVar_buf)
        
        
        
    def prepareToSolve(self):
        '''
        Makes buffers to hold data that will be used by the OpenCL solver kernel,
        including buffers to hold the solution itself.  All buffers are stored as
        attributes of self with the _buf suffix.  Needs primitive parameters to be
        defined, and the update() method to have run (so that IncomeDstn exists).
        This method only works for cycles != 0.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        # Make buffers with preference parameters
        CRRA_vec = np.array([agent.CRRA for agent in self.agents])
        DiscFac_vec = np.array([agent.DiscFac for agent in self.agents])
        BoroCnst_vec = np.array([agent.BoroCnstArt for agent in self.agents])
        self.CRRA_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,CRRA_vec)
        self.DiscFac_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,DiscFac_vec)
        self.BoroCnst_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,BoroCnst_vec)
        
        # Make buffer for the aXtraGrid
        aXtraGrid_vec = np.concatenate([agent.aXtraGrid for agent in self.agents])
        self.aXtraGrid_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,aXtraGrid_vec)
        
        # Prepare to make buffers for income distributions
        T_total_vec = self.T_total_vec
        IncomePrbs_list = []
        PermShks_list = []
        TranShks_list = []
        IncDstnAddress_vec = np.zeros(np.sum(T_total_vec)+1,dtype=int)
        WorstIncPrb_vec = np.zeros(np.sum(T_total_vec)+1)
        idx = 0
        loc = 0
        for j in range(len(self.agents)):
            this_agent = self.agents[j]
            lengths = this_agent.cycles*[this_agent.IncomeDstn[t][0].size for t in range(this_agent.T_cycle)]
            lengths.append(0)
            IncDstnAddress_vec[(idx+1):(idx+T_total_vec[j]+1)] = loc + np.cumsum(lengths)
            loc += np.sum(lengths)
            IncomePrbs_temp = np.concatenate([this_agent.IncomeDstn[t][0] for t in range(this_agent.T_cycle)])
            IncomePrbs_list.append(np.tile(IncomePrbs_temp,this_agent.cycles))
            PermShks_temp = np.concatenate([this_agent.IncomeDstn[t][1] for t in range(this_agent.T_cycle)])
            PermShks_list.append(np.tile(PermShks_temp,this_agent.cycles))
            TranShks_temp = np.concatenate([this_agent.IncomeDstn[t][2] for t in range(this_agent.T_cycle)])
            TranShks_list.append(np.tile(TranShks_temp,this_agent.cycles))
            
            # Get the worst income probability by t for this agent type
            PermShkMin = [np.min(agent.IncomeDstn[t][1]) for t in range(agent.T_cycle)]
            TranShkMin = [np.min(agent.IncomeDstn[t][2]) for t in range(agent.T_cycle)]
            WorstIncPrb = [np.sum(agent.IncomeDstn[t][0]*(agent.IncomeDstn[t][1]==PermShkMin[t])*(agent.IncomeDstn[t][2]==TranShkMin[t])) for t in range(agent.T_cycle)]
            WorstIncPrb_adj = agent.cycles*WorstIncPrb + [0.0]
            WorstIncPrb_vec[idx:(idx+T_total_vec[j])] = WorstIncPrb_adj
            idx += T_total_vec[j]
        
        # Make buffers for income distributions                
        IncomePrbs_vec = np.concatenate(IncomePrbs_list)
        PermShks_vec = np.concatenate(PermShks_list)
        TranShks_vec = np.concatenate(TranShks_list)
        self.IncomePrbs_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,IncomePrbs_vec)
        self.PermShks_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,PermShks_vec)
        self.TranShks_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,TranShks_vec)
        self.WorstIncPrb_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,WorstIncPrb_vec)
        self.IncDstnAddress_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,IncDstnAddress_vec[:-1])
        #print(IncDstnAddress_vec)
        
        # Make buffers to hold the solution
        mGridSize = np.sum((T_total_vec-1)*np.array([(agent.aXtraCount+1) for agent in self.agents])) # this assumes a terminal period
        empty_soln_vec = np.zeros(mGridSize)
        self.mGrid_buf   = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,empty_soln_vec)
        self.Coeffs0_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,empty_soln_vec)
        self.Coeffs1_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,empty_soln_vec)
        self.Coeffs2_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,empty_soln_vec)
        self.Coeffs3_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,empty_soln_vec)
        self.mLowerBound_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,np.zeros(np.sum(T_total_vec)))
        CoeffsAddress_vec = np.zeros(np.sum(T_total_vec),dtype=int)
        j = 0
        idx = 0 # This is a really lazy way to do this
        for k in range(len(self.agents)):
            this_agent = self.agents[k]
            for t in range(T_total_vec[k]):
                count = (this_agent.aXtraCount+1)*(t < (T_total_vec[k]-1))
                idx += count
                CoeffsAddress_vec[j] = idx
                j += 1
        CoeffsAddress_vec = np.insert(CoeffsAddress_vec[:-1],0,0.0)
        self.CoeffsAddress_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,CoeffsAddress_vec)
        self.SolnSize = mGridSize
        #print(CoeffsAddress_vec)
        
        # Make buffers with temporary global memory
        ThreadCount = np.sum(np.array([agent.aXtraCount for agent in self.agents]))
        self.ThreadCountSoln = ThreadCount
        empty_temp_vec = np.zeros(ThreadCount)
        self.mTemp_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,empty_temp_vec)
        self.cTemp_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,empty_temp_vec)
        self.MPCtemp_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,empty_temp_vec)
        
        # Adjust integer inputs
        self.IntegerInputs[7] = self.agents[0].aXtraCount # assumes all types have same aXtraCount
        queue.write_buffer(self.IntegerInputs_buf,self.IntegerInputs)
        
        # Load the solver kernel
        self.solveConsIndShockKrn = self.program.get_kernel("solveConsIndShock")
        self.solveConsIndShockKrn.set_args(self.IntegerInputs_buf,
                                        self.TypeAddress_buf,
                                        self.T_total_buf,
                                        self.CoeffsAddress_buf,
                                        self.IncDstnAddress_buf,
                                        self.LivPrb_buf,
                                        self.IncomePrbs_buf,
                                        self.PermShks_buf,
                                        self.TranShks_buf,
                                        self.WorstIncPrb_buf,
                                        self.PermGroFac_buf,
                                        self.Rfree_buf,
                                        self.CRRA_buf,
                                        self.DiscFac_buf,
                                        self.BoroCnst_buf,
                                        self.aXtraGrid_buf,
                                        self.mGrid_buf,
                                        self.mLowerBound_buf,
                                        self.Coeffs0_buf,
                                        self.Coeffs1_buf,
                                        self.Coeffs2_buf,
                                        self.Coeffs3_buf,
                                        self.mTemp_buf,
                                        self.cTemp_buf,
                                        self.MPCtemp_buf)
                                       
        

    def makeParameterBuffers(self):
        '''
        Makes buffers to hold primitive and constructed parameters for the agent
        types represented by this instance.  All buffers are stored as attributes
        of self with the _buf suffix.  Needs primitive parameters to be defined.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        for agent in self.agents:
            agent.timeFwd()
        
        # Calculate the total number of periods for each agent type
        T_cycle_vec = np.array([agent.T_cycle for agent in self.agents],dtype=int)
        cycles_vec = np.array([agent.cycles for agent in self.agents],dtype=int)
        term_bool_vec = (1 - np.array([agent.pseudo_terminal for agent in self.agents],dtype=int))*(cycles_vec > 0)
        T_total_vec = T_cycle_vec*np.maximum(cycles_vec,1) + term_bool_vec
        self.TypeAgeCount = np.sum(T_total_vec)
        self.TypeAddress = np.cumsum(np.insert(T_total_vec,0,0))[0:-1]
        self.T_total_vec = T_total_vec

        # Make buffers for type parameters
        self.TypeAddress_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,self.TypeAddress)
        self.T_total_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,T_total_vec)
        Rfree_vec = np.array([agent.Rfree for agent in self.agents],dtype=np.float64)
        aNrmInitMean_vec = np.array([agent.aNrmInitMean for agent in self.agents],dtype=np.float64)
        aNrmInitStd_vec = np.array([agent.aNrmInitStd for agent in self.agents],dtype=np.float64)
        pLvlInitMean_vec = np.array([agent.pLvlInitMean for agent in self.agents],dtype=np.float64)
        pLvlInitStd_vec = np.array([agent.pLvlInitStd for agent in self.agents],dtype=np.float64)
        self.Rfree_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,Rfree_vec)
        self.aNrmInitMean_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,aNrmInitMean_vec)
        self.aNrmInitStd_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,aNrmInitStd_vec)
        self.pLvlInitMean_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,pLvlInitMean_vec)
        self.pLvlInitStd_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,pLvlInitStd_vec)
        
        # Make buffers for time-varying parameters: LivPrb, PermStd, TranStd, PermGroFac, UnempPrb, IncUnemp
        LivPrb_vec = np.empty(self.TypeAgeCount,dtype=float)
        PermStd_vec = np.empty(self.TypeAgeCount,dtype=float)
        TranStd_vec = np.empty(self.TypeAgeCount,dtype=float)
        PermGroFac_vec = np.empty(self.TypeAgeCount,dtype=float)
        UnempPrb_vec = np.empty(self.TypeAgeCount,dtype=float)
        IncUnemp_vec = np.empty(self.TypeAgeCount,dtype=float)
        for j in range(self.TypeCount):
            agent = self.agents[j]
            cycles = np.maximum(agent.cycles,1)
            bot = self.TypeAddress[j]
            top = bot + T_total_vec[j]
            UnempPrb = agent.UnempPrb*np.ones(agent.T_cycle)
            IncUnemp = agent.IncUnemp*np.ones(agent.T_cycle)
            if agent.T_retire > 0:
                UnempPrb[agent.T_retire:] = agent.UnempPrbRet
                IncUnemp[agent.T_retire:] = agent.IncUnempRet
            if agent.cycles > 0:
                UnempPrb = np.insert(np.tile(UnempPrb,cycles),0,agent.UnempPrb)
                IncUnemp = np.insert(np.tile(IncUnemp,cycles),0,agent.IncUnemp)
                PermGroFac = [1.0] + cycles*agent.PermGroFac
                PermStd = [0.0] + cycles*agent.PermShkStd
                TranStd = [0.0] + cycles*agent.TranShkStd
                LivPrb = [1.0] + cycles*agent.LivPrb
            else:
                UnempPrb = [UnempPrb[-1]] + UnempPrb[0:-1]
                IncUnemp = [IncUnemp[-1]] + IncUnemp[0:-1]
                PermGroFac = [PermGroFac[-1]] + PermGroFac[0:-1]
                PermStd = [PermStd[-1]] + PermStd[0:-1]
                TranStd = [TranStd[-1]] + TranStd[0:-1]
                LivPrb = [LivPrb[-1]] + LivPrb[0:-1]
            LivPrb_vec[bot:top] = LivPrb
            PermStd_vec[bot:top] = PermStd
            TranStd_vec[bot:top] = TranStd
            PermGroFac_vec[bot:top] = PermGroFac
            UnempPrb_vec[bot:top] = UnempPrb
            IncUnemp_vec[bot:top] = IncUnemp
        self.LivPrb_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,LivPrb_vec)
        self.PermStd_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,PermStd_vec)
        self.TranStd_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,TranStd_vec)
        self.PermGroFac_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,PermGroFac_vec)
        self.UnempPrb_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,UnempPrb_vec)
        self.IncUnemp_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,IncUnemp_vec)
        self.IntegerInputs[2] = np.sum(T_total_vec)
        queue.write_buffer(self.IntegerInputs_buf,self.IntegerInputs)
        
        
    def makeSimulationBuffers(self):
        '''
        Makes buffers to hold information on the current population of simulated 
        agents represented by this instance.  All buffers are stored as attributes
        of self with the _buf suffix.  Needs the AgentCount attribute to be defined.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        AgentCount_vec = np.array([agent.AgentCount for agent in self.agents],dtype=int)
        AgentCount = np.sum(AgentCount_vec)
        self.AgentCount = AgentCount
        TypeNow_vec = np.empty(AgentCount,dtype=int)
        bot = 0
        for j in range(self.TypeCount):
            top = bot + AgentCount_vec[j]
            TypeNow_vec[bot:top] = j
            bot = top
        blank_int_vec = np.zeros(AgentCount,dtype=int)
        blank_float_vec = np.zeros(AgentCount,np.float64)
        self.TypeNow_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,TypeNow_vec)
        self.t_cycle_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,blank_int_vec)
        self.t_age_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,blank_int_vec)
        self.mNrmNow_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,blank_float_vec)
        self.cNrmNow_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,blank_float_vec)
        self.aNrmNow_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,blank_float_vec)
        self.pLvlNow_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,blank_float_vec)
        self.aLvlNow_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,blank_float_vec)
        self.MPCnow_buf  = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,blank_float_vec)
        self.PermShkNow_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,blank_float_vec)
        self.TranShkNow_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,blank_float_vec)
        self.TestVar_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,blank_float_vec)
        NormDraws_vec = np.random.randn(65536)
        UniDraws_vec = np.random.rand(500000)
        self.NormDraws_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,NormDraws_vec)
        self.UniDraws_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,UniDraws_vec)
        self.IntegerInputs[0] = AgentCount
        self.IntegerInputs[5] = NormDraws_vec.size
        self.IntegerInputs[6] = UniDraws_vec.size
        queue.write_buffer(self.IntegerInputs_buf,self.IntegerInputs)
        
        
    def makeSolutionBuffers(self):
        '''
        Makes buffers to hold representations of the consumption function for
        types represented by this instance.  All buffers are stored as attributes
        of self with the _buf suffix.  Needs the solution attribute to be defined.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        # Extract numbers that characterize the consumption function for each type-age
        mLowerBound_vec = np.zeros(self.TypeAgeCount)
        CoeffsAddress_vec = np.zeros(self.TypeAgeCount+1,dtype=int)
        mGrid_list = []
        mGrid_size = 0
        for j in range(self.TypeCount):
            agent = self.agents[j]
            bot = self.TypeAddress[j]
            top = bot + self.T_total_vec[j]
            T = len(agent.solution)
            TT = T
            if agent.cycles > 0:
                TT -= 1
            temp = [agent.solution[t].cFunc.functions[0].x_list.size for t in range(TT)]
            mGrid_list.append(np.concatenate([agent.solution[t].cFunc.functions[0].x_list for t in range(TT)]))
            if agent.cycles > 0:
                temp.append(0) # Terminal period has a hardcoded solution
            CoeffsAddress_vec[(bot+1):(top+1)] = mGrid_size + np.cumsum(temp)
            mLowerBound_vec[bot:top] = np.array([agent.solution[t].mNrmMin for t in range(T)])
            mGrid_size += np.sum(temp)
        CoeffsAddress_vec = CoeffsAddress_vec[:-1]
        self.CoeffsAddress = CoeffsAddress_vec
        mGrid_vec = np.concatenate(mGrid_list)
        Coeffs0_vec = np.zeros(mGrid_size)
        Coeffs1_vec = np.zeros(mGrid_size)
        Coeffs2_vec = np.zeros(mGrid_size)
        Coeffs3_vec = np.zeros(mGrid_size)
        for j in range(self.TypeCount):
            agent = self.agents[j]
            T = len(agent.solution)
            TT = T
            if agent.cycles > 0:
                TT -= 1
            bot = CoeffsAddress_vec[self.TypeAddress[j]]
            if j == (self.TypeCount-1):
                top = mGrid_size
            else:
                top = CoeffsAddress_vec[self.TypeAddress[j+1]]
            # First coefficient row is lower extrapolation, never used
            Coeffs0_vec[bot:top] = np.concatenate([agent.solution[t].cFunc.functions[0].coeffs[1:,0] for t in range(TT)])
            Coeffs1_vec[bot:top] = np.concatenate([agent.solution[t].cFunc.functions[0].coeffs[1:,1] for t in range(TT)])
            Coeffs2_vec[bot:top] = np.concatenate([agent.solution[t].cFunc.functions[0].coeffs[1:,2] for t in range(TT)])
            Coeffs3_vec[bot:top] = np.concatenate([agent.solution[t].cFunc.functions[0].coeffs[1:,3] for t in range(TT)])
            
        # Make buffers to hold these solutions
        self.CoeffsAddress_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,CoeffsAddress_vec)
        self.mGrid_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,mGrid_vec)
        self.mLowerBound_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,mLowerBound_vec)
        self.Coeffs0_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,Coeffs0_vec)
        self.Coeffs1_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,Coeffs1_vec)
        self.Coeffs2_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,Coeffs2_vec)
        self.Coeffs3_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,Coeffs3_vec)        
        self.IntegerInputs[3] = CoeffsAddress_vec.size
        queue.write_buffer(self.IntegerInputs_buf,self.IntegerInputs)
        
        
    def writeSimVar(self,var_name):
        '''
        Moves current simulation variable of agents into OpenCL memory for use
        by kernels.  Can only be run after makeSolutionBuffers().
        
        Parameters
        ----------
        var_name : string
            Name of simulation variable to write to OpenCL buffer.  Should exist
            as attribute of each agent in agents and have a corresponding X_buf
            attribute in self.
            
        Returns
        -------
        None
        '''
        if not np.all([hasattr(agent,var_name) for agent in self.agents]):
            print("Some agents don't have an attribute named " + var_name + "!")
            return
        buf_name = var_name + '_buf'
        if not hasattr(self,buf_name):
            print("IndShockConsumerTypesOpenCL instance doesn't have buffer named " + buf_name + "!")
            return            
        var_temp = np.concatenate([getattr(agent,var_name) for agent in self.agents])
        queue.write_buffer(getattr(self,buf_name),var_temp)

        
    def readSimVar(self,var_name):
        '''
        Moves current simulation variable of from OpenCL buffers into each agent
        in self.agents.  Can only be run after makeSolutionBuffers().
        
        Parameters
        ----------
        var_name : string
            Name of simulation variable to read from OpenCL buffer.  Should exist
            as attribute of each agent in agents and have a corresponding X_buf
            attribute in self.
            
        Returns
        -------
        None
        '''
        buf_name = var_name + '_buf'
        if not hasattr(self,buf_name):
            print("IndShockConsumerTypesOpenCL instance doesn't have buffer named " + buf_name + "!")
            return
        try:
            type_temp = getattr(self.agents[0],var_name).dtype
        except:
            type_temp = np.float64
        var_temp = np.empty(self.AgentCount,dtype=type_temp)
        queue.read_buffer(getattr(self,buf_name),var_temp)
        bot = 0
        for j in range(self.TypeCount):
            agent = self.agents[j]
            top = bot + agent.AgentCount
            setattr(agent,var_name,var_temp[bot:top])
            bot = top
            
            
            
    def solve(self):
        '''
        Solve all agent types using the OpenCL kernel.  This overwrites AgentType.solve().
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        WorkGroupSize = self.agents[0].aXtraCount
        GlobalThreadCount = self.ThreadCountSoln
        queue.execute_kernel(self.solveConsIndShockKrn, [GlobalThreadCount], [WorkGroupSize])
        queue.read_buffer(self.IntegerInputs_buf,self.IntegerInputs)
            
            
            
    def simOnePeriodOLD(self):
        '''
        Simulates one period of the consumption-saving model for all agents
        represented by this instance.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        queue.execute_kernel(self.getMortalityKrn, [self.AgentCount], None)
        queue.execute_kernel(self.getShocksKrn, [self.AgentCount], None)
        queue.execute_kernel(self.getStatesKrn, [self.AgentCount], None)
        queue.execute_kernel(self.getControlsKrn, [self.AgentCount], None)
        queue.execute_kernel(self.getPostStatesKrn, [self.AgentCount], None)
        self.IntegerInputs[4] += 1 # Advance t_sim, else RNG works badly
        queue.write_buffer(self.IntegerInputs_buf,self.IntegerInputs)
        
        
    def simOnePeriodNEW(self):
        '''
        Simulates one period of the consumption-saving model for all agents
        represented by this instance.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        queue.execute_kernel(self.simOnePeriodKrn, [self.AgentCount], None)
        self.IntegerInputs[4] += 1 # Advance t_sim, else RNG works badly
        queue.write_buffer(self.IntegerInputs_buf,self.IntegerInputs)
        
        
        
    def simNperiods(self,N):
        '''
        Simulates N periods of the consumption-saving model for all agents
        represented by this instance.
        
        Parameters
        ----------
        N : int
            Number of periods to simulate.
            
        Returns
        -------
        None
        '''
        for n in range(N):
            self.simOnePeriodNEW()
            
            
            
if __name__ == '__main__':
    import ConsumerParameters as Params
    from ConsIndShockModel import IndShockConsumerType
    import matplotlib.pyplot as plt
    from copy import copy, deepcopy
    from time import clock
    from HARKinterpolation import CubicInterp
    from HARKutilities import plotFuncs
    
    TestType = IndShockConsumerType(**Params.init_lifecycle)
    TestType.TestVar = np.empty(10000)
    TestType.CubicBool = True
    T_sim = 1000
    TestType.T_sim = T_sim
    TestType.initializeSim()
    
    TypeCount = 32
    CRRAmin = 1.0
    CRRAmax = 5.0
    CRRAset = np.linspace(CRRAmin,CRRAmax,TypeCount)
    TypeList = []
    for j in range(TypeCount):
        NewType = deepcopy(TestType)
        NewType(CRRA = CRRAset[j])
        TypeList.append(NewType)
    TestOpenCL = IndShockConsumerTypesOpenCL(TypeList)
    
    TestOpenCL.prepareToSolve()
    t_start = clock()
    TestOpenCL.solve()
    TestOpenCL.loadSimulationKernels()
    TestOpenCL.writeSimVar('aNrmNow')
    TestOpenCL.writeSimVar('pLvlNow')
    t_end = clock()
    print('Solving ' + str(TestOpenCL.TypeCount) + ' types took ' + str(t_end-t_start) + ' seconds with OpenCL.')
    
    t_start = clock()
    TestType.solve()
    t_end = clock()
    print('Solving 1 type took ' + str(t_end-t_start) + ' seconds with Python.')
    
    t_start = clock()
    TestOpenCL.simNperiods(T_sim)
    TestOpenCL.readSimVar('mNrmNow')
    TestOpenCL.readSimVar('cNrmNow')
    TestOpenCL.readSimVar('TestVar')
    t_end = clock()
    print('Simulating ' + str(TestOpenCL.AgentCount) + ' consumers for ' + str(T_sim) + ' periods took ' + str(t_end-t_start) + ' seconds on OpenCL.')
    
#    C_test = np.zeros(TestType.AgentCount)
#    for t in range(TestType.T_cycle+1):
#        these = TestType.TestVar == t
#        C_test[these] = TestType.solution[t].cFunc(TestType.mNrmNow[these])
#    plt.plot(C_test,TestType.cNrmNow,'.k')
#    plt.show()
    
    t_start = clock()
    TestType.simulate()
    t_end = clock()
    print('Simulating ' + str(TestType.AgentCount) + ' consumers for ' + str(T_sim) + ' periods took ' + str(t_end-t_start) + ' seconds on Python.')
    
  