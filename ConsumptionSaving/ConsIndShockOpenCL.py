'''
This module provides functions and classes for implementing ConsIndShockModel
on OpenCL, enabling the use of GPU computing.
'''

import os
import numpy as np
import opencl4py as cl
os.environ["PYOPENCL_CTX"] = "2" # This is where you choose a device number

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
        self.IntegerInputs = np.zeros(6,dtype=int)
        self.IntegerInputs[1] = self.TypeCount
        self.IntegerInputs_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,self.IntegerInputs)
        
        self.makeParameterBuffers()  # Make buffers with primitive and constructed parameters
        self.makeSimulationBuffers() # Make buffers for holding current simulated variables
        has_solved = [hasattr(agents[j],'solution') for j in range(self.TypeCount)]
        if np.all(np.array(has_solved)):
            self.makeSolutionBuffers() # Make buffers for the consumption function            
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
        NormDraws_vec = np.random.randn(65536)
        self.NormDraws_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,NormDraws_vec)
        self.IntegerInputs[0] = AgentCount
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
            
            
            
if __name__ == '__main__':
    import ConsumerParameters as Params
    from ConsIndShockModel import IndShockConsumerType
    import matplotlib.pyplot as plt
    from copy import copy
    
    TestType = IndShockConsumerType(**Params.init_lifecycle)
    TestType.CubicBool = True
    TestType.solve()
    TestOpenCL = IndShockConsumerTypesOpenCL([TestType])
    
    TestType.T_sim = 10
    TestType.initializeSim()
    TestType.simulate(1)
    
    TestOpenCL.writeSimVar('mNrmNow')
#    TestOpenCL.writeSimVar('cNrmNow')
    TestOpenCL.writeSimVar('pLvlNow')
    TestOpenCL.loadSimulationKernels()
    
    queue.execute_kernel(TestOpenCL.getControlsKrn, [TestOpenCL.AgentCount], None)
    X = copy(TestType.cNrmNow)
    Y = copy(TestType.MPCnow)
    TestOpenCL.readSimVar('cNrmNow')
    TestOpenCL.readSimVar('MPCnow')
    plt.plot(np.sort(TestType.cNrmNow - X))
    plt.show()
    

#    queue.execute_kernel(TestOpenCL.getPostStatesKrn, [TestOpenCL.AgentCount], None)
#    queue.execute_kernel(TestOpenCL.getMortalityKrn, [TestOpenCL.AgentCount], None)
#    queue.execute_kernel(TestOpenCL.getShocksKrn, [TestOpenCL.AgentCount], None)
#    queue.execute_kernel(TestOpenCL.getStatesKrn, [TestOpenCL.AgentCount], None)
#    
#    X = copy(TestType.pLvlNow)
#    TestOpenCL.readSimVar('pLvlNow')
#    plt.plot(np.sort(TestType.pLvlNow/X))
#    plt.show()
    