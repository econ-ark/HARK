'''
This module defines the Tractable Buffer Stock model described in CDC's notes.
'''

from HARKcore import AgentType, NullFunc
from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKutilities import CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityPPP, CRRAutilityPPPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv
from HARKinterpolation import Cubic1DInterpDecay
from copy import copy
from scipy.optimize import newton, brentq
#from cstwMPC import cstwMPCagent

# Define utility function and its derivatives (plus inverses)
utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityPPP = CRRAutilityPPP
utilityPPPP = CRRAutilityPPPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv

class TractableConsumerSolution():
    '''
    A class representing the solution to a tractable buffer saving problem.
    Attributes include a list of money points m_list, a list of consumption points
    c_list, a list of MPCs kappa_list, a perfect foresight consumption function
    while employed, and a perfect foresight consumption function while unemployed.
    The solution includes a consumption function constructed from the lists.
    '''
    
    def __init__(self, m_list=[], c_list=[], kappa_list=[], cFunc_U=NullFunc, cFunc=NullFunc):
        '''
        The constructor for a new TractableConsumerSolution object.
        '''       
        self.m_list = m_list
        self.c_list = c_list
        self.kappa_list = kappa_list
        self.cFunc_U = cFunc_U
        self.cFunc = cFunc
        
    def distance(self,other_soln):
        '''
        The distance between two solutions is the difference in the number of
        stable arm points in each.  This is a very crude measure of distance that
        captures the notion that the process is over when no more points are added.
        '''
        return abs(float(len(self.m_list) - len(other_soln.m_list)))


def findNextPoint(beta,R,rho,Gamma,mho,scriptR,Beth,c_tp1,m_tp1,kappa_tp1,kappa_PF):
    uPP = lambda x : utilityPP(x,gam=rho)
    c_t = Gamma*(beta*R)**(-1.0/rho)*c_tp1*(1 + mho*((c_tp1/(kappa_PF*(m_tp1-1.0)))**rho-1.0))**(-1.0/rho)
    m_t = (Gamma/R)*(m_tp1 - 1.0) + c_t
    cU_tp1 = kappa_PF*(m_t-c_t)*scriptR
    natural = Beth*scriptR*(1.0/uPP(c_t))*((1.0-mho)*uPP(c_tp1)*kappa_tp1 + mho*uPP(cU_tp1)*kappa_PF)
    kappa_t = natural / (natural + 1)
    return m_t, c_t, kappa_t
        

def addToStableArmPoints(solution_tp1,beta,R,rho,Gamma,mho,kappa_PF,scriptR,Beth,m_min,m_max):
    '''
    This is the solveAPeriod function for the Tractable Buffer Stock model.  If
    the bounding levels of m_min (lower) and m_max (upper) have not yet been met
    by a stable arm point in m_list, it adds a point to each end of the arm.  It
    is the contents of the "backshooting" loop.
    '''     
    # Unpack the lists of Euler points
    m_list = copy(solution_tp1.m_list)
    c_list = copy(solution_tp1.c_list)
    kappa_list = copy(solution_tp1.kappa_list)
    
    # Check whether to add a stable arm point to the top
    m_tp1 = m_list[-1]
    if m_tp1 < m_max:
        # Get the rest of the data for the previous top point
        c_tp1 = solution_tp1.c_list[-1]
        kappa_tp1 = solution_tp1.kappa_list[-1]
        
        # Calculate employed levels of c, m, and kappa from next period's values
        m_t, c_t, kappa_t = findNextPoint(beta,R,rho,Gamma,mho,scriptR,Beth,c_tp1,m_tp1,kappa_tp1,kappa_PF)
        
        # Add this point to the top of the stable arm list
        m_list.append(m_t)
        c_list.append(c_t)
        kappa_list.append(kappa_t)
    
    # Check whether to add a stable arm point to the bottom
    m_tp1 = m_list[0]
    if m_tp1 > m_min:
        # Get the rest of the data for the previous bottom point
        c_tp1 = solution_tp1.c_list[0]
        kappa_tp1 = solution_tp1.kappa_list[0]
        
        # Calculate employed levels of c, m, and kappa from next period's values
        m_t, c_t, kappa_t = findNextPoint(beta,R,rho,Gamma,mho,scriptR,Beth,c_tp1,m_tp1,kappa_tp1,kappa_PF)
        
        # Add this point to the top of the stable arm list
        m_list.insert(0,m_t)
        c_list.insert(0,c_t)
        kappa_list.insert(0,kappa_t)
        
    # Construct and return this period's solution
    solution_t = TractableConsumerSolution(m_list=m_list, c_list=c_list, kappa_list=kappa_list)
    return solution_t
    

class TractableConsumerType(AgentType):

    def __init__(self,cycles=0,time_flow=False,**kwds):
        '''
        Instantiate a new TractableConsumerType with given data.
        '''       
        # Initialize a basic AgentType
        AgentType.__init__(self,cycles=cycles,time_flow=time_flow,pseudo_terminal=True,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = []
        self.time_inv = ['beta','R','rho','Gamma','mho','kappa_PF','scriptR','Beth','m_min','m_max']
        self.solveAPeriod = addToStableArmPoints
        
    def preSolve(self):
        '''
        This method calculates all of the solution objects that can be obtained
        before conducting the backshooting routine, including the target levels,
        the perfect foresight solution, (marginal) consumption at m=0, and the
        small perturbations around the steady state.
        '''
        # Define utility functions
        uPP = lambda x : utilityPP(x,gam=self.rho)
        uPPP = lambda x : utilityPPP(x,gam=self.rho)
        uPPPP = lambda x : utilityPPPP(x,gam=self.rho)
        
        # Define some useful constants from model primitives
        self.Gamma = self.G/(1.0-self.mho) #"uncertainty compensated" wage growth factor
        self.scriptR = self.R/self.Gamma # net interest factor (R normalized by wage growth)
        self.kappa_PF= 1.0-(self.R**(-1.0))*(self.R*self.beta)**(1.0/self.rho) # MPC for a perfect forsight consumer
        self.Beth = self.scriptR*self.beta*self.Gamma**(1.0-self.rho)
        
        # Verify that this consumer is impatient
        scriptPGrowth = (self.R*self.beta)**(1.0/self.rho)/self.Gamma 
        scriptPReturn = (self.R*self.beta)**(1.0/self.rho)/self.R
        if scriptPReturn >= 1.0:
            raise Exception("Employed consumer not return impatient, cannot solve!")
        if scriptPGrowth >= 1.0:
            raise Exception("Employed consumer not growth impatient, cannot solve!")
            
        # Find target money and consumption
        Pi = (1+(scriptPGrowth**(-self.rho)-1.0)/self.mho)**(1/self.rho)
        self.h = (1.0/(1.0-self.G/self.R))
        zeta = self.scriptR*self.kappa_PF*Pi
        self.m_targ = 1.0+(self.R/(self.Gamma+zeta*self.Gamma-self.R))
        self.c_targ = (1.0-self.scriptR**(-1.0))*self.m_targ+self.scriptR**(-1.0)
        m_targU = (self.m_targ - self.c_targ)*self.scriptR
        c_targU = m_targU*self.kappa_PF
        self.epsilon = self.m_targ*0.1
        
        # Find the MPC, MMPC, and MMMPC at the target
        mpcTargFixedPointFunc = lambda k : k*uPP(self.c_targ) - self.Beth*((1.0-self.mho)*(1.0-k)*k*self.scriptR*uPP(self.c_targ)+self.kappa_PF*self.mho*(1.0-k)*self.scriptR*uPP(c_targU))
        self.kappa_targ = newton(mpcTargFixedPointFunc,0)
        mmpcTargFixedPointFunc = lambda kk : kk*uPP(self.c_targ) + self.kappa_targ**2.0*uPPP(self.c_targ) - self.Beth*(-(1.0 - self.mho)*self.kappa_targ*kk*self.scriptR*uPP(self.c_targ)+(1.0-self.mho)*(1.0 - self.kappa_targ)**2.0*kk*self.scriptR**2.0*uPP(self.c_targ)-self.kappa_PF*self.mho*kk*self.scriptR*uPP(c_targU)+(1.0-self.mho)*(1.0-self.kappa_targ)**2.0*self.kappa_targ**2.0*self.scriptR**2.0*uPPP(self.c_targ)+self.kappa_PF**2.0*self.mho*(1.0-self.kappa_targ)**2.0*self.scriptR**2.0*uPPP(c_targU))
        self.kappaP_targ = newton(mmpcTargFixedPointFunc,0)
        mmmpcTargFixedPointFunc = lambda kkk : kkk * uPP(self.c_targ) + 3 * self.kappa_targ * self.kappaP_targ * uPPP(self.c_targ) + self.kappa_targ**3 * uPPPP(self.c_targ) - self.Beth * (-(1 - self.mho) * self.kappa_targ * kkk * self.scriptR * uPP(self.c_targ) - 3 * (1 - self.mho) * (1 - self.kappa_targ) * self.kappaP_targ**2 * self.scriptR**2 * uPP(self.c_targ) + (1 - self.mho) * (1 - self.kappa_targ)**3 * kkk * self.scriptR**3 * uPP(self.c_targ) - self.kappa_PF * self.mho * kkk * self.scriptR * uPP(c_targU) - 3 * (1 - self.mho) * (1 - self.kappa_targ) * self.kappa_targ**2 * self.kappaP_targ * self.scriptR**2 * uPPP(self.c_targ) + 3 * (1 - self.mho) * (1 - self.kappa_targ)**3 * self.kappa_targ * self.kappaP_targ * self.scriptR**3 * uPPP(self.c_targ) - 3 * self.kappa_PF**2 * self.mho * (1 - self.kappa_targ) * self.kappaP_targ * self.scriptR**2 * uPPP(c_targU) + (1 - self.mho) * (1 - self.kappa_targ)**3 * self.kappa_targ**3 * self.scriptR**3 * uPPPP(self.c_targ) + self.kappa_PF**3 * self.mho * (1 - self.kappa_targ)**3 * self.scriptR**3 * uPPPP(c_targU))
        self.kappaPP_targ = newton(mmmpcTargFixedPointFunc,0)
        
        # Find the MPC at m=0
        f_temp = lambda k : self.Beth*self.scriptR*self.mho*(self.kappa_PF*self.scriptR*((1.0-k)/k))**(-self.rho-1.0)*self.kappa_PF
        mpcAtZeroFixedPointFunc = lambda k : k - f_temp(k)/(1 + f_temp(k))
        #self.kappa_max = newton(mpcAtZeroFixedPointFunc,0.5)
        self.kappa_max = brentq(mpcAtZeroFixedPointFunc,self.kappa_PF,0.99,xtol=0.00000001,rtol=0.00000001)
        
        # Make the initial list of Euler points: target and perturbation to either side
        m_list = [self.m_targ-self.epsilon, self.m_targ, self.m_targ+self.epsilon]
        c_perturb_lo = self.c_targ - self.epsilon*self.kappa_targ + 0.5*self.epsilon**2.0*self.kappaP_targ - (1.0/6.0)*self.epsilon**3.0*self.kappaPP_targ
        c_perturb_hi = self.c_targ + self.epsilon*self.kappa_targ + 0.5*self.epsilon**2.0*self.kappaP_targ + (1.0/6.0)*self.epsilon**3.0*self.kappaPP_targ
        c_list = [c_perturb_lo, self.c_targ, c_perturb_hi]
        kappa_perturb_lo = self.kappa_targ - self.epsilon*self.kappaP_targ + 0.5*self.epsilon**2.0*self.kappaPP_targ
        kappa_perturb_hi = self.kappa_targ + self.epsilon*self.kappaP_targ + 0.5*self.epsilon**2.0*self.kappaPP_targ
        kappa_list = [kappa_perturb_lo, self.kappa_targ, kappa_perturb_hi]
        
        # Set bounds for money (stable arm construction stops when these are exceeded)
        self.m_min = 1.0
        self.m_max = 2.0*self.m_targ
        
        # Make the terminal period solution
        solution_terminal = TractableConsumerSolution(m_list=m_list,c_list=c_list,kappa_list=kappa_list)
        self.solution_terminal = solution_terminal
        
        # Make two linear steady state functions
        self.cSSfunc = lambda m : m*((self.scriptR*self.kappa_PF*Pi)/(1.0+self.scriptR*self.kappa_PF*Pi))
        self.mSSfunc = lambda m : (self.Gamma/self.R)+(1.0-self.Gamma/self.R)*m
                
    def postSolve(self):
        '''
        This method adds consumption at m=0 to the list of stable arm points,
        then constructs the consumption function as a cubic interpolation over
        those points.  Should be run after the backshooting routine is complete.
        '''
        # Add bottom point to the stable arm points
        self.solution[0].m_list.insert(0,0.0)
        self.solution[0].c_list.insert(0,0.0)
        self.solution[0].kappa_list.insert(0,self.kappa_max)
        
        # Construct an interpolation of the consumption function from the stable arm points
        self.solution[0].cFunc = Cubic1DInterpDecay(self.solution[0].m_list,self.solution[0].c_list,self.solution[0].kappa_list,self.kappa_PF*(self.h-1.0),self.kappa_PF)
        self.solution[0].cFunc_U = lambda m : self.kappa_PF*m
        #self.cFunc = self.solution[0].cFunc
        
    def update():
        '''
        This method does absolutely nothing.
        '''
        