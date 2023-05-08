# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## A Description of the Gothic class
# This notebook briefly outlines how the "Gothic" class operates.
#
# The Gothic class to define the end-of-period "gothic" functions: $\mathfrak{v}$, $\mathfrak{v}'$, and $\mathfrak{c}$, as well as the interpolations of each of these functions. 
#     
# Defining these in one class allows us to bundle the parameters for the problem in one place, and then hide them from the user. We have likewise bundled the parameters for the utility function and the discrete distribution approximation in their own classes. The class structure additionally allows us to bundle useful fucnitonality with the utility function and discrete distribution, such as the marginal utility in the utility class, and the expectation operator associated with the discrete distribution. The layers of abstraction provided by the object-oriented framework allow us to use the bare minimum additional parameters for each level of the code. See the notebook regarding these classes for further explanation.
#
# We define a Gothic object with a utility function $u, \beta,$ the risk parameter $\rho, \gamma,$ R, and $\theta$.
#
# Once initilized, we will have access to these methods in the Gothic class:
#
#
#         V_Tminus1:              GothicV at {T-1}, in levels
#     
#         VP_Tminus1:             GothicV\' at {T-1}, in marginal values
#         
#         V_Tminus1_interp, and 
#         VP_Tminus1_interp:      Both above, interpolated on an a-grid
#     
#     Usage:
#     
#         Gothic.V_Tminus1(a):    Return the gothicV value for a at T-1.
#         Gothic.VP_Tminus1(a):   Return the gothicV\' value for a at T-1.
#         
#         Gothic.V_Tminus1_interp(a_grid): Return gothicV(a) as an interpolated 
#                                          function, interpolated on the a_grid 
#                                          provided.
#         Gothic.VP_Tminus1_interp(a_grid):   As above, return gothicV\'(a) as
#                                             an interpolation function on the 
#                                             a_grid.
#
#         Gothic.C_Tminus1(a):    Return the gothicC value for a at T-1.
#
#         Gothic.C_Tminus1_interp(a_grid): Return gothicC(a) as an interpolated 
#                                          function, interpolated on the a_grid 
#                                          provided.
#
# ## The Gothic class:

from __future__ import division
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np

# + code_folding=[2, 144]
class Gothic:
    
    def __init__(self, u, beta, rho, Gamma, R, Income, variable_variance=False):
        """
        Initialize a Gothic object. 

        Args:
            u (object):     Utility function. Should accept a real number & have
                            a "prime" method which is the first derivative.
            beta (float):   Time discount factor.
            rho (float):    Risk aversion.
            gamma (array):  Array of gamma values, time series indexed by t.
            R (float):      The real return factor. Fixed in time. 
            Income (object): Approximated distribution for a two-shock method. 
                             Must have method: "Income.E()." 
                             NOTE: The convention is that permanent shock to 
                             incom (psi) comes first, and the temporary shock
                             (eta) comes second in the ordered pair of the
                             shocks to income. Any function of which we need 
                             to find an expectation, with respect to income, 
                             should be defined as such.
            variable_variance (boolean):  If true, the Income is a list of 
                                          income objects.
        Returns:
            Nothing.
        Raises:
            []
        """
        self.u = u
        self.beta = beta
        self.rho = rho
        self.Gamma = Gamma
        self.Gamma_to_1minusRho = Gamma ** (1.0 - rho)  # Define here once.   
        self.Gamma_to_minusRho = Gamma ** (-rho)        # Define here once.   
        self.R = R
        self.Income = Income
        self.variable_variance = variable_variance


        
    def V(self, a, t=-1, v_prime=None):
        """
        Given an end-of-period a value, return the GothicV_{T-1} value. 
        For t = None, implements equation (22) from MicroDSOP: value function at T-1 
        For t != None, v_prime != None, implements equation (17) from MicroDSOP. 
        """
        
        # Define function describing tomorrow:
        if t == -1:
            tp1 = -1    # Selects final value in a vector.
            t = -2
            V_func = lambda tinc_shk:  self.u(self.R/(self.Gamma[tp1]) * a + tinc_shk) 
        elif v_prime is not None:
            tp1 = t + 1
            V_func = lambda tinc_shk:  v_prime(self.R/(self.Gamma[tp1]) * a + tinc_shk) 
        else:
            raise Exception("Please either specify that t=-1 (indicating solution for period T-1) or specify *both* t and v_prime.")
        
        if self.variable_variance:
            gothicV = self.beta * self.Gamma_to_1minusRho[tp1] * self.Income[tp1].E(V_func)
            # TODO: confirm that 
        else:
            gothicV = self.beta * self.Gamma_to_1minusRho[tp1] * self.Income.E(V_func)
            
        return(gothicV)

    
    def V_prime(self, a, t=-1, c_prime=None):
        """
        Given an end-of-period a-value, return the GothicV_prime value. 
        If t=-1, return T-1 value; else return the t-value.
        
        This implements equation (19) and (30) from MicroDSOP for T-1, and 
        equation (18) for all previous time periods. 
        """        

        if t == -1:
            tp1 = -1    # Selects final value in a vector.
            t = -2
            Vp_func = lambda tinc_shk:  psi**(-self.rho) * self.u.prime(self.R/(self.Gamma[tp1]) * a + tinc_shk) 
        elif c_prime is not None:
            tp1 = t+1
            
            #mtp1 = self.R/(self.Gamma[tp1]*psi) * a + eta
            #print "mtp1", mtp1
            #g = lambda psi, eta:  psi**(-self.rho) * self.u.prime(c_prime(mtp1))
            # one possible solution:
            Vp_func = lambda tinc_shk, R=self.R, gamma=self.Gamma[tp1],aa=a, rho=self.rho, uP=self.u.prime, ctp1=c_prime:  uP(ctp1(R/(gamma) * aa + tinc_shk))
            
        else:
            raise Exception("Please either specify that t=-1 (indicating solution for period T-1) or specify *both* t and c_prime.")
        
        if self.variable_variance:
            gothicV_prime = self.beta * self.R * self.Gamma_to_minusRho[tp1] * self.Income[tp1].E(Vp_func)
        else:
            gothicV_prime = self.beta * self.R * self.Gamma_to_minusRho[tp1] * self.Income.E(Vp_func)
        
        return(gothicV_prime)
    
    
    def C(self, a, t=-1, c_prime=None):
        """
        Return the gothicC value for a. If t=-1, return the value for T-1. 
        
        Implements equation (34) in MicroDSOP for T-1; implements equation (20) 
        for all other periods. 
        """
        
        if t == -1:
            scriptC = self.V_prime(a,t=-1)**(-1.0/self.rho)
        elif c_prime is not None:
            scriptC = self.V_prime(a, t=t, c_prime=c_prime)**(-1.0/self.rho)
        else:
            raise Exception("Please either specify that t=-1 (indicating solution for period T-1) or specify *both* t and c_prime.")
        
        return(scriptC)
    
    # copied from ./Code/Python/active_development/archive/Gothic Class 1shock.ipynb
    def C_Tminus1(self, a):
        """
        Return the gothicC value for a at T-1. Equation (34) in MicroDSOP.
        """
        return self.VP_Tminus1(a)**(-1.0/self.rho)

    # copied from ./Code/Python/active_development/archive/Gothic Class 1shock.ipynb
    # changed Theta -> Income
    def VP_Tminus1(self, a):
        """
        Given an end-of-period a-value, return the GothicV_prime_Tminus1 value. 
        Vectorize to work on a grid. 
        
        This implements function (30) from MicroDSOP.
        """        
        # Convenience definitions. Note we take the last value of Gamma:
        fancyR_T = self.R/self.Gamma[-1] 
        
        Vp_func = lambda tinc_shk: self.u.prime(fancyR_T * a + tinc_shk)
        # The value:
        GVTm1P = self.beta * self.R * self.Gamma_to_minusRho[-1] * self.Income.E(Vp_func)

        return GVTm1P

    # copied from ./Code/Python/active_development/archive/Gothic Class 1shock.ipynb
    # changed Theta -> Income
    def C_t(self, a, c_prime, t=None):
        """
        Return the gothicC value for a at t. 

        This employs Equation (20) in MicroDSOP.
        """
        # Quick comparison test against hand-coded equation (76):
        
        if t is None:
            t = -1
        
        E_sum = 0.0
        for theta in self.Income.X:
            fancyR_tp1 = self.R/self.Gamma[t+1]
            c_tp1 = c_prime(fancyR_tp1*a + theta)
            
            E_sum += c_tp1**(-self.rho)
        
        alt_scriptC = (self.beta * self.R * (self.Gamma[t+1] ** (-self.rho)) * (1.0/self.Income.N) * E_sum) ** (-1.0/self.rho)
                
        scriptC = self.VP_t(a, c_prime, t)**(-1.0/self.rho)
        
        #print "alt_scriptC", alt_scriptC
        #print "scriptC", scriptC
 
        tempdiff = alt_scriptC - scriptC
        assert np.abs(tempdiff) < 1e-10, "in Gothic.C_t, manually calculated scriptC(a) != computed scriptC, by this much: " + str(tempdiff) + " values: alt_scriptC: " + str(alt_scriptC) + " scriptC: " + str(scriptC) 
        
        return scriptC

    # copied from ./Code/Python/active_development/archive/Gothic Class 1shock.ipynb
    # changed Theta -> Income
    def VP_t(self, a, c_prime, t=None):
        """
        Given a next-period consumption function, find the Vprime function for this period. 
        
        This implements function (__) from MicroDSOP.
        """        
        
        if t is None:
            Gamma_to_mRho = self.Gamma_to_minusRho[0]
            scriptR_tp1 = self.R/self.Gamma[0]
        else:
            Gamma_to_mRho = self.Gamma_to_minusRho[t+1]
            scriptR_tp1 = self.R/self.Gamma[t+1]
        
        Vp_func = lambda tinc_shk: self.u.prime(c_prime(scriptR_tp1 * a + tinc_shk))
        # The value:
        GVPt = self.beta * self.R * Gamma_to_mRho * self.Income.E(Vp_func)
        
        return GVPt
# -





# ### Demonstrating Functionality
#
# First import and define a number of items needed:

# In[2]:

if __name__ == "__main__":
    
    # Only execute the demonstrations if this is the main file; 
    # do not run when this is imported.
    import numpy as np
    import scipy.stats as stats
    import pylab as plt
    from scipy.optimize import brentq
    from resources import DiscreteApproximation, Utility, DiscreteApproximationTwoMeanOneIndependentLognormalDistribs, DiscreteApproximationToTwoMeanOneIndependentLognormalDistribsWithDiscreteProb_Z_Event
    
    # General parameters:
    rho = 2.0
    beta = 0.96
    gamma = np.array([1.0,1.0,1.0]) # A three-element "time series;" this
                            # structure needed for gothic class below 
    R = 1.02
    
    # Define discrete approximation:
    sigma = 1.0
    #mu = -0.5*(sigma**2)
    #z = stats.lognorm(sigma, 0, np.exp(mu))    # Create "frozen" distribution instance
    theta_grid_N = 7
    
    sigma2 = 1.0
    N2 = 7
    p0 = 0.01
    
    # Create a discrete approximation instance:
    #theta = DiscreteApproximation(N=theta_grid_N, cdf=z.cdf, pdf=z.pdf, invcdf=z.ppf)
    income = DiscreteApproximationTwoMeanOneIndependentLognormalDistribs(
                N1=theta_grid_N, sigma1=sigma, N2=N2, sigma2=sigma2)
    #DiscreteApproximationToTwoMeanOneIndependentLognormalDistribsWithDiscreteProb_Z_Event(
    #            N1=theta_grid_N, sigma1=sigma, N2=N2, sigma2=sigma2, pZevent=p0, z=0.0)
    
    # M grid:
    m_min, m_max, m_size = 0.0, 4.0, 5          # Assign multiple values on a line
    m_grid = np.linspace(m_min, m_max, m_size)
    
    # Set up a-grid:
    a_min, a_max, a_size = 0.0, 4.0, 5
    a_grid = np.linspace(a_min, a_max, a_size)
    
    self_a_min = -min(income.X2) * R/gamma[0]     # Self-imposed minimum a
    self_c_min = min(m_grid) - self_a_min       # Self-imposed min c
    
    # Define utility:
    u = Utility(gamma=rho)
    
    
    # Create a Gothic object with these specific parameters:
    gothic = Gothic(u, beta, rho, gamma, R, income)

    


# ### Plot some of the functions:
#
# Examine consumption functions.

# In[3]:

if __name__ == "__main__":
    # Examine the GothicC function:
    #f = gothic.C_Tminus1_interp(a_grid, self_a_min)
    
    temp_a_grid = [self_a_min] + [a for a in a_grid]
    c_grid = [0.0]
    m_grid = [self_a_min]
    for a in a_grid:
        c = gothic.C(a, t=-1)
        m = a + c
        c_grid.append(c)
        m_grid.append(m)
            
    
    # Define a consumption function: 
    c_prime = InterpolatedUnivariateSpline(m_grid, c_grid, k=1)
    plt.plot(m_grid, c_grid, 'g-')
    plt.show()

    # Examine the GothicC function for (t != T-1):
    c_grid2 = [0.0]
    m_grid2 = [self_a_min]    # This needs to be ... falling back? 
                              # because each period can potentially be borrowing 
                              # more?
    for a in a_grid:
        c = gothic.C(a, t=0, c_prime=c_prime)
        m = a + c
        c_grid2.append(c)
        m_grid2.append(m)

    c_prime2 = InterpolatedUnivariateSpline(m_grid2, c_grid2, k=1)

    plt.plot(m_grid, c_grid, 'g-')
    plt.plot(m_grid2, c_grid2, 'r--')
    plt.title("Consumption for T-1 and T-2")
    plt.show()

    # Examine the GothicC function for (t != T-1):
    c_grid3 = [0.0]
    m_grid3 = [self_a_min]    # This needs to be ... falling back? 
                              # because each period can potentially be borrowing 
                              # more?
    for a in a_grid:
        c = gothic.C(a, t=0, c_prime=c_prime2)
        m = a + c
        c_grid3.append(c)
        m_grid3.append(m)

    
    plt.plot(m_grid, c_grid, 'g-')
    plt.plot(m_grid2, c_grid2, 'r--')
    plt.plot(m_grid3, c_grid3, 'b:')
    plt.title("Consumption for T-1, T-2, and T-3")
    plt.show()



# In[8]:




# ## We will see that the $\mathfrak{v}$ and $\mathfrak{v}'$ replicate desired values. 

# In[ ]:

# Code saved for possible future use:
'''
if __name__ == "__main__":
    # Examine the GothicV function:
    big_a_grid = np.linspace(0,4, 100)
    f = gothic.V_Tminus1_interp(a_grid)
    vals = [gothic.V_Tminus1(a) for a in a_grid]
    f2 = gothic.V_Tminus1_interp(big_a_grid)
    #plt.plot(a_grid, f(a_grid), 'r--')   
      # NOTE: the in-class interpolation method is not working quite right.
      # Only use "external" interpolation to solve for consumption functions.
    plt.plot(a_grid, vals, 'r--')
    plt.plot(big_a_grid, f2(big_a_grid), 'k-')
    plt.ylim(-2, 0.1)
    plt.show()
    print(gothic.V_Tminus1(1.0))
    print(f(1  +0.00001))               # Note: The interpolation is the issue. Look into. 
    print(f(1.0+0.000000000000000001))

    
    # Examine the GothicV' function:
    big_a_grid = np.linspace(0,4, 100)
    #f = gothic.VP_Tminus1_interp(a_grid)
    vals = [gothic.VP_Tminus1(a) for a in a_grid]
    f2 = gothic.VP_Tminus1_interp(big_a_grid)
    plt.plot(a_grid, vals, 'r--')
    plt.plot(big_a_grid, f2(big_a_grid), 'k-')
    plt.ylim(0.0, 1.0)
    plt.show()
'''
'''
    def V_Tminus1_interp(self, a_grid):
        """
        Given an grid of end-of-period a values, return the GothicV_{T-1} 
        function interpolated between these a_grid points. 
        
        This implements function (22) from MicroDSOP, interpolated across a_grid.

        **NOTE: currently a bug here. Need to find. For now find externally.
        """
        values = [self.V_Tminus1(a) for a in a_grid]
        return InterpolatedUnivariateSpline(a_grid, values, k=1)


    def VP_Tminus1_interp(self, a_grid):
        """
        Given a grid of end-of-period a-values, return the GothicV'_{T-1} 
        function interpolated between the points on a_grid. 
        
        This implements function (30) from MicroDSOP, interpolated across a_grid.

        **NOTE: currently a bug here. Need to find. For now find externally.
        """        
        values = [self.VP_Tminus1(a) for a in a_grid]
        return InterpolatedUnivariateSpline(a_grid, values, k=1)

   def C_Tminus1_interp(self, a_grid, a_min=None):
        """
        NOTE: not used in main program. Retained for future use.
        
        Return the gothicC value interpolated across the a-grid.
        
        a_min here refers to the a_underbar_{T-1} value in section 5.7. Recall
        that:
                a_underbar_{T-1} = -theta_underbar/fancyR_T,
                
        that is, the min PDV of income in period T. That is:
            
                fancy_R_T * a_underbar_Tminus1 = -theta_1. 
                
        When we provide a_min, it must be the correct a_min.
        """
        if a_min is not None:
            a_grid = np.append(a_min, a_grid)
            Y = [self.C_Tminus1(a) for a in a_grid]
            Y[0] = 0.0
        else:
            Y = [self.C_Tminus1(a) for a in a_grid]
        return InterpolatedUnivariateSpline(a_grid, Y, k=1)     
'''

