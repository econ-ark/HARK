'''
This module contains functions useful for estimating structural models, including
optimization methods and bootstrapping tools.
'''

# The following libraries are part of the standard python distribution
from __future__ import division                         # Use new division function
import numpy as np                                      # Numerical Python
from time import time                                   # Used to time execution
from copy import deepcopy
from scipy.optimize import fmin, fmin_powell, brute     # Minimizers
from HARKutilities import warnings                      # Import modified "warnings" library

def minimizeNelderMead(objectiveFunction, parameter_guess, verbose=False, **kwargs):
    '''
    Minimizes the objective function using the Nelder-Mead simplex algorithm,
    starting from an initial parameter guess.
    
    Parameters:
    -----------
    objectiveFunction : function
        The function to be minimized.  It should take only a single argument, which
        should be a list representing the parameters to be estimated.
    parameter_guess : [float]
        A starting point for the Nelder-Mead algorithm, which must be a valid
        input for objectiveFunction.
    verbose : boolean
        A flag for the amount of output to print.
        
    Returns:
    ----------
    xopt : [float]
        The values that minimize objectiveFunction.
    '''

    # Execute the minimization step using initial values from the parameters file.
    # Time the process.
    t0 = time()
    OUTPUT = fmin(objectiveFunction, parameter_guess, full_output=1, maxiter=1000, disp=verbose, **kwargs)
    t1 = time()

    # Extract values from optimization output:
    xopt = OUTPUT[0]        # Parameter that minimizes function.
    fopt = OUTPUT[1]        # Value of function at minimum: ``fopt = func(xopt)``.
    optiter = OUTPUT[2]     # Number of iterations performed.
    funcalls = OUTPUT[3]    # Number of function calls made.
    warnflag = OUTPUT[4]    # warnflag : int
                            #   1 : Maximum number of function evaluations made.
                            #   2 : Maximum number of iterations reached.
    # Check that optimization succeeded:
    if warnflag != 0:
        warnings.warn("Minimization failed! xopt=" + str(xopt) + ', fopt=' + str(fopt) + ', optiter=' + str(optiter) +', funcalls=' + str(funcalls) +', warnflag=' + str(warnflag))

    # Display the results:
    if verbose:
        print("Time to estimate is " + str(t1-t0) +  " seconds.")

    return xopt
    
    
def minimizePowell(objectiveFunction, parameter_guess, verbose=False):
    '''
    Minimizes the objective function using derivative-free Powell algorithm,
    starting from an initial parameter guess.
    
    Parameters:
    -----------
    objectiveFunction : function
        The function to be minimized.  It should take only a single argument, which
        should be a list representing the parameters to be estimated.
    parameter_guess : [float]
        A starting point for the Powell algorithm, which must be a valid
        input for objectiveFunction.
    verbose : boolean
        A flag for the amount of output to print.
        
    Returns:
    ----------
    xopt : [float]
        The values that minimize objectiveFunction.
    '''

    # Execute the minimization step using initial values from the parameters file.
    # Time the process.
    t0 = time()
    OUTPUT = fmin_powell(objectiveFunction, parameter_guess, full_output=1, maxiter=1000, disp=verbose)
    t1 = time()

    # Extract values from optimization output:
    xopt = OUTPUT[0]        # Parameter that minimizes function.
    fopt = OUTPUT[1]        # Value of function at minimum: ``fopt = func(xopt)``.
    direc = OUTPUT[2]
    optiter = OUTPUT[3]     # Number of iterations performed.
    funcalls = OUTPUT[4]    # Number of function calls made.
    warnflag = OUTPUT[5]    # warnflag : int
                            #   1 : Maximum number of function evaluations made.
                            #   2 : Maximum number of iterations reached.
    # Check that optimization succeeded:
    if warnflag != 0:
        warnings.warn("Minimization failed! xopt=" + str(xopt) + ', fopt=' + str(fopt) + ', direc=' + str(direc) + ', optiter=' + str(optiter) +', funcalls=' + str(funcalls) +', warnflag=' + str(warnflag))

    # Display the results:
    if verbose:
        print("Time to estimate is " + str(t1-t0) +  " seconds.")

    return xopt


def bootstrapSampleFromData(data,weights=None,seed=0):
    '''
    Samples rows from the input array of data, generating a new data array with
    an equal number of rows (records).  Rows are drawn with equal probability
    by default, but probabilities can be specified with weights (must sum to 1).
    
    Parameters:
    -----------
    data : np.array
        An array of data, with each row representing a record.
    seed : int
        A seed for the random number generator.
        
    Returns:
    -----------
    new_data : np.array
        A resampled version of input data.
    '''
    
    # Set up the random number generator
    RNG = np.random.RandomState(seed)
    N = data.shape[0]
    
    # Set up weights
    if weights is not None:
        cutoffs = np.cumsum(weights)
    else:
        cutoffs = np.linspace(0,1,N)
    
    # Draw random indices
    
    #indices_temp = np.floor(N*RNG.uniform(size=N))
    #indices = indices_temp.astype(int)
    indices = np.searchsorted(cutoffs,RNG.uniform(size=N))
    
    # Create a bootstrapped sample
    new_data = deepcopy(data[indices,])
    return new_data
    
