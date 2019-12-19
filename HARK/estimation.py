'''
Functions for estimating structural models, including optimization methods
and bootstrapping tools.
'''

# The following libraries are part of the standard python distribution
from __future__ import division                         # Use new division function
from __future__ import print_function
from builtins import str
import numpy as np                                      # Numerical Python
from time import time                                   # Used to time execution
from copy import deepcopy                               # For replicating complex objects
from scipy.optimize import fmin, fmin_powell            # Minimizers
import warnings


def minimizeNelderMead(objectiveFunction, parameter_guess, verbose=False, which_vars=None, **kwargs):
    '''
    Minimizes the objective function using the Nelder-Mead simplex algorithm,
    starting from an initial parameter guess.
    
    Parameters
    ----------
    objectiveFunction : function
        The function to be minimized.  It should take only a single argument, which
        should be a list representing the parameters to be estimated.
    parameter_guess : [float]
        A starting point for the Nelder-Mead algorithm, which must be a valid
        input for objectiveFunction.
    which_vars : np.array or None
        Array of booleans indicating which parameters should be estimated.  When
        not provided, estimation is performed on all parameters.
    verbose : boolean
        A flag for the amount of output to print.
        
    Returns
    -------
    xopt : [float]
        The values that minimize objectiveFunction.
    '''
    # Specify a temporary "modified objective function" that restricts parameters to be estimated
    if which_vars is None:
        which_vars = np.ones(len(parameter_guess),dtype=bool)
    def objectiveFunctionMod(params):
        params_full = np.copy(parameter_guess)
        params_full[which_vars] = params
        out = objectiveFunction(params_full)
        return out
    # convert parameter guess to np array to slice it with boolean array
    parameter_guess_mod = np.array(parameter_guess)[which_vars]

    # Execute the minimization, starting from the given parameter guess
    t0 = time() # Time the process
    OUTPUT = fmin(objectiveFunctionMod, parameter_guess_mod, full_output=1, disp=verbose, **kwargs)
    t1 = time()

    # Extract values from optimization output:
    xopt = OUTPUT[0]        # Parameters that minimize function.
    fopt = OUTPUT[1]        # Value of function at minimum: ``fopt = func(xopt)``.
    optiter = OUTPUT[2]     # Number of iterations performed.
    funcalls = OUTPUT[3]    # Number of function calls made.
    warnflag = OUTPUT[4]    # warnflag : int
                            #   1 : Maximum number of function evaluations made.
                            #   2 : Maximum number of iterations reached.
    # Check that optimization succeeded:
    if warnflag != 0:
        warnings.warn("Minimization failed! xopt=" + str(xopt) + ', fopt=' + str(fopt) + 
                      ', optiter=' + str(optiter) +', funcalls=' + str(funcalls) +
                      ', warnflag=' + str(warnflag))
    xopt_full = np.copy(parameter_guess)
    xopt_full[which_vars] = xopt

    # Display and return the results:
    if verbose:
        print("Time to estimate is " + str(t1-t0) +  " seconds.")
    return xopt_full


def minimizePowell(objectiveFunction, parameter_guess, verbose=False):
    '''
    Minimizes the objective function using a derivative-free Powell algorithm,
    starting from an initial parameter guess.

    Parameters
    ----------
    objectiveFunction : function
        The function to be minimized.  It should take only a single argument, which
        should be a list representing the parameters to be estimated.
    parameter_guess : [float]
        A starting point for the Powell algorithm, which must be a valid
        input for objectiveFunction.
    verbose : boolean
        A flag for the amount of output to print.

    Returns
    -------
    xopt : [float]
        The values that minimize objectiveFunction.
    '''

    # Execute the minimization, starting from the given parameter guess
    t0 = time()   # Time the process
    OUTPUT = fmin_powell(objectiveFunction, parameter_guess, full_output=1, maxiter=1000, disp=verbose)
    t1 = time()

    # Extract values from optimization output:
    xopt = OUTPUT[0]        # Parameters that minimize function.
    fopt = OUTPUT[1]        # Value of function at minimum: ``fopt = func(xopt)``.
    direc = OUTPUT[2]
    optiter = OUTPUT[3]     # Number of iterations performed.
    funcalls = OUTPUT[4]    # Number of function calls made.
    warnflag = OUTPUT[5]    # warnflag : int
    #                           1 : Maximum number of function evaluations made.
    #                           2 : Maximum number of iterations reached.

    # Check that optimization succeeded:
    if warnflag != 0:
        warnings.warn("Minimization failed! xopt=" + str(xopt) + ', fopt=' + str(fopt) + ', direc=' + str(direc) +
                      ', optiter=' + str(optiter) + ', funcalls=' + str(funcalls) + ', warnflag=' + str(warnflag))

    # Display and return the results:
    if verbose:
        print("Time to estimate is " + str(t1-t0) + " seconds.")
    return xopt


def bootstrapSampleFromData(data, weights=None, seed=0):
    '''
    Samples rows from the input array of data, generating a new data array with
    an equal number of rows (records).  Rows are drawn with equal probability
    by default, but probabilities can be specified with weights (must sum to 1).

    Parameters
    ----------
    data : np.array
        An array of data, with each row representing a record.
    weights : np.array
        A weighting array with length equal to data.shape[0].
    seed : int
        A seed for the random number generator.

    Returns
    -------
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
        cutoffs = np.linspace(0, 1, N)

    # Draw random indices
    indices = np.searchsorted(cutoffs, RNG.uniform(size=N))

    # Create a bootstrapped sample
    new_data = deepcopy(data[indices, ])
    return new_data


def main():
    print("Sorry, HARK.estimation doesn't actually do anything on its own.")
    print("To see some examples of its functions in actions, check out an application")
    print("like /SolvingMicroDSOPs/StructEstimation or /cstwMPC/cstwMPC.")


if __name__ == '__main__':
    main()
