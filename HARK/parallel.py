'''
Early version of multithreading in HARK. To use most of this module, you should first install dill
and joblib.  Packages can be installed by typing "conda install dill" (etc) at
a command prompt.
'''
from __future__ import division, print_function
from builtins import next
from builtins import zip
from builtins import str
from builtins import range
import multiprocessing
import numpy as np
from time import perf_counter
import csv


# We want to be able to import this module even if joblib and dill are not installed.
# If we can't import joblib and dill, define the functions we tried to import
# such that they will raise useful errors if called.
def raiseImportError(moduleStr):
    def defineImportError(*args, **kwargs):
        raise ImportError(moduleStr + ' could not be imported, and is required for this'+\
        ' function.  See HARK documentation for more information on how to install the ' \
        + moduleStr + ' module.')
    return defineImportError

try:
    # Try to import joblib
    from joblib import Parallel, delayed
except:
    print("Warning: Could not import joblib.")
    Parallel = raiseImportError('joblib')
    delayed  = raiseImportError('joblib')

try:
    # Try to import dill
    import dill as pickle
except:
    print("Warning: Could not import dill.")
    pickle   = raiseImportError('dill')


def multiThreadCommandsFake(agent_list,command_list,num_jobs=None):
    '''
    Executes the list of commands in command_list for each AgentType in agent_list
    in an ordinary, single-threaded loop.  Each command should be a method of
    that AgentType subclass.  This function exists so as to easily disable
    multithreading, as it uses the same syntax as multithreadCommands.

    Parameters
    ----------
    agent_list : [AgentType]
        A list of instances of AgentType on which the commands will be run.
    command_list : [string]
        A list of commands to run for each AgentType.
    num_jobs : None
        Dummy input to match syntax of multiThreadCommands.  Does nothing.

    Returns
    -------
    none
    '''
    for agent in agent_list:
        for command in command_list:
            exec('agent.' + command)

def multiThreadCommands(agent_list,command_list,num_jobs=None):
    '''
    Executes the list of commands in command_list for each AgentType in agent_list
    using a multithreaded system. Each command should be a method of that AgentType subclass.

    Parameters
    ----------
    agent_list : [AgentType]
        A list of instances of AgentType on which the commands will be run.
    command_list : [string]
        A list of commands to run for each AgentType in agent_list.

    Returns
    -------
    None
    '''
    if len(agent_list) == 1:
        multiThreadCommandsFake(agent_list,command_list)
        return None

    # Default number of parallel jobs is the smaller of number of AgentTypes in
    # the input and the number of available cores.
    if num_jobs is None:
        num_jobs = min(len(agent_list),multiprocessing.cpu_count())

    # Send each command in command_list to each of the types in agent_list to be run
    agent_list_out = Parallel(n_jobs=num_jobs)(delayed(runCommands)(*args) for args in zip(agent_list, len(agent_list)*[command_list]))

    # Replace the original types with the output from the parallel call
    for j in range(len(agent_list)):
        agent_list[j] = agent_list_out[j]

def runCommands(agent,command_list):
    '''
    Executes each command in command_list on a given AgentType.  The commands
    should be methods of that AgentType's subclass.

    Parameters
    ----------
    agent : AgentType
        An instance of AgentType on which the commands will be run.
    command_list : [string]
        A list of commands that the agent should run, as methods.

    Returns
    -------
    agent : AgentType
        The same AgentType instance passed as input, after running the commands.
    '''
    for command in command_list:
        exec('agent.' + command)
    return agent


#=============================================================
# ========  Define a parallel Nelder-Mead algorithm ==========
#=============================================================

def parallelNelderMead(objFunc,guess,perturb=None,P=1,ftol=0.000001,xtol=0.00000001,maxiter=np.inf,maxeval=np.inf,r_param=1.0,e_param=1.0,c_param=0.5,s_param=0.5,maxcores=None,name=None,resume=False,savefreq=None,verbose=1):
    '''
    A parallel implementation of the Nelder-Mead minimization algorithm, as
    described in Lee and Wiswall.  For long optimization procedures, it can
    save progress between iterations and resume later.

    Parameters
    ----------
    objFunc : function
        The objective function to be minimized. Takes a single 1D array as input.
    guess : np.array
        Initial starting point for the simplex, representing an input for objFunc.
    perturb : np.array
        Perturbation vector for the simplex, of the same length as an input to
        objFunc.  If perturb[j] is non-zero, a simplex point will be created
        that perturbs the j-th element of guess by perturb[j]; if it is zero,
        then the j-th parameter of objFunc will not be optimized over.  By
        default, guess=None, indicating that all parameters should be optimized,
        with an initial perturbation of 0.1*guess.
    P : int
        Degree of parallelization: the number of vertices of the simplex to try
        to update on each iteration of the process.
    ftol : float
        Absolute tolerance of the objective function for convergence.  If suc-
        cessive iterations return minimum function values that differ by less
        than ftol, the process terminates successfully.
    xtol : float
        Absolute tolerance of the input values for convergence.  If the maximum
        distance between the current minimum point and the worst point in the
        simplex is less than xtol, then the process terminates successfully.
    maxiter : int
        Maximum number of Nelder-Mead iterations; reaching iters=maxiter is
        reported as an "unsuccessful" minimization.
    maxeval : int
        Maximum number of evaluations of objFunc (across all processes); reaching
        evals=maxeval is reported as an "unsuccessful" minimization.
    r_param: float
        Parameter indicating magnitude of the reflection point calculation.
    e_param: float
        Parameter indicating magnitude of the expansion point calculation.
    c_param: float
        Parameter indicating magnitude of the contraction point calculation.
    s_param: float
        Parameter indicating magnitude of the shrink calculation.
    maxcores : int
        The maximum number of CPU cores that the optimization should use,
        regardless of the size of the problem.
    name : string
        A filename for (optionally) saving the progress of the Nelder-Mead search,
        and for resuming a previous search (when resume=True).  Useful for long
        searches that could potentially be interrupted by computer down time.
    resume : boolean
        An indicator for whether the search should resume from earlier progress.
        When True, the process will load a progress file named in input name.
    savefreq : int
        When not None, search progress will be saved to name.txt every savefreq
        iterations, to be loaded later with resume=True).
    verbose : int
        Indicator for the verbosity of the optimization routine.  Higher values
        generate more text output; verbose=0 produces no text output.

    Returns
    -------
    min_point : np.array
        The input that minimizes objFunc, as found by the minimization.
    fmin : float
        The minimum of objFunc; fmin = objFunc(min_point).
    '''
    # If this is a resumed search, load the data
    if resume:
        simplex, fvals, iters, evals = loadNelderMeadData(name)
        dim_count = fvals.size - 1
        N = dim_count+1 # Number of points in simplex
        K = simplex.shape[1] # Total number of parameters
    # Otherwise, construct the initial simplex and array of function values
    else:
        if perturb is None: # Default: perturb each parameter by 10%
            perturb = 0.1*guess
            guess[guess == 0] = 0.1
        params_to_opt = np.where(perturb != 0)[0] # Indices of which parameters to optimize
        dim_count = params_to_opt.size # Number of parameters to search over
        N = dim_count+1 # Number of points in simplex
        K = guess.size # Total number of parameters
        simplex = np.tile(guess,(N,1))
        for j in range(dim_count): # Perturb each parameter to optimize by the specified distance
            simplex[j+1,params_to_opt[j]] = simplex[j+1,params_to_opt[j]] + perturb[params_to_opt[j]]
        # Initialize a few
        fvals = np.zeros(dim_count+1) + np.nan
        iters = 0
        evals = 0

    # Create the pool of worker processes
    cpu_cores = multiprocessing.cpu_count() # Total number of available CPU cores
    cores_to_use = min(cpu_cores,dim_count)
    if maxcores is not None: # Cap the number of cores if desired
        cores_to_use = min(cores_to_use,maxcores)
    parallel = Parallel(n_jobs=cores_to_use)

    # Begin a new Nelder-Mead search
    if not resume:
        temp_simplex = list(simplex) # Evaluate the initial simplex
        fvals = np.array(parallel(delayed(objFunc)(params) for params in temp_simplex))
        evals += N
        # Reorder the initial simplex
        order = np.argsort(fvals)
        fvals = fvals[order]
        simplex = simplex[order,:]
        fmin = fvals[0]
        f_dist = np.abs(fmin - fvals[-1])
        x_dist = np.max(np.sqrt(np.sum(simplex**2.0 - np.tile(simplex[0,:],(N,1))**2.0,axis=1)))
        if verbose > 0:
            print('Evaluated the initial simplex: fmin=' + str(fmin) + ', f_dist=' + str(f_dist) + ', x_dist=' + str(x_dist))
    else: # Resume an existing search that was cut short
        if verbose > 0:
            print('Resuming search after ' + str(iters) + ' iterations and ' + str(evals) + ' function evaluations.')

    # Initialize some inputs for the multithreader
    j_list = list(range(N-P,N))
    opt_params= [r_param,c_param,e_param]

    # Run the Nelder-Mead algorithm until a terminal condition is met
    go = True
    while go:
        t_start = perf_counter()
        iters += 1
        if verbose > 0:
            print('Beginning iteration #' + str(iters) + ' now.')

        # Update the P worst points of the simplex
        output = parallel(delayed(parallelNelderMeadWorker)(objFunc,simplex,fvals,j,P,opt_params) for j in j_list)
        new_subsimplex = np.zeros((P,K)) + np.nan
        new_vals = np.zeros(P) + np.nan
        new_evals = 0
        for i in range(P):
            new_subsimplex[i,:] = output[i][0]
            new_vals[i] = output[i][1]
            new_evals += output[i][2]
        evals += new_evals

        # Check whether any updates actually happened
        old_subsimplex = simplex[(N-P):N,:]
        if np.max(np.abs(new_subsimplex - old_subsimplex)) == 0:
            if verbose > 0:
                print('Updated the simplex, but must perform a shrink step.')
            # If every attempted update was unsuccessful, must shrink the simplex
            simplex = s_param*np.tile(simplex[0,:],(N,1)) + (1.0-s_param)*simplex
            temp_simplex = list(simplex[1:N,:])
            fvals = np.array([fvals[0]] + parallel(delayed(objFunc)(params) for params in temp_simplex))
            new_evals += N-1
            evals += N-1
        else:
            if verbose > 0:
                print('Updated the simplex successfully.')
            # Otherwise, update the simplex with the new results
            simplex[(N-P):N,:] = new_subsimplex
            fvals[(N-P):N] = new_vals

        # Reorder the simplex from best to worst
        order = np.argsort(fvals)
        fvals = fvals[order]
        simplex = simplex[order,:]
        fmin = fvals[0]
        f_dist = np.abs(fmin - fvals[-1])
        x_dist = np.max(np.sqrt(np.sum(simplex**2.0 - np.tile(simplex[0,:],(N,1))**2.0,axis=1)))
        t_end = perf_counter()
        if verbose > 0:
            t_iter = t_end - t_start
            print('Finished iteration #' + str(iters) +' with ' + str(new_evals) + ' evaluations (' + str(evals) + ' cumulative) in ' + str(t_iter) + ' seconds.')
            print('Simplex status: fmin=' + str(fmin) + ', f_dist=' + str(f_dist) + ', x_dist=' + str(x_dist))

        # Check for terminal conditions
        if iters >= maxiter:
            go = False
            print('Maximum iterations reached, terminating unsuccessfully.')
        if evals >= maxeval:
            go = False
            print('Maximum evaluations reached, terminating unsuccessfully.')
        if f_dist < ftol:
            go = False
            print('Function tolerance reached, terminating successfully.')
        if x_dist < xtol:
            go = False
            print('Parameter tolerance reached, terminating successfully.')

        # Save the progress of the estimation if desired
        if savefreq is not None:
            if (iters % savefreq) == 0:
                 saveNelderMeadData(name, simplex, fvals, iters, evals)
                 if verbose > 0:
                     print('Saved search progress in ' + name + '.txt')

    # Return the results
    xopt = simplex[0,:]
    return xopt, fmin


def saveNelderMeadData(name, simplex, fvals, iters, evals):
    '''
    Stores the progress of a parallel Nelder-Mead search in a text file so that
    it can be resumed later (after manual termination or a crash).

    Parameters
    ----------
    name : string
        Name of the txt file in which to store search progress.
    simplex : np.array
        The current state of the simplex of parameter guesses.
    fvals : np.array
        The objective function value at each row of simplex.
    iters : int
        The number of completed Nelder-Mead iterations.
    evals : int
        The cumulative number of function evaluations in the search process.

    Returns
    -------
    none
    '''
    f = open(name + '.txt','w')
    my_writer = csv.writer(f,delimiter=' ')
    my_writer.writerow(simplex.shape)
    my_writer.writerow([iters, evals])
    my_writer.writerow(simplex.flatten())
    my_writer.writerow(fvals)
    f.close()


def loadNelderMeadData(name):
    '''
    Reads the progress of a parallel Nelder-Mead search from a text file, as
    created by saveNelderMeadData().

    Parameters
    ----------
    name : string
        Name of the txt file from which to read search progress.

    Returns
    -------
    simplex : np.array
        The current state of the simplex of parameter guesses.
    fvals : np.array
        The objective function value at each row of simplex.
    iters : int
        The number of completed Nelder-Mead iterations.
    evals : int
        The cumulative number of function evaluations in the search process.
    '''
    f = open(name + '.txt','rb')
    my_reader = csv.reader(f,delimiter=' ')
    my_shape_txt = next(my_reader)
    shape0 = int(my_shape_txt[0])
    shape1 = int(my_shape_txt[1])
    my_nums_txt = next(my_reader)
    iters = int(my_nums_txt[0])
    evals = int(my_nums_txt[1])
    simplex_flat = np.array(next(my_reader),dtype=float)
    simplex = np.reshape(simplex_flat,(shape0,shape1))
    fvals = np.array(next(my_reader),dtype=float)
    f.close()

    return simplex, fvals, iters, evals


def parallelNelderMeadWorker(objFunc,simplex,f_vals,j,P,opt_params):
    '''
    A worker process for the parallel Nelder-Mead algorithm.  Updates one point
    in the simplex, returning its function value as well.  Should basically
    never be called directly, only by parallelNelderMead().

    Parameters
    ----------
    objFunc : function
        The function to be minimized; takes a single 1D array as input.
    simplex : numpy.array
        The current simplex for minimization; simplex[k,:] is an input for objFunc.
    f_vals : numpy.array
        The values of the objective function at each point of the simplex:
        f_vals[k] = objFunc(simplex[k,:])
    j : int
        Index of the point in the simplex to update: simplex[j,:]
    P : int
        Degree of parallelization of the algorithm.
    opt_params : numpy.array
        Three element array with parameters for reflection, contraction, expansion.

    Returns
    -------
    new_point : numpy.array
        An updated point for the simplex; might be the same as simplex[j,:].
    new_val : float
        The value of the objective function at the new point: objFunc(new_point).
    evals : int
        Number of evaluations of objFunc by this worker.
    '''
    # Unpack the input parameters
    alpha = opt_params[0] # reflection parameter
    beta = opt_params[1] # contraction parameter
    gamma = opt_params[2] # expansion parameter
    my_point = simplex[j,:] # vertex to update
    my_val = f_vals[j] # value at the vertex to update
    best_val = f_vals[0] # best value in the vertex
    next_val = f_vals[j-1] # next best point in the simplex
    evals = 0

    # Calculate the centroid of the "good" simplex points
    N = simplex.shape[0] # number of points in simplex
    centroid = np.mean(simplex[0:(N-P),:],axis=0)

    # Calculate the reflection point and its function value
    r_point = centroid + alpha*(centroid - my_point)
    r_val = objFunc(r_point)
    evals += 1

    # Case 1: the reflection point is better than best point
    if r_val < best_val:
        e_point = r_point + gamma*(r_point - centroid)
        e_val = objFunc(e_point) # Calculate expansion point
        evals += 1
        if e_val < r_val:
            new_point = e_point
            new_val = e_val
        else:
            new_point = r_point
            new_val = r_val
    # Case 2: the reflection point is better than the next best point
    elif r_val < next_val:
        new_point = r_point # Report reflection point
        new_val = r_val
    # Case 3: the reflection point is worse than the next best point
    else:
        if r_val < my_val:
            temp_point = r_point # Check whether reflection or original point
            temp_val = r_val     # is better and use it temporarily
        else:
            temp_point = my_point
            temp_val = my_val
        c_point = beta*(centroid + temp_point)
        c_val = objFunc(c_point) # Calculate contraction point
        evals += 1
        if c_val < temp_val:
            new_point = c_point
            new_val = c_val   # Check whether the contraction point is better
        else:
            new_point = temp_point
            new_val = temp_val

    # Return the outputs
    return new_point, new_val, evals

#=============================================================================
#=============================================================================

def main():
    print("Sorry, HARKparallel doesn't actually do much on its own.")
    print("To see an example of multithreading in HARK, see /Testing/MultithreadDemo.")
    print('To ensure full compatibility "out of the box", multithreading is not')
    print('used in our models and applications; users can turn it on by modifying')
    print('the source code slightly.')

    K = 36
    P = 24
    my_guess = np.random.rand(K) - 0.5
    def testFunc1(x):
        return np.sum(x**2.0)/x.size

    xopt, fmin = parallelNelderMead(testFunc1,my_guess,P=P,maxiter=300,savefreq=100,name='testfile',resume=False)
    xopt2, fmin2 = parallelNelderMead(testFunc1,xopt,P=P)

if __name__ == "__main__":
    main()
