"""
Functions for estimating structural models, including optimization methods
and bootstrapping tools.
"""
import numpy as np  # Numerical Python
from time import time  # Used to time execution
from copy import deepcopy  # For replicating complex objects
from scipy.optimize import fmin, fmin_powell  # Minimizers
import warnings
import csv
import multiprocessing
from joblib import Parallel, delayed

__all__ = [
    "minimize_nelder_mead",
    "minimize_powell",
    "bootstrap_sample_from_data",
    "parallelNelderMead",
]


def minimize_nelder_mead(
    objective_func, parameter_guess, verbose=False, which_vars=None, **kwargs
):
    """
    Minimizes the objective function using the Nelder-Mead simplex algorithm,
    starting from an initial parameter guess.
    
    Parameters
    ----------
    objective_func : function
        The function to be minimized.  It should take only a single argument, which
        should be a list representing the parameters to be estimated.
    parameter_guess : [float]
        A starting point for the Nelder-Mead algorithm, which must be a valid
        input for objective_func.
    which_vars : np.array or None
        Array of booleans indicating which parameters should be estimated.  When
        not provided, estimation is performed on all parameters.
    verbose : boolean
        A flag for the amount of output to print.
        
    Returns
    -------
    xopt : [float]
        The values that minimize objective_func.
    """
    # Specify a temporary "modified objective function" that restricts parameters to be estimated
    if which_vars is None:
        which_vars = np.ones(len(parameter_guess), dtype=bool)

    def objective_func_mod(params):
        params_full = np.copy(parameter_guess)
        params_full[which_vars] = params
        out = objective_func(params_full)
        return out

    # convert parameter guess to np array to slice it with boolean array
    parameter_guess_mod = np.array(parameter_guess)[which_vars]

    # Execute the minimization, starting from the given parameter guess
    t0 = time()  # Time the process
    OUTPUT = fmin(
        objective_func_mod, parameter_guess_mod, full_output=1, disp=verbose, **kwargs
    )
    t1 = time()

    # Extract values from optimization output:
    xopt = OUTPUT[0]  # Parameters that minimize function.
    fopt = OUTPUT[1]  # Value of function at minimum: ``fopt = func(xopt)``.
    optiter = OUTPUT[2]  # Number of iterations performed.
    funcalls = OUTPUT[3]  # Number of function calls made.
    warnflag = OUTPUT[4]  # warnflag : int
    #   1 : Maximum number of function evaluations made.
    #   2 : Maximum number of iterations reached.
    # Check that optimization succeeded:
    if warnflag != 0:
        warnings.warn(
            "Minimization failed! xopt="
            + str(xopt)
            + ", fopt="
            + str(fopt)
            + ", optiter="
            + str(optiter)
            + ", funcalls="
            + str(funcalls)
            + ", warnflag="
            + str(warnflag)
        )
    xopt_full = np.copy(parameter_guess)
    xopt_full[which_vars] = xopt

    # Display and return the results:
    if verbose:
        print("Time to estimate is " + str(t1 - t0) + " seconds.")
    return xopt_full


def minimize_powell(objective_func, parameter_guess, verbose=False):
    """
    Minimizes the objective function using a derivative-free Powell algorithm,
    starting from an initial parameter guess.

    Parameters
    ----------
    objective_func : function
        The function to be minimized.  It should take only a single argument, which
        should be a list representing the parameters to be estimated.
    parameter_guess : [float]
        A starting point for the Powell algorithm, which must be a valid
        input for objective_func.
    verbose : boolean
        A flag for the amount of output to print.

    Returns
    -------
    xopt : [float]
        The values that minimize objective_func.
    """

    # Execute the minimization, starting from the given parameter guess
    t0 = time()  # Time the process
    OUTPUT = fmin_powell(
        objective_func, parameter_guess, full_output=1, maxiter=1000, disp=verbose
    )
    t1 = time()

    # Extract values from optimization output:
    xopt = OUTPUT[0]  # Parameters that minimize function.
    fopt = OUTPUT[1]  # Value of function at minimum: ``fopt = func(xopt)``.
    direc = OUTPUT[2]
    optiter = OUTPUT[3]  # Number of iterations performed.
    funcalls = OUTPUT[4]  # Number of function calls made.
    warnflag = OUTPUT[5]  # warnflag : int
    #                           1 : Maximum number of function evaluations made.
    #                           2 : Maximum number of iterations reached.

    # Check that optimization succeeded:
    if warnflag != 0:
        warnings.warn(
            "Minimization failed! xopt="
            + str(xopt)
            + ", fopt="
            + str(fopt)
            + ", direc="
            + str(direc)
            + ", optiter="
            + str(optiter)
            + ", funcalls="
            + str(funcalls)
            + ", warnflag="
            + str(warnflag)
        )

    # Display and return the results:
    if verbose:
        print("Time to estimate is " + str(t1 - t0) + " seconds.")
    return xopt


def bootstrap_sample_from_data(data, weights=None, seed=0):
    """
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
    """
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
    new_data = deepcopy(data[indices,])
    return new_data


def parallelNelderMead(
    obj_func,
    guess,
    perturb=None,
    P=1,
    ftol=0.000001,
    xtol=0.00000001,
    maxiter=np.inf,
    maxeval=np.inf,
    r_param=1.0,
    e_param=1.0,
    c_param=0.5,
    s_param=0.5,
    maxthreads=None,
    name=None,
    resume=False,
    savefreq=None,
    verbose=1,
):
    """

    A parallel implementation of the Nelder-Mead minimization algorithm, as
    described in Lee and Wiswall.  For long optimization procedures, it can
    save progress between iterations and resume later.
    
    Parameters
    ----------
    obj_func : function
        The objective function to be minimized. Takes a single 1D array as input.
    guess : np.array
        Initial starting point for the simplex, representing an input for obj_func.
    perturb : np.array
        Perturbation vector for the simplex, of the same length as an input to
        obj_func.  If perturb[j] is non-zero, a simplex point will be created
        that perturbs the j-th element of guess by perturb[j]; if it is zero,
        then the j-th parameter of obj_func will not be optimized over.  By
        default, perturb=None, indicating that all parameters should be optimized,
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
        Maximum number of evaluations of obj_func (across all processes); reaching
        evals=maxeval is reported as an "unsuccessful" minimization.
    r_param: float
        Parameter indicating magnitude of the reflection point calculation.
    e_param: float
        Parameter indicating magnitude of the expansion point calculation.
    c_param: float
        Parameter indicating magnitude of the contraction point calculation.
    s_param: float
        Parameter indicating magnitude of the shrink calculation.
    maxthreads : int
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
        The input that minimizes obj_func, as found by the minimization.
    fmin : float
        The minimum of obj_func; fmin = obj_func(min_point).
    """
    # If this is a resumed search, load the data
    if resume:
        simplex, fvals, iters, evals = load_nelder_mead_data(name)
        dim_count = fvals.size - 1
        N = dim_count + 1  # Number of points in simplex
        K = simplex.shape[1]  # Total number of parameters

    # Otherwise, construct the initial simplex and array of function values
    else:
        if perturb is None:  # Default: perturb each parameter by 10%
            perturb = 0.1 * guess
            guess[guess == 0] = 0.1

        params_to_opt = np.where(perturb != 0)[
            0
        ]  # Indices of which parameters to optimize
        dim_count = params_to_opt.size  # Number of parameters to search over
        N = dim_count + 1  # Number of points in simplex
        K = guess.size  # Total number of parameters
        simplex = np.tile(guess, (N, 1))
        for j in range(
            dim_count
        ):  # Perturb each parameter to optimize by the specified distance
            simplex[j + 1, params_to_opt[j]] = (
                simplex[j + 1, params_to_opt[j]] + perturb[params_to_opt[j]]
            )

        # Initialize iteration and evaluation counts, plus a 1D array of function values
        fvals = np.zeros(dim_count + 1) + np.nan

        iters = 0
        evals = 0

    # Make sure degree of parallelization is not illegal
    if P > N - 1:
        print(
            "Requested degree of simplex parallelization is "
            + str(P)
            + ", but dimension of optimization problem is only "
            + str(N - 1)
            + "."
        )
        print("Degree of parallelization has been reduced to " + str(N - 1) + ".")
        P = N - 1

    # Create the pool of worker processes
    cpu_cores = multiprocessing.cpu_count()  # Total number of available CPU cores
    cores_to_use = min(cpu_cores, dim_count)
    if maxthreads is not None:  # Cap the number of cores if desired
        cores_to_use = min(cores_to_use, maxthreads)
    parallel = Parallel(n_jobs=cores_to_use)

    # Begin a new Nelder-Mead search
    if not resume:
        temp_simplex = list(simplex)  # Evaluate the initial simplex
        fvals = np.array(parallel(delayed(obj_func)(params) for params in temp_simplex))
        evals += N
        # Reorder the initial simplex
        order = np.argsort(fvals)
        fvals = fvals[order]
        simplex = simplex[order, :]
        fmin = fvals[0]
        f_dist = np.abs(fmin - fvals[-1])
        x_dist = np.max(
            np.sqrt(np.sum((simplex - np.tile(simplex[0, :], (N, 1))) ** 2.0, axis=1))
        )
        if verbose > 0:
            print(
                "Evaluated the initial simplex: fmin="
                + str(fmin)
                + ", f_dist="
                + str(f_dist)
                + ", x_dist="
                + str(x_dist)
            )
        if savefreq is not None:
            save_nelder_mead_data(name, simplex, fvals, iters, evals)
            if verbose > 0:
                print("Saved search progress in " + name + ".txt")
    else:  # Resume an existing search that was cut short
        if verbose > 0:
            print(
                "Resuming search after "
                + str(iters)
                + " iterations and "
                + str(evals)
                + " function evaluations."
            )

    # Initialize some inputs for the multithreader
    j_list = range(N - P, N)
    opt_params = [r_param, c_param, e_param]

    # Run the Nelder-Mead algorithm until a terminal condition is met
    go = True
    while go:
        t_start = time()
        iters += 1
        if verbose > 0:
            print("Beginning iteration #" + str(iters) + " now.")

        # Update the P worst points of the simplex
        output = parallel(
            delayed(parallel_nelder_mead_worker)(obj_func, simplex, fvals, j, P, opt_params)
            for j in j_list
        )
        new_subsimplex = np.zeros((P, K)) + np.nan
        new_vals = np.zeros(P) + np.nan
        new_evals = 0
        for i in range(P):
            new_subsimplex[i, :] = output[i][0]
            new_vals[i] = output[i][1]
            new_evals += output[i][2]
        evals += new_evals

        # Check whether any updates actually happened
        old_subsimplex = simplex[(N - P) : N, :]
        if np.max(np.abs(new_subsimplex - old_subsimplex)) == 0:
            if verbose > 0:
                print("Updated the simplex, but must perform a shrink step.")
            # If every attempted update was unsuccessful, must shrink the simplex
            simplex = (
                s_param * np.tile(simplex[0, :], (N, 1)) + (1.0 - s_param) * simplex
            )
            temp_simplex = list(simplex[1:N, :])
            fvals = np.array(
                [fvals[0]]
                + parallel(delayed(obj_func)(params) for params in temp_simplex)
            )
            new_evals += N - 1
            evals += N - 1
        else:
            if verbose > 0:
                print("Updated the simplex successfully.")
            # Otherwise, update the simplex with the new results
            simplex[(N - P) : N, :] = new_subsimplex
            fvals[(N - P) : N] = new_vals

        # Reorder the simplex from best to worst
        order = np.argsort(fvals)
        fvals = fvals[order]
        simplex = simplex[order, :]
        fmin = fvals[0]
        f_dist = np.abs(fmin - fvals[-1])
        x_dist = np.max(
            np.sqrt(np.sum((simplex - np.tile(simplex[0, :], (N, 1))) ** 2.0, axis=1))
        )
        t_end = time()
        if verbose > 0:
            t_iter = t_end - t_start
            print(
                "Finished iteration #"
                + str(iters)
                + " with "
                + str(new_evals)
                + " evaluations ("
                + str(evals)
                + " cumulative) in "
                + str(t_iter)
                + " seconds."
            )
            print(
                "Simplex status: fmin="
                + str(fmin)
                + ", f_dist="
                + str(f_dist)
                + ", x_dist="
                + str(x_dist)
            )

        # Check for terminal conditions
        if iters >= maxiter:
            go = False
            print("Maximum iterations reached, terminating unsuccessfully.")
        if evals >= maxeval:
            go = False
            print("Maximum evaluations reached, terminating unsuccessfully.")
        if f_dist < ftol:
            go = False
            print("Function tolerance reached, terminating successfully.")
        if x_dist < xtol:
            go = False
            print("Parameter tolerance reached, terminating successfully.")

        # Save the progress of the estimation if desired
        if savefreq is not None:
            if (iters % savefreq) == 0:
                save_nelder_mead_data(name, simplex, fvals, iters, evals)
                if verbose > 0:
                    print("Saved search progress in " + name + ".txt")

    # Return the results
    xopt = simplex[0, :]
    return xopt, fmin


def save_nelder_mead_data(name, simplex, fvals, iters, evals):
    """
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
    None
    """
    N = simplex.shape[0]  # Number of points in simplex
    K = simplex.shape[1]  # Total number of parameters

    with open(name + ".txt", "w") as f:
        my_writer = csv.writer(f, delimiter=",")
        my_writer.writerow(simplex.shape)
        my_writer.writerow([iters, evals])
        for n in range(N):
            my_writer.writerow(simplex[n, :])
        my_writer.writerow(fvals)


def load_nelder_mead_data(name):
    """
    Reads the progress of a parallel Nelder-Mead search from a text file, as
    created by save_nelder_mead_data().
    
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
    """
    # Open the Nelder-Mead progress file
    with open(name + ".txt", "r") as f:
        my_reader = csv.reader(f, delimiter=",")

        # Get the shape of the simplex and initialize it
        my_shape_txt = my_reader.next()
        N = int(my_shape_txt[0])
        K = int(my_shape_txt[1])
        simplex = np.zeros((N, K)) + np.nan

        # Get number of iterations and cumulative evaluations from the next line
        my_nums_txt = my_reader.next()
        iters = int(my_nums_txt[0])
        evals = int(my_nums_txt[1])

        # Read one line per point of the simplex
        for n in range(N):
            simplex[n, :] = np.array(my_reader.next(), dtype=float)

        # Read the final line to get function values
        fvals = np.array(my_reader.next(), dtype=float)

    return simplex, fvals, iters, evals


def parallel_nelder_mead_worker(obj_func, simplex, f_vals, j, P, opt_params):
    """
    A worker process for the parallel Nelder-Mead algorithm.  Updates one point
    in the simplex, returning its function value as well.  Should basically
    never be called directly, only by parallelNelderMead().
    
    Parameters
    ----------
    obj_func : function
        The function to be minimized; takes a single 1D array as input.
    simplex : numpy.array
        The current simplex for minimization; simplex[k,:] is an input for obj_func.
    f_vals : numpy.array
        The values of the objective function at each point of the simplex:
        f_vals[k] = obj_func(simplex[k,:])
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
        The value of the objective function at the new point: obj_func(new_point).
    evals : int
        Number of evaluations of obj_func by this worker.
    """
    # Unpack the input parameters
    alpha = opt_params[0]  # reflection parameter
    beta = opt_params[1]  # contraction parameter
    gamma = opt_params[2]  # expansion parameter
    my_point = simplex[j, :]  # vertex to update
    my_val = f_vals[j]  # value at the vertex to update
    best_val = f_vals[0]  # best value in the vertex
    next_val = f_vals[j - 1]  # next best point in the simplex
    evals = 0

    # Calculate the centroid of the "good" simplex points
    N = simplex.shape[0]  # number of points in simplex
    centroid = np.mean(simplex[0 : (N - P), :], axis=0)

    # Calculate the reflection point and its function value
    r_point = centroid + alpha * (centroid - my_point)
    r_val = obj_func(r_point)
    evals += 1

    # Case 1: the reflection point is better than best point
    if r_val < best_val:
        e_point = r_point + gamma * (r_point - centroid)
        e_val = obj_func(e_point)  # Calculate expansion point
        evals += 1
        if e_val < r_val:
            new_point = e_point
            new_val = e_val
        else:
            new_point = r_point
            new_val = r_val
    # Case 2: the reflection point is better than the next best point
    elif r_val < next_val:
        new_point = r_point  # Report reflection point
        new_val = r_val
    # Case 3: the reflection point is worse than the next best point
    else:
        if r_val < my_val:
            temp_point = r_point  # Check whether reflection or original point
            temp_val = r_val  # is better and use it temporarily
        else:
            temp_point = my_point
            temp_val = my_val
        c_point = temp_point + beta * (centroid - temp_point)
        c_val = obj_func(c_point)  # Calculate contraction point
        evals += 1
        if c_val < temp_val:
            new_point = c_point
            new_val = c_val  # Check whether the contraction point is better
        else:
            new_point = temp_point
            new_val = temp_val

    # Return the outputs
    return new_point, new_val, evals
