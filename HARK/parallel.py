from typing import List, Type, Any
from joblib import Parallel, delayed
import multiprocessing


def multiThreadCommandsFake(
    agent_list: List, command_list: List, num_jobs=None
) -> None:
    """
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
    """
    for agent in agent_list:
        for command in command_list:
            exec("agent." + command)


def multiThreadCommands(agent_list: List, command_list: List, num_jobs=None) -> None:
    """
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
    """
    if len(agent_list) == 1:
        multiThreadCommandsFake(agent_list, command_list)
        return None

    # Default number of parallel jobs is the smaller of number of AgentTypes in
    # the input and the number of available cores.
    if num_jobs is None:
        num_jobs = min(len(agent_list), multiprocessing.cpu_count())

    # Send each command in command_list to each of the types in agent_list to be run
    agent_list_out = Parallel(n_jobs=num_jobs)(
        delayed(runCommands)(*args)
        for args in zip(agent_list, len(agent_list) * [command_list])
    )

    # Replace the original types with the output from the parallel call
    for j in range(len(agent_list)):
        agent_list[j] = agent_list_out[j]


def runCommands(agent: Any, command_list: List) -> Any:
    """
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
    """
    for command in command_list:
        exec("agent." + command)
    return agent
