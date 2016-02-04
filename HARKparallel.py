'''
This is a very minimal module for an early version of multithreading in HARK.
This code previously lived in HARKcore, but has been split out because the pack-
ages used are not native to Anaconda.  To run this module, first install dill
and joblib.  Packages can be installed by typing "conda install dill" (etc) at
a command prompt.
'''
import multiprocessing
from joblib import Parallel, delayed
import dill as pickle


def multiThreadCommandsFake(agent_list,command_list):
    '''
    Executes the list of commands in command_list for each AgentType in agent_list
    using a multithreaded system.  Just kidding, it's just a loop for now.
    Each command should be a method of that AgentType subclass.
    
    Parameters:
    -----------
    agent_list : [AgentType]
        A list of instances of AgentType on which the commands will be run.
    command_list : [string]
        A list of commands to run for each AgentType.
        
    Returns:
    ----------
    none
    '''
    for agent in agent_list:
        for command in command_list:
            exec('agent.' + command)

       
def multiThreadCommands(agent_list,command_list,num_jobs=None):
    '''
    Executes the list of commands in command_list for each AgentType in agent_list
    using a multithreaded system. Each command should be a method of that AgentType subclass.
    
    Parameters:
    -----------
    agent_list : [AgentType]
        A list of instances of AgentType on which the commands will be run.
    command_list : [string]
        A list of commands to run for each AgentType.
        
    Returns:
    ----------
    none
    '''
    if num_jobs is None:
        num_jobs = min(len(agent_list),multiprocessing.cpu_count())
    agent_list_out = Parallel(n_jobs=num_jobs)(delayed(runCommands)(*args) for args in zip(agent_list, len(agent_list)*[command_list]))
    for j in range(len(agent_list)):
        agent_list[j] = agent_list_out[j]

    
def runCommands(agent,command_list):
    for command in command_list:
        exec('agent.' + command)
    return agent
