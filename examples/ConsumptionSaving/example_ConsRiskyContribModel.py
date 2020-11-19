# %%
'''
Example implementations of HARK.ConsumptionSaving.ConsPortfolioModel
'''
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyContribConsumerType, init_riskyContrib
from time import time
import numpy as np

# %% Define a plotting function
def plotFuncs3D(functions,bottom,top,N=300,titles = None, ax_labs = None):
    '''
    Plots 3D function(s) over a given range.

    Parameters
    ----------
    functions : [function] or function
        A single function, or a list of functions, to be plotted.
    bottom : (float,float)
        The lower limit of the domain to be plotted.
    top : (float,float)
        The upper limit of the domain to be plotted.
    N : int
        Number of points in the domain to evaluate.
    titles: None, or list of string
        If not None, the titles of the subplots

    Returns
    -------
    none
    '''
    import matplotlib.pyplot as plt
    if type(functions)==list:
        function_list = functions
    else:
        function_list = [functions]
    
    nfunc = len(function_list)
    
    # Initialize figure and axes
    fig = plt.figure(figsize=plt.figaspect(1.0/nfunc))
    # Create a mesh
    x = np.linspace(bottom[0],top[0],N,endpoint=True)
    y = np.linspace(bottom[1],top[1],N,endpoint=True)
    X,Y = np.meshgrid(x, y)
    
    for k in range(nfunc):
        
        # Add axisplt
        ax = fig.add_subplot(1, nfunc, k+1, projection='3d')
        #ax = fig.add_subplot(1, nfunc, k+1)
        Z = function_list[k](X,Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        if ax_labs is not None:
            ax.set_xlabel(ax_labs[0])
            ax.set_ylabel(ax_labs[1])
            ax.set_zlabel(ax_labs[2])
        #ax.imshow(Z, extent=[bottom[0],top[0],bottom[1],top[1]], origin='lower')
        #ax.colorbar();
        if titles is not None:
            ax.set_title(titles[k]);
            
        ax.set_xlim([bottom[0], top[0]])
        ax.set_ylim([bottom[1], top[1]])
        
    plt.show()

def plotSlices3D(functions,bot_x,top_x,y_slices,N=300,y_name = None,
                 titles = None, ax_labs = None):

    import matplotlib.pyplot as plt
    if type(functions)==list:
        function_list = functions
    else:
        function_list = [functions]
    
    nfunc = len(function_list)
    
    # Initialize figure and axes
    fig = plt.figure(figsize=plt.figaspect(1.0/nfunc))
    
    # Create x grid
    x = np.linspace(bot_x,top_x,N,endpoint=True)
    
    for k in range(nfunc):
        ax = fig.add_subplot(1, nfunc, k+1)
                
        for y in y_slices:
            
            if y_name is None:
                lab = ''
            else:
                lab = y_name + '=' + str(y)
            
            z = function_list[k](x, np.ones_like(x)*y)
            ax.plot(x,z, label = lab)
            
        if ax_labs is not None:
            ax.set_xlabel(ax_labs[0])
            ax.set_ylabel(ax_labs[1])
            
        #ax.imshow(Z, extent=[bottom[0],top[0],bottom[1],top[1]], origin='lower')
        #ax.colorbar();
        if titles is not None:
            ax.set_title(titles[k]);
            
        ax.set_xlim([bot_x, top_x])
        
        if y_name is not None:
            ax.legend()
        
    plt.show()

def plotSlices4D(functions,bot_x,top_x,y_slices,w_slices,N=300,
                 slice_names = None, titles = None, ax_labs = None):

    import matplotlib.pyplot as plt
    if type(functions)==list:
        function_list = functions
    else:
        function_list = [functions]
    
    nfunc = len(function_list)
    nws   = len(w_slices)
    
    # Initialize figure and axes
    fig = plt.figure(figsize=plt.figaspect(1.0/nfunc))
    
    # Create x grid
    x = np.linspace(bot_x,top_x,N,endpoint=True)
    
    for j in range(nws):
        w = w_slices[j]
        
        for k in range(nfunc):
            ax = fig.add_subplot(nws, nfunc, j*nfunc + k+1)
                    
            for y in y_slices:
                
                if slice_names is None:
                    lab = ''
                else:
                    lab = slice_names[0] + '=' + str(y) + ',' + \
                          slice_names[1] + '=' + str(w)
                
                z = function_list[k](x, np.ones_like(x)*y, np.ones_like(x)*w)
                ax.plot(x,z, label = lab)
                
            if ax_labs is not None:
                ax.set_xlabel(ax_labs[0])
                ax.set_ylabel(ax_labs[1])
                
            #ax.imshow(Z, extent=[bottom[0],top[0],bottom[1],top[1]], origin='lower')
            #ax.colorbar();
            if titles is not None:
                ax.set_title(titles[k]);
                
            ax.set_xlim([bot_x, top_x])
            
            if slice_names is not None:
                ax.legend()
        
    plt.show()

# %%
# Solve an infinite horizon version

# Get initial parameters
par_infinite = init_riskyContrib.copy()
# And make the problem infinite horizon
par_infinite['cycles']   = 0
# and sticky
par_infinite['AdjustPrb'] = 0.5
# and with a withdrawal tax
par_infinite['tau'] = 0.1

par_infinite['DiscreteShareBool'] = True
par_infinite['vFuncBool'] = True

# Create agent and solve it.
InfAgent = RiskyContribConsumerType(**par_infinite)
print('Now solving infinite horizon version')
t0 = time()
InfAgent.solve(verbose = True)
t1 = time()
print('Converged!')
print('Solving took ' + str(t1-t0) + ' seconds.')

# Plot policy functions
periods = [0]
n_slices = [0,2,6]
mMax = 20

DFuncAdj     = [InfAgent.solution[t].stageSols['Reb'].DFuncAdj for t in periods]
ShareFuncSha = [InfAgent.solution[t].stageSols['Sha'].ShareFuncAdj for t in periods]
cFuncFxd     = [InfAgent.solution[t].stageSols['Cns'].cFunc for t in periods]

# Rebalancing
plotSlices3D(DFuncAdj,0,mMax,y_slices = n_slices,y_name = 'n',
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m','d'])
# Share
plotSlices3D(ShareFuncSha,0,mMax,y_slices = n_slices,y_name = 'n',
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m','S'])

# Consumption
shares = [0., 0.9]
plotSlices4D(cFuncFxd,0,mMax,y_slices = n_slices,w_slices = shares,
             slice_names = ['n_til','s'],
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m_til','c'])

# %%
# Solve a short, finite horizon version
par_finite = init_riskyContrib.copy()

# Four period model
par_finite['PermGroFac'] = [2.0, 1.0, 0.1, 1.0]
par_finite['PermShkStd'] = [0.1, 0.1, 0.0, 0.0]
par_finite['TranShkStd'] = [0.2, 0.2, 0.0, 0.0]
par_finite['AdjustPrb']  = [0.5, 0.5, 0.95, 0.95]
par_finite['tau']        = [0.1, 0.1, 0.0, 0.0]
par_finite['LivPrb']     = [1.0, 1.0, 1.0, 1.0]
par_finite['T_cycle']    = 4
par_finite['T_retire']   = 0
par_finite['T_age']      = 4

# Adjust discounting and returns distribution so that they make sense in a 
# 4-period model
par_finite['DiscFac']  = 0.95**15
par_finite['Rfree']    = 1.03**15
par_finite['RiskyAvg'] = 1.08**15 # Average return of the risky asset
par_finite['RiskyStd'] = 0.20*np.sqrt(15) # Standard deviation of (log) risky returns


# Create and solve
ContribAgent = RiskyContribConsumerType(**par_finite)
print('Now solving')
t0 = time()
ContribAgent.solve()
t1 = time()
print('Solving took ' + str(t1-t0) + ' seconds.')

# Plot Policy functions
periods = [0,2,3]

DFuncAdj     = [ContribAgent.solution[t].stageSols['Reb'].DFuncAdj for t in periods]
ShareFuncSha = [ContribAgent.solution[t].stageSols['Sha'].ShareFuncAdj for t in periods]
cFuncFxd     = [ContribAgent.solution[t].stageSols['Cns'].cFunc for t in periods]

# Rebalancing
plotSlices3D(DFuncAdj,0,mMax,y_slices = n_slices,y_name = 'n',
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m','d'])
# Share
plotSlices3D(ShareFuncSha,0,mMax,y_slices = n_slices,y_name = 'n',
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m','S'])
# Consumption
plotSlices4D(cFuncFxd,0,mMax,y_slices = n_slices,w_slices = shares,
             slice_names = ['n_til','s'],
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m_til','c'])

# %%  Simulate the finite horizon consumer
ContribAgent.track_vars = ['pLvlNow','t_age','AdjustNow',
                           'mNrmNow','nNrmNow','mNrmTildeNow','nNrmTildeNow','aNrmNow',
                           'cNrmNow', 'ShareNow', 'DNrmNow']
ContribAgent.T_sim = 4
ContribAgent.AgentCount = 10
ContribAgent.initializeSim()
ContribAgent.simulate()

# %% Format simulation results

import pandas as pd

Data = ContribAgent.history

# Add an id to the simulation results
agent_id = np.arange(ContribAgent.AgentCount)
Data['id'] = np.tile(agent_id,(ContribAgent.T_sim,1))

# Flatten variables
Data = {k: v.flatten(order = 'F') for k, v in Data.items()}

# Make dataframe
Data = pd.DataFrame(Data)