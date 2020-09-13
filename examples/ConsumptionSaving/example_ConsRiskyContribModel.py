# %%
'''
Example implementations of HARK.ConsumptionSaving.ConsPortfolioModel
'''
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyContribConsumerType, init_riskyContrib
from copy import copy
from time import time
import numpy as np
import matplotlib.pyplot as plt

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
# Make and solve an example of the risky pension contribution consumer type
init_sticky_share = init_riskyContrib.copy()
init_sticky_share['DiscreteShareBool'] = True
init_sticky_share['vFuncBool'] = True
#init_sticky_share['UnempPrb'] = 0
#init_sticky_share['UnempPrbRet'] = 0
init_sticky_share['IncUnemp'] = 0.0

# Three period model just to check
init_sticky_share['PermGroFac'] = [2.0, 1.0, 0.5]
init_sticky_share['PermShkStd'] = [0.1, 0.1, 0.2]
init_sticky_share['TranShkStd'] = [0.2, 0.2, 0.2]
init_sticky_share['AdjustPrb']  = [0.1, 0.1, 1]
init_sticky_share['tau']        = 0.1
init_sticky_share['LivPrb']     = [1.0, 1.0, 1.0]
init_sticky_share['T_cycle']    = 3
init_sticky_share['T_retire']   = 0
init_sticky_share['T_age']      = 4

ContribAgent = RiskyContribConsumerType(**init_sticky_share)
# %%
# Make and solve a discrete portfolio choice consumer type
print('Now solving')
t0 = time()
ContribAgent.solve()
t1 = time()
print('Solving took ' + str(t1-t0) + ' seconds.')

# %% Policy function inspection

periods = [0,2,4]
n_slices = [0,2,6]
mMax = 20

cFuncFxd     = [ContribAgent.solution[t].cFuncFxd for t in periods]
DFuncAdj     = [ContribAgent.solution[t].DFuncAdj for t in periods]
ShareFuncSha = [ContribAgent.solution[t].ShareFuncSha for t in periods]

# %% Adjusting agent

# Share and Rebalancing
plotSlices3D(DFuncAdj,0,mMax,y_slices = n_slices,y_name = 'n',
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m','d'])

plotSlices3D(ShareFuncSha,0,mMax,y_slices = n_slices,y_name = 'n',
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m','S'])

# %% Constrained agent
from copy import deepcopy
# Create projected consumption functions at different points of the share grid
shares = [0., 0.9]

plotSlices4D(cFuncFxd,0,mMax,y_slices = n_slices,w_slices = shares,
             slice_names = ['n_til','s'],
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m_til','c'])
