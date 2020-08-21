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
def plotFuncs3D(functions,bottom,top,N=300,titles = None):
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
        #ax = fig.add_subplot(1, nfunc, k+1, projection='3d')
        ax = fig.add_subplot(1, nfunc, k+1)
        Z = function_list[k](X,Y)
        #ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
        #                cmap='viridis', edgecolor='none')
        ax.imshow(Z, extent=[bottom[0],top[0],bottom[1],top[1]], origin='lower')
        #ax.colorbar();
        if titles is not None:
            ax.set_title(titles[k]);
            
        ax.set_xlim([bottom[0], top[0]])
        ax.set_ylim([bottom[1], top[1]])
        
    plt.show()

# %%
# Make and solve an example of the risky pension contribution consumer type
init_sticky_share = init_riskyContrib.copy()
init_sticky_share['AdjustPrb'] = 0.9
init_sticky_share['DiscreteShareBool'] = True
init_sticky_share['vFuncBool'] = True


ContribAgent = RiskyContribConsumerType(**init_sticky_share)
# %%
# Make and solve a discrete portfolio choice consumer type
print('Now solving')
t0 = time()
ContribAgent.solve()
t1 = time()
print('Solving took ' + str(t1-t0) + ' seconds.')

# %% Policy function inspection
cFuncAdj = [ContribAgent.solution[t].cFuncAdj for t in range(ContribAgent.T_cycle)]
cFuncFxd = [ContribAgent.solution[t].cFuncFxd for t in range(ContribAgent.T_cycle)]
DFuncAdj = [ContribAgent.solution[t].DFuncAdj for t in range(ContribAgent.T_cycle)]
ShareFuncAdj = [ContribAgent.solution[t].ShareFuncAdj for t in range(ContribAgent.T_cycle)]

plotFuncs3D(cFuncAdj[4:10], bottom = (0,0), top = (5,10))
plotFuncs3D(DFuncAdj[4:10], bottom = (0,0), top = (5,10))
plotFuncs3D(ShareFuncAdj, bottom = (0,0), top = (5,5))