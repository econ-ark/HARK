# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:34:31 2016

@author: ganong
"""

import os
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
os.chdir("/Users/ganong/repo/HARK-comments-and-cleanup/gn")
import settings

import sys 
#xxx want to change this default path
#sys.path.remove('/Users/ganong/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages')
#sys.path.remove('/Users/ganong/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/PIL')
sys.path.insert(0,'../')
#sys.path.insert(0,'../ConsumptionSavingModel')
sys.path.insert(0,'../ConsumptionSaving')
sys.path.insert(0,'../SolvingMicroDSOPs')
#test
from copy import copy, deepcopy
import numpy as np
#from HARKcore_gn import AgentType, Solution, NullFunc
#from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKinterpolation import LinearInterp  #, LowerEnvelope, CubicInterp
from HARKutilities import   plotFuncs

#xx why does this error out on later runs but not on the first run? that's weird.
import ConsumptionSavingModel_gn as Model
#import ConsumerParameters as Params
import EstimationParameters_old as Params


#from time import clock
mystr = lambda number : "{:.4f}".format(number)

do_simulation           = True
T_series = 30
baseline_params = Params.init_consumer_objects

import pandas as pd
from rpy2 import robjects
import rpy2.robjects.lib.ggplot2 as gg
from rpy2.robjects import pandas2ri
import make_plots as mp

#xx I'd like to be able to move around parameters here but I haven't figured out how yet! Need to understand self method better.
#NBCExample.assignParameters(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac)  

#enable plotting inside of iPython notebook (default rpy2 pushes to a semi-broken R plot-viewer)
import uuid
from rpy2.robjects.packages import importr 
from IPython.core.display import Image
grdevices = importr('grDevices')
def ggplot_notebook(gg, width = 800, height = 600):
    fn = 'tmp/{uuid}.png'.format(uuid = uuid.uuid4())
    grdevices.png(fn, width = width, height = height)
    gg.plot()
    grdevices.dev_off()
    return Image(filename=fn)
    
###########################################################################
#set economic parameters 
age = 45
t_eval = age - 25

###########################################################################
# Solve consumer problems
#does this still work?
settings.init()
settings.t_rebate = age
settings.rebate_size = 0

IndShockExample = Model.IndShockConsumerType(**baseline_params)
IndShockExample.solve()
IndShockExample.unpack_cFunc()
IndShockExample.timeFwd()

settings.rebate_size = 1
settings.init()

FutRebateExample = Model.IndShockConsumerType(**baseline_params)
FutRebateExample.solve()
FutRebateExample.unpack_cFunc()
FutRebateExample.timeFwd()

settings.rebate_size = 0
settings.init()

IndShockExample.cFunc[19](2)
FutRebateExample.cFunc[19](2)

pandas2ri.activate() 
loc = robjects.r('c(1,0)')

def gg_funcs(functions,bottom,top,N=1000,labels = ["Baseline"],
             title = "Consumption and Cash-on-Hand", ylab = "y", xlab="x", 
             loc = loc, ltitle = 'Variable',
             file_name = None):
    if type(functions)==list:
        function_list = functions
    else:
        function_list = [functions]
       
    step = (top-bottom)/N
    x = np.arange(bottom,top,step)
    fig = pd.DataFrame({'x': x})
    #xx there's got to be a better way to scroll through this list
    i = 0
    for function in function_list:
        fig[labels[i]] = function(x)
        #print labels[i]
        i=i+1
    fig = pd.melt(fig, id_vars=['x'])  
    #print(fig)
    g = gg.ggplot(fig) + \
        mp.base_plot + mp.line + mp.point +  \
        mp.theme_bw(base_size=9) + mp.fte_theme +mp.colors +  \
        gg.labs(title=title,y=ylab,x=xlab) + mp.legend_f(loc) + mp.legend_t_c(ltitle) + mp.legend_t_s(ltitle) #+ \
        #
        #gg.geom_text(data=pd.DataFrame(data={'l':"test"},index=np.arange(1)), x = "1", y = "1",group="1",colour="1", label = "plot mpg vs. wt")
        #gg.geom_text(data=pd.DataFrame(data={'l':"test"},index=np.arange(1)), mapping=gg.aes_string(x="1", y="1",group="1",colour="1",shape="1", mapping="l")) 
    if file_name is not None:
        mp.ggsave(file_name,g)
    return(g)


###########################################################################
# Begin Plots
cf_exo = IndShockExample.cFunc[t_eval]
#cf_nbc = NBCExample.cFunc[t_eval]
cf_fut = FutRebateExample.cFunc[t_eval-1]
cf_fut_tm3 = FutRebateExample.cFunc[t_eval-3]
cf_fut_tm5 = FutRebateExample.cFunc[t_eval-5]

#right now takes 0.0855 seconds per run
#in the future, consider saving each function rather than just saving the output for a certain temp inc realization
def c_future_wealth(fut_period = 1, coh = 1, exo = True):
    c_list = []
    rebate_fut_vals = np.linspace(0, 1, num=11)
    rebate_curr_vals = rebate_fut_vals[1:]
    for rebate_fut in rebate_fut_vals:
        settings.rebate_size = rebate_fut
        settings.init()
        if exo:
            IndShockExample = Model.IndShockConsumerType(**baseline_params)
        else :
            IndShockExample = Model.IndShockConsumerType(**init_natural_borrowing_constraint)
        IndShockExample.solve()
        IndShockExample.unpack_cFunc()
        IndShockExample.timeFwd()
        c_list = np.append(c_list,IndShockExample.cFunc[t_eval-fut_period](coh))
    for rebate_cur in rebate_curr_vals:
        c_list = np.append(c_list,IndShockExample.cFunc[t_eval-fut_period](coh+rebate_cur))
    c_func = LinearInterp(np.linspace(0, 2, num=21),np.array(c_list))             
    return(c_func)
    
yr = gg.ylim(range=robjects.r('c(0.55,1)'))


###########################################################################
#slide 3. consumption function out of future wealth
g = gg_funcs([cf_fut_tm5,cf_fut, cf_exo],0.01,2.5, N=50, loc=robjects.r('c(1,0)'),
         ltitle = '',
         labels = ['Rebate Arrives In 5 Years','Rebate Arrives Next Year', 'No Rebate'],
        title = "Consumption Function With Predictable Rebate of One Year's Income",
        ylab = "Consumption", xlab = "Cash-on-Hand")
mp.ggsave("future_rebate",g)
ggplot_notebook(g, height=300,width=400)


###########################################################################
#slide 4  -- convex consumpion function out of debt forgiveness
convex_c_1 = c_future_wealth(fut_period = 1)
convex_c_2 = c_future_wealth(fut_period = 2)
convex_c_3 = c_future_wealth(fut_period = 3)
convex_c_4 = c_future_wealth(fut_period = 4)
g = gg_funcs([convex_c_1,convex_c_2, convex_c_3,convex_c_4],0.0,2, N=50, 
         labels = ['1 Year','2 Years','3 Years','4 Years'],
        xlab="Wealth Grant",
        title = 'Impact of Pseudo-Debt Forgivenss \n \
        From 0 to 1.0 is Future Grant. From 1.0 to 2.0 is Present Grant.\n Temp Inc = 1',
        ylab = "Consumption", ltitle = "Years Until Future Grant")
g += gg.geom_vline(xintercept=1, linetype=2, colour="red", alpha=0.25)
g += yr     
mp.ggsave("convex_cons_func",g)
ggplot_notebook(g, height=300,width=400)

