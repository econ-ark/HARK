# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:55:59 2016

@author: ganong
"""
import os
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
os.chdir("/Users/ganong/repo/HARK-comments-and-cleanup/gn")
import settings
import sys 
sys.path.insert(0,'../')
sys.path.insert(0,'../ConsumptionSaving')
sys.path.insert(0,'../SolvingMicroDSOPs')
from copy import deepcopy
import numpy as np
#from HARKcore_gn import AgentType, Solution, NullFunc
from HARKinterpolation import LinearInterp  #, LowerEnvelope, CubicInterp
#xx why does this error out on later runs but not on the first run? that's weird.
import ConsumptionSavingModel_gn as Model
import EstimationParameters as Params
#from time import clock
mystr = lambda number : "{:.4f}".format(number)
do_simulation           = True
baseline_params = Params.init_consumer_objects
import pandas as pd
#this line errors out sometimes. Driven by issues with the Canopy_64bit path
from rpy2 import robjects
import rpy2.robjects.lib.ggplot2 as gg
from rpy2.robjects import pandas2ri
import make_plots as mp
#read in HAMP parameters from google docs
import gspread
from oauth2client.service_account import ServiceAccountCredentials
scope = ['https://spreadsheets.google.com/feeds']
credentials = ServiceAccountCredentials.from_json_keyfile_name('gspread-oauth.json', scope)
gc = gspread.authorize(credentials)
g_params = gc.open("HAMPRA Model Parameters").sheet1 #in this case
df = pd.DataFrame(g_params.get_all_records())
hamp_params = df[['Param','Value']].set_index('Param')['Value'][:6].to_dict()
initialize_hamp_recip = df[['Param','Value']].set_index('Param')['Value'][6:7].to_dict()

#xxx want to modify this default path not to have all these bad things starting out
#sys.path.remove('/Users/ganong/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages')
#sys.path.remove('/Users/ganong/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/PIL')
#xx I'd like to be able to move around parameters here but I haven't figured out how yet! Need to understand self method better.
#NBCExample.assignParameters(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac)  
import uuid     #enable plotting inside of iPython notebook (default rpy2 pushes to a semi-broken R plot-viewer)
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
rebate_years_until_death = 25
age_of_rebate = 90 - rebate_years_until_death
t_eval = age_of_rebate - 25 - 20
def mpc(cF, rebate = 1, a = 0.1):
    return round((cF(a+rebate) - cF(a)) / rebate,2)
tmp_lo = 0.8
tmp_norm = 1
tmp_hi = 1.2

###########################################################################
#calculate rebate and borrowing constraints under HAMP
#in future, when simulating histories, need to adjust for realized income shocks as well
def hsg_wealth(debt, annual_hp_growth, collateral_constraint, initial_debt, initial_price, int_rate, pra_forgive, age_at_mod = 45):
    if initial_debt < 0:
        print("Error: cannot have negative mortgage")
        return
    print "Hsg wealth params: ", annual_hp_growth, collateral_constraint, initial_debt, initial_price, int_rate
    T = 65 - age_at_mod
    price = [initial_price]
    debt = [debt]
    for i in range(1,T):
        #print "age: " + str(i + age_at_mod) + " has growth fac: " + str(Params.PermGroFac[(i-1) + age_at_mod - 25])
        perm_gro = Params.PermGroFac[i + age_at_mod - 26]
        price.append(price[-1]*(1+annual_hp_growth)/perm_gro)
        debt.append(debt[-1]/perm_gro) #this is the no amortization condition
    equity = np.array(price) - np.array(debt)
    limit_from_mod_to_retire = np.min(np.vstack((-(np.array(price)*(1-collateral_constraint) - np.array(debt)),np.zeros(T))),axis=0).tolist()
    limit = [0.0] * (age_at_mod - 26) + limit_from_mod_to_retire + [0.0] * 26
    if equity[T-1] < 0:
        print("Error: still underwater at sale date")
        return
    return equity[T-1], limit

#hamp_params['collateral_constraint'] = 0
#r, L = hsg_wealth(debt =  hamp_params['initial_debt'], **hamp_params)
hamp_params['collateral_constraint'] = 0.2
#r2, L2 = hsg_wealth(debt =  hamp_params['initial_debt'], **hamp_params)
#L[30:40], L2[30:40]

uw_house_params = deepcopy(baseline_params)
uw_house_params['rebate_age_65'], uw_house_params['BoroCnstArt'] = hsg_wealth(debt =  hamp_params['initial_debt'], **hamp_params)
pra_params = deepcopy(baseline_params)
pra_params['rebate_age_65'], pra_params['BoroCnstArt'] = hsg_wealth(debt =  hamp_params['initial_debt'] - hamp_params['pra_forgive'], **hamp_params)

#??? I would like to know how to pass a consumption function in rather than specifying it like I did here.
#I would love to know how to clean up this code using *args or using lambda.
#also this is helpful: https://pythontips.com/2013/08/04/args-and-kwargs-in-python-explained/
def saving_rate(x,tran_shk, cf):
    return(tran_shk-cf(x+tran_shk))   
def cons_bop(x,tran_shk, cf):
    return(cf(x+tran_shk)) 
from functools import partial

pandas2ri.activate() 
loc = robjects.r('c(1,0)')

def gg_funcs(functions,bottom,top,N=1000,labels = [],
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
        if i > len(labels)-1:
            labels.append("func" + str(i))
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
# Solve consumer problems

settings.init()
settings.t_rebate = rebate_years_until_death

def solve_unpack(params):
    #settings.rebate_size = 
    settings.rebate_size = params['rebate_age_65']
    print "Rebate is " + str(round(settings.rebate_size,2)) + " at age " + str(90-settings.t_rebate)
    IndShockConsumerType = Model.IndShockConsumerType(**params)
    IndShockConsumerType.solve()
    IndShockConsumerType.unpack_cFunc()
    IndShockConsumerType.timeFwd()
    return IndShockConsumerType
baseline_params['aXtraCount'] = 30



#DEBUGGING Nans in progress
#xx divide by zero encountered in log
#xx error: All-NaN axis encountered. bottom five values of grid are Nan
#what's going wrong?
#starting only during working years...
#p self.vPfuncNext(self.mNrmNext)[0] has five nan values at bottom
#p self.vPfuncNext(self.mNrmNext)[1] and every subsequent value is defined. so it's just this [0].
#can also be recast as p self.vPfuncNext(self.mNrmNext[0])
#vPfuncNext goes back to vPfunc goes back to vPfuncNow goes back to MargValueFunc
#this is why we are getting the results wrong is we are estimating two slopes
#self.mNrmNext[24] looks pretty good
#MargValueFunc fails when given a number < -1 (if you give it -1, it returns 0)
#it calls cFunc
#why can't cFunc handle arguments < -1? I don't know
#baseline_params['DiscFac'] = (np.array(Params.DiscFac_timevary)*0.96).tolist()
baseline_params['DiscFac'] = (np.ones(65)*0.96).tolist()
baseline_params['vFuncBool'] = True
IndShockExample = solve_unpack(baseline_params)

# Simulate some data for num_agents defaults to 10000; results stored in cHist, mHist, bHist, aHist, MPChist, and pHist
IndShockExample.sim_periods = IndShockExample.T_total + 1
IndShockExample.makeIncShkHist()
IndShockExample.initializeSim()
IndShockExample.simConsHistory()

#how do these change with age
np.mean(IndShockExample.cHist,axis=1) #rising w age and then falling right before death
np.mean(IndShockExample.mHist,axis=1) #rising then falling
np.mean(IndShockExample.bHist,axis=1) #rising then falling bank balances before labor income
np.mean(IndShockExample.aHist,axis=1) #rising then falling
np.mean(IndShockExample.MPChist,axis=1) #falling very rapidly in first five years.
np.mean(IndShockExample.pHist,axis=1) #rising permanent income level

np.mean(IndShockExample.MPChist[:40]) 
np.mean(IndShockExample.MPChist[:20]) 

#redo lifecycle model with housing
lifecycle_hsg_params = deepcopy(baseline_params)
#xx at age 65 you are selling your house for 5. this should be reflected in your asset balances starting at age 65
hamp_params['collateral_constraint'] = 0
rebate, equity = hsg_wealth(debt = 3.25, age_at_mod = 30, **hamp_params)
hamp_params['collateral_constraint'] = 0.2
lifecycle_hsg_params['rebate_age_65'], lifecycle_hsg_params['BoroCnstArt'] = hsg_wealth(debt = 3.25, age_at_mod = 30, **hamp_params)
lifecycle_hsg_example = solve_unpack(lifecycle_hsg_params)
lifecycle_hsg_example.sim_periods = lifecycle_hsg_example.T_total + 1
lifecycle_hsg_example.makeIncShkHist()
lifecycle_hsg_example.initializeSim()
lifecycle_hsg_example.simConsHistory()

np.mean(lifecycle_hsg_example.cHist,axis=1) #rising w age and then falling right before death
np.mean(lifecycle_hsg_example.mHist,axis=1) #rising then falling
np.mean(lifecycle_hsg_example.bHist,axis=1) #rising then falling bank balances before labor income
np.mean(lifecycle_hsg_example.aHist,axis=1) #rising then falling
np.mean(lifecycle_hsg_example.MPChist,axis=1) #falling very rapidly in first five years.
np.mean(lifecycle_hsg_example.pHist,axis=1) #rising permanent income level

np.mean(lifecycle_hsg_example.MPChist[:40]) 
np.mean(lifecycle_hsg_example.MPChist[:20]) 

#array of x-values

#array of y-values


#df = pd.DataFrame({'Loan-To-Value': 100*(1+np.array(equity[4:38]/rebate)), 
#                      'MPC': np.mean(lifecycle_hsg_example.MPChist,axis=1)[4:38]})
#g = gg.ggplot2(df) + \
#    gg.aes_string(x='Loan-To-Value', y='MPC')
#    
#    + mp.line + mp.point     +  \
#    mp.theme_bw(base_size=9) + mp.fte_theme +mp.colors + \
#    gg.labs(title="Home Equity and Marginal Propensity to Consume",
#                  y="MPC", x = "Loan-to-Value Ratio")
#mp.ggsave("ltv_and_mpc",g)
#ggplot_notebook(g, height=300,width=400)
#np.vstack((-1*np.array(lifecycle_hsg_params['BoroCnstArt'][4:38]),np.mean(lifecycle_hsg_example.MPChist,axis=1)[:40]))
#

#alternative consumption functions
example_params = deepcopy(baseline_params)
settings.t_rebate = 35
example_params['rebate_age_65'] = 1
RebateAge55 = solve_unpack(example_params)
settings.t_rebate = 44
RebateAge46 = solve_unpack(example_params)
#settings.verbose = True
example_params['rebate_age_65'] = 0
l = example_params['BoroCnstArt']
for i in range(len(l)):
    l[i] = -1
Boro1YrInc = solve_unpack(example_params)
#settings.verbose = False


#complete package: rebate and borrowing constraint relaxed
settings.t_rebate = rebate_years_until_death
settings.rebate_size = uw_house_params['rebate_age_65']
wealth_grant_only = solve_unpack(baseline_params)
uw_house_example = solve_unpack(uw_house_params)

settings.rebate_size = pra_params['rebate_age_65']
pra_example = solve_unpack(pra_params)

#resolve for 10 different rebate values
hw_cf_params = deepcopy(baseline_params)

#xx in future start with 2D interpolation
hamp_params['collateral_constraint'] = 0.2
tmp_vlo = 0.6
cFuncs = []
hw_cf_list = []
hw_cf_coh_hi_list = []
hw_cf_coh_vlo_list = []
grid_len = 16
grid_int = 0.25
grid_max = grid_len*grid_int
for i in range(grid_len):
    hw_cf_params['rebate_age_65'], hw_cf_params['BoroCnstArt'] = hsg_wealth(debt =  hamp_params['initial_debt'] - i*grid_int, **hamp_params)
    #print hw_cf_params['BoroCnstArt'][20:40]
    settings.rebate_size = hw_cf_params['rebate_age_65']
    print settings.rebate_size
    cf = solve_unpack(hw_cf_params)
    cFuncs.append(cf.cFunc)
    hw_cf_list.append(cf.cFunc[t_eval](initialize_hamp_recip['cash_on_hand']))
    hw_cf_coh_hi_list.append(cf.cFunc[t_eval](tmp_hi))
    hw_cf_coh_vlo_list.append(cf.cFunc[t_eval](tmp_vlo))

#first index says. second index says age 44, I believe 
#t_eval = 19
#cFuncs[0][t_eval](initialize_hamp_recip['cash_on_hand'])
equity_initial = hamp_params['initial_debt'] - hamp_params['initial_price']
#gr_min = -equity_initial
#gr_max = grid_len*grid_int - equity_initial
gr_min = 100*-equity_initial/hamp_params['initial_price']
gr_max = 100*(grid_len*grid_int - equity_initial)/hamp_params['initial_price']
grid_int2 = (gr_max-gr_min)/grid_len
hw_cf = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_list))
hw_cf_coh_hi = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_coh_hi_list))
hw_cf_coh_vlo = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_coh_vlo_list))


hamp_params['collateral_constraint'] = 0
cFuncs_0_pct = []
hw_cf_0_pct_list = []
for i in range(grid_len):
    hw_cf_params['rebate_age_65'], hw_cf_params['BoroCnstArt'] = hsg_wealth(debt =  hamp_params['initial_debt'] - i*grid_int, **hamp_params)
    settings.rebate_size = hw_cf_params['rebate_age_65']
    cf = solve_unpack(hw_cf_params)
    cFuncs_0_pct.append(cf.cFunc)
    hw_cf_0_pct_list.append(cf.cFunc[t_eval](initialize_hamp_recip['cash_on_hand']))
#gr_min = 100*-equity_initial/hamp_params['initial_price']
#gr_max = 100*(grid_len*grid_int - equity_initial)/hamp_params['initial_price']
#grid_int2 = (gr_max-gr_min)/grid_len
hw_cf_0_pct = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_0_pct_list))

#slide 1 -- consumption function out of future wealth
g = gg_funcs([IndShockExample.cFunc[t_eval],RebateAge46.cFunc[t_eval],RebateAge55.cFunc[t_eval]],
        -1.001,3, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Future Wealth",
        labels = ["Baseline (No Grant)","1 Year Away", "10 Years Away"],
        ylab = "Consumption", xlab = "Cash-on-hand")
mp.ggsave("cf_fut_wealth",g)
ggplot_notebook(g, height=300,width=400)

#slide 2 -- Consumption function out of collateral
g = gg_funcs([IndShockExample.cFunc[t_eval],Boro1YrInc.cFunc[t_eval]],
        -1.001,3, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Collateral",
        labels = ["Baseline (No Collateral)","Collateral = 1 Year's Inc"],
        ylab = "Consumption", xlab = "Cash-on-hand")
mp.ggsave("cf_fut_collateral",g)
ggplot_notebook(g, height=300,width=400)

#slide 3 -- housing equity 
#Add proceeds of house sale at age 65 as a vertical arrow.
def neg(x): return -1*x
boro_cnst_pre_pra = list(map(neg, uw_house_params['BoroCnstArt']))
boro_cnst_post_pra = list(map(neg, pra_params['BoroCnstArt']))
#this is 1.1. should be 1.5, i think.
#pra_params['BoroCnstArt'][38] - uw_house_params['BoroCnstArt'][38]

g = gg_funcs([LinearInterp(np.arange(25,90),boro_cnst_pre_pra),LinearInterp(np.arange(25,90),boro_cnst_post_pra)],
              25.11,64, N=40, loc=robjects.r('c(0,0.5)'),
        title = "Borrowing Limits by Year",
        labels = ["Baseline (No PRA)", "With PRA"],
        ylab = "Borrowing Limit", xlab = "Age")
mp.ggsave("borrowing_limits_and_pra",g)
ggplot_notebook(g, height=300,width=400)

#xx anchor on 100 and figure out how to reverse the order 
#slide 4 -- Consumption function out of principal forgiveness
g = gg_funcs(hw_cf,gr_min,gr_max, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Principal Forgiveness",
        labels = ["Baseline"],
        ylab = "Consumption", xlab = "Collateral Position (Loan-to-Value w < 0 as Underwater)")
mp.ggsave("cons_and_prin_forgive",g)
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([hw_cf,hw_cf_0_pct,],gr_min,gr_max, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Principal Forgiveness",
        labels = ["Baseline","Housing Equity >= 0%"],
        ylab = "Consumption", xlab = "Collateral Position (Loan-to-Value w < 0 as Underwater)")
mp.ggsave("cons_and_prin_forgive_v2",g)
ggplot_notebook(g, height=300,width=400)


g = gg_funcs([hw_cf,hw_cf_0_pct,hw_cf_coh_hi],gr_min,gr_max, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Principal Forgiveness",
        labels = ["Baseline","Housing Equity >= 0%","Cash-on-Hand = " + str(tmp_hi)],
        ylab = "Consumption", xlab = "Collateral Position (Loan-to-Value w < 0 as Underwater)")
mp.ggsave("cons_and_prin_forgive_v2",g)
ggplot_notebook(g, height=300,width=400)


g = gg_funcs([hw_cf,hw_cf_coh_vlo,hw_cf_coh_hi],gr_min,gr_max, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Principal Forgiveness By Initial Cash-On-Hand",
        labels = ["Baseline","Cash-on-Hand = " + str(tmp_vlo),"Cash-on-Hand = " + str(tmp_hi)],
        ylab = "Consumption", xlab = "Collateral Position (Loan-to-Value w < 0 as Underwater)")
mp.ggsave("cons_and_prin_forgive_coh",g)
ggplot_notebook(g, height=300,width=400)

hw_cf_comp_coh = lambda x: 0.33*hw_cf(x) + 0.33*hw_cf_coh_vlo(x) + 0.34*hw_cf_coh_hi(x)
hw_cf_comp_collat = lambda x: 0.5*hw_cf(x) + 0.5*hw_cf_0_pct(x)
g = gg_funcs([hw_cf,hw_cf_comp_coh,hw_cf_comp_collat],gr_min,gr_max, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Principal Forgiveness By Initial Cash-On-Hand",
        labels = ["Baseline","Composite w Hetero CoH","Composite w Hetero Collateral"],
        ylab = "Consumption", xlab = "Collateral Position (Loan-to-Value w < 0 as Underwater)")
mp.ggsave("cons_and_prin_forgive_coh_v2",g)
ggplot_notebook(g, height=300,width=400)

#xx figure out the jags near zero. these are appearing only when collateral is below zero

#xx graphs are generating a division by zero error. need to step through this to figure out the issues


#diagnostic plot with consumption functions
cFuncs44 = []
for i in range(0,16,2):
    cFuncs44.append(cFuncs[i][t_eval])
g = gg_funcs(cFuncs44,-1.5,3, N=200, loc=robjects.r('c(1,0)'),
        title = "Consumption Functions. Each line is 0.5 more of Principal Forgiveness",
        ylab = "Consumption", xlab = "Cash on Hand")
mp.ggsave("cfuncs_prin_forgive",g)
ggplot_notebook(g, height=300,width=400)

cFuncs44 = []
for i in range(0,16,2):
    cFuncs44.append(cFuncs_0_pct[i][t_eval])
g = gg_funcs(cFuncs44,-1.5,3, N=200, loc=robjects.r('c(1,0)'),
        title = "Consumption Functions. Each line is 0.5 more of Principal Forgiveness",
        ylab = "Consumption", xlab = "Cash on Hand")
mp.ggsave("cfuncs_prin_forgive_0_pct",g)
ggplot_notebook(g, height=300,width=400)













settings.rebate_size = uw_house_params['rebate_age_65']
settings.init()
FutRebateExample = Model.IndShockConsumerType(**baseline_params)
FutRebateExample.solve()
FutRebateExample.unpack_cFunc()
FutRebateExample.timeFwd()

IndShockExample.cFunc[19](2)
FutRebateExample.cFunc[19](2)

IndShockExample.cFunc[38](2)
FutRebateExample.cFunc[38](2)

#plot borrowing constraint from age 45 to 65
-uw_house_params['BoroCnstArt'][20:40]
-pra_params['BoroCnstArt'][20:40]




yr = gg.ylim(range=robjects.r('c(0.55,1)'))



###########################################################################
# Study moving around collateral constraint

#l = baseline_params['BoroCnstArt']
#for i in range(len(l)):
#    l[i] = 0
        

#check consumption function w diff borrowing constraints
t_eval = 35
cf_list = [IndShockExample.cFunc[t_eval],uw_house_example.cFunc[t_eval],pra_example.cFunc[t_eval]]
g = gg_funcs(cf_list,-2,2.5, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption at Age " + str(25+t_eval) + " With Relaxed Collateral Constraint",
        ylab = "Consumption Function", xlab = "Cash-on-Hand")
mp.ggsave("future_collateral",g)
ggplot_notebook(g, height=300,width=400)

#check consumption function w rebates borrowing constraints
t_eval = 30
cf_list = [IndShockExample.cFunc[t_eval],FutRebateExample.cFunc[t_eval]]
g = gg_funcs(cf_list,0,2.5, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption at Age " + str(25+t_eval) + " With Age 65 Rebate",
        ylab = "Consumption Function", xlab = "Cash-on-Hand")
mp.ggsave("future_rebate_v2",g)
ggplot_notebook(g, height=300,width=400)
#
#cf_list = []
#for t_eval in range(0,40,5):
#    print t_eval
#    dc = lambda x, t_eval = t_eval: RelaxCollat.cFunc[t_eval](x) - IndShockExample.cFunc[t_eval](x)
#    cf_list.append(dc)
#g = gg_funcs(cf_list,0.01,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption With Predictable Collateral at Age " + str(age_of_rebate),
#        ylab = "Consumption Change From Predictable Collateral Movement", xlab = "Cash-on-Hand")
#mp.ggsave("future_collateral_cons_ages",g)
#ggplot_notebook(g, height=300,width=400)
#
#t_eval = 20
#cf_list = [IndShockExample.cFunc[t_eval],RelaxCollat.cFunc[t_eval]]
#g = gg_funcs(cf_list,-1,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption at Age " + str(25+t_eval) + " With Predictable Collateral at Age " + str(age_of_rebate),
#        ylab = "Consumption Change From Predictable Collateral Grant", xlab = "Cash-on-Hand")
#mp.ggsave("future_collateral_cons_age" + str(25+t_eval),g)
#ggplot_notebook(g, height=300,width=400)

###########################################################################
# Begin Plots


t_eval = 39
cf_exo = IndShockExample.cFunc[t_eval-1]
cf_fut = FutRebateExample.cFunc[t_eval-1]
cf_fut_tm3 = FutRebateExample.cFunc[t_eval-3]
cf_fut_tm5 = FutRebateExample.cFunc[t_eval-5]

#slide 3. consumption function out of future wealth
g = gg_funcs([cf_fut_tm5,cf_fut, cf_exo],0.01,2.5, N=50, loc=robjects.r('c(1,0)'),
         ltitle = '',
         labels = ['Rebate Arrives In 5 Years','Rebate Arrives Next Year', 'No Rebate'],
        title = "Consumption Function With Predictable Rebate of One Year's Income at Age " + str(age_of_rebate),
        ylab = "Consumption", xlab = "Cash-on-Hand")
mp.ggsave("future_rebate_v2" + str(age_of_rebate),g)
ggplot_notebook(g, height=300,width=400)


###########################################################################
# nBC consumption function
#
##right now takes 0.0855 seconds per run
##in the future, consider saving each function rather than just saving the output for a certain temp inc realization
#init_natural_borrowing_constraint = deepcopy(baseline_params)
##init_natural_borrowing_constraint['BoroCnstArt'] = None #min is at -0.42 with a natural borrowing constraint
#l = init_natural_borrowing_constraint['BoroCnstArt']
#for i in range(len(l)):
#    l[i] = -10
#NbcExample = Model.IndShockConsumerType(**init_natural_borrowing_constraint)
#NbcExample.solve()
#NbcExample.unpack_cFunc()
#NbcExample.timeFwd()
#cf_nbc = NbcExample.cFunc[t_eval]
#
#
#g = gg_funcs([cf_exo,cf_nbc],-0.5,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption Functions age 45", labels = ['Exo Constraint','Nat Borrowing Constraint'],
#        ylab = "Consumption", xlab = "Cash-on-Hand")
#mp.ggsave("nbc_age_45",g)
#ggplot_notebook(g, height=300,width=400)
#
#
#g = gg_funcs([IndShockExample.cFunc[10],NbcExample.cFunc[10]],-0.5,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption Functions age 35", labels = ['Exo Constraint','Nat Borrowing Constraint'],
#        ylab = "Consumption", xlab = "Cash-on-Hand")
#mp.ggsave("nbc_age_35",g)
#ggplot_notebook(g, height=300,width=400)
#
#
#
#g = gg_funcs([IndShockExample.cFunc[40],NbcExample.cFunc[40]],-10,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption Functions age 65", labels = ['Exo Constraint','Nat Borrowing Constraint'],
#        ylab = "Consumption", xlab = "Cash-on-Hand")
#mp.ggsave("nbc_age_65",g)
#ggplot_notebook(g, height=300,width=400)


###########################################################################

#Study moving rebate date
###########################################################################
cf_list = []
for i in range(10):
    cf_list.append(FutRebateExample.cFunc[40-i+5])
g = gg_funcs(cf_list,0.01,2.5, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function With Predictable Rebate of One Year's Income at Age " + str(age_of_rebate),
        ylab = "Consumption", xlab = "Cash-on-Hand")
ggplot_notebook(g, height=300,width=400)

#slide 3 with moving around date of rebate
#if you give rebate during retirement then no change (except that your wealth is higher)
#because there is no income uncertainty left

#figure out much younger and much older
#bumping age of rebate by ONE year bumps age of cons func affected by TWO years. why?
#there is s ahre cons bump at 

#baseline cons function is much higher (and has bigger h2m region) after retirement
#cons function with rebate at 45 is much higher after age 50

#when we had a rebate, then we find that cons funcs change sharply at age 50
# at age 50 inc growth rate  from 2.5% per year to 1% per year
#i don't understand conceptually why this matters
#also, tried this again later and couldn't reproduce
age_at_rebate = 50
age_eval_min = 44
years_eval = 9
dyear = 3
settings.t_rebate = baseline_params['T_total'] + 25 - age_at_rebate 
settings.rebate_size = 1
rebate_alt_age = Model.IndShockConsumerType(**baseline_params)
rebate_alt_age.solve()
rebate_alt_age.unpack_cFunc()
rebate_alt_age.timeFwd()
cf_list = [IndShockExample.cFunc[age_eval_min-25]]
for i in range(years_eval):
    cf_list.append(rebate_alt_age.cFunc[age_eval_min-25+i*dyear])
g = gg_funcs(cf_list,0.01,2.5, N=50, loc=robjects.r('c(1,0)'),
         ltitle = '', labels = ['no rebate'],
        title = "Rebate at Age " + str(age_at_rebate) + " cons func age " + str(age_eval_min) + " to " + str(age_eval_min + dyear*years_eval),
        ylab = "Consumption", xlab = "Cash-on-Hand")
ggplot_notebook(g, height=300,width=400)



###########################################################################
#slide 1: traditional consumption function
base_hi = 2
base_lo = 0.8
rebate = 0.2
dy = 0.05
mpc_exo_hi = mpc(cf_exo,rebate=rebate, a=base_hi)
mpc_exo_lo = mpc(cf_exo,rebate=rebate, a=base_lo)
g = gg_funcs([cf_exo],0.01,2.5, N=50, 
         labels = ['Exogenous'],xlab="Cash-on-Hand",
        ylab = "Consumption", ltitle = "Borrowing Constraint")
g += mp.annotate(geom = "text", x = 0.9, y = 0.45, label = "MPC: " + str(mpc_exo_lo))
g += gg.geom_segment(gg.aes_string(x = base_lo+dy, y = float(cf_exo(base_lo+rebate)), xend = base_lo+rebate, yend = float(cf_exo(base_lo+rebate))),color= robjects.r.palette_lines[2])
g += gg.geom_segment(gg.aes_string(x = base_lo+dy, y = float(cf_exo(base_lo+dy)), xend = base_lo+dy, yend = float(cf_exo(base_lo+rebate))),color= robjects.r.palette_lines[2])
g += mp.annotate(geom = "text", x = 2.1, y = 0.8, label = "MPC: " + str(mpc_exo_hi))
g += gg.geom_segment(gg.aes_string(x = base_hi+dy, y = float(cf_exo(base_hi+rebate)), xend = base_hi+rebate, yend = float(cf_exo(base_hi+rebate))),color= robjects.r.palette_lines[2])
g += gg.geom_segment(gg.aes_string(x = base_hi+dy, y = float(cf_exo(base_hi+dy)), xend = base_hi+dy, yend = float(cf_exo(base_hi+rebate))),color= robjects.r.palette_lines[2])
g += gg.ylim(range=robjects.r('c(0,1)'))
mp.ggsave("trad_cf_exo",g)
ggplot_notebook(g, height=300,width=400)


###########################################################################
#slide 2. consumption and temporary income
#"MPCs assume  initial assets = 0.5"
mpc_lo = mpc(cf_exo,rebate=rebate, a=tmp_lo)
mpc_norm = mpc(cf_exo,rebate=rebate, a=tmp_norm)
mpc_hi = mpc(cf_exo,rebate=rebate, a=tmp_hi)
label_str = "MPC @ 80%: " + str(mpc_lo) + "\n MPC @ 100%: " + str(mpc_norm) + "\n MPC @ 120%: " + str(mpc_hi)
cons_2 = partial(cons_bop,tran_shk=tmp_hi, cf = cf_exo) 
cons_1 = partial(cons_bop,tran_shk=tmp_norm, cf = cf_exo)
cons_0 = partial(cons_bop,tran_shk=tmp_lo, cf = cf_exo)  
g = gg_funcs([cons_0,cons_1,cons_2],0.01,2.5, N=10, loc=robjects.r('c(1,0)'),
         ltitle = 'Ratio of Temp Inc to Perm Inc',
         labels = ['80%','100%', '120%'],
        title = "Impact of Temporary Income on Consumption\n MPCs assume beginning of period assets are zero.",
        ylab = "Consumption", xlab = "Beginning-of-Period Assets")
g += mp.annotate(geom = "text", x = 1, y = 0.7, label = label_str)
mp.ggsave("cons_bop_exo",g)
ggplot_notebook(g, height=300,width=400)


#slide 2 backup
saving_2 = partial(saving_rate,tran_shk=tmp_hi, cf = cf_exo) 
saving_1 = partial(saving_rate,tran_shk=tmp_norm, cf = cf_exo)
saving_0 = partial(saving_rate,tran_shk=tmp_lo, cf = cf_exo)  
g = gg_funcs([saving_0,saving_1,saving_2],0.01,2.5, N=10, loc=robjects.r('c(1,1)'),
         ltitle = 'Temp / Perm Inc',
         labels = ['80%','100%', '120%'],
        title = "Saving Rate ",
        ylab = "Net Saving as Share of Perm Inc", xlab = "Beginning-of-Period Assets")
g += gg.ylim(range=robjects.r('c(-0.2,0.45)'))
mp.ggsave("saving_rate_exo",g)
ggplot_notebook(g, height=300,width=400)
  

#slide 3 backup
#xx these can be cleaned up by sharing a common baseplot.
mpc_fut = mpc(cf_fut,rebate=rebate, a=0.5)
mpc_norm = mpc(cf_exo,rebate=rebate, a=0.5)
label_str = "MPC w/Standard Cons Func: " + str(mpc_norm) + "\n MPC w/Rebate Next Year: " + str(mpc_fut) 
g = gg_funcs([cf_fut_tm5,cf_fut, cf_exo],0.01,2.5, N=50, loc=robjects.r('c(1,0)'),
         ltitle = '',
         labels = ['Rebate Arrives In 5 Years','Rebate Arrives Next Year', 'No Rebate'],
        title = "Consumption Function With Predictable Rebate of One Year's Income \n MPCs assume cash-on-hand is 0.5",
        ylab = "Consumption", xlab = "Cash-on-Hand")
g += mp.annotate(geom = "text", x = 1.5, y = 0.5, label = label_str)
mp.ggsave("future_rebate_tmp_vlo",g)
ggplot_notebook(g, height=300,width=400)


#slide 3 backup
mpc_fut = mpc(cf_fut,rebate=rebate, a=tmp_lo)
mpc_norm = mpc(cf_exo,rebate=rebate, a=tmp_lo)
label_str = "MPC w/Standard Cons Func: " + str(mpc_norm) + "\n MPC w/Rebate Next Year: " + str(mpc_fut) 
g = gg_funcs([cf_fut_tm5,cf_fut, cf_exo],0.01,2.5, N=50, loc=robjects.r('c(1,0)'),
         ltitle = '',
         labels = ['Rebate Arrives In 5 Years','Rebate Arrives Next Year', 'No Rebate'],
        title = "Consumption Function With Predictable Rebate of One Year's Income \n MPCs assume cash-on-hand is 0.8",
        ylab = "Consumption", xlab = "Cash-on-Hand")
g += mp.annotate(geom = "text", x = 1.5, y = 0.5, label = label_str)
mp.ggsave("future_rebate_tmp_lo",g)
ggplot_notebook(g, height=300,width=400)


#slide 3 backup
mpc_fut = mpc(cf_fut,rebate=rebate, a=2)
mpc_norm = mpc(cf_exo,rebate=rebate, a=2)
label_str = "MPC w/Standard Cons Func: " + str(mpc_norm) + "\n MPC w/Rebate Next Year: " + str(mpc_fut) 
g = gg_funcs([cf_fut_tm5,cf_fut, cf_exo],0.01,2.5, N=50, loc=robjects.r('c(1,0)'),
         ltitle = '',
         labels = ['Rebate Arrives In 5 Years','Rebate Arrives Next Year', 'No Rebate'],
        title = "Consumption Function With Predictable Rebate of One Year's Income \n MPCs assume cash-on-hand is 2",
        ylab = "Consumption", xlab = "Cash-on-Hand")
g += mp.annotate(geom = "text", x = 1.5, y = 0.5, label = label_str)
mp.ggsave("future_rebate_coh_hi",g)
ggplot_notebook(g, height=300,width=400)

###########################################################################
#slide 4  -- convex consumpion function out of debt forgiveness


def c_future_wealth(fut_period = 1, coh = 1, exo = True):
    c_list = []
    rebate_fut_vals = np.linspace(0, 2, num=11)
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
    c_func = LinearInterp(np.linspace(0, 4, num=21),np.array(c_list))             
    return(c_func)
    
    
rebate_years_until_death = 45
age_of_rebate = 90 - rebate_years_until_death
t_eval = age_of_rebate - 25
convex_c_1 = c_future_wealth(fut_period = 1)
convex_c_2 = c_future_wealth(fut_period = 2)
convex_c_3 = c_future_wealth(fut_period = 3)
convex_c_4 = c_future_wealth(fut_period = 4)
convex_c_5 = c_future_wealth(fut_period = 20)
g = gg_funcs([convex_c_1,convex_c_2, convex_c_3,convex_c_4,convex_c_5],0.0,4, N=50, 
         labels = ['1 Year','2 Years','3 Years','4 Years', '20 Years'],
        xlab="Wealth Grant",
        title = 'Impact of Pseudo-Debt Forgivenss \n \
        From 0 to 1.0 is Future Grant. From 1.0 to 2.0 is Present Grant.\n Temp Inc = ' + str(tmp_norm),
        ylab = "Consumption", ltitle = "Years Until Future Grant")
g += gg.geom_vline(xintercept=2, linetype=2, colour="red", alpha=0.25)
g += yr     
mp.ggsave("convex_cons_func_v2",g)
ggplot_notebook(g, height=300,width=400)


#slide 4 
convex_c_1 = c_future_wealth(fut_period = 1, coh = tmp_lo)
convex_c_2 = c_future_wealth(fut_period = 2, coh = tmp_lo)
convex_c_3 = c_future_wealth(fut_period = 3, coh = tmp_lo)
convex_c_4 = c_future_wealth(fut_period = 4, coh = tmp_lo)
g = gg_funcs([convex_c_1,convex_c_2, convex_c_3,convex_c_4],0.0,2, N=50, 
         labels = ['1 Year','2 Years','3 Years','4 Years'],
        xlab="Wealth Grant",
        title = 'Impact of Pseudo-Debt Forgivenss \n \
        From 0 to 1.0 is Future Grant. From 1.0 to 2.0 is Present Grant.\n Temp Inc = ' + str(tmp_lo),
        ylab = "Consumption", ltitle = "Years Until Future Grant")
g += gg.geom_vline(xintercept=1, linetype=2, colour="red", alpha=0.25)    
g += yr 
mp.ggsave("convex_cons_func_temp_low",g)
ggplot_notebook(g, height=300,width=400)



convex_c_1 = c_future_wealth(fut_period = 1, coh = tmp_hi)
convex_c_2 = c_future_wealth(fut_period = 2, coh = tmp_hi)
convex_c_3 = c_future_wealth(fut_period = 3, coh = tmp_hi)
convex_c_4 = c_future_wealth(fut_period = 4, coh = tmp_hi)
g = gg_funcs([convex_c_1,convex_c_2, convex_c_3,convex_c_4],0.0,2, N=50, 
         labels = ['1 Year','2 Years','3 Years','4 Years'],
        xlab="Wealth Grant",
        title = 'Grant From 0 to 1.0 is Future. From 1.0 to 2.0 is Present.\n Temp Inc = ' + str(tmp_hi),
        ylab = "Consumption", ltitle = "Years Until Grant")
g += gg.geom_vline(xintercept=1, linetype=2, colour="red", alpha=0.25)
g += yr         
mp.ggsave("convex_cons_func_temp_hi",g)
ggplot_notebook(g, height=300,width=400)


#slide 4 backup
base_a = 1
convex_c_1 = c_future_wealth(fut_period = 1, coh = tmp_norm + base_a)
convex_c_2 = c_future_wealth(fut_period = 2, coh = tmp_norm + base_a)
convex_c_3 = c_future_wealth(fut_period = 3, coh = tmp_norm + base_a)
convex_c_4 = c_future_wealth(fut_period = 4, coh = tmp_norm + base_a)
g = gg_funcs([convex_c_1,convex_c_2, convex_c_3,convex_c_4],0.0,2, N=50, 
         labels = ['1 Year','2 Years','3 Years','4 Years'],
        xlab="Wealth Grant",
        title = 'Impact of Pseudo-Debt Forgivenss \n \
        From 0 to 1.0 is Future Grant. From 1.0 to 2.0 is Present Grant.\n Beginning-of-Period Assets = ' + str(base_a) + ' Temp Inc = ' + str(tmp_norm),
        ylab = "Consumption", ltitle = "Years Until Future Grant")
g += gg.geom_vline(xintercept=1, linetype=2, colour="red", alpha=0.25)
g += yr     
mp.ggsave("convex_cons_func_base_a",g)
ggplot_notebook(g, height=300,width=400)


#slide 4 
convex_c_1 = c_future_wealth(fut_period = 1, coh = tmp_lo + base_a)
convex_c_2 = c_future_wealth(fut_period = 2, coh = tmp_lo + base_a)
convex_c_3 = c_future_wealth(fut_period = 3, coh = tmp_lo + base_a)
convex_c_4 = c_future_wealth(fut_period = 4, coh = tmp_lo + base_a)
g = gg_funcs([convex_c_1,convex_c_2, convex_c_3,convex_c_4],0.0,2, N=50, 
         labels = ['1 Year','2 Years','3 Years','4 Years'],
        xlab="Wealth Grant",
        title = 'Impact of Pseudo-Debt Forgivenss \n \
        From 0 to 1.0 is Future Grant. From 1.0 to 2.0 is Present Grant.\n Beginning-of-Period Assets = ' + str(base_a) + ' Temp Inc = ' + str(tmp_lo),
        ylab = "Consumption", ltitle = "Years Until Future Grant")
g += gg.geom_vline(xintercept=1, linetype=2, colour="red", alpha=0.25)    
g += yr 
mp.ggsave("convex_cons_func_temp_low_base_a",g)
ggplot_notebook(g, height=300,width=400)



convex_c_1 = c_future_wealth(fut_period = 1, coh = tmp_hi + base_a)
convex_c_2 = c_future_wealth(fut_period = 2, coh = tmp_hi + base_a)
convex_c_3 = c_future_wealth(fut_period = 3, coh = tmp_hi + base_a)
convex_c_4 = c_future_wealth(fut_period = 4, coh = tmp_hi + base_a)
g = gg_funcs([convex_c_1,convex_c_2, convex_c_3,convex_c_4],0.0,2, N=50, 
         labels = ['1 Year','2 Years','3 Years','4 Years'],
        xlab="Wealth Grant",
        title = 'Grant From 0 to 1.0 is Future. From 1.0 to 2.0 is Present.\n Beginning-of-Period Assets = ' + str(base_a) + ' Temp Inc = ' + str(tmp_hi),
        ylab = "Consumption", ltitle = "Years Until Grant")
g += gg.geom_vline(xintercept=1, linetype=2, colour="red", alpha=0.25)
g += yr         
mp.ggsave("convex_cons_func_temp_hi_base_a",g)
ggplot_notebook(g, height=300,width=400)

###########################################################################
#slide 5 -- consumption impact of principal reduction 

#make a consumption impact figure
yr = gg.ylim(range=robjects.r('c(-0.01,0.75)'))
def c_impact_now(x,cF,rebate, tran_shk = 1):
    return (cF(x+tran_shk+rebate) - cF(x+tran_shk)) / rebate
def c_impact_fut(x,cF_new, cF_base = cf_exo, rebate = 1, tran_shk = 1):
    return (cF_new(x+tran_shk) - cF_base(x+tran_shk)) / rebate
baseline = partial(c_impact_now,cF = cf_exo, rebate = 1,tran_shk = 0) 
yr_1 = partial(c_impact_fut,cF_new = cf_fut, tran_shk = 0) 
yr_3 = partial(c_impact_fut,cF_new = cf_fut_tm3, tran_shk = 0) 
yr_5 = partial(c_impact_fut,cF_new = cf_fut_tm5, tran_shk = 0) 
g = gg_funcs([baseline,yr_1,yr_3,yr_5],0.01,2.5, N=50, 
         labels = ['Today','1 Year','3 Years','5 Years'],xlab="Cash-on-Hand",
        ylab = "Change in Consumption", ltitle = "Timing",
        title = "Consumption Impact from Rebate Equal to One Year's Income")
g += yr
mp.ggsave("cons_impact_from_policy",g)
ggplot_notebook(g, height=300,width=400)

#add different temporary income values
baseline = partial(c_impact_now,cF = cf_exo, rebate = 1) 
yr_1 = partial(c_impact_fut,cF_new = cf_fut) 
yr_3 = partial(c_impact_fut,cF_new = cf_fut_tm3) 
yr_5 = partial(c_impact_fut,cF_new = cf_fut_tm5) 
g = gg_funcs([baseline,yr_1,yr_3,yr_5],0.01,2.5, N=50, 
         labels = ['Today','1 Year','3 Years','5 Years'],xlab="Beginning-of-Period Assets",
        ylab = "Change in Consumption", ltitle = "Timing",
        title = "Consumption Impact from Rebate Equal to One Year's Income \n Temporary Income = 1")
g += yr
mp.ggsave("cons_impact_from_policy_bop",g)
ggplot_notebook(g, height=300,width=400)


#add different temporary income values
baseline = partial(c_impact_now,cF = cf_exo, rebate = 1, tran_shk = tmp_lo) 
yr_1 = partial(c_impact_fut,cF_new = cf_fut, tran_shk = tmp_lo) 
yr_3 = partial(c_impact_fut,cF_new = cf_fut_tm3, tran_shk = tmp_lo) 
yr_5 = partial(c_impact_fut,cF_new = cf_fut_tm5, tran_shk = tmp_lo) 
g = gg_funcs([baseline,yr_1,yr_3,yr_5],0.01,2.5, N=50, 
         labels = ['Today','1 Year','3 Years','5 Years'],xlab="Beginning-of-Period Assets",
        ylab = "Change in Consumption", ltitle = "Timing",
        title = "Consumption Impact from Rebate Equal to One Year's Income \n Temporary Income = " + str(tmp_lo))
g += yr
mp.ggsave("cons_impact_from_policy_bop_tmp_lo",g)
ggplot_notebook(g, height=300,width=400)



#slides which i'd like to delete
convex_c_1 = c_future_wealth(fut_period = 1, exo = False)
convex_c_2 = c_future_wealth(fut_period = 2, exo = False)
convex_c_3 = c_future_wealth(fut_period = 3, exo = False)

g = gg_funcs([convex_c_1,convex_c_2, convex_c_3],0.0,2, N=50, 
         labels = ['1 Year','2 Years','3 Years'],
        xlab="Wealth Grant",
        title = 'Grant From 0 to 1.0 is Future. From 1.0 to 2.0 is Present.\n Temp Inc = ' + str(tmp_norm) + ', Natural Borrowing Constraint',
        ylab = "Consumption", ltitle = "Years Until Grant")
g += gg.geom_vline(xintercept=1, linetype=2, colour="red", alpha=0.25)
g += yr     
mp.ggsave("convex_cons_func_nbc",g)
ggplot_notebook(g, height=300,width=400)


convex_c_1 = c_future_wealth(fut_period = 1, coh = tmp_lo, exo = False)
convex_c_2 = c_future_wealth(fut_period = 2, coh = tmp_lo, exo = False)
convex_c_3 = c_future_wealth(fut_period = 3, coh = tmp_lo, exo = False)
g = gg_funcs([convex_c_1,convex_c_2, convex_c_3],0.0,2, N=50, 
         labels = ['1 Year','2 Years','3 Years'],
        xlab="Wealth Grant",
        title = 'Grant From 0 to 1.0 is Future. From 1.0 to 2.0 is Present.\n Temp Inc = ' + str(tmp_lo) + ', Natural Borrowing Constraint',
        ylab = "Consumption", ltitle = "Years Until Grant")
g += gg.geom_vline(xintercept=1, linetype=2, colour="red", alpha=0.25) 
g += yr        
mp.ggsave("convex_cons_func_temp_low_nbc",g)
ggplot_notebook(g, height=300,width=400)



base = 0.5
mpc_exo = mpc(cf_exo,rebate=rebate, a=base)
mpc_nbc = mpc(cf_nbc,rebate=rebate, a=base)
g = gg_funcs([cf_exo,cf_nbc],0.01,2.5, N=50, 
         labels = ['Exogenous','Natural'],xlab="Cash-on-Hand",
        ylab = "Consumption", ltitle = "Borrowing Constraint")
g += mp.annotate(geom = "text", x = 1, y = 0.2, label = "Exogenous MPC: " + str(mpc_exo))
g += mp.annotate(geom = "text", x = 1, y = 0.9, label = "Natural MPC: " + str(mpc_nbc))
g += gg.geom_segment(gg.aes_string(x = base+dy, y = float(cf_exo(base+rebate)), xend = base+rebate, yend = float(cf_exo(base+rebate))),color= robjects.r.palette_lines[1])
g += gg.geom_segment(gg.aes_string(x = base+dy, y = float(cf_exo(base+dy)), xend = base+dy, yend = float(cf_exo(base+rebate))),color= robjects.r.palette_lines[1])
g += gg.geom_segment(gg.aes_string(x = base, y = float(cf_nbc(base+rebate)), xend = base+rebate, yend = float(cf_nbc(base+rebate))),color= robjects.r.palette_lines[2])
g += gg.geom_segment(gg.aes_string(x = base, y = float(cf_nbc(base)), xend = base, yend = float(cf_nbc(base+rebate))),color= robjects.r.palette_lines[2])
mp.ggsave("exo_vs_nbc",g)
ggplot_notebook(g, height=300,width=400)


base = 1
mpc_exo = mpc(cf_exo,rebate=rebate, a=base)
mpc_nbc = mpc(cf_nbc,rebate=rebate, a=base)
g = gg_funcs([cf_exo,cf_nbc],0.01,2.5, N=50, 
         labels = ['Exogenous','Natural'],xlab="Cash-on-Hand",
        ylab = "Consumption", ltitle = "Borrowing Constraint")
g += mp.annotate(geom = "text", x = 1.7, y = 0.7, label = "Exogenous MPC: " + str(mpc_exo))
g += mp.annotate(geom = "text", x = 0.4, y = 0.9, label = "Natural MPC: " + str(mpc_nbc))
g += gg.geom_segment(gg.aes_string(x = base+dy, y = float(cf_exo(base+rebate)), xend = base+rebate, yend = float(cf_exo(base+rebate))),color= robjects.r.palette_lines[1])
g += gg.geom_segment(gg.aes_string(x = base+dy, y = float(cf_exo(base+dy)), xend = base+dy, yend = float(cf_exo(base+rebate))),color= robjects.r.palette_lines[1])
g += gg.geom_segment(gg.aes_string(x = base, y = float(cf_nbc(base+rebate)), xend = base+rebate, yend = float(cf_nbc(base+rebate))),color= robjects.r.palette_lines[2])
g += gg.geom_segment(gg.aes_string(x = base, y = float(cf_nbc(base)), xend = base, yend = float(cf_nbc(base+rebate))),color= robjects.r.palette_lines[2])
mp.ggsave("exo_vs_nbc_a1",g)
ggplot_notebook(g, height=300,width=400)

base = 1.5
mpc_exo = mpc(cf_exo,rebate=rebate, a=base)
mpc_nbc = mpc(cf_nbc,rebate=rebate, a=base)
g = gg_funcs([cf_exo,cf_nbc],0.01,2.5, N=50, 
         labels = ['Exogenous','Natural'],xlab="Cash-on-Hand",
        ylab = "Consumption", ltitle = "Borrowing Constraint")
g += mp.annotate(geom = "text", x = 1.7, y = 0.8, label = "Exogenous MPC: " + str(mpc_exo))
g += mp.annotate(geom = "text", x = 1.3, y = 0.95, label = "Natural MPC: " + str(mpc_nbc))
g += gg.geom_segment(gg.aes_string(x = base+dy, y = float(cf_exo(base+rebate)), xend = base+rebate, yend = float(cf_exo(base+rebate))),color= robjects.r.palette_lines[1])
g += gg.geom_segment(gg.aes_string(x = base+dy, y = float(cf_exo(base+dy)), xend = base+dy, yend = float(cf_exo(base+rebate))),color= robjects.r.palette_lines[1])
g += gg.geom_segment(gg.aes_string(x = base, y = float(cf_nbc(base+rebate)), xend = base+rebate, yend = float(cf_nbc(base+rebate))),color= robjects.r.palette_lines[2])
g += gg.geom_segment(gg.aes_string(x = base, y = float(cf_nbc(base)), xend = base, yend = float(cf_nbc(base+rebate))),color= robjects.r.palette_lines[2])
mp.ggsave("exo_vs_nbc_a2",g)
ggplot_notebook(g, height=300,width=400)


cons_2 = partial(cons_bop,tran_shk=tmp_hi, cf = cf_nbc) 
cons_1 = partial(cons_bop,tran_shk=tmp_norm, cf = cf_nbc)
cons_0 = partial(cons_bop,tran_shk=tmp_lo, cf = cf_nbc)  
g = gg_funcs([cons_0,cons_1,cons_2],0.01,2.5, N=10, loc=robjects.r('c(1,0)'),
         ltitle = 'Temp Income',
         labels = ['Low','Normal', 'High'],
        title = "Consumption with Natural Borrowing Constraint",
        ylab = "Consumption", xlab = "Beginning-of-Period Assets",
        file_name = "cons_bop_nbc")
ggplot_notebook(g, height=300,width=400)



