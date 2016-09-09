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
from operator import sub, add
import pandas as pd
#this line errors out sometimes. Driven by issues with the Canopy_64bit path
from rpy2 import robjects
import rpy2.robjects.lib.ggplot2 as gg
from rpy2.robjects import pandas2ri
import make_plots as mp
import pickle
#read in HAMP parameters from google docs

#import gspread
#from oauth2client.service_account import ServiceAccountCredentials
#scope = ['https://spreadsheets.google.com/feeds']
#credentials = ServiceAccountCredentials.from_json_keyfile_name('gspread-oauth.json', scope)
#gc = gspread.authorize(credentials)
#g_params = gc.open("HAMPRA Model Parameters").sheet1 #in this case
#df = pd.DataFrame(g_params.get_all_records())
#pickle.dump( df, open("params_google_df.p", "wb" ) )
df = pickle.load( open( "params_google_df.p", "rb" ) )

hamp_params = df[['Param','Value']].set_index('Param')['Value'][:8].to_dict()
inc_params = df[['Param','Value']].set_index('Param')['Value'][8:11].to_dict()
hamp_coh = float(inc_params['cash_on_hand'])
boom_params = df[['Param','Value']].set_index('Param')['Value'][12:17].to_dict()
heloc_L = float(df[['Param','Value']].set_index('Param')['Value'][18:19])

#g_params = gc.open("HAMPRA Loan-to-Value Distribution")
#ltv_wksheet = g_params.worksheet("PythonInput")
#df_ltv = pd.DataFrame(ltv_wksheet.get_all_records())
#pickle.dump( df_ltv, open("params_google_df_ltv.p", "wb" ) )
df_ltv = pickle.load( open( "params_google_df_ltv.p", "rb" ) )

#ltv_params = df_ltv.set_index('LTV_Midpoint')

#xxx want to modify this default path not to have all these bad things starting out
#sys.path.remove('/Users/ganong/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages')
#sys.path.remove('/Users/ganong/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/PIL')
#xx I'd like to be able to move around parameters here but I haven't figured out how yet! Need to understand self method better.
#NBCExample.assignParameters(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac)  

    
###########################################################################
#set economic parameters 
###########################################################################
rebate_years_until_death = 25
age_of_rebate = 90 - rebate_years_until_death
t_eval = age_of_rebate - 25 - 20
def mpc(cF, rebate = 1, a = 0.1):
    return round((cF(a+rebate) - cF(a)) / rebate,3)
tmp_vlo = Params.IncUnemp
tmp_lo = 0.8
tmp_norm = 1
tmp_hi = 3
tmp_vhi = 6

def mpc_pih(cF, rebate = 1, a = 10):
    return round((cF(a+rebate) - cF(a)) / rebate,3)

settings.init()
settings.hsg_pay = 0.25 #code is currently designed to force choosing this value explicitly
settings.lil_verbose = True
settings.min_age, settings.max_age = 60, 65
###########################################################################
#plotting functions 
###########################################################################
#xx I tried to move to make_plots.py but failed. Not sure why.
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
    i = 0
    for function in function_list:
        if i > len(labels)-1:
            labels.append("func" + str(i))
        fig[labels[i]] = function(x)
        i=i+1
    fig = pd.melt(fig, id_vars=['x'])  
    g = gg.ggplot(fig) + \
        mp.base_plot + mp.line + mp.point +  \
        mp.theme_bw(base_size=9) + mp.fte_theme + \
        gg.labs(title=title,y=ylab,x=xlab) + mp.legend_f(loc) + mp.legend_t_c(ltitle) + mp.colors #+ mp.legend_t_s(ltitle) 
    if file_name is not None:
        mp.ggsave(file_name,g)
    return(g)
    
    

###########################################################################
#housing wealth functions
###########################################################################
#xx move to outside script and import in the future. got errors because I was modifying the wrong do file before...
#xxx the version in the default script is now more up to date
#remark: right now you are actually selling the house one year before retirement rather than at retirement. not sure if this is a problem.
def hsg_wealth(initial_debt, annual_hp_growth, collateral_constraint, baseline_debt, 
               initial_price, int_rate, pra_forgive, hsg_rent_p, hsg_own_p, 
               d_house_price = 0, age_at_mod = 45, hsg_pmt_wk_own = True, hsg_pmt_ret_y = False,
               annual_hp_growth_base = None):
    '''
    Calculates annual mortgage contract using parameters at date of mod.
    Everything is measured in years of income.
    
    Parameters
    ----------
    initial_debt : float
        Debt at date of mod
    annual_hp_growth : float
        Annual house price growth
    collateral_constraint : float
        Fraction of house needed for collateral (can borrow against rest)
    baseline_debt : float
        parameter from HAMP spreadsheet. Unused currently.        
    initial_price : float
        Value of house at origination
    int_rate: float
        Interest rate on liquid assets
    pra_forgive: float
        Amount of debt forgiven by PRA
    hsg_rental_rate : float
        int_rate minus annual_hp_growth
    age_at_mod : float
        Age at date of origination
                    
    Returns
    -------
    sale_proceeds : float
        Amount rebated at age 65 when house sold
    equity : list
        Value of house minus debt at each date (can be negative) 
    limit : list 
        Borrowing limit each period max{(1-collateral)*price - debt,0}
    mtg_pmt : list
        Annual mortgage payment from origination to age 65
    '''
    if initial_debt < 0:
        print("Error: cannot have negative mortgage")
        return
    price = [initial_price + d_house_price]
    if settings.verbose:
        print "Hsg wealth params: P=", price, " D=", baseline_debt, " g=", annual_hp_growth, " r=", int_rate, " phi=", collateral_constraint, " owner cost while work:", hsg_pmt_wk_own , " share inc while ret: ", hsg_pmt_ret_y
    T = 65 - age_at_mod
    
    #working life housing payments and collateral    
    debt = [initial_debt]
    amort = int_rate*(1+int_rate)**30/((1+int_rate)**30-1)
    hsg_pmt = [initial_debt*amort]
    for i in range(1,T):
        #print "age: " + str(i + age_at_mod) + " has growth fac: " + str(Params.PermGroFac[(i-1) + age_at_mod - 25])
        perm_gro = Params.PermGroFac[i + age_at_mod - 26]
        price.append(price[-1]*(1+annual_hp_growth)/perm_gro)
        hsg_pmt.append(hsg_pmt[-1]/perm_gro)
        debt.append((debt[-1]*(1+int_rate))/perm_gro - hsg_pmt[-1]) #xx double-check timing assumptions here
    equity = np.array(price) - np.array(debt)
    limit = np.min(np.vstack((-(np.array(price)*(1-collateral_constraint) - np.array(debt)),np.zeros(T))),axis=0).tolist()
    if hsg_pmt_wk_own:
        user_cost = [x * hamp_params['hsg_own_p'] for x in price]
        hsg_pmt = map(add, hsg_pmt, user_cost)
    if max(hsg_pmt) >= baseline_params['IncUnemp'] + 0.3:
        print("Error: cannot have housing payment > UI Benefit")
        print hsg_pmt
        return
    
    #housing payments in retirement
    if hsg_pmt_ret_y:
        if annual_hp_growth_base is None:
            annual_hp_growth_base = annual_hp_growth
        price_baseline = [initial_price]
        for i in range(1,T):
            perm_gro = Params.PermGroFac[i + age_at_mod - 26]
            price_baseline.append(price[-1]*(1+annual_hp_growth_base)/perm_gro)        
        hsg_pmt_ret = [hamp_params['hsg_rent_p']*price_baseline[-1]/Params.PermGroFac[39]]
        hsg_pmt_ret = hsg_pmt_ret * 26
    else:
        hsg_pmt_ret = [hamp_params['hsg_rent_p']*price[-1]*(1+annual_hp_growth)/Params.PermGroFac[39]]
        for i in range(25):
            hsg_pmt_ret.append(hsg_pmt_ret[-1]*(1+annual_hp_growth))  
    if max(hsg_pmt_ret) >= 1:
        print("Error: cannot have housing payment > income")
        print hsg_pmt_ret  
        return
    #fill out arguments to return
    hsg_pmt = [0.0] * (age_at_mod - 26) + hsg_pmt + hsg_pmt_ret
    equity = [0.0] * (age_at_mod - 26) + equity.tolist() + [0.0] * 26
    limit = [0.0] * (age_at_mod - 26) + limit + [0.0] * 26
    sale_proceeds = max(equity[38],0)
    #print hsg_pmt
    return sale_proceeds, equity, limit, hsg_pmt


def pra_pmt(annual_hp_growth, collateral_constraint, baseline_debt, initial_price, 
            int_rate, pra_forgive, hsg_rent_p, hsg_own_p, hsg_pmt_wk_own = True, hsg_pmt_ret_y = False,
            age = 45, forgive = hamp_params['pra_forgive']):
    r, e, L, d_hamp = hsg_wealth(initial_debt =  hamp_params['baseline_debt'], age_at_mod = age, hsg_pmt_wk_own = hsg_pmt_wk_own, hsg_pmt_ret_y = hsg_pmt_ret_y, **hamp_params)
    r, e, L, d_prin_red = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - forgive, age_at_mod = age, hsg_pmt_wk_own = hsg_pmt_wk_own, hsg_pmt_ret_y = hsg_pmt_ret_y, **hamp_params)
    pra_pmt = d_prin_red[:age-26] + d_hamp[age-26:age-21] + d_prin_red[age-21:]
    return pra_pmt
#pra_pmt(age = 45, forgive = 1, **hamp_params)[20:40]

#reload(Params)  
baseline_params = Params.init_consumer_objects

#construct results w default specification for how to set up house prices (neg wealth effect)
uw_house_params = deepcopy(baseline_params)
uw_house_params['rebate_amt'], e, uw_house_params['BoroCnstArt'], uw_house_params['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'], **hamp_params)
pra_params = deepcopy(baseline_params)
pra_params['rebate_amt'], e, pra_params['BoroCnstArt'], pra_params['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - hamp_params['pra_forgive'], **hamp_params)
pra_params['HsgPay'] = pra_pmt(age = 45, **hamp_params)

#slide 3 -- housing equity 
labels = ["Payment Reduction", "Payment & Principal Reduction"]
def neg(x): return -1*x
boro_cnst_pre_pra = LinearInterp(np.arange(25,90),list(map(neg, uw_house_params['BoroCnstArt'])))
boro_cnst_post_pra = LinearInterp(np.arange(25,90),list(map(neg, pra_params['BoroCnstArt'])))
g = gg_funcs([boro_cnst_pre_pra,boro_cnst_post_pra],
              45.001,75, N=round(75-45.001), loc=robjects.r('c(0,0.5)'),
        title = "Borrowing Limits \n Receive Treatment at Age 45",
        labels = labels,
        ylab = "Borrowing Limit (Years of Income)", xlab = "Age", file_name = "borrowing_limits_and_pra_diag")
#ggplot_notebook(g, height=300,width=400)
g = gg_funcs([boro_cnst_pre_pra,boro_cnst_post_pra],
              45.001,64, N=round(65-44.001), loc=robjects.r('c(0,0.5)'),
        title = "Borrowing Limits \n Receive Treatment at Age 45",
        labels = labels,
        ylab = "Borrowing Limit (Years of Income)", xlab = "Age", file_name = "borrowing_limits_and_pra")
ggplot_notebook(g, height=300,width=400)

#slide 4 -- housing payments 
pmt_pre_pra =  LinearInterp(np.arange(25,90),uw_house_params['HsgPay']) 
pmt_post_pra = LinearInterp(np.arange(25,90),pra_pmt(age = 45, **hamp_params))
g = gg_funcs([pmt_pre_pra,pmt_post_pra],
              45.0001,75, N=round(75-45.001), loc=robjects.r('c(1,1)'),
        title = "Mortgage Payments \n Receive Treatment at Age 45",
        labels = labels,
        ylab = "Payment As Share of Income", xlab = "Age", file_name = "hsg_pmt_and_pra_diag")
ggplot_notebook(g, height=300,width=400)
g = gg_funcs([pmt_pre_pra,pmt_post_pra],
              45.0001,65, N=round(65-45.001), loc=robjects.r('c(1,1)'),
        title = "Mortgage Payments \n Receive Treatment at Age 45",
        labels = labels,
        ylab = "Payment As Share of Income", xlab = "Age", file_name = "hsg_pmt_and_pra")
g += gg.ylim(robjects.r('c(0.1,0.25)'))
ggplot_notebook(g, height=300,width=400)

#construct change in payments and change in borrowing limits
d_boro_cnst = LinearInterp(np.arange(25,90),map(sub, uw_house_params['BoroCnstArt'],pra_params['BoroCnstArt']))
d_pmt =  LinearInterp(np.arange(25,90),map(sub,pra_pmt(age = 45, **hamp_params),uw_house_params['HsgPay'])) 
g = gg_funcs([d_boro_cnst,d_pmt],
              45.0001,65, N=round(65-45.001), loc=robjects.r('c(0,1)'),
        title = "Impact of Principal Forgiveness on Mortgage Payments and Borrowing Limits",
        labels = ["Borrowing Constraint", "Annual Payment"],
        ylab = "Change As Share of Annual Income", xlab = "Age", file_name = "suff_stat_for_pra")
ggplot_notebook(g, height=300,width=400)

#housing payments diagnostic
params_hsg_own_inc = deepcopy(baseline_params)
params_hsg_own_inc['rebate_amt'], e, params_hsg_own_inc['BoroCnstArt'], params_hsg_own_inc['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'],\
               hsg_pmt_wk_own = True, hsg_pmt_ret_y = True, **hamp_params)
params_hsg_mtg_inc = deepcopy(baseline_params)               
params_hsg_mtg_inc['rebate_amt'], e, params_hsg_mtg_inc['BoroCnstArt'], params_hsg_mtg_inc['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'], \
               hsg_pmt_wk_own = False, hsg_pmt_ret_y = True, **hamp_params)
pmt_own_inc =  LinearInterp(np.arange(25,90),params_hsg_own_inc['HsgPay']) 
pmt_mtg_inc =  LinearInterp(np.arange(25,90),params_hsg_mtg_inc['HsgPay']) 
g = gg_funcs([pmt_pre_pra,pmt_own_inc,pmt_mtg_inc],
              44.0001,75, N=round(75-44.001), loc=robjects.r('c(0,0.4)'),
        title = "Mortgage Payments",
        labels = ["Work: Owner Cost, Ret: User Cost", "Work: Owner Cost, Ret: Inc Share","Work: Mortgage, Ret: Inc Share"],
        ylab = "Payment As Share of Income", xlab = "Age", file_name = "hsg_pmt_diag")
ggplot_notebook(g, height=300,width=400)




###########################################################################
# Solve consumer problems
pandas2ri.activate() 

def solve_unpack(params):
    settings.rebate_size = params['rebate_amt']
    settings.t_rebate = params['rebate_age']
    params['Rfree'] = 1+ hamp_params['int_rate']
    if settings.lil_verbose:
        print "Rebate is " + str(round(settings.rebate_size,2)) + " at age " + str(90-settings.t_rebate) #+ " & int rate is: " + str(params['Rfree'])
    IndShockConsumerType = Model.IndShockConsumerType(**params)
    IndShockConsumerType.solve()
    IndShockConsumerType.unpack_cFunc()
    IndShockConsumerType.timeFwd()
    return IndShockConsumerType
    
#rrr want to reset this back to a lower number now to see what happens to speed
baseline_params['aXtraCount'] = 30
baseline_params['DiscFac'] = (np.ones(65)*0.96).tolist()
baseline_params['vFuncBool'] = True
settings.verbose = False
#baseline_params['IncUnemp'] = inc_params['inc_unemp']
#baseline_params['UnempPrb'] = inc_params['prb_unemp']

IndShockExample = solve_unpack(baseline_params)
#function call
mpc_pih(IndShockExample.cFunc[0]), mpc_pih(IndShockExample.cFunc[20]), mpc_pih(IndShockExample.cFunc[40])
#RESULTS
#baseline Carroll (perm inc risk on, baseline disc fac timevary)  (0.035, 0.044, 0.059)
#perm inc risk off, baseline disc fac param: (.033,.040,.059)
#perm inc risk off, disc fac 0.96: (.059,.044,.065)
#perm inc risk on, disc fac 0.96: (0.045, 0.048, 0.065)
#perm inc risk on, disc fac 0.96, R = 1.01: (0.035, 0.039, 0.058)
#perm inc risk on, disc fac 0.96, R = 1.02: (0.04, 0.017, 0.061)
#perm inc risk off, disc fac 0.96, R = 1.01: (0.059, 0.035, 0.058)


#alternative consumption functions
example_params = deepcopy(baseline_params)
example_params['rebate_age'] = 39
example_params['rebate_amt'] = 1
RebateAge51 = solve_unpack(example_params)
example_params['rebate_age'] = 44
RebateAge46 = solve_unpack(example_params)
settings.t_rebate = rebate_years_until_death
grant_now = lambda x: IndShockExample.cFunc[t_eval](x+1)

#relax collateral constraint
example_params = deepcopy(baseline_params)
example_params['rebate_amt'] = 0
l = example_params['BoroCnstArt']
for i in range(len(l)):
    l[i] = -1
Boro1YrInc = solve_unpack(example_params)
l = example_params['BoroCnstArt']
for i in range(len(l)):
    l[i] = heloc_L
Boro_heloc = solve_unpack(example_params)

#slide 1.1 -- consumption function out of future wealth
g = gg_funcs([IndShockExample.cFunc[t_eval]],
        -0.001,3, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Wealth & Collateral",
        labels = ["Baseline","Grant 0 Years Away (Now)"],
        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "cf_fut_slide1")
ggplot_notebook(g, height=300,width=400)

#slide 1.2 -- consumption function out of future wealth
g = gg_funcs([IndShockExample.cFunc[t_eval],grant_now],
        -0.001,3, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Wealth & Collateral",
        labels = ["Baseline","Grant 0 Years Away (Now)"],
        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "cf_fut_slide2")
ggplot_notebook(g, height=300,width=400)

#slide 1.3 -- consumption function out of future wealth
g = gg_funcs([IndShockExample.cFunc[t_eval],grant_now,RebateAge46.cFunc[t_eval]],
        -0.001,3, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Wealth & Collateral",
        labels = ["Baseline","Grant 0 Years Away (Now)","Grant 1 Year Away", "Grant 6 Years Away"],
        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "cf_fut_slide3")
ggplot_notebook(g, height=300,width=400)

#slide 1.4 -- consumption function out of future wealth
g = gg_funcs([IndShockExample.cFunc[t_eval],grant_now,RebateAge46.cFunc[t_eval],RebateAge51.cFunc[t_eval]],
        -0.001,3, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Wealth & Collateral",
        labels = ["Baseline","Grant 0 Years Away (Now)","Grant 1 Year Away", "Grant 6 Years Away"],
        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "cf_fut_slide4")
ggplot_notebook(g, height=300,width=400)

#slide 1.5 -- consumption function out of future wealth
g = gg_funcs([IndShockExample.cFunc[t_eval],grant_now,RebateAge46.cFunc[t_eval],RebateAge51.cFunc[t_eval],Boro1YrInc.cFunc[t_eval]],
        -0.001,3, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Wealth & Collateral",
        labels = ["Baseline","Grant 0 Years Away (Now)","Grant 1 Year Away", "Grant 6 Years Away","Raise Collateral Now"],
        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "cf_fut_slide5")
ggplot_notebook(g, height=300,width=400)

#consumption function w and without HELOC
g = gg_funcs([IndShockExample.cFunc[t_eval],Boro_heloc.cFunc[t_eval]],
        -0.001,3, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Wealth & Collateral",
        labels = ["Baseline","HELOC Borrow Limit: " + str(heloc_L)],
        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "cf_heloc_diag")
ggplot_notebook(g, height=300,width=400)

#####################################
#analyze principal forgiveness
######################################

cFuncs = []
cFuncs_w = []
cFuncs_L = []
cFuncs_rL = []
cFuncs_0_pct =[]
cFuncs_heloc =[]
hw_cf_list = []
hw_cf_coh_hi_list = []
hw_cf_coh_vhi_list = []
hw_cf_w_list = []
hw_cf_rL_list = []
hw_cf_L_list = []
hw_cf_0_pct_list = []
hw_cf_heloc_list = []
grid_len = 20
grid_int = 0.25
grid_max = grid_len*grid_int

#rrr switch to a class
for i in range(grid_len):
    #full specification
    hw_cf_params = deepcopy(baseline_params)
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - i*grid_int, **hamp_params)
    hw_cf_params['HsgPay'] =  pra_pmt(age = 45, forgive = i*grid_int, **hamp_params)   
    cf = solve_unpack(hw_cf_params)
    cFuncs.append(cf.cFunc)
    hw_cf_list.append(cf.cFunc[t_eval](hamp_coh))
    hw_cf_coh_hi_list.append(cf.cFunc[t_eval](tmp_hi))
    hw_cf_coh_vhi_list.append(cf.cFunc[t_eval](6))

    #full specification with heloc
    hw_cf_params['BoroCnstArt']   = map(lambda x:x-heloc_L, hw_cf_params['BoroCnstArt'])
    cf = solve_unpack(hw_cf_params)
    cFuncs_heloc.append(cf.cFunc)
    hw_cf_heloc_list.append(cf.cFunc[t_eval](hamp_coh))

    #cash payments only
    hw_cf_params = deepcopy(baseline_params)
    hw_cf_params['rebate_amt'], e, L, d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - i*grid_int, **hamp_params)
    hw_cf_params['HsgPay'] =  pra_pmt(age = 45, forgive = i*grid_int, **hamp_params)   
    cf = solve_unpack(hw_cf_params)
    cFuncs_w.append(cf.cFunc)
    hw_cf_w_list.append(cf.cFunc[t_eval](hamp_coh))
    
    #collateral only
    hw_cf_params = deepcopy(baseline_params)
    r, e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - i*grid_int, **hamp_params)
    hw_cf_params['HsgPay']  = uw_house_params['HsgPay']
    hw_cf_params['rebate_amt'] = 0
    cf = solve_unpack(hw_cf_params)
    cFuncs_L.append(cf.cFunc)
    hw_cf_L_list.append(cf.cFunc[t_eval](hamp_coh))
    
    #collateral and rebate specification
    hw_cf_params = deepcopy(baseline_params)
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - i*grid_int, **hamp_params)
    hw_cf_params['HsgPay']  = uw_house_params['HsgPay']
    cf = solve_unpack(hw_cf_params)
    cFuncs_rL.append(cf.cFunc)
    hw_cf_rL_list.append(cf.cFunc[t_eval](hamp_coh))
    
    #0% LTV specification
    hamp_params['collateral_constraint'] = 0
    hw_cf_params = deepcopy(baseline_params)
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - i*grid_int, **hamp_params)
    hw_cf_params['HsgPay'] =  pra_pmt(age = 45, forgive = i*grid_int, **hamp_params)   
    cf = solve_unpack(hw_cf_params)
    cFuncs_0_pct.append(cf.cFunc)
    hw_cf_0_pct_list.append(cf.cFunc[t_eval](hamp_coh))
    hamp_params['collateral_constraint'] = 0.20
    

    
    
#first index says. second index says age 44, I believe 
#t_eval = 19
equity_initial = hamp_params['baseline_debt'] - hamp_params['initial_price']
gr_min = 100*-equity_initial/hamp_params['initial_price']
gr_max = 100*(grid_len*grid_int - equity_initial)/hamp_params['initial_price']
grid_int2 = (gr_max-gr_min)/grid_len
hw_cf = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_list))
hw_cf_coh_hi = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_coh_hi_list))
hw_cf_coh_vhi = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_coh_vhi_list))
#xxx this is a temporary hack to deal with a bump in this function
hw_cf_coh_hi_hack = LinearInterp(np.arange(gr_min+ 2*grid_int2,gr_max,grid_int2),np.array(hw_cf_coh_hi_list[2:]), lower_extrap = True)
hw_cf_w = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_w_list))
hw_cf_L = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_L_list))
hw_cf_rL = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_rL_list))
hw_cf_0_pct = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_0_pct_list))
hw_cf_heloc = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_heloc_list))

ltv_start = -100*(hamp_params['baseline_debt']/hamp_params['initial_price']-1)
ltv_end = -100*((hamp_params['baseline_debt'] - hamp_params['pra_forgive'])/hamp_params['initial_price']-1)

#slide 5 -- Consumption function out of principal forgiveness
g = gg_funcs(hw_cf,gr_min,gr_max, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Principal Forgiveness",
        labels = ["Baseline"],
        ylab = "Consumption", xlab = "Housing Equity Position (< 0 is Underwater)")
g += gg.geom_vline(xintercept=hamp_params['collateral_constraint']*100, linetype=2, colour="#66C2A5", alpha=0.75)
mp.ggsave("cons_and_prin_forgive",g)
ggplot_notebook(g, height=300,width=400)
g += gg.geom_segment(gg.aes_string(x = ltv_start, y = hw_cf_list[0] + 0.03, xend = ltv_end, yend =  hw_cf_list[0] + 0.03),
                     arrow = robjects.r('arrow(length = unit(0.5, "cm"))'),
                     color= robjects.r.palette_lines[1])
mp.ggsave("cons_and_prin_forgive_arrow",g)
ggplot_notebook(g, height=300,width=400)


g = gg_funcs([hw_cf,hw_cf_0_pct],gr_min,gr_max, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Principal Forgiveness",
        labels = ["Baseline","Housing Equity >= 0%"],
        ylab = "Consumption", xlab = "Housing Equity Position (< 0 is Underwater)")
g += gg.geom_vline(xintercept=hamp_params['collateral_constraint']*100, linetype=2, colour="#66C2A5", alpha=0.75)
g += gg.geom_vline(xintercept=0, linetype=2, colour="#FC8D62", alpha=0.75)
mp.ggsave("cons_and_prin_forgive_0-pct",g)
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([hw_cf,hw_cf_0_pct, hw_cf_coh_hi_hack],gr_min,gr_max, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Principal Forgiveness",
        labels = ["Baseline","Housing Equity >= 0%","PIH: Cash-on-Hand = " + str(tmp_hi)],
        ylab = "Consumption", xlab = "Housing Equity Position (< 0 is Underwater)")
g += gg.geom_vline(xintercept=hamp_params['collateral_constraint']*100, linetype=2, colour="#66C2A5", alpha=0.75)
g += gg.geom_vline(xintercept=0, linetype=2, colour="#FC8D62", alpha=0.75)
mp.ggsave("cons_and_prin_forgive_0-pct_coh-hi",g)
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([hw_cf,hw_cf_0_pct, hw_cf_coh_hi_hack,hw_cf_coh_vhi],gr_min,gr_max, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Function Out of Principal Forgiveness",
        labels = ["Baseline","Housing Equity >= 0%","PIH: Cash-on-Hand = " + str(tmp_hi), "PIH: Cash-on-Hand = 6"],
        ylab = "Consumption", xlab = "Housing Equity Position (< 0 is Underwater)")
g += gg.geom_vline(xintercept=hamp_params['collateral_constraint']*100, linetype=2, colour="#66C2A5", alpha=0.75)
g += gg.geom_vline(xintercept=0, linetype=2, colour="#FC8D62", alpha=0.75)
mp.ggsave("cons_and_prin_forgive_0-pct_coh-hi_diag",g)
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([hw_cf,hw_cf_w,hw_cf_L],gr_min,gr_max, N=50, loc=robjects.r('c(0,1)'), #hw_cf_rL
        title = "Consumption Function Out of Principal Forgiveness: Channels",
        labels = ["Collateral & Future Cash","Future Cash Only","Collateral Only"],
        ylab = "Consumption", xlab = "Housing Equity Position (< 0 is Underwater)")
g += gg.geom_vline(xintercept=hamp_params['collateral_constraint']*100, linetype=2, colour="#66C2A5", alpha=0.75)
mp.ggsave("cons_and_prin_forgive_decomp",g)
ggplot_notebook(g, height=300,width=400)


########## diagnostic plots########## 
#below threshold
cFuncsBelow = [cFuncs[0][t_eval],cFuncs_w[0][t_eval],cFuncs_L[0][t_eval],cFuncs_rL[0][t_eval]]
g = gg_funcs(cFuncsBelow,-1.5,3, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Functions For Underwater HHs",
        labels = ["Full Mod (Collateral, Rebates, Payments)","Rebates and Payments Only","Collateral Only", "Collateral and Rebates Only"],
        ylab = "Consumption", xlab = "Cash on Hand", file_name = "cfuncs_below_collat_threshold_diag")
ggplot_notebook(g, height=300,width=400)
#
##at LTV 80
cFuncsAt = [cFuncs[11][t_eval],cFuncs_w[11][t_eval],cFuncs_L[11][t_eval],cFuncs_rL[11][t_eval]]
g = gg_funcs(cFuncsAt,-1.5,3, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Functions For HHs At LTV = 80",
        labels = ["Full Mod (Collateral, Rebates, Payments)","Rebates and Payments Only","Collateral Only", "Collateral and Rebates Only"],
        ylab = "Consumption", xlab = "Cash on Hand", file_name = "cfuncs_at_collat_threshold_diag")
ggplot_notebook(g, height=300,width=400)

#far above threshold
cFuncsAbove = [cFuncs[19][t_eval],cFuncs_w[19][t_eval],cFuncs_L[19][t_eval],cFuncs_rL[19][t_eval]]
g = gg_funcs(cFuncsAbove,-2.001,3, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Functions For HHs At LTV = 0",
        labels = ["Full Mod (Collateral, Rebates, Payments)","Rebates and Payments Only","Collateral Only", "Collateral and Rebates Only"],
        ylab = "Consumption", xlab = "Cash on Hand", file_name = "cfuncs_above_collat_threshold_diag")
ggplot_notebook(g, height=300,width=400)

# consumption functions
cFuncs44 = []
for i in range(0,grid_len-4,2):
    cFuncs44.append(cFuncs[i][t_eval])
g = gg_funcs(cFuncs44,-1.5,6, N=200, loc=robjects.r('c(1,0)'),
        title = "Consumption Functions. Each line is 0.5 more of Principal Forgiveness",
        ylab = "Consumption", xlab = "Cash on Hand", file_name = "cfuncs_prin_forgive_diag")
ggplot_notebook(g, height=300,width=400)

cFuncs44 = []
for i in range(7):
    cFuncs44.append(cFuncs[i][t_eval])
g = gg_funcs(cFuncs44,2.01,4, N=200, loc=robjects.r('c(1,0)'),
        title = "Consumption Functions. Each line is 0.25 more of Principal Forgiveness",
        ylab = "Consumption", xlab = "Cash on Hand", file_name = "cfuncs_prin_forgive_high_coh_diag")
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([hw_cf,hw_cf_heloc],gr_min,gr_max, N=50, loc=robjects.r('c(0,1)'), #hw_cf_rL
        title = "Consumption Function Out of Principal Forgiveness -- Agent w HELOC",
        labels = ["Baseline","Agent has HELOC"],
        ylab = "Consumption", xlab = "Housing Equity Position (< 0 is Underwater)")
g += gg.geom_vline(xintercept=hamp_params['collateral_constraint']*100, linetype=2, colour="red", alpha=0.25)
mp.ggsave("cons_and_prin_forgive_heloc_diag",g)
ggplot_notebook(g, height=300,width=400)


######################################
##analyze house price increases
#######################################
cFuncs = []
cFuncs_pos_w = []
hp_cf_list = []
hp_cf_pos_w_list = []
hw_cf_pos_w_list = []
grid_len = 20
grid_int = 0.25
grid_max = grid_len*grid_int
#housing payments identifcal in both cases
#raising house prices induces a larger age 65 rebate and smaller opportunity to borrow along the way
for i in range(grid_len): #
    print "raise house prices by ", i*grid_int
    hw_cf_params = deepcopy(baseline_params)
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'], d_house_price = i*grid_int, **hamp_params)
    cf = solve_unpack(hw_cf_params)
    cFuncs.append(cf.cFunc)
    hp_cf_list.append(cf.cFunc[t_eval](hamp_coh))

    #house prices in pos wealth effect spec
    hp_pos_w_params = deepcopy(baseline_params)
    hp_pos_w_params['rebate_amt'], e, hp_pos_w_params['BoroCnstArt'], d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - i*grid_int, hsg_pmt_wk_own = False, hsg_pmt_ret_y = True, **hamp_params)
    hp_pos_w_params['HsgPay']  = uw_house_params['HsgPay']
    cf = solve_unpack(hp_pos_w_params)
    cFuncs_pos_w.append(cf.cFunc)
    hp_cf_pos_w_list.append(cf.cFunc[t_eval](hamp_coh))
    
    #prin forgive in pos wealth effect spec
    hw_cf_params = deepcopy(baseline_params)
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - i*grid_int, hsg_pmt_wk_own = False, hsg_pmt_ret_y = True, **hamp_params)
    hw_cf_params['HsgPay'] =  pra_pmt(age = 45, forgive = i*grid_int, **hamp_params)   
    cf = solve_unpack(hw_cf_params)
    cFuncs.append(cf.cFunc)
    hw_cf_pos_w_list.append(cf.cFunc[t_eval](hamp_coh))
  
gr_min = 0
gr_max =  grid_len*grid_int
grid_int2 = (gr_max-gr_min)/grid_len
hp_cf = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hp_cf_list))
hw_cf_yrs = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_list))
hp_cf_pos_w = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hp_cf_pos_w_list))
hw_cf_pos_w = LinearInterp(np.arange(gr_min,gr_max,grid_int2),np.array(hw_cf_pos_w_list))


g = gg_funcs([hw_cf_yrs,hw_cf_pos_w,hp_cf,hp_cf_pos_w],gr_min,gr_max, N=50, loc=robjects.r('c(0,1)'),
        title = "Consumption Function Out of Principal Forgiveness",
        labels = ["Principal Forgiveness (Neg Wealth Effect, Baseline)","Principal Forgiveness (Pos Wealth Effect)","House Price Increase (Neg Wealth Effect)","House Price Increase (Pos Wealth Effect)"],
        ylab = "Consumption", xlab = "House Price Increase Measured In Years of Income",
        file_name = "cons_dprice_forgive_years_of_inc_diag")
ggplot_notebook(g, height=300,width=400)


cFuncs44 = []
for i in range(0,grid_len-4,2):
    cFuncs44.append(cFuncs[i][t_eval])
g = gg_funcs(cFuncs44,-1.5,8, N=50, loc=robjects.r('c(0,1)'),
        title = "Consumption Functions. Each line is 0.5 Increase in house value",
        ylab = "Consumption", xlab = "Cash on Hand", file_name = "cfuncs_house_price_diag")
ggplot_notebook(g, height=300,width=400)


############################################################################
## Housing MPC and Cash MPC by LTV
############################################################################
ltv_rows = range(10,170,10)
#ltv_rows = list(df_ltv['LTV_Midpoint'])
equity_a = np.arange(-60,100,10) #np.arange(-60,90,10)
index = pd.Index(ltv_rows, name='rows')
scenarios = ['cash','hsg','debt']
columns = pd.Index(scenarios, name='cols')
dhp = 0.25
coh = 2

hp_mpc = pd.DataFrame(np.zeros((len(ltv_rows), len(scenarios))), index=index, columns=columns) 
hp_mpc_low_coh = pd.DataFrame(np.zeros((len(ltv_rows), len(scenarios))), index=index, columns=columns) 
for eq in ltv_rows: #
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  (eq/100.)*hamp_params['initial_price'] , **hamp_params)
    hsg_pay_pre_neg = deepcopy(hw_cf_params['HsgPay'])
    #settings.verbose = True
    cf_pre = solve_unpack(hw_cf_params)  
    #settings.verbose = False
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt = (eq/100.)*hamp_params['initial_price'] - dhp, **hamp_params)
    cf_debt = solve_unpack(hw_cf_params)    
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt = (eq/100.)*hamp_params['initial_price'],  d_house_price = dhp, **hamp_params)
    cf_hp = solve_unpack(hw_cf_params)
    #print "Change in housing payment after HP increase, spec: negative", map(sub,hw_cf_params['HsgPay'],hsg_pay_pre_neg)
    hp_mpc.loc[eq,'hsg'] = (cf_hp.cFunc[t_eval](coh) - cf_pre.cFunc[t_eval](coh))/dhp
    hp_mpc.loc[eq,'debt'] = (cf_debt.cFunc[t_eval](coh) - cf_pre.cFunc[t_eval](coh))/dhp
    hp_mpc.loc[eq,'cash'] = (cf_pre.cFunc[t_eval](coh+dhp) - cf_pre.cFunc[t_eval](coh))/dhp
    
    hp_mpc_low_coh.loc[eq,'hsg'] = (cf_hp.cFunc[t_eval](hamp_coh) - cf_pre.cFunc[t_eval](hamp_coh))/dhp
    hp_mpc_low_coh.loc[eq,'debt'] = (cf_debt.cFunc[t_eval](hamp_coh) - cf_pre.cFunc[t_eval](hamp_coh))/dhp
    hp_mpc_low_coh.loc[eq,'cash'] = (cf_pre.cFunc[t_eval](hamp_coh+dhp) - cf_pre.cFunc[t_eval](hamp_coh))/dhp
    
    #zero wealth shock spec
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  (eq/100.)*hamp_params['initial_price'] ,  hsg_pmt_wk_own = True, hsg_pmt_ret_y = True, **hamp_params)
    hsg_pay_pre_zero = deepcopy(hw_cf_params['HsgPay'])  
    cf_pre_zero = solve_unpack(hw_cf_params)
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt = (eq/100.)*hamp_params['initial_price'],  d_house_price = dhp, hsg_pmt_wk_own = True, hsg_pmt_ret_y = True, **hamp_params)
    hw_cf_params['HsgPay'] = hw_cf_params['HsgPay'][:39] + hsg_pay_pre_zero[39:]   
    #print "Change in housing payment after HP increase, spec: zero", map(sub,hw_cf_params['HsgPay'],hsg_pay_pre_zero)   
    cf_hp_zero = solve_unpack(hw_cf_params)
    
    #positive wealth shock spec
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  (eq/100.)*hamp_params['initial_price'] ,  hsg_pmt_wk_own = False, hsg_pmt_ret_y = True, **hamp_params)
    hsg_pay_pre_pos = deepcopy(hw_cf_params['HsgPay'])
    cf_pre_pos = solve_unpack(hw_cf_params)
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt = (eq/100.)*hamp_params['initial_price'],  d_house_price = dhp, hsg_pmt_wk_own = False, hsg_pmt_ret_y = True, **hamp_params)
    hw_cf_params['HsgPay'] = hsg_pay_pre_pos  
    #print "Change in housing payment after HP increase, spec: pos", map(sub,hw_cf_params['HsgPay'],hsg_pay_pre_pos)
    cf_hp_pos = solve_unpack(hw_cf_params)
    
    hp_mpc.loc[eq,'hsg_pos'] = (cf_hp_pos.cFunc[t_eval](coh) - cf_pre_pos.cFunc[t_eval](coh))/dhp
    hp_mpc.loc[eq,'hsg_zero'] = (cf_hp_zero.cFunc[t_eval](coh) - cf_pre_zero.cFunc[t_eval](coh))/dhp
    hp_mpc_low_coh.loc[eq,'hsg_pos'] = (cf_hp_pos.cFunc[t_eval](hamp_coh) - cf_pre_pos.cFunc[t_eval](hamp_coh))/dhp
    hp_mpc_low_coh.loc[eq,'hsg_zero'] = (cf_hp_zero.cFunc[t_eval](hamp_coh) - cf_pre_zero.cFunc[t_eval](hamp_coh))/dhp


hp_mpc.to_csv("~/dropbox/hampra/out2/tbl_hp_mpc.csv")
hp_mpc
hp_mpc_low_coh

#make table for CSV output
ltv_rows_alt = list(df_ltv['LTV_Midpoint'])
index_alt = pd.Index(ltv_rows_alt, name='rows')
hp_mpc_tbl = pd.DataFrame(np.zeros((len(ltv_rows_alt), len(scenarios))), index=index_alt, columns=columns) 
for eq in ltv_rows_alt: #
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  (eq/100.)*hamp_params['initial_price'] , **hamp_params)
    hsg_pay_pre_neg = deepcopy(hw_cf_params['HsgPay'])
    cf_pre = solve_unpack(hw_cf_params)  
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt = (eq/100.)*hamp_params['initial_price'] - dhp, **hamp_params)
    cf_debt = solve_unpack(hw_cf_params)    
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt = (eq/100.)*hamp_params['initial_price'],  d_house_price = dhp, **hamp_params)
    cf_hp = solve_unpack(hw_cf_params)
    hp_mpc_tbl.loc[eq,'hsg'] = (cf_hp.cFunc[t_eval](coh) - cf_pre.cFunc[t_eval](coh))/dhp
    hp_mpc_tbl.loc[eq,'debt'] = (cf_debt.cFunc[t_eval](coh) - cf_pre.cFunc[t_eval](coh))/dhp
    hp_mpc_tbl.loc[eq,'cash'] = (cf_pre.cFunc[t_eval](coh+dhp) - cf_pre.cFunc[t_eval](coh))/dhp

hp_mpc_tbl


#average MPC out of housing wealth right now is 3.6 cents w 4% user cost and 6.8 cents with 1.5% user cost
hp_mpc_tbl_ltv = pd.merge(df_ltv, hp_mpc_tbl, left_on="LTV_Midpoint", right_index=True)
hp_mpc_tbl_ltv = hp_mpc_tbl_ltv[df_ltv['LTV_Midpoint'] <= 95]
hp_mpc_tbl_ltv.loc[:,'hsg'] = hp_mpc_tbl_ltv['hsg']*hp_mpc_tbl_ltv['Share_2005']/sum(hp_mpc_tbl_ltv['Share_2005'])
hp_mpc_tbl_ltv.loc[:,'cash'] = hp_mpc_tbl_ltv['cash']*hp_mpc_tbl_ltv['Share_2005']/sum(hp_mpc_tbl_ltv['Share_2005'])
hp_mpc_tbl_ltv.loc[:,'debt'] = hp_mpc_tbl_ltv['debt']*hp_mpc_tbl_ltv['Share_2005']/sum(hp_mpc_tbl_ltv['Share_2005'])

hp_mpc_tbl_sum = pd.DataFrame({"Mean": hp_mpc_tbl_ltv[['cash','hsg','debt']].sum(),
                                "Levered":hp_mpc_tbl.loc[95,:],
                               "Underwater":hp_mpc_tbl.loc[150,:]}).round(3)
hp_mpc_tbl_sum.to_csv("~/dropbox/hampra/out2/tbl_mpc_ltv.csv")


mpc_hsg_f = LinearInterp(equity_a,np.array(hp_mpc['hsg'])[::-1])
mpc_cash_f = LinearInterp(equity_a,np.array(hp_mpc['cash'])[::-1])
mpc_debt_f = LinearInterp(equity_a,np.array(hp_mpc['debt'])[::-1])

#xxx consider turning arrow into an mp item
g = gg_funcs([mpc_hsg_f,mpc_cash_f,mpc_debt_f],-60,90, N=len(ltv_rows), loc=robjects.r('c(1,1)'),
        title = "Marginal Propensity to Consume by Home Equity\n Cash-On-Hand = " + str(coh),
        labels = ["Housing Price MPC","Cash MPC","Housing Debt MPC"],
        ylab = "MPC", xlab = "Home Equity (< 0 is Underwater)",
        file_name = "mpc_cash_hsg")
ggplot_notebook(g, height=300,width=400)
g += mp.annotate(geom = "text", x = 40, y = 0.02, label = "MPC\n Avg = 6.8 cents")
g += mp.annotate(geom = "text", x = 5, y = 0.05, label = "MPC\n Levered / Avg = 1.5")
g += gg.geom_segment(gg.aes_string(x = 0, y = 0.07, xend = 0, yend = 0.10),
                     arrow = robjects.r('arrow(length = unit(0.4, "cm"))'),
                     color= robjects.r.palette_lines[1])
mp.ggsave("mpc_cash_hsg_arrow_backup",g)
ggplot_notebook(g, height=300,width=400)


mpc_hsg_pos_f = LinearInterp(equity_a,np.array(hp_mpc['hsg_pos'])[::-1])
#mpc_hsg_zero_f = LinearInterp(equity_a,np.array(hp_mpc['hsg_zero'])[::-1])
g = gg_funcs([mpc_debt_f,mpc_hsg_f,mpc_hsg_pos_f],-60,90, N=len(ltv_rows), loc=robjects.r('c(0,1)'), #mpc_hsg_zero_f
        title = "Marginal Propensity to Consume Out of Housing by Home Equity",
        labels = ["Debt","Hsg Neg Wealth Effect","Hsg Pos Wealth Effect"], #"Hsg Zero Wealth Effect",
        ylab = "MPC", xlab = "Home Equity  (< 0 is Underwater)",
        file_name = "mpc_hsg_backup")
ggplot_notebook(g, height=300,width=400)



mpc_hsg_low_coh_f = LinearInterp(equity_a,np.array(hp_mpc_low_coh['hsg'])[::-1])
mpc_cash_low_coh_f = LinearInterp(equity_a,np.array(hp_mpc_low_coh['cash'])[::-1])
mpc_debt_low_coh_f = LinearInterp(equity_a,np.array(hp_mpc_low_coh['debt'])[::-1])
g = gg_funcs([mpc_cash_low_coh_f,mpc_debt_low_coh_f],-60,90, N=len(ltv_rows), loc=robjects.r('c(1,1)'),
        title = "Marginal Propensity to Consume by Home Equity",
        labels = ["Cash MPC","Housing Debt MPC","Housing Price MPC"],
        ylab = "MPC", xlab = "Home Equity  (< 0 is Underwater)",
        file_name = "mpc_cash_hsg_low_coh")
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([mpc_cash_low_coh_f,mpc_debt_low_coh_f,mpc_hsg_low_coh_f],-60,90, N=len(ltv_rows), loc=robjects.r('c(1,1)'),
        title = "Marginal Propensity to Consume by Home Equity",
        labels = ["Cash MPC","Housing Debt MPC","Housing Price MPC"],
        ylab = "MPC", xlab = "Home Equity  (< 0 is Underwater)",
        file_name = "mpc_cash_hsg_low_coh_backup")
ggplot_notebook(g, height=300,width=400)


#backup plot with alternative collateral constraint
hamp_params['collateral_constraint'] = 0
hp_mpc_0 = pd.DataFrame(np.zeros((len(ltv_rows), len(scenarios))), index=index, columns=columns) 
for eq in ltv_rows:
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  (eq/100.)*hamp_params['initial_price'] , **hamp_params)
    cf_pre = solve_unpack(hw_cf_params)
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt = (eq/100.)*hamp_params['initial_price'] - dhp, **hamp_params)
    cf_debt = solve_unpack(hw_cf_params)
    hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt = (eq/100.)*hamp_params['initial_price'],  d_house_price = dhp, **hamp_params)
    cf_hp = solve_unpack(hw_cf_params)
    hp_mpc_0.loc[eq,'hsg'] = (cf_hp.cFunc[t_eval](hamp_coh) - cf_pre.cFunc[t_eval](hamp_coh))/dhp
    hp_mpc_0.loc[eq,'debt'] = (cf_debt.cFunc[t_eval](hamp_coh) - cf_pre.cFunc[t_eval](hamp_coh))/dhp
    hp_mpc_0.loc[eq,'cash'] = (cf_pre.cFunc[t_eval](hamp_coh+dhp) - cf_pre.cFunc[t_eval](hamp_coh))/dhp
hamp_params['collateral_constraint'] = 0.2
hp_mpc_0.to_csv("~/dropbox/hampra/out2/tbl_hp_mpc_collat_0.csv")
hp_mpc_0

mpc_hsg_0_f = LinearInterp(equity_a,np.array(hp_mpc_0['hsg'])[::-1])
mpc_cash_0_f = LinearInterp(equity_a,np.array(hp_mpc_0['cash'])[::-1])
mpc_debt_0_f = LinearInterp(equity_a,np.array(hp_mpc_0['debt'])[::-1])

g = gg_funcs([mpc_hsg_0_f,mpc_cash_0_f,mpc_debt_0_f],-60,90, N=len(ltv_rows), loc=robjects.r('c(1,1)'),
        title = "Marginal Propensity to Consume by Home Equity \n Collateral Constraint = 0%",
        labels = ["Housing Price MPC","Cash MPC","Housing Debt MPC"],
        ylab = "MPC", xlab = "Home Equity  (< 0 is Underwater)",
        file_name = "mpc_cash_hsg_0_backup")
ggplot_notebook(g, height=300,width=400)

#important to do this for different models of housing costs wealth values




#
#df_ltv = df_ltv.loc[5:10,:]
#ltv_rows = list(df_ltv['LTV_Midpoint'])
##index = pd.Index(ltv_rows, name='rows')
#scenarios = [ 'mpc_L0_g16','mpc_L20_g16']
#columns = pd.Index(scenarios, name='cols')
#hp_mpc = pd.DataFrame(np.zeros((len(df_ltv), len(scenarios))), columns=columns) #index=index, 
#coh = 2
#dhp = 0.1
#hw_cf_params = deepcopy(baseline_params)
#  
##baseline case
#for spec in scenarios:
#    #set parameters [is there a more efficient way to do this?]
#    if spec is 'mpc_L0_g16':
#        hamp_params['collateral_constraint'] = 0
#        hamp_params['annual_hp_growth'] = 0.016
#    elif spec is 'mpc_L20_g16':
#        hamp_params['collateral_constraint'] = 0.20
#        hamp_params['annual_hp_growth'] = 0.016
#    #compute MPCs
#    for eq in ltv_rows:
#        hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  (eq/100.)*hamp_params['initial_price'] , **hamp_params)
#        cf_pre = solve_unpack(hw_cf_params)
#        hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt = (eq/100.)*hamp_params['initial_price'] , d_house_price = dhp, **hamp_params)
#        cf_post = solve_unpack(hw_cf_params)
#        hp_mpc.loc[eq,spec] = (cf_post.cFunc[t_eval](coh) - cf_pre.cFunc[t_eval](coh))/dhp
#
##even w/coh = 2 we are still getting high MPCs out of housing prices!
##xxx could look up typical coh for a 45 year old in order to be a big more disciplned
#
#hp_mpc_ltv = pd.merge(df_ltv, hp_mpc, left_on="LTV_Midpoint", right_index=True)
#
#output_rows = []
#for spec in scenarios:
#    for yr in ["_2005","_2010"]:
#        hp_mpc_ltv[spec+yr] = hp_mpc_ltv[spec]*hp_mpc_ltv["Share"+yr]/100
#        output_rows.append(spec+yr)
##study only people above 80% LTV
##hp_mpc_ltv.set_index('LTV_Midpoint')
##hp_mpc_ltv[hp_mpc_ltv['LTV_Midpoint'] > 80]
#hp_mpc_select_rows = df_ltv[df_ltv['LTV_Midpoint'] > 80]
##hp_mpc_select_rows['Share_2005_adj'] = hp_mpc_select_rows['Share_2005']/sum(hp_mpc_select_rows['Share_2005'])
##[output_rows].sum()
#hp_mpc_ltv.to_csv("~/dropbox/hampra/out2/tbl_mpc_ltv_src.csv")
#hp_mpc_ltv[output_rows].sum().to_csv("~/dropbox/hampra/out2/tbl_mpc_ltv_summary.csv")
#
#df = pd.DataFrame({'Loan-To-Value': 100*(1+np.array(equity[4:38]/rebate)), 
#                      'MPC': np.mean(lifecycle_hsg_example.MPChist,axis=1)[4:38]})
#g = gg.ggplot2(df) + \
#    gg.aes_string(x='Loan-To-Value', y='MPC') + mp.line + mp.point     +  \
#    mp.theme_bw(base_size=9) + mp.fte_theme +mp.colors + \
#    gg.labs(title="Home Equity and Marginal Propensity to Consume",
#                  y="MPC", x = "Loan-to-Value Ratio")
#mp.ggsave("ltv_and_mpc",g)
#ggplot_notebook(g, height=300,width=400)
#np.vstack((-1*np.array(lifecycle_hsg_params['BoroCnstArt'][4:38]),np.mean(lifecycle_hsg_example.MPChist,axis=1)[:40]))


#pseudo code:
#make a table (and then a figure) with lines for average MPCs        
#loc = robjects.r('c(1,1)')         
#title = "Consumption and Cash-on-Hand"
#ylab = "y"
#xlab="Loan-To-Value Ratio"
#ltitle = 'cols'
#labels = ['MPC: Collateral >= 0%','MPC: Collateral >= 20%']
#tmp = hp_mpc_ltv
#tmp = tmp.rename(columns={scenarios[0]: labels[0], scenarios[1]: labels[1]})
#fig = pd.melt(tmp, id_vars =['LTV_Midpoint'], value_vars = labels)
#fig = fig.rename(columns={"LTV_Midpoint": "x"})

#g = gg.ggplot(fig) + \
#    mp.base_plot + mp.line + mp.point +  \
#    mp.theme_bw(base_size=9) + mp.fte_theme +mp.colors +  \
#    gg.labs(title=title,y=ylab,x=xlab) + mp.legend_f(loc) + mp.legend_t_c(ltitle) + mp.legend_t_s(ltitle)
#ggplot_notebook(g, height=300,width=400)
#make x values into a factor variable so that they are equally spaced
#

#rrr graphs are generating a division by zero error. need to step through this to figure out the issues


##############################################
#try out alternative consumption functions
##############################################
IndShockExample = solve_unpack(baseline_params)

tmp_params = deepcopy(baseline_params)
tmp_params['PermShkStd'] = Params.PermShkStdPos
tmp_params["PermShkCount"] = 7
AddBackPermIncRisk = solve_unpack(tmp_params)

tmp_params = deepcopy(baseline_params)
tmp_params['TranShkStd'] =  [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, # <-- no transitory income shocs after retirement
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DoubleTempIncRisk = solve_unpack(tmp_params)

tmp_params = deepcopy(baseline_params)
tmp_params["UnempPrb"] = 0.2
QuadrupleURisk = solve_unpack(tmp_params)

tmp_params = deepcopy(baseline_params)
tmp_params['HsgPay'] = uw_house_params['HsgPay']
AddHsgPay = solve_unpack(tmp_params)

#double temp inc risk not working
g = gg_funcs([IndShockExample.cFunc[t_eval],AddBackPermIncRisk.cFunc[t_eval],AddHsgPay.cFunc[t_eval]], #DoubleTempIncRisk.cFunc[t_eval],QuadrupleURisk.cFunc[t_eval],
        -.001,8, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Functions",
        labels = ["Baseline (No Housing, No Perm Inc Risk)","Add Perm Inc Risk", "Add Housing Payments"], #"Double Temp Inc SD", "Quadruple U risk", 
        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "cf_concavity")
ggplot_notebook(g, height=300,width=400)

###########################################################################
# Calculate consumption impact of PRA
###########################################################################

index = pd.Index(list(["High Cash-on-Hand"]), name='rows')
columns = pd.Index(['c_pre', 'c_post'], name='cols')
pra_mpc = pd.DataFrame(np.zeros((1,2)), index=index, columns=columns)

#baseline case & heloc case
hw_cf_params = deepcopy(baseline_params)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Baseline','c_pre'] = cf.cFunc[t_eval](hamp_coh)
pra_mpc.loc['Payments Reduced Immediately','c_pre'] = cf.cFunc[t_eval](hamp_coh)
pra_mpc.loc['Low Cash-on-Hand','c_pre'] = cf.cFunc[t_eval](tmp_vlo)
pra_mpc.loc['High Cash-on-Hand','c_pre'] = cf.cFunc[t_eval](tmp_hi)
pra_mpc.loc["Cash-on-Hand = " + str(tmp_vhi),'c_pre'] = cf.cFunc[t_eval](tmp_vhi)
hw_cf_params['BoroCnstArt']   = map(lambda x:x-heloc_L, hw_cf_params['BoroCnstArt'])
cf_heloc = solve_unpack(hw_cf_params)
pra_mpc.loc['Has HELOC','c_pre'] = cf_heloc.cFunc[t_eval](hamp_coh)

hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] -hamp_params['pra_forgive'] , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Payments Reduced Immediately','c_post'] = cf.cFunc[t_eval](hamp_coh)
hw_cf_params['HsgPay'] =  pra_pmt(age = 45, forgive = hamp_params['pra_forgive'] , **hamp_params)   
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Baseline','c_post'] = cf.cFunc[t_eval](hamp_coh)
pra_mpc.loc['Low Cash-on-Hand','c_post'] = cf.cFunc[t_eval](tmp_vlo)
pra_mpc.loc['High Cash-on-Hand','c_post'] = cf.cFunc[t_eval](tmp_hi)
pra_mpc.loc["Cash-on-Hand = " + str(tmp_vhi),'c_post'] = cf.cFunc[t_eval](tmp_vhi)
hw_cf_params['BoroCnstArt']   = map(lambda x:x-heloc_L, hw_cf_params['BoroCnstArt'])
cf_heloc = solve_unpack(hw_cf_params)
pra_mpc.loc['Has HELOC','c_post'] = cf_heloc.cFunc[t_eval](hamp_coh)

#lower interest rate
hw_cf_params = deepcopy(baseline_params)
hamp_params['int_rate'] = 0.001
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['High Cash-on-Hand & r = 0','c_pre'] = cf.cFunc[t_eval](tmp_hi)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] -hamp_params['pra_forgive'] , **hamp_params)
hw_cf_params['HsgPay'] =  pra_pmt(age = 45, forgive = hamp_params['pra_forgive'] , **hamp_params)   
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['High Cash-on-Hand & r = 0','c_post'] = cf.cFunc[t_eval](tmp_hi)
hamp_params['int_rate'] = 0.02

#change baseline debt to 135% LTV
tmp = deepcopy(hamp_params['baseline_debt'])
hamp_params['baseline_debt'] = 1.35*hamp_params['initial_price']
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt']  , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Write down to 90% LTV','c_pre'] = cf.cFunc[t_eval](hamp_coh)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt = hamp_params['baseline_debt']  -hamp_params['pra_forgive'] , **hamp_params)
hw_cf_params['HsgPay'] =  pra_pmt(age = 45, forgive = hamp_params['pra_forgive'] , **hamp_params)   
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Write down to 90% LTV','c_post'] = cf.cFunc[t_eval](hamp_coh)
hamp_params['baseline_debt'] = tmp


#here MPCs are the same for house price change and debt change
#change baseline debt to 95% LTV
grant_size = 0.1
hamp_params['collateral_constraint'] = 0
hw_cf_params = deepcopy(baseline_params)
col_frgv = 'MPC Forgive Debt From 95% LTV'
col_hp = 'MPC House Price From 95% LTV'
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  0.95*hamp_params['initial_price']  , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc[col_frgv,'c_pre'] = cf.cFunc[t_eval](hamp_coh)
pra_mpc.loc[col_hp,'c_pre'] = cf.cFunc[t_eval](hamp_coh)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt = 0.95*hamp_params['initial_price']  - grant_size , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc[col_frgv,'c_post'] = cf.cFunc[t_eval](hamp_coh)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay']  = hsg_wealth(initial_debt = 0.95*hamp_params['initial_price'], d_house_price = grant_size, **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc[col_hp,'c_post'] = cf.cFunc[t_eval](hamp_coh)
hamp_params['collateral_constraint'] = 0.20


#collateral constraint
hamp_params['collateral_constraint'] = 0
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Collateral Constraint = 0','c_pre'] = cf.cFunc[t_eval](hamp_coh)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] -hamp_params['pra_forgive'] , **hamp_params)
hw_cf_params['HsgPay'] =  pra_pmt(age = 45, forgive = hamp_params['pra_forgive'] , **hamp_params)   
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Collateral Constraint = 0','c_post'] = cf.cFunc[t_eval](hamp_coh)
tmp = deepcopy(hamp_params['baseline_debt'])
hamp_params['baseline_debt'] = 1.35*hamp_params['initial_price']
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt = hamp_params['baseline_debt']   , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['90% LTV & Constraint = 0','c_pre'] = cf.cFunc[t_eval](hamp_coh)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt = hamp_params['baseline_debt']   -hamp_params['pra_forgive'] , **hamp_params)
hw_cf_params['HsgPay'] =  pra_pmt(age = 45, forgive = hamp_params['pra_forgive'] , **hamp_params)   
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['90% LTV & Constraint = 0','c_post'] = cf.cFunc[t_eval](hamp_coh)
hamp_params['collateral_constraint'] = 0.20
#hamp_params['baseline_debt'] = tmp

#age 35 and 55
t_eval_35 = 10
age_young = 35
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'], age_at_mod = age_young , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Age At Mod = 35','c_pre'] = cf.cFunc[t_eval_35](hamp_coh)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] -hamp_params['pra_forgive'], age_at_mod = age_young , **hamp_params)
hw_cf_params['HsgPay'] =  pra_pmt(age = 35, forgive = hamp_params['pra_forgive'] , **hamp_params)   
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Age At Mod = 35','c_post'] = cf.cFunc[t_eval_35](hamp_coh)
t_eval_55 = 30
age_old = 55
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'], age_at_mod = age_old , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Age At Mod = 55','c_pre'] = cf.cFunc[t_eval_55](hamp_coh)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] -hamp_params['pra_forgive'], age_at_mod = age_old , **hamp_params)
hw_cf_params['HsgPay'] =  pra_pmt(age = age_old, forgive = hamp_params['pra_forgive'] , **hamp_params)   
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Age At Mod = 55','c_post'] = cf.cFunc[t_eval_55](hamp_coh)

#why is the most optimistic combo less than the original PIH guy?
#(1) he has more collateral -- this is pushing up MPC slighly
#(2) we are looking at an older guy
#most optimistic: 0 collateral, age 55, high cash-on-hand
age_eval = 45
t_eval_55 = t_eval # was 30 
hamp_params['collateral_constraint'] = 0 # was 0
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'], age_at_mod = age_eval , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Most Optimistic Combo','c_pre'] = cf.cFunc[t_eval_55](tmp_hi)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], d = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] -hamp_params['pra_forgive'], age_at_mod = age_eval , **hamp_params)
hw_cf_params['HsgPay'] =  pra_pmt(age = age_eval, forgive = hamp_params['pra_forgive'] , **hamp_params)   
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Most Optimistic Combo','c_post'] = cf.cFunc[t_eval_55](tmp_hi)
hamp_params['collateral_constraint'] = 0.20



#rapid house price growth
tmp = deepcopy(hamp_params['annual_hp_growth'])
hamp_params['annual_hp_growth'] = 0.05
hw_cf_params = deepcopy(baseline_params)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] , **hamp_params)
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Fast House Price Growth (5%)','c_pre'] = cf.cFunc[t_eval](hamp_coh)
hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] -hamp_params['pra_forgive'] , **hamp_params)
hw_cf_params['HsgPay'] =  pra_pmt(age = 45, forgive = hamp_params['pra_forgive'] , **hamp_params)   
cf = solve_unpack(hw_cf_params)
pra_mpc.loc['Fast House Price Growth (5%)','c_post'] = cf.cFunc[t_eval](hamp_coh)
hamp_params['annual_hp_growth'] = tmp


#remark: I don't know how we interpret the high PIH agent. just say that we can reject this behavior?
#why is consumption going down for the low coh agent and the 90% LTV agent?
pra_mpc['mpc'] = (pra_mpc['c_post'] - pra_mpc['c_pre'])/hamp_params['pra_forgive']
pra_mpc.loc[col_frgv,'mpc'] = (pra_mpc.loc[col_frgv,'c_post'] - pra_mpc.loc[col_frgv,'c_pre']) / grant_size
pra_mpc.loc[col_hp,'mpc'] = (pra_mpc.loc[col_frgv,'c_post'] - pra_mpc.loc[col_frgv,'c_pre']) / grant_size
pra_mpc = pra_mpc.round(3)
pra_mpc.to_csv("~/dropbox/hampra/out2/tbl_pra_mpc.csv")
pra_mpc




#
#
############################################################################
## Calculate consumption impact of House Price Increases
############################################################################
#
#df_ltv = df_ltv.loc[5:10,:]
#ltv_rows = list(df_ltv['LTV_Midpoint'])
##index = pd.Index(ltv_rows, name='rows')
#scenarios = [ 'mpc_L0_g16','mpc_L20_g16']
#columns = pd.Index(scenarios, name='cols')
#hp_mpc = pd.DataFrame(np.zeros((len(df_ltv), len(scenarios))), columns=columns) #index=index, 
#coh = 2
#dhp = 0.1
#hw_cf_params = deepcopy(baseline_params)
#  
##baseline case
#for spec in scenarios:
#    #set parameters [is there a more efficient way to do this?]
#    if spec is 'mpc_L0_g16':
#        hamp_params['collateral_constraint'] = 0
#        hamp_params['annual_hp_growth'] = 0.016
#    elif spec is 'mpc_L20_g16':
#        hamp_params['collateral_constraint'] = 0.20
#        hamp_params['annual_hp_growth'] = 0.016
#    #compute MPCs
#    for eq in ltv_rows:
#        hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt =  (eq/100.)*hamp_params['initial_price'] , **hamp_params)
#        cf_pre = solve_unpack(hw_cf_params)
#        hw_cf_params['rebate_amt'], e, hw_cf_params['BoroCnstArt'], hw_cf_params['HsgPay'] = hsg_wealth(initial_debt = (eq/100.)*hamp_params['initial_price'] , d_house_price = dhp, **hamp_params)
#        cf_post = solve_unpack(hw_cf_params)
#        hp_mpc.loc[eq,spec] = (cf_post.cFunc[t_eval](coh) - cf_pre.cFunc[t_eval](coh))/dhp
#
##even w/coh = 2 we are still getting high MPCs out of housing prices!
##xxx could look up typical coh for a 45 year old in order to be a big more disciplned
#
#hp_mpc_ltv = pd.merge(df_ltv, hp_mpc, left_on="LTV_Midpoint", right_index=True)
#
#output_rows = []
#for spec in scenarios:
#    for yr in ["_2005","_2010"]:
#        hp_mpc_ltv[spec+yr] = hp_mpc_ltv[spec]*hp_mpc_ltv["Share"+yr]/100
#        output_rows.append(spec+yr)
##study only people above 80% LTV
##hp_mpc_ltv.set_index('LTV_Midpoint')
##hp_mpc_ltv[hp_mpc_ltv['LTV_Midpoint'] > 80]
#hp_mpc_select_rows = df_ltv[df_ltv['LTV_Midpoint'] > 80]
##hp_mpc_select_rows['Share_2005_adj'] = hp_mpc_select_rows['Share_2005']/sum(hp_mpc_select_rows['Share_2005'])
##[output_rows].sum()
#hp_mpc_ltv.to_csv("~/dropbox/hampra/out2/tbl_mpc_ltv_src.csv")
#hp_mpc_ltv[output_rows].sum().to_csv("~/dropbox/hampra/out2/tbl_mpc_ltv_summary.csv")
#
#df = pd.DataFrame({'Loan-To-Value': 100*(1+np.array(equity[4:38]/rebate)), 
#                      'MPC': np.mean(lifecycle_hsg_example.MPChist,axis=1)[4:38]})
#g = gg.ggplot2(df) + \
#    gg.aes_string(x='Loan-To-Value', y='MPC') + mp.line + mp.point     +  \
#    mp.theme_bw(base_size=9) + mp.fte_theme +mp.colors + \
#    gg.labs(title="Home Equity and Marginal Propensity to Consume",
#                  y="MPC", x = "Loan-to-Value Ratio")
#mp.ggsave("ltv_and_mpc",g)
#ggplot_notebook(g, height=300,width=400)
#np.vstack((-1*np.array(lifecycle_hsg_params['BoroCnstArt'][4:38]),np.mean(lifecycle_hsg_example.MPChist,axis=1)[:40]))


#pseudo code:
#make a table (and then a figure) with lines for average MPCs        
#loc = robjects.r('c(1,1)')         
#title = "Consumption and Cash-on-Hand"
#ylab = "y"
#xlab="Loan-To-Value Ratio"
#ltitle = 'cols'
#labels = ['MPC: Collateral >= 0%','MPC: Collateral >= 20%']
#tmp = hp_mpc_ltv
#tmp = tmp.rename(columns={scenarios[0]: labels[0], scenarios[1]: labels[1]})
#fig = pd.melt(tmp, id_vars =['LTV_Midpoint'], value_vars = labels)
#fig = fig.rename(columns={"LTV_Midpoint": "x"})

#g = gg.ggplot(fig) + \
#    mp.base_plot + mp.line + mp.point +  \
#    mp.theme_bw(base_size=9) + mp.fte_theme +mp.colors +  \
#    gg.labs(title=title,y=ylab,x=xlab) + mp.legend_f(loc) + mp.legend_t_c(ltitle) + mp.legend_t_s(ltitle)
#ggplot_notebook(g, height=300,width=400)
#make x values into a factor variable so that they are equally spaced
#
##make a table (and then a figure) with overlapping histogram of population shares
#tmp = df_ltv
#labels = ['Share of Mortgages 2005', 'Share of Mortgages 2010']
#tmp = tmp.rename(columns={df_ltv.columns[2]: labels[0], df_ltv.columns[3]: labels[1]})
#fig2 = pd.melt(tmp, id_vars =['LTV_Midpoint'], value_vars = labels)
#fig2['value'] = fig2['value']/100
#fig2 = fig2.rename(columns={"LTV_Midpoint": "x"})
#fig.append(fig2).to_csv("~/dropbox/hampra/out2/tbl_ltv_ggplot.csv")

#h = gg.ggplot(fig2) + \
#    gg.aes_string(x='x', y='value',fill = 'variable') + \
#    gg.geom_bar(stat = "identity", position = "dodge") + \
#    mp.theme_bw(base_size=9) + mp.fte_theme +mp.colors +  \
#    gg.labs(title=title,y=ylab,x=xlab) + mp.legend_f(loc) + mp.legend_t_c(ltitle) + mp.legend_t_s(ltitle)
##xx ggplot_notebook  should be called by default and should have height & width set by default
#ggplot_notebook(h, height=300,width=400)
# 
#i = gg.ggplot(fig) + \
#    mp.base_plot + mp.line + mp.point +  \
#    gg.ggplot(fig2) + \
#    gg.aes_string(x='x', y='value',fill = 'variable') + \
#    gg.geom_bar(stat = "identity", position = "dodge") 
#ggplot_notebook(i, height=300,width=400) 
#
##sample code
#import pandas.rpy.common as com
#df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C':[7,8,9]},
#                   index=["one", "two", "three"]) 
#r_dataframe = com.convert_to_r_dataframe(df)
#print(type(r_dataframe))
#print(r_dataframe)
#
#r_fig = com.convert_to_r_dataframe(fig)
##big fat R code block
#g = robjects.r(' \
#    ggplot2(fig, aes(x,value) \
#    ')


#remark to Pascal: can't have two separate y-axes https://gist.github.com/tomhopper/faa24797bb44addeba79
#calculate population average MPC and write as text on plot



##################################
#Simulate data and study MPC over the lifecycle
##################################

## Simulate some data for num_agents defaults to 10000; results stored in cHist, mHist, bHist, aHist, MPChist, and pHist
#IndShockExample.sim_periods = IndShockExample.T_total + 1
#IndShockExample.makeIncShkHist()
#IndShockExample.initializeSim()
#IndShockExample.simConsHistory()
#
##how do these change with age
#np.mean(IndShockExample.cHist,axis=1) #rising w age and then falling right before death
#np.mean(IndShockExample.mHist,axis=1) #rising then falling
#np.mean(IndShockExample.bHist,axis=1) #rising then falling bank balances before labor income
#np.mean(IndShockExample.aHist,axis=1) #rising then falling
#np.mean(IndShockExample.MPChist,axis=1) #falling very rapidly in first five years.
#np.mean(IndShockExample.pHist,axis=1) #rising permanent income level
#
#np.mean(IndShockExample.MPChist[:40]) 
#np.mean(IndShockExample.MPChist[:20]) 
#
##redo lifecycle model with housing
#lifecycle_hsg_params = deepcopy(baseline_params)
##xx at age 65 you are selling your house for 5. this should be reflected in your asset balances starting at age 65
#hamp_params['collateral_constraint'] = 0
#rebate, equity, limit, d = hsg_wealth(initial_debt = 3.25, age_at_mod = 30, **hamp_params)
#hamp_params['collateral_constraint'] = 0.2
#lifecycle_hsg_params['rebate_amt'], e, lifecycle_hsg_params['BoroCnstArt'], d = hsg_wealth(initial_debt = 3.25, age_at_mod = 30, **hamp_params)
#lifecycle_hsg_example = solve_unpack(lifecycle_hsg_params)
#lifecycle_hsg_example.sim_periods = lifecycle_hsg_example.T_total + 1
#lifecycle_hsg_example.makeIncShkHist()
#lifecycle_hsg_example.initializeSim()
#lifecycle_hsg_example.simConsHistory()
#
#np.mean(lifecycle_hsg_example.cHist,axis=1) #rising w age and then falling right before death
#np.mean(lifecycle_hsg_example.mHist,axis=1) #rising then falling
#np.mean(lifecycle_hsg_example.bHist,axis=1) #rising then falling bank balances before labor income
#np.mean(lifecycle_hsg_example.aHist,axis=1) #rising then falling
#np.mean(lifecycle_hsg_example.MPChist,axis=1) #falling very rapidly in first five years.
#np.mean(lifecycle_hsg_example.pHist,axis=1) #rising permanent income level
#
#np.mean(lifecycle_hsg_example.MPChist[:40]) 
#np.mean(lifecycle_hsg_example.MPChist[:20]) 
#


###########################################################################
# Study moving around collateral constraint

#l = baseline_params['BoroCnstArt']
#for i in range(len(l)):
#    l[i] = 0
        

##check consumption function w diff borrowing constraints
#t_eval = 35
#cf_list = [IndShockExample.cFunc[t_eval],uw_house_example.cFunc[t_eval],pra_example.cFunc[t_eval]]
#g = gg_funcs(cf_list,-2,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption at Age " + str(25+t_eval) + " With Relaxed Collateral Constraint",
#        ylab = "Consumption Function", xlab = "Cash-on-Hand (Ratio to Permanent Income)")
#mp.ggsave("future_collateral",g)
#ggplot_notebook(g, height=300,width=400)
#
##check consumption function w rebates borrowing constraints
#t_eval = 30
#cf_list = [IndShockExample.cFunc[t_eval],FutRebateExample.cFunc[t_eval]]
#g = gg_funcs(cf_list,0,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption at Age " + str(25+t_eval) + " With Age 65 Rebate",
#        ylab = "Consumption Function", xlab = "Cash-on-Hand (Ratio to Permanent Income)")
#mp.ggsave("future_rebate_v2",g)
#ggplot_notebook(g, height=300,width=400)
#
#cf_list = []
#for t_eval in range(0,40,5):
#    print t_eval
#    dc = lambda x, t_eval = t_eval: RelaxCollat.cFunc[t_eval](x) - IndShockExample.cFunc[t_eval](x)
#    cf_list.append(dc)
#g = gg_funcs(cf_list,0.01,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption With Predictable Collateral at Age " + str(age_of_rebate),
#        ylab = "Consumption Change From Predictable Collateral Movement", xlab = "Cash-on-Hand (Ratio to Permanent Income)")
#mp.ggsave("future_collateral_cons_ages",g)
#ggplot_notebook(g, height=300,width=400)
#
#t_eval = 20
#cf_list = [IndShockExample.cFunc[t_eval],RelaxCollat.cFunc[t_eval]]
#g = gg_funcs(cf_list,-1,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption at Age " + str(25+t_eval) + " With Predictable Collateral at Age " + str(age_of_rebate),
#        ylab = "Consumption Change From Predictable Collateral Grant", xlab = "Cash-on-Hand (Ratio to Permanent Income)")
#mp.ggsave("future_collateral_cons_age" + str(25+t_eval),g)
#ggplot_notebook(g, height=300,width=400)

###########################################################################
# Begin Plots


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
#        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)")
#mp.ggsave("nbc_age_45",g)
#ggplot_notebook(g, height=300,width=400)
#
#
#g = gg_funcs([IndShockExample.cFunc[10],NbcExample.cFunc[10]],-0.5,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption Functions age 35", labels = ['Exo Constraint','Nat Borrowing Constraint'],
#        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)")
#mp.ggsave("nbc_age_35",g)
#ggplot_notebook(g, height=300,width=400)
#
#
#
#g = gg_funcs([IndShockExample.cFunc[40],NbcExample.cFunc[40]],-10,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption Functions age 65", labels = ['Exo Constraint','Nat Borrowing Constraint'],
#        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)")
#mp.ggsave("nbc_age_65",g)
#ggplot_notebook(g, height=300,width=400)


############################################################################
#
##Study moving rebate date
############################################################################
#cf_list = []
#for i in range(10):
#    cf_list.append(FutRebateExample.cFunc[40-i+5])
#g = gg_funcs(cf_list,0.01,2.5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Consumption Function With Predictable Rebate of One Year's Income at Age " + str(age_of_rebate),
#        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)")
#ggplot_notebook(g, height=300,width=400)
#
##slide 3 with moving around date of rebate
##if you give rebate during retirement then no change (except that your wealth is higher)
##because there is no income uncertainty left
#
##figure out much younger and much older
##bumping age of rebate by ONE year bumps age of cons func affected by TWO years. why?
##there is s ahre cons bump at 
#
##baseline cons function is much higher (and has bigger h2m region) after retirement
##cons function with rebate at 45 is much higher after age 50
#
##when we had a rebate, then we find that cons funcs change sharply at age 50
## at age 50 inc growth rate  from 2.5% per year to 1% per year
##i don't understand conceptually why this matters
##also, tried this again later and couldn't reproduce
#age_at_rebate = 50
#age_eval_min = 44
#years_eval = 9
#dyear = 3
#settings.t_rebate = baseline_params['T_total'] + 25 - age_at_rebate 
#settings.rebate_size = 1
#rebate_alt_age = Model.IndShockConsumerType(**baseline_params)
#rebate_alt_age.solve()
#rebate_alt_age.unpack_cFunc()
#rebate_alt_age.timeFwd()
#cf_list = [IndShockExample.cFunc[age_eval_min-25]]
#for i in range(years_eval):
#    cf_list.append(rebate_alt_age.cFunc[age_eval_min-25+i*dyear])
#g = gg_funcs(cf_list,0.01,2.5, N=50, loc=robjects.r('c(1,0)'),
#         ltitle = '', labels = ['no rebate'],
#        title = "Rebate at Age " + str(age_at_rebate) + " cons func age " + str(age_eval_min) + " to " + str(age_eval_min + dyear*years_eval),
#        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)")
#ggplot_notebook(g, height=300,width=400)
#
