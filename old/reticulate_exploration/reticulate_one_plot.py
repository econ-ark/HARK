#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:35:40 2018

@author: peterganong
"""

import os, sys
#os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
#if getpass.getuser() == 'peterganong':
os.chdir("/Users/peterganong/repo/HARK/gn") 
out_path = "~/dropbox/hampra/out_arch/out_test3/"
#elif getpass.getuser() == 'pascalnoel':
#    os.chdir("/Users/Pascal/repo/HARK/gn") 
sys.path.insert(0,".")
sys.path.insert(0,'../')
sys.path.insert(0,'../ConsumptionSaving')
sys.path.insert(0,'../SolvingMicroDSOPs')
import settings
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
import pdb
import pandas as pd
#this line errors out sometimes. Driven by issues with the Canopy_64bit path
#from rpy2 import robjects
#import rpy2.robjects.lib.ggplot2 as gg
#from rpy2.robjects import pandas2ri
#import make_plots as mp
import pickle
#read in HAMP parameters from google docs
#import gspread
#from oauth2client.service_account import ServiceAccountCredentials
#scope = ['https://spreadsheets.google.com/feeds']
#credentials = ServiceAccountCredentials.from_json_keyfile_name('gspread-oauth-2017.json', scope)
#gc = gspread.authorize(credentials)
#g_params = gc.open("HAMPRA Model Parameters").sheet1 
#df = pd.DataFrame(g_params.get_all_records())
#pickle.dump( df, open("params_google_df.p", "wb" ) )
df = pd.read_pickle( open( "params_google_df.p", "rb" ) )

hamp_params = df[['Param','Value']].set_index('Param')['Value'][:9].to_dict()
inc_params = df[['Param','Value']].set_index('Param')['Value'][9:13].to_dict()
hamp_coh = float(inc_params['cash_on_hand'])
boom_params = df[['Param','Value']].set_index('Param')['Value'][14:20].to_dict()
heloc_L = float(df[['Param','Value']].set_index('Param')['Value'][21:22])
rd_params = df[['Param','Value']].set_index('Param')['Value'][23:26].to_dict()

#g_params = gc.open("HAMPRA Loan-to-Value Distribution")
#ltv_wksheet = g_params.worksheet("PythonInput")
#df_ltv = pd.DataFrame(ltv_wksheet.get_all_records())
#pickle.dump( df_ltv, open("params_google_df_ltv.p", "wb" ) )
#df_ltv = pickle.load( open( "params_google_df_ltv.p", "rb" ) )
    
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
settings.lil_verbose = True
settings.min_age, settings.max_age = 60, 65
###########################################################################
#plotting functions 
###########################################################################
#xx I tried to move to make_plots.py but failed. Not sure why.
#import uuid     #enable plotting inside of iPython notebook (default rpy2 pushes to a semi-broken R plot-viewer)
#from rpy2.robjects.packages import importr
#from IPython.core.display import Image
#grdevices = importr('grDevices')
#def ggplot_notebook(gg, width = 800, height = 600):
#    fn = 'tmp/{uuid}.png'.format(uuid = uuid.uuid4())
#    grdevices.png(fn, width = width, height = height)
#    gg.plot()
#    grdevices.dev_off()
#    return Image(filename=fn)
#    
#    
#loc = robjects.r('c(1,0)')
#def gg_funcs(functions,bottom,top,N=1000,labels = [],
#             title = "Consumption and Cash-on-Hand", ylab = "y", xlab="x", 
#             loc = loc, ltitle = 'Variable',
#             file_name = None):
#    if type(functions)==list:
#        function_list = functions
#    else:
#        function_list = [functions]       
#    step = (top-bottom)/N
#    x = np.arange(bottom,top,step)
#    fig = pd.DataFrame({'x': x})
#    i = 0
#    for function in function_list:
#        if i > len(labels)-1:
#            labels.append("func" + str(i))
#        fig[labels[i]] = function(x)
#        i=i+1
#    fig = pd.melt(fig, id_vars=['x'])  
#    g = gg.ggplot(fig) + \
#        mp.base_plot + mp.line + mp.point +  \
#        mp.theme_bw(base_size=9) + mp.fte_theme + \
#        gg.labs(title=title,y=ylab,x=xlab) + mp.legend_f(loc) + mp.legend_t_c(ltitle) + mp.colors #+ mp.legend_t_s(ltitle) 
#    if file_name is not None:
#        mp.ggsave(file_name,g)
#    return(g)
#    
    

###########################################################################
#housing wealth functions
###########################################################################
#xx move to outside script and import in the future. got errors because I was modifying the wrong do file before...
#xxx the version in the default script is now more up to date
#remark: right now you are actually selling the house one year before retirement rather than at retirement. not sure if this is a problem.
def hsg_wealth(initial_debt, annual_hp_growth, collateral_constraint, baseline_debt, 
               initial_price, int_rate, pra_forgive, hsg_rent_p, hsg_own_p, maint,
               d_house_price = 0, age_at_mod = 45, hsg_pmt_wk_own = True, hsg_pmt_ret_y = False, default = False,
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
    maint_pmt = [initial_price*maint]
    for i in range(1,T):
        #print "age: " + str(i + age_at_mod) + " has growth fac: " + str(Params.PermGroFac[(i-1) + age_at_mod - 25])
        perm_gro = Params.PermGroFac[i + age_at_mod - 26]
        price.append(price[-1]*(1+annual_hp_growth)/perm_gro)
        debt.append((debt[-1]*(1+int_rate))/perm_gro - hsg_pmt[-1]/perm_gro) #xx double-check timing assumptions here
        maint_pmt.append(maint_pmt[-1]/perm_gro)
        hsg_pmt.append(hsg_pmt[-1]/perm_gro)
    equity = np.array(price) - np.array(debt)
    limit = np.min(np.vstack((-(np.array(price)*(1-collateral_constraint) - np.array(debt)),np.zeros(T))),axis=0).tolist()
    if hsg_pmt_wk_own and not default:
        user_cost = [x * hamp_params['hsg_own_p'] for x in price]
        hsg_pmt = map(add, hsg_pmt, user_cost)
    elif default:
        hsg_pmt = [x * hamp_params['hsg_rent_p'] for x in price]
    if max(hsg_pmt) >= baseline_params['IncUnemp'] + 0.3:
        print("Error: cannot have housing payment > UI Benefit")
        print hsg_pmt
        return
    hsg_pmt = map(add, hsg_pmt, maint_pmt)
    if default:
        hsg_pmt_ret_y = False
    if hsg_pmt_ret_y: #housing payments in retirement
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
    #pdb.set_trace()
    hsg_pmt_ret = map(add, hsg_pmt_ret, [maint_pmt[-1]/Params.PermGroFac[39]] * 26)
    #fill out arguments to return
    hsg_pmt = [0.0] * (age_at_mod - 26) + hsg_pmt + hsg_pmt_ret
    equity = [0.0] * (age_at_mod - 26) + equity.tolist() + [0.0] * 26
    limit = [0.0] * (age_at_mod - 26) + limit + [0.0] * 26
    sale_proceeds = max(equity[38],0)
    return sale_proceeds, equity, limit, hsg_pmt


def pra_pmt(annual_hp_growth, collateral_constraint, baseline_debt, initial_price, 
            int_rate, pra_forgive, hsg_rent_p, hsg_own_p, maint, hsg_pmt_wk_own = True, hsg_pmt_ret_y = False,
            age = 45, forgive = hamp_params['pra_forgive']):
    r, e, L, d_hamp = hsg_wealth(initial_debt =  hamp_params['baseline_debt'], age_at_mod = age, hsg_pmt_wk_own = hsg_pmt_wk_own, hsg_pmt_ret_y = hsg_pmt_ret_y, **hamp_params)
    r, e, L, d_prin_red = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - forgive, age_at_mod = age, hsg_pmt_wk_own = hsg_pmt_wk_own, hsg_pmt_ret_y = hsg_pmt_ret_y, **hamp_params)
    pra_pmt = d_prin_red[:age-26] + d_hamp[age-26:age-21] + d_prin_red[age-21:]
    return pra_pmt

#reload(Params)  
baseline_params = Params.init_consumer_objects
#pra_pmt(age = 45, forgive = 1, **hamp_params)[20:40]
#hsg_wealth(initial_debt =  hamp_params['baseline_debt'], **hamp_params)
#hamp_params['maint'] = 0.
#hsg_wealth(initial_debt =  hamp_params['baseline_debt'], **hamp_params)
#hamp_params['maint'] = 0.025


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
boro_cnst_pre_pra = LinearInterp(np.arange(26,91),list(map(neg, uw_house_params['BoroCnstArt'])))
boro_cnst_post_pra = LinearInterp(np.arange(26,91),list(map(neg, pra_params['BoroCnstArt'])))
#g = gg_funcs([boro_cnst_pre_pra,boro_cnst_post_pra],
#              45.001,75, N=round(75-45.001), loc=robjects.r('c(0,0.5)'),
#        title = "Borrowing Limits \n Receive Treatment at Age 45",
#        labels = labels,
#        ylab = "Borrowing Limit (Years of Income)", xlab = "Age", file_name = "borrowing_limits_and_pra_diag")
#ggplot_notebook(g, height=300,width=400)
