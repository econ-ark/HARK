# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:55:59 2016

@author: ganong
"""
import os
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
os.chdir("/Users/ganong/repo/HARK-comments-and-cleanup/gn")
out_path = "~/dropbox/hampra/out2/"
import settings
import sys 
sys.path.insert(0,'../')
sys.path.insert(0,'../ConsumptionSaving')
sys.path.insert(0,'../SolvingMicroDSOPs')
from copy import deepcopy
import numpy as np
#from HARKcore_gn import AgentType, Solution, NullFunc
from HARKinterpolation import LinearInterp  #, LowerEnvelope, CubicInterp
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


#read in HAMP parameters from google docs
import gspread
from oauth2client.service_account import ServiceAccountCredentials
scope = ['https://spreadsheets.google.com/feeds']
credentials = ServiceAccountCredentials.from_json_keyfile_name('gspread-oauth.json', scope)
gc = gspread.authorize(credentials)
g_params = gc.open("HAMPRA Model Parameters").sheet1 #in this case
df = pd.DataFrame(g_params.get_all_records())
sh = gc.open("HAMPRA Model Parameters")
py_out = sh.worksheet("py_out") #sh.add_worksheet(title = "py_out", rows = "10", cols = "5")
#import pickle
#df = pickle.load( open( "params_google_df.p", "rb" ) )

hamp_params = df[['Param','Value']].set_index('Param')['Value'][:8].to_dict()
inc_params = df[['Param','Value']].set_index('Param')['Value'][8:12].to_dict()
hamp_coh = float(inc_params['cash_on_hand'])
boom_params = df[['Param','Value']].set_index('Param')['Value'][13:19].to_dict()
heloc_L = float(df[['Param','Value']].set_index('Param')['Value'][20:21])

#imports specific to default code
from scipy.optimize import fsolve
def findIntersection(fun1,fun2,x0):
    return fsolve(lambda x : fun1(x) - fun2(x),x0)
from functools import partial

###########################################################################
#set economic parameters 
###########################################################################
rebate_years_until_death = 25
age_of_rebate = 90 - rebate_years_until_death
t_eval = age_of_rebate - 25 - 20
def mpc(cF, rebate = 1, a = 0.1):
    return round((cF(a+rebate) - cF(a)) / rebate,3)
tmp_vlo = Params.IncUnemp


settings.init()
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
    
    
def gg_funcs(functions,bottom,top,N=1000,labels = [],
             title = "Consumption and Cash-on-Hand", ylab = "y", xlab="x", 
             loc = robjects.r('c(1,0)'), ltitle = 'Variable',
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
#remark: right now you are actually selling the house one year before retirement rather than at retirement. not sure if this is a problem.
def hsg_wealth(initial_debt, annual_hp_growth, collateral_constraint, baseline_debt, 
               initial_price, int_rate, pra_forgive, hsg_rent_p, hsg_own_p, 
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
    for i in range(1,T):
        #print "age: " + str(i + age_at_mod) + " has growth fac: " + str(Params.PermGroFac[(i-1) + age_at_mod - 25])
        perm_gro = Params.PermGroFac[i + age_at_mod - 26]
        price.append(price[-1]*(1+annual_hp_growth)/perm_gro)
        hsg_pmt.append(hsg_pmt[-1]/perm_gro)
        debt.append((debt[-1]*(1+int_rate))/perm_gro - hsg_pmt[-1]) #xx double-check timing assumptions here
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
    
    #housing payments in retirement
    if default:
        hsg_pmt_ret_y = False
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
    return sale_proceeds, equity, limit, hsg_pmt


def pra_pmt(annual_hp_growth, collateral_constraint, baseline_debt, initial_price, 
            int_rate, pra_forgive, hsg_rent_p, hsg_own_p, hsg_pmt_wk_own = True, hsg_pmt_ret_y = False,
            age = 45, forgive = hamp_params['pra_forgive']):
    r, e, L, d_hamp = hsg_wealth(initial_debt =  hamp_params['baseline_debt'], age_at_mod = age, hsg_pmt_wk_own = hsg_pmt_wk_own, hsg_pmt_ret_y = hsg_pmt_ret_y, **hamp_params)
    r, e, L, d_prin_red = hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - forgive, age_at_mod = age, hsg_pmt_wk_own = hsg_pmt_wk_own, hsg_pmt_ret_y = hsg_pmt_ret_y, **hamp_params)
    pra_pmt = d_prin_red[:age-26] + d_hamp[age-26:age-21] + d_prin_red[age-21:]
    return pra_pmt

#calculate NPV of mortgage payments at mod date
#remark: this calculation ignores the higher payments in the first five years (which we are assuming get flushed down the toilet)
def npv_mtg_nominal(initial_debt, annual_hp_growth, collateral_constraint, 
                    baseline_debt, initial_price, int_rate, pra_forgive, 
                    hsg_rent_p, hsg_own_p,  age_at_mod = 45):
    if settings.verbose:
        print "Hsg wealth params: P=", initial_price, " D=", baseline_debt, " g=", annual_hp_growth, " r=", int_rate, " phi=", collateral_constraint
    T = 65 - age_at_mod
    price = [initial_price]
    debt = [initial_debt]
    amort = int_rate*(1+int_rate)**30/((1+int_rate)**30-1)
#    mtg_pmt = [initial_debt*amort]
    nom_pmt = initial_debt*amort
    npv_debt = nom_pmt
    for i in range(1,T):
        price.append(price[-1]*(1+annual_hp_growth))
#        mtg_pmt.append(mtg_pmt[-1])
        debt.append((debt[-1]*(1+int_rate)) - nom_pmt) #xx double-check timing assumptions here
        npv_debt += nom_pmt/((1+int_rate)**i)
    npv_debt = npv_debt + debt[T-1]/((1+int_rate)**T) 
    npv_asset = price[T-1]/((1+int_rate)**T)   
    npv_stay = npv_asset - npv_debt
    return npv_asset, npv_debt, npv_stay

a, d, npv_stay = npv_mtg_nominal(initial_debt =  hamp_params['baseline_debt'],**hamp_params)
a, d, npv_stay_pra = npv_mtg_nominal(initial_debt =  hamp_params['baseline_debt'] -hamp_params['pra_forgive'],**hamp_params)
py_out.update_acell('B2', npv_stay_pra-npv_stay)


###########################################################################
# Solve consumer problems
###########################################################################
pandas2ri.activate() 
reload(Params)  
baseline_params = Params.init_consumer_objects

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
    

baseline_params['aXtraCount'] = 30
baseline_params['DiscFac'] = (np.ones(65)*0.96).tolist()
baseline_params['vFuncBool'] = True
settings.verbose = False
#baseline_params['IncUnemp'] = inc_params['inc_unemp']
#baseline_params['UnempPrb'] = inc_params['prb_unemp']
#xxx this code exists only on default side and not on consumption side
py_out.update_acell('B9', baseline_params['PermGroFac'][0])
py_out.update_acell('B10', baseline_params['PermGroFac'][39])
py_out.update_acell('B11', baseline_params['CRRA'])
py_out.update_acell('B12', baseline_params['DiscFac'][0])


IndShockExample = solve_unpack(baseline_params)

###########################################################################
# Set parameters
###########################################################################
def inc_params(TranShkCount = 15, inc_shock_rescale = 1, p_unemp = 0.05, CRRA = 4  ):
    agent_params = deepcopy(baseline_params)
    agent_params['TranShkCount'] = TranShkCount
    agent_params['TranShkStd'] = [item*inc_shock_rescale for item in agent_params['TranShkStd']]
    agent_params['UnempPrb'] = p_unemp
    agent_params['CRRA'] = CRRA
    return agent_params


#can also be set further down
def_p = pd.DataFrame({ "grid_n": 75, "inc_sd": 3.5, "p_unemp": 0., "stig": -7, "CRRA": 4}, index=["std"])
params_std = inc_params(TranShkCount = int(def_p.loc["std","grid_n"]), inc_shock_rescale = def_p.loc["std","inc_sd"], p_unemp = def_p.loc["std","p_unemp"])
def_p = def_p.append(pd.DataFrame({ "grid_n": 25, "inc_sd": 1.4, "p_unemp": 0.07, "stig": -7, "CRRA": 4}, index=["u"]))
params_u = inc_params(TranShkCount = int(def_p.loc["u","grid_n"]), inc_shock_rescale = def_p.loc["u","inc_sd"], p_unemp = def_p.loc["u","p_unemp"])
def_p = def_p.append(pd.DataFrame({ "grid_n": 25, "inc_sd": 1.4, "p_unemp": 0.07, "stig": -7, "CRRA": 2}, index=["u_crra"]))
params_u_crra = inc_params(TranShkCount = int(def_p.loc["u","grid_n"]), inc_shock_rescale = def_p.loc["u","inc_sd"], p_unemp = def_p.loc["u","p_unemp"], CRRA = def_p.loc["u_crra","CRRA"])
def_p.to_csv(out_path + "default_params.csv")



uw_house_params = deepcopy(params_u) #uw_house_params['BoroCnstArt']
uw_house_params['rebate_amt'], e, L, uw_house_params['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'], **hamp_params) 
pra_params = deepcopy(params_u) #pra_params['BoroCnstArt']
pra_params['rebate_amt'], e, L, pra_params['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - hamp_params['pra_forgive'], **hamp_params)
default_params = deepcopy(params_u)
r, e, L, default_params['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'], default = True, **hamp_params) 

   
###########################################################################
# Solve basic consumption problem
###########################################################################
HouseExample = solve_unpack(uw_house_params)
PrinFrgvExample = solve_unpack(pra_params)
agent_d = solve_unpack(default_params)

def v_stig(m,stig,vf):
    return(vf(m)+stig)
stig_cnst = 7 #was 12
py_out.update_acell('B6', stig_cnst)
v_def_stig = partial(v_stig, stig = - stig_cnst, vf = agent_d.solution[t_eval].vFunc) 
stig_cnst_lo = 3
v_def_stig_lo = partial(v_stig, stig = - stig_cnst_lo, vf = agent_d.solution[t_eval].vFunc) 

yr = robjects.r('c(-30,-5)')
labels = ["Pay Mortgage                ",
          "Default, Utility Cost of " + str(stig_cnst),
          "Pay Mortgage (Post-PRA)"] #"Default, Utility Cost of " + str(stig_cnst_lo)
funcs = [HouseExample.solution[t_eval].vFunc,
         v_def_stig,
         PrinFrgvExample.solution[t_eval].vFunc] #v_def_stig_lo agent_d.solution[t_eval].vFunc
g = gg_funcs(funcs,0.3,3, N=20, loc=robjects.r('c(1,0)'),
        title = "Value Functions", labels = labels, ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)")
g+= gg.ylim(yr)
mp.ggsave("value_funcs_house_backup",g)        
ggplot_notebook(g, height=300,width=400)

###########################################################################
# Default rates and loan-to-value
###########################################################################
#reload(Params)  
#baseline_params = Params.init_consumer_objects
#agent = Model.IndShockConsumerType(**baseline_params)
#calculate a default rate
#(1) calculate m* where value funcs cross
#m_star = findIntersection(v_def_stig,HouseExample.solution[t_eval].vFunc,1.0)
#x = np.linspace(0.3,2,50)
#pylab.plot(x,v_def_stig(x),x,HouseExample.solution[t_eval].vFunc(x),result,v_def_stig(m_star),'ro')
#pylab.show()
#(2) calculate the fraction of income shocks below this thresholds
#sum(HouseExample.IncomeDstn[0][0][HouseExample.IncomeDstn[0][2]<m_star])


def hsg_params(params,ltv,default = False, pra = False, pra_start = hamp_params['baseline_debt'], add_hsg = 0): #xxx code is missing uw_house_params['BoroCnstArt']
    new_params = deepcopy(params)
    new_params['rebate_amt'], e, L, new_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['initial_price']*(ltv/100.),default = default, **hamp_params) 
    if default:
        new_params['rebate_amt'] = 0
    if pra:
        forgive_amt = pra_start - hamp_params['initial_price']*(ltv/100.)
        new_params['rebate_amt'], e, L, d = \
            hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - forgive_amt, **hamp_params)
        new_params['HsgPay'] = pra_pmt(age = 45, forgive = forgive_amt, **hamp_params) 
        #new_params['BoroCnstArt'] = L
    new_params['HsgPay'] = map(add,new_params['HsgPay'],[0] * 19 + [add_hsg] * 20 + [0] * 26)
    return new_params
    
def default_rate_solved(v_func_def,agent):
    if agent.solution[t_eval].vFunc(3) < v_func_def(3):
        m_star = 3
    elif agent.solution[t_eval].vFunc(0.3) > v_func_def(0.3):
        m_star = 0.3
    else:
        m_star = findIntersection(v_func_def,agent.solution[t_eval].vFunc,0.5)
    share_default = sum(agent.IncomeDstn[0][0][agent.IncomeDstn[0][2]<m_star])
    return m_star, share_default


settings.lil_verbose = False
ltv_rows = range(30,190,10) 
def_p.loc["std","stig"] = -6
def_p.loc["u","stig"] = -6
def_p.loc["u_crra","stig"] = -3
m_std_pra_list = []
def_std_pra_list = []
m_u_pra_list = []
def_u_pra_list = []
m_u_pra_stig_lo_list = []
def_u_pra_stig_lo_list = []
m_u_pra_stig_hi_list = []
def_u_pra_stig_hi_list = []
m_u_pra_crra_list = []
def_u_pra_crra_list = []
m_star_hi_dti_list = []
def_hi_dti_list = []
m_star_hi_dti_list2 = []
def_hi_dti_list2 = []
def_params = hsg_params(params_u, ltv = hamp_params['baseline_debt']/hamp_params['initial_price'], default = True)
agent_d = solve_unpack(def_params)
v_def_stig_tmp = partial(v_stig, stig = def_p.loc["u","stig"], vf = agent_d.solution[t_eval].vFunc) 
v_def_stig_tmp_lo = partial(v_stig, stig = def_p.loc["u","stig"] + 0.25, vf = agent_d.solution[t_eval].vFunc) 
v_def_stig_tmp_hi = partial(v_stig, stig = def_p.loc["u","stig"] - 0.25, vf = agent_d.solution[t_eval].vFunc) 
def_params = hsg_params(params_u_crra, ltv = hamp_params['baseline_debt']/hamp_params['initial_price'], default = True)
agent_d_crra = solve_unpack(def_params)
v_def_stig_crra = partial(v_stig, stig = def_p.loc["u_crra","stig"], vf = agent_d_crra.solution[t_eval].vFunc) 
def_params = hsg_params(params_std, ltv = hamp_params['baseline_debt']/hamp_params['initial_price'], default = True)
agent_d = solve_unpack(def_params)
v_def_stig_std = partial(v_stig, stig = def_p.loc["std","stig"], vf = agent_d.solution[t_eval].vFunc) 

for ltv in ltv_rows:
    mtg_params = hsg_params(params_std, ltv = ltv, pra = True) 
    agent_nd = solve_unpack(mtg_params)
    m_star, share_default = default_rate_solved(v_def_stig_std,agent_nd)
    m_std_pra_list.append(m_star)
    if ltv <= 100: share_default = 0
    def_std_pra_list.append(share_default)
    
    mtg_params = hsg_params(params_u, ltv = ltv, pra = True) 
    agent_nd = solve_unpack(mtg_params)
    m_star, share_default = default_rate_solved(v_def_stig_tmp,agent_nd)
    m_u_pra_list.append(m_star)
    if ltv <= 100: share_default = 0
    def_u_pra_list.append(share_default)
   
    m_star, share_default = default_rate_solved(v_def_stig_tmp_lo,agent_nd)
    m_u_pra_stig_lo_list.append(m_star)
    if ltv <= 100: share_default = 0
    def_u_pra_stig_lo_list.append(share_default)
 
    m_star, share_default = default_rate_solved(v_def_stig_tmp_hi,agent_nd)
    m_u_pra_stig_hi_list.append(m_star)
    if ltv <= 100: share_default = 0
    def_u_pra_stig_hi_list.append(share_default)
  
    mtg_hi_dti_params = hsg_params(params_u, ltv = ltv, pra = True, add_hsg = 0.02)
    agent_hi_dti_nd = solve_unpack(mtg_hi_dti_params)
    v_def_stig_tmp2 = partial(v_stig, stig = def_p.loc["u","stig"], vf = agent_d.solution[t_eval].vFunc) 
    m_star, share_default = default_rate_solved(v_def_stig_tmp2,agent_hi_dti_nd)
    if ltv <= 100: share_default = 0
    m_star_hi_dti_list.append(m_star)
    def_hi_dti_list.append(share_default)
    
    mtg_hi_dti_params = hsg_params(params_u, ltv = ltv, pra = True, add_hsg = 0.05)
    agent_hi_dti_nd2 = solve_unpack(mtg_hi_dti_params)
    v_def_stig_tmp3 = partial(v_stig, stig = def_p.loc["u","stig"], vf = agent_d.solution[t_eval].vFunc) 
    m_star, share_default = default_rate_solved(v_def_stig_tmp2,agent_hi_dti_nd2)
    if ltv <= 100: share_default = 0
    m_star_hi_dti_list2.append(m_star)
    def_hi_dti_list2.append(share_default)
    
    print v_def_stig_tmp(1), agent_nd.solution[t_eval].vFunc(1), agent_hi_dti_nd2.solution[t_eval].vFunc(1)
    
    mtg_params = hsg_params(params_u_crra, ltv = ltv, pra = True) 
    agent_nd = solve_unpack(mtg_params)
    m_star, share_default = default_rate_solved(v_def_stig_crra,agent_nd)
    m_u_pra_crra_list.append(m_star)
    if ltv <= 100: share_default = 0
    def_u_pra_crra_list.append(share_default)
    print v_def_stig_crra(1), agent_nd.solution[t_eval].vFunc(1)
    

mstar_std_pra_f = LinearInterp(ltv_rows,np.array(m_std_pra_list))
def_std_pra_f = LinearInterp(ltv_rows,np.array(def_std_pra_list))
mstar_u_pra_f = LinearInterp(ltv_rows,np.array(m_u_pra_list))
def_u_pra_f = LinearInterp(ltv_rows,np.array(def_u_pra_list))
mstar_u_pra_stig_lo_f = LinearInterp(ltv_rows,np.array(m_u_pra_stig_lo_list))
def_u_pra_stig_lo_f = LinearInterp(ltv_rows,np.array(def_u_pra_stig_lo_list))
mstar_u_pra_stig_hi_f = LinearInterp(ltv_rows,np.array(m_u_pra_stig_hi_list))
def_u_pra_stig_hi_f = LinearInterp(ltv_rows,np.array(def_u_pra_stig_hi_list))
mstar_u_pra_crra_f = LinearInterp(ltv_rows,np.array(m_u_pra_crra_list))
def_u_pra_crra_f = LinearInterp(ltv_rows,np.array(def_u_pra_crra_list))
mstar_hi_dti_f = LinearInterp(ltv_rows,np.array(m_star_hi_dti_list))
def_hi_dti_f = LinearInterp(ltv_rows,np.array(def_hi_dti_list))
mstar_hi_dti_f2 = LinearInterp(ltv_rows,np.array(m_star_hi_dti_list2))
def_hi_dti_f2 = LinearInterp(ltv_rows,np.array(def_hi_dti_list2))

py_out.update_acell('B4', def_hi_dti_f2(130)-def_hi_dti_f2(119))
py_out.update_acell('B5', def_std_pra_f(130)-def_std_pra_f(119))


labels = ["Model: Baseline","Model: Match Xsec Correlation"]
labels_pmt = ["Baseline","Raise Mortgage Pmts by 2% of Income","Raise Mortgage Pmts by 5% of Income"]
labels_stig = ["Baseline","Default Stigma Low","Default Stigma High"]
labels_pra = ["PRA, CRRA = 4","PRA, CRRA = 2"]

g = gg_funcs([def_u_pra_f,def_hi_dti_f2], #
            min(ltv_rows),max(ltv_rows)+10, N=len(ltv_rows), loc=robjects.r('c(0,1)'),
        title = "Default Rate", labels = [labels_pmt[0] , labels_pmt[2]],
        ylab = "Share Defaulting ", xlab = "Loan-to-Value", file_name = "default_rate")
ggplot_notebook(g, height=300,width=400)


g = gg_funcs([def_u_pra_f], #
            min(ltv_rows),max(ltv_rows)+10, N=len(ltv_rows), loc=robjects.r('c(0,1)'),
        title = "Default Rate", labels = [labels_pmt[0]],
        ylab = "Share Defaulting ", xlab = "Loan-to-Value")
g += gg.ylim(robjects.r('c(0,1)'))
mp.ggsave("default_rate_slide1",g)
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([def_u_pra_f,def_std_pra_f], #
            min(ltv_rows),max(ltv_rows)+10, N=len(ltv_rows), loc=robjects.r('c(0,1)'),
        title = "Default Rate", labels = labels,
        ylab = "Share Defaulting ", xlab = "Loan-to-Value", file_name = "default_rate_inc_backup")
ggplot_notebook(g, height=300,width=400)


g = gg_funcs([def_u_pra_f,def_hi_dti_f,def_hi_dti_f2], 
            min(ltv_rows),max(ltv_rows)+10, N=len(ltv_rows), loc=robjects.r('c(0,1)'),
        title = "Default Rate", labels = labels_pmt,
        ylab = "Share Defaulting ", xlab = "Loan-to-Value", file_name = "default_rate_pmt_backup")
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([def_u_pra_f,def_u_pra_stig_lo_f,def_u_pra_stig_hi_f], 
            min(ltv_rows),max(ltv_rows)+10, N=len(ltv_rows), loc=robjects.r('c(0,1)'),
        title = "Default Rate", labels = labels_stig,
        ylab = "Share Defaulting ", xlab = "Loan-to-Value", file_name = "default_rate_stig_backup")
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([def_u_pra_f,def_u_pra_crra_f], 
            min(ltv_rows),max(ltv_rows)+10, N=len(ltv_rows), loc=robjects.r('c(0,1)'),
        title = "Default Rate", labels = labels_pra,
        ylab = "Share Defaulting ", xlab = "Loan-to-Value", file_name = "default_rate_crra_backup")
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([mstar_u_pra_f,mstar_hi_dti_f2], 
            min(ltv_rows),max(ltv_rows)+10, N=50, loc=robjects.r('c(0,1)'),
        title = "Cutoff Values to Leave House \n Default or Sell if Income + Assets < Threshold",
        labels = [labels_pmt[0] , labels_pmt[2]],
        ylab = "Income + Assets Threshold", xlab = "Loan-to-Value")
g += gg.ylim(robjects.r('c(0.5,' + str(mstar_hi_dti_f2(180)+0.05) + ')'))  
mp.ggsave("default_inc_threshold",g)      
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([mstar_u_pra_f], 
            min(ltv_rows),max(ltv_rows)+10, N=50, loc=robjects.r('c(0,1)'),
        title = "Cutoff Values to Leave House \n Default or Sell if Income + Assets < Threshold",
        labels = [labels_pmt[0] , labels_pmt[2]], 
        ylab = "Income + Assets Threshold", xlab = "Loan-to-Value")
g += gg.ylim(robjects.r('c(0.5,' + str(mstar_hi_dti_f2(180+0.05)) + ')'))
mp.ggsave("default_inc_threshold_slide1",g)
ggplot_notebook(g, height=300,width=400)


g = gg_funcs([mstar_u_pra_f,mstar_std_pra_f], #mstar_f
            min(ltv_rows),max(ltv_rows)+10, N=50, loc=robjects.r('c(0,1)'),
        title = "Cutoff Values to Leave House \n Default or Sell if Income + Assets < Threshold",
        labels = labels, 
        ylab = "Income + Assets Threshold", xlab = "Loan-to-Value", file_name = "default_inc_threshold_inc_backup")
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([mstar_u_pra_f,mstar_hi_dti_f,mstar_hi_dti_f2], #mstar_f
            min(ltv_rows),max(ltv_rows)+10, N=50, loc=robjects.r('c(0,1)'),
        title = "Cutoff Values to Leave House \n Default or Sell if Income + Assets < Threshold",
        labels = labels_pmt, #ltitle = ["Income Risk Process"]
        ylab = "Income + Assets Threshold", xlab = "Loan-to-Value", file_name = "default_inc_threshold_pmt_backup")
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([mstar_u_pra_f,mstar_u_pra_stig_lo_f,mstar_u_pra_stig_hi_f], #mstar_f
            min(ltv_rows),max(ltv_rows)+10, N=50, loc=robjects.r('c(0,1)'),
        title = "Cutoff Values to Leave House \n Default or Sell if Income + Assets < Threshold",
        labels = labels_stig, #ltitle = ["Income Risk Process"]
        ylab = "Income + Assets Threshold", xlab = "Loan-to-Value", file_name = "default_inc_threshold_stig_backup")
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([mstar_u_pra_f,mstar_u_pra_crra_f], #mstar_f
            min(ltv_rows),max(ltv_rows)+10, N=50, loc=robjects.r('c(0,1)'),
        title = "Cutoff Values to Leave House \n Default or Sell if Income + Assets < Threshold",
        labels = labels_pra, #ltitle = ["Income Risk Process"]
        ylab = "Income + Assets Threshold", xlab = "Loan-to-Value", file_name = "default_inc_threshold_crra_backup")
ggplot_notebook(g, height=300,width=400)



###########################################################################
# Income Shock distn
###########################################################################

loc = robjects.r('c(1,1)')
agent_std = solve_unpack(params_std)
agent_u = solve_unpack(params_u) 
df_inc = pd.DataFrame({"x":agent_std.IncomeDstn[0][2], 
                       "value":agent_std.IncomeDstn[0][0],
                        "variable":labels[1]})
df_inc_u = pd.DataFrame({"x":agent_u.IncomeDstn[0][2], #xxx these are the wrong bins!
                       "value":agent_u.IncomeDstn[0][0],
                        "variable":labels[0]})    
df_inc = df_inc.append(df_inc_u)                       
g = gg.ggplot(df_inc) + mp.base_plot + mp.colors  + gg.aes_string(fill='variable') +  \
        gg.geom_bar(stat="identity", width = 0.01, position = gg.position_dodge(width=0.01))  + \
        mp.theme_bw(base_size=9) + mp.fte_theme + gg.xlim(robjects.r('c(0.3,1.7)')) + \
        gg.labs(title="Temporary Income Shock Distribution",y="Probability ",x="Ratio of Temporary Income to Permanent Income") + \
        mp.legend_f(loc) + mp.legend_t_c("variable") 
mp.ggsave("inc_dstn_xsec_backup",g)
ggplot_notebook(g, height=300,width=400)

g = gg.ggplot(df_inc_u) + mp.base_plot + mp.colors  + gg.aes_string(fill='variable') +  \
        gg.geom_bar(stat="identity", width = 0.01, position = gg.position_dodge(width=0.01))  + \
        mp.theme_bw(base_size=9) + mp.fte_theme + gg.xlim(robjects.r('c(0.3,1.7)')) + \
        gg.labs(title="Temporary Income Shock Distribution",y="Probability ",x="Ratio of Temporary Income to Permanent Income") + \
        mp.legend_f(loc) + mp.legend_t_c("variable") 
mp.ggsave("inc_dstn_backup",g)
ggplot_notebook(g, height=300,width=400)

###########################################################################
# NPV of asset and of mortgage
###########################################################################

npv_a_list = []
npv_d_list = []
for ltv in ltv_rows:
    npv_a, npv_d, npv_diff = npv_mtg_nominal(initial_debt =  hamp_params['initial_price']*(ltv/100.),**hamp_params)
    npv_a_list.append(npv_a)
    npv_d_list.append(npv_d)
npv_a_f = LinearInterp(ltv_rows,np.array(npv_a_list))
npv_d_f = LinearInterp(ltv_rows,np.array(npv_d_list))

g = gg_funcs([npv_a_f,npv_d_f],
            min(ltv_rows),max(ltv_rows), N=50, loc=robjects.r('c(0,1)'),
        title = "Net Present Value at Age 45 of Asset and Debt", labels = ["Asset","Debt"],
        ylab = "Net Present Value (Years of Income)", xlab = "Loan-to-Value", 
        file_name = "npv_ltv_backup")
ggplot_notebook(g, height=300,width=400)

tmp = deepcopy(hamp_params['annual_hp_growth'])
hamp_params['annual_hp_growth'] = 0.02
npv_a_list = []
npv_d_list = []
for ltv in ltv_rows:
    npv_a, npv_d, npv_diff = npv_mtg_nominal(initial_debt =  hamp_params['initial_price']*(ltv/100.),**hamp_params)
    npv_a_list.append(npv_a)
    npv_d_list.append(npv_d)
npv_a_f = LinearInterp(ltv_rows,np.array(npv_a_list))
npv_d_f = LinearInterp(ltv_rows,np.array(npv_d_list))

g = gg_funcs([npv_a_f,npv_d_f],
            min(ltv_rows),max(ltv_rows), N=50, loc=robjects.r('c(0,1)'),
        title = "Net Present Value at Age 45 of Asset and Debt \n Set House Price Growth = Interest Rate on Assets", labels = ["Asset","Debt"],
        ylab = "Net Present Value (Years of Income)", xlab = "Loan-to-Value", 
        file_name = "npv_ltv_interest_parity_diag")
ggplot_notebook(g, height=300,width=400)
hamp_params['annual_hp_growth'] = tmp

###########################################################################
# Begin Diagnostic plots
###########################################################################

#find lowest utility cost code
#xx this is 
inc_min = agent_u.IncomeDstn[0][2][1]
stig_grid = np.arange(-10,0.5,0.25)
stig_list = []
stig_list_asset = []
stig_list_crra = []
for ltv in ltv_rows:
    mtg_params = hsg_params(params_std, ltv = ltv)
    def_params = hsg_params(params_std, ltv = ltv, default = True)
    agent_nd = solve_unpack(mtg_params)
    agent_d = solve_unpack(def_params)
    mtg_params['CRRA'] = 2
    def_params['CRRA'] = 2
    agent_nd_low_crra = solve_unpack(mtg_params)
    agent_d_low_crra = solve_unpack(def_params)
    stig_lowest = min(stig_grid)
    stig_lowest_asset = min(stig_grid)
    stig_lowest_crra = min(stig_grid)
    for stig_cnst in stig_grid:
        v_def_stig_tmp = partial(v_stig, stig = stig_cnst, vf = agent_d.solution[t_eval].vFunc) 
        m_star, share_default = default_rate_solved(v_def_stig_tmp,agent_nd)
        if v_def_stig_tmp(inc_min) < agent_nd.solution[t_eval].vFunc(inc_min):
            stig_lowest = stig_cnst
        if v_def_stig_tmp(inc_min+1) < agent_nd.solution[t_eval].vFunc(inc_min+1):
            stig_lowest_asset = stig_cnst
        v_def_stig_tmp = partial(v_stig, stig = stig_cnst, vf = agent_d_low_crra.solution[t_eval].vFunc) 
        if v_def_stig_tmp(inc_min) < agent_nd_low_crra.solution[t_eval].vFunc(inc_min):
            stig_lowest_crra = stig_cnst
        
    print ltv, stig_lowest
    stig_list.append(stig_lowest)   
    stig_list_asset.append(stig_lowest_asset)  
    stig_list_crra.append(stig_lowest_crra)  
    print "V(0.3): ", v_def_stig_tmp(0.3), agent_nd.solution[t_eval].vFunc(0.3)
    print "V(3): ", v_def_stig_tmp(3), agent_nd.solution[t_eval].vFunc(3)

stig_f = LinearInterp(ltv_rows,np.array(stig_list))
stig_asset_f = LinearInterp(ltv_rows,np.array(stig_list_asset))
stig_crra_f = LinearInterp(ltv_rows,np.array(stig_list_crra))
labels_asset = ["Baseline (Asset = 0)","Assets = 1"]
g = gg_funcs([stig_f,stig_asset_f], #stig_crra_f
            min(ltv_rows),max(ltv_rows)+10, N=len(ltv_rows), loc=robjects.r('c(0,0)'),
        title = "Moving Cost Such That Agent Sells or Defaults Only If Unemployed", labels = labels_asset,
        ylab = "Utility Cost ", xlab = "Loan-to-Value", file_name = "util_cost_by_ltv_diag")
ggplot_notebook(g, height=300,width=400)

#
#g = gg_funcs([agent_d.solution[t_eval].vPPfunc,HouseExample.solution[t_eval].vPPfunc],
#            0.9,2, N=50, loc=robjects.r('c(1,0)'),
#        title = "Marginal Marginal Value Functions.", labels = ["Default","Pay Mortgage"],
#        ylab = "Marg Marg Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_vPPfunc_diag")
#ggplot_notebook(g, height=300,width=400)
#
#g = gg_funcs([agent_d.solution[t_eval].vPfunc,HouseExample.solution[t_eval].vPfunc],
#            0.9,2, N=50, loc=robjects.r('c(1,0)'),
#        title = "Marginal Value Functions.", labels = ["Default","Pay Mortgage"],
#        ylab = "Marg Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_vPfunc_diag")
#ggplot_notebook(g, height=300,width=400)
#
#g = gg_funcs([agent_d.solution[t_eval].vPfunc,HouseExample.solution[t_eval].vPfunc],
#            0.2,0.9, N=50, loc=robjects.r('c(1,0)'),
#        title = "Marginal Value Functions.", labels = ["Default","Pay Mortgage"],
#        ylab = "MargValue", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_vPfunc_low_a_diag")
#ggplot_notebook(g, height=300,width=400)

g = gg_funcs([IndShockExample.solution[t_eval].cFunc,HouseExample.solution[t_eval].cFunc],
            0.5,6, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Functions.", labels = ["Baseline","With House"],
        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "c_funcs_house_diag")
ggplot_notebook(g, height=300,width=400)

g = gg_funcs([IndShockExample.solution[t_eval].cFunc,HouseExample.solution[t_eval].cFunc],
            0.1,1.5, N=50, loc=robjects.r('c(1,0)'),
        title = "Consumption Functions.", labels = ["Baseline","With House"],
        ylab = "Consumption", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "c_funcs_house_low_diag")
ggplot_notebook(g, height=300,width=400)


##############################################
#show how value function changes with parameters of problem
##############################################

#rrr this code is highly repetitive so it can be cleaned up
#plot value function by age
#it is steepest right before retirement and most shallow before death
vFuncs_age = []
for i in range(0,65,9):
    vFuncs_age.append(IndShockExample.solution[i].vFunc)
g = gg_funcs(vFuncs_age,0.5,4, N=200, loc=robjects.r('c(1,0)'),
        title = "Value Functions. Each line is 9 years forward in time",
        ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_age_diag")
ggplot_notebook(g, height=300,width=400)

#plot value function by housing costs
HsgPay_solution = []
vFuncs_d = []
for i in np.arange(0.2,0.3,0.02):
    tmp_params = deepcopy(baseline_params)
    tmp_params['HsgPay'] = map(add,tmp_params['HsgPay'], [i]*65)
    HsgPay_solution.append(solve_unpack(tmp_params).solution)
    vFuncs_d.append(HsgPay_solution[-1][t_eval].vFunc)
g = gg_funcs(vFuncs_d,0.5,5, N=20, loc=robjects.r('c(1,0)'),
        title = "Value Functions. Each line is 0.02 higher housing cost starting w base of 0.2. Age 45",
        ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_d_diag")
ggplot_notebook(g, height=300,width=400)

#plot value function by age 65 rebate
rebate_solution = []
vFuncs_reb = []
for i in np.arange(5):
    tmp_params = deepcopy(baseline_params)
    tmp_params['rebate_amt'] = tmp_params['rebate_amt'] + i
    map(add,tmp_params['HsgPay'], [0.2]*65)
    rebate_solution.append(solve_unpack(tmp_params).solution)
    vFuncs_reb.append(rebate_solution[-1][t_eval].vFunc)
g = gg_funcs(vFuncs_reb,0.5,5, N=20, loc=robjects.r('c(1,0)'),
        title = "Value Functions. Each line is 1 higher rebate. Age 45",
        ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_rebate_diag")
ggplot_notebook(g, height=300,width=400)


#plot value function by borrowing limit 
Boro_solution = []
vFuncs_L = []
for i in np.arange(0,1,0.2):
    tmp_params = deepcopy(baseline_params)
    tmp_params['BoroCnstArt'] = map(sub,tmp_params['BoroCnstArt'], [i]*65)
    Boro_solution.append(solve_unpack(tmp_params).solution)
    vFuncs_L.append(Boro_solution[-1][t_eval].vFunc)
g = gg_funcs(vFuncs_L,0.5,2, N=200, loc=robjects.r('c(1,0)'),
        title = "Value Functions. Each line is 0.2 higher borrowing limit. Age 45",
        ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_L_diag")
ggplot_notebook(g, height=300,width=400)

#borrow only while working
Boro_solution = []
vFuncs_L = []
for i in np.arange(0,0.6,0.2):
    tmp_params = deepcopy(baseline_params)
    tmp_params['BoroCnstArt'] = map(sub,tmp_params['BoroCnstArt'], [0] * 20 + [i]*20 + [0]*25)
    tmp_params['rebate_amt'] = i
    Boro_solution.append(solve_unpack(tmp_params).solution)
    vFuncs_L.append(Boro_solution[-1][t_eval].vFunc)
g = gg_funcs(vFuncs_L[:3],0.5,2, N=200, loc=robjects.r('c(1,0)'),
        title = "Value Functions. Each line is 0.2 higher borrowing limit. Age 45",
        ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_L_wrk_diag")
ggplot_notebook(g, height=300,width=400)

for i in range(65):
    print "Vfunc ", i, Boro_solution[-1][i].vFunc(0.2)
    

###########################################################################
# Value functions DEBUGGING
###########################################################################
#diagnostic: calculate default discount
#map(sub,uw_house_params['HsgPay'],default_params['HsgPay'])  

hamp_params['collateral_constraint'] = 0
i_d = hamp_params['initial_price']*0.8
default_params = deepcopy(baseline_params)
r, e, L, default_params['HsgPay'] = \
    hsg_wealth(initial_debt =  i_d, default = True, **hamp_params) 
default_params_mtg = deepcopy(params_u)
r, e, L, default_params_mtg['HsgPay'] = \
    hsg_wealth(initial_debt =  i_d, **hamp_params) 
default_params_mtg_pay = deepcopy(params_u)
default_params_mtg_pay['rebate_amt'], e, L, default_params_mtg_pay['HsgPay'] = \
    hsg_wealth(initial_debt =  i_d, **hamp_params) 
#default = solve_unpack(default_params)
#default_mtg = solve_unpack(default_params_mtg)
#default_mtg_pay = solve_unpack(default_params_mtg_pay)
house_params = deepcopy(baseline_params)
house_params['rebate_amt'], e, house_params['BoroCnstArt'], house_params['HsgPay'] = \
    hsg_wealth(initial_debt =  i_d, **hamp_params)
house_params['rebate_amt'] += 0.01
#house_params['BoroCnstArt'] = map(sub,house_params['BoroCnstArt'], [0] * 20 + [0]*20 + [0]*25)
house_params2 = deepcopy(baseline_params)
house_params2['BoroCnstArt'] = map(sub,house_params2['BoroCnstArt'], [0] * 20 + [0.4]*20 + [0]*25)
house_params2['rebate_amt'] = 0.4
hamp_params['collateral_constraint'] = 0.2 

settings.verbose = False
reload(Model)
settings.min_age, settings.max_age = 64,65
house = solve_unpack(house_params)
for i in range(65):
    print "vFunc ", i, house.solution[i].vFunc(5)
settings.verbose = False
house2 = solve_unpack(house_params2)


#g = gg_funcs([house.solution[t_eval].vFunc,house2.solution[t_eval].vFunc],
#             0.5,5, N=50, loc=robjects.r('c(1,0)'),
#        title = "Value Functions, LTV = 80", 
#        ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_house_diag")
#ggplot_notebook(g, height=300,width=400)


    
    
#figure out where value func breaks. what value and what period starting w vfunc terminal. step thru debugger.
#KEY ISSUE: THIS EVALUATES TO nan: house2.solution[t_eval].vFunc(4)
#KEY ISSUE: THIS EVALUATES TO nan: Boro_solution.solution[t_eval].vFunc for limit = -0.8
#why does the valu function evaluation to Nan when the borrowing limit  gets too large?
#need to find the cutoff by trial and error and then step through the code to figure out what's going on


###########################################################################
# Value functions
###########################################################################
#xxx note that is redundant to code run earlier.
#I am re-running since these items may have changed in the interim. in future, make sure ethat for loop does not change them
uw_house_params = deepcopy(params_u) #uw_house_params['BoroCnstArt']
uw_house_params['rebate_amt'], e, L, uw_house_params['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'], **hamp_params) 
pra_params = deepcopy(params_u) #pra_params['BoroCnstArt']
pra_params['rebate_amt'], e, L, pra_params['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'] - hamp_params['pra_forgive'], **hamp_params)
default_params = deepcopy(params_u)
r, e, L, default_params['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'], default = True, **hamp_params) 

HouseExample = solve_unpack(uw_house_params)
PrinFrgvExample = solve_unpack(pra_params)
agent_d = solve_unpack(default_params)
    
v_def_stig = partial(v_stig, stig = - stig_cnst, vf = agent_d.solution[t_eval].vFunc) 

v_def_stig_12 = partial(v_stig, stig =-12, vf = agent_d.solution[t_eval].vFunc) 
v_def_stig_4 = partial(v_stig,stig = -4, vf = agent_d.solution[t_eval].vFunc) 
v_def_stig_1 = partial(v_stig,stig = -1.5, vf = agent_d.solution[t_eval].vFunc) 

g = gg_funcs([HouseExample.solution[t_eval].vFunc,v_def_stig,
              v_def_stig_12,v_def_stig_4],
        0.4,2, N=20, loc=robjects.r('c(1,0)'),
        title = "Value Functions", ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", 
        labels = ["Pay Mortgage","Default, Utility Cost of " + str(stig_cnst),
                  "Default, Utility Cost of 12","Default, Utility Cost of 4"])
#g+= gg.ylim(yr)
mp.ggsave("value_funcs_vary_util_cost_diag",g)        
ggplot_notebook(g, height=300,width=400)


uw_parity_house_params = deepcopy(params_u)
default_parity_params = deepcopy(params_u)
hamp_params['annual_hp_growth'] = 0.02
uw_parity_house_params['rebate_amt'], e, L, uw_parity_house_params['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'], **hamp_params) 
r, e, L, default_parity_params['HsgPay'] = \
    hsg_wealth(initial_debt =  hamp_params['baseline_debt'], default = True, **hamp_params) 
hamp_params['annual_hp_growth'] = 0.009

agent_d_parity = solve_unpack(default_parity_params)
v_def_stig_parity = partial(v_stig, stig =-12, vf = agent_d_parity.solution[t_eval].vFunc) 
HouseExample_parity = solve_unpack(uw_parity_house_params)
g = gg_funcs([HouseExample.solution[t_eval].vFunc,
              HouseExample_parity.solution[t_eval].vFunc,
              v_def_stig_12,v_def_stig_parity],
        0.5,4, N=20, loc=robjects.r('c(1,0)'),
        title = "Value Functions", ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", 
        labels = ["Pay Mortgage","Pay Mortgage w/Asset Return Parity",
                  "Default","Default w/Asset Return Parity"])
mp.ggsave("value_funcs_vary_parity_diag",g)        
ggplot_notebook(g, height=300,width=400)

default_pih_params = deepcopy(default_params)
pih_hi = 20
default_pih_params['PermGroFac'][20] = (100. - pih_hi)/100.
agent_d_c_equiv_hi = solve_unpack(default_pih_params)
pih_med = 18
default_pih_params['PermGroFac'][20] = (100. - pih_med)/100.
agent_d_c_equiv_med = solve_unpack(default_pih_params)
pih_lo = 12
default_pih_params['PermGroFac'][20] = (100. - pih_lo)/100.
agent_d_c_equiv_lo = solve_unpack(default_pih_params)

g = gg_funcs([agent_d_c_equiv_hi.solution[t_eval].vFunc,
              agent_d_c_equiv_med.solution[t_eval].vFunc,
            agent_d_c_equiv_lo.solution[t_eval].vFunc,
              v_def_stig,HouseExample.solution[t_eval].vFunc],
             0.5,2, N=50, loc=robjects.r('c(1,0)'),
        title = "Value Functions", labels = 
        ['PIH Loss Hi ' + str(pih_hi),'PIH Loss Med ' + str(pih_med),'PIH Loss Lo ' + str(pih_lo),
         'Default w/Stigma of ' + str(stig_cnst),'Pay Mortgage'],
        ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_house_pih_diag")
ggplot_notebook(g, height=300,width=400)
py_out.update_acell('B7', pih_med)

default_hsg_pay_params = deepcopy(default_params)
default_hsg_pay_params['HsgPay'] = map(add,default_hsg_pay_params['HsgPay'],[0]*20 + [0.17]*45)
agent_d_c_equiv_hi = solve_unpack(default_hsg_pay_params)
default_hsg_pay_params['HsgPay'] = map(sub,default_hsg_pay_params['HsgPay'],[0]*20 + [0.03]*45)
agent_d_c_equiv_med = solve_unpack(default_hsg_pay_params)
default_hsg_pay_params['HsgPay'] = map(sub,default_hsg_pay_params['HsgPay'],[0]*20 + [0.02]*45)
agent_d_c_equiv_lo = solve_unpack(default_hsg_pay_params)

g = gg_funcs([agent_d_c_equiv_hi.solution[t_eval].vFunc,
              agent_d_c_equiv_med.solution[t_eval].vFunc,
            agent_d_c_equiv_lo.solution[t_eval].vFunc,
              v_def_stig,HouseExample.solution[t_eval].vFunc],
             0.5,2, N=50, loc=robjects.r('c(1,0)'),
        title = "Value Functions", labels = 
        ['Default, Pay 17% Extra of Inc','Default, Pay 14% Extra','Default, Pay 12% Extra',
         'Default w/Stigma','Pay Mortgage'],
        ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_house_hsg_pay_diag")
ggplot_notebook(g, height=300,width=400)

default_hsg_pay_params = deepcopy(default_params)
default_hsg_pay_params['HsgPay'] = map(add,default_hsg_pay_params['HsgPay'],[0]*20 + [0.095]*45)
agent_d_c_equiv_hi = solve_unpack(default_hsg_pay_params)
default_hsg_pay_params['HsgPay'] = map(sub,default_hsg_pay_params['HsgPay'],[0]*20 + [0.015]*45)
agent_d_c_equiv_med = solve_unpack(default_hsg_pay_params)
default_hsg_pay_params['HsgPay'] = map(sub,default_hsg_pay_params['HsgPay'],[0]*20 + [0.02]*45)
agent_d_c_equiv_lo = solve_unpack(default_hsg_pay_params)
g = gg_funcs([agent_d_c_equiv_hi.solution[t_eval].vFunc,
              agent_d_c_equiv_med.solution[t_eval].vFunc,
            agent_d_c_equiv_lo.solution[t_eval].vFunc,
              v_def_stig,PrinFrgvExample.solution[t_eval].vFunc],
             0.5,2, N=50, loc=robjects.r('c(1,0)'),
        title = "Value Functions", labels = 
        ['Default, Pay 9.5% Extra of Inc','Default, Pay 8% Extra','Default, Pay 6% Extra',
         'Default w/Stigma','Pay Mortgage (Post-PRA)'],
        ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_house_hsg_pay_pra_diag")
ggplot_notebook(g, height=300,width=400)

tmp = deepcopy(hamp_params['hsg_rent_p'])
hamp_params['hsg_rent_p'] = 0.075
r, e, L, default_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'], default = True, **hamp_params) 
agent_d_c_equiv_hi = solve_unpack(default_params)
hamp_params['hsg_rent_p'] = 0.07
r, e, L, default_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'], default = True, **hamp_params) 
agent_d_c_equiv_med = solve_unpack(default_params)
hamp_params['hsg_rent_p'] = 0.06
r, e, L, default_params['HsgPay'] = hsg_wealth(initial_debt =  hamp_params['baseline_debt'], default = True, **hamp_params) 
agent_d_c_equiv_lo = solve_unpack(default_params)
hamp_params['hsg_rent_p'] = tmp

g = gg_funcs([agent_d_c_equiv_hi.solution[t_eval].vFunc,
              agent_d_c_equiv_med.solution[t_eval].vFunc,
            agent_d_c_equiv_lo.solution[t_eval].vFunc,
              v_def_stig,HouseExample.solution[t_eval].vFunc],
             0.5,2, N=50, loc=robjects.r('c(1,0)'),
        title = "Value Functions", labels = 
        ['Default, Rent 7.5% of P','Default, Rent 7% of P','Default, Rent 6% of P',
         'Default w/Stigma','Pay Mortgage'],
        ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_house_rent_p_diag")
ggplot_notebook(g, height=300,width=400)

#
##borrow only while working
#Boro_solution = []
#vFuncs_L = []
#for i in np.arange(0,1,0.2):
#    tmp_params = deepcopy(baseline_params)
#    tmp_params['BoroCnstArt'] = map(sub,tmp_params['BoroCnstArt'], [0] * 20 + [i]*20 + [0]*25)
#    #without this, everything breaks    
#    tmp_params['rebate_amt'] = i
#    Boro_solution.append(solve_unpack(tmp_params).solution)
#    vFuncs_L.append(Boro_solution[-1][t_eval].vFunc)
#g = gg_funcs(vFuncs_L,0.5,2, N=200, loc=robjects.r('c(1,0)'),
#        title = "Value Functions. Each line is 0.2 higher borrowing limit. Age 45",
#        ylab = "Value", xlab = "Cash-on-Hand (Ratio to Permanent Income)", file_name = "value_funcs_L_wrk_diag")
#ggplot_notebook(g, height=300,width=400)
