# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:21:09 2016

@author: ganong
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
scope = ['https://spreadsheets.google.com/feeds']
credentials = ServiceAccountCredentials.from_json_keyfile_name('gspread-oauth.json', scope)
gc = gspread.authorize(credentials)
g_params = gc.open("HAMPRA Model Parameters").sheet1 #in this case
df = pd.DataFrame(g_params.get_all_records())
hamp_params = df[['Param','Value']].set_index('Param')['Value'].to_dict()

import EstimationParameters as Params


#in future, when simulating histories, need to adjust for realized income shocks as well
def hsg_wealth(debt, annual_hp_growth, collateral_constraint, initial_debt, initial_price, int_rate, pra_forgive, age_at_mod = 45):
    print annual_hp_growth, collateral_constraint, initial_debt, initial_price, int_rate
    T = 65 - age_at_mod
    price = [initial_price]
    debt = [debt]
    for i in range(1,T):
        print "age: " + str(i + age_at_mod) + " has growth fac: " + str(Params.PermGroFac[(i-1) + age_at_mod - 25])
        perm_gro = Params.PermGroFac[i + age_at_mod - 26]
        price.append(price[-1]*(1+annual_hp_growth)/perm_gro)
        debt.append(debt[-1]/perm_gro) #this is the no amortization condition
    equity = np.array(price) - np.array(debt)
    limit_from_mod_to_retire = np.min(np.vstack((-equity*(1-collateral_constraint),np.zeros(T))),axis=0).tolist()
    limit = [0.0] * (age_at_mod - 26) + limit_from_mod_to_retire + [0.0] * 26
    if equity[T-1] < 0:
        print("Error: still underwater at sale date")
        return
    return equity[T-1], limit

#test output:
#39 is the first period of retirement and is when it all happens
#wealth grant and borrowing limits fall, when income falls by 30%
rebate, limit = hsg_wealth(debt =  hamp_params['initial_debt'], **hamp_params)
rebate_pra, limit_pra = hsg_wealth(debt = hamp_params['initial_debt'] - hamp_params['pra_forgive'], **hamp_params)

df = pd.DataFrame({'index': range(65),'PermGroFac': Params.PermGroFac, 'Limit': limit})
df
df_pra = pd.DataFrame({'index': range(65),'PermGroFac': Params.PermGroFac, 'Limit': limit_pra})

#NNN calculations to see if we are approximating the PRA policy well enough.
age_at_mod = 45
#how many years from until HH is above water?
min(df[df['Limit']<0]['index']) - (age_at_mod-25)
#how many years does PRA bring this forward?
min(df[df['Limit']<0]['index'])  - min(df_pra[df_pra['Limit']<0]['index']) 
#NPV of principal reduction
hamp_params['pra_forgive']/((1+hamp_params['int_rate'])**(65-age_at_mod))

hsg_wealth(debt =  hamp_params['initial_debt'], **hamp_params)


hamp_params['collateral_constraint'] = 0.2
hsg_wealth(debt =  hamp_params['initial_debt'],  **hamp_params)


#updates to cons function
#Update to asset grid for marginal value function. 
#	Before mod: [0, 10]
#Between mod and age 65: [-BoroConsArt_t , -BoroConsArt_t  + 10]
#		Perceived asset grid t+1 at age 64: [-Equity_64 + Rebate_T, -Equity_64 + Rebate_T + 10]
#65 and beyond: [0,10]

#
#house_wealth function takes params above as arguments
#Update in a for loop
#Price_t+1 = price_t * (1+ann_hp_growth) / PermGroFac[t]
#Debt_t+1 = Debt_t / PermGroFac[t]
#Equity_t+1 = (1-down_pay)*price_t+1 - debt_t+1
#Assert Price_T > Debt_T. If not, error out.
#BoroConsArt_t = max(0,Equity_t) if age < 65 and 0 thereafter
#Rebate_T = Price_T - Debt_T
#Returns Rebate_T, BoroCnstArt 