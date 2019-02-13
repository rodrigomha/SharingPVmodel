import numpy as np
import pandas as pd
from IPython.display import clear_output

def obtain_dataid_2017(df):
    "Return all dataid as a nparray of a dataframe"
    return df[df.localhour == '2017-01-01 00:00:00']['dataid'].values.tolist()

def f_gen_per_kw(df):
    "Return gen_per_kw as a matrix on which rows are timeslots and columns are firms"
    dataids = obtain_dataid_2017(df)
    validdata = len(dataids)
    T = len(df[df.dataid==26].gen.values)
    gen_kw = np.empty([T,validdata])
    for idx, val in enumerate(dataids):
        gen_kw[:,idx] = df[df.dataid == val]['gen_per_kW'].values
    return gen_kw
        
def f_load_kw(df):
    "Return gen_per_kw as a matrix on which rows are timeslots and columns are firms"
    dataids = obtain_dataid_2017(df)
    validdata = len(dataids)
    T = len(df[df.dataid==26].gen.values)
    load_kw = np.empty([T,validdata])
    for idx, val in enumerate(dataids):
        load_kw[:,idx] = df[df.dataid == val]['use'].values
    return load_kw

def f_cap_firms(gen_per_m2, load_kw):
    avgPVUsers = np.mean(gen_per_m2, axis=0) #Obtain average generation per m2 of each home
    avgLoadUsers = np.mean(load_kw, axis=0) #Obtain average demand in kW of each home
    a_cap_firms = avgLoadUsers/avgPVUsers  #Obtain cap on investment to not be a producer through the year
    return a_cap_firms

def firms_investment_sharing(sol_sharing, a_max_size_firms, firms): #sol_sharing is what it is returned from the sharing algorithm
    investment_pv = np.zeros(firms)
    investment_pv[sol_sharing] = a_max_size_firms[sol_sharing]
    return investment_pv

def utility_profit_no_investment(load_kw, pi_r):
    T = len(load_kw)
    avg_profit = np.sum(pi_r*load_kw)/T
    print("Average profit per time slot without PV investment for utility is " + str(avg_profit))
    return avg_profit

def utility_profit_standalone(gen_per_m2, load_kw, standalone_inv, pi_r, pi_nm):
    firms = len(standalone_inv)
    profit = 0
    for i in range(firms):
        net_load = load_kw[:,i] - gen_per_m2[:,i]*standalone_inv[i] #Compute net load for investment a[k]
        net_gen = -net_load
        net_load[net_load<=0] = 0 #Set to zero if the net load is negative (gen)
        net_gen[net_gen <=0] = 0 #Set to zero if the net gen is negative (load)
        profit = profit + np.mean(pi_r * net_load) - np.mean(pi_nm * net_gen) #Receive pi_r if net load is positive and pay pi_nm if its negative
    print("Average profit per time slot for utility (with standalone PV investment) is " + str(profit))
    return profit

def utility_profit_sharing(gen_per_m2, load_kw, sharing_inv, pi_r):
    firms = len(sharing_inv)
    profit = 0
    T = len(gen_per_m2)
    for t in range(T):
        InstalledGen = gen_per_m2[t,:]*sharing_inv
        CollectiveGen = np.sum(InstalledGen)
        CollectiveLoad = np.sum(load_kw[t,:])
        if CollectiveLoad > CollectiveGen:
            profit = profit + pi_r * (CollectiveLoad - CollectiveGen)
    avg_profit = profit/T
    print("Average profit per time slot for utility (with sharing PV investment) is " + str(avg_profit))
    return avg_profit
    