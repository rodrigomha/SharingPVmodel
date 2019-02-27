import numpy as np
import pandas as pd
import cvxpy as cvx
from IPython.display import clear_output


def solve_standalone(gen_per_m2, load_kw, valid_index, a_max_firms, pi_s, pi_r, pi_nm):
    num_points = 500
    firms = len(a_max_firms)
    J_firms = np.zeros([num_points,firms])
    standalone_pv_m2 = np.zeros(firms)
    for i in range(firms):
        clear_output()
        print('Solving firm '+ str(i+1)+ ' (of '+ str(firms) +') with dataid ' + str(valid_index[i]))
        a = np.linspace(0,a_max_firms[i], num_points) #Create a vector between 0 and a_max.
        for k in range(len(a)):
            net_load = load_kw[:,i] - gen_per_m2[:,i]*a[k] #Compute net load for investment a[k]
            net_gen = -net_load
            net_load[net_load<=0] = 0 #Set to zero if the net load is negative (gen)
            net_gen[net_gen <=0] = 0 #Set to zero if the net gen is negative (load)
            J = pi_s*a[k] + np.mean(pi_r * net_load) - np.mean(pi_nm * net_gen)
            J_firms[k,i] = J
        standalone_pv_m2[i] = a[np.argmin(J_firms[:,i])] #Save investment decision
    return standalone_pv_m2

def solve_sharing_collective(gen_per_m2, load_kw, valid_index, a_max_size_firms, pi_s, pi_r):
    #Data initialization
    T = len(gen_per_m2)
    avgPVFirms = np.mean(gen_per_m2, axis=0) #Obtain average generation per m2 per firm
    firms = len(a_max_size_firms)
    MaxGenFirms = avgPVFirms*a_max_size_firms #Obtain Max possible generation per firm
    idx_SortFirms = np.argsort(-MaxGenFirms) #Sort firms from highest maximum possible generation to lowest
    dataid_SortFirms = valid_index[idx_SortFirms] #Obtain dataid of firms, sorted from highest to lowest max possible generation
    mid_firm = int(np.floor(firms/2)) #Divide by two the total number of firms, necessary to initialize the algorithm.
    
    #Initialize sets
    SetInvest = idx_SortFirms[0:mid_firm] #Set S that invest on PV. Initialized by picking the highest firms (half of them). Contain the index position used in MaxGenFirms or a_max_size_firms (and not dataid)
    SetNonInvest = idx_SortFirms[mid_firm:] #Set T initialized on PV. Complement of S. Contain the index used in MaxGenFirms or a_max_size_firms (and not dataid)
    SetInvest = SetInvest.tolist() #Convert array to list
    SetNonInvest = SetNonInvest.tolist() #Convert array to list
    
    Condition = True #Condition to stop the algorithm
    t = 1 #Iteration counter
    threshold = 5 #Threshold of how many repeated changes I admit
    counter = 0 #Counter of how many times the same change has been done

    #Initial values to start when the algorithm should stop
    dataid_S_old_1 = -2
    dataid_T_old_1 = -3
    dataid_removed_S = -4
    dataid_removed_T = -5
    
    while Condition:
        print('\n')
        print('Iteration: ' + str(t))
        ## Compute Statistics of Collective PV and Load
        gen_max = gen_per_m2[:,SetInvest]*a_max_size_firms[SetInvest] #Users on S invest its max capacity
        collective_gen = np.sum(gen_max,axis=1) #collective gen in kW per time step
        collective_load = np.sum(load_kw,axis=1) #collective load in kW per time step
        net_load_pos = collective_load >= collective_gen #Timesteps when there is positive netload
        probdeficit = np.mean(net_load_pos) #Probability of deficit
        theta = pi_s/(probdeficit*pi_r) #threshold for users
        

        ## Compute Statistics of users regarding their merit site
        net_load_pos_vec = np.reshape(net_load_pos, (T,1)) #reshape vector of when net_load is positive
        W_pos = net_load_pos_vec*gen_per_m2 #Create vector of generation only when demand is positive
        expected_W_pos = np.sum(W_pos,axis=0)/np.sum(net_load_pos_vec) #Compute expected generation when netload is pos.
        merit_site_S = expected_W_pos[SetInvest] #Compute the merit of households on S
        merit_site_T = expected_W_pos[SetNonInvest] #Compute the merit of households on T
        firms_remove_from_S = merit_site_S < theta #If the merit site on S is below theta, they can be removed from S
        firms_remove_from_T = merit_site_T > theta #If the merit on site T is above theta, they can be removed from T

        aux_remove_S = -1 #Initialization of auxiliar variable to remove firm from S
        aux_remove_T = -1 #Initialization of auxiliar variable to remove firm from T


        if sum(firms_remove_from_S) > 0: #If there are firms to remove from S
            merit_site_S[~firms_remove_from_S]= float('Inf') #Set merit of non-removable firms to infinity
            aux_remove_S = np.argmin(merit_site_S) #Remove firm with the worst merit (to add to T)
            dataid_removed_S = SetInvest[aux_remove_S] #Save the index and dataid of the removed firm
            print('firm removed from S: '+ str(dataid_removed_S) + ' with dataid '+ str(valid_index[dataid_removed_S]))

        if sum(firms_remove_from_T) > 0: #If there are firms to remove from T
            merit_site_T[~firms_remove_from_T]= -float('Inf') #Set merit of non-removable firms to -infinity
            aux_remove_T = np.argmax(merit_site_T) #Remove firm with the best merit (to add to S)
            dataid_removed_T = SetNonInvest[aux_remove_T] #Save the index and dataid of the removed firm
            print('firm removed from T: '+ str(dataid_removed_T) + ' with dataid '+ str(valid_index[dataid_removed_T]))

        if aux_remove_S != -1: #If we have something to remove from S
            SetInvest.pop(aux_remove_S) #Remove the worst firm from set S
            SetNonInvest.append(dataid_removed_S) #Add worst firm removed from S to set T

        if aux_remove_T != -1: #If we have something to remove from T
            SetNonInvest.pop(aux_remove_T) #Remove the best firm from set T
            SetInvest.append(dataid_removed_T) #Add the best firm removed from T to set S
        if t==1:
            print('Set of firms that invest:')
            print(SetInvest)
            print('Set of firms that do not invest:')
            print(SetNonInvest)
        t = t+1
        if aux_remove_S + aux_remove_T == -2: #If no need to remove:
            Condition = False #End algorithm
        if dataid_T_old_1 == dataid_removed_S | dataid_S_old_1 == dataid_removed_T: #There is a repetition of removal and addition:
            counter = counter + 1
        if counter >= threshold: #If there is more than threshold repetition
            Condition = False #End algorithm
        dataid_T_old_1 = dataid_removed_T
        dataid_S_old_1 = dataid_removed_S
        print('Number of times doing the same change: '+ str(counter))
    print('\nTotal investment of PV in sharing case is ' + str(sum(a_max_size_firms[SetInvest])) + ' in m2')
    return SetInvest #Return which firms invest its maximum
    
    
    

def solve_wholesale_aggregator(gen_per_m2, load_kw, a_max_firms, pi_s, pi_g):
    firms = len(a_max_firms)
    a_inv = cvx.Variable(firms)
    T = len(gen_per_m2[:,0])
    load_cost_matrix = (load_kw.T * pi_g).T
    gen_profit_matrix = (gen_per_m2.T * pi_g).T
    obj = pi_s*cvx.sum(a_inv) + np.sum(load_cost_matrix)/T - cvx.sum(a_inv*np.sum(gen_profit_matrix,axis=0))/T
    constraints = [a_inv >= 0, a_inv <= a_max_firms]
    prob = cvx.Problem(cvx.Minimize(obj), constraints)
    prob.solve(solver=cvx.GUROBI, verbose=True)
    print("optimal value with GUROBI:", prob.value)
    return a_inv.value


def solve_wholesale_aggregator_demandcharge(gen_per_m2, load_kw, a_max_firms, pi_s, pi_g, demand_charge):
    firms = len(a_max_firms)
    a_inv = cvx.Variable(firms)
    T = len(gen_per_m2[:,0])
    
    #obtain timeseries for each month
    load_jan = load_kw[0:24*31, :]
    gen_jan = gen_per_m2[0:24*31, :]
    load_feb =  load_kw[24*31:24*59, :]
    gen_feb = gen_per_m2[24*31:24*59, :]
    load_mar = load_kw[24*59:24*90, :]
    gen_mar = gen_per_m2[24*59:24*90, :]
    load_apr = load_kw[24*90:24*120 , :]
    gen_apr = gen_per_m2[24*90:24*120 , :]
    load_may = load_kw[24*120:24*151 , :]
    gen_may = gen_per_m2[24*120:24*151 , :]
    load_jun = load_kw[24*151:24*181 , :]
    gen_jun = gen_per_m2[24*151:24*181 , :]
    load_jul = load_kw[24*181:24*212 , :]
    gen_jul = gen_per_m2[24*181:24*212 , :]
    load_aug = load_kw[24*212:24*243 , :]
    gen_aug = gen_per_m2[24*212:24*243 , :]
    load_sep = load_kw[24*243:24*273 , :]
    gen_sep = gen_per_m2[24*243:24*273 , :]
    load_oct = load_kw[24*273:24*304 , :]
    gen_oct = gen_per_m2[24*273:24*304 , :]
    load_nov = load_kw[24*304:24*334 , :]
    gen_nov = gen_per_m2[24*304:24*334 , :]
    load_dec = load_kw[24*334: , :]
    gen_dec = gen_per_m2[24*334: , :]
    
    d_charge = demand_charge/12  #average monthly payment 
    load_cost_matrix = (load_kw.T * pi_g).T #define a matrix of size (8754, 1000) that contains the cost of purchasing electricity
    gen_profit_matrix = (gen_per_m2.T * pi_g).T #define a matrix of size (8754, 1000) that contains the cost of selling electricity per m2
    obj = pi_s*cvx.sum(a_inv) + np.sum(load_cost_matrix)/T - cvx.sum(a_inv*np.sum(gen_profit_matrix,axis=0))/T
    obj = obj + d_charge* cvx.norm(np.sum(load_jan,axis=1) - gen_jan*a_inv,'inf') + d_charge* cvx.norm(np.sum(load_feb,axis=1) - gen_feb*a_inv,'inf') 
    obj = obj + d_charge* cvx.norm(np.sum(load_mar,axis=1) - gen_mar*a_inv,'inf') + d_charge* cvx.norm(np.sum(load_apr,axis=1) - gen_apr*a_inv,'inf') 
    obj = obj + d_charge* cvx.norm(np.sum(load_may,axis=1) - gen_may*a_inv,'inf') + d_charge* cvx.norm(np.sum(load_jun,axis=1) - gen_jun*a_inv,'inf') 
    obj = obj + d_charge* cvx.norm(np.sum(load_jul,axis=1) - gen_jul*a_inv,'inf') + d_charge* cvx.norm(np.sum(load_aug,axis=1) - gen_aug*a_inv,'inf') 
    obj = obj + d_charge* cvx.norm(np.sum(load_sep,axis=1) - gen_sep*a_inv,'inf') + d_charge* cvx.norm(np.sum(load_oct,axis=1) - gen_oct*a_inv,'inf') 
    obj = obj + d_charge* cvx.norm(np.sum(load_nov,axis=1) - gen_nov*a_inv,'inf') + d_charge* cvx.norm(np.sum(load_dec,axis=1) - gen_dec*a_inv,'inf') 
    constraints = [a_inv >= 0, a_inv <= a_max_firms]
    prob = cvx.Problem(cvx.Minimize(obj), constraints)
    prob.solve(solver=cvx.GUROBI, verbose=True)
    print("optimal value with GUROBI:", prob.value)
    return a_inv.value





