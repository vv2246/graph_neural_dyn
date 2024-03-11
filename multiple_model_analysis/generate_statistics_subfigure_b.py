from utilities import  load_results, compute_d_statistics_one_sample,compute_critical_val,  compute_d_statistics, compute_pval ,set_seeds,acceptance_ratio, get_acc_ratio_sample_vs_null
from dynamics import Dynamics
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
import torch 
from torchdiffeq import odeint
import numpy as np
import os
import pickle

def load_neural_nets(number_networks,name_dynamics , size, alpha , scale,dynamic_weight,std_reg, equiv_reg)  :  
    # 
    nn_list = []
    loss = []
    for i in range(50):
        # folder = f"results/multiple_nn/experiment_{name_dynamics}_size_{size}_{i}"
        folder = f"results/er_experiment_{name_dynamics}_size_{size}_{i}_alpha_{alpha}_scale_{scale}_dynwght_{dynamic_weight}_std_reg_{std_reg}_equiv_reg_{equiv_reg}"
        
        A1, training_params1, dynamics_config1, dyn1, func1, loss_list1, x_train1, y_train1, x_test1, y_test1 = load_results(folder)
        nn_list.append(func1)
        pred_y = [func1(0, x_train1[0][i][:,None], A1[0]) for i in range(len(x_train1[0]))]
        y_train = [y_train1[0][i] for i in range(len(y_train1[0]))] # use this line if you want observed loss
        y_train = [dyn1(0, x_train1[0][i]) for i in range(len(x_train1[0]))] # use this line if you want true loss
        l =torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train,1)).mean()
        loss.append(float(l.detach().numpy()))
    loss = [name_dynamics,np.mean(loss),np.std(loss) ]
    return  nn_list, x_test1 , loss , torch.linspace(0, dynamics_config1.T,int( dynamics_config1.T/dynamics_config1.dt ) ), A1[0], training_params1.train_distr, dyn1
    
        
        
if __name__ == "__main__":
    set_seeds()
    warnings.filterwarnings('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    name_dynamics_list = [ "Diffusion","MAK", "MM", "PD", "SIS"]
    M = 50
    size =10
    number_bootstraps = 10
    delta_plot = 0. # shift
    alpha_sig = 0.05 # significance leven
    # generate_stats= False
    alpha = 0
    scale = 100 
    dynamic_weight = False
    std_reg = 1.0
    equiv_reg = 0.0
    new_network = True    
    network_size = 15
    
    if new_network == True:
        connected = False
        while connected == False:
            gnew = nx.erdos_renyi_graph(network_size,0.3)
            if nx.is_connected(gnew) == True:
                connected = True
        A = torch.FloatTensor(np.array(nx.adjacency_matrix(gnew).todense()))
        
    np.save("er_n_15_p_03.npy",np.array(nx.adjacency_matrix(gnew).todense()))

    #####################
    # iterate over dynamics
    #####################
    # test_significance_functions()
    for itr in range(5):
        print(itr)
        name_dynamics = name_dynamics_list[itr]
        nn_list, x_train, loss_list , time , A_train  , m , dyn_train = load_neural_nets(number_networks = 50,
                                                                                          name_dynamics = name_dynamics,
                                                                                          size= size,
                                                                                          alpha = alpha, 
                                                                                          scale = scale, 
                                                                                          dynamic_weight= dynamic_weight,
                                                                                          std_reg = std_reg, equiv_reg=equiv_reg)  
        
        # generate statistics from training
        d_stat_full_sample = compute_d_statistics(list_of_experiments = nn_list,x_test =  x_train , M = 20,
                                                  direct_fun = True, number_of_draws = 1000, A = A_train)
        
        # prediction and true trajectories
        if itr == 0:
            for i in range(10): 
                x0 = m.sample([network_size]) + delta_plot
            print(x0)
        if new_network ==False:
            A = A_train
            gnew = nx.from_numpy_array(A.detach().numpy())
        # else:
            
        dyn = Dynamics(A, model=dyn_train.model, B=dyn_train.B, R=dyn_train.R,
                            H=dyn_train.H, F=dyn_train.F, a=dyn_train.a, b=dyn_train.b)
        y = odeint( dyn, x0, time, method="dopri5").squeeze().t()
        sol = odeint(lambda y, t: nn_list[0](y, t, A), x0[:,None], time, method="dopri5"  ).squeeze().detach()

    
        # generate statistics from test
        x_pred  = [] 
        for k in range(y.shape[1]):
            x_pred.append(sol[k,:][:,None])
        sigpred = compute_d_statistics(list_of_experiments = nn_list, 
                                        x_test = x_pred, 
                                        M = 20 , direct_fun = True, A = A, number_of_draws = 1000)
        
        with open(f"results/subfigure_b_{name_dynamics}.pkl","wb+") as f:
            pickle.dump([d_stat_full_sample, sigpred, y, sol, time], f)
        