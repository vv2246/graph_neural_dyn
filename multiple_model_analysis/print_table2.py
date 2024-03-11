from utilities import  load_results, compute_d_statistics_one_sample,compute_critical_val,  compute_d_statistics, compute_pval ,set_seeds,acceptance_ratio, get_acc_ratio_sample_vs_null
from dynamics import Dynamics
from generate_statistics_subfigure_b import load_neural_nets
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


if __name__ == "__main__":

    set_seeds()
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
    A_test = np.load('er_n_15_p_03.npy')
    A_test = torch.FloatTensor(A_test)
    name_dynamics_list = ["Diffusion","MAK","MM","PD","SIS"]
    m =torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1]))
    x0_train = m.sample([10])
    x0_test = m.sample([15])
    for itr in range(len(name_dynamics_list)):
        # print(itr)
        name_dynamics = name_dynamics_list[itr]
        nn_list, x_train, loss_list , time , A_train  , m , dyn_config = load_neural_nets(number_networks = 50,
                                                                                          name_dynamics = name_dynamics,
                                                                                          size= size,
                                                                                          alpha = alpha, 
                                                                                          scale = scale, 
                                                                                          dynamic_weight= dynamic_weight,
                                                                                          std_reg = std_reg, equiv_reg=equiv_reg)  
        
    
    
        
        loss = [] 
        dyn_test = Dynamics(A = A_train, model=name_dynamics_list[itr], B=dyn_config.B, R=dyn_config.R, 
                       H=dyn_config.H, F=dyn_config.F, a=dyn_config.a, b=dyn_config.b)
        traj_true = odeint(dyn_test, x0_train, time).squeeze()  
        for fun in nn_list:
            traj_pred = odeint(lambda y, t: fun(y, t, A_train), x0_train[:,None], time, method="dopri5"  ).squeeze().detach()
            l_fun = float(((abs(traj_true - traj_pred).sum())/ (abs(traj_true).sum()))*100)
            loss.append(l_fun )
        print(name_dynamics_list[itr] ,"& $",round( np.mean(loss),2),"\\pm", round(np.std(loss),2), "$", end=" & $ ")
    
    
        loss = [] 
        dyn_test = Dynamics(A = A_test, model=name_dynamics_list[itr], B=dyn_config.B, R=dyn_config.R, 
                       H=dyn_config.H, F=dyn_config.F, a=dyn_config.a, b=dyn_config.b)
        traj_true = odeint(dyn_test, x0_test, time).squeeze()  
        for fun in nn_list:
            traj_pred = odeint(lambda y, t: fun(y, t, A_test), x0_test[:,None], time, method="dopri5"  ).squeeze().detach()
            
            l_fun = float(((abs(traj_true - traj_pred).sum())/ (abs(traj_true).sum()))*100)
            loss.append(l_fun )
        print(round( np.mean(loss),2),"\\pm", round(np.std(loss),2), "$ ")
