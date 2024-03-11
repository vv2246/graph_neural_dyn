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
    

def generate_statistics(A_test , delta_test, true_dyn, time_array,  d_stat_full_sample,
                        neural_net_list, M , alpha_sig = 0.05, number_of_iterations =10, a = 1,b = 1 ):
    res = []
    m_test =  torch.distributions.Beta(torch.FloatTensor([a]),torch.FloatTensor([b]))
    for niter in range(number_of_iterations) :
        x0 = m_test.sample([A_test.shape[0]]) + delta_test
        y = odeint( true_dyn, x0, time_array, method= 'dopri5' ).squeeze().t()
        # sol = pooled_integral(neural_net_list, x0[:,None], time_array, A_test)
        idx = np.random.randint(M)
        sol = odeint(lambda y, t: neural_net_list[idx](y, t, A_test), x0[:,None], time_array, method="dopri5").squeeze().detach()
        loss = [] 
        x_pred  = [] 
        for k in range(y.shape[1]):
            x_pred.append(sol[k,:][:,None]) 
            # loss.append(torch.stack( [ abs(dyn(0, y[:,k][:,None]).squeeze() - neural_net_list[idx](0, y[:,k][:,None,None], A_test ).squeeze()) ])#for func in neural_net_list ]) )
            loss.append(abs(dyn(0, y[:,k][:,None]).squeeze() - neural_net_list[idx](0, y[:,k][:,None,None], A_test ).squeeze()))#for func in neural_net_list ]) )
        
        loss = float(torch.stack(loss).mean().detach().numpy())
        sigpred = compute_d_statistics(list_of_experiments = neural_net_list, 
                                       x_test = x_pred, M = 20 , direct_fun = True, 
                                       A = A_test, number_of_draws = 100)
        accepted = get_acc_ratio_sample_vs_null(null_samples = d_stat_full_sample, 
                                       testing_samples = sigpred, alpha = alpha_sig )
        res.append([true_dyn.model, a, loss, accepted])
    return res 
        
        
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
    new_network = False    
    network_size = 10
    
    #####################
    # generate a new network if used
    #####################
    if new_network == True:
        connected = False
        while connected == False:
            gnew = nx.erdos_renyi_graph(network_size,0.3)
            if nx.is_connected(gnew) == True:
                connected = True
        A = torch.FloatTensor(np.array(nx.adjacency_matrix(gnew).todense()))

    #####################
    # generate stats
    #####################
    for i in range(len(name_dynamics_list)):
        print(i)
        fname = f"results/beta_ab_vs_pval_vs_loss_network_size_{network_size}_new_network_{new_network}_alpha_{alpha_sig}.csv"
        fname1 = f"results/training_loss_network_size_{size}.csv"
        nn_list, x_train, loss_list , time , A_train  , m , dyn_config = load_neural_nets(number_networks = 50,
                                                                                          name_dynamics = name_dynamics_list[i],
                                                                                          size = size,
                                                                                          alpha = alpha, scale = scale , 
                                                                                          dynamic_weight=dynamic_weight ,
                                                                                          std_reg=std_reg, equiv_reg=equiv_reg )  
        # generate statistics from training
        d_stat_full_sample = compute_d_statistics(list_of_experiments = nn_list, 
                                                   x_test = x_train , M = M, 
                                                  direct_fun = True, number_of_draws = 100, A = A_train)
        if new_network == False:
            A = A_train 
        dyn = Dynamics(A, model=name_dynamics_list[i], B=dyn_config.B, R=dyn_config.R, 
                       H=dyn_config.H, F=dyn_config.F, a=dyn_config.a, b=dyn_config.b)

        res_all = []
        for a in [1,2,3,4,5]:
            # print(delta)
            res =  generate_statistics(A_test = A, delta_test = 0, true_dyn = dyn,
                                       d_stat_full_sample = d_stat_full_sample ,
                                       time_array = time ,  neural_net_list = nn_list , M= M ,  
                                       alpha_sig= 0.05, number_of_iterations =10 ,a =a, b= a )
            res_all.append(res)
            
            df = pd.DataFrame(res, columns = ["Dynamics", "delta", "loss", "pval"])
            if os.path.isfile(fname) == True:
                df.to_csv(fname, header = False, mode = "a+")
            else:
                df.to_csv(fname, header = True)
                
