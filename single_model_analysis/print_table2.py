# from experiment_class import DynamicsParameters, TrainingParameters, Experiment
from utilities import load_results
import numpy as np 
import networkx as nx
from dynamics import Dynamics 
import torch
import warnings


m1 = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1]))
a,b = 5,2
m2 = torch.distributions.Beta(torch.FloatTensor([a]),torch.FloatTensor([b]))
size = 100
alpha = 0
scale = 100 
dynamic_weight = False
std_reg = 1.0
equiv_reg = 0.0
warnings.filterwarnings('ignore')
round_val= 2
for name_dynamics in ["Diffusion", "MAK", "MM","PD", "SIS"]:
# name_dynamics = ""
    folder = f"results/er_experiment_{name_dynamics}_size_{size}_0_alpha_{alpha}_scale_{scale}_dynwght_{dynamic_weight}_std_reg_{std_reg}_equiv_reg_{equiv_reg}"
    scale = 100
    string = ""
    
    if name_dynamics == "Diffusion":
        print("Heat &", end=" ")
        # string += "Heat\\tnote{a} & -- & $B(x_j-x_i)$  & "
    else:
        print(name_dynamics  + " & " , end= " ")
    print("$\\mathcal{G}\\equiv \\mathcal{H}$ &", end ="")
    
    A1, training_params1, dynamics_config1, dyn1, func1, loss_list1, x_train1, y_train1, x_test1, y_test1 = load_results(folder)
    y_train1 = y_train1[0]
    ntrain =     len(y_train1)
    x_train1 = x_train1[0]
    A1 = A1[0]
    
    dyn_self  = Dynamics(A1, model=dynamics_config1.model_name, B=dynamics_config1.B, R=dynamics_config1.R,
                        H=dynamics_config1.H, F=dynamics_config1.F, a=dynamics_config1.a, b=dynamics_config1.b,  self_interaction = True, nbr_interaction = False)
    dyn_nbr = Dynamics(A1, model=dynamics_config1.model_name, B=dynamics_config1.B, R=dynamics_config1.R,
                        H=dynamics_config1.H, F=dynamics_config1.F, a=dynamics_config1.a, b=dynamics_config1.b,  self_interaction = False, nbr_interaction = True)

    #### in sample
    pred_y_self = [func1(0, x_train1[i][:,None], nbr_interaction = False)  for i in range(ntrain)]
    pred_y_self = torch.squeeze(torch.hstack(pred_y_self))
    pred_y_nbr = [func1(0, x_train1[i][:,None], self_interaction = False)  for i in range(ntrain)]
    pred_y_nbr = torch.squeeze(torch.hstack(pred_y_nbr))
    
    true_y_nbr = [dyn_nbr(0, x_train1[i])  for i in range(ntrain)]
    true_y_nbr = torch.squeeze(torch.hstack(true_y_nbr))
    true_y_self = [dyn_self(0, x_train1[i])  for i in range(ntrain)]
    true_y_self = torch.squeeze(torch.hstack(true_y_self))
    
    loss_self = abs(true_y_self - pred_y_self)
    loss_nbr = abs(true_y_nbr - pred_y_nbr)
    
    loss_self_mean = scale*  loss_self.mean() #torch.mean(scale *  torch.abs(torch.squeeze(torch.cat(pred_y_self,1),2)-torch.cat(true_y_self,1))) 
    loss_self_err = scale*  loss_self.mean(0).std() #torch.std(scale *  torch.abs(torch.squeeze(torch.cat(pred_y_self,1),2)-torch.cat(true_y_self,1))) 
    loss_self_mean = round(float(loss_self_mean.detach().numpy()),round_val) 
    loss_self_err = round(float(loss_self_err.detach().numpy()) ,round_val)
    print("$", loss_self_mean, "\pm" , loss_self_err , "$", end="&")
    
    loss_nbr_mean = scale* loss_nbr.mean() #torch.mean(scale *  torch.abs(torch.squeeze(torch.cat(pred_y_nbr,1),2)-torch.cat(true_y_nbr,1))) 
    loss_nbr_err =   scale* loss_nbr.mean(0).std() #torch.std(scale *  torch.abs(torch.squeeze(torch.cat(pred_y_nbr,1),2)-torch.cat(true_y_nbr,1))) 
    loss_nbr_mean = round(float(loss_nbr_mean.detach().numpy()),round_val) 
    loss_nbr_err = round(float(loss_nbr_err.detach().numpy()) ,round_val)
    print("$", loss_nbr_mean, "\pm" , loss_nbr_err , "$", end=" \\\\ \n")
    # 
    # print("\n")
    
    
    #### in sample
    p_size_list = [(0.1, size ) ,(0.6, size ) ]
    for p, size_ in p_size_list:
        gnew = nx.erdos_renyi_graph(size_,p)
        A = torch.FloatTensor(np.array(nx.adjacency_matrix(gnew).todense()))
        x_test = [m1.sample([size]) for i in range(1000)]
        pred_y_self = torch.squeeze(torch.hstack([func1(0, x_test[i][:,None],A, nbr_interaction = False) for i in range(1000)]))
        pred_y_nbr = torch.squeeze(torch.hstack([func1(0, x_test[i][:,None],A, self_interaction = False) for i in range(1000)]))
        
        dyn_self  = Dynamics(A, model=dynamics_config1.model_name, B=dynamics_config1.B, R=dynamics_config1.R,
                            H=dynamics_config1.H, F=dynamics_config1.F, a=dynamics_config1.a, b=dynamics_config1.b,  self_interaction = True, nbr_interaction = False)
        dyn_nbr = Dynamics(A, model=dynamics_config1.model_name, B=dynamics_config1.B, R=dynamics_config1.R,
                            H=dynamics_config1.H, F=dynamics_config1.F, a=dynamics_config1.a, b=dynamics_config1.b,  self_interaction = False, nbr_interaction = True)

        true_y_nbr = torch.squeeze(torch.hstack([dyn_nbr(0, x_test[i])  for i in range(1000)]))
        true_y_self =torch.squeeze(torch.hstack( [dyn_self(0, x_test[i])  for i in range(1000)]))
        
        
        loss_self = abs(true_y_self - pred_y_self) #torch.mean( scale * torch.abs(true_y_self - pred_y_self )) .detach().numpy()
        loss_nbr =  abs(true_y_nbr - pred_y_nbr) #torch.mean( scale * torch.abs(true_y_self - pred_y_self )) .detach().numpy()
        
            
        loss_self_mean =  scale* loss_self.mean() #torch.mean(scale *  torch.abs(torch.squeeze(torch.cat(pred_y_self,1),2)-torch.cat(true_y_self,1))) 
        loss_self_err =  scale* loss_self.mean(0).std() #torch.std(scale *  torch.abs(torch.squeeze(torch.cat(pred_y_self,1),2)-torch.cat(true_y_self,1))) 
        loss_self_mean = round(float(loss_self_mean.detach().numpy()),round_val) 
        loss_self_err = round(float(loss_self_err.detach().numpy()) ,round_val)
        if p == 0.1:
            print("&$\mathcal{H}\sim \mathcal{P}(\mathfrak{G})$ &", end = "")
        else:
            print("&$\mathcal{H}\sim \mathcal{P}(\mathfrak{H})$ &", end=" ")
        print("$", loss_self_mean, "\pm" , loss_self_err , "$", end=" & ")
        
        loss_nbr_mean =scale*  loss_nbr.mean() #torch.mean(scale *  torch.abs(torch.squeeze(torch.cat(pred_y_nbr,1),2)-torch.cat(true_y_nbr,1))) 
        loss_nbr_err =   scale* loss_nbr.mean(0).std() #torch.std(scale *  torch.abs(torch.squeeze(torch.cat(pred_y_nbr,1),2)-torch.cat(true_y_nbr,1))) 
        loss_nbr_mean = round(float(loss_nbr_mean.detach().numpy()),round_val) 
        loss_nbr_err = round(float(loss_nbr_err.detach().numpy()) ,round_val)
        print("$", loss_nbr_mean, "\pm" , loss_nbr_err , "$", end=" \\\\ \n")
        # print("\n")
    # print("\\\\\n")