# from experiment_class import DynamicsParameters, TrainingParameters, Experiment
from utilities import load_results
import numpy as np 
import networkx as nx
from dynamics import Dynamics 
import torch
import warnings
import random

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
m1 = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1]))
a,b = 5,2
m2 = torch.distributions.Beta(torch.FloatTensor([a]),torch.FloatTensor([b]))
size = 100
alpha = 0
scale = 100
std_reg = 1.0
equiv_reg = 0.0
round_val= 2
relative = False
# cosine = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

A_new1 =torch.FloatTensor( np.load("er_n_100_p_01.npy"))
A_new2 =torch.FloatTensor( np.load("er_n_100_p_06.npy"))


for name_dynamics in ["Diffusion", "MAK", "MM","PD", "SIS"]:
    folder = f"results/er_experiment_{name_dynamics}_size_{size}_std_reg_{std_reg}"
    string = ""
    
    if name_dynamics == "Diffusion":
        print("Heat& -- & $B(x_j-x_i)$  &", end=" ")
        string += "Heat\\tnote{a} & -- & $B(x_j-x_i)$  & "
    
    if name_dynamics == "MAK":
        print("MAK & $F-Bx_i^b$ & $Rx_j$ &", end=" ")
        string += "MAK & $F-Bx_i^b$ & $Rx_j$  & "
        
    if name_dynamics == "PD":
        print("PD& $-Bx_i^b$ & $Rx_j^a$ &", end=" ")
        string += "PD & $-Bx_i^b$ & $Rx_j^a$ & "
        
    if name_dynamics == "MM":
        print("MM &  $-Bx_i$& $R\\frac{x_j^h}{1+x_{j}^h}$ &", end=" ")
        string += "MM &  $-Bx_i$& $R\\frac{x_j^h}{1+x_{j}^h}$ & "
        
    if name_dynamics == "SIS":
        print("SIS &$-Bx_i$ &$R(1-x_i)x_j$   &", end=" ")
        string += "SIS &$-Bx_i$ &$R(1-x_i)x_j$   & "
    
    
    A1, training_params1, dynamics_config1, dyn1, func1, loss_list1, x_train1, y_train1, x_test1, y_test1 = load_results(folder)
    y_train1 = y_train1
    ntrain =     len(y_train1)
    x_train1 = x_train1
    
    loss_tot = []
    pred_y = [func1(0, x_train1[i][:,None])  for i in range(ntrain)]
    for i in range(len(y_train1)):
        pred_yi = pred_y[i].squeeze()
        true_yi = y_train1[i].squeeze()
        li = sum(abs(pred_yi - true_yi))
        if relative:
            li = li/sum(abs(true_yi))
        loss_tot.append(float(li))
    loss_tot_scaled= torch.tensor(loss_tot) * scale
    loss = torch.mean(loss_tot_scaled)
    loss_err = loss_tot_scaled.std()
    loss = round(float(loss.detach().numpy()),round_val) 
    loss_err = round(float(loss_err.detach().numpy()) ,round_val)
    print("$", loss, "\pm" , loss_err , "$", end="&")
    string +=  " $ " + str( loss)  + " \pm "  + str(loss_err ) + " $ & "
    
    #### out of sample 
    x_test = [m1.sample([size]) for i in range(1000)]
    pred_y = [func1(0, x_test[i][:,None]) for i in range(1000)]
    true_y = [dyn1(0, x_test[i]) for i in range(1000)]
    loss_tot = []
    for i in range(len(pred_y)):
        pred_yi = pred_y[i].squeeze()
        true_yi = true_y[i].squeeze()
        li = sum(abs(pred_yi - true_yi))
        if relative:
            li = li/sum(abs(true_yi))
        loss_tot.append(float(li))
    loss_tot_scaled= torch.tensor(loss_tot) * scale
    loss = torch.mean(loss_tot_scaled)
    loss_err = loss_tot_scaled.std()
    loss = round(float(loss.detach().numpy()),round_val)
    loss_err = round(float(loss_err.detach().numpy()) ,round_val)
    print("$", loss, "\pm" , loss_err , "$", end="&")
    string += " $ " + str( loss) +  " \pm " +str( loss_err )+ "$ &" 
    
    
    #### out of sample 2
    x_test = [m2.sample([size]) for i in range(1000)]
    pred_y = [func1(0, x_test[i][:,None]) for i in range(1000)]
    true_y = [dyn1(0, x_test[i]) for i in range(1000)]
    loss_tot = []
    for i in range(len(pred_y)):
        pred_yi = pred_y[i].squeeze()
        true_yi = true_y[i].squeeze()
        li = sum(abs(pred_yi - true_yi))
        if relative:
            li = li/sum(abs(true_yi))
        loss_tot.append(float(li))
    loss_tot_scaled= torch.tensor(loss_tot) * scale
    loss = torch.mean(loss_tot_scaled)
    loss_err = loss_tot_scaled.std()
    loss = round(float(loss.detach().numpy()),round_val)
    loss_err = round(float(loss_err.detach().numpy()) ,round_val)
    print("$", loss, "\pm" , loss_err , "$", end="&")
    string += " $ " +str( loss)+ " \pm " +str( loss_err ) + "$ &"
    
    
    #### network 1, 2
    for A_test in [A_new1,A_new2]:
        x_test = [m1.sample([size]) for i in range(1000)]
        dyn = Dynamics(A_test, model=dynamics_config1.model_name, B=dynamics_config1.B, R=dynamics_config1.R,
                            H=dynamics_config1.H, F=dynamics_config1.F, a=dynamics_config1.a, b=dynamics_config1.b)
        pred_y = [func1(0, x_test[i][:,None],A_test) for i in range(1000)]
        true_y =[dyn(0, x_test[i]) for i in range(1000)]
        loss_tot = []
        for i in range(len(pred_y)):
            pred_yi = pred_y[i].squeeze()
            true_yi = true_y[i].squeeze()
            li = sum(abs(pred_yi - true_yi))
            if relative:
                li = li/sum(abs(true_yi))
            loss_tot.append(float(li))
        loss_tot_scaled= torch.tensor(loss_tot) * scale
        loss = torch.mean(loss_tot_scaled)
        loss_err = loss_tot_scaled.std()
        loss = round(float(loss),round_val)
        loss_err = round(float(loss),round_val)
        print("$", loss, "\pm" , loss_err , "$", end = " & ")
        string += " $ " +str( loss)+ " \pm " +str( loss_err ) + "$ & "
    print("\n")
    with open(f"results/table1_var_reg_relative_{relative}.txt" , mode = "a+") as f:
        f.write(string + "\n")
    
