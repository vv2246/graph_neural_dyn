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
    folder = f"results/er_experiment_{name_dynamics}_size_{size}_std_reg_{std_reg}"
    scale = 100
    string = ""
    
    if name_dynamics == "Diffusion":
        print("Heat &", end=" ")
        # string += "Heat\\tnote{a} & -- & $B(x_j-x_i)$  & "
    else:
        print(name_dynamics  + " & " , end= " ")
    
    A1, training_params1, dynamics_config1, dyn1, func1, loss_list1, x_train1, y_train1, x_test1, y_test1 = load_results(folder)
    A1 = A1[0]
    
    folder = f"results/er_experiment_{name_dynamics}_size_{size}_std_reg_{std_reg}_beta_5_2"
    A2, training_params2, dynamics_config2, dyn2, func2, loss_list2, x_train2, y_train2, x_test2, y_test2 = load_results(folder)
    A2 = A2[0]
    
    folder = f"results/er_experiment_{name_dynamics}_size_{size}_std_reg_{std_reg}_p_06"
    A3, training_params3, dynamics_config3, dyn3, func3, loss_list3, x_train3, y_train3, x_test3, y_test3 = load_results(folder)
    A3 = A3[0]
    
    
    #### out of sample  1

    
    #B(5,2) model
    # x_test = [m2.sample([size]) for i in range(1000)]
    pred_y = [func2(0, x_train2[i][:,None]) for i in range(len(x_train2))]
    # true_y = [dyn2(0, x_test[i]) for i in range(1000)]
    loss = torch.mean(scale * torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train2,1))) 
    loss_err = torch.std(scale * torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train2,1))) 
    loss = round(float(loss.detach().numpy()),round_val)
    loss_err = round(float(loss_err.detach().numpy()) ,round_val)
    print("$", loss, "\pm" , loss_err , "$", end="&")
    string += " $ " + str( loss) +  " \pm " +str( loss_err )+ "$ &" 
    
    
    
    x_test = [m2.sample([size]) for i in range(1000)]
    pred_y = [func2(0, x_test[i][:,None]) for i in range(len(x_test))]
    true_y = [dyn2(0, x_test[i]) for i in range(1000)]
    loss = torch.mean(scale * torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(true_y,1))) 
    loss_err = torch.std(scale * torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(true_y,1))) 
    loss = round(float(loss.detach().numpy()),round_val)
    loss_err = round(float(loss_err.detach().numpy()) ,round_val)
    print("$", loss, "\pm" , loss_err , "$", end="&")
    string += " $ " + str( loss) +  " \pm " +str( loss_err )+ "$ &" 
    
    #out of sample 3
    
    #p=0.6 model 
    # x_test = [m1.sample([size]) for i in range(1000)]
    pred_y = [func3(0, x_train3[i][:,None]) for i in range(len(x_train3))]
    # true_y = [dyn3(0, x_test[i]) for i in range(1000)]
    loss = torch.mean(scale * torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train3,1))) 
    loss_err = torch.std(scale * torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train3,1))) 
    loss = round(float(loss.detach().numpy()),round_val)
    loss_err = round(float(loss_err.detach().numpy()) ,round_val)
    print("$", loss, "\pm" , loss_err , "$", end="\\ \\ ")
    string += " $ " + str( loss) +  " \pm " +str( loss_err )+ "$ "
    
    
    x_test = [m1.sample([size]) for i in range(1000)]
    pred_y = [func3(0, x_test[i][:,None]) for i in range(len(x_test))]
    true_y = [dyn3(0, x_test[i]) for i in range(1000)]
    loss = torch.mean(scale * torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(true_y,1))) 
    loss_err = torch.std(scale * torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(true_y,1))) 
    loss = round(float(loss.detach().numpy()),round_val)
    loss_err = round(float(loss_err.detach().numpy()) ,round_val)
    print("$", loss, "\pm" , loss_err , "$", end="&")
    string += " $ " + str( loss) +  " \pm " +str( loss_err )+ "$ \\\\" 
    
    
    
    
    
    