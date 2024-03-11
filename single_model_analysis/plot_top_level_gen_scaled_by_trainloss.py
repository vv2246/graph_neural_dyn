# top level generalization
# from experiment_class import DynamicsParameters, TrainingParameters, Experiment
from utilities import load_results
import torch
import numpy as np
import networkx as nx
from dynamics import Dynamics 
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnchoredText
size = 100
alpha = 0
scale = 100 
dynamic_weight = False
std_reg = 1.0
equiv_reg = 0.0
list_of_dynamics = ["Diffusion", "PD","MAK", "MM","SIS"]
from mycolorpy import colorlist as mcp
colors = mcp.gen_color(cmap="rainbow",n=5)

generate_results = False
warnings.filterwarnings('ignore')
relative = True
if generate_results:     
    print("Generating data")
    res_p = []
    res = []
    number_of_models = 1
    res_network = []
    for name_dynamics in list_of_dynamics:
        print(name_dynamics)
        p_size_list = [(p, size ) for p in [0.05,0.1] +list(np.linspace(0.2, 1,9)) ]
        for i in range(number_of_models):
            folder = f"results/er_experiment_{name_dynamics}_size_{size}_std_reg_{std_reg}"
            A1, training_params1, dynamics_config1, dyn1, func1, loss_list1, x_train1, y_train1, x_test1, y_test1 = load_results(folder)
        m1 = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1]))
        y_pred1 = torch.stack([func1(0,xi[:,None]) for xi in x_train1]).squeeze()
        train_loss = float(torch.mean(torch.abs(torch.stack(y_train1).squeeze() - y_pred1)).detach())
        print("Subfigure a data")
        for p, size_ in p_size_list:
            loss_list = []
            for niter in range(10):
                connected = False
                while connected == False:
                    gnew = nx.erdos_renyi_graph(size_, p)
                    if nx.is_connected(gnew): 
                        connected = True
                A = torch.FloatTensor(np.array(nx.adjacency_matrix(gnew).todense()))
                x_test = [m1.sample([size_]) for i in range(100)]
                pred_y = [func1(0,x_test[i][:,None], A) for i in range(100)]
                dyn = Dynamics(A, model=dynamics_config1.model_name, B=dynamics_config1.B, R=dynamics_config1.R,
                                    H=dynamics_config1.H, F=dynamics_config1.F, a=dynamics_config1.a, b=dynamics_config1.b)
                true_y = [dyn(0, x_test[i]) for i in range(100)]
                # loss =torch.mean(torch.abs(pred_y - true_y)) # number of nodes x number of samples 
                # loss = loss.detach().numpy()/train_loss
                
                loss_tot = []
                for i in range(len(pred_y)):
                    pred_yi = pred_y[i].squeeze()
                    true_yi = true_y[i].squeeze()
                    li = sum(abs(pred_yi - true_yi))
                    if relative:
                        li = li/sum(abs(true_yi))
                    loss_tot.append(float(li))
                loss_tot_scaled= torch.tensor(loss_tot) * scale
                loss = float(torch.mean(loss_tot_scaled))
                loss_abs = float(torch.mean(loss_tot_scaled))
                
                
                
                
                loss_list.append(loss)
            loss = (np.mean(loss_list))
            loss_err = (np.std(loss_list))
            res_p.append((name_dynamics, size_, p, loss, loss_err ))
            
        # print("Subfigure b data")
        # for delta in torch.linspace(0,2,11):
        #     x_test = [m1.sample([size])+ delta for i in range(1000)]
        #     pred_y = torch.stack([ func1(0, x_test[i][:,None], A1[0]) for i in range(1000)]).squeeze()
        #     true_y = torch.stack([dyn1(0, x_test[i]) for i in range(1000)]).squeeze()
        #     loss = torch.mean(  torch.abs(true_y - pred_y)/train_loss)
        #     loss_err = torch.std(  torch.abs(true_y - pred_y)/train_loss)
        #     res.append((name_dynamics, float(delta), float(loss.detach().numpy()), float(loss_err.detach().numpy())) )
            
        p_size_list = [(0.1, N ) for N in range(50,500,50) ]
        print("Subfigure c data")
        for p, size_ in p_size_list:
            loss_list = []
            for niter in range(10):
                connected = False
                while connected == False:
                    gnew = nx.erdos_renyi_graph(size_, p)
                    if nx.is_connected(gnew): 
                        connected = True
                A = torch.FloatTensor(np.array(nx.adjacency_matrix(gnew).todense()))
                x_test = [m1.sample([size_]) for i in range(100)]
                pred_y = [ func1(0, x_test[i][:,None], A) for i in range(100)]
                dyn = Dynamics(A, model=dynamics_config1.model_name, B=dynamics_config1.B, R=dynamics_config1.R,
                                    H=dynamics_config1.H, F=dynamics_config1.F, a=dynamics_config1.a, b=dynamics_config1.b)
                true_y = [dyn(0, x_test[i]) for i in range(100)]
                # loss = torch.mean(  torch.abs(true_y - pred_y)/train_loss)
                # loss =loss.detach().numpy()
                
                
                loss_tot = []
                for i in range(len(pred_y)):
                    pred_yi = pred_y[i].squeeze()
                    true_yi = true_y[i].squeeze()
                    li = sum(abs(pred_yi - true_yi))
                    if relative:
                        li = li/sum(abs(true_yi))
                    loss_tot.append(float(li))
                loss_tot_scaled= torch.tensor(loss_tot) * scale
                loss = float(torch.mean(loss_tot_scaled))
                loss_abs = float(torch.mean(loss_tot_scaled))
                
                
                loss_list.append(loss)
            loss = (np.mean(loss_list))
            loss_err = (np.std(loss_list))
            res_network.append((name_dynamics, size_, p, loss, loss_err ))
   
    Df1 = pd.DataFrame(res_network, columns = ["dynamics","N","p","loss","loss_err"])
    Df = pd.DataFrame(res, columns = ["dynamics","delta","loss","loss_err"])
    Df1.to_csv(f"results/loss_vs_N_size_{size}_std_reg_{std_reg}_scaled_by_trainloss.csv")
    Df.to_csv(f"results/loss_vs_delta_size_{size}_std_reg_{std_reg}_scaled_by_trainloss.csv")
    Df2 = pd.DataFrame(res_p, columns = ["dynamics","N","p","loss","loss_err"])
    Df2.to_csv(f"results/loss_vs_p_size_{size}_std_reg_{std_reg}_scaled_by_trainloss.csv")
else:
    print("Loading data from files")
    Df1 = pd.read_csv(f"results/loss_vs_N_size_{size}_std_reg_{std_reg}_scaled_by_trainloss.csv")
    Df = pd.read_csv(f"results/loss_vs_delta_size_{size}_std_reg_{std_reg}_scaled_by_trainloss.csv")
    Df2 = pd.read_csv(f"results/loss_vs_p_size_{size}_std_reg_{std_reg}_scaled_by_trainloss.csv")



        

markers = ["o","^","s","D", "X"] 
fig,axs= plt.subplots(figsize= (15,6), ncols= 2,  layout='constrained')
count = 0

### size
ax= axs[1]
for key, df in Df1.groupby("dynamics"):
    if key == "Diffusion":
        key = "Heat"
    ax.plot(df.N, df.loss, marker = markers.pop(), label = key, markersize =15,lw = 1, color = colors[count])
    ax.fill_between(df.N, df.loss - df.loss_err,df.loss + df.loss_err, alpha = 0.2, color = colors[count])
    count +=1
ax.set_ylabel("$\\langle\\eta(\\mathcal{G})\\rangle$",fontsize = 35)
ax.set_xlabel("$|\\mathcal{G}|$",fontsize = 35)
ax.set_xticks(range(50,500,50) )
ax.set_yticks(np.linspace(0,5,6))
ax.tick_params(axis="x", rotation = 90)
markers = ["o","^","s","D", "X"]

axs[0].text(-0.3, 1.2, "a)", transform=axs[0].transAxes, verticalalignment='top', horizontalalignment='left', fontsize=40, fontname='Times New Roman')#, fontweight='bold')
axs[1].text(-0.3, 1.2, "b)", transform=axs[1].transAxes, verticalalignment='top', horizontalalignment='left', fontsize=40, fontname='Times New Roman')#, fontweight='bold')

ax = axs[0]
markers = ["o","^","s","D", "X"] 
count = 0
for key, df in Df2.groupby("dynamics"):
    if key == "Diffusion":
        key = "Heat"
    ax.plot(df.p, df.loss, marker = markers.pop(), label = key, markersize =15,lw = 1, color = colors[count])
    ax.fill_between(df.p, df.loss - df.loss_err,df.loss + df.loss_err, alpha = 0.2, color = colors[count])
    count += 1
ax.set_ylabel("$\\langle\\eta(\\mathcal{G}(p))\\rangle$",fontsize = 35)
ax.set_xlabel("$p$",fontsize = 35)
ax.tick_params(axis="x", rotation = 90)
# ax.set_ylim(0,0.2)
ax.legend(loc =0, fontsize =20)
ax.set_yticks(np.linspace(0,5,6))
ax.set_xticks(np.linspace(0,1,11))
ax.tick_params(axis="x", rotation = 90)
# ax.set_xscale("symlog")
plt.savefig(f"loss_vs_p_vs_N_size_{size}_std_reg_{std_reg}_relpercerr.pdf")


        
