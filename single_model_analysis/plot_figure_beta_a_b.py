# plot heatmap a, b of beta distribution vs loss 

import matplotlib
# from tqdm import tqdm
# from experiment_class import DynamicsParameters, TrainingParameters, Experiment
from utilities import load_results
import torch 
import matplotlib.pyplot as plt 
import numpy as np
# import networkx as nx
import warnings
import scipy.stats as ss
# from torchdiffeq import odeint
import pandas as pd 
warnings.filterwarnings('ignore')


size = 100
alpha = 0
scale = 100 
dynamic_weight = False
std_reg = 1.0
equiv_reg = 0.0

#######################
#
# Figure 4
#
####################### 

generate_data = True
if generate_data:
    for name_dynamics in ["Diffusion","MAK" , "MM", "PD", "SIS"]:
        print(name_dynamics)
        folder = f"results/er_experiment_{name_dynamics}_size_{size}_std_reg_{std_reg}"
        A1, training_params1, dynamics_config1, dyn1, func1, loss_list1, x_train1, y_train1, x_test1, y_test1 = load_results(folder)
        ntrain =     len(y_train1)
        ab_list = list(range(1,11))
        ab_loss = []
        ab_loss_err = []
        a_list, b_list = [], []
        
        m2 = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1]))
        # x_test = [m2.sample([size]) for i in range(1000)]
        pred_y = [func1(0, x_train1[i][:,None]) for i in range(len(x_train1))]
        # true_y = [dyn1(0, x_test[i]) for i in range(x_train)]
        loss_ref=   torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train1,1)) .detach().numpy().mean(0)
        # sig_test = []
        for a in ab_list :
            for b in ab_list:
                a_list.append(a)
                b_list.append(b)
                m2 = torch.distributions.Beta(torch.FloatTensor([a]),torch.FloatTensor([b]))
                x_test = [m2.sample([size]) for i in range(1000)]
                pred_y = [func1(0, x_test[i][:,None]) for i in range(1000)]
                true_y = [dyn1(0, x_test[i]) for i in range(1000)]
                loss = torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(true_y,1)).detach().numpy().mean(0)
                # res = ss.ttest_1samp(loss/loss_ref.mean(), 1)
                loss= loss.mean()/loss_ref.mean()
                loss_err = torch.std(torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(true_y,1)).detach()/loss_ref.mean()) 
                loss_err = float(loss_err.detach().numpy())
                # print(a, b ,"$", loss, "\pm" , loss_err , "$", end="\n")
                ab_loss.append(loss)
                ab_loss_err.append(loss_err)
                # sig_test.append(res[1])
        d = pd.DataFrame([ab_loss, ab_loss_err, a_list, b_list]  ).T
        d.columns = ["ab_loss", "ab_loss_err", "a_list", "b_list"]
        d.to_csv(f"results/er_{name_dynamics}_loss_test_beta_dist_a_b_size_{size}_0_alpha_{alpha}_scale_{scale}_dynwght_{dynamic_weight}_std_reg_{std_reg}_equiv_reg_{equiv_reg}.csv")
    

fig,axs = plt.subplots(1,5, figsize=(25,6),sharey=True,  layout='constrained')
i = 0
labels = ["a)","b)","c)","d)","e)"]
for name_dynamics in ["Diffusion", "MAK", "MM", "PD", "SIS"]:
    d = pd.read_csv(f"results/er_{name_dynamics}_loss_test_beta_dist_a_b_size_{size}_0_alpha_{alpha}_scale_{scale}_dynwght_{dynamic_weight}_std_reg_{std_reg}_equiv_reg_{equiv_reg}.csv")
    ax = axs[i]
    x = np.array(d.a_list)
    y = np.array(d.b_list)
    xy = np.stack((x, y))
    n = len(set(x))
    g = np.array(d.ab_loss)[:,None].reshape((n,n))
    print(g.min(), g.max())    

    im1 =ax.pcolormesh(x.reshape((n,n)),y.reshape((n,n)), g, cmap=plt.cm.get_cmap('gist_rainbow'), )#gist_rainbow
    im1.norm =matplotlib.colors.LogNorm (vmin = 0.1, vmax = 11)
    if name_dynamics == "Diffusion":
        ax.set_title("Heat" )
        ax.set_ylabel("$a$")
    else:
        ax.set_title(f"{name_dynamics}")
    ax.set_yticks(range(1,11))
    ax.set_xticks(range(1,11))
    ax.text(-0.3, 1.2, labels[i], transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=40, fontname='Times New Roman')#, fontweight='bold')
    
    if name_dynamics == "MM":
        ax.set_xlabel("$b$")
    if name_dynamics =="SIS":
        cbar = fig.colorbar(im1)
        # cbar.ax.set_title("$\\frac{\\langle\\mathcal{L}(a,b)\\rangle}{\\langle\\mathcal{L}(1,1)\\rangle}$",fontsize= 40, rotation=0)
        # cbar.ax.tick_params(rotation=45)
        cbar.ax.set_title("$\\frac{\\mathcal{L}_{(a,b)}}{\\mathcal{L}_{(1,1)}}$",fontsize= 40, rotation=0)
       
    i+=1
plt.savefig(f"loss_test_beta_dist_a_b_size_{size}_0_alpha_{alpha}_scale_{scale}_dynwght_{dynamic_weight}_std_reg_{std_reg}_equiv_reg_{equiv_reg}.pdf")




# #######################
# #
# # Figure 3
# #
# ####################### 
# size = 50
# fig,axs = plt.subplots(1,5, figsize=(25,5),sharey=True,  layout='constrained')
# i = 0
# a, b = 1,1
# m2 = torch.distributions.Beta(torch.FloatTensor([a]),torch.FloatTensor([b]))
# for name_dynamics in ["Diffusion", "MAK", "MM", "PD", "SIS"]:
#     edgeprob=0.4
#     ax=axs[i]
#     folder = f"results/multiple_nn_2/experiment_{name_dynamics}_size_{size}_0"
#     A1, training_params1, dynamics_config1, dyn1, func1, loss_list1, x_train1, y_train1, x_test1, y_test1 = load_results(folder)
#     gnew = nx.erdos_renyi_graph(size, edgeprob)
#     A = torch.FloatTensor(np.array(nx.adjacency_matrix(gnew).todense()))
#     t = torch.Tensor(np.linspace(0,10,200))
#     x0 = m2.sample([size])
#     dyn = Dynamics(A, model=dynamics_config1.model_name, B=dynamics_config1.B, R=dynamics_config1.R,
#                         H=dynamics_config1.H, F=dynamics_config1.F, a=dynamics_config1.a, b=dynamics_config1.b)
#     y = odeint( dyn, x0, t, method=training_params1.method).squeeze().t()
#     sol = odeint(lambda y, t: func1(y, t, A), x0[:,None], t, method=training_params1.method)[:,:,0,0].detach()
#     i+=1
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.plot(t,y.T, linewidth =5,alpha =0.5)
#     ax.plot(t,sol,linestyle ="dotted", linewidth =5)
#     # ax.set_yticks(np.linspace(0,1,5))
#     # ax.set_xticks(np.round(np.linspace(0,3,6),2),rotation=90)
#     ax.set_ylim(0,1.01)
#     if name_dynamics != "Diffusion":
#         ax.text(.3, 0.9, f'{name_dynamics}', fontsize=25)
#     else:
#         ax.text(.3, 0.9, 'Heat', fontsize=25)
#     if name_dynamics == "MM":
#         ax.set_xlabel("$t$")
# plt.savefig(f"results/x0_beta_{a}_{b}_graph_ER_{edgeprob}.pdf")
# plt.show()

