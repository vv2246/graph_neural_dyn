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
relative = True
generate_data = False
scale = 100
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
        pred_y = [func1(0, x_train1[i][:,None]) for i in range(len(x_train1))]
        loss_tot = []
        for i in range(len(pred_y)):
            pred_yi = pred_y[i].squeeze()
            true_yi = y_train1[i].squeeze()
            li = sum(abs(pred_yi - true_yi))
            if relative:
                li = li/sum(abs(true_yi))
            loss_tot.append(float(li))
        
        loss_ref= float(torch.mean(torch.tensor(loss_tot) * scale))
        for a in ab_list :
            for b in ab_list:
                a_list.append(a)
                b_list.append(b)
                m2 = torch.distributions.Beta(torch.FloatTensor([a]),torch.FloatTensor([b]))
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
                loss = float(torch.mean(loss_tot_scaled))
                loss_abs = float(torch.mean(loss_tot_scaled))
                
                loss = loss / loss_ref
                # print(a, b ,"$", loss, "\pm" , loss_err , "$", end="\n")
                ab_loss.append(loss)
                ab_loss_err.append(loss_abs)
        d = pd.DataFrame([ab_loss, ab_loss_err, a_list, b_list]  ).T
        d.columns = ["ab_loss_rel", "ab_loss_abs", "a_list", "b_list"]
        d.to_csv(f"results/er_{name_dynamics}_loss_test_beta_dist_a_b_size_{size}_std_reg_{std_reg}_relpercerr.csv")
    

fig,axs = plt.subplots(1,5, figsize=(25,6),sharey=True,  layout='constrained')
i = 0
labels = ["a)","b)","c)","d)","e)"]
for name_dynamics in ["Diffusion", "MAK", "MM", "PD", "SIS"]:
    d = pd.read_csv(f"results/er_{name_dynamics}_loss_test_beta_dist_a_b_size_{size}_std_reg_{std_reg}_relpercerr.csv")
    ax = axs[i]
    x = np.array(d.a_list)
    y = np.array(d.b_list)
    xy = np.stack((x, y))
    n = len(set(x))
    g = np.array(d.ab_loss_rel)[:,None].reshape((n,n))
    print(g.min(), g.max())    

    im1 =ax.pcolormesh(x.reshape((n,n)),y.reshape((n,n)), g, cmap=plt.cm.get_cmap('gist_rainbow'), )#gist_rainbow
    im1.norm =matplotlib.colors.LogNorm (vmin = 0.1, vmax = 33)
    if name_dynamics == "Diffusion":
        ax.set_title("Heat" )
        ax.set_ylabel("$b$")
    else:
        ax.set_title(f"{name_dynamics}")
    ax.set_yticks(range(1,11))
    ax.set_xticks(range(1,11))
    ax.text(-0.3, 1.2, labels[i], transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=40, fontname='Times New Roman')#, fontweight='bold')
    
    if name_dynamics == "MM":
        ax.set_xlabel("$a$")
    if name_dynamics =="SIS":
        cbar = fig.colorbar(im1)
        # cbar.ax.set_title("$\\frac{\\eta_{(a,b)}}{\\mathcal{L}_{(1,1)}}$",fontsize= 40, rotation=0)
        cbar.ax.set_title("$R_{a,b}$",fontsize= 32, rotation=0)
       
    i+=1
plt.savefig(f"loss_test_beta_dist_a_b_size_{size}_std_reg_{std_reg}_relpercerr.pdf")


