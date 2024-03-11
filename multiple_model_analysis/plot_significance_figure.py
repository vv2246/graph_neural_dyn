


import pandas as pd
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch 
from utilities import set_seeds, compute_critical_val,get_acc_ratio_sample_vs_null
import numpy as np
import pickle
import seaborn as sns 

if __name__ == "__main__":
    cm = plt.cm.get_cmap('rainbow')
    set_seeds()
    warnings.filterwarnings('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    name_dynamics_list = [ "Diffusion","MAK", "MM", "PD", "SIS"]
    
    ##########
    # Initialize figure
    ##########
    
    fig,axs = plt.subplots(nrows =2 , ncols = 5 , figsize=(30,14),sharey=True , sharex = False,  layout='constrained')
    axins = []
    axs[0][0].text(-0.3, 1.2, 'a)', transform=axs[0][0].transAxes, verticalalignment='top', horizontalalignment='left', fontsize=50, fontname='Times New Roman')#, fontweight='bold')
    axs[1][0].text(-0.3, 1.2, 'b)', transform=axs[1][0].transAxes, verticalalignment='top', horizontalalignment='left', fontsize=50, fontname='Times New Roman')#, fontweight='bold')

    for i in range(5):
        if i == 3 : 
            axins.append(inset_axes(axs[1,i], width=2.5, height=2, loc = 4))
        else:
            
            axins.append(inset_axes(axs[1,i], width=2.5, height=2))

    for axi in  axins:
        axi.tick_params(labelleft=False, labelbottom=False)
        axi.patch.set_alpha(0.3)
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.patch.set_facecolor("white")
        axi.patch.set_alpha(1)
        axi.zorder = 10
        # axi.set_yscale("symlog")
        
    for ax in axs[1]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(-0.1, 1.1 )
        ax.patch.set_alpha(0)
        ax.zorder = 0
        
    shared_axes = [ax.twinx() for ax in axs[0]]
    for sax in shared_axes[1:]:
        shared_axes[0].get_shared_y_axes().join(shared_axes[1], sax)
        
    for sax in shared_axes[:-1]:
        sax.yaxis.set_tick_params(labelright=False)
        
        
    df = pd.read_csv( f"results/delta_vs_pval_vs_loss_network_size_10_new_network_False_alpha_0.05.csv")
    df = df[df.columns[1:]]
        
    df1 = pd.read_csv( f"results/training_loss_network_size_10.csv")
    df1 = df1[df1.columns[1:]]
    
    i = 0
    for key in  name_dynamics_list :
        d = df[df.Dynamics == key]
        ax = axs[1,i]
        ax2 = shared_axes[i]
        d = d[d.columns[1:]]
        ax2.errorbar([-0.1], list(df1[df1.Dynamics == key].loss_mean) , list(df1[df1.Dynamics == key].loss_std) ,  linestyle="none", marker = "^",markersize =12, capsize = 5, color="forestgreen")
        ax2.errorbar(d.groupby("delta").mean().index,d.groupby("delta").mean().loss, d.groupby("delta").std().loss, linestyle="none", marker = "o",markersize =12, capsize = 5, color="slateblue")
        ax.errorbar(d.groupby("delta").mean().index+0.02, 1-d.groupby("delta").mean().pval, d.groupby("delta").std().pval, linestyle="none", marker = "o",markersize =12, capsize = 5, color="darkorange")
                
        ax.yaxis.label.set_color('darkorange')
        ax.tick_params(axis='y', colors='darkorange')
        ax2.yaxis.label.set_color('slateblue')
        ax2.tick_params(axis='y', colors='slateblue')
        ax.set_xticks(np.linspace(0,1,6))#rotation = 90)
        if i ==2:
            ax.set_xlabel("$\\Delta$")
        if i == 0:
            ax.set_ylabel("Fraction rejected")
        if i==4:
            ax2.set_ylabel("Loss")
        if key == "Diffusion":
            ax.set_title("Heat")
        else:
            ax.set_title(key)
        

        ##### 
        # Subfigure b
        #####        
        with open(f"results/subfigure_b_{key}.pkl", "rb") as f:
            d_stat_full_sample, sigpred, y, sol, time = pickle.load(f)    
        
        network_size = y.shape[0]
        for j in range(network_size):
            axs[0,i].plot(time,y.T[:,j], linewidth =5,alpha =0.5, zorder = j, color = cm(j/network_size))
            axs[0,i].plot(time,sol[:,j], linewidth =5,alpha =1, linestyle = "dotted", zorder =j, color = cm(j/network_size))
            
            
        
        sns.ecdfplot(sigpred, complementary= True, color = "darkorange", linewidth=5,alpha= 1, ax = axins[i])
        sns.ecdfplot(d_stat_full_sample, complementary = True, color = "slateblue", linewidth=5,alpha= 1,ax = axins[i])
        axins[i].set_xlim(0,compute_critical_val(d_stat_full_sample , alpha =0.05))
        # compute confidence in prediction
        accepted = get_acc_ratio_sample_vs_null(null_samples = d_stat_full_sample, testing_samples = sigpred, alpha = 0.05)
        accepted = round( accepted * 100 )
        if key == "Diffusion":
            name = "Heat"
        else:
            name = key
        axs[1,i].set_title(name + f": {accepted}%")
    

        i += 1
    
    axs[1,2].set_xlabel("$t$")
    plt.tight_layout()
    plt.savefig(f"significance_figure.pdf")


    
    