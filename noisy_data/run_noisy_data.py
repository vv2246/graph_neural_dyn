# from tqdm import tqdm
from experiment_class import DynamicsParameters, TrainingParameters, Experiment
from utilities import  set_seeds
import warnings
# import seaborn as sns
import torch
import networkx as nx
from torchdiffeq import odeint
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes, inset_axes

cm = plt.cm.get_cmap('rainbow')


def plot_true_predicted(t_plot,t_true ,  x  , y    ,dyn, func , name_dyn, itr, scale_obs, scale_t ):
    t_plot [0 ] = -0.001
    plt.rcParams.update({'font.size': 18})
    t_true[0] = -0.001
    x0 = x[:,0][:,None]
    y_true_part = odeint( dyn, x0 ,t_plot, method=training_params.method).squeeze().t()
    y_true_full = odeint( dyn, x0 ,t_true, method=training_params.method).squeeze().t()
    
    dt_val =t_true.diff()[-1]
    
    y_pred  = []
    for j in range(y.shape[1]-1):
        x0_curr = x[:,j][:,None,None]
        # print(dt_val,j,t_plot[j+1] , t_plot[j] ,  torch.linspace(t_plot[j], t_plot[j+1],int((t_plot[j+1] - t_plot[j] ) /dt_val )))
        y_pred_curr = odeint( func, x0_curr ,torch.linspace(t_plot[j], t_plot[j+1],int((t_plot[j+1] - t_plot[j] ) /dt_val )),
                             method=training_params.method).squeeze().t().detach()
        # print(y_pred_curr)
        y_pred.append(y_pred_curr)
    # y_pred = torch.stack(y_pred).T
    fig, ax = plt.subplots(figsize = (4,4), layout='constrained')
    axins = inset_axes(ax, 1.5,1.5 , loc=4)
    for j in range(y.shape[0]):
        ax.scatter(t_plot[1:],  y[j], marker = "s", s = 2, facecolor = "none", edgecolor = cm(j/10)) # label 
        axins.scatter(t_plot[1:],  y[j] , marker = "s", s = 30, facecolor = "none", edgecolor = cm(j/10)) # label inset 
        # axins.plot(t_plot[1:],  y[j] ,color = cm(j/10))
        ax.plot(t_true,  y_true_full[j], lw=1, color = cm(j/10)) # predict  for dt base
        axins.plot(t_true,  y_true_full[j], lw=1, color = cm(j/10))# predict  for dt base  inset 
        
        ax.scatter(t_plot,  y_true_part[j] , marker = "o",s = 2, color = cm(j/10)) # ground truth at true sampling 
        axins.scatter(t_plot,  y_true_part[j], marker = "o" , s = 30,color = cm(j/10)) # ground truth at true sampling 

        for k in range(len(y_pred)):
            axins.plot( torch.linspace(t_plot[k], t_plot[k+1],int((t_plot[k+1] - t_plot[k] ) /dt_val )) , y_pred[k][j] ,color = cm((j)/10), linestyle = "--" )
            try:
                axins.scatter( t_plot[k+1] , y_pred[k][j][-1] ,color = cm((j)/10), marker = "^", s= 30,facecolor = "none")
            except:
                axins.scatter( t_plot[k+1] , y_pred[k][j] ,color = cm((j)/10), marker = "^", s= 30 ,facecolor = "none")
        

    x1, x2, y1, y2 = 0.25, 0.30, 0.63, 0.68 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", lw=1, ec='black', zorder = 10)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    ax.set_yticks(np.linspace(0,1,6))
    ax.set_xticks(np.linspace(0,1,6))
    ax.set_xlabel("$t$")
    # fig.suptitle(f"Epoch:{itr}")
    # plt.savefig(f"figures/noisy_data_{name_dyn}_iter_{itr}_noise_obs_{scale_obs}_noise_t_{scale_t}.pdf")
    plt.show()
    
    
if __name__ == "__main__":
    set_seeds()
    warnings.filterwarnings('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    network_size = 10
    model_name = "Diffusion"
    connected = False
    while connected == False:
        g = nx.erdos_renyi_graph(network_size, 0.3)
        if nx.is_connected(g) ==True:
            connected = True 
    if model_name == "Diffusion":
        dynamics_params = DynamicsParameters(model_name = "Diffusion", B=1.5, T = 1, dt = 10**-4)     
    if model_name == "MAK":
        dynamics_params = DynamicsParameters(model_name = "MAK", B=0.1, F = 0.5, R  = 1,  T = 1, dt = 10**-4)     
    if model_name == "PD":
        dynamics_params = DynamicsParameters(model_name = "PD", B=2, R  = 0.3 , a =1.5 , b =3, T = 1, dt = 10**-4)     
    if model_name == "MM":
        dynamics_params = DynamicsParameters(model_name = "MM", B=4, R  = 0.5 , H =3 , T = 1, dt = 10**-4)     
    if model_name == "SIS":
        dynamics_params = DynamicsParameters(model_name = "SIS", B =4, R  = 0.5 , T = 1, dt = 10**-4)     
    
    scale_t = 0.01
    scale_obs = 0.01
    training_params = TrainingParameters(setting=1, 
                                         train_distr = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1])),
                                         test_distr = torch.distributions.Beta(torch.FloatTensor([5]),torch.FloatTensor([2])), 
                                         train_samples=1000,test_samples =100, epochs = 2000, lr=0.0001, nsample = 50, weight_decay= 10e-3,
                                         h=20, h2=20, h3 = 20, h4 = 20,  Q_factorized = True, bias = True)
    experiment = Experiment(device = device, dynamics_parameters = dynamics_params, training_parameters = training_params, graph = g)

    Delta=   10**-2
    dt_base = dynamics_params.dt
    t = torch.linspace(0,dynamics_params.T, int(1/dt_base))
    irreg_index = torch.sort(torch.randint(0,t.shape[0],(1,int(Delta/dt_base)))).values[0]
    irreg_index[0] = 0
    irreg_index = irreg_index[torch.concat([irreg_index.diff(), torch.tensor([1])]) >1]
    t_irreg = t[irreg_index]
    y_nonoise = []
    y_train = []
    x_train = []
    t_train = []
    niter =int( training_params.train_samples / t.shape[0] )
    # if niter == 0:
    niter = 1
    x_test_1 = [torch.rand([network_size,1]) for i in range(1000)]
    
    for i in range(niter):
        for xx in range(5):
            x0 = torch.rand([network_size,1])
        y = odeint( experiment.dyn, x0, t, method=training_params.method).squeeze().t()
        y_irreg = y[:,irreg_index]
        try:
            m = torch.distributions.normal.Normal(loc = 0, scale = scale_obs)
            noise = m.sample([network_size, t_irreg.shape[0]])#* scale
        except:
            noise = 0
            print("var is zero")
        y_noise =  y_irreg + noise
        y_train_i = y_noise.T
        y_train_i = y_train_i[1:,:,None] # T - 1, N, 1
        x_train_i = y_noise[:,:-1] 
        x_train_i = x_train_i.T
        x_train_i = x_train_i[:,:,None]
        t_diff = t_irreg.diff()
        for i in range(len(y_train_i)):
            y_nonoise.append(y.T [i, :, None])
            y_train.append(y_train_i[i])
            x_train.append(x_train_i[i])
            t_i = torch.linspace(0,t_diff[i], int(t_diff[i]/dt_base ))
            t_train.append(t_i)
            if len(t_i) ==0:
                print(t_diff[i])
                
    ntrain = len(y_train)
    print(len(y_train))
    loss_list_tot= []
    
    plt.plot(t_irreg, y_irreg.T)
    plt.plot(t_irreg, y_noise.T)
    plt.plot(t, y.T, "--")
    plt.show()
    
    t_test = torch.linspace(0,Delta,int(Delta/dt_base))
    for itr in range(training_params.epochs +1):
        experiment.optimizer.zero_grad()
        index = torch.randint(0,ntrain,(1,training_params.nsample))
        y_train_batch = [y_train[i] for i in index[0]]
        pred_y = [ ] 
        for i in index[0]:
            t_curr = t_train[i]
            pred_y_i =  odeint(lambda y, t: experiment.func(y, t), x_train[i][:,None], t_curr, method=training_params.method)[-1,:,:,:] 
            pred_y.append(pred_y_i)
        # loss = torch.sum(torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train_batch,1))) #+ torch.cat(noise_batch,1) )) 
        loss = torch.sum((abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train_batch,1))).sum()) #+ torch.cat(noise_batch,1) )) 
        
        loss_list_tot.append(float(loss.detach().numpy()))
        if itr % 100==0 :
            print("EPOCH: ", itr, "TRAIN loss: ",round( float(loss.detach()),5), end=" ")
            loss_test = []
            pred_test = []
            true_test= []
            for xt in x_test_1:
                # loss_test.append((abs(odeint(lambda y, t: experiment.dyn(y, t), xt[:], t_test, method=training_params.method)[-1,:,:]-odeint(lambda y, t: experiment.func(y, t), xt[:,None], t_test, method=training_params.method)[-1,:,:,0])).sum())
                # loss_test.append(((odeint(lambda y, t: experiment.dyn(y, t), xt[:], t_test, method=training_params.method)[-1,:,:]-odeint(lambda y, t: experiment.func(y, t), xt[:,None], t_test, method=training_params.method)[-1,:,:,0])**2))#.sum())
                pred_test.append( torch.squeeze(odeint(lambda y, t: experiment.func(y, t), xt[:,None], t_test, method=training_params.method)[-1,:,:]))
                true_test.append( torch.squeeze(odeint(lambda y, t: experiment.dyn(y, t), xt[:], t_test, method=training_params.method)[-1,:,:]))
            loss_test = (abs(torch.stack(pred_test) - torch.stack(true_test)))
            # print("TEST loss: " , round(float(torch.stack(loss_test).mean().detach().numpy()),5)," pm ",round(float(torch.stack(loss_test).std().detach().numpy()),5))
            print("TEST loss: " , round(float(loss_test.sum(1).mean().detach().numpy()),5)," pm ",round(float(loss_test.sum(1).std().detach().numpy()),5))
                
        loss.backward()
        experiment.optimizer.step()
        
    # with torch.no_grad():
    #     y = torch.stack([y_train[i] for i in range(int(len(y_train)/niter))]).T.squeeze() 
    #     x = torch.stack([x_train[i] for i in range(int(len(y_train)/niter))]).T.squeeze()
    #     plot_true_predicted(t_irreg, t  ,  x  , y    ,experiment.dyn, experiment.func , model_name, itr, scale_obs, scale_t )
    # experiment.save(f"results/noisy_data/experiment_{model_name}_size_{network_size}_scale_t_{scale_t}_scale_obs_{scale_obs}_Delta_{Delta}", loss_list_tot, x_train, y_train, t, t_train)


    ######################### 
    # Comparison with the untrained network
    # #########################
    # modelRND = Experiment(device=device, dynamics_parameters=dynamics_params, training_parameters=training_params,
    #            graph=g)
    # pred_test = []
    # true_test= []
    # for xt in x_test_1:
    #     pred_test.append( torch.squeeze(odeint(lambda y, t: modelRND.func(y, t), xt[:,None], t_test, method=training_params.method)[-1,:,:]))
    #     true_test.append( torch.squeeze(odeint(lambda y, t: experiment.dyn(y, t), xt[:], t_test, method=training_params.method)[-1,:,:]))
    # loss = (abs(torch.stack(pred_test) - torch.stack(true_test))).sum(1)#.mean()
    
        
    # print(f"TEST loss of untrained model: ", round(float(torch.mean(torch.tensor(loss))),5)," pm ",round(float(torch.std(torch.tensor(loss))),5))


    
        
    # # trained vs learnt dynamics in-sample
    # x0 = y[:,0][:,None]
    # with torch.no_grad():
    #     sol_pred=odeint(experiment.func, x0[:,None] ,t, method=training_params.method).squeeze().t().T
    #     sol_true =odeint(experiment.dyn, x0[:] ,t, method=training_params.method).squeeze().t().T
    #     plt.plot(sol_pred)
    #     plt.plot(sol_true,"--")
        
    # print("insample traj loss",torch.mean(abs( sol_pred - sol_true)))
       
    # # out sample
    # err = []
    # for i in range(100):
    #     x0 = torch.rand([network_size,1])
    #     with torch.no_grad():
    #         sol_pred=odeint(experiment.func, x0[:,None] ,t, method=training_params.method).squeeze().t().T
    #         sol_true =odeint(experiment.dyn, x0[:] ,t, method=training_params.method).squeeze().t().T
    #         plt.plot(sol_pred)
    #         plt.plot(sol_true,"--")
    #     err.append(torch.mean(abs( sol_pred - sol_true)))
    # print("outsample traj loss",torch.mean(torch.tensor(err)), torch.std(torch.tensor(err)))
       
    
    
    
    modelRND = Experiment(device=device, dynamics_parameters=dynamics_params, training_parameters=training_params,
               graph=g)
    
    
    # out sample
    err = []
    err_rdn = []
    for i in range(100):
        x0 = torch.rand([network_size,1])
        with torch.no_grad():
            sol_pred=odeint(experiment.func, x0[:,None] ,t, method=training_params.method).squeeze().t().T
            sol_true =odeint(experiment.dyn, x0[:] ,t, method=training_params.method).squeeze().t().T
            sol_rdn = torch.squeeze(odeint(lambda y, t: modelRND.func(y, t), x0[:,None], t, method=training_params.method)[-1,:,:])
        err_rdn.append( torch.mean(abs( sol_rdn - sol_true)))
        err.append(torch.mean(abs( sol_pred - sol_true)))
    print("outsample traj loss",torch.mean(torch.tensor(err)), torch.std(torch.tensor(err)))
    print("outsample traj loss RND ",torch.mean(torch.tensor(err_rdn)), torch.std(torch.tensor(err_rdn)))
    

# for no noise :     
# outsample traj loss tensor(0.0728) tensor(0.0424)
# outsample traj loss RND  tensor(0.1936) tensor(0.0300)

# with noise:
# outsample traj loss tensor(0.0871) tensor(0.0566)
# outsample traj loss RND  tensor(0.2409) tensor(0.0269)


      
    