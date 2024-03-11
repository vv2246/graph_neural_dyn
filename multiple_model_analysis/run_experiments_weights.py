# from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from experiment_class import DynamicsParameters, TrainingParameters, Experiment
from utilities import set_seeds 
import warnings
import torch
import networkx as nx 
import numpy as np
from torchdiffeq import odeint
# import operator
# import matplotlib.pyplot as plt
import random
from dynamics import Dynamics


def calculate_probabilities(arr):
    unique_values, counts = np.unique(arr, return_counts=True)
    total_elements = len(arr)
    probabilities = counts / total_elements
    return dict(zip(unique_values, probabilities))


if __name__ == "__main__":
    set_seeds()
    warnings.filterwarnings('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    multiple_nn = True
    if multiple_nn:
        M_tot = 50
        bootstrap_fraction = 0.9
        results_root = "results/multiple_nn"
        trajectories = True # FOR MULTIPLE NEURAL NETWORKS AND TRAJECTORY TEST DATA
    else:
        M_tot = 1
        bootstrap_fraction  = 1
        results_root = "results/dynamic_weights"
        trajectories = False # FOR MULTIPLE NEURAL NETWORKS AND TRAJECTORY TEST DATA
    model_name = "MAK"
    network_name = "er"
    number_of_graphs = 1
    for N in [10]:# + list(range(100,300,100)):
    # N = 10
        network_sizes = [N for i in range(number_of_graphs)]
        densities = np.linspace(0.5,1,2)    
        graphs = []
        adjacencies = []
        for itr in range(number_of_graphs):
            size = network_sizes[int(itr)]
            p = densities[itr]
            connected = False
            while connected == False:
                if network_name == "ba":
                    g = nx.barabasi_albert_graph(size,3)
                if network_name =="er":
                    g = nx.erdos_renyi_graph(size, p)
                # g= directed_small_world(N, 2, 3)
                print(g.degree())
                if nx.is_connected(g) ==True:
                    connected = True 
                    graphs.append(g)
                    adjacencies.append(torch.FloatTensor(np.array(nx.adjacency_matrix(g).todense())))
                print(g.number_of_edges())
       ############################
        #
        # Definition of regularizers
        #
        ############################
        dynamic_weights = False # if do dynamic weighing 
        node_invariance_reg_lambda = 0.0 # if do a regularizer wrt to differences in interaction term per neighbor
        regularizer_lambda = 1.0 # regularizer that minimizes variance in the loss across nodes
        alpha = 0 # exponent of degree distribution
        scale = 100 # scaling factor for p(k)  as it can be very small
        # regularizer_lambda = 0 # regularizer that minimizes variance in the loss across nodes
        degree_probabilities = []
        weights = []
        for A in adjacencies:
            degrees = A.sum(0).squeeze()
            degree_probabilities.append(calculate_probabilities(degrees))
            probs = torch.tensor([ degree_probabilities[0][k]**alpha for k in list(degrees.detach().numpy())] )
            weights.append((probs)**alpha )
            
        # normalize weights to sum to 1  
        weights_norm = []
        for wgths in weights:
            wights_norm = [w/sum(wgths) for w in wgths]
            #wights_norm = [1.0 for w in wgths]
            weights_norm.append(torch.tensor(wights_norm)* scale)
        weights = weights_norm
        
        #### group nodes according to degree
        grouped_degrees = {k: [] for k in set(list(degrees.numpy())) }
        for i in range(N):
            k = list(degrees.numpy())[i]
            grouped_degrees[k].append(i)
            
        if model_name == "Diffusion":
            dynamics_params = DynamicsParameters(model_name = "Diffusion", B=0.5)     
        if model_name == "MAK":
            dynamics_params = DynamicsParameters(model_name = "MAK", B=0.1, F = 0.5, R  = 1, b=3)
        if model_name == "PD":
            dynamics_params = DynamicsParameters(model_name = "PD", B=2, R  = 0.3 , a =1.5 , b =3 )
        if model_name == "MM":
            dynamics_params = DynamicsParameters(model_name = "MM", B=4, R  = 0.5 , H =3 )
        if model_name == "SIS":
            dynamics_params = DynamicsParameters(model_name = "SIS", B =4, R  = 0.5 )
    
        training_params = TrainingParameters(setting=1, train_distr = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1])),
                                             test_distr = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1])), 
                                             train_samples=1000,test_samples =100, epochs = 2000, lr=0.005, nsample = 50, weight_decay= 0.001,
                                             h=30, h2=30, h3 = 30, h4 = 30, Q_factorized = True, bias = True )
        experiment_list = [Experiment(device = device, dynamics_parameters = dynamics_params, training_parameters = training_params,
                                      graph = g) for i in range(M_tot)]
    
        x_train_full,y_train_full,x_test_full,y_test_full, x_val_full, y_val_full  = [],[],[],[],[],[]
        if trajectories == False:
            for A in adjacencies:
                x_train, y_train= experiment_list[0].generate_arbitrary_data( A, training_params.train_samples, training_params.train_distr)
                x_test, y_test= experiment_list[0].generate_arbitrary_data( A, training_params.test_samples, training_params.test_distr)
                x_val, y_val = experiment_list[0].generate_arbitrary_data( A, training_params.test_samples, training_params.test_distr)
                x_train_full.append(x_train)
                y_train_full.append(y_train)
                x_test_full.append(x_test)
                y_test_full.append(y_test)
                x_val_full.append(x_val)
                y_val_full.append(y_val)
            
        else:    
            t = torch.linspace(0,dynamics_params.T,int(dynamics_params.T/ dynamics_params.dt))
            n_traj_train =int(np.ceil( training_params.train_samples /(dynamics_params.T/dynamics_params.dt)))
            n_traj_test =int(np.ceil( training_params.test_samples /(dynamics_params.T/dynamics_params.dt)))
            print(n_traj_train)
            x_train_full,y_train_full,x_test_full,y_test_full  = [],[],[],[]
            
            for A in adjacencies:
                x_train, x_test, y_train, y_test = [], [], [], []
                for i in range(n_traj_train):
                    x = training_params.train_distr.sample([N])
                    y = odeint( experiment_list[0].dyn, x, t, method=training_params.method).squeeze().t()
                    for j in range(t.shape[0]-1): 
                        x_train.append(y[:,j][:,None])
                        y_train.append(experiment_list[0].dyn(0,y[:,j][:,None]))
                        # y_train.append((y[:,j+1][:,None] -y[:,j][:,None])/dynamics_params.dt )
            
                for i in range(n_traj_test):
                    x = training_params.test_distr.sample([N])
                    y = odeint( experiment_list[0].dyn, x, t, method=training_params.method).squeeze().t()
                    for j in range(t.shape[0]): 
                        x_test.append(y[:,j][:,None])
                        y_test.append(experiment_list[0].dyn(0,y[:,j][:,None]))
                x_train_full.append(x_train)
                y_train_full.append(y_train)
                x_test_full.append(x_test)
                y_test_full.append(y_test)
            x_val_full , y_val_full = x_test_full,y_test_full
    
        ntrain,ntest = len(x_train_full[0]), len(y_test_full[0])
        n_bootstrap = int(bootstrap_fraction * ntrain)
        
        #########################
        # do bootstrap of DATA
        #########################
        x_train_tot , y_train_tot = [],[]
        for m in range(M_tot):
            x_train_m = []
            y_train_m = []
            for j in range(number_of_graphs):
                index = torch.randint(0,ntrain,(1,n_bootstrap))
                x_train_m.append( [x_train_full[j][i] for i in index[0]] )
                y_train_m.append([y_train_full[j][i] for i in index[0]] )
            x_train_tot.append(x_train_m)
            y_train_tot.append(y_train_m)
        
        loss_list_tot = []
        
        #########################
        # Setup variables and functions for invariance regularizer 
        #########################
        node_degree = {i: int(A.sum(0)[i]) for i in range(A.shape[0])}
        x0 = torch.zeros([size,1,1])+ 1
        x0.to(device)
        y0 = experiment_list[0].generate_arbitrary_point(A, x0.squeeze(dim=2))
        y0 = y0[:,None]
        y0.to(device)

        dyn_self  = Dynamics(A.to(device), model=dynamics_params.model_name, B=dynamics_params.B, R=dynamics_params.R,
                            H=dynamics_params.H, F=dynamics_params.F, a=dynamics_params.a, b=dynamics_params.b,  self_interaction = True, nbr_interaction = False).to(device)
        dyn_nbr = Dynamics(A.to(device), model=dynamics_params.model_name, B=dynamics_params.B, R=dynamics_params.R,
                            H=dynamics_params.H, F=dynamics_params.F, a=dynamics_params.a, b=dynamics_params.b,  self_interaction = False, nbr_interaction = True).to(device)

        adjacencies = [x.to(device) for x in adjacencies]
        x_train_tot = [[[x.to(device) for x in x_sample] for x_sample in x_exp_list] for x_exp_list in x_train_tot]
        y_train_tot = [[[y.to(device) for y in y_sample] for y_sample in y_exp_list] for y_exp_list in y_train_tot]
        
        print(f"####################\n SETUP \n####################\nN={size} {model_name} Dynamics on {network_name}\nTrain samples:{training_params.train_samples}\n", end="")
        print(f"Learning rate:{training_params.lr}\nBatch size:{training_params.nsample}\nWeight decay:{training_params.weight_decay}")
        print(f"Regularizers:\nDynamic weighting:{dynamic_weights}\nVariance Regularizer:{bool(regularizer_lambda)}\nDegree weighting:{bool(alpha)}")
        print(f"BSS const term regularizer: {node_invariance_reg_lambda}\n")

        #########################
        # Training
        #########################
        rv = random.random()
        for exp_iter in range(M_tot):
            print(exp_iter)
            experiment = experiment_list[exp_iter]
            scheduler = ReduceLROnPlateau(experiment.optimizer, 'min', patience= 50, cooldown=10)
            x_train_exp = x_train_tot[exp_iter]
            y_train_exp = y_train_tot[exp_iter]
            # x_test_exp = x_test_full[exp_iter]
            # y_test_exp = y_test_full[exp_iter]
            y_val_exp = [y.to(device) for y in y_val_full[0]]
            x_val_exp = [x.to(device) for x in x_val_full[0]]
            for itr in range(training_params.epochs+1):
                experiment.optimizer.zero_grad()
                loss = []
                for idx in range(number_of_graphs):
                    # take bootstrapped sample 
                    index = torch.randint(0,n_bootstrap,(1,training_params.nsample))
                    pred_y = [experiment.func(0, x_train_exp[idx][i][:,None], adjacencies[idx]) for i in index[0]]
                    y_train_batch = [y_train_exp[idx][i] for i in index[0]]
                    
                    # compute loss
                    v1= torch.squeeze(torch.cat(pred_y,1),2)
                    v2 = torch.cat(y_train_batch,1)
                    l_idx =torch.abs(v1-v2)
                    
                    # weigh wrt to variance in the loss
                    node_var = l_idx.var(0)
                    variance_reg = (node_var.mean() )
                    # variance_reg_degree =torch.stack([l_idx[grouped_degrees[key]].var() for key in grouped_degrees.keys()]).mean()
                    
                    # regularizer wrt to differences in interaction term per neighbor
                    # node_invariance_reg =  torch.stack([experiment.func(0, x0.to(device) , adjacencies[idx].to(device), self_interaction = False)[j]/node_degree[j] for j in range(size)]).var()
                    dynamics_out = experiment.func(0, x0.to(device) , adjacencies[idx].to(device), self_interaction = True)
                    node_invariance_reg = torch.abs(dynamics_out - y0.to(device)).mean()
                    #node_invariance_reg_sample = torch.stack([interaction_dynamics_out[j]/node_degree[j] for j in range(size)])
                    #node_invariance_reg =  node_invariance_reg_sample.var()

                    
                    # weigh wrt degree 
                    l_idx =  torch.vstack([l_idx[i]* float(weights[0][i].numpy())  for i in range(N)])
                    l_idx_weighted = l_idx.clone()
                    
                    # dynamic weighting
                    if dynamic_weights :
                        for i in range(l_idx.shape[1]):
                            batch_loss = l_idx[:,i]
                            topmost =list(torch.topk(batch_loss,10).indices.detach().numpy())
                            l_idx_weighted[:,i][topmost] = batch_loss[topmost] * 10
                            
                    # mean l1 loss for the batch 
                    l = l_idx_weighted.mean()*1 + variance_reg * regularizer_lambda + node_invariance_reg  * node_invariance_reg_lambda
                    
                    loss.append(l)
                loss = sum(loss)/number_of_graphs

                # compute val loss
                pred_y_val = [experiment.func(0, x_val_exp[i][:, None], adjacencies[idx]) for i in
                              range(len(x_val_exp[0]))]
                y_val = [y_val_exp[i] for i in range(len(y_val_exp[0]))]
                loss_tot_val = torch.abs(torch.squeeze(torch.cat(pred_y_val, 1), 2) - torch.cat(y_val, 1)).mean()
                prev_lr = experiment.optimizer.param_groups[0]['lr']
                if itr>1000:
                    scheduler.step(loss_tot_val)
                if prev_lr != experiment.optimizer.param_groups[0]['lr']:
                    print(f"learning rate scheduler update: ", {experiment.optimizer.param_groups[0]['lr']})
                
                # printing statements
                if itr % 1000==0 :
                    rv = random.random()
                    with torch.no_grad():

                        #compute total loss
                        pred_y = [experiment.func(0, x_train_exp[idx][i][:,None], adjacencies[idx]) for i in range(len(x_train_exp[0]))]
                        y_train = [y_train_exp[idx][i] for i in range(len(x_train_exp[0]))]
                        loss_tot =torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train,1)).mean()


                        #compute val loss
                        pred_y_val = [experiment.func(0, x_val_exp[i][:,None], adjacencies[idx]) for i in range(len(x_val_exp[0]))]
                        y_val= [y_val_exp[i] for i in range(len(y_val_exp[0]))]
                        loss_tot_val =torch.abs(torch.squeeze(torch.cat(pred_y_val,1),2)-torch.cat(y_val,1)).mean()

                        # compute loss wrt q
                        y_pred_q = [experiment.func(0, x_train_exp[idx][i][:,None], adjacencies[idx],self_interaction = False) for i in range(len(x_train_exp[0]))]
                        y_true_q = [dyn_nbr(0, x_train_exp[idx][i]) for i in range(len(x_train_exp[0]))]
                        loss_q = torch.abs(torch.squeeze(torch.cat(y_pred_q,1),2) - torch.cat(y_true_q,1)).mean()

                        # compute loss wrt l
                        y_pred_l = [experiment.func(0, x_train_exp[idx][i][:,None], adjacencies[idx],nbr_interaction = False) for i in range(len(x_train_exp[0]))]
                        y_true_l = [dyn_self(0, x_train_exp[idx][i]) for i in range(len(x_train_exp[0]))]
                        loss_l = torch.abs(torch.squeeze(torch.cat(y_pred_l,1),2) - torch.cat(y_true_l,1)).mean()

                        print(itr, "total loss: ",round(float(loss_tot),5),"loss Q :", round(float(loss_q),5), "loss L : " , round(float(loss_l),5 ), "degree reg:", round(float(variance_reg),10), "validation loss: ", round(float(loss_tot_val),5 )) #

                        loss_list_tot.append((float(loss_tot.detach().cpu().numpy()),float(loss_q.detach().cpu().numpy()),float(loss_l.detach().cpu().numpy()), float(variance_reg.detach().cpu().numpy()),float(loss_tot_val.detach().cpu().numpy())))

                loss.backward()
                experiment.optimizer.step()

            with torch.no_grad():
                pred_y = [experiment.func(0, x_train_exp[idx][i][:,None], adjacencies[idx]) for i in range(len(x_train_exp[0]))]
                y_train_batch = [y_train_exp[idx][i] for i in range(len(x_train_exp[0]))]
                l_idx =torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train_batch,1))
                print(l_idx.mean())
            
            #######
            # Save 
            #######
            if exp_iter == 0:
                experiment.save(f"{results_root}/{network_name}_experiment_{model_name}_size_{N}_{exp_iter}_alpha_{alpha}_scale_{scale}_dynwght_{dynamic_weights}_std_reg_{regularizer_lambda}_equiv_reg_{node_invariance_reg_lambda}", [], x_train_tot[exp_iter], y_train_tot[exp_iter], x_train, y_train, adjacencies)
            else:
                experiment.save(f"{results_root}/{network_name}_experiment_{model_name}_size_{N}_{exp_iter}_alpha_{alpha}_scale_{scale}_dynwght_{dynamic_weights}_std_reg_{regularizer_lambda}_equiv_reg_{node_invariance_reg_lambda}", [], x_train_tot[exp_iter], y_train_tot[exp_iter], x_train, y_train)
            np.save(f"{results_root}/{network_name}_experiment_loss_{model_name}_size_{N}_{exp_iter}_alpha_{alpha}_scale_{scale}_dynwght_{dynamic_weights}_std_reg_{regularizer_lambda}_equiv_reg_{node_invariance_reg_lambda}.npy", np.array(loss_list_tot))

        print(f"min - max value of agg input after training: ({experiment.func.agg_input_min}, {experiment.func.agg_input_max})")
        # experiment.func.agg_input_min = 0.0
        # experiment.func.agg_input_max = 0.0

        for size2 in [100, 200, 300]:
            x0_test = torch.zeros([size2,1,1])+ 1
            gnew = nx.erdos_renyi_graph(size2 , 0.1)
            A = torch.FloatTensor(np.array(nx.adjacency_matrix(gnew).todense())).to(device)
            x_test = [experiment.train_distr.sample([size2]) for i in range(1000)]
            pred_y = [experiment.func(0, x_test[i][:,None].to(device), A.to(device)) for i in range(1000)]
            dyn_test = Dynamics(A, model=dynamics_params.model_name, B=dynamics_params.B, R=dynamics_params.R,
                                H=dynamics_params.H, F=dynamics_params.F, a=dynamics_params.a, b=dynamics_params.b)
            true_y = [dyn_test(0, x_test[i].to(device)) for i in range(1000)]
            loss = torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(true_y,1)).detach()
            variance_reg_test = loss.var(0).mean()
            loss = loss.mean()
            dyn_self_test  = Dynamics(A, model=dynamics_params.model_name, B=dynamics_params.B, R=dynamics_params.R,
                                H=dynamics_params.H, F=dynamics_params.F, a=dynamics_params.a, b=dynamics_params.b,  self_interaction = True, nbr_interaction = False)
            dyn_nbr_test = Dynamics(A, model=dynamics_params.model_name, B=dynamics_params.B, R=dynamics_params.R,
                                H=dynamics_params.H, F=dynamics_params.F, a=dynamics_params.a, b=dynamics_params.b,  self_interaction = False, nbr_interaction = True)
            
            y_pred_q = [experiment.func(0, x_test[i][:,None].to(device), A,self_interaction = False) for i in range(len(x_test))]
            y_pred_l = [experiment.func(0, x_test[i][:,None].to(device), A,nbr_interaction = False) for i in range(len(x_test))]
            y_true_l = [dyn_self_test(0, x_test[i].to(device)) for i in range(len(x_test))]
            y_true_q = [dyn_nbr_test(0, x_test[i].to(device)) for i in range(len(x_test))]
            loss_q = torch.abs(torch.squeeze(torch.cat(y_pred_q,1),2) - torch.cat(y_true_q,1)).mean()
            loss_l = torch.abs(torch.squeeze(torch.cat(y_pred_l,1),2) - torch.cat(y_true_l,1)).mean()
            
            # node_degree = {i: int(A.sum(0)[i]) for i in range(A.shape[0])}
            # node_invariance_reg_test =  torch.stack([experiment.func(0, x0_test.to(device) , A.to(device), self_interaction = False)[j]/node_degree[j] for j in range(size2)]).var()

            print(f"TEST {size2}\n", "total loss: ",round(float(loss),5),
                      "loss Q :", round(float(loss_q),5), 
                      "loss L : " , round(float(loss_l),5 ), 
                      # "invariance reg:", round(float(node_invariance_reg_test),10), 
                      "degree reg:", round(float(variance_reg_test),10))

            rdn_loss = []
            rdn_loss_q = []
            rdn_loss_l = []
            for rnd_iter in range(10):
                modelRND = Experiment(device=device, dynamics_parameters=dynamics_params, training_parameters=training_params,
                           graph=gnew)
    
                pred_y = [modelRND.func(0, x_test[i][:,None].to(device), A.to(device)) for i in range(1000)]
                true_y = [dyn_test(0, x_test[i].to(device)) for i in range(1000)]
                loss = torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(true_y,1)).detach()
                rdn_loss.append(loss.mean())
                
                y_pred_q = [modelRND.func(0, x_test[i][:,None].to(device), A,self_interaction = False) for i in range(len(x_test))]
                y_pred_l = [modelRND.func(0, x_test[i][:,None].to(device), A,nbr_interaction = False) for i in range(len(x_test))]
                y_true_l = [dyn_self_test(0, x_test[i].to(device)) for i in range(len(x_test))]
                y_true_q = [dyn_nbr_test(0, x_test[i].to(device)) for i in range(len(x_test))]
                loss_q = torch.abs(torch.squeeze(torch.cat(y_pred_q,1),2) - torch.cat(y_true_q,1)).mean()
                loss_l = torch.abs(torch.squeeze(torch.cat(y_pred_l,1),2) - torch.cat(y_true_l,1)).mean()
                rdn_loss_q.append(loss_q.mean())
                rdn_loss_l.append(loss_l.mean())
            print(f"TEST loss of untrained model {size2}: ", round(float(torch.mean(torch.tensor(rdn_loss))),5))

            print(f"Loss Q on untrained model ", round(float(loss_q),5))
            print(f"Loss L on untrained model " , round(float(loss_l),5))

            # print(
            #     f"min - max value of agg input during inference: ({experiment.func.agg_input_min}, {experiment.func.agg_input_max})")
            # experiment.func.agg_input_min = 0.0
            # experiment.func.agg_input_max = 0.0



   