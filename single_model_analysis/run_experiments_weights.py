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
import matplotlib.pyplot as plt
import random
from dynamics import Dynamics


def calculate_probabilities(arr):
    unique_values, counts = np.unique(arr, return_counts=True)
    total_elements = len(arr)
    probabilities = counts / total_elements
    return dict(zip(unique_values, probabilities))


if __name__ == "__main__":
    
    warnings.filterwarnings('ignore')
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    multiple_nn = False
    if multiple_nn:
        M_tot = 20
        bootstrap_fraction = 0.9
    else:
        M_tot = 1
        bootstrap_fraction  = 1
        
    model_name = "SIS"
    results_root = "results"
    network_name  = "er"
    N=100
    p = 0.1
    
    
    connected = False
    while connected == False:
        print("mm")
        if network_name == "ba":
            g = nx.barabasi_albert_graph(N,3)
        if network_name =="er":
            g = nx.erdos_renyi_graph(N, p)
        print(g.degree())
        if nx.is_connected(g) ==True:
            connected = True 
            A = torch.FloatTensor(np.array(nx.adjacency_matrix(g).todense()))
        print(g.number_of_edges())
    # A = np.load("er_n_100_p_06.npy")
    # g = nx.from_numpy_array(A)
    # A = torch.FloatTensor(A)
       ############################
        # Definition of regularizers
        ############################
    regularizer_lambda = 1.0 # regularizer that minimizes variance in the loss across nodes
        
    if model_name == "Diffusion":
        dynamics_params = DynamicsParameters(model_name = "Diffusion", B=0.5)     
    if model_name == "MAK":
        dynamics_params = DynamicsParameters(model_name = "MAK", B=0.1, F = 0.5, R  = 1, b= 3)
    if model_name == "PD":
        dynamics_params = DynamicsParameters(model_name = "PD", B=2, R  = 0.3 , a =1.5 , b =3 )
    if model_name == "MM":
        dynamics_params = DynamicsParameters(model_name = "MM", B=4, R  = 0.5 , H =3 )
    if model_name == "SIS":
        dynamics_params = DynamicsParameters(model_name = "SIS", B =4, R  = 0.5 )
    
    
    a,b = 5,2
    m2 = torch.distributions.Beta(torch.FloatTensor([a]),torch.FloatTensor([b]))
    training_params = TrainingParameters(setting=1, train_distr = m2,#torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1])),
                                         test_distr = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1])), 
                                         train_samples=5000,test_samples =100, epochs = 2000, lr=0.01, nsample = 100, weight_decay= 0.001,
                                         h=30, h2=30, h3 = 30, h4 = 30, Q_factorized = True, bias = True )
    experiment = Experiment(device = device, dynamics_parameters = dynamics_params, training_parameters = training_params,
                                  graph = g)

    x_train_full,y_train_full,x_test_full,y_test_full, x_val_full, y_val_full  = [],[],[],[],[],[]

    x_train, y_train= experiment.generate_arbitrary_data( A, training_params.train_samples, training_params.train_distr)
    # x_test, y_test= experiment.generate_arbitrary_data( A, training_params.test_samples, training_params.test_distr)
    x_val, y_val = experiment.generate_arbitrary_data( A, training_params.test_samples, training_params.test_distr)
    n_bootstrap =len(x_train)
    loss_list_tot = []
    

    
    print(f"####################\n SETUP \n####################\nN={N} {model_name} Dynamics on {network_name}\nTrain samples:{training_params.train_samples}\n", end="")
    print(f"Learning rate:{training_params.lr}\nBatch size:{training_params.nsample}\nWeight decay:{training_params.weight_decay}")

    #########################
    # Training
    #########################
    rv = random.random()
    xtest = torch.rand([N, 1])
    t=torch.linspace(0,2,500)
    y_test = odeint( experiment.dyn, xtest, t[:], method="dopri5").squeeze()
    scheduler = ReduceLROnPlateau(experiment.optimizer, 'min', patience= 50, cooldown=10)
    y_val_exp = [y.to(device) for y in y_val]
    x_val_exp = [x.to(device) for x in x_val]
    for itr in range(training_params.epochs+1):
        experiment.optimizer.zero_grad()
        index = torch.randint(0,n_bootstrap,(1,training_params.nsample))
        pred_y = [experiment.func(0, x_train[i][:,None], A) for i in index[0]]
        y_train_batch = [y_train[i] for i in index[0]]
        
        # compute loss
        v1= torch.squeeze(torch.cat(pred_y,1),2)
        v2 = torch.cat(y_train_batch,1)
        l_idx =torch.abs(v1-v2)
        
        # weigh wrt to variance in the loss
        node_var = l_idx.var(0)
        variance_reg = (node_var.mean() )
        loss = l_idx.mean()*1 + variance_reg * regularizer_lambda #+ node_invariance_reg  * node_invariance_reg_lambda
            
        loss.backward()
        experiment.optimizer.step()
        
    
        if itr>1000:
            # compute val loss
            pred_y_val = [experiment.func(0, x_val[i][:, None], A) for i in range(len(x_val))]
            # y_val = [y_val_exp[i] for i in range(len(y_val_exp[0]))]
            loss_tot_val = torch.abs(torch.squeeze(torch.cat(pred_y_val, 1), 2) - torch.cat(y_val, 1)).mean()
            prev_lr = experiment.optimizer.param_groups[0]['lr']
            scheduler.step(loss_tot_val)
            if prev_lr != experiment.optimizer.param_groups[0]['lr']:
                print(f"learning rate scheduler update: ", {experiment.optimizer.param_groups[0]['lr']})
        
        # printing statements
        if itr % 100==0 :
            rv = random.random()
            with torch.no_grad():
                print(itr, loss)
                pred_full = [experiment.func(0, x_train[i][:,None], A)  for i in range(len(x_train))]
                loss_full = torch.abs(torch.cat(pred_full,1).squeeze() - torch.cat(y_train,1)).mean()
                print(loss_full)
                print(index)
                y_test_pred = odeint(experiment.func, xtest[:,None], t[:], method="dopri5").detach().squeeze()
                plt.plot(t,y_test)
                plt.plot(t,y_test_pred,"--")
                plt.show()
    #######
    # Save 
    #######
    experiment.save(f"{results_root}/{network_name}_experiment_{model_name}_size_{N}_std_reg_{regularizer_lambda}_beta_5_2", [], x_train, y_train, [],[], [A])
    # np.save(f"{results_root}/{network_name}_experiment_loss_{model_name}_size_{N}_{exp_iter}_alpha_{alpha}_scale_{scale}_dynwght_{dynamic_weights}_std_reg_{regularizer_lambda}_equiv_reg_{node_invariance_reg_lambda}.npy", np.array(loss_list_tot))
