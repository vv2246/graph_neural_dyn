
from tqdm import tqdm
from experiment_class import DynamicsParameters, TrainingParameters, Experiment
from utilities import *
import warnings


def plot_loss_mesh(func_nn, func_dyn, mesh, ax = None,  self_interaction = True):
    
    xy, x, y = mesh
    ny=int(xy.shape[1] **0.5)
    g = []
    pred_y = torch.vstack([func_nn(0, xy[:,i][:,None,None], self_interaction = self_interaction)[:,0,0] for i in range(len(xy.T))]).T.detach()
    true_y = func_dyn(0, xy)
    loss_list = [torch.mean(torch.abs(pred_y[:,i]-true_y[:,i])) for i in range(len(xy[0,:]))]
    for val in loss_list:
        try:
            g.append(float(val.detach().numpy()))
        except:
            g.append(0)
    g = np.array(g)[:,None].reshape((ny,ny))
    # print()
    # print(g)
    im =ax.pcolormesh(x.reshape((ny,ny)),y.reshape((ny,ny)), g, cmap=plt.cm.get_cmap('gist_rainbow'), )
    return im

def plot_vector_field(function,c ,mesh,ax = None,arrowsize= 5, noise = None,self_interaction = True , nbr_interaction = True):
    
    xy, x, y = mesh
    try:
        g = torch.vstack([function(0, xy[:,i][:,None,None], self_interaction = self_interaction, nbr_interaction = nbr_interaction)[:,0,0] for i in range(len(xy.T))]).T.detach() 
    except:
        g = function(0, xy)
    print(g)
    if noise != None:
        g += noise 
    Fx  = np.array(g[0,:])#.unflatten(0,(NY,NY)))
    Fy  =  np.array(g[1,:])#.unflatten(0,(NY,NY)))
    F = [Fx, Fy]
    ax.quiver(x,y,Fx,Fy, color= c, alpha = 0.9, headwidth =arrowsize)


def plot_gradient_vector_field(func_nn, func_dyn, mesh, ax = None,self_interaction = True):
    
    xy, x, y = mesh
    try:
        g = torch.vstack([func_nn(0, xy[:,i][:,None,None], self_interaction = self_interaction)[:,0,0] for i in range(len(xy.T))]).T.detach()
    except:
        g = func_nn(0, xy)
    Fx  = np.array(g[0,:])#.unflatten(0,(NY,NY)))
    Fy  =  np.array(g[1,:])#.unflatten(0,(NY,NY)))
    F = [Fx, Fy]
    
    
    xy, x, y = mesh
    g = []
    pred_y = torch.vstack([func_nn(0, xy[:,i][:,None,None],self_interaction = self_interaction)[:,0,0] for i in range(len(xy.T))]).T.detach()
    true_y = func_dyn(0, xy)
    loss_list = [torch.sum(torch.abs(pred_y[:,i]-true_y[:,i])) for i in range(len(xy[0,:]))]
    for val in loss_list:
        try:
            g.append(float(val.detach().numpy()))
        except:
            g.append(0)
    g = np.array(g)#[:,None].reshape((10,10))
    
    ax.quiver(x,y,Fx,Fy,  g ,alpha =1, headwidth =5)


if __name__ == "__main__":
    set_seeds()
    warnings.filterwarnings('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    network_size = 2
    g = nx.erdos_renyi_graph(network_size, 1)
    scale = 0.1
    dynamics_params = DynamicsParameters(model_name = "Diffusion", B=1, F=2,R = 0.5)
    training_params = TrainingParameters(setting=1, train_distr = torch.distributions.Beta(torch.FloatTensor([5]),torch.FloatTensor([2])),
                                         test_distr = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1])),
                                         train_samples=100,test_samples =100, epochs = 1000, Q_factorized=True, bias = True)
    experiment = Experiment(device = device, dynamics_parameters = dynamics_params, training_parameters = training_params, graph = g)
    x_train, y_train, x_test, y_test = experiment.generate_train_test_data()
    m = torch.distributions.normal.Normal(loc = 0, scale = scale)
    # noise = m.sample([network_size, training_params.train_samples])
    ntrain = len(y_train)
    ntest = len(y_test)
    loss_list_tot = []
    regularizer_lambda = 1
    # from scipy.stats import qmc
    # list_of_samples = []
    # sampler = qmc.Sobol(d=2, scramble=False)
    # sample = sampler.random_base2(m=7)
    # for s in sample:
    #     list_of_samples.append(torch.tensor(s,dtype =torch.float32)[:,None])
      
    folder = f"results/small_example_{dynamics_params.model_name}_regularizer_{regularizer_lambda}"
    for itr in tqdm(range(training_params.epochs +1)):
        experiment.optimizer.zero_grad()
        pred_y = [experiment.func(0, x_train[i][:,None]) for i in range(ntrain)]
        # loss = torch.sum(torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train,1) ))#+ noise)) #L1
        
        v1= torch.squeeze(torch.cat(pred_y,1),2)
        v2 = torch.cat(y_train,1)
        l_idx =torch.abs(v1-v2)
        
        # weigh wrt to variance in the loss
        node_var = l_idx.var(0)
        variance_reg = (node_var.mean() )
        loss = l_idx.mean() + variance_reg * regularizer_lambda
        loss.backward()
        experiment.optimizer.step()
        loss_list_tot.append(float(loss.detach().numpy()))
        
        if itr % 100==0 :
            with torch.no_grad():
                loss_train = torch.sum(torch.abs(torch.squeeze(torch.cat(pred_y,1),2)-torch.cat(y_train,1))) #L1
                pred_y_test = [experiment.func(0, x_test[i][:,None]) for i in range(ntest)]
                loss_test = torch.sum(torch.abs(torch.squeeze(torch.cat(pred_y_test,1),2)-torch.cat(y_test,1))) #L1
                print('Iter {:04d} | Current Loss {:.6f} | Train Loss {:.6f} | Test Loss {:.6f}'.format(itr, loss ,loss_train , loss_test))
                if itr == 1000 or itr == 500 or itr == 0 :
                   
                        fig,ax = plt.subplots(1,2, figsize=(12,6),sharey = False, sharex= False)
                        plot_vector_field(experiment.func,"hotpink", experiment.get_custom_mesh(x_train), ax = ax[0], arrowsize =10, noise= None)
                        plot_vector_field(experiment.dyn,"royalblue", experiment.get_custom_mesh(x_train), ax=ax[0], arrowsize =5, noise = None)
                        ax[0].set_title("Train data~$\mathcal{B}(5,2)$")
                        xy,x, y  = experiment.get_uniform_mesh(10, 2)
                        im = plot_loss_mesh(experiment.func, experiment.dyn, (xy,x, y),ax = ax[1])
                        plot_vector_field(experiment.func,"hotpink",(xy,x, y ), ax = ax[1], arrowsize =10)
                        plot_vector_field(experiment.dyn,"royalblue", (xy,x, y ), ax=ax[1], arrowsize =5)
                        ax[1].set_title("Test data: Mesh")
                        ax[0].set_ylabel("$x_2$")
                        ax[0].set_xlabel("$x_1$")
                        ax[1].set_xlabel("$x_1$")
                        ax[0].set_ylim(-0.05,1.05)
                        ax[0].set_xlim(-0.05,1.05)
                        fig.colorbar(im, ax=ax[1], label = "$\\mathcal{L}$")
                        im.set_clim(0, 2)
                        pos = ax[1].get_position()
                        pos2 = ax[0].get_position()
                        ax[0].set_position([pos.x0-0.35,pos2.y0,pos.width,pos2.height])
                        plt.savefig(f"{folder}/small_example_{itr}_reg_{regularizer_lambda}.pdf")
                        plt.show()
                    
    experiment.save(folder, loss_list_tot, x_train, y_train, x_test, y_test)
    