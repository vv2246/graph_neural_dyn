import networkx as nx
import torch 
import random
import numpy as np


np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

gnew = nx.erdos_renyi_graph(100,0.1)
A_new1 = np.array(nx.adjacency_matrix(gnew).todense())


gnew = nx.erdos_renyi_graph(100,0.6)
A_new2 = np.array(nx.adjacency_matrix(gnew).todense())
    
np.save("er_n_100_p_01.npy", A_new1)
np.save("er_n_100_p_06.npy", A_new2)