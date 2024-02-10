import os.path as osp 
import torch 
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, zinc, QM7b
from torch_geometric.utils import to_networkx, homophily, erdos_renyi_graph, stochastic_blockmodel_graph, scatter, loop
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.nn.aggr import DeepSetsAggregation #key to implement DeepSet input (batch_features, batch_index)
from torch_geometric.nn.models import MLP
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, Dropout
import torch_geometric.transforms as T
import torch.nn.functional as F

class ScalarModel(torch.nn.Module):
    def __init__(self, hid_dim, out_dim, dropout=0):
        super(ScalarModel, self).__init__()
        #Deepset for diagonal scalars
        phi_d = MLP([1,hid_dim, hid_dim*2], dropout=[dropout]*2)
        rho_d = MLP([hid_dim*2, hid_dim], dropout=[dropout]*1)
        self.deepset_d = DeepSetsAggregation(local_nn=phi_d, global_nn=rho_d)
        #Deepset for off-diagonal scalars
        phi_o = MLP([1,hid_dim, hid_dim*2], dropout=[dropout]*2)
        rho_o = MLP([hid_dim*2, hid_dim], dropout=[dropout]*1)
        self.deepset_o = DeepSetsAggregation(local_nn=phi_o, global_nn=rho_o) 
        #MLP_s for f_star
        self.MLP_s = MLP([1,hid_dim, hid_dim], dropout=[dropout]*2) 
        #MLP for (Deepset(f_d), Deepset(f_o), f_star)
        self.MLP_out = MLP([hid_dim*3, hid_dim, out_dim], dropout=[dropout]*2)

    def reset_parameters(self):
        for net in [self.deepset_d, self.deepset_o, self.MLP_s, self.MLP_out]:
            net.reset_parameters()
    
    def forward(self, data):
        out_d = self.deepset_d(data.f_d, data.batch) # bs x hid_dim
        out_o = self.deepset_o(data.f_o, data.edge_batch) # bs x hid_dim
        out_star = self.MLP_s(data.f_star) #bs x hid_dim
        #concat and output final embedding
        out = self.MLP_out(torch.concat([out_d, out_o, out_star], dim=-1)) # bs x hid_dim*3 -> bs x out_dim
        return out
