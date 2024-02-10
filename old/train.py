import os.path as osp 
import numpy as np
import torch 
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, homophily, erdos_renyi_graph, stochastic_blockmodel_graph, scatter, loop
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.nn.aggr import DeepSetsAggregation #key to implement DeepSet input (batch_features, batch_index)
from torch_geometric.nn.models import MLP
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, Dropout
import torch_geometric.transforms as T
import torch.nn.functional as F

import time
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from architecture import ScalarModel

def get_fs(data):
  '''
  input: DataBatch from pytorch geometric data loader (let bs denote batch_size)
    edge_index=[bs, num_edges]
    edge_attr=[num_edges]
    y=[bs, num_targets]
    x=[bs*num_nodes, 1]
    batch=[bs*num_nodes] (of the form [0, 0, ..,0, 1,...,1,..., bs-1...,bs-1])

  output: updated DataBatch with the following additional invariant features
   f_d is the set of diagonal edge attributes, 
   f_o is the set of off-diagonal (upper) edge attributes, 
   f_star is \sum_{i \neq j} X_ii X_ij where X is the n by n edge attribute graph
  '''
  bs = data.y.shape[0]
  #get f_d, f_o
  loop_mask = data.edge_index[0] == data.edge_index[1]
  data.f_d = data.edge_attr[loop_mask].unsqueeze(1) #use data.batch as batch index (for nodes)
  data.f_o = data.edge_attr[~loop_mask].unsqueeze(1)
  #extract batch index for edges, adapted per https://github.com/pyg-team/pytorch_geometric/issues/1827
  #with self-loops remove for off-digaonal entries
  row, col = loop.remove_self_loops(data.edge_index)[0]
  data.edge_batch = data.batch[row]
  #get f_star
  sum_fo = scatter(data.edge_attr, data.edge_index[1], dim=0) #sum over off-diagonal elements
  data.f_star = torch.zeros((bs, 1))
  data.f_star.index_add_(0, data.batch, data.f_d.squeeze(1) * ( sum_fo - data.f_d.squeeze(1)) )
  return data

def train(train_loader, model, optimizer, device):
    model.train()
    loss_all = 0

    lf = torch.nn.L1Loss()

    for data in train_loader:
        data = get_fs(data)
        #print(data)
        data = data.to(device)
        optimizer.zero_grad()
        loss = lf(model(data), data.y)

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return (loss_all / len(train_loader.dataset))


def test(loader, model, device):
    model.eval()
    error = 0

    for data in loader:
        data = get_fs(data)
        data = data.to(device)
        error += (model(data) - data.y).abs().sum().item()
    return error / len(loader.dataset)


# 10-CV for NN training and hyperparameter selection.
def nn_evaluation(dataset, hid_dim, out_dim, max_num_epochs=300, batch_size=128, 
                                  start_lr=0.02, min_lr = 0.000001, factor=0.8, patience=10,
                       num_repetitions=1, all_std=True, verbose=True, dropout=0):
    #reproducibility
    torch.manual_seed(0)

    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_all = []
    test_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_error = []
        kf = KFold(n_splits=10, shuffle=True, random_state=i)
        #dataset.shuffle()

        for train_index, test_index in kf.split(list(range(len(dataset)))):
            # Sample 10% split from training split for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=i)
            best_val_error = None
            best_test = None

            # Split data.
            train_dataset = dataset[train_index.tolist()]
            val_dataset = dataset[val_index.tolist()]
            test_dataset = dataset[test_index.tolist()]

            # Prepare batching.
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #reproducible

            # Collect val. and test acc. over all hyperparameter combinations.
            # Setup model.
            model = ScalarModel(hid_dim, out_dim, dropout).to(device)
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=factor, patience=patience,
                                                                    min_lr=0.0000001)
            times = []
            for epoch in range(1, max_num_epochs + 1):
                lr = scheduler.optimizer.param_groups[0]['lr']
                #count training time per epoch
                torch.cuda.synchronize() 
                start_time = time.time()
                train(train_loader, model, optimizer, device)
                torch.cuda.synchronize()
                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)

                loss = train(train_loader, model, optimizer, device)
                val_error = test(val_loader, model, device)
                scheduler.step(val_error)

                if best_val_error is None or val_error <= best_val_error:
                    best_val_error = val_error
                    best_test = test(test_loader, model, device)
                if verbose and (epoch+1) % 50 == 0:
                    print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.4f},'
                      'Val MAE: {:.4f}'.format(epoch, lr, loss, val_error))
                # Break if learning rate is smaller 10**-6.
                if lr < min_lr:
                    break

            test_error.append(best_test)
            if all_std:
                test_complete.append(best_test)
        test_all.append(float(np.array(test_error).mean()))

    if all_std:
        return (np.array(test_all).mean(), np.array(test_all).std(),
                np.array(test_complete).std(),  np.array(times).mean(), np.array(times).std())
    else:
        return (np.array(test_all).mean(), np.array(test_all).std(),  np.array(times).mean(), np.array(times).std())
