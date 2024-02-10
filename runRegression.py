import os
import torch
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, zinc, QM7b, ModelNet
from torch_geometric.utils import to_networkx, homophily, erdos_renyi_graph, stochastic_blockmodel_graph, scatter, loop
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.nn.aggr import DeepSetsAggregation #key to implement DeepSet input (batch_features, batch_index)
from torch_geometric.nn.models import MLP
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, Dropout
import torch_geometric.transforms as T
from torch_geometric.transforms import BaseTransform
import torch.nn.functional as F

from torch.optim import SGD
from tqdm.auto import tqdm

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pickle
import argparse

#step 1: pre-compute fs in dataset
#step 2: modify collate_fn in dataloader to store the edge batch index

def get_fs(data):
  '''
  input: Data (graph) from pytorch geometric dataset
    edge_index=[num_edges]
    edge_attr=[num_edges]
    y=[num_targets]
    x=[num_nodes, 1]

  output: updated DataBatch with the following additional invariant features
   f_d is the set of diagonal edge attributes,
   f_o is the set of off-diagonal (upper) edge attributes,
   f_star is \sum_{i \neq j} X_ii X_ij where X is the n by n edge attribute graph
  '''
  #get f_d, f_o
  loop_mask = data.edge_index[0] == data.edge_index[1]
  data.f_d = data.edge_attr[loop_mask].unsqueeze(1)
  data.f_o = data.edge_attr[~loop_mask].unsqueeze(1)

  #get f_star
  sum_fo = scatter(data.edge_attr, data.edge_index[1], dim=0) #sum over rows
  data.f_star = (data.f_d.squeeze(1) @ ( sum_fo - data.f_d.squeeze(1))).reshape((1,1))
  return data

#The kernel trick that projects a scalar to higher-dimensional space
#src: https://arxiv.org/pdf/1305.7074.pdf, Appendix B
def binary_expansion(f, num_radial, theta=1):
  '''
  input: f (bs x 1); num_radial - number of basis expansion
  output: phi(f) (bs x num_radial)
  phi(f) = [..., sigmoid(f-theta/theta), sigmoid(f/theta), sigmoid(f+theta/thera),...]
  '''
  bs = f.shape[0]
  out = torch.zeros((bs, num_radial))
  max_val = (num_radial - 1)//2
  offsets = np.arange(start=-max_val, stop=max_val+1)
  for i, offset in enumerate(offsets):
    out[:, i:(i+1)] = F.sigmoid( (f - theta*offset) / theta )
  return out

def get_binary_expansion(data, num_radial=100, theta=1):
  '''
  Apply binary_expansion for all fs
  try: num_radial \in {100, 1000}
  '''
  #print(f"num_basis={num_radial}")
  data.f_d = binary_expansion(data.f_d, num_radial, theta)
  data.f_o = binary_expansion(data.f_o, num_radial, theta)
  data.f_star = binary_expansion(data.f_star, num_radial, theta)
  return data

def train(train_loader, model, optimizer, device):
    model.train()
    loss_all = 0

    lf = torch.nn.L1Loss()

    for data in train_loader:
        #print(data)
        data = data.to(device)
        optimizer.zero_grad()
        loss = lf(model(data), data.y)

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return (loss_all / len(train_loader.dataset))


def test(loader, model, device, per_target=False):
    model.eval()
    out_dim = model.MLP_out.channel_list[-1]
    if per_target:
      error_pt = torch.zeros(out_dim)
    error = 0

    for data in loader:
        data = data.to(device)
        error_multi = (model(data) - data.y).abs()
        if per_target:
          error_pt += error_multi.sum(axis=0).cpu()

        error += error_multi.sum().item()
    if per_target:
      return error / len(loader.dataset), error_pt / len(loader.dataset)
    else:
      return error / len(loader.dataset), None


# 10-CV for NN training and hyperparameter selection.
def nn_evaluation(dataset, hid_dim, out_dim, max_num_epochs=200, batch_size=128, start_lr=0.01, min_lr = 0.000001, factor=0.5, patience=50,
                       num_repetitions=5, verbose=True, dropout=0, per_target=False):
    #reproducibility
    torch.manual_seed(0)

    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_all = []

    # Input dim
    input_dim = dataset.data.f_star.shape[1]

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

            # Prepare batching. (follow_batch: batchify both f_d, f_o)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, follow_batch=['f_d', 'f_o'])
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, follow_batch=['f_d', 'f_o'])
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, follow_batch=['f_d', 'f_o']) #reproducible

            # Collect val. and test acc. over all hyperparameter combinations.
            # Setup model.
            model = ScalarModel(hid_dim, out_dim, dropout, input_dim).to(device)
            #model = ScalarModel_Bessel(hid_dim, out_dim, dropout).to(device)
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=factor, patience=patience,
                                                                    min_lr=min_lr)
            for epoch in range(1, max_num_epochs + 1):
                lr = scheduler.optimizer.param_groups[0]['lr']
                loss = train(train_loader, model, optimizer, device)
                val_error, val_pt = test(val_loader, model, device, per_target)
                scheduler.step(val_error)

                if best_val_error is None or val_error <= best_val_error:
                    best_val_error = val_error
                    best_test, test_pt = test(test_loader, model, device, per_target)
                if verbose and (epoch+1) % 50 == 0:
                    print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.4f},'
                      'Val MAE: {:.4f}'.format(epoch, lr, loss, val_error))
                # Break if learning rate is smaller 10**-6.
                if lr < min_lr:
                    break
            if per_target:
                print(f"per_target Val MAE = {val_pt}")
            test_error.append(best_test)
            print(f"Finish run with best_test={best_test:.4f}")

        test_all.append(test_error)
    
    test_all = np.array(test_all)

    return test_all

class ExtractTarget(BaseTransform):
    def __init__(self, target):
        self.target = target
    def forward(self, data):
        data.y = data.y[:, self.target:(self.target+1)]
        return data

class ScalarModel(torch.nn.Module):
    def __init__(self, hid_dim, out_dim, dropout=0, input_dim=1):
        super(ScalarModel, self).__init__()
        #Deepset for diagonal scalars
        phi_d = MLP([input_dim,hid_dim, hid_dim*2], dropout=[dropout]*2)
        rho_d = MLP([hid_dim*2, hid_dim], dropout=[dropout]*1) #MLP([hid_dim*2, hid_dim, hid_dim], dropout=[dropout]*2) #
        self.deepset_d = DeepSetsAggregation(local_nn=phi_d, global_nn=rho_d)
        #Deepset for off-diagonal scalars
        phi_o = MLP([input_dim,hid_dim, hid_dim*2], dropout=[dropout]*2)
        rho_o = MLP([hid_dim*2, hid_dim], dropout=[dropout]*1) #MLP([hid_dim*2, hid_dim, hid_dim], dropout=[dropout]*2)
        self.deepset_o = DeepSetsAggregation(local_nn=phi_o, global_nn=rho_o)
        #MLP_s for f_star
        self.MLP_s = MLP([input_dim,hid_dim, hid_dim], dropout=[dropout]*2)  #MLP([1,hid_dim*2, hid_dim], dropout=[dropout]*2) #
        #MLP for (Deepset(f_d), Deepset(f_o), f_star)
        self.MLP_out = MLP([hid_dim*3, hid_dim, out_dim], dropout=[dropout]*2)

    def reset_parameters(self):
        for net in [self.deepset_d, self.deepset_o, self.MLP_s, self.MLP_out]:
        #for net in [self.deepset_d, self.deepset_o, self.MLP_out]:
            net.reset_parameters()

    def forward(self, data):
        out_d = self.deepset_d(data.f_d, data.f_d_batch) # bs x hid_dim
        out_o = self.deepset_o(data.f_o, data.f_o_batch) # bs x hid_dim
        out_star = self.MLP_s(data.f_star) #bs x hid_dim
        #concat and output final embedding
        out = self.MLP_out(torch.concat([out_d, out_o, out_star], dim=-1)) # bs x hid_dim*3 -> bs x out_dim
        return out


def main(args):
	dataset = QM7b(args.path, pre_transform=T.Compose([get_fs, get_binary_expansion]),
               transform=T.Compose([ExtractTarget(args.target)]))
	dataset.name = "QM7b"
	#loader = DataLoader(dataset, batch_size=args.bs, follow_batch=['f_d', 'f_o']) 
	test_all = nn_evaluation(dataset, args.hid_dim, out_dim =1, 
									batch_size=args.bs, max_num_epochs=1000,
                                  start_lr=0.02, min_lr = 0.0000001, factor=0.8, patience=20,
                       num_repetitions=1, verbose=False, dropout=0)
	file_name = os.path.join(args.result_path, "target_" + str(args.target) + ".pkl")
	pickle.dump(test_all, open(file_name, "wb" ))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molecule Invariant Regression")
    parser.add_argument("--target", type=int, default=0, help="the target variable index")   
    parser.add_argument("--bs", type=int, default=128, help="batch size")
    parser.add_argument("--hid_dim", type=int, default=512, help="hidden dimension in NN")   
    parser.add_argument("--path", type=str, default="./QM7b/dataset_basis", help="dataset folder path")
    parser.add_argument("--result_path", type=str, default="./results/", help="result folder path")

    args = parser.parse_args()

    print(args)
    main(args)













 

