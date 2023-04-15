import os
import os.path as osp 
import argparse
import torch_geometric.transforms as T
from torch_geometric.datasets import QM7b
import pickle

from architecture import ScalarModel
from train import get_fs, nn_evaluation


def main(args):

    #get dataset

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets')
    
    results = []

    for target in range(14):

        class Univariate(object):
            def __call__(self, data):
                # Specify target.
                data.y = data.y[:, target:(target+1)]
                return data

        #get univariate dataset
        dataset = QM7b(path, transform=T.Compose([Univariate()]))

        #train
        mae, s_1, s_2, time, t_std = nn_evaluation(dataset, args.hid_dim, out_dim=1, 
                                                  num_repetitions=args.num_reps, start_lr=args.lr, max_num_epochs=args.n_epochs)
        print(str(mae) + " " + str(s_1) + " " + str(s_2) + " " + str(time) + " " + str(t_std))
        results.append(str(mae) + " " + str(s_1) + " " + str(s_2) + " " + str(time) + " " + str(t_std))

        #save at every univariate trial
        file_name = os.path.join(args.result_path, + str(args.hid_dim) + ".pkl")
        pickle.dump(results, open(file_name, "wb" ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScalarModel on QM7b invariant regression")
    parser.add_argument("--num_reps", type=int, default=5, help="number of experiment runs")
    parser.add_argument("--hid_dim", type=int, default=16, help="hidden dimension")
    parser.add_argument("--lr", type=float, default=0.02, help="starting learning rate")
    parser.add_argument("--n_epochs", type=int, default=300, help="maximum number of epochs")
    parser.add_argument("--result_path", type=str, default="./results/", help="dataset folder path")

    args = parser.parse_args()

    print(args)
    main(args)