# InvariantFeatures
Implementation of Invariant Features based on the field generators of invariant polynomials for O(d) x S_n, specifically
- Conjugation-Invariant DeepSet (CI-DS)
- $O(d)$-Invariant DeepSet (OI-DS)

## Dependencies
- Python 3.7+
- Pytorch 1.10+
- [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) version 2.4.x


You can follow the code below to install pytorch-geometric
```
import os
import torch
os.environ['TORCH'] = torch.__version__
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```

## Experiments
- To run experiment on QM7b (Section 5.1): ```python runRegression.py --target [TARGET_INDEX]```
- To run experiment on GW distance regression (Section 5.1): ```Pointcloud_GWdist.ipynb```
  - ModelNet40 data (sub-sampling with 100 points) [here](https://drive.google.com/file/d/1-0p827HBi4ralvP5Z5bPqUXwks5WuXMr/view?usp=sharing)
  - Computed GW distances from ModelNet40, subset of class 2 and 7: [train set](https://drive.google.com/file/d/1uo6uIab5HynUIcyKrvpYxi6_UD9KGHUI/view?usp=sharing), [test set](https://drive.google.com/file/d/1KwAdkJKrMKFeuX3-_9zL9e6UoJgoyIvu/view?usp=sharing)
