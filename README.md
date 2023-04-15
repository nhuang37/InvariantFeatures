# ScalarModel
Implementation of ScalarModel based on the field generators of invariant polynomials for O(d) x S_n

## Dependencies
- Python 3.7+
- Pytorch 1.10+
- [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)


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
- Examples and quick demo: ```Pointcloud_lightweight.ipynb```
- To run experiment on QM7b: ```python main.py```
