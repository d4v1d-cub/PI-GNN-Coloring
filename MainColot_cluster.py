import random
import torch
import warnings
import numpy as np
import networkx as nx
import os
import dgl

from time import time


from Utils_orig import(get_adjacency_matrix, saver, plotter, plotter_g, get_gnn, 
                      loss_func_color_hard, run_gnn_training, SyntheticDataset, chromatic_numbers)


# fix seed to ensure consistent results
SEED_VALUE = 0
random.seed(SEED_VALUE)        # seed python RNG
np.random.seed(SEED_VALUE)     # seed global NumPy RNG
torch.manual_seed(SEED_VALUE)  # seed torch RNG

# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')