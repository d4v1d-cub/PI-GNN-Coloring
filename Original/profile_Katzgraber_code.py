import random
import torch
import warnings
import numpy as np
import sys
import os
import dgl
from torch.profiler import profile, record_function, ProfilerActivity

from time import time

warnings.filterwarnings('ignore')


# local imports: we load a few general utility functions from `minimal_utils.py`.

from minimal_utils import (
    get_adjacency_matrix, get_gnn, run_gnn_training,
    build_graph_from_color_file, chromatic_numbers
)


# fix seed to ensure consistent results
SEED_VALUE = 0
random.seed(SEED_VALUE)        # seed python RNG
np.random.seed(SEED_VALUE)     # seed global NumPy RNG
torch.manual_seed(SEED_VALUE)  # seed torch RNG

# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')

# Specify the problem instance to solve and where to find the dataset(s) here:

problem_file = sys.argv[1]

input_parent = sys.argv[2]

nepochs = int(float(sys.argv[3]))

path_prof = sys.argv[4]

hypers = {
        'model': 'GraphSAGE',   # set either with 'GraphConv' or 'GraphSAGE'. It cannot take other input
        'dim_embedding': 112,
        'dropout': 0.1571,
        'learning_rate': 0.14426,
        'hidden_dim': 199,
        'seed': SEED_VALUE
}

# Default meta parameters
solver_hypers = {
    'tolerance': 1e-3,           # Loss must change by more than tolerance, or add towards patience count
    'number_epochs': nepochs,   # Max number training steps
    'patience': 500,             # Number early stopping triggers before breaking loop
    'graph_file': problem_file,  # Which problem is being solved
    'layer_agg_type': 'mean',    # How aggregate neighbors sampled within graphSAGE
    'number_classes': chromatic_numbers[problem_file]
}

# Combine into a single set
hypers.update(solver_hypers)

# Establish full input location
input_fpath = os.path.join(input_parent, problem_file)

# Load in graph
nx_graph = build_graph_from_color_file(input_fpath, node_offset=-1, parent_fpath='')

# Get DGL graph from networkx graph
# Ensure relevant objects are placed onto proper torch device
dgl_graph = dgl.from_networkx(nx_graph)
dgl_graph = dgl_graph.to(TORCH_DEVICE)


# Retrieve known optimizer hypers
opt_hypers = {
    'lr': hypers.get('learning_rate', None)
}

# Get adjacency matrix for use in calculations
adj_ = get_adjacency_matrix(nx_graph, TORCH_DEVICE, TORCH_DTYPE)

# See minimal_utils.py for description. Constructs GNN and optimizer objects from given hypers. 
# Initializes embedding layer to use as initial model input
net, embed, optimizer = get_gnn(dgl_graph, nx_graph.number_of_nodes(), hypers, opt_hypers, TORCH_DEVICE, TORCH_DTYPE)


with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function(f'{nepochs} epochs'):
        t_start = time()
        run_gnn_training(nx_graph, dgl_graph, adj_, net, embed, optimizer, hypers['number_epochs'], 
                         hypers['patience'], hypers['tolerance'], seed=SEED_VALUE)
        runtime_gnn = round(time() - t_start, 4)


filename_without_ext = os.path.splitext(os.path.basename(problem_file))[0]
foutname = f'Profiler_Katzgraber_{TORCH_DEVICE}_q_{chromatic_numbers[problem_file]}_nep_{nepochs}_filename_{filename_without_ext}'

fname = f'{path_prof}/{foutname}.txt'

fout = open(fname, "w")

fout.write(prof.key_averages().table(sort_by="cuda_time_total"))

fout.close()

print(f'GNN runtime: {runtime_gnn}s')