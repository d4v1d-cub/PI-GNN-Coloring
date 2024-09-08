import random
import torch
import numpy as np
import sys
import os

from time import time


from Utils_orig_single import(get_adjacency_matrix, saver, saver_colorings, get_gnn, 
                               run_gnn_training_early_stop, SyntheticDataset)


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
q = int(sys.argv[1])
filename = sys.argv[2]
nepochs = int(float(sys.argv[3]))
path_loss = sys.argv[4]
path_chroms = sys.argv[5]
path_colorings = sys.argv[6]
model = sys.argv[7]
emb_dim = int(sys.argv[8])
hid_dim = int(sys.argv[9])

filename_without_ext = os.path.splitext(os.path.basename(filename))[0]

data_train = SyntheticDataset(filename, q)
print('Dataset ready\n')


# Sample hyperparameters
if model == 'GraphConv':  # example with CPU
    hypers = {
        'model': 'GraphConv',   # set either with 'GraphConv' or 'GraphSAGE'. It cannot take other input
        'dim_embedding': emb_dim,
        'dropout': 0.1,
        'learning_rate': 0.0001,
        'hidden_dim': hid_dim,
        'seed': SEED_VALUE
    }
elif model == 'GraphSAGE':                           # example with GPU
    hypers = {
        'model': 'GraphSAGE',
        'dim_embedding': emb_dim,
        'dropout': 0.3784,
        'learning_rate': 0.02988,
        'hidden_dim': hid_dim,
        'seed': SEED_VALUE
    }
else:
    print("The model should be GraphConv or GraphSAGE")

torch.set_printoptions(threshold=3705,linewidth=160)


probs=[]
best_colorings=[]
best_losses=[] 
final_colorings=[] 
final_losses=[] 
epoch_nums = []
names=[]
lossesses=[]
hard_lossesses=[]
chromss=[]


# Retrieve known optimizer hypers
opt_hypers = {
    'lr': hypers.get('learning_rate', None)
}

t_start = time()
try:
    # Default meta parameters
    solver_hypers = {
        'tolerance': 1e-3,           # Loss must change by more than tolerance, or add towards patience count
        'number_epochs': nepochs,   # Max number training steps
        'patience': 10000,             # Number early stopping triggers before breaking loop
        'graph_file': data_train.fname,  # Which problem is being solved
        'layer_agg_type': 'mean',    # How aggregate neighbors sampled within graphSAGE
        'number_classes': data_train.chr_n  #data_train.GetChrom(i)#
    }

        # Combine into a single set
    hypers.update(solver_hypers)

        # Get adjacency matrix for use in calculations
    adj_ = get_adjacency_matrix(data_train.nxgraph, TORCH_DEVICE, TORCH_DTYPE)

    # See minimal_utils.py for description. Constructs GNN and optimizer objects from given hypers. 
    # Initializes embedding layer to use as initial model input
    net, embed, optimizer = get_gnn(data_train.chr_n, data_train.fname, data_train.graph, 
                                    data_train.nxgraph.number_of_nodes(), hypers, opt_hypers, 
                                    TORCH_DEVICE, TORCH_DTYPE)

    name,losses,hard_losses,chroms, prob,best_coloring,best_loss,final_coloring,final_loss,epoch_num = run_gnn_training_early_stop(
            hypers['graph_file'], data_train.nxgraph, data_train.graph, adj_, net, embed, 
            optimizer, hypers['number_epochs'], hypers['patience'], hypers['tolerance'], seed=SEED_VALUE)
    
    runtime_gnn = round(time() - t_start, 4)

        # report results
    print(f'GNN runtime: {runtime_gnn}s')

    str_file = f'q_{data_train.chr_n}_model_{model}_embdim_{emb_dim}_hidim_{hid_dim}_filename_{filename_without_ext}'

    loss_filename = "loss_" + str_file
    cols_filename = "coloring_" + str_file
    saver(losses, path_loss, loss_filename, "loss", hard_losses)
    saver_colorings(best_coloring, path_colorings, cols_filename, data_train.nx_orig, final_coloring)

except IndexError:
    print(f'index error for graph {data_train.fname}')