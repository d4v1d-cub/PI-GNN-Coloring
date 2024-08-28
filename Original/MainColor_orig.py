import random
import torch
import numpy as np
import sys

from time import time


from Utils_orig_cluster import(get_adjacency_matrix, saver, saver_colorings, get_gnn, 
                               run_gnn_training, SyntheticDataset)


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
q = 3
TrPath = "Original/data/train"
nepochs = int(1e5)
path_loss = "Original/losses"
path_chroms = "Original/chroms"
path_colorings = "Original/colorings"
n_data=10000
data_train = SyntheticDataset(TrPath, n_data, q)
print('Dataset ready\n')

dict_graphs={data_train.fnames[i] : [data_train.graphs[i], data_train.nxgraphs[i]] for i in range(len(data_train.nxgraphs))}

# Sample hyperparameters
if TORCH_DEVICE.type == 'cpu':  # example with CPU
    hypers = {
        'model': 'GraphConv',   # set either with 'GraphConv' or 'GraphSAGE'. It cannot take other input
        'dim_embedding': 64,
        'dropout': 0.1,
        'learning_rate': 0.0001,
        'hidden_dim': 64,
        'seed': SEED_VALUE
    }
else:                           # example with GPU
    hypers = {
        'model': 'GraphSAGE',
        'dim_embedding': 77,
        'dropout': 0.3784,
        'learning_rate': 0.02988,
        'hidden_dim': 32,
        'seed': SEED_VALUE
    }

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
for i in range(len(data_train)):
    try:
        # Default meta parameters
        solver_hypers = {
            'tolerance': 1e-3,           # Loss must change by more than tolerance, or add towards patience count
            'number_epochs': nepochs,   # Max number training steps
            'patience': 10000,             # Number early stopping triggers before breaking loop
            'graph_file': data_train.fnames[i],  # Which problem is being solved
            'layer_agg_type': 'mean',    # How aggregate neighbors sampled within graphSAGE
            'number_classes': data_train.all_chr_ns[i]#data_train.GetChrom(i)#
        }

        # Combine into a single set
        hypers.update(solver_hypers)

        # Get adjacency matrix for use in calculations
        adj_ = get_adjacency_matrix(data_train.nxgraphs[i], TORCH_DEVICE, TORCH_DTYPE)

        # See minimal_utils.py for description. Constructs GNN and optimizer objects from given hypers. 
        # Initializes embedding layer to use as initial model input
        net, embed, optimizer = get_gnn(data_train.all_chr_ns[i], data_train.fnames[i], 
                                        data_train.graphs[i], data_train.nxgraphs[i].number_of_nodes(), 
                                        hypers, opt_hypers, TORCH_DEVICE, TORCH_DTYPE)

        #print(len(nx_graph.nodes()))
        name,losses,hard_losses,chroms, prob,best_coloring,best_loss,final_coloring,final_loss,epoch_num = run_gnn_training(
            hypers['graph_file'], data_train.nxgraphs[i], data_train.graphs[i], adj_, net, embed, 
            optimizer, hypers['number_epochs'], hypers['patience'], hypers['tolerance'], seed=SEED_VALUE)
        names.append(name) #all this cuz sometimes can't be processesd by net and triggers exception, cuzing different lengths.
        probs.append(prob)
        lossesses.append(losses)
        hard_lossesses.append(hard_losses)
        chromss.append(chroms)
        best_colorings.append(best_coloring)
        best_losses.append(best_loss)
        final_colorings.append(final_coloring)
        final_losses.append(final_loss)
        epoch_nums.append(epoch_num)

        runtime_gnn = round(time() - t_start, 4)

        # report results
        print(f'GNN runtime: {runtime_gnn}s')
    except IndexError:
        print(f'index error for graph {data_train.fnames[i]}')


for i in range(len(names)):
    saver(lossesses[i], path_loss, names[i], "loss", hard_lossesses[i])
    saver(chromss[i], path_chroms, names[i], "chroma")
    saver_colorings(best_colorings[i], path_colorings, names[i], data_train.nx_orig[i], final_colorings[i])