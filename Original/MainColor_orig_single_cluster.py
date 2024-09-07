import random
import torch
import numpy as np
import sys
import os

from time import time


from Utils_orig_single_cluster import(get_adjacency_matrix, saver, saver_colorings, get_gnn, 
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

n_data=10000
data_train = SyntheticDataset(filename, n_data, q)
print('Dataset ready\n')

dict_graphs={data_train.fnames[i] : [data_train.graphs[i], data_train.nxgraphs[i]] for i in range(len(data_train.nxgraphs))}

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
        name,losses,hard_losses,chroms, prob,best_coloring,best_loss,final_coloring,final_loss,epoch_num = run_gnn_training_early_stop(
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

        str_file = f'q_{data_train.all_chr_ns[i]}_model_{model}_embdim_{emb_dim}_hidim_{hid_dim}_filename_{filename_without_ext}'

        loss_filename = "loss_" + str_file
        chroms_filename = "chroms_" + str_file
        cols_filename = "coloring_" + str_file
        saver(lossesses[i], path_loss, loss_filename, "loss", hard_lossesses[i])
        saver(chromss[i], path_chroms, chroms_filename, "chroma")
        saver_colorings(best_colorings[i], path_colorings, cols_filename, data_train.nx_orig[i], final_colorings[i])

    except IndexError:
        print(f'index error for graph {data_train.fnames[i]}')