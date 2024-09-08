import random
import torch
import numpy as np
import sys
import os
from hyperopt import hp, fmin, tpe


from Utils_orig_single import(get_adjacency_matrix, saver, saver_colorings, get_gnn, 
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
q = int(sys.argv[1])
path_to_files = sys.argv[2]
nepochs = int(float(sys.argv[3]))
model = sys.argv[4]

filenames = []
all_data_train = []

for k, fname in enumerate(os.listdir(path_to_files)):
    print(f'Reading file: "{fname}"')
    filenames.append(fname)
    all_data_train.append(SyntheticDataset(fname, q))

print('Dataset ready\n')


space = (hp.quniform('embdim', 10, 300, 1), hp.quniform('hiddim', 10, 300, 1), 
        hp.uniform('dout', 0, 0.5), hp.uniform('lrate', 0, 0.2))

def train_all(embdim, hiddim, dout, lrate):
    # Sample hyperparameters
    if model == 'GraphConv':  # example with CPU
        hypers = {
            'model': 'GraphConv',   # set either with 'GraphConv' or 'GraphSAGE'. It cannot take other input
            'dim_embedding': embdim,
            'dropout': dout,
            'learning_rate': lrate,
            'hidden_dim': hiddim,
            'seed': SEED_VALUE
        }
    elif model == 'GraphSAGE':                           # example with GPU
        hypers = {
            'model': 'GraphSAGE',
            'dim_embedding': embdim,
            'dropout': dout,
            'learning_rate': lrate,
            'hidden_dim': hiddim,
            'seed': SEED_VALUE
        }
    else:
        print("The model should be GraphConv or GraphSAGE")

    torch.set_printoptions(threshold=3705,linewidth=160)


    # Retrieve known optimizer hypers
    opt_hypers = {
        'lr': hypers.get('learning_rate', None)
    }

    cumul_loss = 0

    for j in range(len(all_data_train)):
        try:
            # Default meta parameters
            solver_hypers = {
                'tolerance': 1e-3,           # Loss must change by more than tolerance, or add towards patience count
                'number_epochs': nepochs,   # Max number training steps
                'patience': 10000,             # Number early stopping triggers before breaking loop
                'graph_file': all_data_train[j].fname,  # Which problem is being solved
                'layer_agg_type': 'mean',    # How aggregate neighbors sampled within graphSAGE
                'number_classes': all_data_train[j].chr_n  #data_train.GetChrom(i)#
            }

                # Combine into a single set
            hypers.update(solver_hypers)

                # Get adjacency matrix for use in calculations
            adj_ = get_adjacency_matrix(all_data_train[j].nxgraph, TORCH_DEVICE, TORCH_DTYPE)

            # See minimal_utils.py for description. Constructs GNN and optimizer objects from given hypers. 
            # Initializes embedding layer to use as initial model input
            net, embed, optimizer = get_gnn(all_data_train[j].chr_n, all_data_train[j].fname, 
                                            all_data_train[j].graph, all_data_train[j].nxgraph.number_of_nodes(), 
                                            hypers, opt_hypers, TORCH_DEVICE, TORCH_DTYPE)

            name,losses,hard_losses,chroms, prob,best_coloring,best_loss,final_coloring,final_loss,epoch_num = run_gnn_training(
                    hypers['graph_file'], all_data_train[j].nxgraph, all_data_train[j].graph, adj_, net, embed, 
                    optimizer, hypers['number_epochs'], hypers['patience'], hypers['tolerance'], seed=SEED_VALUE)
            
            cumul_loss += np.sum(hard_losses)
            
        except IndexError:
            print(f'index error for graph {all_data_train[j].fname}')
    return cumul_loss
    

def objective(args):
    embdim, hiddim, dout, lrate = args
    return train_all(embdim, hiddim, dout, lrate)


best = fmin(objective, space=space, algo=tpe.suggest, max_evals=100)

print(best)


