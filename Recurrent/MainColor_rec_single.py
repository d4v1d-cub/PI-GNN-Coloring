import random
import torch
import numpy as np
import sys
import os

from time import time


from Utils_rec_single import(get_adjacency_matrix, saver_loss, saver_colorings, get_gnn, 
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


def read_params(fileparams):
    fpar = open(fileparams, "r")
    fpar.readline()
    line = fpar.readline().split()
    fpar.close()
    return int(line[0]), int(line[1]), float(line[2]), float(line[3])

# Specify the problem instance to solve and where to find the dataset(s) here:
q = int(sys.argv[1])
filename = sys.argv[2]
nepochs = int(float(sys.argv[3]))
path_loss = sys.argv[4]
path_colorings = sys.argv[5]

fileparams = sys.argv[6]

randdim, hiddim, dout, lrate = read_params(fileparams)

print("Parameters are:")
print(f'randdim={randdim}   hiddim={hiddim}    dout={dout}    lrate={lrate}')

filename_without_ext = os.path.splitext(os.path.basename(filename))[0]

data_train = SyntheticDataset(filename, q)
print('Dataset ready\n')


hypers = {
        'model': 'GraphSAGE',
        'dim_embedding': randdim + 2 * q,
        'dim_rand_input': randdim,
        'dropout': dout,
        'learning_rate': lrate,
        'hidden_dim': hiddim,
        'seed': SEED_VALUE
}

torch.set_printoptions(threshold=3705,linewidth=160)


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
    net, inputs, optimizer = get_gnn(data_train.chr_n, data_train.fname, 
                                    data_train.nxgraph.number_of_nodes(), hypers, opt_hypers, 
                                    TORCH_DEVICE, TORCH_DTYPE)

    name,losses,hard_losses, prob,best_coloring,best_loss,final_coloring,final_loss,epoch_num = run_gnn_training_early_stop(
            hypers['graph_file'], data_train.nxgraph, data_train.graph, adj_, net, inputs, 
            optimizer, randdim, hypers['number_epochs'], hypers['patience'], hypers['tolerance'], seed=SEED_VALUE)
    
    runtime_gnn = round(time() - t_start, 4)

        # report results
    print(f'GNN runtime: {runtime_gnn}s')

    str_file = f'recurrent_q_{data_train.chr_n}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_filename_{filename_without_ext}'

    loss_filename = "loss_" + str_file
    cols_filename = "coloring_" + str_file
    saver_loss(losses, path_loss, loss_filename, hard_losses)
    saver_colorings(best_coloring, path_colorings, cols_filename, data_train.nx_orig, final_coloring)

except IndexError:
    print(f'index error for graph {data_train.fname}')