import torch
import sys
import os
from torch.profiler import profile, record_function, ProfilerActivity

from time import time


from Utils_rec_single import(get_adjacency_matrix, get_gnn, SyntheticDataset)


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
path_prof = sys.argv[4]

fileparams = sys.argv[5]

init_seed = int(sys.argv[6])

randdim, hiddim, dout, lrate = read_params(fileparams)

print("Parameters are:")
print(f'randdim={randdim}   hiddim={hiddim}    dout={dout}    lrate={lrate}')

filename_without_ext = os.path.splitext(os.path.basename(filename))[0]

data_train = SyntheticDataset(filename, q)
print('Dataset ready\n')


hypers = {
        'model': 'GraphSAGE',
        'dim_embedding': randdim + 2 * data_train.chr_n,
        'dim_rand_input': randdim,
        'dropout': dout,
        'learning_rate': lrate,
        'hidden_dim': hiddim,
        'seed': init_seed
}

torch.set_printoptions(threshold=3705,linewidth=160)


# Retrieve known optimizer hypers
opt_hypers = {
    'lr': hypers.get('learning_rate', None)
}

t_start = time()

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

cond = True
min_cost = data_train.nxgraph.number_of_edges()
print("\nTrying seed=", hypers['seed'])
net, embed, optimizer = get_gnn(data_train.chr_n, data_train.fname, 
                                data_train.nxgraph.number_of_nodes(), hypers, opt_hypers, 
                                TORCH_DEVICE, TORCH_DTYPE)

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("forward step"):
        net(data_train.graph, embed.weight)


foutname = f'Profiler_recurrent_q_{data_train.chr_n}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_nep_{nepochs}_filename_{filename_without_ext}'

fname = f'{path_prof}/{foutname}.txt'

fout = open(fname, "w")

fout.write(prof.key_averages().table(sort_by="cuda_time_total"))

fout.close()