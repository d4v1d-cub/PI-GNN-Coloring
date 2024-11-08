import torch
import sys
import numpy as np

from time import time


from Utils_rec_parallel import(get_adjacency_matrix, saver_loss, saver_colorings, 
                                         saver_colorings_final, saver_others, get_gnn, 
                                         run_gnn_training, get_full_colors, get_graph_index, 
                                         SyntheticDataset)


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
folder = sys.argv[2]
nepochs = int(sys.argv[3])
path_loss = sys.argv[4]
path_colorings = sys.argv[5]
path_others = sys.argv[6]

fileparams = sys.argv[7]

init_seed = int(sys.argv[8])
ntries = int(sys.argv[9])

randdim, hiddim, dout, lrate = read_params(fileparams)

print("Parameters are:")
print(f'randdim={randdim}   hiddim={hiddim}    dout={dout}    lrate={lrate}')


hypers = {
        'model': 'GraphSAGE',
        'dim_embedding': randdim + 2 * q,
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



# Default meta parameters
solver_hypers = {
    'tolerance': 1e-3,           # Loss must change by more than tolerance, or add towards patience count
    'number_epochs': nepochs,   # Max number training steps
    'patience': 10000,             # Number early stopping triggers before breaking loop
    'layer_agg_type': 'mean',    # How aggregate neighbors sampled within graphSAGE
    'number_classes': q  #data_train.GetChrom(i)#
}

        # Combine into a single set
hypers.update(solver_hypers)

        # Get adjacency matrix for use in calculations

graph_index, ngraphs = get_graph_index(folder)
best_costs = np.full(ngraphs, np.inf)
best_colorings = [[] for _ in range(ngraphs)]
best_seed = np.full(ngraphs, np.inf)

str_file = f'recurrent_parallel_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntries}_nep_{nepochs}'

name_start_col = "coloring_" + str_file

cond = True
times = []
while hypers['seed'] < init_seed + ntries and cond:
    t_start = time()
    data_train = SyntheticDataset(folder)
    print('Dataset ready\n')
    all_adj_ = get_adjacency_matrix(data_train.nx_clean_all, TORCH_DEVICE, TORCH_DTYPE)

    print("\nTrying seed=", hypers['seed'])
    net, embed, optimizer = get_gnn(data_train.graph.num_nodes(), hypers, opt_hypers, 
                                    TORCH_DEVICE, TORCH_DTYPE)

    losses, final_coloring, final_loss, final_cost = run_gnn_training(
            data_train.nx_clean_all, data_train.graph, all_adj_, net, embed, 
            optimizer, randdim, hypers['number_epochs'], hypers['patience'], hypers['tolerance'], hypers['seed'])
    
    full_colors = get_full_colors(final_coloring, data_train.nx_orig_all, 
                                  data_train.graph.batch_num_nodes())
    saver_colorings(full_colors, best_colorings, best_costs, path_colorings, name_start_col, 
                    data_train, graph_index, hypers['seed'], best_seed)

    if final_cost < 0.5:
        cond = False
        print("Success with all graphs at seed=", hypers['seed'])
    runtime_gnn = round(time() - t_start, 4)
    times.append(runtime_gnn)

    hypers['seed'] += 1

        # report results
print(f'GNN runtime: {sum(times)}s')

loss_filename = "loss_" + str_file
others_filename = "others_" + str_file
saver_loss(losses, path_loss, loss_filename)
saver_colorings_final(best_colorings, best_costs, path_colorings, name_start_col, 
                      data_train, graph_index, best_seed)
others = [hypers['seed'] - 1]
for i in range(len(times)):
    others.append(times[i])
saver_others(others, path_others, others_filename)