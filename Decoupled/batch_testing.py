import random
import torch
import warnings
import numpy as np
import networkx as nx
import os
import torch.nn as nn
import torch_geometric
dev = torch.device('cuda')
from torch_geometric.loader import DataLoader
from itertools import chain
import torch_geometric.utils as ut


from time import time

# local imports: we load a few general utility functions from `minimal_utils.py`.
from Utils_new_batch import(get_gnn, loss_func_color_hard,
                      run_gnn_training, run_gnn_testing, SyntheticDataset)#, SyntheticDatasetTest)
# fix seed to ensure consistent results
SEED_VALUE = 0
random.seed(SEED_VALUE)        # seed python RNG
np.random.seed(SEED_VALUE)     # seed global NumPy RNG
torch.manual_seed(SEED_VALUE)  # seed torch RNG

# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')


TePath="/raid/data/silveste/data/500nods/7chrom/30.000000connect/Test" #"/raid/data/silveste/data/500nods/5chrom/experim/13.100000connect/Test"
n_data=300
data_test = SyntheticDataset(TePath, n_data).process()#Test
print('TestDataset pronto\n')

hypers = {
	'model': 'GraphSAGE',
	'dim_embedding': 77,
	'dropout': 0.3784,
	'learning_rate': 0.02988,
	'hidden_dim': 32,
	'seed': SEED_VALUE,
	'tolerance': 1e-3,           # Loss must change by more than tolerance, or add towards patience count
	'layer_agg_type': 'mean',    # How aggregate neighbors sampled within graphSAGE
	'number_epochs': int(5e4),   # Max number training steps
	'patience': 10000             # Number early stopping triggers before breaking loop
}

opt_hypers = {
    'lr': hypers.get('learning_rate', None)
}

####   LOAD MODELS   ####
modeload={}
for i in set (data_test.keys()):
    net, embeds = get_gnn(hypers, opt_hypers, i, TORCH_DEVICE, TORCH_DTYPE)
    checkpoint = torch.load(f'./SaveModel{i}permutated')
    net.load_state_dict(checkpoint['state_dict'])
    for n in embeds.keys():
        embeds[n].load_state_dict(checkpoint[f'embed_dict{n}'])
    modeload.update({i:(net, embeds)})


####   TESTING FOR COMPETING WITH SA   ####
test_dataloader={}
Sat=0
t_start = time()
for i in set(data_test.keys()):
    test_dataloader.update({ i : DataLoader(data_test[i], batch_size=1, shuffle=False, drop_last=True) })
    with torch.no_grad():
        for teBatchJ, test_batch in enumerate(test_dataloader[i]):
            test_batch.to(dev)
            losses=[]
            modeload[i][0].eval()
            net, embeds = modeload[i]
            embedde=torch.vstack(tuple([embeds[nnods.item()].weight for nnods in test_batch.nnods]))
            name, Tcost_hard, prob, coloring = run_gnn_testing(
                        test_batch, net, embedde, hypers['tolerance'], seed=SEED_VALUE)
            if Tcost_hard==0:
                Sat+=1
            print(f'TESTING: chr_n {i} | HardCost: {Tcost_hard.item():.1f} \
| batch {teBatchJ} | #nodes: {test_batch.num_nodes} | GPU memory {torch.cuda.memory_allocated(device=dev)}')
print(f'{100*Sat/(len(test_dataloader[i])):.3f}%')
print(f'{round(time() - t_start, 4)}')