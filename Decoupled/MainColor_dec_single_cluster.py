#Substitute TrPath and TePath in lines 38, 39 with right path.
import random
import torch
import numpy as np
import os
from torch_geometric.loader import DataLoader
from itertools import chain
import torch_geometric.utils as ut
from time import time
import sys



# local imports: we load a few general utility functions from `minimal_utils.py`.
from Utils_dec_single import(get_gnn, run_gnn_training, run_gnn_testing, SyntheticDataset)

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

filename=sys.argv[2]
test_every=int(sys.argv[3])

save_path = sys.argv[4]
models_path = sys.argv[5]
coloring_path = sys.argv[6]
nepochs=int(float(sys.argv[7]))
batch_size = int(sys.argv[8])
model = sys.argv[9]
emb_dim = int(sys.argv[10])
hid_dim = int(sys.argv[11])
print_every = 1
test_every = 50

filename_without_ext = os.path.splitext(os.path.basename(filename))[0]

n_data=10000
data_train = SyntheticDataset(filename, n_data, q).process()
print('TrainDataset ready\n')
data_test = SyntheticDataset(filename, n_data, q).process()#Test
print('TestDataset ready\n')

for i in set(data_train.keys()):
    print(f'{i}: {len(data_train[i])}')
	# Sample hyperparameters
if model == 'GraphConv':  # example with CPU
    hypers = {
        'model': 'GraphConv',   # set either with 'GraphConv' or 'GraphSAGE'. It cannot take other input
        'dim_embedding': emb_dim,
        'dropout': 0.1,
        'learning_rate': 0.0001,
        'hidden_dim': hid_dim,
        'seed': SEED_VALUE,
        'tolerance': 1e-3,           # Loss must change by more than tolerance, or add towards patience count
        'number_epochs': nepochs,   # Max number training steps
        'layer_agg_type': 'mean',    # How aggregate neighbors sampled within graphSAGE
        'patience': 10000             # Number early stopping triggers before breaking loop
    }
elif model == 'GraphSAGE':                           # example with GPU
    hypers = {
        'model': 'GraphSAGE',
        'dim_embedding': emb_dim,
        'dropout': 0.3784,
        'learning_rate': 0.02988,
        'hidden_dim': hid_dim,
        'seed': SEED_VALUE,
        'tolerance': 1e-3,           # Loss must change by more than tolerance, or add towards patience count
        'layer_agg_type': 'mean',    # How aggregate neighbors sampled within graphSAGE
        'number_epochs': nepochs,   # Max number training steps
        'patience': 10000             # Number early stopping triggers before breaking loop
    }
else:
    print("The model should be GraphConv or GraphSage")
    


# Retrieve known optimizer hypers
opt_hypers = {
    'lr': hypers.get('learning_rate', None)
}

probs=[]

epoch_nums = []
names=[]

chromss=[]

#Function to either create new key:val instance or to append new val to existing key
#Aim: dictionary of [chi1:[loss_epoch1,...,loss_epochN],...,chiM:[loss_epoch1,...,loss_epochN]]
def DictoListUpdate(dicto, i, val):
    if i not in dicto.keys():
        dicto.update({i:[val]})
    else: 
        dicto[i].append(val)
    return dicto

torch.set_printoptions(threshold=3705,linewidth=260)

models={}
train_dataloader={}
test_dataloader={}

losses={}
MeanEpochLoss={}
hard_losses={}
TestLosses={}
TestMeanEpochLoss={}
TestHardMeanEpochLoss={}

final_colorings={}
final_losses={}
final_hard_losses={}
best_colorings={}


nnodes_train = []
for i in set(data_train.keys()):
    for j in range(len(data_train[i])):
        nnodes_train.append(data_train[i][j]['nnods'])

nnodes_test = []
for i in set(data_train.keys()):
    for j in range(len(data_train[i])):
        nnodes_test.append(data_train[i][j]['nnods'])

margin = 0.1 # fraction of the minimum to substract, or of the maximum to add
nnodes_min = min(min(nnodes_train), min(nnodes_test))
nnodes_min = int(nnodes_min *(1 - margin))
nnodes_max = max(max(nnodes_train), max(nnodes_test))
nnodes_max = int(nnodes_max *(1 + margin))

#### MODELS INITIALIZATION, BEHEADED NETWORK AND EMBEDDINGS ####
for i in set(data_train.keys()): # data_train.keys() are the dataset's chromatic numbers.
    if i!=8:
        net, embeds = get_gnn(hypers, i, TORCH_DEVICE, TORCH_DTYPE, nnodes_min, nnodes_max)
        models.update({i:(net, embeds)})
        train_dataloader.update({ i : DataLoader(data_train[i], batch_size=batch_size, shuffle=True, drop_last=True) })
        test_dataloader.update({ i : DataLoader(data_test[i], batch_size=batch_size, shuffle=False, drop_last=True) })

#### SAVER CALLABLE DURING TRRAINING ####
def InstaSaver(path, lista3, name):
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    file1 = open(f'{path}/{name}.txt', "a")  # write mode
    file1.write(f'{[ lista3[0], float("{0:.4f}".format(lista3[1])) ]},')
    file1.close()

def ModelSaver(path, models_i, i, name):
    state={'state_dict': models_i[0].state_dict()}
    for n in models_i[1].keys():
        state.update({f'embed_dict{n}':models_i[1][n].state_dict()})
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    torch.save(state, f'{path}/{name}')

def print_final_colorings(path, colorings, i, filename):
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    fout = open(f'{path}/{filename}', "a")
    if (len(colorings[i][1]) == 1):
        list_cols=colorings[i][0].tolist()
        fout.write(f'{i}\t{colorings[i][1][0]}\t{list_cols}\n')
    else:
        for j in range(len(colorings[i][1])):
            fout.write(f'{i}\t{colorings[i][1][j]}\t{list_cols[j]}\n')
    fout.close()

#### SAVING PATHS ####


for i in set(data_train.keys()): #aka: for every subdataset with a certain chi
    #i=5
    # Tracking
    best_cost = torch.tensor(float('Inf'))  # high initialization
    best_loss = torch.tensor(float('Inf'))
    best_coloring = None
    t_start = time()
    net, embeds = models[i]
    for epoch in range(hypers['number_epochs']):
        #if epoch<=15:
        #    test_every=1
        models[i][0].train()
        EpochCumLossPerBatch=0
        EpochCumHardLossPerBatch=0
        EpochCumLossPerBatchEval=0
        EpochCumHardLossPerBatchEval=0
        #### Training ####
        for batchJ, batch in enumerate(train_dataloader[i]):
            batch.to(TORCH_DEVICE)
            #Call the embeddings u need within the initialized ones, then stack them.
            embed=torch.vstack(tuple([embeds[nnods.item()].weight for nnods in batch.nnods])) #Keep the duplicates here! (obv).
            #Retrieve nnods for every graph in batch. Use them to chain a list of batch_size parameter-sets.
            batch_nnods=set([nnods.item() for nnods in set(batch.nnods)]) #'set' necessary to avoid duplicates
            ProtoParams=[embeds[nnods].parameters() for nnods in batch_nnods]#.insert(0, net.parameters())
            ProtoParams.insert(0, net.parameters())#Insert the params of the Beheaded network.
            params=chain.from_iterable(ProtoParams)
            #params=chain(net.parameters(), embeds[40].parameters(), embeds[50].parameters())
            #Feed the chain to the optimizer, in order to make one of the called embeddings learn from training.
            optimizer=torch.optim.AdamW(params, **opt_hypers, weight_decay=1e-2)

            try:
                name, loss, prob= run_gnn_training(batch, net, optimizer, embed, batchJ, hypers['tolerance'], seed=SEED_VALUE)
                names.append(name) #all this cuz sometimes can't be processesd by net and triggers exception, cuzing different lengths.


                # run optimization with backpropagation
                optimizer.zero_grad()  # clear gradient for step
                loss.backward()  # calculate gradient through compute graph
                optimizer.step()  # take step, update weights

                detloss=loss.cpu().detach()
                EpochCumLossPerBatch+=detloss
                detprob=prob.cpu().detach()
                #detHardLoss=cost_hard.cpu().detach()
                #EpochCumHardLossPerBatch=detHardLoss
                DictoListUpdate(losses, i, detloss)
                probs.append(detprob)
                #DictoListUpdate(hard_losses, i, detHardLoss)

            except IndexError:
                print(f'index error for batch {batch.fnames}')
        
        #Printing avg loss at the end of every epoch.
        #"Insta" because saving loss for every forward, as opposed to every epoch. 
        InstaSaver(f'{save_path}/train',[epoch,EpochCumLossPerBatch/len(train_dataloader[i])],
                   f'q_{i}_{model}_embdim_{emb_dim}_hidim_{hid_dim}_filename_{filename_without_ext}_TrData')
        if EpochCumLossPerBatch/len(train_dataloader[i])<0.0089:
            print(f'BREAKING | Epoch {epoch} | Soft Loss: {EpochCumLossPerBatch/len(train_dataloader[i]):.3f} | batch {batchJ}')
            break

        # Printing the epoch recap once every [1] epochs.
        if epoch % print_every == 0:
            print(f'Epoch {epoch} | Soft Loss: {EpochCumLossPerBatch/len(train_dataloader[i]):.3f} | ChroNu: {i} |\
 time: {round(time() - t_start, 4)}s | GPU memory {torch.cuda.memory_allocated(device=TORCH_DEVICE)}')#Discrete Cost: {EpochCumHardLossPerBatch/len(train_dataloader[i]):.1f} |
        #Append to avg losses for this chi the loss averaged on this last epoch.
        DictoListUpdate(MeanEpochLoss, i, EpochCumLossPerBatch/len(train_dataloader[i]))

        ####  Testing  ####
        if epoch%test_every==0:
            models[i][0].eval()
            with torch.no_grad():
                for teBatchJ, test_batch in enumerate(test_dataloader[i]):
                    test_batch.to(TORCH_DEVICE)
                    embedde=torch.vstack(tuple([embeds[nnods.item()].weight for nnods in test_batch.nnods]))
                    name, Tcost_hard, prob, coloring = run_gnn_testing(
                        test_batch, net, embedde, hypers['tolerance'], seed=SEED_VALUE)

                    #EpochCumLossPerBatchEval=Tloss.cpu()
                    EpochCumHardLossPerBatchEval+=Tcost_hard.cpu()
                    if Tcost_hard < best_cost:
                        #best_loss = loss
                        best_cost = Tcost_hard
                        best_colorings.update({i:[coloring, batch.fnames]})
                    #InstaSaver(f'{savplot_path_dict[typecurrent]}/test',[epoch,Tcost_hard.cpu()],f'{i}chr_{len(data_train[i])}TrData')


                    #if j%1==0:            #  | sat:{torch.sum(test_batch.labels)/8} ## RIMETTERE PER DISCRIMINATING COLORING ##
#                     if epoch % print_every == 0:
#                         print(f'TESTING: chr_n {i} | Epoch {epoch} | HardCost: {Tcost_hard.item():.1f} \
# | batch {teBatchJ} | #nodes: {test_batch.num_nodes} | GPU memory {torch.cuda.memory_allocated(device=TORCH_DEVICE)}')# | Soft Loss: {Tloss.item():.3f}
                #DictoListUpdate(TestMeanEpochLoss, i, [epoch, EpochCumLossPerBatchEval/len(test_dataloader[i])])
                DictoListUpdate(TestHardMeanEpochLoss, i, [epoch, EpochCumHardLossPerBatchEval/len(test_dataloader[i])])
            if epoch % print_every == 0:
                print(f'TEST-Epoch: {epoch//test_every} | Hard Cost: {EpochCumHardLossPerBatchEval/len(test_dataloader[i]):.1f}\
 | time:{round(time() - t_start, 4)}s')#| Soft Loss: {EpochCumLossPerBatchEval/len(test_dataloader[i]):.3f}
            InstaSaver(f'{save_path}/test',[epoch,EpochCumHardLossPerBatchEval/len(test_dataloader[i])],
                       f'q_{i}_{model}_embdim_{emb_dim}_hidim_{hid_dim}_filename_{filename_without_ext}_TstData')
            ModelSaver(models_path, models[i], i, f'Model_q_{i}_{model}_embdim_{emb_dim}_hidim_{hid_dim}_filename_{filename_without_ext}')
        ####  End Testing  ####

    final_colorings.update({i:(coloring, batch.fnames)})
    # Print final loss
    print('Epoch %d | Final loss: %.3f | Lowest discrete cost: %.1f' % (epoch, loss.item(), best_cost))

    # Final coloring
    final_loss = detloss
    print(f'Final coloring: {final_colorings[i][0]}, soft loss: {final_loss:.3f}, chromatic_number: {torch.max(coloring)+1}')
    print_final_colorings(coloring_path, final_colorings, i, f'Colorings_q_{i}_{model}_embdim_{emb_dim}_hidim_{hid_dim}_filename_{filename_without_ext}.txt')
    #final_batch.update({i:batch.fnames})

    runtime_gnn = round(time() - t_start, 4)
    print(f'GNN runtime: {runtime_gnn}s')