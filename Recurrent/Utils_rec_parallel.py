import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN1D
import random
import numpy as np
import dgl
from dgl.nn.pytorch import SAGEConv
from dgl.data import DGLDataset
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from time import time
import psutil
import os


def set_seed(seed):
    """
    Sets random seeds for training.
    :param seed: Integer used for seed.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_adjacency_matrix(nx_graph, torch_device, torch_dtype):
    """
    Pre-load adjacency matrix, map to torch device
    :param nx_graph: Graph object to pull adjacency matrix for
    :type nx_graph: networkx.Graph
    :param torch_device: Compute device to map computations onto (CPU vs GPU)
    :type torch_dtype: str
    :param torch_dtype: Specification of pytorch datatype to use for matrix
    :type torch_dtype: str
    :return: Adjacency matrix for provided graph
    :rtype: torch.tensor
    """

    adj = nx.linalg.graphmatrix.adjacency_matrix(nx_graph).tocoo()
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj.shape

    adj_ = torch.sparse_coo_tensor(i, v, torch.Size(shape)).type(torch_dtype).to(torch_device)

    return adj_


def saver_loss(lista1, path, name): #IGNORE
    print(f'saving losses in {name}')
    # Write-Overwrites
    file1 = open(f'{path}/{name}.txt', "w")  # write mode
    epoch_i = 0
    while epoch_i < min(100, len(lista1)):
        file1.write(f'{epoch_i}\t{"{0:.4f}".format(lista1[epoch_i])}\n')
        epoch_i += 1
    while epoch_i < len(lista1):
        file1.write(f'{epoch_i}\t{"{0:.4f}".format(lista1[epoch_i])}\n')
        epoch_i = int(epoch_i * 1.02)
    file1.close()


def get_full_colors(final_colors, nx_orig):
    final_colors_new = []
    isol = list(nx.isolates(nx_orig))
    offset = 0
    for i in range(nx_orig.number_of_nodes()):
        if i in isol:
            final_colors_new.append(0)
            offset += 1
        else:
            final_colors_new.append(final_colors[i - offset].item())
    return final_colors_new


def saver_colorings(full_colors, best_colorings, best_costs, path, name_start, 
                    data_train, index_graphs, seed, best_seed):    

    ncumul = 0
    for fname in os.listdir(data_train.folder):
        cost = 0
        with open(f'{data_train.folder}/{fname}', 'r') as f:
                content = f.read().strip()
        lines = content.split('\n')  # skip comment line(s)
        n=int(lines[0].split()[1])
        nedges=int(lines[1].split()[1])
        for line in lines[2:nedges+2]:
            edge = parse_line(line, data_train.node_offset + ncumul)
            if full_colors[edge[0]] == full_colors[edge[1]]:
                cost += 1
        if cost < best_costs[index_graphs[fname]]:
            best_costs[index_graphs[fname]] = cost
            best_colorings[index_graphs[fname]] = full_colors[ncumul:ncumul + n]
            best_seed[index_graphs[fname]] = seed
            if cost == 0:
                fname_without_ext = os.path.splitext(os.path.basename(fname))[0]
                file3 = open(f'{path}/{name_start}_filename_{fname_without_ext}.txt', "w")  # write mode
                file3.write(str(best_colorings[index_graphs[fname]])+'\n')#final coloring
                file3.write(f'{seed}\t{best_costs[index_graphs[fname]]}')
                file3.close()
                os.remove(f'{data_train.folder}/{fname}')
        ncumul += n


def saver_colorings_final(best_colorings, best_costs, path, name_start, 
                            data_train, index_graphs, best_seed):    

    for fname in os.listdir(data_train.folder):
        fname_without_ext = os.path.splitext(os.path.basename(fname))[0]
        file3 = open(f'{path}/{name_start}_filename_{fname_without_ext}.txt', "w")  # write mode
        file3.write(str(best_colorings[index_graphs[fname]])+'\n')#final coloring
        file3.write(f'{best_seed[index_graphs[fname]]}\t{best_costs[index_graphs[fname]]}')
        file3.close()
        os.remove(f'{data_train.folder}/{fname}')


    


def saver_others(others, path, name):
    file3 = open(f'{path}/{name}.txt', "w")
    file3.write(str(others[0]))
    for i in range(1, len(others)):
        file3.write("\t" + str(others[i]))
    file3.close()


def parse_line(file_line, node_offset=0):
    splitted = file_line.split()
    x = int(splitted[1])
    y = int(splitted[2])
    x, y = x+node_offset, y+node_offset  # nodes in file are 1-indexed, whereas python is 0-indexed
    return x, y


def find_disc_nodes(edges_nx, n):
    all_nodes = []
    for edge in edges_nx:
        if edge[0] not in all_nodes:
            all_nodes.append(edge[0])
        if edge[1] not in all_nodes:
            all_nodes.append(edge[1])
    missing = []
    for i in range(n):
        if i not in all_nodes:
            missing.append(i)
    return missing


def get_graph_index(folder):
    graph_index = {}
    counter = 0
    for fname in os.listdir(folder):
        graph_index[fname] = counter
        counter += 1
    return graph_index, counter

class SyntheticDataset(DGLDataset):
    def __init__(self, folder, node_offset=-1):
        self.folder=folder
        self.node_offset=node_offset
        super().__init__(name="synthetic")

    def process(self):

        nx_orig = nx.Graph()
        total_n = 0
        edgesnx = []
        for fname in os.listdir(self.folder):
            print(f'Building graph from contents of file: {fname}')
            with open(f'{self.folder}/{fname}', 'r') as f:
                content = f.read().strip()
            lines = content.split('\n')  # skip comment line(s)
            n=int(lines[0].split()[1])
            nedges=int(lines[1].split()[1])
            edgesnx+=[parse_line(line, self.node_offset + total_n) for line in lines[2:nedges+2]]
            total_n += n

        for i in range(total_n):
            nx_orig.add_node(i)
        for edge in edgesnx:
            nx_orig.add_edge(edge[0], edge[1])

        nx_clean = nx_orig.copy()
        nx_clean.remove_nodes_from(list(nx.isolates(nx_clean)))
        nx_clean = nx.convert_node_labels_to_integers(nx_clean)

        self.nx_orig = nx_orig
        self.nxgraph = nx_clean
        dgl_graph = dgl.from_networkx(nx_clean, device=dev)        
        self.graph = dgl_graph


# Define GNN GraphSage object
class GNNSage(nn.Module):
    """
    Basic GraphSAGE-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters - in this
    case, just dropout.
    """

    def __init__(self, in_feats, hidden_size, num_classes, dropout, TORCH_DEV, TORCH_DTYPE):
        """
        Initialize the model object. Establishes model architecture and relevant hypers (`dropout`, `num_classes`, `agg_type`)
        :param g: Input graph object
        :type g: dgl.DGLHeteroGraph
        :param in_feats: Size (number of nodes) of input layer
        :type in_feats: int
        :param hidden_size: Size of hidden layer
        :type hidden_size: int
        :param num_classes: Size of output layer (one node per class)
        :type num_classes: int
        :param dropout: Dropout fraction, between two convolutional layers
        :type dropout: float
        :param agg_type: Aggregation type for each SAGEConv layer. All layers will use the same agg_type
        :type agg_type: str
        """
        
        super(GNNSage, self).__init__()

        self.num_classes = num_classes
        
        # input layers
        self.l1_mean = SAGEConv(in_feats, hidden_size, "mean", activation=F.relu)
        self.l1_pool = SAGEConv(in_feats, hidden_size, "pool", activation=F.relu)
        
        # output layer
        self.outlayer = SAGEConv(hidden_size, num_classes, "mean")
        self.dropout = nn.Dropout(p=dropout)

        self.batch_norm_mean = BN1D(hidden_size, device=TORCH_DEV, dtype=TORCH_DTYPE)
        self.batch_norm_pool = BN1D(hidden_size, device=TORCH_DEV, dtype=TORCH_DTYPE)

        self.relu = nn.ReLU()

    def forward(self, g, features):
        """
        Define forward step of netowrk. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution.
        :param features: Input node representations
        :type features: torch.tensor
        :return: Final layer representation, pre-activation (i.e. class logits)
        :rtype: torch.tensor
        """
        h = features

        h = self.relu(torch.add(self.batch_norm_mean(self.l1_mean(g, h)), 
                             self.batch_norm_pool(self.l1_pool(g, h))))
        h = self.dropout(h)
        h = self.outlayer(g, h)

        return h


# Construct graph to learn on #
# Construct graph to learn on #
def get_gnn(n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    """
    Helper function to load in GNN object, optimizer, and initial embedding layer.
    :param n_nodes: Number of nodes in graph
    :type n_nodes: int
    :param gnn_hypers: Hyperparameters to provide to GNN constructor
    :type gnn_hypers: dict
    :param opt_params: Hyperparameters to provide to optimizer constructor
    :type opt_params: dict
    :param torch_device: Compute device to map computations onto (CPU vs GPU)
    :type torch_dtype: str
    :param torch_dtype: Specification of pytorch datatype to use for matrix
    :type torch_dtype: str
    :return: Initialized GNN instance, embedding layer, initialized optimizer instance
    :rtype: GNN_Conv or GNN_SAGE, torch.nn.Embedding, torch.optim.AdamW
    """

    try:
        print(f'Function get_gnn(): Setting seed to {gnn_hypers["seed"]}')
        set_seed(gnn_hypers['seed'])
    except KeyError:
        print('!! Function get_gnn(): Seed not specified in gnn_hypers object. Defaulting to 0 !!')
        set_seed(0)

    model = gnn_hypers['model']
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']
    randdim = gnn_hypers['dim_rand_input']

    # instantiate the GNN
    print(f'Building {model}...')
    net = GNNSage(dim_embedding, hidden_dim, number_classes, dropout, torch_device, torch_dtype)
    
    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)
    with torch.no_grad():
        embed.weight[:, randdim:] = 0

    # set up Adam optimizer
    # params = chain(net.parameters())

    print(f'Building ADAM-W optimizer...')
    optimizer = torch.optim.AdamW(net.parameters(), **opt_params, weight_decay=1e-2)

    return net, embed, optimizer


# helper function for graph-coloring loss
def loss_func_mod(probs, adj_tensor):
    """
    Function to compute cost value based on soft assignments (probabilities)
    :param probs: Probability vector, of each node belonging to each class
    :type probs: torch.tensor
    :param adj_tensor: Adjacency matrix, containing internode weights
    :type adj_tensor: torch.tensor
    :return: Loss, given the current soft assignments (probabilities)
    :rtype: float
    """

    # Multiply probability vectors, then filter via elementwise application of adjacency matrix.
    #  Divide by 2 to adjust for symmetry about the diagonal
    loss_ = torch.mul(adj_tensor, (probs @ probs.T)).sum() / 2

    return loss_


# helper function for custom loss according to Q matrix
def loss_func_color_hard(coloring, nx_graph):
    """
    Function to compute cost value based on color vector (0, 2, 1, 4, 1, ...)
    :param coloring: Vector of class assignments (colors)
    :type coloring: torch.tensor
    :param nx_graph: Graph to evaluate classifications on
    :type nx_graph: networkx.Graph
    :return: Cost of provided class assignments
    :rtype: torch.tensor
    """

    cost_ = 0
    for (u, v) in nx_graph.edges:
        cost_ += 1*(coloring[u] == coloring[v])*(u != v) #for self loops loss func not to be incremented obv.

    return cost_


def run_gnn_training(nx_graph, graph_dgl, adj_mat, net, embed, optimizer,
                                randdim, number_epochs=int(1e5), patience=1000, tolerance=1e-4, seed=1):
    t_start = time()

    """
    Function to run model training for given graph, GNN, optimizer, and set of hypers.
    Includes basic early stopping criteria. Prints regular updates on progress as well as
    final decision.
    :param nx_graph: Graph instance to solve
    :param graph_dgl: Graph instance to solve
    :param adj_mat: Adjacency matrix for provided graph
    :type adj_mat: torch.tensor
    :param net: GNN instance to train
    :type net: GNN_Conv or GNN_SAGE
    :param embed: Initial embedding layer
    :type embed: torch.nn.Embedding
    :param optimizer: Optimizer instance used to fit model parameters
    :type optimizer: torch.optim.AdamW
    :param number_epochs: Limit on number of training epochs to run
    :type number_epochs: int
    :param patience: Number of epochs to wait before triggering early stopping
    :type patience: int
    :param tolerance: Minimum change in cost to be considered non-converged (i.e.
        any change less than tolerance will add to early stopping count)
    :type tolerance: float
    :return: Final model probabilities, best color vector found during training, best loss found during training,
    final color vector of training, final loss of training, number of epochs used in training
    :rtype: torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, int
    """
    losses=[]
    # Ensure RNG seeds are reset each training run
    print(f'Function run_gnn_training(): Setting seed to {seed}')
    set_seed(seed)

    # Tracking
    
    # Early stopping to allow NN to train to near-completion
    prev_loss = 1.  # initial loss value (arbitrary)
    cnt = 0  # track number times early stopping is triggered

    # Training logic
    epoch = 0
    while epoch < number_epochs:
        # get soft prob assignments
        logits = net(graph_dgl, embed.weight)
        # apply softmax for normalization
        probs = F.softmax(logits, dim=1)

        # get cost value with POTTS cost function
        #weight_classes=weight_classes_orig*factor
        loss = loss_func_mod(probs, adj_mat)

        
        if (abs(loss - prev_loss) <= tolerance) | ((loss - prev_loss) > 0):
            cnt += 1
        else:
            cnt = 0
        # print(f'epoch: {epoch}, cost_hard.item(): {cost_hard.item()}')
        
        losses.append(loss.item())
        '''
        if cost_hard.item()==0:#epoch>=int(2e3) and 
            print('Epoch %d | Soft Loss: %.5f | Discrete Cost: %.5f | calculated ChroNu: %d | ChroNu: %d' % (epoch, loss.item(), cost_hard.item(), torch.max(coloring)+1, chromatic_numbers[graphname]))
            break
        '''
        # if cost_hard.item()==0:
        #     print("0 reached")

        # update loss tracking
        prev_loss = loss

        if cnt >= patience:
            print(f'Stopping early on epoch {epoch}. Patience count: {cnt}')
            break

        # run optimization with backpropagation
        optimizer.zero_grad()  # clear gradient for step
        loss.backward()  # calculate gradient through compute graph
        optimizer.step()  # take step, update weights

        # Rebuild input parameters for recurrence
        with torch.no_grad():
            embed.weight[:, randdim:] = torch.cat((logits, probs), dim=1)

        # tracking: print intermediate loss at regular interval
        if epoch % 500 == 0:
            print(f'Epoch {epoch} | Soft Loss: {loss.item():.5f} \
                   | time: {round(time() - t_start, 4)} |  CPU Usage: {psutil.cpu_percent()} \
                   | RAM Usage: {psutil.virtual_memory().used / (1024 ** 3)} GB  |  GPU memory {torch.cuda.memory_allocated(device=dev)}')
        epoch += 1
    # Print final loss
    # Final coloring
    final_loss = loss
    final_coloring = torch.argmax(probs, 1)
    print(f'Final soft loss: {final_loss:.3f}, chromatic_number: {torch.max(final_coloring)+1}')

    final_cost = loss_func_color_hard(final_coloring, nx_graph)
    print('Epoch %d | Final loss: %.5f | Final cost: %.5f' % (epoch, loss.item(), final_cost))

    
    
    return losses, final_coloring, final_loss, final_cost