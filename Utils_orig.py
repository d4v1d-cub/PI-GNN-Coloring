import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import dgl
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GraphConv
from dgl.data import DGLDataset
from itertools import chain
# dev = torch.device('cuda')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
from time import time

# Known chromatic numbers for specified problems (from references)
chromatic_numbers = {
    # COLOR graphs
    'jean.col': 10,
    'anna.col': 11,
    'huck.col': 11,
    'david.col': 11,
    'homer.col': 13,
    'myciel2.col': 3,
    'myciel3.col': 4,
    'myciel4.col': 5, #io
    'myciel5.col': 6,
    'myciel6.col': 7,
    'queen5_5.col': 5,
    'queen6_6.col': 7,
    'queen7_7.col': 7,
    'queen8_8.col': 9,
    'queen9_9.col': 10,
    'queen8_12.col': 12,
    'queen11_11.col': 11,
    'queen12_12.col': 12, #io
    'queen13_13.col': 13,
    'le450_5a.col': 5,#io from here
    'le450_5b.col': 5,#
    'le450_5c.col': 5,#
    'le450_5d.col': 5,#
    'le450_15a.col': 15,#
    'le450_15b.col': 15,#
    'le450_15c.col': 15,#
    'le450_15d.col': 15,#to here
    # Citations graphs
    'cora.cites': 5,
    'citeseer.cites': 6,
    'pubmed.cites': 8
}


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

    adj = nx.linalg.graphmatrix.adjacency_matrix(nx_graph).todense()
    adj_ = torch.tensor(adj).type(torch_dtype).to(torch_device) #torch_dtype is float32 in their case

    return adj_

def saver(lista1, name, type, lista2=0): #IGNORE
    if type=='loss':
        print(f'saving {type} of {name}')
        path="./losses"
        # Write-Overwrites
        file1 = open(f'{path}/{name}.txt', "w")  # write mode
        file1.write(str([ [ epoch_i, float("{0:.4f}".format(lista1[epoch_i])), float("{0:.4f}".format(lista2[epoch_i])) ]for epoch_i in range(len(lista1)) ]))
        file1.close()
    if type=='chroma':
        print(f'saving {type} of {name}')
        cpath="./chroms"
        # Write-Overwrites
        file2 = open(f'{cpath}/{name}.txt', "w")  # write mode
        file2.write(str([[epoch_i, int(lista1[epoch_i]) ]for epoch_i in range(len(lista1))]))
        file2.close()
    if type=='coloring':
        print(f'saving {type} of {name}')
        colpath='./colorings'
        file3 = open(f'{colpath}/{name}.txt', "w")  # write mode
        file3.write(str([lista1[i].item() for i in range(len(lista1))])+'\n')#best coloring
        file3.write(str([lista2[i].item() for i in range(len(lista1))])+'\n')#final coloring
        file3.close()

def plotter(graphname, n_nodes, best_cost, best_loss): #IGNORE
    myfile=open(f'./losses/{graphname}.txt')
    myline=myfile.readline()
    b=[i for i in myline[2:-2].split('], [')] #[2:-2] to avoid [[ or ]] accolled to 1st or last entry.
    myfile.close()
    a=[b[i].split(', ') for i in range(len(b))]
    #print(a)

    cfile=open(f'./chroms/{graphname}.txt')
    mycline=cfile.readline()
    ep_chrs=[i for i in mycline[2:-2].split('], [')] #[2:-2] to avoid [[ or ]] accolled to 1st or last entry.
    chrs=[ep_chrs[i].split(', ') for i in range(len(ep_chrs))]
    
    cfile.close()

    print(f'plotting losses of {graphname}')
    fig,ax=plt.subplots()

    plt.subplot(2,1,2)
    plt.xlabel('epochs')    
    epochs=[int(a[i][0]) for i in range(len(a))]
    plt.plot(epochs, [0.0 for _ in range(len(a))], color='black')
    plt.plot(epochs, [float(a[i][1]) for i in range(len(a))], color='blue', label='loss')
    plt.plot(epochs, [float(a[i][2]) for i in range(len(a))], color='navy', label='hard loss')
    plt.plot(epochs, [best_cost for _ in range(len(a))], '--', color='navy', linewidth=0.3, label='best hard loss')
    plt.plot(epochs, [best_loss for _ in range(len(a))], '--', color='blue', linewidth=0.3, label='best loss')
    plt.legend()
    
    print(f'plotting predicted chromatic numbers of {graphname}')
    plt.subplot(2,1,1)
    plt.xlabel('epochs')
    plt.plot(epochs, [int(chrs[i][1]) for i in range(len(chrs))], color='orange', label='pred chromatic number')
    if chromatic_numbers[graphname] is not None:
        plt.plot(epochs, [chromatic_numbers[graphname] for _ in range(len(chrs))], '--', color='red', label='chromatic number')
    plt.legend()

    fig.suptitle(f'{graphname}, {n_nodes} nodes')
    fig.savefig(f'./losses/{graphname}.png')

def plotter_g(name, graph): #IGNORE
    graphs=[graph, graph]
    colors=[]
    mycolfile=open(f'./colorings/{name}.txt')
    mycolines=mycolfile.readlines()
    mycolfile.close()
    g_i=0
    for line in mycolines:
        coloring=[int(i) for i in line[1:-2].split(', ')]#[1:-2] cuz each line has \n at the end in addition to ']'.
        #print(f'g_i: {g_i}')
        #print(f'graphs[g_i].edges: {graphs[g_i].edges}')
        for i, j in list(graphs[g_i].edges):
            #print(f'i,j: {i,j}')
            #print(f'(i,j): {(i,j)}, colors: {(coloring[i], coloring[j])}')
            if coloring[i]==coloring[j]:
                graphs[g_i].edges[i, j]['color'] = 'red'
        colors.append( nx.get_edge_attributes(graphs[g_i],'color').values() )
        g_i+=1

    fig,ax=plt.subplots(1,2)
    print(f'drawing coloured graph for {name}')
    for i in range(g_i):
        plt.subplot(1,2,i+1)
        nx.draw(graphs[i], node_color=coloring, edge_color = colors[i], node_size=30, with_labels = True, width=0.4)#, with_labels=True, node_color=color_map[i])
        fig.set_facecolor('dimgrey')
    fig.suptitle(f'{name}, {len(graph.nodes())} nodes')
    fig.savefig(f'./colorings/{name}.png')


def parse_line(file_line, node_offset):
    """
    Helper function to parse lines out of COLOR files - skips first character, which
    will be an "e" to denote an edge definition, and returns node0, node1 that define
    the edge in the line.
    :param file_line: Line to be parsed
    :type file_line: str
    :param node_offset: How much to add to account for file numbering (i.e. offset by 1)
    :type node_offset: int
    :return: Set of nodes connected by edge defined in the line (i.e. node_from, node_to)
    :rtype: int, int
    """

    x, y = file_line.split(' ')[1:]  # skip first character - specifies each line is an edge definition
    x, y = int(x)+node_offset, int(y)+node_offset  # nodes in file are 1-indexed, whereas python is 0-indexed
    return x, y


class SyntheticDataset(DGLDataset):
    def __init__(self, root, stop, node_offset=-1):
        self.stop=stop-2
        self.root=root
        self.node_offset=node_offset
        super().__init__(name="synthetic")

    def process(self):
        self.graphs = []
        self.nxgraphs=[]
        #self.labels = []
        self.fnames=[]
        for k, fname in enumerate(os.listdir(self.root)):
            self.fnames.append(fname)
            self.fpath = os.path.join(self.root, fname)
            print(f'Building graph from contents of file: {fname}')
            with open(self.fpath, 'r') as f:
                content = f.read().strip()

            # Identify where problem definition starts.
            # All lines prior to this are assumed to be miscellaneous descriptions of file contents
            # which start with "c ".
            start_idx = [idx for idx, line in enumerate(content.split('\n')) if line.startswith('p')][0]
            lines = content.split('\n')[start_idx:]  # skip comment line(s)
            edges = [parse_line(line, self.node_offset) for line in lines[1:] if len(line) > 0]

            nx_temp = nx.from_edgelist(edges)

            # nx_graph = nx.OrderedGraph()
            nx_graph = nx.Graph()
            nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
            nx_graph.add_edges_from(nx_temp.edges, color='blue')
            self.nxgraphs.append(nx_graph)
            dgl_graph = dgl.from_networkx(nx_graph, device=dev)        
            self.graphs.append(dgl_graph)
            if k > self.stop:
                break # store only #graphs = Batch_size 

    def __getitem__(self, i):
        return self.graphs[i]


    def __len__(self):
        return len(self.graphs)


# Define GNN GraphSage object
class GNNSage(nn.Module):
    """
    Basic GraphSAGE-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters - in this
    case, just dropout.
    """

    def __init__(self, in_feats, hidden_size, num_classes, dropout, agg_type='mean'):
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
        
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, agg_type, activation=F.relu))
        # output layer
        self.layers.append(SAGEConv(hidden_size, num_classes, agg_type))
        self.dropout = nn.Dropout(p=dropout)

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
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        return h


# Define GNN GraphConv object
class GNNConv(nn.Module):
    """
    Basic GraphConv-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters - in this
    case, just dropout.
    """
    
    def __init__(self, in_feats, hidden_size, num_classes, dropout):
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
        """
        
        super(GNNConv, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, hidden_size, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(hidden_size, num_classes))
        self.dropout = nn.Dropout(p=dropout)

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
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


# Construct graph to learn on #
# Construct graph to learn on #
def get_gnn(name, g, n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    if name not in chromatic_numbers.keys():
        chromatic_numbers[name] = 'unknown'
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
    agg_type = gnn_hypers['layer_agg_type'] or 'mean'

    # instantiate the GNN
    print(f'Building {model} model for graph {name}, chrom number: {chromatic_numbers[name]}...')
    if model == "GraphConv":
        net = GNNConv(dim_embedding, hidden_dim, number_classes, dropout)
    elif model == "GraphSAGE":
        net = GNNSage(dim_embedding, hidden_dim, number_classes, dropout, agg_type)
    else:
        raise ValueError("Invalid model type input! Model type has to be in one of these two options: ['GraphConv', 'GraphSAGE']")

    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())

    print(f'Building ADAM-W optimizer for graph {name}...')
    optimizer = torch.optim.AdamW(params, **opt_params, weight_decay=1e-2)

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


def run_gnn_training(graphname, nx_graph, graph_dgl, adj_mat, net, embed, optimizer,
                     number_epochs=int(1e5), patience=1000, tolerance=1e-4, seed=1):
    t_start = time()

    if graphname not in chromatic_numbers.keys():
        chromatic_numbers[graphname] = 'unknown'
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
    chroms=[]
    hard_losses=[]
    # Ensure RNG seeds are reset each training run
    print(f'Function run_gnn_training(): Setting seed to {seed}')
    set_seed(seed)

    inputs = embed.weight

    # Tracking
    best_cost = torch.tensor(float('Inf'))  # high initialization
    best_loss = torch.tensor(float('Inf'))
    best_coloring = None

    # Early stopping to allow NN to train to near-completion
    prev_loss = 1.  # initial loss value (arbitrary)
    cnt = 0  # track number times early stopping is triggered

    # Training logic
    for epoch in range(number_epochs):
        # get soft prob assignments
        logits = net(graph_dgl, inputs)
        # apply softmax for normalization
        probs = F.softmax(logits, dim=1)

        coloring = torch.argmax(probs, dim=1)
        #print(len(coloring))
        # get cost value with POTTS cost function
        #weight_classes=weight_classes_orig*factor
        loss = loss_func_mod(probs, adj_mat)

        # get cost based on current hard class assignments
        # update cost if applicable

        cost_hard = loss_func_color_hard(coloring, nx_graph)
        
        #if cost_hard.item()==0 or cost_hard.item()==1:
        #    weight_classes=weight_classes_orig #if it's already 0 (or 1), challenge the net to explore low occupancy colours encodings.
            #(by annulling the flattening of weights proportional to the earliness)
        
        if cost_hard < best_cost:
            best_loss = loss
            best_cost = cost_hard
            best_coloring = coloring
        
        # Early stopping check
        # If loss increases or change in loss is too small, trigger
        
        if (abs(loss - prev_loss) <= tolerance) | ((loss - prev_loss) > 0):
            cnt += 1
        else:
            cnt = 0
        # print(f'epoch: {epoch}, cost_hard.item(): {cost_hard.item()}')
        
        losses.append(loss.item())
        hard_losses.append(cost_hard.item())
        chroms.append((torch.max(coloring)+1).item())
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

        # tracking: print intermediate loss at regular interval
        if epoch % 500 == 0:
            print(f'Epoch {epoch} | Soft Loss: {loss.item():.5f}  | ChroNu: {torch.max(coloring)+1} | time: {round(time() - t_start, 4)} | Discrete Cost:{cost_hard.item()}')
    # Print final loss
    print('Epoch %d | Final loss: %.5f | Lowest discrete cost: %.5f' % (epoch, loss.item(), best_cost))

    # Final coloring
    final_loss = loss
    final_coloring = torch.argmax(probs, 1)
    print(f'Final coloring: {final_coloring}, soft loss: {final_loss:.3f}, chromatic_number: {torch.max(coloring)+1}')
    
    return graphname, losses, hard_losses, chroms, probs, best_coloring, best_loss.item(), final_coloring, final_loss, epoch 