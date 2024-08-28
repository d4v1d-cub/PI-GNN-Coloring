import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
from torch_geometric.data import Data
import torch_geometric.utils as ut
from itertools import chain
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch_geometric.nn as geonn


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


def parse_line(file_line, node_offset=0):
    x, y = [int(x) for x in file_line.split()]
    x, y = x+node_offset, y+node_offset  # nodes in file are 1-indexed, whereas python is 0-indexed
    return x, y


class SyntheticDataset():
    def __init__(self, root, stop, chr_n):
        self.stop=stop-2
        self.root=root
        self.key=0
        self.chr_n = chr_n
    def process(self):
        self.graphs = {}
        self.nxgraphs={}
        #self.labels = []
        self.fnames={}
        print(f'Building graph from contents of file: ')
        for k, fname in enumerate(os.listdir(self.root)):
            edgesnx=[]
            self.fpath = os.path.join(self.root, fname)
            #print(f'{fname}, ', end='')
            print(f'{100*k/min(self.stop,len(os.listdir(self.root))):.2f}%', end="\r")
            with open(self.fpath, 'r') as f:
                content = f.read().strip()
            lines = content.split('\n')  # skip comment line(s)
            n=int(lines[0])
            nedges=int(lines[1])
            edgesnx=[parse_line(line, -1) for line in lines[2:nedges+2]]
            
            nx_orig = nx.Graph()
            for i in range(n):
                nx_orig.add_node(i)
            for edge in edgesnx:
                nx_orig.add_edge(edge[0], edge[1])

            nx_clean = nx_orig.copy()
            nx_clean.remove_nodes_from(list(nx.isolates(nx_clean)))
            nx_clean = nx.convert_node_labels_to_integers(nx_clean)
            
            #graph=ut.convert.from_networkx(nx_graph)
            g = ut.from_networkx(nx_clean)

            if len(lines) == nedges+3:
                self.chr_n = int(lines[nedges+2])

            if self.chr_n not in self.graphs.keys():
                self.graphs.update({self.chr_n:[Data(edge_index=g.edge_index, nx_graph=nx_clean, nnods=nx_clean.number_of_nodes(), num_nodes=nx_clean.number_of_nodes(), nedges=nedges, fnames=fname)]})
            else:
                self.graphs[self.chr_n].append(Data(edge_index=g.edge_index, nx_graph=nx_clean, nnods=nx_clean.number_of_nodes(), num_nodes=nx_clean.number_of_nodes(), nedges=nedges, fnames=fname))


            if k > self.stop:
                break # store only #graphs = Batch_size
        return self.graphs 

    def __getitem__(self, i):
        return self.graphs[self.key][i]

    def __len__(self):
        return len(self.graphs[self.key])

    def len(self, key):
        return len(self.graphs[key])



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
        #self.v_init = nn.Parameter(torch.randn(in_feats))
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(geonn.SAGEConv(in_feats, hidden_size, agg_type))
        # output layer

        self.layers.append(geonn.SAGEConv(hidden_size, num_classes, agg_type))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, embeds, num_tot_edges=50*1):
        """
        Define forward step of netowrk. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution.
        :param vertex_initial_embeddings: Input node representations
        :type vertex_initial_embeddings: torch.tensor
        :return: Final layer representation, pre-activation (i.e. class logits)
        :rtype: torch.tensor
        """
        #vertex_initial_embeddings=torch.tile(self.v_init, (num_tot_nodes,1))
        h=embeds#torch.vstack(tuple([embeds[i%len(embeds)] for i in range(num_tot_edges)]))
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, g)

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
        self.layers.append(geonn.GraphConv(in_feats, hidden_size))
        # output layer
        self.layers.append(geonn.GraphConv(hidden_size, num_classes))
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
            # h = layer(g, h)
            h = layer(h, g)
        return h


# Construct graph to learn on #
def get_gnn(gnn_hypers, chrom, torch_device, torch_dtype, nndes_min, nndes_max):
    """
    Helper function to load in GNN object, optimizer, and initial embedding layer.
    :param n_nodes: Number of nodes in graph
    :type n_nodes: int
    :param gnn_hypers: Hyperparameters to provide to GNN constructor
    :type gnn_hypers: dict
    :param torch_device: Compute device to map computations onto (CPU vs GPU)
    :type torch_dtype: str
    :param torch_dtype: Specification of pytorch datatype to use for matrix
    :type torch_dtype: str
    :return: Initialized GNN instance, embedding layer, initialized optimizer instance
    :rtype: GNN_Conv or GNN_SAGE, torch.nn.Embedding, torch.optim.AdamW
    """

    try:
        print(f'Function get_gnn(): Setting seed to {gnn_hypers["seed"]} for model of chr_num {chrom}')
        set_seed(gnn_hypers['seed'])
    except KeyError:
        print('!! Function get_gnn(): Seed not specified in gnn_hypers object. Defaulting to 0 !!')
        set_seed(0)

    model = gnn_hypers['model']
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = chrom   ####  CAMBIA PER GESU'  ####
    agg_type = gnn_hypers['layer_agg_type'] or 'mean'

    # instantiate the GNN
    if model == "GraphConv":
        net = GNNConv(dim_embedding, hidden_dim, number_classes, dropout)
    elif model == "GraphSAGE":
        net = GNNSage(dim_embedding, hidden_dim, number_classes, dropout, agg_type)
    else:
        raise ValueError("Invalid model type input! Model type has to be in one of these two options: ['GraphConv', 'GraphSAGE']")

    net = net.type(torch_dtype).to(torch_device)
    #embed = nn.Embedding(n_nodes, dim_embedding, device=dev)
    embeds={}
    for nnods in range(nndes_min, nndes_max):
        embeds.update({nnods: nn.Embedding(nnods, dim_embedding, device=dev)})
        # set up Adam optimizer
        #params = chain(net.parameters(), embeds[nnods].parameters())

    #return net, embeds#, optimizers
    # set up Adam optimizer
    #params = chain(net.parameters(), embed.parameters())

    #optimizer = torch.optim.AdamW(params, **opt_params, weight_decay=1e-2)

    return net, embeds#, optimizer


# helper function for graph-coloring loss
def loss_func_mod(probs, adj_tensor):
    loss_ = torch.mul(adj_tensor, (probs @ probs.T)).sum() / 2#)**2 + torch.sum(torch.norm(probs, dim=1, p=2))**2)
    return loss_


# helper function for custom loss according to Q matrix
def loss_func_color_hard(coloring, edge0s, edge1s, batch):

    nods=[n.item() for n in batch.nnods]
    cost_ = 0
    for i in range(len(batch)):
        collocal=coloring[sum(nods[0:i]):(sum(nods[0:i])+nods[i])]
        #for (u, v) in batch.nx_graph[i].edges:
        for (u, v) in zip(edge0s[i], edge1s[i]):
            cost_ += 1*(collocal[u] == collocal[v])*(u != v) #for self loops loss func not to be incremented obv.

    return cost_


def run_gnn_training(batch, net, optimizer, embeds, batchj, tolerance=1e-4, seed=1):
    # Ensure RNG seeds are reset each training run
    #print(f'Function run_gnn_training(): Setting seed to {seed}')
    set_seed(seed)
    #CUMnedges=[sum(batch.nedges[0:i]) for i in range(len(batch.nedges)+1)]
    # get soft prob assignments
    logits = net(batch.edge_index, embeds) #batch.num_nodes check with a print
    # apply softmax for normalization
    probs = F.softmax(logits, dim=1)
    #print(coloring)
    # get cost value with POTTS cost function
    #edge0s=[(batch.edge_index[0][CUMnedges[i]:CUMnedges[i+1]])-min(batch.edge_index[0][CUMnedges[i]:CUMnedges[i+1]]) for i in range(len(CUMnedges)-1)]
    #edge1s=[(batch.edge_index[1][CUMnedges[i]:CUMnedges[i+1]])-min(batch.edge_index[1][CUMnedges[i]:CUMnedges[i+1]]) for i in range(len(CUMnedges)-1)]
    adj_mat=ut.to_dense_adj(torch.vstack((batch.edge_index[0], batch.edge_index[1]))).squeeze(0)
#    print(f'name: {batch.fnames[7]}')
#    print(f'probs.size(): {probs.size()}')
#    avva=[adj_mat[i].size() for i in range(len(adj_mat))]
#    print(f'adj_mats.sizes: {avva} batch.nedges: {batch.nedges}')
#    porbs=torch.split(probs, batch.nnods.tolist())
#    vava=[torch.split(probs, [n for n in batch.nnods])[i].size() for i in range(len(torch.split(probs, [n for n in batch.nnods])))]
#    print(f'size of probsplit: {vava}')
#    print(f'edge0s: {edge0s[7]}')
#    print(f'edge1s: {edge1s[7]}')
    loss = loss_func_mod(probs, adj_mat)

    # get cost based on current hard class assignments
    # update cost if applicable
    return batch.fnames, loss, probs

def run_gnn_testing(batch, net, embeds, tolerance=1e-4, seed=1):
    # Ensure RNG seeds are reset each training run
    #print(f'Function run_gnn_training(): Setting seed to {seed}')
    set_seed(seed)
    # get soft prob assignments
    logits = net(batch.edge_index, embeds) #batch.num_nodes check with a print
    # apply softmax for normalization
    probs = F.softmax(logits, dim=1)
    #print(coloring)
    # get cost value with POTTS cost function
    CUMnedges=[sum(batch.nedges[0:i]) for i in range(len(batch.nedges)+1)]
    edge0s=[(batch.edge_index[0][CUMnedges[i]:CUMnedges[i+1]])-min(batch.edge_index[0][CUMnedges[i]:CUMnedges[i+1]]) for i in range(len(CUMnedges)-1)]
    edge1s=[(batch.edge_index[1][CUMnedges[i]:CUMnedges[i+1]])-min(batch.edge_index[1][CUMnedges[i]:CUMnedges[i+1]]) for i in range(len(CUMnedges)-1)]
 
    # get cost based on current hard class assignments
    # update cost if applicable
    #print(coloring)
    #print(coloring.size())
    coloring = torch.argmax(probs, dim=1)
    cost_hard = loss_func_color_hard(coloring, edge0s, edge1s, batch)
    return batch.fnames, cost_hard, probs, coloring   