import networkx as nx
import os
import sys
import numpy as np
import random


def print_graph(graph, path, filename):
    graph_to_print = nx.Graph()
    graph_to_print.add_nodes_from(sorted(graph.nodes()))
    graph_to_print.add_edges_from(graph.edges)
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    fout = open(f'{path}/{filename}', "w")
    fout.write(f'{graph.number_of_nodes()}\n{2 * graph.number_of_edges()}\n')
    for i in range(graph.number_of_nodes()):
        for j in graph.neighbors(i):
            fout.write(f'{i+1}\t{j+1}\n')
    fout.close()


def print_all(n, c, ngraphs_each, seed0, path):
    random.seed(seed0)
    np.random.seed(seed0)
    for id in range(ngraphs_each):
        g = nx.erdos_renyi_graph(n, c / (n - 1))
        filename = f'ErdosRenyi_N_{n}_c_{"{0:.3f}".format(c)}_id_{id}.txt'
        print_graph(g, path, filename)


s0 = int(sys.argv[1])
ng_each = int(sys.argv[2])
c = float(sys.argv[3])
N = int(sys.argv[4])

path_to_graphs = sys.argv[5]

print_all(N, c, ng_each, s0, path_to_graphs)