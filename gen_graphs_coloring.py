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
    fout.write(f'N\t{graph_to_print.number_of_nodes()}\nM\t{graph_to_print.number_of_edges()}\n')
    for i, j in graph_to_print.edges():
        fout.write(f'e\t{i+1}\t{j+1}\n')
    fout.close()


def print_all(n, c, ngraphs_each, seed0, path):
    random.seed(seed0)
    np.random.seed(seed0)
    for id in range(ngraphs_each):
        m = int(round(c * n / 2))
        g = nx.gnm_random_graph(n, m)
        filename = f'ErdosRenyi_N_{n}_M_{m}_id_{id + 1}.txt'
        print_graph(g, path, filename)


s0 = int(sys.argv[1])
ng_each = int(sys.argv[2])
c = float(sys.argv[3])
N = int(sys.argv[4])

path_to_graphs = sys.argv[5]

print_all(N, c, ng_each, s0, path_to_graphs)