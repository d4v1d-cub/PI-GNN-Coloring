import networkx as nx
import os


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


def print_all(min_exp, max_exp, c_list, ngraphs_each, seed0, path):
    for exp in range(min_exp, max_exp + 1):
        for c in c_list:
            for seed in range(seed0, seed0 + ngraphs_each):
                n = int(2 ** exp)
                g = nx.erdos_renyi_graph(n, c / (n - 1), seed=seed)
                filename = f'ErdosRenyi_N_{n}_c_{"{0:.3f}".format(c)}_seed_{seed}.txt'
                print_graph(g, path, filename)


s0 = 2
ng_each = 1
c_l = [3]
min_e = 5
max_e = 5

path_to_graphs = "./random_graphs/ErdosRenyi"

print_all(min_e, max_e, c_l, ng_each, s0, path_to_graphs)