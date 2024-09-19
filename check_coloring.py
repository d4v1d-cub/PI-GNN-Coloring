import networkx as nx
import numpy as np


def parse_line(file_line, node_offset=-1):
    x, y = [int(x) for x in file_line.split()]
    x, y = x+node_offset, y+node_offset  # nodes in file are 1-indexed, whereas python is 0-indexed
    return x, y


def check(filecols, path_to_graph, graphname, q=3):
    filegraph=f'{path_to_graph}/{graphname}'
    with open(filegraph, 'r') as f:
        content = f.read().strip()
    lines = content.split('\n')
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
    

    fcol = open(filecols, "r")
    cols = []
    while True:
        j = fcol.readline()
        if not j:
            print(f'graph {graphname} not found inside {filecols}')
            break
        line_col = j.split("\t")
        gname_try = line_col[1]
        if gname_try == graphname:
            print(f'graph {graphname} found')
            cols = line_col[2][1:-2].split(",")
            for i in range(len(cols)):
                cols[i] = int(cols[i])
            break
    cond = True
    for i in range(len(cols)):
        if cols[i] > q:
            print(f'coloring with q={q} not achieved for graph {graphname}')
            print(f'color={cols[i]} at site {i}')
            break
        else:
            for j in nx.neighbors(nx_clean, i):
                if cols[j] == cols[i]:
                    cond = False
                    break
            if not cond:
                print(f'the neighboring nodes ({i}, {j})  have the same color={cols[i]}')
                break
    fcol.close()
    if cond:
        print(f'The graph "{graphname}" is well colored with q={q}')



def check_orig(filecols, path_to_graph, graphname, q=3):
    
    try:    
        fcol = open(filecols, "r")
        cols = []
        j = fcol.readline()
        cols = j[1:-2].split(",")
        for i in range(len(cols)):
            cols[i] = int(cols[i])
        
        
        filegraph=f'{path_to_graph}/{graphname}'
        with open(filegraph, 'r') as f:
            content = f.read().strip()
        lines = content.split('\n')
        n=int(lines[0])
        nedges=int(lines[1])
        edgesnx=[parse_line(line, -1) for line in lines[2:nedges+2]]
        
        nx_orig = nx.Graph()
        for i in range(n):
            nx_orig.add_node(i)
        for edge in edgesnx:
            nx_orig.add_edge(edge[0], edge[1])
        

        cond = True
        for i in range(len(cols)):
            if cols[i] > q:
                print(f'coloring with q={q} not achieved for graph {graphname}')
                print(f'color={cols[i]} at site {i}')
                break
            else:
                for j in nx.neighbors(nx_orig, i):
                    if cols[j] == cols[i]:
                        cond = False
                        break
                if not cond:
                    print(f'the neighboring nodes ({i}, {j})  have the same color={cols[i]}')
                    break
        fcol.close()
        if cond:
            print(f'The graph "{graphname}" is well colored with q={q}')
            return True, True
        else:
            print(f'The graph "{graphname}" is NOT well colored with q={q}')
            return False, True
    except (IOError, OSError):
        print(f'file "{filecols}" not found')
        return False, False


def check_all(N_list, c_list, q, seedmin, seedmax, path_to_graph, path_to_cols, 
              model, embdim, hiddim, fileout):
    fout = open(fileout, "w")
    fout.write("# N  c  nsamples  solved\n")
    for N in N_list:
        path_to_graph_new = path_to_graph + f'N_{N}'
        for c in c_list:
            nsamples = 0
            solved = 0.0
            for seed in range(seedmin, seedmax + 1):
                graphname = f'ErdosRenyi_N_{N}_c_{"{0:.3f}".format(c)}_seed_{seed}.txt'
                filecols = f'{path_to_cols}/coloring_q_{q}_model_{model}_embdim_{embdim}_hidim_{hiddim}_filename_{graphname}'
                colored, found = check_orig(filecols, path_to_graph_new, graphname, q)
                solved += colored
                nsamples += found
            if nsamples > 0:
                fout.write(str(N) + "\t" + str(c) + "\t" + str(nsamples) + "\t" + str(solved / nsamples) + "\n")
    fout.close()


def read_params(fileparams):
    fpar = open(fileparams, "r")
    fpar.readline()
    line = fpar.readline().split()
    fpar.close()
    return int(line[0]), int(line[1]), float(line[2]), float(line[3])


def check_all_hyperopt(N_list, c_list, q, seedmin, seedmax, path_to_graph, path_to_cols, 
                       model, fileout, path_to_params, nep_hyper, ngr_hyper, 
                       ntr_hyper):
    fout = open(fileout, "w")
    fout.write("# N  c  nsamples  solved\n")
    for N in N_list:
        path_to_graph_new = path_to_graph + f'N_{N}'
        for c in c_list:
            nsamples = 0
            solved = 0.0
            fileparams = f'{path_to_params}/best_params_q_{q}_N_{N}_c_{"{0:.2f}".format(c)}_model_{model}_nepochs_{nep_hyper}_ngraphs_{ngr_hyper}_ntrials_{ntr_hyper}.txt'
            embdim, hiddim, dout, lrate = read_params(fileparams)
            for seed in range(seedmin, seedmax + 1):
                graphname = f'ErdosRenyi_N_{N}_c_{"{0:.3f}".format(c)}_id_{seed}.txt'
                filecols = f'{path_to_cols}/coloring_q_{q}_model_{model}_embdim_{embdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_filename_{graphname}'
                colored, found = check_orig(filecols, path_to_graph_new, graphname, q)
                solved += colored
                nsamples += found
            if nsamples > 0:
                fout.write(str(N) + "\t" + str(c) + "\t" + str(nsamples) + "\t" + str(solved / nsamples) + "\n")
    fout.close()


N_list = [16, 32, 64, 128, 256]
c_list = np.arange(2.96, 5.01, 0.18)
# c_list = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
q = 3
seedmin = 1
seedmax = 201
model = "GraphSAGE"

path_to_graph = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/random_graphs/ErdosRenyi/'
path_to_cols = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Opt_params/colorings"

path_to_params = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/hyperopt/opt_all"
nep_hyper = "1e2"
ngr_hyper = 20
ntr_hyper = 1000

path_out = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Opt_params/Stats/"
fileout = path_out + f'Solved_q_{q}_ErdosRenyi_model_{model}_nephyp_{nep_hyper}_ngrhyp_{ngr_hyper}_ntrhyp_{ntr_hyper}.txt'

solv_frac = check_all_hyperopt(N_list, c_list, q, seedmin, seedmax, path_to_graph, path_to_cols,
                               model, fileout, path_to_params, nep_hyper, ngr_hyper, ntr_hyper)