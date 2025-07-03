import networkx as nx
import numpy as np
import os


def parse_line(file_line, node_offset=0):
    splitted = file_line.split()
    x = int(splitted[1])
    y = int(splitted[2])
    x, y = x+node_offset, y+node_offset  # nodes in file are 1-indexed, whereas python is 0-indexed
    return x, y



def check_orig(filecols, path_to_graph, graphname, q=3):
    
    try:    
        fcol = open(filecols, "r")
        cols = []
        j = fcol.readline()
        cols = j[1:-2].split(",")
        for i in range(len(cols)):
            cols[i] = int(cols[i])
        
        fcol.close()
        
        filegraph=f'{path_to_graph}/{graphname}'
        with open(filegraph, 'r') as f:
            content = f.read().strip()
        lines = content.split('\n')
        n=int(lines[0].split()[1])
        nedges=int(lines[1].split()[1])
        edgesnx=[parse_line(line, -1) for line in lines[2:nedges+2]]
        
        nx_orig = nx.Graph()
        for i in range(n):
            nx_orig.add_node(i)
        for edge in edgesnx:
            nx_orig.add_edge(edge[0], edge[1])
        

        ener = 0
        for e in nx_orig.edges():
            if cols[e[0]] == cols[e[1]]:
                ener += 1
        if ener == 0:
            print(f'The graph "{graphname}" is well colored with q={q}')
            return True, ener
        else:
            print(f'The graph "{graphname}" is NOT well colored with q={q}')
            return True, ener
    except (IOError, OSError):
        # print(f'file "{filecols}" not found')
        return False, -1


def read_params(fileparams):
    fpar = open(fileparams, "r")
    fpar.readline()
    line = fpar.readline().split()
    fpar.close()
    return int(line[0]), int(line[1]), float(line[2]), float(line[3])




def check_all_rec_varnep(N_list, c_list, q, seedmin, seedmax, path_to_graph, path_to_cols, 
                  path_to_others, path_to_params, ntrials, nepochs_list, str_program):
    fileparams = f'{path_to_params}/params_paper_recurrence.txt'
    randdim, hiddim, dout, lrate = read_params(fileparams)
    for j in range(len(N_list)):
        N = N_list[j]
        nepochs = nepochs_list[j]
        path_to_graph_new = path_to_graph + f'N_{N}'
        for c in c_list:
            nsamples = 0
            for seed in range(seedmin, seedmax + 1):
                m = int(round(N * c / 2))
                graphname = f'ErdosRenyi_N_{N}_M_{m}_id_{seed}.txt'
                fileothers_orig = f'{path_to_others}/others_recurrent_less_hardloss_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'
                if not os.path.exists(fileothers_orig):
                    filecols = f'{path_to_cols}/coloring_recurrent_{str_program}_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'
                    found, ener = check_orig(filecols, path_to_graph_new, graphname, q)
                    if found:
                        nsamples += 1
                        fileothers = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/GPU/single_graph/less_hardloss/temp/others/others_recovered_recurrent_less_hardloss_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'
                        with open(fileothers, 'w') as fout:
                            fout.write(f"{ener}\n")
                    else:
                        print(f'no results for the graph "{graphname}"')
            if nsamples > 0:
                print(str(N) + "\t" + str(c) + "\t" + str(nsamples))



path_to_graph = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/random_graphs/ErdosRenyi/New_graphs/'


N_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
c_list = np.arange(3.32, 5.01, 0.18)
q = 3
# c_list = np.arange(9.9, 13.6, 0.4)
# q = 5
seedmin = 1
seedmax = 400
ntrials = 5

A = 100
nepochs_list = []
for i in range(len(N_list)):
    nepochs_list.append(A * N_list[i])

graph_version = "New_graphs"
cluster = ""
program_version = "less_hardloss"
processor = "GPU"

path_to_others = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/{processor}/single_graph/{program_version}/q_{q}/{graph_version}/others{cluster}'

path_to_params = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/params"

path_to_cols = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/{processor}/single_graph/{program_version}/q_{q}/{graph_version}/colorings{cluster}'


solv_frac = check_all_rec_varnep(N_list, c_list, q, seedmin, seedmax, path_to_graph, path_to_cols,
                        path_to_others, path_to_params, ntrials, nepochs_list, program_version)