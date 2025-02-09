import networkx as nx
import numpy as np
import pandas as pd


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
            print(f'The graph "{graphname}" is well colored with q={q}', end='')
            return True, True
        else:
            print(f'The graph "{graphname}" is NOT well colored with q={q}', end='')
            return False, True
    except (IOError, OSError):
        # print(f'file "{filecols}" not found')
        return False, False


def read_params(fileparams):
    fpar = open(fileparams, "r")
    fpar.readline()
    line = fpar.readline().split()
    fpar.close()
    return int(line[0]), int(line[1]), float(line[2]), float(line[3])


def read_csv_cadical(q, path_to_csv):
    filename = f'{path_to_csv}/q{q}_col_labels.csv'
    df = pd.read_csv(filename)
    return df


def is_sat(df_cadical, N, M, seed):
    row = df_cadical.loc[df_cadical['cnf_file']==f'sat_{N}_{M}_{seed - 1}.dimacs']
    if row.empty:
        return True, False 
    elif row['sat'].item() == 1:
        return True, True
    else:
        return False, True


def check_all(N_list, c_list, q, seedmin, seedmax, path_to_graph, path_to_cols_list, 
              fileout, path_to_params, ntrials, nepochs_list, str_program_list, df_cadical):
    fout = open(fileout, "w")
    fout.write("# N  c  nsamples  P(sol)  P(sol|SAT) \n")
    for j in range(len(N_list)):
        N = N_list[j]
        path_to_graph_new = path_to_graph + f'N_{N}'
        for c in c_list:
            m = int(round(N * c / 2))
            nsamples = 0
            solved = 0.0
            solved_sat = 0.0
            n_sat = 0.0
            in_cadical = True
            fileparams = f'{path_to_params}/params_paper_recurrence.txt'
            randdim, hiddim, dout, lrate = read_params(fileparams)
            for seed in range(seedmin, seedmax + 1):
                found = False
                sat, in_cadical_seed = is_sat(df_cadical, N, m, seed)
                in_cadical = in_cadical_seed and in_cadical
                l = 0
                while l < len(path_to_cols_list) and not found:
                    nepochs = nepochs_list[l][j]
                    graphname = f'ErdosRenyi_N_{N}_M_{m}_id_{seed}.txt'
                    filecols = f'{path_to_cols_list[l]}/coloring_recurrent_{str_program_list[l]}_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'
                    colored, found = check_orig(filecols, path_to_graph_new, graphname, q)
                    if found:
                        print(f'  (found in {str_program_list[l]} l={l})')
                    solved += colored
                    nsamples += found
                    if sat:
                        solved_sat += colored
                        n_sat += found
                    l += 1
            if nsamples > 0:
                if in_cadical:
                    if n_sat > 0:
                        str_solved_sat = str(solved_sat / n_sat)
                    else:
                        str_solved_sat = "AllSAT"
                else:
                    str_solved_sat = "NoData"
                fout.write(str(N) + "\t" + str(c) + "\t" + str(nsamples) + "\t" + str(solved / nsamples) + "\t" + str_solved_sat + "\n")
            else:
                print(f'No data for q={q}  N={N}  c={c}')
    fout.close()



# FOR RECURRENT CODE


N_list = [128, 256, 512, 1024, 2048, 4096, 8192]
c_list = np.arange(2.96, 5.01, 0.18)
q = 3
# c_list = np.arange(9.9, 13.5, 0.4)
# q = 5
seedmin = 1
seedmax = 400
ntrials = 5
nepochs_par = [600000, 600000, 600000, 600000, 600000, 1000000, 600000]

# Q=3
nepochs_list_cpu = [100000, 100000, 100000, 102400, 204800, 409600, 819200]

# Q=5
# nepochs_list_cpu = [102400, 102400, 102400, 102400, 204800, 409600, 819200]

nepochs_list = [nepochs_list_cpu, nepochs_list_cpu, nepochs_par]

graph_version = "New_graphs"
program_version_list = ["less_hardloss", "less_hardloss", "parallel"]
processor_list = ["CPU", "CPU", "GPU"]
cluster_list = ["_dresden", "", ""]

path_to_cols_list = []
for i in range(len(program_version_list)):
    path_to_cols_list.append(f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/{processor_list[i]}/{program_version_list[i]}/q_{q}/{graph_version}/colorings{cluster_list[i]}')


path_to_graph = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/random_graphs/ErdosRenyi/{graph_version}/'

path_to_params = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/params"

path_out = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/Mixed/q_{q}/Stats/'
fileout = path_out + f'Solved_recurrent_mixed_q_{q}_ErdosRenyi_ntrials_{ntrials}.txt'

path_to_csv = '/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/CaDiCal'
df_cadical = read_csv_cadical(q, path_to_csv)

solv_frac = check_all(N_list, c_list, q, seedmin, seedmax, path_to_graph, path_to_cols_list,
                      fileout, path_to_params, ntrials, nepochs_list, program_version_list, 
                      df_cadical)