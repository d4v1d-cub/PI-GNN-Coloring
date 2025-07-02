import numpy as np
import csv


def read_params(fileparams):
    fpar = open(fileparams, "r")
    fpar.readline()
    line = fpar.readline().split()
    fpar.close()
    return int(line[0]), int(line[1]), float(line[2]), float(line[3])


def read_others(fileothers):
    try:
        fin = open(fileothers, "r")
    except (IOError, OSError):
        return -1, False
    line = fin.readline().split()
    return int(line[0]), True


def read_coloring(filecol):
    try:
        fin = open(filecol, "r")
    except (IOError, OSError):
        return -1, False
    fin.readline()
    line = fin.readline().split()
    return int(line[1]), True


def parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others_list, 
              fileout, path_to_params, ntrials, nepochs_list):
    fout = open(fileout, "w")
    writer = csv.writer(fout)
    writer.writerow(["N", "M",  "id",  "E",  "ntrials"])
    for j in range(len(N_list)):
        N = N_list[j]
        for c in c_list:
            fileparams = f'{path_to_params}/params_paper_recurrence.txt'
            randdim, hiddim, dout, lrate = read_params(fileparams)
            nsamples = 0
            nsampl_gpu = 0
            for seed in range(seedmin, seedmax + 1):
                found = False
                l = 0
                while l < len(path_to_others_list) and not found:
                    nepochs = nepochs_list[j]
                    m = int(round(N * c / 2))
                    graphname = f'ErdosRenyi_N_{N}_M_{m}_id_{seed}.txt'
                    fileothers = f'{path_to_others_list[l]}/others_recurrent_less_hardloss_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'
                    e, found = read_others(fileothers)
                    nsamples += found
                    nsampl_gpu += found
                    if found:
                        writer.writerow([N, m, seed, e, ntrials])
                    l += 1
            if nsamples > 0:
                print(f'q={q}  N={N}  c={"{0:.3f}".format(c)}  Nsamples={nsamples}   GPU={nsampl_gpu}')
            else:
                print(f'q={q}  N={N}  c={"{0:.3f}".format(c)}  NOT FOUND')
    fout.close()


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
cluster_list = ["", "_dresden"]

path_to_others_list = []
for i in range(len(cluster_list)):
    path_to_others_list.append(f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/GPU/single_graph/less_hardloss/q_{q}/{graph_version}/others{cluster_list[i]}')

path_to_params = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/params"

path_out = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/Mixed/q_{q}/Stats/'
fileout = path_out + f'{q}COL_rPI-GNN_ntrials={ntrials}_GPU_single_graph.csv'

parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others_list, fileout, path_to_params, ntrials, nepochs_list)