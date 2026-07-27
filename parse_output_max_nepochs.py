import numpy as np
import csv
from os.path import isfile


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


def parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others_gpu, path_to_others_cpu,
              fileout, path_to_params, ntrials, nepochs_list_gpu, 
              nepochs_list_cpu, fileparams):
    randdim, hiddim, dout, lrate = read_params(f'{path_to_params}/{fileparams}')
    fout = open(fileout, "w")
    writer = csv.writer(fout)
    writer.writerow(["N", "M",  "id",  "E",  "ntrials"])
    for j in range(len(N_list)):
        N = N_list[j]
        for c in c_list:
            nsamples = 0
            for seed in range(seedmin, seedmax + 1):
                m = int(round(N * c / 2))
                graphname = f'ErdosRenyi_N_{N}_M_{m}_id_{seed}.txt'
                found = False
                nepochs_max = 0
                fileothers_max = ""
                l = 0
                while l < len(path_to_others_gpu):
                    nepochs = nepochs_list_gpu[j]
                    if nepochs > nepochs_max:
                        fileothers = f'{path_to_others_gpu[l]}/others_recurrent_less_hardloss_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'                    
                        if isfile(fileothers):
                            nepochs_max = nepochs
                            fileothers_max = fileothers
                    l += 1
                l = 0
                while l < len(path_to_others_cpu):
                    nepochs = nepochs_list_cpu[j]
                    if nepochs > nepochs_max:
                        fileothers = f'{path_to_others_cpu[l]}/others_recurrent_less_hardloss_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'
                        if isfile(fileothers):
                            nepochs_max = nepochs
                            fileothers_max = fileothers
                    l += 1                                    
                e, found = read_others(fileothers_max)
                nsamples += found
                if found:
                    writer.writerow([N, m, seed, e, ntrials])


            if nsamples > 0:
                print(f'q={q}  N={N}  c={"{0:.3f}".format(c)}  Nsamples={nsamples}  nepochs={nepochs_max}')
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

nepochs_list_gpu = [1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600]

# Q=3
nepochs_list_cpu = [100000, 100000, 100000, 100000, 100000, 100000, 102400, 204800, 409600]

# Q=5
# nepochs_list_cpu = [102400, 102400, 102400, 102400, 102400, 102400, 102400, 204800, 409600, 819200]

graph_version = "New_graphs"
cluster_list = ["_dresden", ""]

path_to_others_gpu = []
for i in range(len(cluster_list)):
    path_to_others_gpu.append(f'/run/media/david/Data/Research/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/GPU/single_graph/less_hardloss/q_{q}/{graph_version}/others{cluster_list[i]}')


path_to_others_cpu = []
for i in range(len(cluster_list)):
    path_to_others_cpu.append(f'/run/media/david/Data/Research/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/CPU/less_hardloss/q_{q}/{graph_version}/others{cluster_list[i]}')


path_to_params = "/run/media/david/Data/Research/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/params"

path_out = f'/run/media/david/Data/Research/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/Mixed/q_{q}/Stats/'
fileout = path_out + f'{q}COL_rPI-GNN_ntrials={ntrials}_hiddim=50_nlayers=2_max_nepochs.csv'

fileparams = "params_paper_recurrence.txt"
parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others_gpu, path_to_others_cpu, 
          fileout, path_to_params, ntrials, nepochs_list_gpu, nepochs_list_cpu, 
          fileparams)