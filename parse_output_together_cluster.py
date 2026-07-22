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


def parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others,
              fileout, path_to_params, ntrials, nepochs_list, 
              fileparams):
    randdim, hiddim, dout, lrate = read_params(f'{path_to_params}/{fileparams}')
    fout = open(fileout, "w")
    writer = csv.writer(fout)
    writer.writerow(["N", "M",  "id",  "E",  "ntrials"])
    for j in range(len(N_list)):
        N = N_list[j]
        for c in c_list:
            nsamples = 0
            for seed in range(seedmin, seedmax + 1):
                found = False
                nepochs = nepochs_list[j]
                m = int(round(N * c / 2))
                graphname = f'ErdosRenyi_N_{N}_M_{m}_id_{seed}.txt'
                fileothers = f'{path_to_others}/others_recurrent_less_hardloss_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'
                e, found = read_others(fileothers)
                nsamples += found
                if found:
                    writer.writerow([N, m, seed, e, ntrials])
            if nsamples > 0:
                print(f'q={q}  N={N}  c={"{0:.3f}".format(c)}  Nsamples={nsamples}')
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

nepochs_list = [1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600]



path_to_others = "./"


path_to_params = "/home/2a/dm27124/PI-GNN/best_params/rec"

path_out = './'

hiddim_list = [20, 30, 40]

for hiddim in hiddim_list:

    fileout = path_out + f'{q}COL_rPI-GNN_ntrials={ntrials}_hiddim={hiddim}.csv'

    fileparams = f"params_paper_recurrence_hiddim_{hiddim}.txt"
    parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others, fileout, 
              path_to_params, ntrials, nepochs_list, fileparams)