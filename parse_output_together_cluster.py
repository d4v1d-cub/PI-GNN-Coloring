import numpy as np
import csv


def read_params(fileparams):
    fpar = open(fileparams, "r")
    fpar.readline()
    line = fpar.readline().split()
    fpar.close()
    if len(line) == 4:
        return int(line[0]), int(line[1]), float(line[2]), float(line[3])
    elif len(line) == 5:
        return int(line[0]), int(line[1]), float(line[2]), float(line[3]), int(line[4])
    else:
        print(f"# The number of parameters in the file {fileparams} is not 4 nor 5")
        return line


def read_others(fileothers):
    try:
        fin = open(fileothers, "r")
    except (IOError, OSError):
        return -1, False
    line = fin.readline().split()
    return int(line[0]), True



def parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others,
              fileout, path_to_params, ntrials, nepochs_list, 
              fileparams):
    params = read_params(f'{path_to_params}/{fileparams}')
    if len(params) == 4:
        randdim, hiddim, dout, lrate = params
        string_nlayers = "" 
    elif len(params)== 5:
        randdim, hiddim, dout, lrate, nlayers = params
        string_nlayers = f"_nlayers_{nlayers}" 
    else:
        return
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
                fileothers = f'{path_to_others}/others_recurrent_less_hardloss_q_{q}{string_nlayers}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'
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

path_out = '../../../../'

print("# Processing nlayers=2")
hiddim_list = [20, 30, 40]

for hiddim in hiddim_list:

    fileout = path_out + f'{q}COL_rPI-GNN_ntrials={ntrials}_hiddim={hiddim}_nlayers=2.csv'

    fileparams = f"params_paper_recurrence_hiddim_{hiddim}.txt"
    parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others, fileout, 
              path_to_params, ntrials, nepochs_list, fileparams)

print("\n")

hiddim_list_morelayers = [50]
nlayers_list = [3]
for nlayers in nlayers_list:
    print(f"# Processing nlayers={nlayers}")
    for hiddim in hiddim_list_morelayers:
        fileout = path_out + f'{q}COL_rPI-GNN_ntrials={ntrials}_hiddim={hiddim}_nlayers={nlayers}.csv'

        fileparams = f"params_paper_recurrence_hiddim_{hiddim}_nlayers_{nlayers}.txt"
        parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others, fileout, 
              path_to_params, ntrials, nepochs_list, fileparams)