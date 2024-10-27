import numpy as np


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
        print(f'file "{fileothers}" not read')
        return -1, -1, -1, False
    line = fin.readline().split()
    return int(line[0]), float(line[1]), int(line[3]), True


def read_loss(fileloss):
    try:
        fin = open(fileloss, "r")
    except (IOError, OSError):
        print(f'file "{fileloss}" not read')
        return -1, False
    while True:
        j = fin.readline()
        if not j:
            break
        line = j.split()
    
    return int(line[0]), True


def parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others, 
              fileout, path_to_params, ntrials, nepochs):
    fout = open(fileout, "w")
    fout.write("# N  c  id  E  runtime  nepochs  ntrials   max_nepochs\n")
    for N in N_list:
        for c in c_list:
            fileparams = f'{path_to_params}/params_paper_recurrence.txt'
            randdim, hiddim, dout, lrate = read_params(fileparams)
            for seed in range(seedmin, seedmax + 1):
                m = int(round(N * c / 2))
                graphname = f'ErdosRenyi_N_{N}_M_{m}_id_{seed}.txt'
                fileothers = f'{path_to_others}/others_recurrent_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'
                e, runtime, nep, found = read_others(fileothers)
                if found:
                    print(f'file {fileothers} read')
                    fout.write(f'{N}\t{"{0:.3f}".format(c)}\t{seed}\t{e}\t{runtime}\t{nep}\t{ntrials}\t{nepochs}\n')
    fout.close()


version = "New_graphs"
processor = "CPU"


N_list = [128, 256, 512, 1024]
# c_list = np.arange(3.32, 5.01, 0.18)
# q = 3
c_list = np.arange(9.9, 13.5, 0.4)
q = 5
seedmin = 1
seedmax = 400
ntrials = 5
nepochs = int(1e5)

path_to_graph = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/random_graphs/ErdosRenyi/{version}/'

path_to_others = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/{processor}/q_{q}/{version}/others'

path_to_params = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/params"

path_out = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/{processor}/q_{q}/{version}/Stats/'
fileout = path_out + f'Data_each_graph_recurrent_q_{q}_ErdosRenyi_ntrials_{ntrials}_nep_{nepochs}.txt'

parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others, fileout, path_to_params, ntrials, nepochs)