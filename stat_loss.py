import ast
import numpy as np


def process_files(N_list, c_list, q, seedmin, seedmax, path_to_loss, 
                  model, embdim, hiddim, path_out, niter):
    for N in N_list:
        for c in c_list:
            nsamples = 0
            all_soft = [[] for i in range(niter)]
            all_hard = [[] for i in range(niter)]
            for seed in range(seedmin, seedmax + 1):
                graphname = f'ErdosRenyi_N_{N}_c_{"{0:.3f}".format(c)}_seed_{seed}.txt'
                filein = f'{path_to_loss}/loss_q_{q}_model_{model}_embdim_{embdim}_hidim_{hiddim}_filename_{graphname}'
                try:
                    fin = open(filein, "r")
                    print(f'Reading file "{filein}"')
                    j = fin.readline()
                    losses = ast.literal_eval(j)
                    for i in range(len(losses)):
                        all_soft[i].append(losses[i][1])
                        all_hard[i].append(losses[i][2])
                    fin.close()
                    nsamples += 1
                except (IOError, OSError):
                    print(f'file  "{filein}"  not read')
            if nsamples > 0:
                fileout = f'{path_out}/Loss_Stat_q_{q}_N_{N}_c_{c}_model_{model}_embdim_{embdim}_hidim_{hiddim}_nsamples_{nsamples}.txt'
                fout = open(fileout, "w")
                fout.write("# iter  av(soft_loss)  min(soft_loss)  av(hard_loss)  min(hard_loss)  \n")
                for i in range(len(all_soft)):
                    soft_av = np.mean(all_soft[i])
                    soft_min = np.min(all_soft[i])
                    hard_av = np.mean(all_hard[i])
                    hard_min = np.min(all_hard[i])
                    fout.write(str(i) + "\t" + str(soft_av) + "\t" + str(soft_min) + "\t" + str(hard_av) + "\t" + str(hard_min) + "\n")
                fout.close()


N_list = [32, 64, 128, 256]
c_list = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
q = 3
seedmin = 1
seedmax = 201
model = "GraphSAGE"
embdim = 80
hiddim = 80
niter = int(1e5)

path_to_cols = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/losses"

path_out = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Stats/"

solv_frac = process_files(N_list, c_list, q, seedmin, seedmax, path_to_cols,
                          model, embdim, hiddim, path_out, niter)