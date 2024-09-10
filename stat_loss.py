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
                    if len(all_soft[i]) > 0:
                        soft_av = np.mean(all_soft[i])
                        soft_min = np.min(all_soft[i])
                        hard_av = np.mean(all_hard[i])
                        hard_min = np.min(all_hard[i])
                    else:
                        soft_av = 0
                        soft_min = 0
                        hard_av = 0
                        hard_min = 0
                    fout.write(str(i) + "\t" + str(soft_av) + "\t" + str(soft_min) + "\t" + str(hard_av) + "\t" + str(hard_min) + "\n")
                fout.close()


def read_params(fileparams):
    fpar = open(fileparams, "r")
    fpar.readline()
    line = fpar.readline().split()
    fpar.close()
    return int(line[0]), int(line[1]), float(line[2]), float(line[3])


def process_files_opt_pars(N_list, c_list, q, seedmin, seedmax, path_to_loss, 
                           model, path_out, niter, path_to_params, nep_hyper, ngr_hyper, 
                           ntr_hyper):
    for N in N_list:
        for c in c_list:
            nsamples = 0
            all_soft = [[] for i in range(niter)]
            all_hard = [[] for i in range(niter)]
            fileparams = f'{path_to_params}/best_params_q_{q}_N_{N}_c_{c}_model_{model}_nepochs_{nep_hyper}_ngraphs_{ngr_hyper}_ntrials_{ntr_hyper}.txt'
            embdim, hiddim, dout, lrate = read_params(fileparams)
            
            for seed in range(seedmin, seedmax + 1):
                graphname = f'ErdosRenyi_N_{N}_c_{"{0:.3f}".format(c)}_seed_{seed}.txt'
                filein = f'{path_to_loss}/loss_q_{q}_model_{model}_embdim_{embdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_filename_{graphname}'
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
                fileout = f'{path_out}/Loss_Stat_q_{q}_N_{N}_c_{c}_model_{model}_embdim_{embdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_nsamples_{nsamples}.txt'
                fout = open(fileout, "w")
                fout.write("# iter  av(soft_loss)  min(soft_loss)  av(hard_loss)  min(hard_loss)  \n")
                for i in range(len(all_soft)):
                    if len(all_soft[i]) > 0:
                        soft_av = np.mean(all_soft[i])
                        soft_min = np.min(all_soft[i])
                        hard_av = np.mean(all_hard[i])
                        hard_min = np.min(all_hard[i])
                    else:
                        soft_av = 0
                        soft_min = 0
                        hard_av = 0
                        hard_min = 0
                    fout.write(str(i) + "\t" + str(soft_av) + "\t" + str(soft_min) + "\t" + str(hard_av) + "\t" + str(hard_min) + "\n")
                fout.close()


# N_list = [32, 64, 128, 256]
N_list = [64, 128, 256]
c_list = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
q = 3
seedmin = 1
seedmax = 201
model = "GraphSAGE"
embdim = 80
hiddim = 80
niter = int(1e5)

path_to_cols = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Opt_params/losses"

path_out = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Opt_params/Stats/"

path_to_params = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/hyperopt"
nep_hyper = "1e2"
ngr_hyper = 20
ntr_hyper = 250


solv_frac = process_files_opt_pars(N_list, c_list, q, seedmin, seedmax, path_to_cols,
                                   model, path_out, niter, path_to_params, nep_hyper, ngr_hyper, ntr_hyper)