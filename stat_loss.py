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
                           model, path_out, maxniter, path_to_params, nep_hyper, ngr_hyper, 
                           ntr_hyper):
    for N in N_list:
        for c in c_list:
            nsamples = 0
            all_soft = [[] for i in range(maxniter)]
            all_hard = [[] for i in range(maxniter)]
            fileparams = f'{path_to_params}/best_params_q_{q}_N_{N}_c_{"{0:.2f}".format(c)}_model_{model}_nepochs_{nep_hyper}_ngraphs_{ngr_hyper}_ntrials_{ntr_hyper}.txt'
            embdim, hiddim, dout, lrate = read_params(fileparams)
            
            for seed in range(seedmin, seedmax + 1):
                graphname = f'ErdosRenyi_N_{N}_c_{"{0:.3f}".format(c)}_id_{seed}.txt'
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
                fileout = f'{path_out}/Loss_Stat_q_{q}_N_{N}_c_{"{0:.2f}".format(c)}_model_{model}_embdim_{embdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_nsamples_{nsamples}.txt'
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


def get_times(maxniter): #IGNORE
    epoch_i = 0
    times = []
    while epoch_i < 100:
        times.append(epoch_i)
        epoch_i += 1
    while epoch_i < maxniter:
        times.append(epoch_i)
        epoch_i = int(epoch_i * 1.02)
    return times


def process_files_rec(N_list, c_list, q, seedmin, seedmax, path_to_loss, 
                      path_out, maxniter, path_to_params, ntrials):
    times = get_times(maxniter)
    for N in N_list:
        for c in c_list:
            nsamples = 0
            all_soft = [[] for _ in range(len(times))]
            all_hard = [[] for _ in range(len(times))]
            fileparams = f'{path_to_params}/params_paper_recurrence.txt'
            randdim, hiddim, dout, lrate = read_params(fileparams)
            
            for seed in range(seedmin, seedmax + 1):
                graphname = f'ErdosRenyi_N_{N}_c_{"{0:.3f}".format(c)}_id_{seed}.txt'
                filein = f'{path_to_loss}/loss_recurrent_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_filename_{graphname}'
                try:
                    fin = open(filein, "r")
                    print(f'Reading file "{filein}"')
                    i = 0
                    while True:
                        j = fin.readline()
                        if not j:
                            break
                        line = j.split()
                        all_soft[i].append(float(line[1]))
                        all_hard[i].append(float(line[2]))
                        i += 1
                    fin.close()
                    nsamples += 1
                except (IOError, OSError):
                    print(f'file  "{filein}"  not read')
            if nsamples > 0:
                fileout = f'{path_out}/Loss_Stat_recurrent_q_{q}_N_{N}_c_{"{0:.2f}".format(c)}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nsamples_{nsamples}.txt'
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
                    fout.write(str(times[i]) + "\t" + str(soft_av) + "\t" + str(soft_min) + "\t" + str(hard_av) + "\t" + str(hard_min) + "\n")
                fout.close()


# N_list = [16, 32, 64, 128, 256]
# c_list = np.arange(2.96, 5.00, 0.18)
# q = 3
# seedmin = 1
# seedmax = 201
# model = "GraphSAGE"
# maxniter = int(1e5)

# path_to_loss = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Opt_params/losses"

# path_out = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Opt_params/Stats/"

# path_to_params = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/hyperopt"
# nep_hyper = "1e2"
# ngr_hyper = 20
# ntr_hyper = 1000


# solv_frac = process_files_opt_pars(N_list, c_list, q, seedmin, seedmax, path_to_loss,
#                                    model, path_out, maxniter, path_to_params, nep_hyper, ngr_hyper, ntr_hyper)


N_list = [16, 32, 64, 128, 256]
# c_list = np.arange(2.96, 5.00, 0.18)
# q = 3
c_list = np.arange(9.1, 13.5, 0.4)
q = 5
seedmin = 1
seedmax = 401
maxniter = int(1e5)
ntrials = 5

path_to_loss = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/q_5/losses"

path_out = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/q_5/Stats/"

path_to_params =  "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/params"


solv_frac = process_files_rec(N_list, c_list, q, seedmin, seedmax, path_to_loss, path_out, maxniter, path_to_params, ntrials)
