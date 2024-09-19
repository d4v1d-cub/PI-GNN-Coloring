import numpy as np


def read_params(fileparams):
    fpar = open(fileparams, "r")
    fpar.readline()
    line = fpar.readline().split()
    fpar.close()
    return int(line[0]), int(line[1]), float(line[2]), float(line[3])


def average_all_params(N_list, c_list, q, model, fileout, path_to_params, nep_hyper, ngr_hyper, 
                       ntr_hyper):
    av_ed = 0
    av_ed_sqr = 0
    av_hd = 0
    av_hd_sqr = 0
    av_dout = 0
    av_dout_sqr = 0
    av_lr = 0
    av_lr_sqr = 0 
    counter = 0
    for N in N_list:
        for c in c_list:
            fileparams = f'{path_to_params}/best_params_q_{q}_N_{N}_c_{"{0:.2f}".format(c)}_model_{model}_nepochs_{nep_hyper}_ngraphs_{ngr_hyper}_ntrials_{ntr_hyper}.txt'
            embdim, hiddim, dout, lrate = read_params(fileparams)
            av_ed += embdim
            av_ed_sqr += embdim * embdim
            av_hd += hiddim
            av_hd_sqr += hiddim * hiddim
            av_dout += dout
            av_dout_sqr += dout * dout
            av_lr += lrate
            av_lr_sqr += lrate * lrate
            counter += 1
    av_ed /= counter
    av_ed_sqr /= counter
    av_hd /= counter
    av_hd_sqr /= counter
    av_dout /= counter
    av_dout_sqr /= counter
    av_lr /= counter
    av_lr_sqr /= counter

    err_ed = np.sqrt((av_ed_sqr - av_ed * av_ed) / counter)
    err_hd = np.sqrt((av_hd_sqr - av_hd * av_hd) / counter)
    err_dout = np.sqrt((av_dout_sqr - av_dout * av_dout) / counter)
    err_lr = np.sqrt((av_lr_sqr - av_lr * av_lr) / counter)

    fout = open(fileout, "w")
    fout.write(f'embdim\t{av_ed}\t{err_ed}\n')
    fout.write(f'hiddim\t{av_hd}\t{err_hd}\n')
    fout.write(f'dout\t{av_dout}\t{err_dout}\n')
    fout.write(f'lrate\t{av_lr}\t{err_lr}\n')
    fout.close()


N_list = [16, 32, 64, 128, 256]
c_list = np.arange(2.96, 5.01, 0.18)
q = 3
model = "GraphSAGE"

path_to_params = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/hyperopt/opt_all"
nep_hyper = "1e2"
ngr_hyper = 20
ntr_hyper = 1000

path_out = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/hyperopt/opt_all/"
fileout = path_out + f'Average_params_q_{q}_ErdosRenyi_model_{model}_nephyp_{nep_hyper}_ngrhyp_{ngr_hyper}_ntrhyp_{ntr_hyper}.txt'

average_all_params(N_list, c_list, q, model, fileout, path_to_params, nep_hyper, ngr_hyper, ntr_hyper)