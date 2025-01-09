__author__ = 'david'

import numpy as np
from scipy.optimize import leastsq


def read_data(path_to_data, filein, n_min, n_max, c_min, c_max, column):
    fin = open(f'{path_to_data}/{filein}', "r")
    fin.readline()
    c_list = []
    n_list = []
    y_data = []
    sigma_y = []
    while True:
        j = fin.readline()
        if not j:
            break
        line = j.split()
        n = int(line[0])
        c = float(line[1])
        if n_min <= n <= n_max:
            if c_min <= c <= c_max:
                if c not in c_list:
                    c_list.append(c)
                    n_list.append([])
                    y_data.append([])
                    sigma_y.append([])
                    pos = len(n_list) - 1
                    n_list[pos].append(n)
                    y_data[pos].append(float(line[column]))
                    sigma_y[pos].append(float(line[column + 1]))
                else:
                    pos = c_list.index(c)
                    n_list[pos].append(n)
                    y_data[pos].append(float(line[column]))
                    sigma_y[pos].append(float(line[column + 1]))
    fin.close()
    n_list = np.array(n_list)
    y_data = np.array(y_data)
    sigma_y = np.array(sigma_y)
    return np.log(n_list), np.log(y_data), sigma_y / y_data 



def linear_function(x, slope, intercept):
    return slope * x + intercept


def get_predictions(x_data, y_data, slope, all_intercepts):
    pred = np.zeros(y_data.shape)
    for l in range(len(y_data)):
        for p in range(len(y_data[l])):
            pred[l, p] = linear_function(x_data[l][p], slope, all_intercepts[l])
    return pred


def differences(params, x_data, y_data, sigma_y):
    slope = params[0]
    all_intercepts = params[1:]
    pred = get_predictions(x_data, y_data, slope, all_intercepts)
    diffs = (pred - y_data) / sigma_y
    return diffs.flatten()


def make_fit(slope0, intercepts0, x_data, y_data, sigma_y):
    params = np.zeros(len(intercepts0) + 1)
    params[0] = slope0
    params[1:] = intercepts0
    out = leastsq(differences, params, args=(x_data, y_data, sigma_y), full_output=True)
    opt_pars = out[0]
    std_pars = np.sqrt(np.diag(out[1]))
    chi_sqr = np.sum(out[2]['fvec'] ** 2) / (x_data.size - len(params))
    return opt_pars, std_pars, chi_sqr


def print_pars(pars, std, chi_sqr, path_out, fileout):
    fout = open(f'{path_out}/{fileout}', "w")
    fout.write("# chi_sqr  slope  std(slope)  intercepts...   std(intercepts...)\n")
    fout.write(str(chi_sqr))
    for i in range(len(pars)):
        fout.write(f'\t{pars[i]}\t{std[i]}')
    fout.write("\n")
    fout.close()


def sample_data_from_fit(params, n_max, path, filedata):
    w = open(f'{path}/{filedata}', "w")    
    nvals = np.arange(0, n_max, step=0.1)
    slope = params[0]
    intercepts = params[1:]
    for n in nvals:
        w.write(str(n))
        for l in range(len(intercepts)):
            w.write(f'\t{np.exp(intercepts[l]) * (n ** slope)}')
        w.write("\n")
    w.close()




def main():
    q = 3
    # ntrials = 'combined'
    ntrials = 5
    # nep = int(1e5)
    nep = "100N"
    n_min = 1024
    c_min = 3.68
    c_max = 3.68
    data_kind = 'nepochs'
    n_max_read = 30000
    n_max_sample = 40000

    if data_kind == 'nepochs':
        column = 12
    elif data_kind == 'runtime':
        column = 8
    else:
        print(f'Unknown column for the data on: {data_kind}')

    graph_version = "New_graphs"
    # processor = "CPU"
    processor = "GPU"
    # program_version = "less_hardloss"
    program_version = "all_hardloss"
    parallel_str = "single_graph/"
    
    path_to_data = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/{processor}/{parallel_str}{program_version}/q_{q}/{graph_version}/Stats'

    filein = f'Full_Stats_recurrent_q_{q}_ErdosRenyi_ntrials_{ntrials}_nep_{nep}.txt'
    fileout = f'fit_pars_{data_kind}_recurrent_q_{q}_ErdosRenyi_ntrials_{ntrials}_nep_{nep}_Nmin_{n_min}_cmin_{c_min}_cmax_{c_max}.txt'
    filedata = f'fit_samples_{data_kind}_recurrent_q_{q}_ErdosRenyi_ntrials_{ntrials}_nep_{nep}_Nmin_{n_min}_cmin_{c_min}_cmax_{c_max}.txt'

    x_data, y_data, sigma_y = read_data(path_to_data, filein, n_min, n_max_read, c_min, c_max, column)
    slope0 = 1
    intercepts0 = np.ones(len(x_data))

    opt_pars, std_pars, chi_sqr = make_fit(slope0, intercepts0, x_data, y_data, sigma_y)
    print("Parameters:", opt_pars)
    print("Errors:", std_pars)
    print("Chi squared:", chi_sqr)

    print_pars(opt_pars, std_pars, chi_sqr, path_to_data, fileout)
    sample_data_from_fit(opt_pars, n_max_sample, path_to_data, filedata)


    return 0


if __name__ == '__main__':
    main()
