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
    j = fin.readline()
    if not j:
        print(f'file "{fileothers}" is empty')
        return -1, -1, -1, False
    line = j.split()
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


def parse_all_old(N_list, c_list, q, seedmin, seedmax, path_to_others, 
                  fileout, path_to_params, ntrials):
    fout = open(fileout, "w")
    fout.write("# N  c  nsamples  P(sol)   av(E)   std(E)   av(runtime)[s]  std(runtime)[s]    av(nepochs)    std(nepochs)\n")
    for N in N_list:
        for c in c_list:
            nsamples = 0
            solved = 0.0
            fileparams = f'{path_to_params}/params_paper_recurrence.txt'
            randdim, hiddim, dout, lrate = read_params(fileparams)
            av_e = 0.0
            av_e_sqr = 0.0
            av_runtime = 0.0
            av_runtime_sqr = 0.0
            av_nep = 0.0
            av_nep_sqr = 0.0
            for seed in range(seedmin, seedmax + 1):
                graphname = f'ErdosRenyi_N_{N}_c_{"{0:.3f}".format(c)}_id_{seed}.txt'
                fileothers = f'{path_to_others}/others_recurrent_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_filename_{graphname}'
                e, runtime, nep, found = read_others(fileothers)
                if found:
                    print(f'file {fileothers} read')
                    nsamples += 1
                    if e == 0:
                        solved += 1
                    av_e += e
                    av_e_sqr += e ** 2
                    av_runtime += runtime
                    av_runtime_sqr += runtime ** 2
                    av_nep += nep
                    av_nep_sqr += nep ** 2
            if nsamples > 0:
                av_e /= nsamples
                av_e_sqr /= nsamples
                av_runtime /= nsamples
                av_runtime_sqr /= nsamples
                av_nep /= nsamples
                av_nep_sqr /= nsamples

                std_e = np.sqrt((av_e_sqr - av_e * av_e) / nsamples)
                std_runtime = np.sqrt((av_runtime_sqr - av_runtime * av_runtime) / nsamples)
                std_nep = np.sqrt((av_nep_sqr - av_nep * av_nep) / nsamples)

                fout.write(str(N) + "\t" + str(c) + "\t" + str(nsamples) + "\t" + str(solved / nsamples) + "\t" + str(av_e) + "\t" + str(std_e)
                            + "\t" + str(av_runtime) + "\t" + str(std_runtime) + "\t" + str(av_nep) + "\t" + str(std_nep) + "\n")
    fout.close()


def parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others, 
              fileout, path_to_params, ntrials, nepochs):
    fout = open(fileout, "w")
    fout.write("# N  c  nsamples  P(sol)   av(E)   std(E)   av(runtime)[s]  std(runtime)[s]    av(nepochs)    std(nepochs)\n")
    for N in N_list:
        for c in c_list:
            nsamples = 0
            solved = 0.0
            fileparams = f'{path_to_params}/params_paper_recurrence.txt'
            randdim, hiddim, dout, lrate = read_params(fileparams)
            av_e = 0.0
            av_e_sqr = 0.0
            av_runtime = 0.0
            av_runtime_sqr = 0.0
            av_nep = 0.0
            av_nep_sqr = 0.0
            for seed in range(seedmin, seedmax + 1):
                m = int(round(N * c / 2))
                graphname = f'ErdosRenyi_N_{N}_M_{m}_id_{seed}.txt'
                fileothers = f'{path_to_others}/others_recurrent_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'
                e, runtime, nep, found = read_others(fileothers)
                if found:
                    print(f'file {fileothers} read')
                    nsamples += 1
                    if e == 0:
                        solved += 1
                    av_e += e
                    av_e_sqr += e ** 2
                    av_runtime += runtime
                    av_runtime_sqr += runtime ** 2
                    av_nep += nep
                    av_nep_sqr += nep ** 2
            if nsamples > 0:
                av_e /= nsamples
                av_e_sqr /= nsamples
                av_runtime /= nsamples
                av_runtime_sqr /= nsamples
                av_nep /= nsamples
                av_nep_sqr /= nsamples

                std_e = np.sqrt((av_e_sqr - av_e * av_e) / nsamples)
                std_runtime = np.sqrt((av_runtime_sqr - av_runtime * av_runtime) / nsamples)
                std_nep = np.sqrt((av_nep_sqr - av_nep * av_nep) / nsamples)

                fout.write(str(N) + "\t" + str("{0:.3f}".format(c)) + "\t" + str(nsamples) + "\t" + str(solved / nsamples) + "\t" + str(av_e) + "\t" + str(std_e)
                            + "\t" + str(av_runtime) + "\t" + str(std_runtime) + "\t" + str(av_nep) + "\t" + str(std_nep) + "\n")
    fout.close()


def parse_all_varnep(N_list, c_list, q, seedmin, seedmax, path_to_others, 
              fileout, path_to_params, ntrials, nepochs_list, str_program):
    fout = open(fileout, "w")
    fout.write("# N  c  nsamples  P(sol)   av(E)   std(E)   av(runtime)[s]  std(runtime)[s]    av(nepochs)    std(nepochs)\n")
    for j in range(len(N_list)):
        N = N_list[j]
        nepochs = nepochs_list[j]
        for c in c_list:
            nsamples = 0
            solved = 0.0
            fileparams = f'{path_to_params}/params_paper_recurrence.txt'
            randdim, hiddim, dout, lrate = read_params(fileparams)
            av_e = 0.0
            av_e_sqr = 0.0
            av_runtime = 0.0
            av_runtime_sqr = 0.0
            av_nep = 0.0
            av_nep_sqr = 0.0
            for seed in range(seedmin, seedmax + 1):
                m = int(round(N * c / 2))
                graphname = f'ErdosRenyi_N_{N}_M_{m}_id_{seed}.txt'
                fileothers = f'{path_to_others}/others_recurrent_{str_program}_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs}_filename_{graphname}'
                e, runtime, nep, found = read_others(fileothers)
                if found:
                    print(f'file {fileothers} read')
                    nsamples += 1
                    if e == 0:
                        solved += 1
                    av_e += e
                    av_e_sqr += e ** 2
                    av_runtime += runtime
                    av_runtime_sqr += runtime ** 2
                    av_nep += nep
                    av_nep_sqr += nep ** 2
            if nsamples > 0:
                av_e /= nsamples
                av_e_sqr /= nsamples
                av_runtime /= nsamples
                av_runtime_sqr /= nsamples
                av_nep /= nsamples
                av_nep_sqr /= nsamples

                std_e = np.sqrt((av_e_sqr - av_e * av_e) / nsamples)
                std_runtime = np.sqrt((av_runtime_sqr - av_runtime * av_runtime) / nsamples)
                std_nep = np.sqrt((av_nep_sqr - av_nep * av_nep) / nsamples)

                fout.write(str(N) + "\t" + str("{0:.3f}".format(c)) + "\t" + str(nsamples) + "\t" + str(solved / nsamples) + "\t" + str(av_e) + "\t" + str(std_e)
                            + "\t" + str(av_runtime) + "\t" + str(std_runtime) + "\t" + str(av_nep) + "\t" + str(std_nep) + "\n")
    fout.close()


graph_version = "New_graphs"
processor = "CPU"
program_version = "less_hardloss"



N_list = [128, 256, 512, 1024, 2048, 4096, 8192]
# N_list = [2048, 4096, 8192]
c_list = np.arange(3.32, 5.01, 0.18)
q = 3
# c_list = np.arange(9.9, 13.5, 0.4)
# q = 5
seedmin = 1
seedmax = 400
ntrials = 5
# nepochs = int(1e5)
nepochs_list = 100 * np.array(N_list)

path_to_graph = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/random_graphs/ErdosRenyi/{graph_version}/'

path_to_others = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/{processor}/{program_version}/q_{q}/{graph_version}/others'
# path_to_others = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/Profiling/times_full/q_{q}/others'

path_to_params = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/params"

path_out = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/{processor}/{program_version}/q_{q}/{graph_version}/Stats/'
# path_out = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/Profiling/times_full/q_{q}/Stats/'
# fileout = path_out + f'Full_Stats_recurrent_q_{q}_ErdosRenyi_ntrials_{ntrials}_nep_{nepochs}.txt'
# fileout = path_out + f'Full_Stats_recurrent_q_{q}_ErdosRenyi_ntrials_{ntrials}_nep_{nepochs}_large_N.txt'
fileout = path_out + f'Full_Stats_recurrent_q_{q}_ErdosRenyi_ntrials_{ntrials}_nep_100N.txt'

# parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others, fileout, path_to_params, 
        #   ntrials, nepochs)


# parse_all_old(N_list, c_list, q, seedmin, seedmax, path_to_others,
            #   fileout, path_to_params, ntrials)


parse_all_varnep(N_list, c_list, q, seedmin, seedmax, path_to_others, fileout, path_to_params, 
          ntrials, nepochs_list, program_version)