import numpy as np
import pandas as pd


def read_params(fileparams):
    fpar = open(fileparams, "r")
    fpar.readline()
    line = fpar.readline().split()
    fpar.close()
    return int(line[0]), int(line[1]), float(line[2]), float(line[3])


def read_others(fileothers, nepochs_max):
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
    seed_success = int(line[2])
    if seed_success > 0:
        return int(line[0]), float(line[1]), nepochs_max, True
    else:
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
                  fileout, path_to_params, ntrials, nepochs_max):
    fout = open(fileout, "w")
    fout.write("# N  c  nsamples  P(sol)   av(E)   std(E)   av(runtime)[s]  std(runtime)[s]   av(runtime(Solved))  str(runtime(Solved))   av(nepochs)    std(nepochs)   av(nepochs(Solved))   str(nepochs(Solved))\n")
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
            av_runtime_solved = 0.0
            av_runtime_solved_sqr = 0.0
            av_nep = 0.0
            av_nep_sqr = 0.0
            av_nep_solved = 0.0
            av_nep_solved_sqr = 0.0
            for seed in range(seedmin, seedmax + 1):
                graphname = f'ErdosRenyi_N_{N}_c_{"{0:.3f}".format(c)}_id_{seed}.txt'
                fileothers = f'{path_to_others}/others_recurrent_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_filename_{graphname}'
                e, runtime, nep, found = read_others(fileothers, nepochs_max)
                if found:
                    print(f'file {fileothers} read')
                    nsamples += 1
                    if e == 0:
                        solved += 1
                        av_runtime_solved += runtime
                        av_runtime_solved_sqr += runtime ** 2
                        av_nep_solved += nep
                        av_nep_solved_sqr += nep ** 2
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
                av_runtime_solved /= nsamples
                av_runtime_solved_sqr /= nsamples
                av_nep /= nsamples
                av_nep_sqr /= nsamples
                av_nep_solved /= nsamples
                av_nep_solved_sqr /= nsamples


                std_e = np.sqrt((av_e_sqr - av_e * av_e) / nsamples)
                std_runtime = np.sqrt((av_runtime_sqr - av_runtime * av_runtime) / nsamples)
                std_nep = np.sqrt((av_nep_sqr - av_nep * av_nep) / nsamples)
                std_runtime_solved = np.sqrt((av_runtime_solved_sqr - av_runtime_solved * av_runtime_solved) / nsamples)
                std_nep_solved = np.sqrt((av_nep_solved_sqr - av_nep_solved * av_nep_solved) / nsamples)

                fout.write(str(N) + "\t" + str(c) + "\t" + str(nsamples) + "\t" + str(solved / nsamples) + "\t" + str(av_e) + "\t" + str(std_e)
                            + "\t" + str(av_runtime) + "\t" + str(std_runtime) + "\t" + str(av_runtime_solved) + "\t" + str(std_runtime_solved)
                            + "\t" + str(av_nep) + "\t" + str(std_nep) + "\t" + str(av_nep_solved) + "\t" + str(std_nep_solved) + "\n")
    fout.close()


def parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others, 
              fileout, path_to_params, ntrials, nepochs_max):
    fout = open(fileout, "w")
    fout.write("# N  c  nsamples  P(sol)   av(E)   std(E)   av(runtime)[s]  std(runtime)[s]   av(runtime(Solved))  str(runtime(Solved))   av(nepochs)    std(nepochs)   av(nepochs(Solved))   str(nepochs(Solved))\n")
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
            av_runtime_solved = 0.0
            av_runtime_solved_sqr = 0.0
            av_nep = 0.0
            av_nep_sqr = 0.0
            av_nep_solved = 0.0
            av_nep_solved_sqr = 0.0
            for seed in range(seedmin, seedmax + 1):
                m = int(round(N * c / 2))
                graphname = f'ErdosRenyi_N_{N}_M_{m}_id_{seed}.txt'
                fileothers = f'{path_to_others}/others_recurrent_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs_max}_filename_{graphname}'
                e, runtime, nep, found = read_others(fileothers, nepochs_max)
                if found:
                    print(f'file {fileothers} read')
                    nsamples += 1
                    if e == 0:
                        solved += 1
                        av_runtime_solved += runtime
                        av_runtime_solved_sqr += runtime ** 2
                        av_nep_solved += nep
                        av_nep_solved_sqr += nep ** 2
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
                av_runtime_solved /= nsamples
                av_runtime_solved_sqr /= nsamples
                av_nep /= nsamples
                av_nep_sqr /= nsamples
                av_nep_solved /= nsamples
                av_nep_solved_sqr /= nsamples

                std_e = np.sqrt((av_e_sqr - av_e * av_e) / nsamples)
                std_runtime = np.sqrt((av_runtime_sqr - av_runtime * av_runtime) / nsamples)
                std_nep = np.sqrt((av_nep_sqr - av_nep * av_nep) / nsamples)
                std_runtime_solved = np.sqrt((av_runtime_solved_sqr - av_runtime_solved * av_runtime_solved) / nsamples)
                std_nep_solved = np.sqrt((av_nep_solved_sqr - av_nep_solved * av_nep_solved) / nsamples)

                fout.write(str(N) + "\t" + str("{0:.3f}".format(c)) + "\t" + str(nsamples) + "\t" + str(solved / nsamples) + "\t" + str(av_e) + "\t" + str(std_e)
                            + "\t" + str(av_runtime) + "\t" + str(std_runtime) + "\t" + str(av_runtime_solved) + "\t" + str(std_runtime_solved)
                            + "\t" + str(av_nep) + "\t" + str(std_nep) + "\t" + str(av_nep_solved) + "\t" + str(std_nep_solved) + "\n")
    fout.close()


def read_csv_cadical(q, path_to_csv):
    filename = f'{path_to_csv}/q{q}_col_labels.csv'
    df = pd.read_csv(filename)
    return df


def is_sat(df_cadical, N, M, seed):
    row = df_cadical.loc[df_cadical['cnf_file']==f'sat_{N}_{M}_{seed - 1}.dimacs']
    if row.empty:
        return True, False 
    elif row['sat'].item() == 1:
        return True, True
    else:
        return False, True


def parse_all_varnep(N_list, c_list, q, seedmin, seedmax, path_to_others, 
              fileout, path_to_params, ntrials, nepochs_list, str_program, df_cadical):
    fout = open(fileout, "w")
    fout.write("# N  c  nsamples  P(sol) P(sol | SAT)  av(E)   std(E)   av(runtime)[s]  std(runtime)[s]   av(runtime(Solved))  str(runtime(Solved))   av(nepochs)    std(nepochs)   av(nepochs(Solved))   str(nepochs(Solved))\n")
    for j in range(len(N_list)):
        N = N_list[j]
        nepochs_max = nepochs_list[j]
        for c in c_list:
            nsamples = 0
            solved = 0.0
            solved_sat = 0.0
            n_sat = 0.0
            in_cadical = True
            fileparams = f'{path_to_params}/params_paper_recurrence.txt'
            randdim, hiddim, dout, lrate = read_params(fileparams)
            av_e = 0.0
            av_e_sqr = 0.0
            av_runtime = 0.0
            av_runtime_sqr = 0.0
            av_runtime_solved = 0.0
            av_runtime_solved_sqr = 0.0
            av_nep = 0.0
            av_nep_sqr = 0.0
            av_nep_solved = 0.0
            av_nep_solved_sqr = 0.0
            for seed in range(seedmin, seedmax + 1):
                m = int(round(N * c / 2))
                sat, in_cadical_seed = is_sat(df_cadical, N, m, seed)
                in_cadical = in_cadical_seed and in_cadical
                graphname = f'ErdosRenyi_N_{N}_M_{m}_id_{seed}.txt'
                fileothers = f'{path_to_others}/others_recurrent{str_program}_q_{q}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntrials}_nep_{nepochs_max}_filename_{graphname}'
                e, runtime, nep, found = read_others(fileothers, nepochs_max)
                if found:
                    print(f'file {fileothers} read')
                    nsamples += 1
                    if sat:
                        n_sat += 1
                    if e == 0:
                        solved += 1
                        av_runtime_solved += runtime
                        av_runtime_solved_sqr += runtime ** 2
                        av_nep_solved += nep
                        av_nep_solved_sqr += nep ** 2
                        if sat:
                            solved_sat += 1
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
                av_runtime_solved /= nsamples
                av_runtime_solved_sqr /= nsamples
                av_nep /= nsamples
                av_nep_sqr /= nsamples
                av_nep_solved /= nsamples
                av_nep_solved_sqr /= nsamples

                std_e = np.sqrt((av_e_sqr - av_e * av_e) / nsamples)
                std_runtime = np.sqrt((av_runtime_sqr - av_runtime * av_runtime) / nsamples)
                std_nep = np.sqrt((av_nep_sqr - av_nep * av_nep) / nsamples)
                std_runtime_solved = np.sqrt((av_runtime_solved_sqr - av_runtime_solved * av_runtime_solved) / nsamples)
                std_nep_solved = np.sqrt((av_nep_solved_sqr - av_nep_solved * av_nep_solved) / nsamples)

                if in_cadical:
                    solved_sat_frac = str(solved_sat / n_sat)
                else:
                    solved_sat_frac = "ND"

                fout.write(str(N) + "\t" + str("{0:.3f}".format(c)) + "\t" + str(nsamples) + "\t" + str(solved / nsamples) + "\t" + solved_sat_frac
                            + "\t" + str(av_e) + "\t" + str(std_e)
                            + "\t" + str(av_runtime) + "\t" + str(std_runtime) + "\t" + str(av_runtime_solved) + "\t" + str(std_runtime_solved)
                            + "\t" + str(av_nep) + "\t" + str(std_nep) + "\t" + str(av_nep_solved) + "\t" + str(std_nep_solved) + "\n")
    fout.close()


graph_version = "New_graphs"
# processor = "CPU"
processor = "GPU"
program_version = "less_hardloss"
str_program = "_less_hardloss"
# program_version = "all_hardloss"
# str_program = ""
cluster=""
# cluster="_dresden"
parallel_str = "single_graph/"


N_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
# c_list = np.arange(3.32, 5.01, 0.18)
# q = 3
c_list = np.arange(9.9, 13.5, 0.4)
q = 5
seedmin = 1
seedmax = 400
ntrials = 5
# nepochs = int(1e5)

# Q=3
# nepochs_list = [100000, 100000, 100000, 102400, 204800, 409600, 819200, 1638400, 3276800]

# Q=5
# nepochs_list = [102400, 102400, 102400, 102400, 204800, 409600, 819200]
nepochs_list = [100000, 100000, 100000, 102400, 204800, 409600, 819200, 1638400, 3276800]

path_to_graph = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/random_graphs/ErdosRenyi/{graph_version}/'

path_to_others = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/{processor}/{parallel_str}{program_version}/q_{q}/{graph_version}/others{cluster}'
# path_to_others = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/Profiling/times_full/q_{q}/others'

path_to_params = "/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/params"

path_out = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/{processor}/{parallel_str}{program_version}/q_{q}/{graph_version}/Stats/'
# path_out = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/Profiling/times_full/q_{q}/Stats/'
# fileout = path_out + f'Full_Stats_recurrent_q_{q}_ErdosRenyi_ntrials_{ntrials}_nep_{nepochs}.txt'
fileout = path_out + f'Full_Stats_recurrent{cluster}_q_{q}_ErdosRenyi_ntrials_{ntrials}_nep_100N.txt'

path_to_csv = '/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/CaDiCal'
df_cadical = read_csv_cadical(q, path_to_csv)


# parse_all(N_list, c_list, q, seedmin, seedmax, path_to_others, fileout, path_to_params, 
        #   ntrials, nepochs)


parse_all_varnep(N_list, c_list, q, seedmin, seedmax, path_to_others, fileout, path_to_params, 
          ntrials, nepochs_list, str_program, df_cadical)