
def compute_score(filein):
    fin = open(filein, "r")
    fin.readline()
    nsat = 0.0
    nsolved = 0.0
    while True:
        j = fin.readline()
        if not j:
            break
        line = j.split()
        if line[-3] != 'NoData' and line[-3] != 'AllUNSAT':
            nsat += float(line[-2])
            nsolved += float(line[-1]) 
    fin.close()
    return nsolved / nsat



# FOR RECURRENT CODE


q = 5
ntrials = 5

graph_version = "New_graphs"
program_version_list = ["less_hardloss", "less_hardloss", "parallel"]
processor_list = ["CPU", "CPU", "GPU"]
cluster_list = ["_dresden", "", ""]

path_in = f'/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/Results/Recurrent/random_graphs/Mixed/q_{q}/Stats/'
filein = path_in + f'Solved_recurrent_mixed_q_{q}_ErdosRenyi_ntrials_{ntrials}.txt'

solved_sat_frac = compute_score(filein)
print(solved_sat_frac)