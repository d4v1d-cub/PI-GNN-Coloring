import os

# Known chromatic numbers for specified problems (from references)
chromatic_numbers = {
    # COLOR graphs
    'jean.col': 10,
    'anna.col': 11,
    'huck.col': 11,
    'david.col': 11,
    'homer.col': 13,
    'myciel2.col': 3,
    'myciel3.col': 4,
    'myciel4.col': 5, #io
    'myciel5.col': 6,
    'myciel6.col': 7,
    'queen5_5.col': 5,
    'queen6_6.col': 7,
    'queen7_7.col': 7,
    'queen8_8.col': 9,
    'queen9_9.col': 10,
    'queen8_12.col': 12,
    'queen11_11.col': 11,
    'queen12_12.col': 12, #io
    'queen13_13.col': 13,
    'le450_5a.col': 5,#io from here
    'le450_5b.col': 5,#
    'le450_5c.col': 5,#
    'le450_5d.col': 5,#
    'le450_15a.col': 15,#
    'le450_15b.col': 15,#
    'le450_15c.col': 15,#
    'le450_15d.col': 15,#to here
    # Citations graphs
    'cora.cites': 5,
    'citeseer.cites': 6,
    'pubmed.cites': 8
}


def modify(path_old_graph, old_name, path_new_graph, new_name, chroms):
    if not os.path.exists(f'{path_new_graph}'):
        os.makedirs(f'{path_new_graph}')        
    if old_name in chroms.keys():
        try:
            fold = open(f'{path_old_graph}/{old_name}', "r")
            fnew = open(f'{path_new_graph}/{new_name}', "w") 
            while True:
                j = fold.readline()
                if not j:
                    break
                if j[0] == 'p':
                    line = j.split()
                    fnew.write(line[2] + "\n")
                    fnew.write(line[3] + "\n")
                elif j[0] != 'c':
                    line = j.split()
                    fnew.write(line[1] + "\t" + line[2] + "\n")
            fnew.write(str(chroms[old_name]))
            fold.close()
            fnew.close()
        except (OSError, IOError):
            print(old_name, " not read")
    else:
        print(f'The dictionary does not have the chromatic number for the graph {old_name}')


def mod_all(path_old_graph, path_new_graph, chroms):
    old_names = os.listdir(path_old_graph)
    for name in old_names:
        splitted = name.split(".")
        newname = splitted[0]
        for i in range(1, len(splitted) - 1):
            newname += "." + splitted[i]
        newname += "_mod." + splitted[-1]
        modify(path_old_graph, name, path_new_graph, newname, chroms)

path_old = "./Original/data/input/COLOR/all_graphs"
path_new = "./Decoupled/data/all_graphs"
mod_all(path_old, path_new, chromatic_numbers) 