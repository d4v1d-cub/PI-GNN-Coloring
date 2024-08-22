import networkx as nx


def parse_line(file_line, node_offset=0):
    x, y = [int(x) for x in file_line.split()]
    x, y = x+node_offset, y+node_offset  # nodes in file are 1-indexed, whereas python is 0-indexed
    return x, y


def check(filecols, path_to_graph, graphname):
    filegraph=f'{path_to_graph}/{graphname}'
    with open(filegraph, 'r') as f:
        content = f.read().strip()
    lines = content.split('\n')  # skip comment line(s)
    n=int(lines[0])
    nedges=int(lines[1])
    edgesnx=[parse_line(line) for line in lines[2:nedges+2]]
    g = nx.from_edgelist(edgesnx)
    q = int(lines[nedges+2])
    fcol = open(filecols, "r")
    cols = []
    while True:
        j = fcol.readline()
        if not j:
            print(f'graph {graphname} not found inside {filecols}')
            break
        line_col = j.split("\t")
        gname_try = line_col[1]
        if gname_try == graphname:
            print(f'graph {graphname} found')
            cols = line_col[2][1:-2].split(",")
            for i in range(len(cols)):
                cols[i] = int(cols[i])
            break
    cond = True
    for i in range(len(cols)):
        if cols[i] > q:
            print(f'coloring with q={q} not achieved for graph {graphname}')
            print(f'color={cols[i]} repeated')
            break
        else:
            for j in nx.neighbors(g, i + 1):
                if cols[j - 1] == cols[i]:
                    cond = False
                    break
            if not cond:
                print(f'the neighboring nodes ({i}, {j})  have the same color={cols[i]}')
                break
    fcol.close()
    if cond:
        print(f'The graph "{graphname}" is well colored with q={q}')


fcols="./Decoupled/colorings/trials/coloring.txt"
path="./Decoupled/data/test"
gname="queen5_5_mod.col"

check(fcols, path, gname)

