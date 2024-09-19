#!/bin/bash
export LC_NUMERIC="en_US.UTF-8"

seed0=1
ngr_each=200

for exp in 8 7 6 5 4
do

N=$((2 ** exp))

for c in $(seq 2.96 0.18 5.0)
do

path='/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/random_graphs/ErdosRenyi/N_'$N

python gen_graphs_cluster.py $seed0 $ngr_each $c $N $path

echo "N="$N"   c="$c"   done"

done
done

exit 0

