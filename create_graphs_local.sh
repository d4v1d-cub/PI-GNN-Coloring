#!/bin/bash
export LC_NUMERIC="en_US.UTF-8"

seed0=1
ngr_each=400

for exp in 12 11
do

N=$((2 ** exp))

# for c in $(seq 2.96 0.18 5.0)
# do

for c in $(seq 9.9 0.4 13.5)
do

path='/media/david/Data/UH/Grupo_de_investigacion/Hard_benchmarks/Coloring/PI-GNN/random_graphs/ErdosRenyi/New_graphs/N_'$N

python gen_graphs_coloring.py $seed0 $ngr_each $c $N $path

echo "N="$N"   c="$c"   done"

done
done

exit 0

