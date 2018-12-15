#!/usr/bin/env bash

#SBATCH -J ntsa
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --workdir=/homedtic/stotaro/ntsa/ntsa/
#SBATCH -C intel #request intel node (those have infiniband)
#SBATCH -o /homedtic/stotaro/ntsa/ntsa/%N.%J.out # STDOUT
#SBATCH -e /homedtic/stotaro/ntsa/ntsa/%N.%j.err # STDERR

set -x

declare -a kaf=(0 1)
mkdir -p jobs
for i in "${kaf[@]}"; do
    for j in "${kaf[@]}"; do
        echo "Running DARNN with $i and $j"
#        sbatch run.sh "$i" "$j"
    done
done

