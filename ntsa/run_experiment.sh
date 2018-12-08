#!/usr/bin/env bash

#SBATCH -J ntsa
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -C intel #request intel node (those have infiniband)
#SBATCH -o /homedtic/stotaro/ntsa/jobs/%N.%J.out # STDOUT
#SBATCH -e /homedtic/stotaro/ntsa/jobs/%N.%j.err # STDERR

set -x
module load Tensorflow/1.5.0-foss-2017a-Python-3.6.4
module load Tkinter/3.6.4-foss-2017a-Python-3.6.4
source /homedtic/stotaro/tf_cpu/bin/activate
mkdir -p jobs

python main.py

