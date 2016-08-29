#! /bin/bash
#$ -N ctf
#$ -cwd
#$ -S /bin/bash
#$ -p -10
#$ -e ctf.err
#$ -o ctf.out
#$ -j y
#$ -pe mpi_openmp_16 16

export PATH="/scicore/pfs/c-cina/ctf4:/scicore/pfs/c-cina/anaconda2/bin:$PATH"
export PYTHONPATH="$PYTHONPATH"

python zorroCTFAnalysis.py

