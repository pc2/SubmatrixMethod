#!/bin/sh
#SBATCH -A pc2-mitarbeiter
#SBATCH -p normal
#SBATCH -t 00:10
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -n 5

julia -e 'include("../matrix_io.jl"); S = sprandsymposdef(1000, 1e-6, 2); write_matrix_three_files("sprandsym-s1000-d1-c2-n1", S, Float64, Int32);'

srun -n 5 ../../mpi-matrix-inv 1000 1 2 > output.log
