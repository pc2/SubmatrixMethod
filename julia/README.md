# Julia "Frontend"

The file `matrix_io.jl` contains functions to read / write sparse matrices (`SparseMatrixCSC`) from / to disk in a format that is compatible with the applications `mkl-matrix-inv` and `mpi-matrix-inv` in the parent directory. In also defines `sprandsymposdef` (a rough analogon of MATLABs same-named function)

## Environment / Modules

In the root of the repo lies a file `.envrc`. If you have [direnv](https://direnv.net/) this is automatically picked up and the relevant modules are loaded. If not, simply execute `source .envrc` before you're doing anything else.

## Building

Go to the root of the repo and execute `make`. That's it.

## Usage: `mkl-matrix-inv`

Performs a standard matrix inversion using (multithreaded) Intel MKL. No MPI parallelism. Expects input matrix to be in `.txt` form (use `write_matrix_txt` to create it).

**Full example:**
(Assuming that we're currently in the directory `matrix-io-julia`)

In Julia:
```julia
include("matrix_io.jl");
S = sprandsymposdef(1000,0.001);
write_matrix_txt("input_matrix", S); # produces file 'input_matrix.txt'
```

In Bash:
```
../mkl-matrix-inv 1000 input_matrix.txt output_matrix.txt > output.log
```

The file `output.log` contains the printed output (e.g. timings). The file output_matrix.txt is a sparse matrix which can be read by `read_matrix_txt`

In Julia:
```julia
include("matrix_io.jl");
A = read_matrix_txt("input_matrix");
Ainv = read_matrix_txt("output_matrix");
maximum(abs, inv(Matrix(A)) .- Ainv) # gave ~1e-10 when writing this but will vary
```

## Usage: `mpi-matrix-inv`
Performs a matrix inversion using the submatrix method. Uses MPI for distributed computing and, on each rank, a certain number of Intel MKL threads. Expects input matrix to be in `.cp`, `.ri`, `.val` form (use `write_matrix_three_files` to create it). Moreover, these files must have a prefix like `sprandsym-s1000-d1-c2-n1.cp` where
* `s1000` indicates the size of the sparse matrix (`s1000` means `1000x1000`)
* `d1` indicates the density in units of the number of elements of the matrix (`s1000` and `d1` together mean `1/(1000 * 1000) = 1e-6`)
* `c2` indicates the condition number
* `n1` is an important "repetition index" (just always use `n1`)
Note that it expects the indices (row indices in `.ri` and col pointers in `.cp` to be `Int32`)!

**Full example:**
(Assuming that we're currently in the directory `matrix-io-julia`)

In Julia:
```julia
include("matrix_io.jl");
S = sprandsymposdef(1000, 1e-6, 2);
# produce three files *.cp, *.ri, and *.val. Value type is Float64. Index type is Int32
write_matrix_three_files("sprandsym-s1000-d1-c2-n1", S, Float64, Int32); 
```

In Bash:
```
mpiexec -n 5 ../mpi-matrix-inv 1000 1 2 > output.log
```

The file `output.log` contains the printed output (e.g. timings). The file `sprandsym-s1000-d1-c2-n1.out.val` contains the values of the inverse matrix (the indices are the same as before).

In Julia:
```julia
include("matrix_io.jl");
A = read_matrix_three_files("sprandsym-s1000-d1-c2-n1", Float64, Int32);
Ainv = read_matrix_three_files("sprandsym-s1000-d1-c2-n1", Float64, Int32; outval=true); # outval=true reads `.out.val` instead of `.val` file
maximum(abs, Ainv .- inv(Matrix(A))) # gave 0.0 when writing this
```