julia -e 'include("../matrix_io.jl"); S = sprandsymposdef(1000,0.001); write_matrix_txt("input_matrix", S);'

../../mkl-matrix-inv 1000 input_matrix.txt output_matrix.txt > output.log

julia -E 'include("../matrix_io.jl"); A = read_matrix_txt("input_matrix"); Ainv = read_matrix_txt("output_matrix"); maximum(abs, inv(Matrix(A)) .- Ainv)'