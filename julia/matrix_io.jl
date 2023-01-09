using Mmap
using SparseArrays
using DelimitedFiles
using LinearAlgebra

# reading

"Read in a sparse matrix from three separate files with extensions, `.cp`, `.ri`, and `.val`"
function read_matrix_three_files(
    fname_prefix::AbstractString,
    Tv::DataType=Float64,
    Ti::DataType=Int64;
    mmap=false,
    outval=false,
)
    fcolptr = fname_prefix * ".cp"
    frowval = fname_prefix * ".ri"
    if outval
        fnzval = fname_prefix * ".out.val"
    else
        fnzval = fname_prefix * ".val"
    end
    _read_vector(fname, T) = read!(fname, Vector{T}(undef, filesize(fname) ÷ sizeof(T)))
    if !mmap
        colptr = _read_vector(fcolptr, Ti)
        rowval = _read_vector(frowval, Ti)
        nzval = _read_vector(fnzval, Tv)
    else
        colptr = Mmap.mmap(fcolptr, Vector{Ti})
        rowval = Mmap.mmap(frowval, Vector{Ti})
        nzval = Mmap.mmap(fnzval, Vector{Tv})
    end
    m = n = length(colptr) - 1
    # need to increment indices by 1 here (C -> Julia)
    return SparseMatrixCSC(m, n, colptr .+ 1, rowval .+ 1, nzval)
end

"Read in a sparse matrix from a single txt file (produced by Matlab)."
function read_matrix_txt(fname::AbstractString, T::DataType=Float64)
    if !endswith(fname, ".txt")
        fname *= ".txt"
    end
    M = readdlm(fname, ',', T)
    return sparse(M)
end

# writing

"Write the given sparse matrix into three files (`prefix.cp`, `prefix.ri`, `prefix.val`)."
function write_matrix_three_files(prefix::AbstractString, S::SparseMatrixCSC, Tv=Float64, Ti=Int64)
    write(string(prefix, ".cp"), Ti.(S.colptr .- 1))
    write(string(prefix, ".ri"), Ti.(S.rowval .- 1))
    write(string(prefix, ".val"), Tv.(S.nzval))
    return nothing
end

"Write the given sparse matrix into a single txt file (can be read by Matlab)."
function write_matrix_txt(fname::AbstractString, S::SparseMatrixCSC)
    if !endswith(fname, ".txt")
        fname *= ".txt"
    end
    writedlm(fname, Matrix(S), ',')
    return nothing
end

# Bonus:

"""
    sprandsymposdef(n,p)
Generates a sparse matrix of size `(n,n)` with the following properties:
  - `issymmetric`
  - `isposdef`
  - has approximately `p * n^2` non-zero elements
  - has a condition number of approximately `c ≈ 1`.

*Algorithm:*

Creates a random sparse matrix (`sprandn`), symmetrizes it and makes it
diagonally dominant by adding `n*I` to ensure positive definiteness.

*References:*

The implementation of the function is inspired by [this stackexchange thread](https://math.stackexchange.com/a/358092).
"""
function sprandsymposdef(n, p)
    A = sprandn(n, n, p)
    A = sparse(Symmetric(A))

    # https://math.stackexchange.com/a/358092:
    # Since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
    # is symmetric positive definite, which can be ensured by adding nI
    A = A + n * I(n)
    return A
end


"""
    sprandsymposdef(n,p,c)
Generates a sparse matrix of size `(n,n)` with the following properties:
  - `issymmetric`
  - `isposdef`
  - has approximately `p * n^2` non-zero elements
  - has a condition number of exactly `c`

**WARNING**: While it works okish for some inputs it can take ages to converge (if at all) for others!

*Algorithm:*

Creates a diagonal positive-definite matrix with the desired condition number
and applies random Jacobi rotations to it to create non-zero off-diagonal elements.

*References:*

The function is inspired by MATLABs [`sprandsym(n,density,rc,kind=1)`](https://de.mathworks.com/help/matlab/ref/sprandsym.html).
"""
function sprandsymposdef(n, p, c)
    # check inputs
    0 ≤ p ≤ 1 || throw(ArgumentError("density must be ≥ 0 and ≤ 1"))
    c ≥ 1 || throw(ArgumentError("condition number must be ≥ 1"))
    n ≥ 1 || throw(ArgumentError("n must be ≥ 1"))

    # enforce condition number
    A = spdiagm(n, n, range(1, c; length=n))

    # apply random Jacobi rotations
    #     - creates non-zero off diagonal elements
    #     - preserves eigvals, thus also posdef property and cond
    # TODO: speed up / avoid allocs by updating the elements of A
    #       directly (i.e. don't construct Jacobi matrix explictly.)
    while density(A) ≤ p
        a, b = (rand(1:n), rand(1:n))
        while b == a
            b = rand(1:n)
        end
        J = _jacobi_rot_mat(n, a, b)
        A = J' * A * J
    end

    # enforce symmetry, i.e. iron out numerical inaccuracies
    A .+= A'
    A .*= 0.5
    return A
end

function _jacobi_rot_mat(n, p, q; c=rand(), s=sqrt(1 - c^2))
    J = sparse(I(n) * 1.0)
    J[p, p] = c
    J[q, q] = c
    J[q, p] = s
    J[p, q] = -s
    return J
end

"""
Number of zero-valued elements over the total number of elements.
"""
sparsity(A::AbstractArray{<:Number}) = count(iszero, A) / length(A)

"""
Number of non-zero elements over the total number of elements.
"""
density(A::AbstractArray{<:Number}) = 1 - sparsity(A)
