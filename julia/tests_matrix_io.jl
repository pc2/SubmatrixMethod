include("matrix_io.jl")

using Test

@testset "reading Michaels files" begin
    folder = joinpath(dirname(@__FILE__), "example_matrix")
    B = read_matrix_three_files("$folder/sprandsym-s1000-d1-c2-n1", Float64, Int32)
    A = read_matrix_txt("$folder/sprandsym-s1000-d1-c2-n1.txt")
    A == B
end

@testset "writing + reading three file format" begin
    S = sprand(1000, 1000, 0.01)
    write_matrix_three_files("testmat", S)
    @test read_matrix_three_files("testmat") == S
    rm("testmat.cp")
    rm("testmat.ri")
    rm("testmat.val")
end

@testset "writing + reading txt format" begin
    S = sprand(1000, 1000, 0.01)
    write_matrix_txt("testmat", S)
    @test read_matrix_txt("testmat") == S
    rm("testmat.txt")
end
