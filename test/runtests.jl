using Moose
using Test

@testset "Moose.jl" begin
    @testset "Tests Metrics.jl" begin
        z = Float32.([1.0,1.0,1.0,1.0, NaN])
        ẑ = Float32.([0.0,1.0,1.0,1.0, 1.0])
        @test Moose.GF(z, ẑ)  == 3/4
        @test Moose.MAE(z, ẑ) == 1/4
        @test Moose.MAD(z, ẑ) ==  0.0f0
    end

    #@testset "Tests Strutture.jl" begin
    #    grid  = Moose.Γgrid()
    #    basis = Basis()

    #    @test length(grid.ζ) == length(grid.ζmin : grid.δζ : grid.ζmax)
    #    @test basis.n == length(grid.ζ) 
        #@test Moose.MAD(z, ẑ) ==  0.0f0
    #end

    @testset "Tests Leggere.jl" begin
        path = joinpath(@__DIR__, "../data/spectra_sample/")
        data = leggere_fits(path)
        @test length(data.flux) == 24
    end

end
    