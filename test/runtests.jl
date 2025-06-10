using Moose
using Test

@testset "Moose.jl" begin
    @testset "Tests Metrics.jl" begin
        z = Float32.([1.0,1.0,1.0,1.0, NaN])
        ẑ = Float32.([0.0,1.0,1.0,1.0, 1.0])
        @test Moose.GF(z, ẑ) == 3/4
        @test Moose.MAE(z, ẑ) == 1/4
        @test Moose.MAD(z, ẑ) ==  0.0f0
    end
end
    