using Moose, Test

grid  = Γgrid()
basis = Basis()

@testset "Moose.jl" begin
    @testset "Tests metrics.jl" begin
        z = Float32.([1.0,1.0,1.0,1.0, NaN])
        ẑ = Float32.([0.0,1.0,1.0,1.0, 1.0])
        @test GF(z, ẑ)  == 3/4
        @test Moose.MAE(z, ẑ) == 1/4
        @test Moose.MAD(z, ẑ) ==  0.0f0
    end

    @testset "Tests strutture.jl" begin
    
        @test length(grid.ζ) == length(grid.ζmin : grid.δζ : grid.ζmax)
        @test basis.n == length(grid.ζ) 
    end

    @testset "Tests leggere.jl" begin
        path = joinpath(@__DIR__, "../data/spectra_sample/")
        data = leggere_fits(path)
        @test length(data.flux) == 24

        h5path  = joinpath(@__DIR__, "../output/chi2_files/chi2_testMoose.h5")
        keys, χ2 = leggere_chifile(h5path)
        @test abs(grid.ζ[argmin(χ2[argmax(keys .== "2")])] - 0.41f0) < 1f-2

        cube_data = leggere_cube(joinpath(@__DIR__, "../data/cubes/DATACUBE_test.fits"))
        @test size(cube_data.flux) == (4,4,3721)
        @test size(cube_data.sdev) == (4,4,3721)
        @test any(.!iszero.(cube_data.flux[1,1,:]))
        @test !any(isnan.(cube_data.sdev))
        @test !any(isinf.(cube_data.sdev))        
        @test !any(isnan.(cube_data.flux))
    end

    @testset "Tests methods.jl" begin
        
        path  = joinpath(@__DIR__, "../data/spectra_sample/")
        data  = leggere_fits(path)
        
        fλ1, σλ1, λ1, id1   = data.flux[1], data.sdev[1], data.awave[1], data.ids[1] 
        intrpfλ1 , intrpσλ1 = interpolate(basis, fλ1 .* λ1, σλ1 .* λ1, λ1)
        χ21                 = threaded_fnnls(basis, intrpfλ1 , intrpσλ1)
        
        h5path  = joinpath(@__DIR__, "../output/chi2_files/chi2_testMoose.h5")

        if isfile(h5path)
            rm(h5path)
        end
        χloop(basis, data; output_path = h5path)
        
        keys, χ2 = leggere_chifile(h5path)

        @test abs(grid.ζ[argmin(χ21)] - 0.41f0) < 1f-2
        @test length(keys) == 24
        @test abs(grid.ζ[argmin(χ2[argmax(keys .== "2")])] - 0.41f0) < 1f-2

        χcube(joinpath(@__DIR__, "../data/cubes/DATACUBE_test.fits"), nothing)
        @test isfile(joinpath(@__DIR__, "../output/DATACUBE_test_chi2.h5"))
    end

    @testset "Test fnnls.jl" begin
        # Solve A*x = b for x, subject to x >=0
        A = [ 0.53879488f0  0.65816267f0
              0.12873446f0  0.98669198f0
              0.24555042f0  0.00598804f0
              0.80491791f0  0.32793762f0 ]

        b = [0.888f0,  0.562f0,  0.255f0,  0.077f0]

        # Test that nnls produces the same solution as scipy
        xknown = [0.15512102f0, 0.69328985f0] # approx solution from scipy

        AtA = A'*A
        Atb = A'*b
        @test sqrt((sum(fnnls(AtA,Atb)- xknown) .^2)/2) < 1f-5
    end

    @testset "Test nmf.jl" begin
        X = [ 0.53879488f0  0.65816267f0
              0.12873446f0  0.98669198f0
              0.24555042f0  0.00598804f0
              0.80491791f0  0.32793762f0 ]
              
        nmf = nNMF(X)
        error, iter = nearly!(nmf)
        @test (abs(error -  5.6912923) < 1f-3) && (iter ==1)
    end

end
    
