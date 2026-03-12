using Moose, Test

@testset "Moose.jl" begin
    @testset "Test metrics.jl" begin
        z = Float32.([1.0,1.0,1.0,1.0, NaN])
        ẑ = Float32.([0.0,1.0,1.0,1.0, 1.0])
        @test GF(z, ẑ)  == 3/4
        @test Moose.MAE(z, ẑ) == 1/4
        @test Moose.MAD(z, ẑ) ==  0.0f0
    end

    @testset "Test strutture.jl" begin
        wgrid = Γgrid(λmin = 4700f0, δζ = 0.1f0)
        basis = Basis(wgrid)

        @test length(wgrid.ζ) == length(wgrid.ζmin : wgrid.δζ : wgrid.ζmax)
        @test basis.n == length(wgrid.ζ) 
    end

    @testset "Tests methods.jl" begin
        wgrid = Γgrid(λmin = 4700f0, δζ = 0.001f0)
        basis = Basis(wgrid)

        path  = joinpath(@__DIR__, "../data/spectra_sample/")
        data  = leggere(path, Val(:fits))
        
        #test interpolate()
        f, v, λ, id   = data.flux[1], data.var[1], data.awave[1], data.ids[1] 
        fʳ, vʳ = interpolate(basis, f, v, λ)

        #test interpolate!()
        fʳ, vʳ = zeros(Float32, basis.l), zeros(Float32, basis.l)
        interpolate!(basis, f, v, λ, fʳ, vʳ)

        #test flow() & flow!() functions
        χ²    = flow(basis, fʳ, vʳ, Val(:no_coeffs))
        χ², Ω = flow(basis, fʳ, vʳ, Val(:coeffs))
        display(plot(χ²))

        χ² = zeros(Float32, basis.n)
        Ω  = zeros(Float32, basis.k, basis.n)

        flow!(basis, fʳ, vʳ, χ²)
        flow!(basis, fʳ, vʳ, χ², Ω)

        @test abs(wgrid.ζ[argmin(χ²)] - 0.41f0) < 1f-2

        h5path  = joinpath(@__DIR__, "../output/chi2_files/chi2_testMoose.h5")
        if isfile(h5path)
            rm(h5path)
        end

        flow(wgrid, basis, data; output_path = h5path)
        χdata = leggere(h5path, Val(:chi2file))

        @test length(χdata.id) == 24
        @test abs(wgrid.ζ[argmin(χdata.chi2_1[argmax(χdata.id .== "2")])] - 0.41f0) < 1f-2

        #χcube(joinpath(@__DIR__, "../data/cubes/DATACUBE_test.fits"), nothing)
        #@test isfile(joinpath(@__DIR__, "../output/chi2_files/DATACUBE_test_chi2.h5")) 
    end

    @testset "Test leggere.jl" begin
        wgrid = Γgrid(λmin = 4700f0, δζ = 0.1f0)
        basis = Basis(wgrid)

        path = joinpath(@__DIR__, "../data/spectra_sample/")
        data = leggere(path, Val(:fits))
        @test length(data.flux) == 24

        h5path  = joinpath(@__DIR__, "../output/chi2_files/chi2_testMoose.h5")
        χdata   = leggere(h5path, Val(:chi2file))
        @test length(χdata.id) == 24

        cube_data = leggere(joinpath(@__DIR__, "../data/cubes/DATACUBE_test.fits"), Val(:cube))
        @test size(cube_data.flux) == (4,4,3721)
        @test size(cube_data.sdev) == (4,4,3721)
        @test any(.!iszero.(cube_data.flux[1,1,:]))
        @test !any(isnan.(cube_data.sdev))
        @test !any(isinf.(cube_data.sdev))        
        @test !any(isnan.(cube_data.flux))
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
        @test isfinite(error)
    end

end

path  = joinpath(@__DIR__, "../data/spectra_sample/")
data  = leggere(path, Val(:fits))

wgrid = Γgrid(λmin = 4700f0, δζ = 5f-4)
basis = Basis(wgrid, Val(:observed))
wgrid = Γgrid(λmin = 4750f0, δζ = 5f-4)
basis.n
basis.l
update!(basis, wgrid, Val(:observed))
basis.l

j = 5
f, σ, λ, id   = data.flux[j], data.sdev[j], data.awave[j], data.ids[j] 

@time χ²    = flow(basis, f, σ .^2)
