using Base.Threads, Dates, LinearAlgebra
using ProgressMeter: @showprogress

"""
    interpolate( basis::Basis, f::Vector{Float32}, σ::Vector{Float32}, λ::Vector{Float32}; fₓ = 0.0f0, σₓ = 1f6)

Description
============
This function takes an instance of the Basis Struct and three vectors: fluxes (f), standard deviations (σ) and observed wavelengths (λ), it interpolates 
the flux and std vectors to the rest wavelength grid assuming a redshift of 0.
    
Arguments
==========
- **`basis ::Basis`**             : Basis Struct
- **`f     ::Vector{Float32}`**   : Observed flux densities
- **`σ     ::Vector{Float32}`**   : Flux associated standard deviations
- **`λ     ::Vector{Float32}`**   : Observed wavelengths
- **`fₓ    ::Float32 = 0f0`**   : Extrapolation value for fluxes
- **`σₓ    ::Float32 = 1f6`**   : Extrapolation value for standard deviations
None

returns
=======
- **`fʳ    ::Vector{Float32}`**   : Interpolated flux densities
- **`σʳ    ::Vector{Float32}`**   : Interpolated standard deviations

See also
========
`interpolate!()`, works similarly, the difference is that `interpolate!` accepts two additional
arguments, `fʳ` and `σʳ`, which are the vectors of interpolated flux densities and standard deviations. Sizes of these
vectors are known, so one can pass an initialization of them (e.g. with zeros) and `interpolate!` will fill them.
This is helpful because it avoids reallocations.

Benchmark
=========
```julia
using BenchmarkTools
@btime interpolate!(basis, f, σ, λ, fʳ , σʳ)
233.140 μs (173 allocations: 32.38 KiB)
```
"""
function interpolate(basis::Basis, f::Vector{Float32}, σ::Vector{Float32}, λ::Vector{Float32};
                      fₓ::Float32 = 0.0f0, σₓ::Float32 = 1f6)
    
    Λ₁  = @view basis.Λ[1,:]
    l  = basis.l

    fʳ = zeros(Float32, l)
    σʳ = zeros(Float32, l)

    logλ = log10.(λ)
    lwbn = logλ[1]
    upbn = logλ[end]

    @inbounds @threads for j in 1:l 
        
        logλⱼ = Λ₁[j]
        # Left or right extrapolation
        if logλⱼ < lwbn || logλⱼ > upbn
            fʳ[j] = fₓ
            σʳ[j] = σₓ
            continue
        end
        # Binary search for interpolation index
        idx = searchsortedfirst(logλ, logλⱼ)
        idx = max(1, min(idx - 1, l-1))  # Clamp to valid range

        # Direct memory access
        logλ₁ = logλ[idx]
        logλ₂ = logλ[idx + 1]
            
        inv_denom = 1 / (logλ₂ - logλ₁)
        p = (logλⱼ - logλ₁) * inv_denom
        
        λᵢ     = λ[idx]
        #f₁, f₂ = f[idx]*λᵢ, f[idx + 1]*λᵢ
        #σ₁, σ₂ = σ[idx]*λᵢ, σ[idx + 1]*λᵢ
        
        # Fused multiply-add operations
        fʳ[j] = muladd(p, f[idx + 1]*λᵢ - f[idx]*λᵢ, f[idx]*λᵢ)
        σʳ[j] = muladd(p, σ[idx + 1]*λᵢ - σ[idx]*λᵢ, σ[idx]*λᵢ)
    end
    
    return fʳ, σʳ
end
function interpolate!(basis::Basis, f::Vector{Float32}, σ::Vector{Float32}, λ::Vector{Float32},
                       fʳ::Vector{Float32}, σʳ::Vector{Float32};
                       fₓ::Float32 = 0.0f0, σₓ::Float32 = 1f6)
    
    Λ₁  = @view basis.Λ[1,:]
    l = basis.l

    logλ = log10.(λ)
    lwbn = logλ[1]
    upbn = logλ[end]

    @inbounds @threads for j in 1:l 
        
        logλⱼ = Λ₁[j]
            
        # Left or right extrapolation
        if logλⱼ < lwbn || logλⱼ > upbn
            fʳ[j] = fₓ
            σʳ[j] = σₓ
            continue
        end

        # Binary search for interpolation index
        idx = searchsortedfirst(logλ, logλⱼ)
        idx = max(1, min(idx - 1, l-1))  # Clamp to valid range

        # Direct memory access
        logλ₁ = logλ[idx]
        logλ₂ = logλ[idx + 1]
            
        inv_denom = 1 / (logλ₂ - logλ₁)
        p = (logλⱼ - logλ₁) * inv_denom
            
        λᵢ     = λ[idx]
        #f₁, f₂ = f[idx]*λᵢ, f[idx + 1]*λᵢ
        #σ₁, σ₂ = σ[idx]*λᵢ, σ[idx + 1]*λᵢ
        
        # Fused multiply-add operations
        fʳ[j] = muladd(p, f[idx + 1]*λᵢ - f[idx]*λᵢ, f[idx]*λᵢ)
        σʳ[j] = muladd(p, σ[idx + 1]*λᵢ - σ[idx]*λᵢ, σ[idx]*λᵢ)
    end
end


"""
    flow(basis::Basis, f::Vector{Float32}, σ::Vector{Float32})

Description
===========
This function executes a multi-threaded calculation of the chi-square curve. For each redshift (each gram matrices Hᵢ & HHᵀᵢ
in basis Struct), it reconstructs an input vector f with basis vectors using a Fast Non-Negative Least Squares (FNNLS),
then quantifies the reconstruction error using a χ² metric

#put definition of chi2 here        grid  = Γgrid()

Arguments
=========
- **`basis ::Basis`**           : Basis Struct
- **`f    ::Vector{Float32}`**  : (interpolated) flux densities vector
- **`σ    ::Vector{Float32}`**  : (interpolated) standard deviations vector

Returns
=======
- **`χ²    ::Vector{Float32}`**  : χ² curve

See also
========
`flow!()`

Example
=======
```julia
# 1. Initialize the Basis Struct
wgrid = Γgrid(λmin = 4700f0, δζ = 0.0005f0)
basis = Basis(wgrid)
# 2. Interpolate flux densities fλ and σλ to rest frame grid
fʳ , σʳ = interpolate(basis, f, σ, λ)
# 3. Run flow to get the chi2 curve
χ² = threaded_fnnls(basis, fʳ , σʳ)
#4. to get also the coeffs
χ², Ω = threaded_fnnls(basis, fʳ , σʳ, true)

```
Tips
====
To set the number of threads in julia, add this line to your linux/macos bash
```bash
export JULIA_NUM_THREADS=4
```
"""
function flow(basis::Basis, f::Vector{T}, σ::Vector{T}) where{T}
        
    n,l,k = basis.n, basis.l,basis.k
    nthreads   = Threads.nthreads()
    chunk_size = cld(n, nthreads)

    # Preallocate workspace for each thread
    f_threads  = [Vector{T}(undef,l) for _ in 1:nthreads]
    ω_threads  = [Vector{T}(undef,k) for _ in 1:nthreads]

    # Preallocate χ² vector
    χ² = Vector{T}(undef, n)

    @threads for t in 1:nthreads
        fʰ  = f_threads[t]
        ωʰ  = ω_threads[t]

        i_start = (t-1)*chunk_size + 1
        i_end   = min(t*chunk_size, basis.n)

        @inbounds for i in i_start:i_end
            mul!(ωʰ, basis.Hᵢ[i], f)
            ω = fnnls(basis.HHᵀᵢ[i], ωʰ)
            mul!(fʰ, Transpose(basis.Hᵢ[i]), ω)

            Σ = 0.0f0
            @simd for j in 1:l
                Σ += ((fʰ[j] - f[j]) / σ[j])^2
            end
            χ²[i] = Σ / l
        end
    end
    return χ²
end
function flow(basis::Basis, f::Vector{T}, σ::Vector{T}, coeffs::Bool) where{T}
    n,l,k = basis.n, basis.l,basis.k
    nthreads   = Threads.nthreads()
    chunk_size = cld(n, nthreads)

    # Preallocate workspace for each thread
    f_threads  = [Vector{T}(undef,l) for _ in 1:nthreads]
    ω_threads  = [Vector{T}(undef,k) for _ in 1:nthreads]
    
    # Preallocate χ² & Ω vectors
    χ² = Vector{T}(undef, n)
    Ω = Matrix{T}(undef,k,n)

    @threads for t in 1:nthreads
        fʰ  = f_threads[t]
        ωʰ = ω_threads[t]

        i_start = (t-1)*chunk_size + 1
        i_end   = min(t*chunk_size, basis.n)

        @inbounds for i in i_start:i_end
            mul!(ωʰ, basis.Hᵢ[i], f)
            ω = fnnls(basis.HHᵀᵢ[i], ωʰ)
            mul!(fʰ, Transpose(basis.Hᵢ[i]), ω)

            Σ = 0.0f0
            @simd for j in 1:l
                Σ += ((fʰ[j] - f[j]) / σ[j])^2
            end
            χ²[i] = Σ / l
            Ω[:,i] .= ω
        end
    end
    return χ², Ω
end
function flow!(basis::Basis, f::Vector{T}, σ::Vector{T}, χ²::Vector{T}) where{T}
    n,l,k = basis.n, basis.l,basis.k
    nthreads   = Threads.nthreads()
    chunk_size = cld(n, nthreads)

    # Preallocate workspace for each thread
    f_threads  = [Vector{T}(undef,l) for _ in 1:nthreads]
    ω_threads  = [Vector{T}(undef,k) for _ in 1:nthreads]

    @threads for t in 1:nthreads
        fʰ  = f_threads[t]
        ωʰ = ω_threads[t]

        i_start = (t-1)*chunk_size + 1
        i_end   = min(t*chunk_size, basis.n)

        @inbounds for i in i_start:i_end
            mul!(ωʰ, basis.Hᵢ[i], f)
            ω = fnnls(basis.HHᵀᵢ[i], ωʰ)
            mul!(fʰ, Transpose(basis.Hᵢ[i]), ω)

            Σ = 0.0f0
            @simd for j in 1:l
                Σ += ((fʰ[j] - f[j]) / σ[j])^2
            end
            χ²[i] = Σ / l
        end
    end
end
function flow!(basis::Basis, f::Vector{T}, σ::Vector{T}, χ²::Vector{T}, Ω::Matrix{T}) where{T}
    n,l,k = basis.n, basis.l,basis.k
    nthreads   = Threads.nthreads()
    chunk_size = cld(n, nthreads)

    # Preallocate workspace for each thread
    f_threads  = [Vector{T}(undef,l) for _ in 1:nthreads]
    ω_threads  = [Vector{T}(undef,k) for _ in 1:nthreads]

    @threads for t in 1:nthreads
        fʰ  = f_threads[t]
        ωʰ = ω_threads[t]

        i_start = (t-1)*chunk_size + 1
        i_end   = min(t*chunk_size, basis.n)

        @inbounds for i in i_start:i_end
            mul!(ωʰ, basis.Hᵢ[i], f)
            ω = fnnls(basis.HHᵀᵢ[i], ωʰ)
            mul!(fʰ, Transpose(basis.Hᵢ[i]), ω)

            Σ = 0.0f0
            @simd for j in 1:l
                Σ += ((fʰ[j] - f[j]) / σ[j])^2
            end
            χ²[i] = Σ / l
            Ω[:,i] .= ω
        end
    end
end
function flow!(basis::Basis, f::Vector{T}, σ::Vector{T}, fᵃ::Vector{T}, χ²::Vector{T}, Ω::Matrix{T}) where{T}
    n,l,k = basis.n, basis.l,basis.k
    nthreads   = Threads.nthreads()
    chunk_size = cld(n, nthreads)

    # Preallocate workspace for each thread
    f_threads  = [Vector{T}(undef,l) for _ in 1:nthreads]
    ω_threads  = [Vector{T}(undef,k) for _ in 1:nthreads]

    @threads for t in 1:nthreads
        fʰ  = f_threads[t]
        ωʰ = ω_threads[t]

        i_start = (t-1)*chunk_size + 1
        i_end   = min(t*chunk_size, basis.n)

        @inbounds for i in i_start:i_end
            H₁ = vcat(transpose(fᵃ), basis.Hᵢ[i])
            mul!(ωʰ, H₁, f)
            ω = fnnls(H₁ * transpose(H₁), ωʰ)
            mul!(fʰ, transpose(H₁), ω)

            Σ = 0.0f0
            @simd for j in 1:l
                Σ += ((fʰ[j] - f[j]) / σ[j])^2
            end
            χ²[i] = Σ / l
            Ω[:,i] .= ω
        end
    end
end

"""
    flow(wgrid::Γgrid, basis:Basis, data; output_path::String = nothing)

Description
===========
Runs a threaded FNNLS on a set of spectra loaded into a NamedTuple apriori (`data`). Resulting χ² curves are
saved into an `.h5` file, whose path and name are specified by setting the keyword argument `output_path`. 

Arguments
=========
- **`wgrid       ::Γgrid`**     : Γgrid Struct
- **`basis       ::Basis`**     : Basis Struct
- **`data        ::Struct`**    : Named Tuple with four fields: flux, sdev, wave, ids
- **`output_path ::String`**    : Path where the chi2 file will be saved 

Returns
=======
nothing

Methods
=======
- `flow(data; output_path::String = nothing)`: `wgrid` and `basis` are instantiated within the method.
- `flow(fits_path::String, DataExtName::Union{String, Int}, StatExtName::Union{String, Int}; output_path::String = nothing)`: `wgrid` and `basis` are instantiated within the method, the data is loaded within the function using the provided `fits_path` and the extensions: `DataExtName`, `StatExtName`.
- `flow(wgrid::Γgrid, basis::Basis, fits_path::String, DataExtName::Union{String, Int}, StatExtName::Union{String, Int}; output_path::String = nothing)`: the data  is loaded within the function using the provided `fits_path` and the extensions: `DataExtName`, `StatExtName`.

Example
=======
```julia
# 1. Initialize Basis Struct
wgrid = Γgrid(;λmin, δζ)
basis = Basis(wgrid)
# 2. read data from fits files
data = leggere("../data/spectra_sample", Val(:fits))
#3. Specify output file path and run flow
chi2file = "../output/chi2_files/chi2_moose.h5"
flow(wgrid, basis, data; output_path = chi2file)
```
"""
function flow(wgrid::Γgrid, basis::Basis, data; output_path::Union{String,Nothing}= nothing)

    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path, a file with the same name already exists!"
    end
    k,l,n = basis.k, basis.l, basis.n
    fʳ = Vector{Float32}(undef, l)
    σʳ = Vector{Float32}(undef, l)
    χ² = Vector{Float32}(undef, n)
    Ω  = Matrix{Float32}(undef, k, n)

    N = length(data.flux)

    @showprogress for i in 1:N
        
        f      = data.flux[i]
        σ = data.sdev[i]         
        λ = data.awave[i]

        interpolate!(basis, f, σ, λ, fʳ, σʳ)
        flow!(basis, fʳ, σʳ, χ², Ω)
        
        # predicted redshift ẑ  
        ẑ = wgrid.ζ[argmin(χ²)]
        
        # decomposition coeffs @ ẑ
        ω = Ω[:, argmin(χ²)]
        
        # significance score \Delta\chi2
        Δ = Δχ²(χ²)
        
        # Robustness metric R
        R_ = R(χ²)
        
        dset = Dict( "curve" => χ², "coeffs" => ω, "zhat" => ẑ, "dchi2"  => Δ, "R" => R_ )

        if isfile(output_path)
            h5open(output_path, "r+") do file
                grp = create_group(file, data.ids[i])
                for (key,value) in dset
                    write(grp, key, value)
                end
                #write(file, data.ids[i], collect((χ², ω, ẑ, Δχ²_,R_)) )
            end
        else
            h5open(output_path, "w") do file
                grp = create_group(file, data.ids[i])
                for (key,value) in dset
                    write(grp, key, value)
                end
                #write(file, data.ids[i], collect((χ², ω, ẑ, Δχ²_,R_)))
            end
        end
    end
end

function flow(data; output_path::Union{String,Nothing}= nothing, λmin::Float32=4700f0, δζ::Float32=0.0005f0)
    
    wgrid = Γgrid(;λmin, δζ)
    basis = Basis(wgrid)

    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path,  a file with the same name already exists!"
    end
    
    k,l,n = basis.k, basis.l, basis.n
    fʳ = Vector{Float32}(undef, l)
    σʳ = Vector{Float32}(undef, l)
    χ² = Vector{Float32}(undef, l)
    Ω  = Matrix{Float32}(undef, k,n)

    N = length(data.flux)

    @showprogress for i in 1:N
        
        f      = data.flux[i]
        σ = data.sdev[i]         
        λ = data.awave[i]

        interpolate!(basis, f, σ, λ, fʳ, σʳ)
        flow!(basis, fʳ, σʳ, χ², Ω)

        # predicted redshift ẑ  
        ẑ     = wgrid.ζ[argmin(χ²)]

        # decomposition coeffs 
        ω     = Ω[argmin(χ²)]
        # significance score \Delta\chi2
        Δ  = Δχ²(χ²)
        # Robustness metric R
        R_    = R(χ²)
        
        dset = Dict( "curve" => χ², "coeffs" => ω, "zhat" => ẑ, "dchi2"  => Δ, "R" => R_ )

        if isfile(output_path)
            h5open(output_path, "r+") do file
                grp = create_group(file, data.ids[i])
                for (key,value) in dset
                    write(grp, key, value)
                end
                #write(file, data.ids[i], collect((χ², ω, ẑ, Δχ²_,R_)) )
            end
        else
            h5open(output_path, "w") do file
                grp = create_group(file, data.ids[i])
                for (key,value) in dset
                    write(grp, key, value)
                end
                #write(file, data.ids[i], collect((χ², ω, ẑ, Δχ²_,R_)))
            end
        end
    end
end

function flow(fits_path::String, DataExtName::Union{String, Int}, StatExtName::Union{String, Int}; output_path::Union{String,Nothing}= nothing)
    
    wgrid = Γgrid(;λmin, δζ)
    basis = Basis(wgrid)

    data = leggere(fits_path, Val(:fits); DataExtName = DataExtName, StatExtName = StatExtName)
    
    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path,  a file with the same name already exists!"
    end
    
    k,l,n = basis.k, basis.l, basis.n
    fʳ = Vector{Float32}(undef, l)
    σʳ = Vector{Float32}(undef, l)
    χ² = Vector{Float32}(undef, l)
    Ω  = Matrix{Float32}(undef, k,n)

    N = length(data.flux)

    @showprogress for i in 1:N
        
        f      = data.flux[i]
        σ = data.sdev[i]         
        λ = data.awave[i]

        interpolate!(basis, f, σ, λ, fʳ, σʳ)
        flow!(basis, fʳ, σʳ, χ², Ω)

        # predicted redshift ẑ  
        ẑ     = wgrid.ζ[argmin(χ²)]

        # decomposition coeffs 
        ω     = Ω[argmin(χ²)]
        # significance score \Delta\chi2
        Δ  = Δχ²(χ²)
        # Robustness metric R
        R_    = R(χ²)
        dset = Dict( "curve" => χ², "coeffs" => ω, "zhat" => ẑ, "dchi2"  => Δ, "R" => R_ )

        if isfile(output_path)
            h5open(output_path, "r+") do file
                grp = create_group(file, data.ids[i])
                for (key,value) in dset
                    write(grp, key, value)
                end
                #write(file, data.ids[i], collect((χ², ω, ẑ, Δχ²_,R_)) )
            end
        else
            h5open(output_path, "w") do file
                grp = create_group(file, data.ids[i])
                for (key,value) in dset
                    write(grp, key, value)
                end
                #write(file, data.ids[i], collect((χ², ω, ẑ, Δχ²_,R_)))
            end
        end
    end
end

function flow(wgrid::Γgrid, basis::Basis, fits_path::String, DataExtName::Union{String, Int}, StatExtName::Union{String, Int}; output_path::Union{String,Nothing}= nothing)
    
    data = leggere(fits_path, Val(:fits); DataExtName = DataExtName, StatExtName = StatExtName)

    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path,  a file with the same name already exists!"
    end
    
    k,l,n = basis.k, basis.l, basis.n
    fʳ = Vector{Float32}(undef, l)
    σʳ = Vector{Float32}(undef, l)
    χ² = Vector{Float32}(undef, l)
    Ω  = Matrix{Float32}(undef, k,n)

    N = length(data.flux)

    @showprogress for i in 1:N
        
        f      = data.flux[i]
        σ = data.sdev[i]         
        λ = data.awave[i]

        interpolate!(basis, f, σ, λ, fʳ, σʳ)
        flow!(basis, fʳ, σʳ, χ², Ω)
        # predicted redshift ẑ  
        ẑ     = wgrid.ζ[argmin(χ²)]

        # decomposition coeffs 
        ω     = Ω[argmin(χ²)]
        # significance score \Delta\chi2
        Δ  = Δχ²(χ²)
        # Robustness metric R
        R_    = R(χ²)

        dset = Dict( "curve" => χ², "coeffs" => ω, "zhat" => ẑ, "dchi2"  => Δ, "R" => R_ )
        if isfile(output_path)
            h5open(output_path, "r+") do file
                grp = create_group(file, data.ids[i])
                for (key,value) in dset
                    write(grp, key, value)
                end
                #write(file, data.ids[i], collect((χ², ω, ẑ, Δχ²_,R_)) )
            end
        else
            h5open(output_path, "w") do file
                grp = create_group(file, data.ids[i])
                for (key,value) in dset
                    write(grp, key, value)
                end
                #write(file, data.ids[i], collect((χ², ω, ẑ, Δχ²_,R_)))
            end
        end
    end
end

"""
    flood(path::String, save_path::Union{String, Nothing}; kwargs...)

Description
===========
Reads the data from a datacube in `path`, and runs a `threaded_fnnls` on each spaxel. It then saves 
the resulting chi2 Array into an ".h5" file.

Arguments
=========
- **`path        ::String`**            : Path to the datacube ".fits" file
- **`save_path   ::String`**            : Path where to save the h5file output 

Keyword Arguments
================= 
- **`DataExtName ::String`**   : Name of the data extension
- **`StatExtName ::String`**   : Name of the stat extension

!!! note
    Keyword arguments are passed directly to `leggere_cube`. If the Data extension is named "DATA", there is no need to pass
    this keyword argument. Similarly, if the stat extension is named "STAT", there is no need to pass this keyword

Output
======
An ".h5" file containing  the chi2 Array, located in `save_path`, the key to the h5 dataset is given by
```julia
h5key = splitext(basename(path))[1]
```

Returns
=======
`nothing`

Arguments
=========
- **`fits_path ::String`**     : path to the datacube `.fits` file
- **`h5_path ::String`**       : Name of the stat extension
- **`DataExtName ::String`**   : Name of the data extension
- **`StatExtName ::String`**   : Name of the stat extension

Author(s)
=========
B.Masten
"""
function flow(src::String; δζᵣ::Float32 = 0.0005f0)
    
    @assert isfile(src) "🔴 File not found: $src"
    dst = splitext(src)[1] * "_Moose.h5"
    
    if isfile(dst)
        @warn("⚠️ File already exists @ the default path generated by `Moose.jl`;
               `Moose.jl` will resume writing this file")
    else
        @warn("📄 File generated for the output! @ $(dst)")
    end

    data, stat, metadata = leggere(src, Val(:h5))

    N₁, N₂, N₃, λᵣ, δλ = metadata
    λ = collect(Float32, λᵣ .+ (0:N₃-1) .* δλ)
    
    wgrid  = Γgrid(λmin = λᵣ, δζ = δζᵣ)
    basis  = Basis(wgrid)

    output, iₜ, jₜ = awaken(dst, basis, wgrid, metadata)

    
    #grp_spaxels = output["CHI2_SPAXELS"]
    grp  = output["CHI2_3X3"]
    #grp = output["CHI2_3X3GK"]


    #kernel = [1f0/16f0  1f0/8f0  1f0/16f0;
    #          1f0/8f0   1f0/4f0  1f0/8f0 ;
    #          1f0/16f0  1f0/8f0  1f0/16f0]
    
    kernel = [1f0/9f0  1f0/9f0  1f0/9f0;
              1f0/9f0  1f0/9f0  1f0/9f0;
              1f0/9f0  1f0/9f0  1f0/9f0]          

    stop = false 
    fʳ = zeros(Float32, basis.l)
    σʳ = zeros(Float32, basis.l)
    χ² = Vector{Float32}(undef, basis.n)

    @showprogress for i in iₜ:N₁
        for j in jₜ:N₂
            dset_name = nothing
            try
                dset_name = "$(i)_$(j)"
                status, f, σ = scrub(data, stat, kernel, i, j, N₁, N₂, N₃)
                if status === :good
                    interpolate!(basis, f, σ, λ, fʳ, σʳ)
                    flow!(basis, fʳ, σʳ, χ²)
                    grp[dset_name] = χ²
                else
                    grp[dset_name] = NaN32
                end

            catch err 
                if isa(err, InterruptException)
                    println("🔴 Stop requested! `Moose.jl` will exit after current iteration...")
                    stop = true
                    if haskey(grp, dset_name)
                        delete_object(grp, dset_name)
                    end
                    status, f, σ = scrub(data, stat, kernel, i, j, N₁, N₂, N₃)
                    if status === :good
                        interpolate!(basis, f, σ, λ, fʳ, σʳ)
                        flow!(basis, fʳ, σʳ, χ²)
                        grp[dset_name] = χ²
                    else
                        grp[dset_name] = NaN32
                    end
                else
                    rethrow(err)
                end
            finally
                delete_attribute(output, "j_t") 
                attributes(output)["j_t"] = j + 1
            end
            
            if stop
                break 
            end
        end

        if stop
            break 
        end
        #update iₜ & jₜ attributes
        jₜ = 1
        delete_attribute(output, "i_t")
        attributes(output)["i_t"] = i + 1 
    end
    close(output)
end