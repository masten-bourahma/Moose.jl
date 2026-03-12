using Base.Threads, Dates, LinearAlgebra, Peaks
using ProgressMeter: @showprogress

"""
    interpolate( basis::Basis, f::Vector{T}, v::Vector{T}, λ::Vector{T}; fₓ::T = zero(T), vₓ::T = T(1e12))

Description
============
This function takes an instance of the Basis Struct and three vectors: fluxes (f), variances (v) and observed wavelengths (λ), and interpolates 
the flux and variance vectors to the rest wavelength grid assuming a redshift of 0.
    
Arguments
==========
`T` refers to element type, and must a `Real`, which includes `Float32`, `Float64`, and `BigFloat`. 
- **`basis ::Basis`**       : Instance of a Basis Struct
- **`f     ::Vector{T}`**   : Observed flux densities
- **`v     ::Vector{T}`**   : Flux densities associated variances
- **`λ     ::Vector{T}`**   : Observed wavelengths
- **`fₓ    ::T = zero(T)`** : Extrapolation value for fluxes
- **`vₓ    ::T = T(1e12)`** : Extrapolation value for variances

returns
=======
- **`f̃    ::Vector{T}`**   : Interpolated flux densities
- **`ṽ    ::Vector{T}`**   : Interpolated variances

See also
========
`interpolate!()`, works similarly, the difference is that `interpolate!` accepts two additional
arguments, `f̃` and `ṽ`, which are the vectors of interpolated flux densities and interpolated variances. Sizes of these
vectors are known, so one can pass an initialization of them (e.g. with zeros) and `interpolate!` will fill them.
This is helpful because it avoids reallocations.
"""
function interpolate(basis::Basis, f::Vector{T}, v::Vector{T}, λ::Vector{T};
                      fₓ::T = zero(T), vₓ::T = T(1e12)) where{T<:Real} 
    
    Λ₁  = @view basis.Λ[1,:]
    l  = basis.l

    f̃ = zeros(T, l)
    ṽ = zeros(T, l)

    logλ = log10.(λ)
    lwbn = logλ[1]
    upbn = logλ[end]

    @inbounds @threads for j in 1:l 
        
        logλⱼ = Λ₁[j]
        # Left or right extrapolation
        if logλⱼ < lwbn || logλⱼ > upbn
            f̃[j] = fₓ
            ṽ[j] = vₓ
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
        f̃[j] = muladd(p, f[idx + 1]*λᵢ - f[idx]*λᵢ, f[idx]*λᵢ)
        ṽ[j] = muladd(p, v[idx + 1]*(λᵢ^2) - v[idx]*(λᵢ^2), v[idx]*(λᵢ^2))
    end
    return f̃, ṽ
end

function interpolate!(basis::Basis, f::Vector{T}, v::Vector{T}, λ::Vector{T},
                       f̃::Vector{T}, ṽ::Vector{T}; fₓ::T = zero(T), vₓ::T = T(1e12)) where{T<:Real} 
    
    Λ₁  = @view basis.Λ[1,:]
    l = basis.l

    logλ = log10.(λ)
    lwbn = logλ[1]
    upbn = logλ[end]

    @inbounds @threads for j in 1:l 
        
        logλⱼ = Λ₁[j]
            
        # Left or right extrapolation
        if logλⱼ < lwbn || logλⱼ > upbn
            f̃[j] = fₓ
            ṽ[j] = vₓ
            continue
        end

        # Binary search for interpolation index
        idx = searchsortedfirst(logλ, logλⱼ)
        idx = max(1, min(idx - 1, l-1))  # Clamp to valid range

        # Direct memory access
        logλ₁ = logλ[idx]
        logλ₂ = logλ[idx + 1]
            
        inv_denom = 1 / (logλ₂ - logλ₁)
        p         = (logλⱼ - logλ₁) * inv_denom
        λᵢ        = λ[idx]
        
        # Fused multiply-add operations
        f̃[j] = muladd(p, f[idx + 1]*λᵢ - f[idx]*λᵢ, f[idx]*λᵢ)
        ṽ[j] = muladd(p, v[idx + 1]*(λᵢ^2) - v[idx]*(λᵢ^2), v[idx]*(λᵢ^2))
    end
end

function interpolate!(grid::Vector{T}, λ::Vector{T}, f::Vector{T}, f̃::Vector{T}) where{T<:Real} 
    
    @assert length(λ) == length(f)
    @assert length(grid) == length(f̃)

    lwbn = λ[1]
    upbn = λ[end]
    l = length(λ)
    
    @inbounds @threads for j in eachindex(grid)
        λⱼ = grid[j]
        # Left or right extrapolation
        if λⱼ < lwbn || λⱼ > upbn
            f̃[j] = zero(T)
            continue
        end

        # Binary search for interpolation index
        idx = searchsortedfirst(λ, λⱼ)
        idx = max(1, min(idx - 1, l-1))  # Clamp to valid range

        # Direct memory access
        λ₁ = λ[idx]
        λ₂ = λ[idx + 1]
            
        inv_denom = 1 / (λ₂ - λ₁)
        p         = (λⱼ - λ₁) * inv_denom
        
        # Fused multiply-add operations
        f̃[j] = muladd(p, (f[idx + 1] - f[idx])/λⱼ, f[idx]/λⱼ)
    end
end

"""
    flow(basis::Basis, f::Vector{T}, v::Vector{T})

Description
===========
This function executes a multi-threaded calculation of the chi-square curve. For each redshift (each gram matrices Hᵢ & HHᵀᵢ
in basis Struct), it reconstructs an input vector f with basis vectors using a Fast Non-Negative Least Squares (FNNLS),
then quantifies the reconstruction error using a χ² goodness-of-fit stasitic.

Arguments
=========
- **`basis ::Basis`**     : Basis Struct
- **`f     ::Vector{T}`** : (interpolated) flux densities vector
- **`v     ::Vector{T}`** : (interpolated) variance vector

Returns
=======
- **`χ²::Vector{T}`**  : χ² curve

See also
========
`flow!()`

Example
=======
```julia
# 1. Initialize the Basis Struct
wgrid = Γgrid(λmin = 4700f0, δζ = 5f-4)
basis = Basis(wgrid)

# 2. Read your flux density and variances from your FITS file
f, v = ... # complete code

# 3. Interpolate flux densities f and variances v to the rest frame grid
f̃ , ṽ = interpolate(basis, f, v, λ)

# 4. Run the flow() function to get the chi2 curve
χ² = flow(basis, f̃ , ṽ, Val(:no_coeffs))

# Extra. If you also want the decomposition coefficients, run flow with the Val(:coeffs) dispatch
χ², Ω = flow(basis, f̃ , ṽ, Val(:coeffs))

```
Tips
====
To set the number of threads in julia, add this line to your linux/macos bash
```bash
export JULIA_NUM_THREADS=4
```
"""
function flow(basis::Basis, f::Vector{T}, v::Vector{T}, ::Val{:no_coeffs}) where{T<:Real}
        
    n,l,k      = basis.n, basis.l,basis.k
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
                Σ += (fʰ[j] - f[j])^2 / v[j]
            end
            χ²[i] = Σ / l
        end
    end
    return χ²
end

function flow(basis::Basis, f::Vector{T}, v::Vector{T}, ::Val{:coeffs}) where{T<:Real}
    
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
                Σ += (fʰ[j] - f[j])^2 / v[j]
            end
            χ²[i] = Σ / l
            Ω[:,i] .= ω
        end
    end
    return χ², Ω
end

function flow!(basis::Basis, f::Vector{T}, v::Vector{T}, χ²::Vector{T}) where{T}
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
                Σ += (fʰ[j] - f[j])^2 / v[j]
            end
            χ²[i] = Σ / l
        end
    end
end

function flow!(basis::Basis, f::Vector{T}, v::Vector{T}, χ²::Vector{T}, Ω::Matrix{T}) where{T<:Real}
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
                Σ += (fʰ[j] - f[j])^2 / v[j]
            end
            χ²[i] = Σ / l
            Ω[:,i] .= ω
        end
    end
end

function flow!(basis::Basis, f::Vector{T}, v::Vector{T}, fᵃ::Vector{T}, χ²::Vector{T}, Ω::Matrix{T}) where{T<:Real}
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
                Σ += (fʰ[j] - f[j])^2 / v[j]
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
Runs a threaded FNNLS on a set of spectra loaded into a NamedTuple (`data`). The function saves obtained χ² curves
into an `.h5` file, whose path and name are specified by setting the keyword argument `output_path`. 

Arguments
=========
- **`wgrid       ::Γgrid`**     : Γgrid Struct
- **`basis       ::Basis`**     : Basis Struct
- **`data        ::Struct`**    : Named Tuple with four fields: flux, sdev, wave, ids
- **`output_path ::String`**    : Path where the chi2 file will be saved 

Returns
=======
nothing

Related methods
===============
- `flow(data; output_path::String = nothing)`: `wgrid` and `basis` are instantiated within the method.
- `flow(fits_path::String, DataExtName::Union{String, Int}, StatExtName::Union{String, Int}; output_path::String = nothing)`: `wgrid` and `basis` are instantiated within the method, the data is loaded within the function using the provided `fits_path` and the extensions: `DataExtName`, `StatExtName`.
- `flow(wgrid::Γgrid, basis::Basis, fits_path::String, DataExtName::Union{String, Int}, StatExtName::Union{String, Int}; output_path::String = nothing)`: the data  is loaded within the function using the provided `fits_path` and the extensions: `DataExtName`, `StatExtName`.

Example
=======
```julia
# 1. Initialize the `Γgrid` and `Basis` Structs
wgrid = Γgrid(;λmin, δζ)
basis = Basis(wgrid)

# 2. read data from fits files
data = leggere("../data/spectra_sample", Val(:fits))

#3. Specify the path where to save the outputs (must have an h5 extension!), and call flow()
# if the output path is not specified a default path will be generated @ "../output/results/chi2_files/chi2_$(now).h5"
# where now refers to the current time function from `Dates.jl`
output_path = "../output/chi2_files/chi2_moose.h5" 
flow(wgrid, basis, data; output_path = output_path)

```
"""
function flow(wgrid::Γgrid, basis::Basis, data; output_path::Union{String,Nothing}= nothing)

    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path, a file with the same name already exists!"
    end
    # Preallocate vectors
    T               = eltype(data.flux[1])
    N, k, l, n      = length(data.flux), basis.k, basis.l, basis.n
    fʳ  = Vector{T}(undef, l)
    vʳ  = Vector{T}(undef, l)
    r₁  = Vector{T}(undef, l)
    χ²₁ = Vector{T}(undef, n)
    Ω   = Matrix{T}(undef, k, n)

    @showprogress for i in 1:N
        f, v, λ, id  = data.flux[i], data.var[i], data.awave[i], data.ids[i]
        interpolate!(basis, f, v, λ, fʳ, vʳ)
        flow!(basis, fʳ, vʳ, χ²₁, Ω)
        
        # predicted redshift z₁  
        z₁ = wgrid.ζ[argmin(χ²₁)]
        
        # decomposition coeffs @ z₁
        ω₁ = Ω[:, argmin(χ²₁)]
        
        # reconstruction @ z₁
        r₁ = vec(ω₁' * basis.H)

        # significance score \Delta\chi2
        Δ = Δχ²(χ²₁)
        
        # Robustness metric R
        R_ = R(χ²₁)

        #indices of the first 10 solutions
        indices = findminima(χ²₁, 20).indices
        sorted_indices      = indices[sortperm(Δχ²(χ²₁, indices), rev=true)][1:10]
        Δ1to10   = Δχ²(χ²₁, sorted_indices)
        z1to10   = wgrid.ζ[sorted_indices]

        dset = Dict( "id"          => id,    "chi2_1" => χ²₁, "r_0"     => fʳ,     "r_1" => r₁, 
                     "coeffs_1"    => ω₁,    "z_1"    => z₁,  "z_1to10" => z1to10,
                     "dchi2_1"     => Δ,     "R_1"    => R_,
                     "dchi2_1to10" => Δ1to10)

        if isfile(output_path)
            h5open(output_path, "r+") do file
                grp = create_group(file, data.ids[i])
                for (key,value) in dset
                    write(grp, key, value)
                end
            end
        else
            h5open(output_path, "w") do file
                grp = create_group(file, data.ids[i])
                for (key,value) in dset
                    write(grp, key, value)
                end
            end
        end
    end
end

function flow(data; output_path::Union{String,Nothing}= nothing, λmin::T = T(4700), δζ::T= T(5e-4)) where{T<:Real}
    
    wgrid = Γgrid(;λmin, δζ)
    basis = Basis(wgrid)

    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path,  a file with the same name already exists!"
    end
    
    k,l,n = basis.k, basis.l, basis.n
    fʳ  = Vector{T}(undef, l)
    vʳ  = Vector{T}(undef, l)
    r₁  = Vector{T}(undef, l)
    χ²₁ = Vector{T}(undef, n)
    Ω   = Matrix{T}(undef, k, n)

    N = length(data.flux)

    @showprogress for i in 1:N
        
        f, v, λ, id  = data.flux[i], data.var[i], data.awave[i], data.ids[i]

        interpolate!(basis, f, v, λ, fʳ, vʳ)
        flow!(basis, fʳ, vʳ, χ²₁, Ω)
        
        # predicted redshift z₁  
        z₁ = wgrid.ζ[argmin(χ²₁)]
        
        # decomposition coeffs @ z₁
        ω₁ = Ω[:, argmin(χ²₁)]
        
        # reconstruction @ z₁
        r₁ = vec(ω * basis.H)

        # significance score \Delta\chi2
        Δ = Δχ²(χ²₁)
        
        # Robustness metric R
        R_ = R(χ²₁)

        #indices of the first 10 solutions
        indices = findminima(χ²₁, 20).indices
        sorted_indices      = indices[sortperm(Δχ²(χ²₁, indices), rev=true)][1:10]
        Δ1to10   = Δχ²(χ²₁, sorted_indices)
        z1to10   = wgrid.ζ[sorted_indices]

        dset = Dict( "id"          => id,    "chi2_1" => χ²₁, "r_0"     => fʳ,     "r_1" => r₁, 
                     "coeffs_1"    => ω₁,    "z_1"    => z₁,  "z_1to10" => z1to10,
                     "dchi2_1"     => Δ,     "R_1"    => R_,
                     "dchi2_1to10" => Δ1to10)

        if isfile(output_path)
            h5open(output_path, "r+") do file
                grp = create_group(file, id)
                for (key,value) in dset
                    write(grp, key, value)
                end
            end
        else
            h5open(output_path, "w") do file
                grp = create_group(file, id)
                for (key,value) in dset
                    write(grp, key, value)
                end
            end
        end
    end
end

function flow(fits_path::String, DataExtName::Union{String, Int}, StatExtName::Union{String, Int},
              λmin::T = T(4700), δζ::T= T(5e-4); output_path::Union{String,Nothing}= nothing) where{T<:Real}
    
    wgrid = Γgrid(;λmin, δζ)
    basis = Basis(wgrid)

    data = leggere(fits_path, Val(:fits); DataExtName = DataExtName, StatExtName = StatExtName, T = T)
    
    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path,  a file with the same name already exists!"
    end
    
    N, k,l,n        = length(data.flux), basis.k, basis.l, basis.n
    fʳ  = Vector{T}(undef, l)
    vʳ  = Vector{T}(undef, l)
    r₁  = Vector{T}(undef, l)
    χ²₁ = Vector{T}(undef, n)
    Ω   = Matrix{T}(undef, k, n)

    @showprogress for i in 1:N
        
        f, v, λ, id  = data.flux[i], data.var[i], data.awave[i], data.ids[i]
        interpolate!(basis, f, v, λ, fʳ, vʳ)
        flow!(basis, fʳ, vʳ, χ²₁, Ω)
        
        # predicted redshift z₁  
        z₁ = wgrid.ζ[argmin(χ²₁)]
        
        # decomposition coeffs @ z₁
        ω₁ = Ω[:, argmin(χ²₁)]
        
        # reconstruction @ z₁
        r₁ = vec(ω * basis.H)

        # significance score \Delta\chi2
        Δ = Δχ²(χ²₁)
        
        # Robustness metric R
        R_ = R(χ²₁)

        #indices of the first 10 solutions
        indices = findminima(χ²₁, 20).indices
        sorted_indices      = indices[sortperm(Δχ²(χ²₁, indices), rev=true)][1:10]
        Δ1to10   = Δχ²(χ²₁, sorted_indices)
        z1to10   = wgrid.ζ[sorted_indices]

        dset = Dict( "id"          => id,    "chi2_1" => χ²₁, "r_0"     => fʳ,     "r_1" => r₁, 
                     "coeffs_1"    => ω₁,    "z_1"    => z₁,  "z_1to10" => z1to10,
                     "dchi2_1"     => Δ,     "R_1"    => R_,
                     "dchi2_1to10" => Δ1to10)

        if isfile(output_path)
            h5open(output_path, "r+") do file
                grp = create_group(file, id)
                for (key,value) in dset
                    write(grp, key, value)
                end
            end
        else
            h5open(output_path, "w") do file
                grp = create_group(file, id)
                for (key,value) in dset
                    write(grp, key, value)
                end
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
    T               = eltype(data.flux[1])
    N, k,l,n        = length(data.flux), basis.k, basis.l, basis.n
    fʳ  = Vector{T}(undef, l)
    vʳ  = Vector{T}(undef, l)
    r₁  = Vector{T}(undef, l)
    χ²₁ = Vector{T}(undef, n)
    Ω   = Matrix{T}(undef, k, n)

    @showprogress for i in 1:N
        
        f, v, λ, id  = data.flux[i], data.var[i], data.awave[i], data.ids[i]
        interpolate!(basis, f, v, λ, fʳ, vʳ)
        flow!(basis, fʳ, vʳ, χ²₁, Ω)
        
        # predicted redshift z₁  
        z₁ = wgrid.ζ[argmin(χ²₁)]
        
        # decomposition coeffs @ z₁
        ω₁ = Ω[:, argmin(χ²₁)]
        
        # reconstruction @ z₁
        r₁ = vec(ω * basis.H)

        # significance score \Delta\chi2
        Δ = Δχ²(χ²₁)
        
        # Robustness metric R
        R_ = R(χ²₁)

        #indices of the first 10 solutions
        indices = findminima(χ²₁, 20).indices
        sorted_indices      = indices[sortperm(Δχ²(χ²₁, indices), rev=true)][1:10]
        Δ1to10   = Δχ²(χ²₁, sorted_indices)
        z1to10   = wgrid.ζ[sorted_indices]

        dset = Dict( "id"          => id,    "chi2_1" => χ²₁, "r_0"     => fʳ,     "r_1" => r₁, 
                     "coeffs_1"    => ω₁,    "z_1"    => z₁,  "z_1to10" => z1to10,
                     "dchi2_1"     => Δ,     "R_1"    => R_,
                     "dchi2_1to10" => Δ1to10)

        if isfile(output_path)
            h5open(output_path, "r+") do file
                grp = create_group(file, id)
                for (key,value) in dset
                    write(grp, key, value)
                end
            end
        else
            h5open(output_path, "w") do file
                grp = create_group(file, id)
                for (key,value) in dset
                    write(grp, key, value)
                end
            end
        end
    end
end

#"""
#    flow(src::String, kernel::Matrix{T}; δζᵣ::T = T(5e-4)) where{T<:Real} 
#
#Description
# ===========
#Reads the data from a datacube in `src`, and runs a threaded FNNLS on each spectrum in a sliding 3x3 mean kernel. It then writes 
#the resulting chi-square curves into an HDF5 file.

#Arguments
# =========
#- **`src    :: String`**    : path to the HDF5 file containing the datacube
#- **`kernel :: Matrix{T}`** : PSF kernel

#Keyword arguments
# =================
# **`δζᵣ    :: T`**         : test redshifts' grid spacing for the run

#Output
# ======
#An HDF5 file containing  the chi-square curves and other run metadata. The path to this file is generated by adding the suffix "_Moose" to the source file.
#The HDF5 file contains an HDF5 group named "CHI2_KERNEL". This group contains HDF5 datasets, each dataset corresponds to one chi-square curve with a name corresponding
#to its indices in the cube. For example the chi-square curve at indices (1,1) will have the name "1_1". 

#Returns
# =======
#`nothing`
#Author(s)
# =========
#B.Masten
#"""
function flow(src::String, kernel::Matrix{T}; δζᵣ::T = T(5e-4)) where{T<:Real} 
    
    @assert isfile(src) "🔴 File not found: $src"
    dst = splitext(src)[1] * "_Moose.h5"
    
    if isfile(dst)
        @info("⚠️ File already exists @ the default path generated by `Moose.jl`;
               `Moose.jl` will resume writing this file")
    else
        @info("📄 File generated for the output! @ $(dst)")
    end

    data, stat, metadata = leggere(src, Val(:h5cube))
    N₁, N₂, N₃, λᵣ, δλ   = metadata
    λ = collect(T, λᵣ .+ (0:N₃-1) .* δλ)
    
    wgrid  = Γgrid(λmin = λᵣ, δζ = δζᵣ)
    basis  = Basis(wgrid)

    output, iₜ, jₜ = awaken(dst, basis, wgrid, metadata)
    grp  = output["CHI2_KERNEL"]   
        
    stop = false 

    # Preallocate vectors
    fʳ = zeros(T, basis.l)
    vʳ = zeros(T, basis.l)
    χ² = zeros(T, basis.n)

    @showprogress for i in iₜ:N₁
        for j in jₜ:N₂
            dset_name = nothing
            try
                dset_name = "$(i)_$(j)"
                status, f, v = scrub(data, stat, kernel, i, j, N₁, N₂, N₃)
                if status === :good
                    interpolate!(basis, f, v, λ, fʳ, vʳ)
                    flow!(basis, fʳ, vʳ, χ²)
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
                    status, f, v = scrub(data, stat, kernel, i, j, N₁, N₂, N₃)
                    if status === :good
                        interpolate!(basis, f, v, λ, fʳ, vʳ)
                        flow!(basis, fʳ, vʳ, χ²)
                        grp[dset_name] = χ²
                    else
                        grp[dset_name] = T(NaN)
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

#"""
#    flow(src::String, ::Val{:spaxel}; δζᵣ::Float32 = 0.0005f0)
#
#Description
# ===========
#Reads the data from a datacube in `src`, and runs a threaded FNNLS on each spaxel. It then writes 
#the resulting chi-square curves into an HDF5 file.
#Arguments
# =========
#- **`src ::String`**          : path to the HDF5 file containing the datacube
#- **`Val(:kernel) ::String`** : type-driven dispatch
#Keyword arguments
# =================
#- **`δζᵣ ::Flot32`**          : test redshifts spacing for the run
#Output
# ======
#An HDF5 file containing  the chi-square curves and other run metadata. The path to this file is generated by adding the suffix "_Moose" to the source file.
#The HDF5 file contains an HDF5 group named "CHI2_SPAXELS". This group contains HDF5 datasets, each dataset corresponds to one chi-square curve with a name corresponding
#to its indices in the cube. For example the chi-square curve at indices (1,1) will have the name "1_1". 
#Returns
# =======
#`nothing`
# Author(s)
# =========
#B.Masten
#"""
function flow(src::String, ::Val{:spaxels}; δζᵣ::Float32 = 0.0005f0)
    
    @assert isfile(src) "🔴 File not found: $src"
    dst = splitext(src)[1] * "_Moose.h5"
    
    if isfile(dst)
        @info("⚠️ File already exists @ the default path generated by `Moose.jl`;
               `Moose.jl` will resume writing this file")
    else
        @info("📄 File generated for the output! @ $(dst)")
    end

    data, stat, metadata = leggere(src, Val(:h5cube))

    N₁, N₂, N₃, λᵣ, δλ = metadata
    λ = collect(Float32, λᵣ .+ (0:N₃-1) .* δλ)
    
    wgrid  = Γgrid(λmin = λᵣ, δζ = δζᵣ)
    basis  = Basis(wgrid)

    output, iₜ, jₜ = awaken(dst, basis, wgrid, metadata)
    
    grp = output["CHI2_SPAXELS"]

    stop = false 
    fʳ = zeros(Float32, basis.l)
    σʳ = zeros(Float32, basis.l)
    χ² = Vector{Float32}(undef, basis.n)

    @showprogress for i in iₜ:N₁
        for j in jₜ:N₂
            dset_name = nothing
            try
                dset_name = "$(i)_$(j)"
                status, f, σ = scrub(data, stat, i, j, N₃)
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
                    status, f, σ = scrub(data, stat, i, j, N₃)
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