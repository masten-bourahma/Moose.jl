using LinearAlgebra, Statistics, HDF5, Base.Threads

@doc raw"""
    Γgrid(; λmin = 4500f0, λmax = 9350f0, δλ = 1.25f0,
            ζmin = 0f0,    ζmax = 7f0,    δζ = 0.001f0)

Description
============
This function creates a rest-frame log-wavelength grid with a uniform spacing, and provides
the test redshifts vector. The function has only kerword arguments, and all of them have default values.

Arguments
==========
- **`λmin ::Float32 = 4600f0`**   : MUSE minimal observed wavelength in Å
- **`λmax ::Float32 = 9350f0`**   : MUSE maximal observed wavelength in Å
- **`δλ   ::Float32 = 1.25f0`**   : MUSE spectral sampling in Å
- **`ζmin ::Float32 = 0f0`**      : minimal test redshift
- **`ζmax ::Float32 = 6.7f0`**    : maximal test redshift
- **`δζ   ::Float32 = 0.001f0`**  : test redshifts step
Details
========
This structure fills additional fields: 

- **`δΓ :: Float32`**         : the rest-frame log-wavelength grid uniform spacing
- **`Γ  :: Vector{Float32}`** : the rest-frame grid log-wavelengths vector
- **`ζ  :: Vector{Float32}`** : the vector of test redshifts            

Examples
========
```julia
    # to get the rest-frame log-wavelength run
    Γ = Γgrid().Γ
    # to get the vector of test redshifts
    ζo publish  = Γgrid().ζ
```

!!! note
    The grid resolution `δΓ` is not an input argument. To modify this parameter, the code itself must be changed.  
    `δΓ` is computed as:
    ```math
    \delta \Gamma = \mathrm{mean} \left(
        \log\!\left(\frac{\lambda_{\mathrm{obs,min}}}{1 + z_{\mathrm{max}}}\right)
        : \frac{\delta\lambda}{1 + z_{\mathrm{max}}}
        : \lambda_{\mathrm{obs,max}}
    \right)
    ```
"""
mutable struct Γgrid

    λmin :: Float32
    λmax :: Float32
    δλ   :: Float32
    
    λ    :: Vector{Float32}

    ζmin :: Float32
    ζmax :: Float32
    δζ   :: Float32

    Γmin :: Float32
    Γmax :: Float32
    
    δΓ :: Float32            
    Γ  :: Vector{Float32} 
    ζ  :: Vector{Float32}

    function Γgrid(;λmin = 4600f0, λmax = 9350f0, δλ = 1.25f0,
                    ζmin = 0f0, ζmax = 6.7f0, δζ = 0.0005f0)
        # edges are hard coded, because the full rest frame grid should always be the same
        Γmin  = log10(4500f0/(1f0 + 7f0))
        Γmax  = log10(λmax)

        # observed-frame grid
        λ = collect(Float32, λmin:δλ:λmax)
        
        # reference grid
        βgrid = log10.(collect(4500f0/(1f0 + 7f0): δλ/(1f0 +7f0): λmax))
        
        # rest-frame grid
        δΓ = sum(βgrid[2:end] .- βgrid[1:end-1])/length(βgrid)

        ζ = collect(Float32, ζmin:δζ:ζmax)
        Γ = collect(Float32, Γmin:δΓ:Γmax)
        
        new(λmin, λmax, δλ, λ, ζmin, ζmax, δζ, Γmin, Γmax, δΓ, Γ, ζ)
    end
end


"""
    Basis(wgrid::Γgrid; rank::Int = 10)

Description
============
This function instantiates a Basis Struct. It loads the basis vectors matrix `H`, stores slices of `H` corresponding to test redshifts, and pre-computes several matricial products.
By default it loads a rank-10 NMF basis vectors obtained using a sequential nearly-NMF carried on ~ 7000 MUSE galaxy spectra.

Arguments
==========
-**`wgrid  :: Γgrid`** : rest-frame grid Struct

Optional arguments
==================
-**`rank  :: Int`**   : rank of the basis you want to use, default to 10. Rank ∈ [6, 14]  

Fields
======
- **`H    :: Matrix{Float32}`**: Full basis vectors matrix
- **`Hᵀ   :: Matrix{Float32}`**: Transposed basis vectors matrix
- **`Hᵢ   :: Vector{Matrix{Float32}}`**: Slices of H corresponding to test redshifts in wgrid.ζ
- **`HHᵀᵢ :: Vector{Matrix{Float32}}`**: Hᵢ * Hᵢᵀ matricial products
- **`Λ    :: Matrix{Float32}`**: Rest wavelengths array corresponding to test redshifts in wgrid.ζ
- **`k    :: Int`**: Rank of H
- **`l    :: Int`**: Spectral size of basis vectors
- **`n    :: Int`**: Number of test redshifts
"""
mutable struct Basis
    H       :: Matrix{Float32}   # Full basis vectors matrix
    Hᵀ      :: Matrix{Float32}   # Transposed basis matrix
    HHᵀᵢ    :: Vector{Matrix{Float32}}
    Hᵢ      :: Vector{Matrix{Float32}}
    Λ       :: Union{Nothing, Matrix{Float32}} # rest wavelengths array
    k       :: Int             # Rank of H
    l       :: Int             # spectral dim of basis vectors
    n       :: Int             # number of test redshifts
end
function Basis(wgrid::Γgrid; rank::Int = 10)    
    if (rank < 6) || (rank > 14)
        rank = 10
        @warn ("⚠️ rank ∈ [6, 14], provided value for the rank is outside these limits, defaulting rank to 10")
    end

    script_dir  = @__DIR__
    path = joinpath(script_dir, "../data/basis_vectors/H_v1.2.h5")

    # Load basis matrix 
    H = try
        h5open(path, "r") do file
            read(file, "rank_$rank")  # Ensure this returns Matrix{Float32}
        end
    catch err
        error("Failed to load H: ", err)
    end
    # mask a blended feature in H
    H[2, 8424: 8544] .= 0f0
        
    # clip subnormal values --> cause performance bottlneck
    @. H = ifelse(issubnormal(H), 0f0, H)
    Hᵀ = transpose(H)
        
    n = Int(length(wgrid.ζ))
    k = Int(size(H, 1))
    l = Int(floor.((log10.(wgrid.λmax) - log10.(wgrid.λmin)) / wgrid.δΓ)) 
        
    λₛ       = log10.(wgrid.λmin ./ (1 .+ wgrid.ζ))
    iₛ       = Int.(floor.((λₛ .- wgrid.Γmin) ./ wgrid.δΓ))

    HHᵀᵢ= Vector{Matrix{Float32}}(undef, n)
    Hᵢ  = Vector{Matrix{Float32}}(undef, n)

    @views @threads for j in axes(iₛ,1)
        Hₜ      = H[:, iₛ[j]:iₛ[j] + l - 1]
        Hᵢ[j]   = Hₜ
        HHᵀᵢ[j] = similar(H, k, k)
        mul!(HHᵀᵢ[j], Hₜ, Transpose(Hₜ))  # in-place, avoids temporary matrix
        #H[:, iₛ[j]:iₛ[j] + l] * Hᵀ[iₛ[j]:iₛ[j] + l, :]
    end

    #interpolation basis
    Λ = (log10(wgrid.λmin).+ (0:(l-1)) .* wgrid.δΓ)' .- log10.(1 .+ wgrid.ζ)
    return Basis(H, Hᵀ, HHᵀᵢ, Hᵢ, Λ, k, l, n)
end

function Basis(wgrid::Γgrid, ::Val{:observed}; rank::Int = 10)
        
    if (rank < 6) || (rank > 14)
        rank = 10
        @warn ("⚠️ rank ∈ [6, 14], provided value for the rank is outside these limits, defaulting rank to 10")
    end

    script_dir  = @__DIR__
    path = joinpath(script_dir, "../data/basis_vectors/H_2p170461fm5_0p8X.h5")

    # Load basis matrix 
    H = try
            h5open(path, "r") do file
                read(file, "rank_$rank")  # Ensure this returns Matrix{Float32}
            end
        catch err
            error("Failed to load H: ", err)
        end
    # mask a blended feature in H
    H[2, 8424: 8544] .= 0f0
        
    # clip subnormal values --> cause performance bottlneck
    @. H = ifelse(issubnormal(H), 0f0, H)

    Hᵀ = transpose(H)
        
    n = Int(length(wgrid.ζ))
    k = Int(size(H, 1))
    l = Int(floor.((log10.(wgrid.λmax) - log10.(wgrid.λmin)) / wgrid.δΓ)) 
        
    λₛ       = log10.(wgrid.λmin ./ (1 .+ wgrid.ζ))
    iₛ       = Int.(floor.((λₛ .- wgrid.Γmin) ./ wgrid.δΓ))

    HHᵀᵢ= Vector{Matrix{Float32}}(undef, n)
    Hᵢ = [Matrix{Float32}(undef, k, length(wgrid.λ)) for _ in 1:n]

    @views @threads for j in axes(iₛ,1)
        Hₜ      = H[:, iₛ[j]:iₛ[j] + l - 1]
        λₜ      = (10 .^ (wgrid.Γ[iₛ[j]:iₛ[j] + l - 1])) .* (1 + wgrid.ζ[j])
        observed!(wgrid.λ, λₜ, Hₜ, Hᵢ[j])
        
        HHᵀᵢ[j] = similar(H, k, k)
        mul!(HHᵀᵢ[j], Hᵢ[j], Transpose(Hᵢ[j]))
    end
    return Basis(H, Hᵀ, HHᵀᵢ, Hᵢ, nothing, k, length(wgrid.λ), n)
end
function update!(basis::Basis, wgrid::Γgrid, ::Val{:observed})
    # parameters that might change
    n = Int(length(wgrid.ζ))
    k = basis.k
    l = Int(floor.((log10.(wgrid.λmax) - log10.(wgrid.λmin)) / wgrid.δΓ)) 

    # commit n & l
    basis.n = n
    basis.l = length(wgrid.λ)

    λₛ = log10.(wgrid.λmin ./ (1 .+ wgrid.ζ))
    iₛ = Int.(floor.((λₛ .- wgrid.Γmin) ./ wgrid.δΓ))

    # Resize vectors only if needed
    if length(basis.Hᵢ) != n
        resize!(basis.Hᵢ, n)
        resize!(basis.HHᵀᵢ, n)
    end

     @views @threads for j in 1:n
        Hₜ = basis.H[:, iₛ[j]:iₛ[j] + l - 1]
        λₜ = (10 .^ (wgrid.Γ[iₛ[j]:iₛ[j] + l - 1])) .* (1 + wgrid.ζ[j])

        if basis.Hᵢ[j] === nothing || size(basis.Hᵢ[j]) != (k, basis.l)
            basis.Hᵢ[j] = zeros(Float32, (k, basis.l))
        end
        observed!(wgrid.λ, λₜ, Hₜ, basis.Hᵢ[j])

        if basis.HHᵀᵢ[j] === nothing || size(basis.HHᵀᵢ[j],1) != basis.k
            basis.HHᵀᵢ[j] = Matrix{Float32}(undef, basis.k, basis.k)
        end
        mul!(basis.HHᵀᵢ[j], basis.Hᵢ[j], Transpose(basis.Hᵢ[j]))
    end    
end
