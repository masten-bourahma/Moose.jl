using LinearAlgebra, Statistics, HDF5, Base.Threads

"""
    Γgrid(; λmin = 4500f0, λmax = 9350f0, δλ = 1.25f0,
            ζmin = 0f0,    ζmax = 7f0,    δζ = 0.0015f0)

Description
============
Creates a rest-frame log-wavelength grid with uniform spacing. It also provides
the test redshifts vector. All needed arguments to initialize this struct
have default values, so that, one can get the grid without specifying any
argument.

Arguments
==========
- **`λmin ::Float32 = 4500`**   : MUSE minimal observed wavelength in Å
- **`λmax ::Float32 = 9350`**   : MUSE maximal observed wavelength in Å
- **`δλ   ::Float32 = 1.25`**   : MUSE spectral resolution in Å
- **`ζmin ::Float32 = 0`**      : minimum test redshift
- **`ζmax ::Float32 = 7`**      : maximum test redshift
- **`δζ   ::Float32 = 0.0015`** : step of test redshift


Details
========

This structure fills additional fields: 

-**`δΓ :: Float32`**         : rest-frame spectral resolutionm
-**`Γ  :: Vector{Float32}`** : rest-frame grid log-wavelengths vector
-**`ζ  :: Vector{Float32}`** : vector of test redshifts            

Examples
========
    # to get the rest-frame log-wavelength grid just run
    Γ = Γgrid().Γ
!!! note
    The resolution `δΓ` of the grid is not an argument, if one wants to play with this parameter you have to change the code. `δΓ` is calculated 
    as the mean resolution of the following grid:
    ` \\mathrm{mean}\\!\\left( 
        \\log \\lambda_{\\rm obs,min} / (1 + z_{\\rm max}) 
        : \\delta\\lambda / (1 + z_{\\rm max}) 
        : \\lambda_{\\rm obs,max}
      \\right) `

"""
mutable struct Γgrid

    λmin :: Float32
    λmax :: Float32
    δλ   :: Float32
    
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

        βgrid = log10.(collect(4500f0/(1f0 + 7f0): δλ/(1f0 +7f0): λmax))        
        δΓ = sum(βgrid[2:end] .- βgrid[1:end-1])/length(βgrid)

        ζ = collect(Float32, ζmin:δζ:ζmax)
        Γ = collect(Float32, Γmin:δΓ:Γmax)
        
        new(λmin, λmax, δλ, ζmin, ζmax, δζ, Γmin, Γmax, δΓ, Γ, ζ)
    end
end


"""
    Basis()

Description
============
This struct loads the basis vectors matrix H and pre-computes several matricial products and stores them in a julia Struct.
This storage allows to save computational ressources and speed up the code. By default it loads a rank 10 NMF basis vectors
obtained using a sequential nearly-NMF carried on ~7000 MUSE galaxy spectra

Arguments
==========
-**`wgrid  :: Γgrid`** : rest-frame grid Struct

Optional arguments
==================
-**`rank  :: Int`**   : rank of the basis you want to use, default to 10. Rank ∈ [6, 14]  
"""
struct Basis
    H       :: Matrix{Float32}   # Full basis vectors matrix
    Hᵀ      :: Matrix{Float32}   # Transposed basis matrix
    HHᵀᵢ    :: Vector{Matrix{Float32}}
    Hᵢ      :: Vector{Matrix{Float32}}
    
    Λ       :: Matrix{Float32} # rest wavelengths array

    k       :: Int             # Rank of H
    l       :: Int             # spectral dim of basis vectors
    n       :: Int             # number of test redshifts
     
    function Basis(wgrid::Γgrid; rank::Int = 10)
        
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
        new(H, Hᵀ, HHᵀᵢ, Hᵢ, Λ, k, l, n)
    end
end
