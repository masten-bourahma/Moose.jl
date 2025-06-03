module Strutture

using Statistics

export Γgrid

"""
    Γgrid(; λmin = 4500f0, λmax = 9350f0, δλ = 1.25f0,
            ζmin = 0f0,    ζmax = 7f0,    δζ = 0.0015f0)

Description
============

Creates a rest-frame log-wavelength grid regularly spaced. And provides
the test redshifts vector. All needed arguments to initialize this struct
have default values, so that, one can get the grid without specifying any
argument. Note that the resolution `δΓ` of the grid is not an argument, if one
wants to play with this parameter you have to change the code. `δΓ` is calculated 
as the mean resolution of the rest-frame wavelength grid not regularly spaced:
log10(min observed wavelength/ (1+maximum redshift) : instrument resolution / (1+maximum redshift)):
maximum observed wavelength.

Arguments
==========

- **`λmin ::Float32 = 4500`**   : MUSE minimal observed wavelength in Å
- **`λmax ::Float32 = 9350`**   : MUSE maximal observed wavelength in Å
- **`δλ   ::Float32 = 1.24`**   : MUSE spectral resolution in Å
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

"""
struct Γgrid

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

    function Γgrid(;λmin = 4500f0, λmax = 9350f0, δλ = 1.25f0,
                    ζmin = 0f0,    ζmax = 7f0,    δζ = 0.0015f0)
        
        Γmin  = log10(λmin/(1+ζmax))
        Γmax  = log10(λmax)

        βgrid = log10.(collect(λmin/(1+ζmax):δλ/(1+ζmax):λmax))
        
        δΓ = sum(βgrid[2:end] .- βgrid[1:end-1])/length(βgrid)
        #δΓ = Statistics.quantile((βgrid[2:end] .- βgrid[1:end-1]), 0.75)
        #δΓ =4f-5;#maximum((βgrid[2:end] .- βgrid[1:end-1]))
        ζ = collect(Float32, ζmin:δζ:ζmax)
        Γ = collect(Float32, Γmin:δΓ:Γmax)
        
        new(λmin, λmax, δλ, ζmin, ζmax, δζ, Γmin, Γmax, δΓ, Γ, ζ)
    end
end


"""
    Basis()

Description
============
This struct loads the basis vectors matrix H and pre-computes several matricial products and stores them in its fields.
This storage allows to save computational ressources and speed the code. No arguments are needed to initialize this struct.

Arguments
==========

Details
=======



"""

struct Basis

    H       :: Matrix{Float32}   # Full basis vectors matrix
    Ht      :: Matrix{Float32}   # Transposed basis matrix
    k       :: Int32             # Rank of H
    
    HHti    :: Vector{Matrix{Float32}}
    Hi      :: Vector{Matrix{Float32}}
    l       :: Int32
    n       :: Int32
    
    λinterp :: Matrix{Float32}

    function Basis()

        γ = Γgrid()
        n =  length(γ.ζ)

        # Load basis matrix 
        Hpath = "/home/masten/Desktop/moose/moose_code/results/H_matrices/H.h5"
        H = try
            h5open(Hpath, "r") do file
                read(file, "rank_7")  # Ensure this returns Matrix{Float32}
            end
        catch err
            error("Failed to load H: ", err)
        end
        
        Ht = transpose(H)  # More efficient than H'
        k = Int(size(H, 1))

        l    = Int32(floor.((log10.(γ.λmax) - log10.(4700)) / γ.δΓ)) 
        λstrt       = log10.(4700f0 ./ (1 .+ γ.ζ))
        istrt       = Int32.(floor.((λstrt .- γ.Γmin) ./ γ.δΓ))

        HHti= Vector{Matrix{Float32}}(undef, n)
        Hi  = Vector{Matrix{Float32}}(undef, n)

        Threads.@threads for i in axes(istrt,1)
            temp = copy(H[ :, istrt[i]:istrt[i] + l -1])
            
            for j in eachindex(temp)
                if issubnormal(temp[j])
                    temp[j] = 0.0f0
                end
            end
            
            Hi[i]    = temp
            HHti[i]  = H[:, istrt[i]:istrt[i] + l] * Ht[istrt[i]:istrt[i] + l, :]
        end

        #interpolation basis
        λinterp = transpose(collect(Float32, log10(4700): γ.δΓ : γ.Γmax - γ.δΓ)) .- log10.(1 .+ γ.ζ)
        # Create instance
        new(H, Ht, k, HHti, Hi, l,  n, λinterp)
    end
end


end