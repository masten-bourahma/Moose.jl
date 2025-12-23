using FITSIO, HDF5, LinearAlgebra

#Val() is used to pass values as types

"""
    leggere(path::String, ::Val{:fits}; DataExtName::String= "DATA", StatExtName::String ="STAT", )

Description
===========
This function reads fluxes, standard deviations, wavelengths and ids from a group of fits file in `path` into a NamedTuple.
Output NamedTuple has four fields (flux, sdev, awave, ids). The name of the data and stat extensions can 
be specified by setting the keyword arguments `DataExtName` and `StatExtName`

Arguments
=========
- **`path ::String`**                   : Path to the FITS files folder 
- **`Val{:fits}`**                      : Type-driven dispatch

Keyword arguments
=========
- **`DataExtName ::String = "DATA"`**   : Name of the data extension
- **`StatExtName ::String = "STAT"`**   : Name of the stat extension

returns
=======
- **`data ::NamedTuple`** : NamedTuple with four fields (flux, sdev, awave, ids)

!!! warning
    `DataExtName` and `StatExtName` have to be consistent across fits files
"""
function leggere(path::String, ::Val{:fits}; DataExtName::Union{String, Int}= "DATA", StatExtName::Union{String, Int}= "STAT")

    files = readdir(path; join =true)
    N     = length(files)

    f, σ, λ = [Vector{Vector{Float32}}(undef, N) for i in 1:3]
    fits_ids  = Vector{String}(undef, N)
    
    
    for i in 1:N
        hdul        = FITS(files[i])
        fits_ids[i] = string(read_header(hdul[1])["ID"])
        f[i]       = read(hdul[DataExtName])
        σ[i]       = sqrt.(read(hdul[StatExtName]))

        # clean the input
        replace!(f[i], NaN => 0) 
        replace!(σ[i], NaN => 1f12, Inf => 1f12) 
        σ[i][f[i] .== 0] .= 1f12 
        
        # Read header information
        λᵣ   = read_header(hdul[DataExtName])["CRVAL1"]
        L    = read_header(hdul[DataExtName])["NAXIS1"]
        δλ   = read_header(hdul[DataExtName])["CDELT1"]
        
        # Calculate wavelength array
        λ[i] = λᵣ .+ (0:L-1) .* δλ
        close(hdul);
    end

    return ( flux         = f,
             sdev         = σ,
             awave        = λ,
             ids          = fits_ids)
end

"""
    leggere(path::String, ::Val{:chi2-file})

Description
===========
This function reads chi-square curves, best decomposition coefficients, predicted redshifts, significance scores, robustness scores, ... 
stored in a HDF5 file and to which the argument `path` points at.

Arguments
=========
- **`path ::String`**     : Path to the h5 file 
- **`Val{:chi2-file}`**   : Type-driven dispatch

returns
=======
A NamedTuple with 6 fields (chi2, omega, zhat, dchi2, R)

"""
function leggere(path::String, ::Val{:chi2file})

    if !isfile(path)
        error("🛑 file does not exist!")
    else
        @info "✅ HDF5 file found!"
    end

    file   = h5open(path, "r")
    grp_names         = keys(file)
    N                 = length(grp_names)
    
    χ²₁     = Vector{Vector{Float32}}(undef,N)
    ω₁      = Vector{Vector{Float32}}(undef,N)
    z₁      = Vector{Float32}(undef,N)
    Δχ²₁    = Vector{Float32}(undef,N)
    R₁      = Vector{Float32}(undef,N)

    r₀    = Vector{Vector{Float32}}(undef,N)
    r₁    = Vector{Vector{Float32}}(undef,N)
    z₅₎   = Vector{Vector{Float32}}(undef,N)
    Δχ²₅₎ = Vector{Vector{Float32}}(undef,N)

    for (i,key) in enumerate(grp_names)
        grp      = file[key]
        χ²₁[i]   = read(grp["chi2_1"])
        r₀       = read(grp["r_0"])
        r₁       = read(grp["r_1"])
        ω₁[i]    = read(grp["coeffs_1"]) 
        z₁[i]    = read(grp["z_1"])
        z₅₎      = read(grp["z_1to5"])
        Δχ²₁[i]  = read(grp["dchi2_1"])
        Δχ²₅₎    = read(grp["dchi2_1to5"])
        R₁[i]    = read(grp["R_1"])
    end
    close(file)
    
    data = (id = grp_names, chi2_1 = χ²₁, coeffs_1 = ω₁, r_0 = r₀, r_1 = r₁,
            z_1 = z₁, z_1t05 = z₅₎, dchi2_1 = Δχ²₁ , R_1 = R₁,
            dchi2_1to5 = Δχ²₅₎)
            
    #@info "A NamedTuple will be returned with the following field names and types"
    #println("id", "chi2_1")
    return data
end


"""
    leggere(path::String; DataExtName::String ="DATA", StatExtName::String = "STAT" )

Description
===========
This function reads the flux densities, standard deviations, wavelength array and attributes and id to each spaxel
in the datacube fits file in `path`. It outputs a NamedTuple with four fields (flux, sdev, awave, ids).
The name of the data and stat extension in the fits file can be specified by setting keywords arguments `DataExtName` and `StatExtName`

Arguments
=========
- **`path ::String`**                   : Path to the fits files folder 
- **`DataExtName ::String = "DATA"`**   : Name of the data extension
- **`StatExtName ::String = "STAT"`**   : Name of the stat extension

Returns
=======
- NamedTuple with four fields (flux, sdev, awave, ids)
"""
function leggere(path::String, ::Val{:cube}; DataExtName::Union{String, Int}= "DATA", StatExtName::Union{String, Int}= "STAT")
    
    #check file exists
    if !isfile(path)
        error("🛑 file does not exist!")
    else
        @info "✅ FITS file found!"
    end
    
    #read fits file
    cube = FITS(path)
    
    #get fluxes and stds
    fλ   = read(cube[DataExtName])  
    σλ   = sqrt.(read(cube[StatExtName]))

    replace!(fλ,  NaN => 0)
    replace!(σλ,  0 => 1f6, NaN => 1f6, Inf => 1f6)

    #construct wavelength array
    λref = read_header(cube[DataExtName])["CRVAL3"];
    L    = read_header(cube[DataExtName])["NAXIS3"];
    δλ   = read_header(cube[DataExtName])["CD3_3"];
    λ    = collect(Float32, λref .+ (0:L-1) .* δλ);
    
    close(cube);

    return ( flux         = fλ,
             sdev         = σλ,
             awave        = λ,
             ids          = collect(1:1:size(fλ,1)),)
end
