using FITSIO, HDF5, LinearAlgebra

#Val() is used to pass values as types

"""
    leggere(path::String; DataExtName::String= "DATA", StatExtName::String ="STAT", ::Val{:Nfits})

Description
===========
This function reads fluxes, standard deviations, wavelengths and ids from a group of fits file in `path` into a NamedTuple.
Output NamedTuple has four fields (flux, sdev, awave, ids). The name of the data and stat extensions can 
be specified by setting the keyword arguments `DataExtName` and `StatExtName`

Arguments
=========
- **`path ::String`**                   : Path to the fits files folder 
- **`DataExtName ::String = "DATA"`**   : Name of the data extension
- **`StatExtName ::String = "STAT"`**   : Name of the stat extension
- **`Val{:fits}`**   :

returns
=======
- **`data ::NamedTuple`** : NamedTuple with four fields (flux, sdev, awave, ids)

```warning
    `DataExtName` and `StatExtName` have to be consistent across fits files
````
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
    leggere(path::String)

Description
===========
This function reads χ² curves, best decomposition parameters ω , predicted redshifts ẑ, significance scores Δχ², and robustness scores R, 
stored in a `.h5` file and to which the argument `path` points at.

Arguments
=========
- **`path ::String`** : Path to the h5 file 

returns
=======
- NamedTuple with 6 fields (ids, chi2, omega, zhat, dchi2, R)

"""
function leggere(path::String, ::Val{:h5})

    if !isfile(path)
        error("🛑 file does not exist!")
    else
        @info "✅ HDF5 file found!"
    end

    file   = h5open(path, "r")
    h5keys = keys(file)
    N      = length(h5keys)
    
    χ²     = Vector{Vector{Float32}}(undef,N)
    ω      = Vector{Vector{Float32}}(undef,N)
    ẑ      = Vector{Float32}(undef,N)
    Δχ²    = Vector{Float32}(undef,N)
    R      = Vector{Float32}(undef,N)


    for (i,key) in enumerate(h5keys)
        grp     = file[key]
        χ²[i]   = read(grp["curve"])
        ω[i]    = read(grp["coeffs"]) 
        ẑ[i]    = read(grp["zhat"])
        Δχ²[i]  = read(grp["dchi2"])
        R[i]    = read(grp["R"])
    end
    close(file)
    
    data = (ids = h5keys, curves = χ², coeffs = ω, zhats = ẑ, dchi2s = Δχ² , Rs = R  )
    return data
end


"""
    leggere_cube(path::String; DataExtName::String ="DATA", StatExtName::String = "STAT" )

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
