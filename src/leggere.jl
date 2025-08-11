using FITSIO, HDF5, LinearAlgebra

"""
    leggere_fits(path::String; DataExtName::String= "DATA", StatExtName::String ="STAT")

    Description
===========
This function reads fluxes, standard deviations, wavelengths and ids from a group of fits file in 'path',
and outputs a Struct with four fields (flux, sdev, awave, ids). The name of the data and stat extension can 
be specied by setting keywords 'DataExtName' and 'StatExtName'

Arguments
=========
- **`path ::String`**                   : Path to the fits files folder 
- **`DataExtName ::String = "DATA"`**   : Name of the data extension
- **`StatExtName ::String = "STAT"`**   : Name of the stat extension

returns
=======
- **`data ::Struct`** : Struct with four fields (flux, sdev, awave, ids)

"""


function leggere_fits(path::String; DataExtName::String= "DATA", StatExtName::String ="STAT")

    files = readdir(path; join =true)
    N     = length(files)

    fλ, σλ, λ = [Vector{Vector{Float32}}(undef, N) for i in 1:3]
    fits_ids= Vector{String}(undef, N)
    
    
    for i in 1:N
        hdul    = FITS(files[i])
        fits_ids[i] = string(read_header(hdul[1])["ID"])
        fλ[i]  = read(hdul[DataExtName])
        σλ[i]  = sqrt.(read(hdul[StatExtName]))

        # clean the input
        replace!(fλ[i], NaN => 0) 
        replace!(σλ[i], NaN => 1f6, Inf => 1f6) 
        σλ[i][fλ[i] .== 0] .= 1f6 
        # Read header information
        λref = read_header(hdul[DataExtName])["CRVAL1"]
        L    = read_header(hdul[DataExtName])["NAXIS1"]
        δλ   = read_header(hdul[DataExtName])["CDELT1"]
        # Calculate wavelength array
        λ[i] = λref .+ (0:L-1) .* δλ
        close(hdul);
    end

    return ( flux         = fλ,
             sdev         = σλ,
             awave        = λ,
             ids= fits_ids)
end

"""
    leggere_chifile(path::String)

Description
===========
This function reads fluxes, standard deviations, wavelengths and ids from a group of fits file in 'path',
and outputs a Struct with four fields (flux, sdev, awave, ids). The name of the data and stat extension can 
be specied by setting keywords 'DataExtName' and 'StatExtName'

Arguments
=========
- **`path ::String`**                   : Path to the fits files folder 

returns
=======
- **`data ::Struct`** : Struct with four fields (flux, sdev, awave, ids)

"""
function leggere_chifile(path::String)

    if !isfile(path)
        error("file does not exist!")
    end

    file = h5open(path,"r")
    h5keys          = keys(file)
    n               = length(h5keys)
    χ2  = Vector{Vector{Float32}}(undef,n)

    for (i,key) in enumerate(h5keys)
        χ2[i] = read(file, key)
    end
    close(file)
    
    return h5keys, χ2
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
function leggere_cube(path::String; DataExtName::String ="DATA", StatExtName::String = "STAT" )
    
    #check file exists
    if !isfile(path)
        error("file does not exist!")
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
