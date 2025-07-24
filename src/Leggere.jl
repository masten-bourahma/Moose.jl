using FITSIO

"""
    leggere_fits

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


function leggere_fits(path; DataExtName::String= "DATA", StatExtName::String ="STAT")

    files = readdir(path; join =true);
    N     = length(files);

    fλ, σλ, λ = [Vector{Vector{Float32}}(undef, N) for i in 1:3]
    fits_ids= Vector{String}(undef, N)
    
    
    for i in 1:N
        hdul    = FITS(files[i])
        fits_ids[i] = string(read_header(hdul[1])["ID"])
        fλ[i]  = read(hdul[DataExtName])
        σλ[i]  = sqrt.(read(hdul[StatExtName]))

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