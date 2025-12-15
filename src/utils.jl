using FITSIO, HDF5, Dates

"""
    lines 

# Fields
-
-
"""
#struct lines
#function lines()
#    table = FITS("../data/lines_table.fits", "r")
#    names =
#    λrest = 


#    close(table)
#end
#    new()
#end

function create_fits(; kwargs...)
    # File name
    file_name = kwargs[:output_path]
    
    # Create a new FITS file
    output = FITS(file_name, "w")

    # ---- Primary HDU (header only) ----
    write(output, zeros(Float32, 1, 1))  # creates the primary HDU

    # Add run-specific metadata to the header
    header = read_header(output[1])
    
    # Genneral specifis
    header["Moose version"] = "x.x"
    #set_comment!(header, "N1", "length along x")
    header["Date"] = string(Dates.now())
    set_comment!(header, "Date", "Run date")
    header["input_file"] = kwargs[:input_name]
    set_comment!(header, "input_file", "name of the input file")
    header["output_file"] = kwargs[:output_name]
    set_comment!(header, "output_file", "name of this file")

    # axes specifics
    header["N1"] = kwargs[:N₁]
    set_comment!(header, "N1", "length along x")
    header["N2"] = kwargs[:N₂]
    set_comment!(header, "N2", "length along y")
    header["N_chi2"] = kwargs[:Nᵪ]
    set_comment!(header, "N_chi2", "length of chi2 curves")

    # redshift grid specifics
    header["zmin"] = kwargs[:ζmin]
    set_comment!(header, "zmin", "lowest test redshift")
    header["zmax"] = kwargs[:ζmax]
    set_comment!(header, "zmax", "highest test redshift")
    header["dz"] = kwargs[:δζ]
    set_comment!(header, "dz", "step in test redshifts")

    # processing indexes
    header["i"] = 1
    set_comment!(header, "i", "current processing index along the x-axis")
    header["j"] = 1
    set_comment!(header, "j", "current processing index along the y-axis")

    # Description of the HDUs
    header["Description"] = "This file has two extensions, CHI2_SPAXELS & 
                            CHI2_3X3, the former is a run of Moose on individual
                            spaxels. The latter is a run on an average of 9 spaxels
                            in a 3X3 window, slided across the cube "

    # ---- Extension 1: chi2_spaxels ----
    println("here")
    write(output, zeros(Float32, kwargs[:N₁], kwargs[:N₂], kwargs[:Nᵪ]); name="CHI2_SPAXELS")

    # ---- Extension 2: chi2_3x3 ----
    write(output, zeros(Float32, kwargs[:N₁], kwargs[:N₂], kwargs[:Nᵪ] ); name="CHI2_3X3")

    #close(output)
    println("✅ FITS file created: $fname")
    return output
end

function load_fits(; kwargs...)
    file_name = kwargs[:output_path]
    output = FITS(file_name, "r+")

    iₜ = read_key(output[1], "i")
    jₜ = read_key(output[1], "j")

    println("✅ FITS file loaded: $file_name\nResuming processing at indexes: ($iₜ,$jₜ)")
    return output, iₜ, jₜ
end
    
function create_h5(dst::String, basis::Basis, wgrid::Γgrid, metadata)
    
    # Create a new HDF5 file
    N₁, N₂, N₃, λᵣ, δλ = metadata
    h5 = h5open(dst, "w")

    # ---- Metadata ----
    attrs = attributes(h5)
    attrs["Moose_version"] = "x.x"
    attrs["Date"]          = string(Dates.now())
    #attrs["input_file"]    = kwargs[:input_name]
    #attrs["output_file"]   = kwargs[:output_name]

    # Axes specifics
    attrs["N1"]     = N₁
    attrs["N2"]     = N₂
    attrs["N_chi2"] = basis.n

    # Redshift grid specifics
    attrs["zmin"] = wgrid.ζmin
    attrs["zmax"] = wgrid.ζmax
    attrs["dz"]   = wgrid.δζ

    # Processing indices
    attrs["i_t"] = 1
    attrs["j_t"] = 1

    # Description
    attrs["Description"] = """
    Groups signification: CHI2_SPAXELS, CHI2_3X3 & CHI2_3X3GK.
    `CHI2_SPAXELS` corresponds to a run of Moose on individual spaxels.
    `CHI2_3X3`, corresponds to a run with a simple average in a 3X3 window.
    `CHI2_3X3GK`, corresponds to a run with a 3X3 Guassian kernel.
    """

    # ---- Create groups ----
    create_group(h5, "CHI2_SPAXELS")
    create_group(h5, "CHI2_3X3")
    create_group(h5, "CHI2_3X3GK")

    #close(h5)
    println("✅ HDF5 file created: $dst")
    return h5
end


function load_h5(path::String)
    h5 = h5open(path, "r+")
    iₜ = read_attribute(h5, "i_t")
    jₜ = read_attribute(h5, "j_t")

    println("✅ HDF5 file loaded: $path\nResuming at indices: ($iₜ, $jₜ)")
    return h5, iₜ, jₜ
end

#TODO: implement scrub with a guassian kernel 

function scrub(data::Array{Float32, 3}, stat::Array{Float32, 3}, i::Int, j::Int, N₃::Int)
    fᵢⱼ = Array(@view data[i, j, :])
    σᵢⱼ = Array(@view stat[i, j, :])

    #sanitize
    @inbounds @simd for k in 1:N₃
        σᵢⱼ[k] = sqrt(σᵢⱼ[k]) 
    end
    @inbounds @simd for k in 1:N₃
        f = fᵢⱼ[k]
        σ = σᵢⱼ[k]
        if !isfinite(f)
            fᵢⱼ[k] = 0f0
        end
        if !isfinite(σ) || σ == 0f0 || f == 0f0
            σᵢⱼ[k] = 1f6
        end
    end

    if sum(iszero.(fᵢⱼ)) > 250
        return (status=:bad, flux=fᵢⱼ, std=σᵢⱼ)
    end

    return (status=:good, flux=fᵢⱼ, std=σᵢⱼ)
end

function scrub(data::Array{Float32, 3}, stat::Array{Float32, 3}, i::Int, j::Int, N₁::Int, N₂::Int, N₃::Int)

    # subcube indices (3x3 window)
    i₁ = clamp(i-1, 1, N₁)
    i₂ = clamp(i+1, 1, N₁)
    j₁ = clamp(j-1, 1, N₂)
    j₂ = clamp(j+1, 1, N₂)

    # read sub cube
    f, σ² = zeros(Float32, N₃),  zeros(Float32, N₃)
    n = 0
    for u in i₁:i₂
        for v in j₁:j₂
            fᵤᵥ =  Array(@view data[u, v, :])
            if count(isnan, fᵤᵥ) > 250
                continue 
            end
            σᵤᵥ² =  Array(@view stat[u, v, :])
            
            #sanitize
            @inbounds @simd for k in 1:N₃
                if !isfinite(fᵤᵥ[k])
                    fᵤᵥ[k] = 0f0
                end
                if !isfinite(σᵤᵥ²[k]) || σᵤᵥ²[k] == 0f0 || fᵤᵥ[k] == 0f0
                    σᵤᵥ²[k] = 1f12
                end
            end
            f .+= fᵤᵥ
            σ² .+= σᵤᵥ²
            n +=1
        end
    end

    return f ./ n,  sqrt.(σ²) ./ n
end

function scrub(data::Array{Float32, 3}, stat::Array{Float32, 3}, kernel::Matrix{Float32},
               i::Int, j::Int, N₁::Int, N₂::Int, N₃::Int)
    
    # subcube indices (3x3 window)
    i₁ = clamp(i-1, 1, N₁)
    i₂ = clamp(i+1, 1, N₁)
    j₁ = clamp(j-1, 1, N₂)
    j₂ = clamp(j+1, 1, N₂)

    # read subcube
    f, σ² = zeros(Float32, N₃),  zeros(Float32, N₃)

    for u in i₁:i₂, v in j₁:j₂

        kᵤᵥ  = kernel[u-i+2, v-j+2]   # u-i+2, v-j+2 map [-1,0,1] → [1,2,3]
        fᵤᵥ  =  @view data[u, v, :]
        σᵤᵥ² =  @view stat[u, v, :] 

        if count(isnan, fᵤᵥ) > 250
            if u == i && v == j
                return (status=:bad, flux=nothing, std=nothing)
            end
            continue 
        end
        #sanitize
        @inbounds @simd for m in 1:N₃
            if !isfinite(fᵤᵥ[m])
                f[m] += 0f0
            else
                f[m]  += kᵤᵥ     * fᵤᵥ[m]
            end

            if !isfinite(σᵤᵥ²[m]) || σᵤᵥ²[m] == 0f0 || fᵤᵥ[m] == 0f0
                σ²[m] += kᵤᵥ^2 * 1f12
            else
                σ²[m] += kᵤᵥ^2 * σᵤᵥ²[m]
            end
            #f[m]  += kᵤᵥ     * fᵤᵥ[m]
            #σ²[m] += kᵤᵥ^2   * σᵤᵥ²[m]
        end
    end
    return (status=:good, flux=f, std=sqrt.(σ²))
end
"""
    fits_to_h5(fits_path::String, h5_path::Union{String,Nothing};
                    DataExtName::String="DATA",
                    StatExtName::String="STAT")
Description
===========
Reads data and stat HDUs from a fits file and saves them in an h5 file.
Arguments
=========
- **`fits_path ::String`**                : Path to the datacube ".fits" file
- **`h5_path   ::Union{String,Nothing}`** : Path where to save the h5 file output, if nothing a default file will be generated

Keyword Arguments
================= 
- **`DataExtName ::String`**   : Name of the data extension
- **`StatExtName ::String`**   : Name of the stat extension

!!! note
    This function may take some time to finish, especially if the datacube has a large size (~ 13 minutes!)
"""
function fits_to_h5(fits_path::String, h5_path::Union{String,Nothing};
                    DataExtName::String="DATA",
                    StatExtName::String="STAT")
    
    if isnothing(h5_path)
        h5_path = splitext(fits_path)[1] * ".h5"
        @warn("⚠️ Output path not provided, a default path/file was generated @:$h5_path")
    end
    # Open FITS file
    cube = FITS(fits_path, "r")

    data_hdu = cube[DataExtName]
    stat_hdu = cube[StatExtName]

    N₁ = read_key(data_hdu, "NAXIS1")[1] # 1st axis
    N₂ = read_key(data_hdu, "NAXIS2")[1] # 2nd axis
    N₃ = read_key(data_hdu, "NAXIS3")[1] # 3rd axis

    λᵣ   = read_key(data_hdu,"CRVAL3")[1] |> Float32
    δλ   = read_key(data_hdu, "CD3_3")[1]

    #DATA
    println("⚪ Reading '$DataExtName' HDU")
    data = read(data_hdu)
    
    println("✅ FITS '$DataExtName' size: ", size(data))
    println("⚪ Writing '$DataExtName' to HDF5 file")
    
    h5open(h5_path, "w") do h5
        attributes(h5)["N₁"] = N₁
        attributes(h5)["N₂"] = N₂
        attributes(h5)["N₃"] = N₃
        attributes(h5)["λᵣ"] = λᵣ
        attributes(h5)["δλ"] = δλ
        write_dataset(h5, "data", data)
    end
    
    # Free memory from data
    data = nothing  
    GC.gc()     
    println("🧹 Freed memory from '$DataExtName'")
    
    #STAT
    println("⚪ Reading '$StatExtName' HDU")
    stat = read(stat_hdu)
    println("✅ FITS '$StatExtName' size: ", size(stat))
    
    println("⚪ Writing '$StatExtName' to HDF5 file")
    h5open(h5_path, "r+") do h5
        write_dataset(h5, "stat", stat)
    end
    
    # Free memory from stat
    stat = nothing
    GC.gc()
    println("🧹 Freed memory from '$StatExtName'")

    # Close fits file
    close(cube)
    println("✅ HDF5 file saved @:$h5_path")
end

"""
    leggere(h5::HDF5File)

Description
===========
maps to memory the "data" and "stat" datasets from a `.h5` file ... created with fits_to_h5() function

Arguments
=========
- **`h5 ::HDF5File`** : an object of type `HDF5File` with "data" and "stat" datasets

"""
#λ    = collect(Float32, λᵣ .+ (0:N₃-1) .* δλ)
function leggere(path::String, ::Val{:h5})
    h5open(path, "r") do h5 

        N₁ = read_attribute(h5, "N₁")
        N₂ = read_attribute(h5, "N₂")
        N₃ = read_attribute(h5, "N₃")
        λᵣ = read_attribute(h5, "λᵣ")
        δλ = read_attribute(h5, "δλ")

        metadata = (N₁,N₂,N₃,λᵣ,δλ)

        data, stat = h5["data"], h5["stat"]
        
        if HDF5.ismmappable(data)
            data = HDF5.readmmap(data)
        end
        if HDF5.ismmappable(stat)
            stat = HDF5.readmmap(stat)
        end
        return data, stat, metadata
    end
end




function leggere(h5::HDF5.File)

    data, stat = h5["data"], h5["stat"]
    if HDF5.ismmappable(data)
        data = HDF5.readmmap(data)
    end

    if HDF5.ismmappable(stat)
        stat = HDF5.readmmap(stat)
    end
    return data, stat
end

function leggere(path::String; DataExtName::String="DATA")
    FITS(path, "r") do cube
        N₁ = read_key(cube[DataExtName], "NAXIS1")[1] # 1st axis
        N₂ = read_key(cube[DataExtName], "NAXIS2")[1] # 2nd axis
        N₃ = read_key(cube[DataExtName], "NAXIS3")[1] # 3rd axis

        λᵣ   = read_key(cube[DataExtName],"CRVAL3")[1] |> Float32
        δλ   = read_key(cube[DataExtName], "CD3_3")[1]
        λ    = collect(Float32, λᵣ .+ (0:N₃-1) .* δλ)
        return N₁,N₂,N₃,λᵣ,λ
    end
end

function awaken(dst, basis, wgrid, metadata)
    output, iₜ, jₜ = nothing, 1, 1
    if isfile(dst)
        output, iₜ, jₜ = load_h5(dst)
    else
        output = create_h5(dst, basis, wgrid, metadata)
    end
    return output, iₜ, jₜ
end