using Base.Threads, Dates
using ProgressMeter: @showprogress

"""
    interpolate( basis::Basis, fλ::Vector{Float32}, σλ::Vector{Float32}, λ::Vector{Float32}; extrpfλ = 0.0f0, extrpσλ = 1f6)

Description
============
This function takes three vectors: flux (fλ), standard deviations (σλ) and observed wavelengths (λ), it interpolates 
the flux and std vectors to the rest wavelength grid assuming redshift 0. (shifting is captured in gram matrices stored in basis Struct)
    
Arguments
==========
- **`basis ::Basis`**             : Basis Struct
- **`fλ    ::Vector{Float32}`**   : Observed flux densities
- **`σλ    ::Vector{Float32}`**   : Flux associated standard deviations
- **`λ     ::Vector{Float32}`**   : Observed wavelengths

- **`extrpfλ ::Float32 = 0f0`**   : Extrapolation value for fluxes
- **`extrpσλ ::Float32 = 1f6`**   : Extrapolation value for standard deviations
None

returns
=======
- **`intrpfλ    ::Vector{Float32}`**   : Interpolated flux densities
- **`intrpσλ    ::Vector{Float32}`**   : Interpolated standard deviations

Details
=======

"""
function interpolate( basis::Basis, fλ::Vector{Float32}, σλ::Vector{Float32}, λ::Vector{Float32};
                      extrpfλ::Float32 = 0.0f0, extrpσλ::Float32 = 1f6)
    
    Λ  = basis.λinterp
    l  = basis.l

    intrpfλ = zeros(Float32, l)
    intrpσλ = zeros(Float32, l)

    logλ = log10.(λ)
    lwbn = logλ[1]
    upbn = logλ[end]

    @inbounds @threads for j in 1:l 
        
        λj = Λ[1, j]
            
        # Left or right extrapolation
        if λj < lwbn || λj > upbn
            intrpfλ[j] = extrpfλ
            intrpσλ[j] = extrpσλ
            continue
        end

        # Binary search for interpolation index
        idx = searchsortedfirst(logλ, λj)
        idx = max(1, min(idx - 1, l-1))  # Clamp to valid range

        # Direct memory access
        λ1 = logλ[idx]
        λ2 = logλ[idx + 1]
            
        inv_denom = 1 / (λ2 - λ1)
        p = (λj - λ1) * inv_denom
            
        fλ1, fλ2 = fλ[idx], fλ[idx + 1]
        σλ1, σλ2 = σλ[idx], σλ[idx + 1] 
        
        # Fused multiply-add operations
        intrpfλ[j] = muladd(p, fλ2 - fλ1, fλ1)
        intrpσλ[j] = muladd(p, σλ2 - σλ1, σλ1)
    end
    
    return intrpfλ, intrpσλ
end

"""
    threaded_fnnls(basis::Basis, fλ::Vector{Float32}, σλ::Vector{Float32})

Description
===========
This function executes a multi-threaded calculation of the chi-square curve. For each redshift (each gram matrices Hi & HHti
in basis Struct), it reconstructs an input vector fλ with basis vectors using Fast Non-Negative Least Squares (FNNLS),
then quantifies the reconstruction error using a χ2 metric

#put definition of chi2 here        grid  = Γgrid()


Arguments
=========
- **`basis ::Basis`**            : Basis Struct
- **`fλ    ::Vector{Float32}`**  : Input vector (flux densities)
- **`σλ    ::Vector{Float32}`**  : Error vector (standard deviations)

Returns
=======
- **`χ2    ::Vector{Float32}`**  : χ2 curve, χ2 metric at each test redshift

Example
=======
```julia
# 1. Initialize the Basis Struct
basis = Basis()

# 2. Interpolate flux densities fλ and σλ to rest frame grid
intrpfλ, intrpσλ = interpolate(b, fλ .* λ, σλ .* λ, λ)

# 3. Run threaded_nnls to get the chi2 curve
χ2 = threaded_fnnls(b, intrpfλ, intrpσλ)
```

Tips
====
To set the number of threads in julia, add this line to your linux/macos bash
```bash
export JULIA_NUM_THREADS=4
```
"""
function threaded_fnnls(basis::Basis, fλ::Vector{Float32}, σλ::Vector{Float32})
    
    n, l  = basis.n, basis.l
    
    χ² = Vector{Float32}(undef, n)
    Ω  = Vector{Vector{Float32}}(undef, n)

    f̂λ_threads     = [Vector{Float32}(undef, l) for _ in 1:Threads.nthreads()]


    @threads for i in 1:n

        thread_id = Threads.threadid()
        f̂λ        = f̂λ_threads[thread_id]
        Hi        =  basis.Hi[i]

        @inbounds ω = fnnls(basis.HHti[i],  Hi * fλ)
        mul!(f̂λ, transpose(Hi), vec(ω))

        total_sum = 0.0f0
        @simd for j in 1:l
            @inbounds total_sum += ((f̂λ[j] - fλ[j])/σλ[j])^2
        end

        χ²[i] = total_sum / l
        Ω[i]  = ω

    end

    return χ², Ω
end

"""
    χfits(wgrid::Γgrid, basis:Basis, data; output_path::String = nothing)

Description
===========
Runs a threaded FNNLS on a set of spectra loaded into a data Struct apriori and stored in data. Resulting chi2 curves are
saved into an h5 file, whose path and name are specified by setting the keyword argument 'output_path' 

Arguments
=========
- **`wgrid       ::Γgrid`**     : Γgrid Struct
- **`basis       ::Basis`**     : Basis Struct
- **`data        ::Struct`**    : data Struct with four fields: flux, sdev, wave, idsdefault
- **`output_path ::String`**    : Path where the chi2 file will be saved 

Returns
=======
nothing

Methods
=======
- `χfits(data; output_path::String = nothing)`: `wgrid` and `basis` are instantiated within the method.
- `χfits(fits_path::String, DataExtName::Union{String, Int}, StatExtName::Union{String, Int}; output_path::String = nothing)`: `wgrid` and `basis` are instantiated within the method,
the data is loaded within the function using the provided `fits_path` and the extensions: `DataExtName`, `StatExtName`.
- `χfits(wgrid::Γgrid, basis::Basis, fits_path::String, DataExtName::Union{String, Int}, StatExtName::Union{String, Int}; output_path::String = nothing)`: the data  is loaded within
the function using the provided `fits_path` and the extensions: `DataExtName`, `StatExtName`.
Example
=======
```julia
# 1. Initialize Basis Struct
wgrid = Γgrid()
basis = Basis()

# 2. read data from fits files
data = leggere_fits("../data/spectra_sample")

#3. Specify output file path and run χloop
my_chi2file = "../output/chi2_files/chi2_moose.h5"
χfits(wgrid, basis, data; output_path = my_chi2file)
```
"""
function χfits(wgrid::Γgrid, basis::Basis, data; output_path::String = nothing)

    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path,  a file with the same name already exists!"
    end
    
    n = length(data.flux)

    @showprogress for i in 1:n
        
        fλ = data.flux[i]
        σλ = data.sdev[i]         
        λ  = data.awave[i]

        intrpfλ, intrpσλ = interpolate(basis, fλ .* λ, σλ .* λ , λ)
        #χ² curve
        χ², Ω = threaded_fnnls(basis, intrpfλ, intrpσλ)
        # predicted redshift ẑ  

        ẑ     = wgrid.ζ[argmin(χ²)]
        # decomposition coeffs 
        ω     = Ω[argmin(χ²)]
        # significance score \Delta\chi2
        Δχ²_  = Δχ²(χ²)
        # Robustness metric R
        R_    = R(χ²)
        
        dset = Dict( "chi2"   => χ², "omega_hat"  => ω, "z_hat"  => ẑ, "dchi2"  => Δχ²_, "R"      => R_ )

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

function χfits(data; output_path::String = nothing)
    wgrid = Γgrid()
    basis = Basis()

    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path,  a file with the same name already exists!"
    end
    
    n = length(data.flux)

    @showprogress for i in 1:n

        fλ = data.flux[i]
        σλ = data.sdev[i]         
        λ  = data.awave[i]

        intrpfλ, intrpσλ = interpolate(basis, fλ .* λ, σλ .* λ , λ)

        #χ² curve
        χ², Ω = threaded_fnnls(basis, intrpfλ, intrpσλ)
        # predicted redshift ẑ  
        ẑ     = wgrid.ζ[argmin(χ²)]

        # decomposition coeffs 
        ω     = Ω[argmin(χ²)]
        # significance score \Delta\chi2
        Δχ²_  = Δχ²(χ²)
        # Robustness metric R
        R_    = R(χ²)
        
        dset = Dict( "chi2"   => χ², "omega_hat"  => ω, "z_hat"  => ẑ, "dchi2"  => Δχ²_, "R"      => R_ )

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

function χfits(fits_path::String, DataExtName::Union{String, Int}, StatExtName::Union{String, Int}; output_path::String = nothing)
    
    wgrid = Γgrid()
    basis = Basis()
    data = leggere_fits(fits_path; DataExtName = DataExtName, StatExtName = StatExtName)
    
    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path,  a file with the same name already exists!"
    end
    
    n = length(data.flux)

    @showprogress for i in 1:n

        fλ = data.flux[i]
        σλ = data.sdev[i]         
        λ  = data.awave[i]

        intrpfλ, intrpσλ = interpolate(basis, fλ .* λ, σλ .* λ , λ)

        #χ² curve
        χ², Ω = threaded_fnnls(basis, intrpfλ, intrpσλ)
        # predicted redshift ẑ  
        ẑ     = wgrid.ζ[argmin(χ²)]

        # decomposition coeffs 
        ω     = Ω[argmin(χ²)]
        # significance score \Delta\chi2
        Δχ²_  = Δχ²(χ²)
        # Robustness metric R
        R_    = R(χ²)

        dset = Dict( "chi2"   => χ², "omega_hat"  => ω, "z_hat"  => ẑ, "dchi2"  => Δχ²_, "R"      => R_ )

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

function χfits(wgrid::Γgrid, basis::Basis, fits_path::String, DataExtName::Union{String, Int}, StatExtName::Union{String, Int}; output_path::String = nothing)
    
    data = leggere_fits(fits_path; DataExtName = DataExtName, StatExtName = StatExtName)

    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path,  a file with the same name already exists!"
    end
    
    n = length(data.flux)

    @showprogress for i in 1:n

        fλ = data.flux[i]
        σλ = data.sdev[i]         
        λ  = data.awave[i]

        intrpfλ, intrpσλ = interpolate(basis, fλ .* λ, σλ .* λ , λ)

        #χ² curve
        χ², Ω = threaded_fnnls(basis, intrpfλ, intrpσλ)
        # predicted redshift ẑ  
        ẑ     = wgrid.ζ[argmin(χ²)]

        # decomposition coeffs 
        ω     = Ω[argmin(χ²)]
        # significance score \Delta\chi2
        Δχ²_  = Δχ²(χ²)
        # Robustness metric R
        R_    = R(χ²)

        dset = Dict( "chi2"   => χ², "omega_hat"  => ω, "z_hat"  => ẑ, "dchi2"  => Δχ²_, "R"      => R_ )

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
    χcube(path::String, save_path::Union{String, Nothing}; kwargs...)

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

Author(s)
=========
B.Masten

See also `χfits`
"""
function χcube(path::String, save_path::Union{String, Nothing}; kwargs...)
    
    if isnothing(save_path)
        save_path = joinpath( @__DIR__, "../output/chi2_files/$(splitext(basename(path))[1])_chi2.h5")
    end
    #h5key
    h5key = splitext(basename(path))[1]

    #instantiate the basis Struct
    basis = Basis()
    
    # get cube data
    data   = leggere_cube(path; kwargs...)
    λ      = data.awave
    nx, ny = size(data.flux)

    χ2     = Array{Float32}(undef,(nx,ny,basis.n)) .* NaN32

    for i in 1:nx
        for j in 1:ny
            if any(.!iszero.(data.flux[i,j,:]))
                intrpfλ, intrpσλ = interpolate(basis, data.flux[i,j,:] .* λ, data.sdev[i,j,:] .* λ, λ)
                χ2[i,j,:]       .= threaded_fnnls(basis, intrpfλ, intrpσλ)
            else
                χ2[i,j,:] .= NaN32
            end
        end

        if i % 5 == 0
            println("row $(i) completed, ...")
            h5open(save_path, "w") do file
                write(file, h5key, χ2)
            end
        end
    end

    h5open(save_path, "w") do file
        write(file, h5key, χ2)
    end
end
