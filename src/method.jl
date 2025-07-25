using Base.Threads, Dates

"""
    interpolate()

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

function interpolate( basis, fλ::Vector{Float32}, σλ::Vector{Float32}, λ::Vector{Float32};
                             extrpfλ = 0.0f0, extrpσλ = 1f6)
    
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
    threaded_nnls

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
****
- **`χ2    ::Vector{Float32}`**  : χ2 curve, χ2 metric at each test redshift

Example
=======
# 1. Initialize the Basis Struct
basis = Basis()

# 2. Interpolate flux densities fλ and σλ to rest frame grid
intrpfλ, intrpσλ = interpolate(b, fλ .* λ, σλ .* λ, λ)

# 3. Run threaded_nnls to get the chi2 curve
χ2 = threaded_nnls(b, intrpfλ, intrpσλ)


Tips
====
To set the number of threads in julia, add this line to your linux/macos bash
''' bash
export JULIA_NUM_THREADS=4
'''


"""
function threaded_nnls(basis, fλ::Vector{Float32}, σλ::Vector{Float32})
    
    n, l  = basis.n, basis.l
    
    χ2= Vector{Float32}(undef, n)
    f̂λ_threads     = [Vector{Float32}(undef, l) for _ in 1:Threads.nthreads()]


    @threads for i in 1:n

        thread_id = Threads.threadid()
        f̂λ        = f̂λ_threads[thread_id]
        Hi        =  basis.Hi[i]

        @inbounds θ = fnnls(basis.HHti[i],  Hi * fλ)
        mul!(f̂λ, transpose(Hi), vec(θ))

        total_sum = 0.0f0

        @simd for j in 1:l
            @inbounds total_sum += ((f̂λ[j] - fλ[j])/σλ[j])^2
        end

        χ2[i] =  total_sum / l
    end

    return χ2
end

"""
    χloop

Description
===========
This function runs a threaded_nnls on a set of spectra contained in a data Struct. Resulting chi2 curves are
saved in an h5 file, whose path and name are specified by setting the keyword argument 'output_path' 

Arguments
=========
- **`basis ::Basis`**            : Basis Struct
- **`data  ::Struct`**           : data Struct with four fields: flux, sdev, wave, ids
- **`output_path  ::String`**    : Path where the chi2 file will be saved 

Returns
=======
nothing

Example
=======
# 1. Initialize Basis Struct
basis = Basis()

# 2. read data from fits files
data = leggere_fits("../data/spectra_sample")

#3. Specify output file path and run χloop
my_chi2file = "../output/chi2_files/chi2_moose.h5
χloop(basis, data; output_path = my_chi2file)

"""

function χloop(basis, data; output_path = nothing)

    if isnothing(output_path)
        output_path = joinpath( @__DIR__, "../output/results/chi2_files/chi2_$(now).h5")
    end

    if isfile(output_path)
        @warn "Output path,  a file with the same name already exists!"
    end
    
    n = length(data.flux)

    for i in 1:n

        fλ = data.flux[i]
        σλ = data.sdev[i]         
        λ  = data.awave[i]

        intrpfλ, intrpσλ = interpolate(basis, fλ .* λ, σλ .* λ , λ)
        χ2               = threaded_nnls(basis, intrpfλ, intrpσλ)

        if isfile(output_path)
            h5open(output_path, "r+") do file
                write(file,data.ids[i] , χ2)
            end
        else
            h5open(output_path, "w") do file
                write(file, data.ids[i], χ2)
            end
        end
    end
end


"""
    

"""