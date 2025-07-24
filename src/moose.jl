using Base.Threads

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
    
    Λ  = basis.Λ
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

Arguments
=========
- **`basis ::Basis`**                  : Basis Struct
- **`intrpfλ    ::Vector{Float32}`**   : Interpolated flux densities
- **`intrpσλ    ::Vector{Float32}`**   : Interpolated standard deviations

Returns
=======
****

Example
=======

"""
function threaded_nnls(basis, intrpfλ::Vector{Float32}, intrpσλ::Vector{Float32})
    
    n, l  = basis.n, basis.l
    
    χ2   = Vector{Float32}(undef, n)
    xhat_threads     = [Vector{Float32}(undef, l) for _ in 1:Threads.nthreads()]

    
    @threads for i in 1:n

        thread_id = Threads.threadid()
        xhat = xhat_threads[thread_id]

        Hi   =  basis.Hi[i]
        @inbounds θ = fnnls(basis.HHti[i],  Hi * intrpfλ)
        #@inbounds θ = nonneg_lsq(basis.HHti[i], Hi * intrpfλ; alg=:nnls,gram=true) # NNLS
        mul!(xhat, transpose(Hi), vec(θ))

        total_sum = 0.0f0

        @simd for j in 1:l
            @inbounds total_sum += ((xhat[j] - intrpfλ[j])/intrpσλ[j])^2
        end

        χ2[i] =  total_sum / l
    end
    return χ2
end