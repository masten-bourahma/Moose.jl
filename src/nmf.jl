module nmf

export NMF, nearly!, sequential, rankify
using HDF5, LinearAlgebra, Base.Threads

"""
# summary    

NMF: A struct to hold X, W & H matrices and algorithmic settings for nearly-NMF algorithm.

# Fields
- `X::Matrix{Float32}`: data matrix (n by l).
- `W::Matrix{Float32}`: weights matrix (n by k).
- `H::Matrix{Float32}`: basis vectors matrix (k by l).
- `M::Matrix{Float32}`: missing values matrix (k by l), X[j,j] is missing => M[i,j] = true.
- `n::Int`: number of spectra in X
- `l::Int`: number of spectral segments in X
- `k::Int`: NMF decomposition rank 
- `maxiters::Int`: max iterations for the solver
- `tol::Float32`: stopping criterion, if improvement < tol and the maximum iterations is not reached, solver stops
- `verbose::Bool`: if true, algorithmic settings and the proportion of negative/missing values in X are printed.
"""

struct NMF
    
    X::Matrix{Float32}
    W::Matrix{Float32}
    H::Matrix{Float32}
    V::Matrix{Float32}
    M::Matrix{Bool}
    n::Int
    l::Int
    k::Int
    maxiters::Int
    tol::Float32

    function NMF(X::Matrix{Float32}; W=nothing, H=nothing, V=nothing, M=nothing, k::Int=5, maxiters::Int=5000, tol::Float32=1f-8, verbose::Bool=true)
        #X    = X isa Matrix ? Float32.(X) :  
        n, l = size(X)

        if W === nothing
            W = rand(Float32, n, k)
        else
            W = Float32.(W) 
            @assert size(W) == (n, k) "W has wrong shape, should be ($(n),$(k))"
        end

        if H === nothing
            H = rand(Float32, k, l)
        else
            H = Float32.(H)
            @assert size(H) == (k, l) "H has wrong shape, should be ($(k),$(l))"
        end

        if M === nothing
            M = trues(n, l) 
        else
            M = Bool.(M)
            @assert size(M) == (n, l) "M has wrong shape, should be ($(n),$(l))"
        end
        
        if V === nothing
            V = ones(Float32, n, l) .* M
        else
            V=  typeof(V) == Float32 ? V .* M : Float32.(V) .* M
            @assert size(V) == (n, l) "V has wrong shape, should be ($(n),$(l))"
        end
        if verbose
            println("|   NMF structure successfully initialized! :) \n
            >>> nmf rank set to $(k) \n
            >>> maximum number of iterations set to $(maxiters) \n
            >>> tolerance set to $(tol) \n
            >>> X: $( round(count(X .== 0) * 100 / (n*l), digits=2))% --> missing values \n
            >>> X: $( round(count(X .< 0) *100 / (n*l), digits=2))% --> negative values")

        end
        new(X, W, H, V, M, n,l, k, maxiters, tol);
    end
end


"""
    nearly!(nmf::NMF)

mutating function!, performs nearly NMF algorithm on nmf.X, and updates nmf.W & nmf.H in-place.
Reference: Dylan Green and Stephen Bailey 2024, https://arxiv.org/abs/2311.04855 

# Arguments
- `nmf`: NMF struct.

# Returns
- `χ2`: frobenius norm at last iteration 
- `iter`: exit (last) iteration
"""

function nearly!(nmf::NMF)
    iter    = 0
    nan_eps = 1f-6
    inf_eps = 1f-6 

    VX         = nmf.V .* nmf.X
    χ2         = frobenius(nmf)
    oldchi2    = Inf32 
    
    while iter < nmf.maxiters && ((oldchi2 - χ2) / oldchi2 > nmf.tol)
        
        oldchi2 = χ2
        
        Wt = transpose(nmf.W)
        WtVXp, WtVXn  = pn_split(Wt * VX)
        nmf.H .= nmf.H .* (WtVXp ./ (Wt * (nmf.V .* (nmf.W*nmf.H)) .+ WtVXn))
        nmf.H .= clamp_nan_inf!(nmf.H, nan_eps, inf_eps)

        Ht =  transpose(nmf.H)
        VXHtp, VXHtn  = pn_split(VX * Ht)
        VWHHt = nmf.V .*(nmf.W * nmf.H)
        nmf.W .= nmf.W .* (VXHtp ./ (VWHHt * Ht .+ VXHtn)) 
        nmf.W .= clamp_nan_inf!(nmf.W, nan_eps, inf_eps)
        
        χ2 = frobenius(nmf)
        
        if !isfinite(χ2)
            error("nearly NMF construction failed, χ2 is not finite")
        end
        
        iter += 1
        println(iter)
    end
    println("Done! after $(iter) iterations; χ2 = $(χ2)")
    return χ2, iter
end



function sequential(flux, ivar, mask; ktarget::Int, variant = "nearly")
    n,l   = size(flux)
    nmf   = nothing
    chi2_ = []

    for k in 1:ktarget

        Wi = rand(Float32, n, k)
        Hi = rand(Float32, k, l) 
        
        if k>1
            Wi[:, 1 : Int(k-1)] .= nmf.W
            Hi[1: Int(k-1),:]   .= nmf.H
        end

        nmf = NMF(flux,V=ivar, H=Hi, W=Wi, M=mask, k = k, variant=variant, maxiters= 1300, tol = 1e-3, verbose=false);
        χ2, fiter = nearly!(nmf)
        push!(chi2_, χ2)
        println("rank $(k); χ2 = $(χ2); after $(fiter) iterations")

        if k == ktarget
            nmf = NMF(flux,V=ivar, H=nmf.H, W=nmf.W, M=mask, k = k, variant=variant, maxiters= 1300, tol = 1e-3, verbose=false);
            χ2, fiter = nearly!(nmf)
            push!(chi2_, χ2)
            println("rank $(k) ++; χ2 = $(χ2); after $(fiter) iterations")
        end
        
        if isfile("../results/Hv2.h5")
            h5open("../results/Hv2.h5", "r+") do file
                write(file, "rank$(k)", nmf.H)
            end
        else
            h5open("../results/Hv2.h5", "w") do file
                write(file, "rank$(k)", nmf.H)
            end
        end
    end
    return nmf
end



function rankify(flux, ivar, mask; kmax::Int=25, max_iters::Int=10000)
    n,l = size(flux)
    nmf = nothing
    mdl_container = []

    for k in 1:kmax
        Wi = rand(Float32, n, k)
        Hi = rand(Float32, k, l)
        
        if k>1
            Wi[:, 1 : Int(k-1)] = nmf.W
            Hi[1: Int(k-1),:]   = nmf.H
        end

        nmf       = NMF(flux,V=ivar, H= nothing, W=nothing, M=nothing, k= k,  maxiters= 10000, tol = 1e-8, verbose=false)
        χ2, fiter = nearly!(nmf)
        
        mdl       = calc_MDL(nmf.X, nmf.W, nmf.H, mask)
        total_mdl = sum(mdl)
        
        println("rank $(k); total MDL: ", total_mdl)
        push!(mdl_container, mdl)

        if isfile("./preliminary_results/mdl.h5")
            h5open("./preliminary_results/mdl.h5", "r+") do file
                write(file, "rank_$(k)", mdl)
            end
        else
            h5open("./preliminary_results/mdl.h5", "w") do file
                write(file, "rank_$(k)", mdl)
            end
        end
        
    end
    return mdl_container
end


function calc_MDL(data::Matrix, mat_w::Matrix, mat_h::Matrix,mat_m, d_D=1e-8, threshold_search_num=10)
    threshold = 0.
    min_res = Inf
    res = []
    
    while threshold < d_D
        LW, LW0 = __calc_factorized_MDL(mat_w, threshold, d_D)
        LH, LH0 = __calc_factorized_MDL(mat_h, threshold, d_D)
        LE      = __calc_error_MDL((data - mat_w * mat_h),mat_m, d_D)

        if LW < 0 || LH < 0
            break
        end
        
        if min_res > LW + LW0 + LH + LH0 + LE
            res = [LW, LW0, LH, LH0, LE]
            min_res = sum(res)
        end
        threshold += d_D / threshold_search_num
        #println(threshold)
    end

    if isempty(res)
        error("Result is empty.")
    end

    return res
end

function __calc_factorized_MDL(mat::Matrix, threshold::Float64, d_D::Float64)
    if count(x -> x > threshold, mat) == 0
        return -1.0, -1.0
    end

    values = mat[mat .> threshold]  # Filter values above threshold
    params = fit(Gamma, values)
    a_hat, scale_hat = params.α, params.θ  # Corrected extraction of shape and scale

    P = pdf.(Gamma(a_hat, scale_hat), values) * d_D
    P = clamp.(P, eps(), 1.0)  # Avoid zero probabilities
    L = -sum(log2.(P))
    if L < 0
        error("Negative MDL value encountered.")
    end

    n0 = count(x -> x ≤ threshold, mat)
    if n0 != 0
        nt = length(mat)
        L0 = max(-n0 * log2(n0 / nt) - (nt - n0) * log2((nt - n0) / nt), 0.0)
    else
        L0 = 0.0
    end

    return L, L0
end

function __calc_error_MDL(mat::Matrix,mat_m, d_D::Float64)
    # Flatten the matrix to a vector on the GPU
    values = vec(mat[mat_m])

    # Calculate mean and standard deviation directly on the GPU
    loc_hat = mean(values)
    scale_hat = std(values)

    # Compute probabilities using the Normal distribution, directly on the GPU
    P = pdf.(Normal(loc_hat, scale_hat), values) * d_D

    # Clamp probabilities to avoid zero values
    P = clamp.(P, eps(), 1.0)

    # Calculate the log-likelihood (sum of logs), again optimized for GPU
    L = -sum(log2.(P))

    return L
end

"""
    frobenius(nmf::NMF)

frobenius norm: chi^2 = || X - WH||^{2}_{V}

# Arguments
- `nmf`: NMF struct.

# Returns
- `χ2`: frobenius norm
"""
function frobenius(nmf::NMF)
    χ2   = sum(nmf.V .* ((nmf.X .- (nmf.W * nmf.H)) .^ 2)) / count(nmf.V .> 0)
    return χ2
end

"""
    pn_split(A::Matrix{Float32})

Splits matrix A, into two matrices: A+ & A-:
1. Ap holds positive elements in A with zeros elsewhere.
2. An holds negative elements in A (turned positive) with zeros elsewhere

# Arguments
- `A`: input matrix.

# Returns
- `Ap`: positive elements matrix 
- `An`: negative elements matrix 
"""
function pn_split(A::Matrix{Float32})
    return 0.5f0 * (A .+ abs.(A)),  0.5f0 * (A .- abs.(A))
end

"""
"""
function clamp_nan_inf!(A::Matrix{Float32}, nan_eps::Float32, inf_eps::Float32)
    @threads for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            if isnan(A[i, j]) || issubnormal(A[i, j])
                A[i, j] = nan_eps
            elseif isinf(A[i, j])
                A[i, j] = inf_eps
            end
        end
    end
    return A
end



end


    
