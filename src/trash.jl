"""
    NMF(X::Matrix{Float32}; W=nothing, H=nothing, V=nothing, M=nothing, k::Int=5, maxiters::Int=5000, tol::Float32=1f-8, verbose::Bool=true)

Description
===========
NMF is a type, holds X, W and H matrices and other algorithmic settings for the nearly-NMF algorithm.

Arguments
=========
- **`X ::Matrix{Real}`** : A required positional argument. Data matrix (n by l)
- **`W ::Matrix{Float32}`** : An optional keyword argument. Weights matrix (n by k), if nothing W is randomly initialized. Defaults to `nothing`
- **`H ::Matrix{Float32}`** : An optional keyword argument. Basis vectors matrix (k by l), if nothing H is randomly initialized. Defaults to `nothing`
- **`M ::Matrix{Bool}`**    : An optional keyword argument. Mask matrix (n by l), if X[j,j] is missing then M[i,j] = true. if nothing M is constructed from X. Defaults to `nothing`
- **`k ::Int`**             : An optional keyword argument. NMF decomposition rank (or number of basis vectors in H). Defaults to `10`
- **`maxiters ::Int`**      : An optional keyword argument. Maximum number iterations for the solver. Defaults to `5000`
- **`tol ::Float32`**       : An optional keyword argument. Stopping criterion, if improvement < tol and the maximum iterations is not reached, solver stops. Defaults to `1f-8`
- **`verbose ::Bool`**      : An optional keyword argument. If true, algorithmic settings and the proportion of negative/missing values in X are printed. Defaults to `true`

Author(s)
=========
B. masten

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

    function NMF(X::Matrix{Real}; W=nothing, H=nothing, V=nothing, M=nothing, k::Int=10, maxiters::Int=5000, tol::Float32=1f-8, verbose::Bool=true)
        X    = X isa Matrix ? Float32.(X) :  
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
            println("NMF structure successfully initialized! :) \n
            >>> nmf rank set to $(k) \n
            >>> maximum number of iterations set to $(maxiters) \n
            >>> tolerance set to $(tol) \n
            >>> X: $( round(count(X .== 0) * 100 / (n*l), digits=2))% --> missing values \n
            >>> X: $( round(count(X .< 0) *100 / (n*l), digits=2))% --> negative values")

        end
        new(X, W, H, V, M, n,l, k, maxiters, tol)
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
- `iter`: exit/last iteration
"""

function nearly!(nmf::NMF)
    iter    = 1
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


using FITSIO

infile  = "/media/masten/masten/DATACUBE_A370-DF_v1.0.eff_dc.fits"
outfile = "../data/cubes/DATACUBE_test.fits"
# Open input FITS
f_in = FITS(infile, "r")

# Open output FITS
f_out = FITS(outfile, "w")

ExtNames = ["", "DATA", "STAT"]
# Loop over all HDUs
for hdu_index in 1:3
    hdu_in = f_in[hdu_index]
    hdr = read_header(hdu_in)

    
    if hdu_index == 1
        data = Matrix{Float32}(undef,(1,1))
        write(f_out, data; header= read_header(hdu_in))
        
    else
        subcube = read(hdu_in, 210:213, 210:213, :) 
        write(f_out, subcube; header= hdr, name=ExtNames[hdu_index] )
    end
end

close(f_in)
close(f_out)