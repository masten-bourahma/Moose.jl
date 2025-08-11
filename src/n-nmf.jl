using HDF5, Base.Threads

"""
    nNMF(X::Matrix{Float32}; W=nothing, H=nothing, V=nothing, M=nothing, k::Int=5, maxiters::Int=5000, tol::Float32=1f-8, verbose::Bool=true)

Description
===========
nNMF is a type, holds X, W, H and V matrices and other algorithmic settings for the nearly-NMF algorithm solver.

Arguments
=========
- **`X ::Matrix{Float32}`** : Data matrix (n by l)
- **`W ::Matrix{Float32}`** : Weights matrix (n by k), if nothing W is randomly initialized. Defaults to `nothing`
- **`H ::Matrix{Float32}`** : Basis vectors matrix (k by l), if nothing H is randomly initialized. Defaults to `nothing`
- **`V ::Matrix{Float32}`** : Inverse variance matrix (n by l), if nothing V is randomly initialized with ones. Defaults to `nothing`
- **`M ::Matrix{Float32}`** : Mask matrix (n by l), if X[j,j] is missing then M[i,j] = true. if nothing M is constructed from X. Defaults to `nothing`
- **`k ::Int`**          : NMF decomposition rank (or number of basis vectors in H). Defaults to `10`
- **`maxiters ::Int`**   : Maximum number iterations for the solver. Defaults to `5000`
- **`tol ::Float32`**    : Stopping criterion, if improvement < tol and the maximum iterations is not reached, solver stops. Defaults to `1f-8`
- **`verbose ::Bool`**   : If true, algorithmic settings and the proportion of negative/missing values in X are printed. Defaults to `true`

Details
=======
Element-type of all matrices (X,W,H and V) should be `Float32`, this is to reduce memory storage and speed calculations.

Author(s)
=========
B. masten

"""
struct nNMF
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

    function nNMF(X::Matrix{Float32}; W=nothing, H=nothing, V=nothing, M=nothing, k::Int=10, maxiters::Int=5000, tol::Float32=1f-8, verbose::Bool=true)
        X    = eltype(X) ==  Float32 ? X : Float32.(X)
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
            println("NMF structure successfully initialized! ⌣\n
            >>> nmf rank set to $(k) \n
            >>> maximum number of iterations set to $(maxiters) \n
            >>> tolerance set to $(tol) \n
            >>> X: $( round(count(X .== 0) * 100 / (n*l), digits=2))% --> missing values \n
            >>> X: $( round(count(X .< 0) *100 / (n*l), digits=2))% --> negative values")
        end

        new(X, W, H, V, M, n,l, k, maxiters, tol);
    end
end 

@doc raw"""
    nearly!(nmf::nNMF)

Description
===========
This is a mutating function! runs the nearly NMF solver on the data matrix nmf.X, and updates nmf.W & nmf.H in-place.
nearly NMF tries to solve:

```math
\begin{aligned}
& \quad \quad \quad \min_{W\in \mathbb{R}^{n \times k}, H\in \mathbb{R}^{k \times l}} &&||(X+Y) - (WH + Y)||_{V}^2\\
& \quad \quad \quad \text{\quad \quad s.t.} && W \geq 0 \;\text{and} \; H \geq 0 
\end{aligned}
```
Y holds the minimum shift to make each entry in X non-negative. To minimize this objective, the nearly NMF solver iteratively updates $W$ and $H$ matrices with the following rules:

```math
\begin{array}{l}
H = H \odot \dfrac{\left[ W^{T} (V \odot X) \right]^{+}}{ W^{T}(V \odot (WH)) + \left[W^{T}(V \odot X) \right]^{-}}\\
W = W \odot \dfrac{\left[ (V \odot X)H^{T} \right]^{+}}{ (V \odot (WH))H^{T} + \left[(V \odot X)H^{T} \right]^{-}} \\
\end{array}
```
Where $\odot$ is the element-wise product. Operation $[\;]^{+}$ applied on a matrix results in a matrix with the same shape, in which all negative values are zeroed. $[\;]^{-}$ on the other hand, zeroes the positive values and takes the absolute value of negative values.

Arguments
=========
- **`nmf::nNMF`**: nNMF Struct type

Returns
=======
- **`χ2`**   : reconstruction error (frobenius norm) at last iteration 
- **`iter`** : exit/last iteration


Reference
=========
Reference: [Dylan Green and Stephen Bailey 2024](https://arxiv.org/abs/2311.04855) 

"""
function nearly!(nmf::nNMF)
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

# frobenius norm 
function frobenius(nmf::nNMF)
    χ2   = sum(nmf.V .* ((nmf.X .- (nmf.W * nmf.H)) .^ 2)) / count(nmf.V .> 0)
    return χ2
end
# positive/negative split
function pn_split(A::Matrix{Float32})
    return 0.5f0 * (A .+ abs.(A)),  0.5f0 * (A .- abs.(A))
end

# clamp NaNs and Infs 
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