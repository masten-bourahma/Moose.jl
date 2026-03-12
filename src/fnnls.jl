""" 
    fnnls

Description
===========
Implementation of the Fast Non-Negative Least Squares algortithm (FNNLS) adopted from the NonNegLeastSquares.jl package, 
(see link[https://doi.org/10.1002/(SICI)1099-128X(199709/10)11:5<393::AID-CEM483>3.0.CO;2-L]).
- Problem under consideration:
    Ax = b, subject to x >= 0

- FNNLS algorithm solves this problem in the least squares sense,
    || Ax - b||^2, subject to x >= 0


Arguments
=========
- **`AtA    ::Matrix{Float32}`**       : Gram matrix (A transpose A)
- **`Atb    ::Matrix{Float32}`**       : Gram matrix (A transpose b)
- **`tol    ::Float32 = 1f-8`**        : tolerance for nonnegativity constraints 
- **`max_iter ::Int= 30*size(AtA,2)`** : Maximum number of iterations)

Returns
=======
- **`x̂    ::Vector{Float32}`** : `x̂` that solves `A*x = b` in the least-squares sense, subject to `x >=0

References
==========
Bro R, De Jong S. A fast non-negativitity-constrained least squares algorithm. Journal of Chemometrics. 11, 393–401 (1997)

Credit
======
Tim Holy, (NonNegLeastSquares.jl)

"""
function fnnls(AtA::Matrix{Float32},
               Atb::Vector{Float32};
               tol::Float32=1f-8,
               max_iter=30*size(AtA,2)) 

    n = size(AtA,1)
    x = zeros(Float32, n)
    s = zeros(Float32, n)

    # P is a bool array storing positive elements of x
    # i.e., x[P] > 0 and x[~P] == 0
    P = x .> tol
    w = Atb - AtA*x

    # We have reached an optimum when either:
    #   (a) all elements of x are positive (no nonneg constraints activated)
    #   (b) ∂f/∂x = A' * (b - A*x) > 0 for all nonpositive elements of x
    iter = 0
    while sum(P)<n && any(w[(!).(P)] .> tol) && iter < max_iter

        # find i that maximizes w, restricting i to indices not in P
        # Note: the while loop condition guarantees at least one w[~P]>0
        i = argmax(w .* (!).(P))

        # Move i to P
        P[i] = true

        # Solve least-squares problem, with zeros for columns/elements not in P
        s[P] = AtA[P,P] \ Atb[P]
        s[(!).(P)] .= zero(eltype(s)) # zero out elements not in P

        # Inner loop: deal with negative elements of s
        while any(s[P].<=tol) && iter < max_iter
            iter += 1

            # find indices in P where s is negative
            ind = @__dot__ (s <= tol) & P

            # calculate step size, α, to prevent any xᵢ from going negative
            α = minimum(x[ind] ./ (x[ind] - s[ind]))

            # update solution (pushes some xᵢ to zero)
            x += α*(s-x)

            # Remove all i in P where x[i] == 0
            for i = 1:n
                if P[i] && abs(x[i]) < tol
                    P[i] = false # remove i from P
                end
            end

            # Solve least-squares problem again, zeroing nonpositive columns
            s[P] = AtA[P,P] \ Atb[P]
            s[(!).(P)] .= zero(eltype(s)) # zero out elements not in P
        end

        # update solution
        x = deepcopy(s)
        w .= Atb - AtA*x
    end
    
    return x
end
