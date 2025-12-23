using Statistics

"""
    sanitize(z,ẑ)

Description
===========
Help function, this function checks for any NaN or Inf in the input vectors of true
and predicted redshifts (z & ẑ respectively), if any it masks them and raises a warning
to the user. It also ensures that both vectors have the same size.

Arguments
=========

- **`z`**: vector of true redshifts
- **`ẑ`**: vector of predicted redshifts

Author(s)
=========
B. Masten
"""

function sanitize(z::Vector{Float32},ẑ::Vector{Float32})

    # 1. Check for same size
    if length(z) != length(ẑ)
        err = "Vectors have different sizes: $(length(z)) and $(length(ẑ))"
        error(err)
    end
    # 2. Check for NaN or Inf
    if !all(isfinite, z)
        @warn "z contains non-finite values (NaN or Inf). A mask will be applied!"
    end

    if !all(isfinite, ẑ)
        @warn "ẑ contains non-finite values (NaN or Inf). A mask will be applied!"
    end

    # 3. Mask NaNs & Inf if any and calculate GF 
    isok = isfinite.(z) .&& isfinite.(ẑ)
    z    = z[isok]
    ẑ    = ẑ[isok]
    return z, ẑ
end

@doc raw"""
    GF(z::Vector{Float32},ẑ::Vector{Float32}; tMUSE::Float32 = 0.1f0)

Description
===========
This function calculates the Good Fraction (GF), given vectors of true and predicted redshift z and ẑ.
GF is defined as,
```math
\begin{aligned}
&& GF &= \frac{\rm good}{N} \\
\end{aligned}
```
``N_{\rm good}`` and ``N`` are respectively the number of good predictions and the total number of predictions. A prediction is good when it satistifies ``\Delta z = |z - ẑ| < t``, 
where ``t`` is a threshold that can be set by setting the keyword argument `tMUSE`. 

Arguments    
=========

- **`z`**: vector of true redshifts
- **`ẑ`**: vector of predicted redshifts
- **`tMUSE`**: redshift error tolerance for MUSE

Details
=======

Given the vectors of true redshifts z and the predicted redshifts ẑ, this function
calculates the good fraction GF, defined as the ratio between the number of good predictions
and the total number of predictions. A good prediction is a one that has an error
Δz   = |z - ẑ| less than a threshold t (tMUSE) in the code, the threshold is 
set by the scientific goal.

Author(s)
=========

B. masten
"""
function GF(z::Vector{Float32},ẑ::Vector{Float32}, tMUSE::Float32 = 0.1f0)
    z, ẑ  = sanitize(z, ẑ )
    Δz    = abs.(z .- ẑ)
    Ngood = sum(Δz .< tMUSE)
    N     = length(z)
    return Ngood/N
end

"""
    MAE(z::Vector{Float32}, ẑ::Vector{Float32})

Description
===========
This function calculates the mean absolute error (MAE) score, given the vectors of true
and predicted resdshifts (z & ẑ respectively). MAE is calculated as follow:
MAE = (Σ_i i=1..N |z - ẑ|) / N

Arguments
=========

- **`z`**: vector of true redshifts
- **`ẑ`**: vector of predicted redshifts

Author(s)
=========

B. Masten
"""
function MAE(z::Vector{Float32}, ẑ::Vector{Float32})
    z, ẑ  = sanitize(z, ẑ )
    Δz    = abs.(z .- ẑ)
    N     = length(Δz)
    MAE_   = sum(Δz) /N
    return MAE_
end

"""
    MAD(z::Vector{Float32}, ẑ::Vector{Float32})

Description
===========

This function calculates the median absolute deviation (MAD), given the vectors
of true and predicted resdshifts (z & ẑ respectively). MAD is calculated as follow:
MAD = median(|Δz_i - median(Δz)|)

Arguments
=========

- **`z`**: vector of true redshifts
- **`ẑ`**: vector of predicted redshifts

Author(s)
=========

B. Masten
"""
function MAD(z::Vector{Float32}, ẑ::Vector{Float32})
    z, ẑ   = sanitize(z, ẑ )
    Δz     = abs.(z .- ẑ)
    Δz̃     = median(Δz)
    MAD_   = median(abs.(Δz .- Δz̃))
    return MAD_
end


"""
    Δχ²(χ²::Vector{Float32})

Description
===========

This function calculates the Δχ² significance score, given a chi-square curve
Δχ² is calculated as follow:

```math
\begin{aligned}
&& \\Delta \\chi^2 & = 1 - \frac{\\chi^2_{\rm min}}{\rm{Q1}(\\chi^2)} \\
\end{aligned}
```

Arguments
=========

- **`χ²`**: chi-square curve vector

Returns
=======

- **`Δχ²`**: redshift significance score

Author(s)
=========

B. Masten
"""
function Δχ²(χ²::Vector{Float32})
    Q₁     = quantile(χ², 0.25)
    Δ      = 1 - minimum(χ² ./ Q₁)
    return Δ 
end
function Δχ²(χ²::Vector{Float32}, indices::Vector{Int})
    Q₁     = quantile(χ², 0.25)
    Δ      = 1 .- (χ²[indices] ./ Q₁)
    return Δ 
end

"""
    R(χ2::Vector{Float32})

Description
===========

This function calculates the R robustness score, given a chi-square curve
R is calculated as follow:

```math
\begin{aligned}
&& R & = 1 - \frac{\\chi^2_{\rm min}}{\rm{Q1}(\\chi^2)} \\
\end{aligned}
```

Arguments
=========

- **`χ2`**: chi-square curve vector

Returns
=======

- **`R`**: redshift robustness score

Author(s)
=========

B. Masten
"""
function R(χ²::Vector{Float32})
    N      = length(χ²)
    Q₁     = quantile(χ², 0.25)
    min1   = minimum(χ²) 
    mindex = argmin(χ²)
    mask   = ones(Bool, N)
    mask[max(1, mindex - 25): min(N, mindex + 25)] .= false
    min2   = sort(χ²[mask])[1]

    R_ =  (min2 - min1) / std(χ²[χ² .<= Q₁])
    
    return R_
end

