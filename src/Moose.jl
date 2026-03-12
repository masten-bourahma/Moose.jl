module Moose

include("n-nmf.jl")
export nNMF, nearly!

include("strutture.jl")
export Γgrid, Basis, update!

include("metrics.jl")
export GF, Δχ², R

include("fnnls.jl")
export fnnls

include("utils.jl")
export fits_to_h5, observed!
include("methods.jl")
export interpolate, interpolate!, flow, flow!

include("leggere.jl")
export leggere

end
