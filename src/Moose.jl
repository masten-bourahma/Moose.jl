module Moose

include("n-nmf.jl")
export nNMF, nearly!

include("strutture.jl")
export Γgrid, Basis

include("metrics.jl")
export GF, Δχ²

include("fnnls.jl")
export fnnls

include("utils.jl")
include("methods.jl")
export interpolate, interpolate!, flow, flow!

include("leggere.jl")
export leggere

end
