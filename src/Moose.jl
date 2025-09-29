module Moose

include("n-nmf.jl")
export nNMF, nearly!

include("strutture.jl")
export Γgrid, Basis

include("metrics.jl")
export GF

include("fnnls.jl")
export fnnls

include("methods.jl")
export interpolate, threaded_fnnls, χfits,  χcube

include("leggere.jl")
export leggere_fits, leggere_chifile, leggere_cube


end
