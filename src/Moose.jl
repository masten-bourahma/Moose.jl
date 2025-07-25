module Moose


#include("nmf.jl")
include("strutture.jl")
include("metrics.jl")
include("fnnls.jl")
include("method.jl")
include("leggere.jl")



export GF, MAE, MAD, Γgrid, Basis, fnnls, leggere_fits,leggere_chifile, interpolate, threaded_nnls, χloop

end
