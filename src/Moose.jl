module Moose


include("Moosenmf.jl")
include("Strutture.jl")
include("Metrics.jl")
include("FNNLS.jl")
include("moose.jl")
include("Leggere.jl")



export GF, MAE, MAD, Γgrid, Basis, leggere_fits

end
