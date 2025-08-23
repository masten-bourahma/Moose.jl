# Moose

[![Build Status](https://github.com/masten-bourahma/Moose.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/masten-bourahma/Moose.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Description
`Moose.jl` is a Julia package for estimating galaxy redshifts from observations with the Multi-Unit Spectroscopic Explorer (MUSE). 
This package includes an implementation of [nearly-NMF]()

## Installation 

```julia REPL
julia> using Pkg
julia> Pkg.add("Moose")
```

## Technical Details
`Moose.jl` leverages a data-driven representation of MUSE galaxy spectra in the rest frame, learned through Non-negative Matrix Factorization (NMF), and  exploits this representation to predict redshifts for previously unseen galaxies. 
  
![`Moose.jl workflow`](docs/src/assets/nmf_flow.png)
 

