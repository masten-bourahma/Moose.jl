# Moose

[![Build Status](https://github.com/masten-bourahma/Moose.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/masten-bourahma/Moose.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Paper
[to come]()

## Description
`Moose.jl` is a Julia package for automatic galaxy redshift estimation, developed for the Multi-Unit Spectroscopic Explorer (MUSE). `Moose.jl` uses a learned rest frame representation of galaxy spectra through a non-negative matrix factorization (NMF) in a data-driven setup. `Moose.jl` leverages this learned basis by decomposing a given unseen spectrum into this basis under the assumption of all possible redshift. Recording the error of each decomposition (in the sense of `χ²`) results in a χ²-curve. The minimum of this curve gives the predicted redshift. Moreover, `Moose.jl` quantifies the signficance of this minimum as well as its separation with the second minimum (robustness `R`) to asses its quality.

This package includes,
- An implementation of [nearly-NMF]()
- An implementation of the Fast Non-Negative Least Squares [FNNLS]()
- A set of methods/functions for redshift prediction


## Installation 

```julia REPL
julia> using Pkg
julia> Pkg.add("Moose")
```

## Details
`Moose.jl` leverages a data-driven representation of MUSE galaxy spectra in the rest frame, learned through Non-negative Matrix Factorization (NMF), and  exploits this representation to predict redshifts for previously unseen galaxies. 

![`Moose.jl workflow`](docs/src/assets/nmf_flow.png)
 
The performance of `Moose.jl` is assessed in `learn/test` data split configuration, where the `learn` fraction of the data is used to learn the basis vectors, and the test fraction is used to assess the performance on the redshift prediction task. The metric for the performance is the `Good Fraction` (GF). GF is defined as the fraction of prediction satisfying $$\Delta z = |z_p - z_t| < 0.1 $$ ($$z_p, z_t$$ are the predicted and true redshift vectors) over the total number of predictions. We quote a GF of $$94\%$$.

The plot below shows the GF of `Moose.jl` as a function of the signal-to-noise (SNR) ratio and as function of redshift $$z$$
![`Moose.jl performance`](docs/src/assets/test_z_snr.png)
