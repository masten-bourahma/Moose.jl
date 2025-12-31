# `Moose.jl`
[![Build Status](https://github.com/masten-bourahma/Moose.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/masten-bourahma/Moose.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![CompatHelper](https://github.com/masten-bourahma/Moose.jl/actions/workflows/CompatHelper.yml/badge.svg)](https://github.com/masten-bourahma/Moose.jl/actions/workflows/CompatHelper.yml)
[![Dependabot Updates](https://github.com/masten-bourahma/Moose.jl/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/masten-bourahma/Moose.jl/actions/workflows/dependabot/dependabot-updates)

[![pages-build-deployment](https://github.com/masten-bourahma/Moose.jl/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/masten-bourahma/Moose.jl/actions/workflows/pages/pages-build-deployment)
[![Documentation](https://github.com/masten-bourahma/Moose.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/masten-bourahma/Moose.jl/actions/workflows/documentation.yml)

## Description
`Moose.jl` is an open-source Julia package for automated galaxy redshift estimation from optical spectroscopic data. The package was developed in the context of a doctoral research project aimed at constructing robust redshift inference and source detection methods for hyperspectral observations obtained with the Multi-Unit Spectroscopic Explorer (MUSE) instrument at the Very Large Telescope (VLT). The software is designed to scale efficiently to large spectroscopic datasets and to integrate naturally within modern Julia-based scientific computing workflows.

## Methodology
Internally, `Moose.jl` relies on a **learned low-rank rest-frame representation** of galaxy spectra observed with MUSE. This representation is learned via **Non-negative Matrix Factorization (NMF)** applied to a training set of approximately **7,000 galaxy spectra** with known redshifts. The learned representation consists of **10 non-negative basis vectors in the rest frame**, capturing the stellar continuum, emission and absorption spectral features of the galaxy population.

Redshift inference for new spectra is performed by projecting each observed spectrum onto the learned basis across a grid of trial redshifts. The redshift grid spans a user-defined interval (typically  0 ≤ z ≤ 6.7) with configurable resolution. For each trial redshift, `Moose.jl` solves a **non-negative least-squares (NNLS)** problem to reconstruct the spectrum and evaluates the corresponding **χ² statistic** between the observed and reconstructed fluxes.

This routine results in a χ² curve (error as a function of redshift), from which the global minimum is selected as the predicted redshift. In addition, `Moose.jl` computes significance and robustness scores to quantify the confidence of the prediction. 

## Software features
`Moose.jl` package provides the following features:
* An implementation of the nearly Non-negative Matrix Factorization algorithm
* An implementation of the Fast Non-Negative Least Squares (FNNLS)
* An implementation of a multi-threaded FNNLS
* Code for redshift prediction on fits files / datacubes
* Code for spectral deblending (under construction) 


## Installation 
`Moose.jl` is freely available under an open-source license. The development version can be installed via:
```julia REPL
julia> using Pkg
julia> Pkg.add(Pkg.add(url="https://github.com/masten-bourahma/Moose.jl", rev="dev")
```
The package is under active development, and code and interfaces may evolve.

## Accompanying publication
A detailed methodological description and scientific validation of Moose.jl will be presented in a forthcoming publication.
