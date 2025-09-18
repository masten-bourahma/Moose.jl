# Moose.jl Documentation

Welcome to the documentation for `Moose.jl`!

## Installation 

```julia REPL
julia> using Pkg
julia> Pkg.add("Moose")
```

## Overview

**`Moose.jl`** is a specialized Julia package designed for accurate galaxy redshift prediction. This package was developed for the hyperspectral data of the Multi-Unit Spectroscopic Explorer (MUSE) instrument at the Very Large Telescope (VLT) in Chili.

Moose.jl operates in two main phases:

* **Low-dimensional galaxy representation learning**: It first employs a Non-negative Matrix Factorization (NMF) algorithm to learn a "rest-frame representation" of galaxy spectra. These learned components are referred to as "basis vectors".

* **Redshift Prediction**: Subsequently, these basis vectors are utilized to test all redshift possibilities for any given galaxy spectrum, identifying the most likely redshift.

This package provides the following utilities:
* An implementation of nearly-NMF algorithm (see, )
* An implementation of the Fast Non-Negative Least Squares (FNNLS) algorithm adopted from ()
* An implementation of a Multi-threaded FNNLS
* An implementation of different utility functions, such as a multi-threaded 1D interpolation function.

## Acknowledgement
The development of this package was supervised by Nicolas BOUCHE and Roland BACON 