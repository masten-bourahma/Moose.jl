## Running Non-negative Matrix Factorization (NMF)
We illustrate how to run non-negative matrix factorization on the dataset used in this [paper]().
The data file can be downloaded [here](https://doi.org/10.5281/zenodo.18066943).
```bash
$ wget https://doi.org/10.5281/zenodo.18066943
```
The file, provided in HDF5 format, contains four HDF5 datasets: `spectra_matrix`, `variance_matrix`, `redshifts` and `zconf`.
`spectra_matrix` is a two-dimensional array containing a MUSE galaxy spectrum in each row. Each spectrum represents flux densities sampled on a fixed rest-frame log-wavelengths grid. `variance_matrix` is the associated per-pixel uncertainities (variances). `redshifts` is a vector listing the redshift of each spectrum in the `spectra_matrix` dataset, and `zconf` is a vector listing the redshift confidence score attributed to each spectrum.

To read this file run the following commands
```julia
file = h5open("./X_2p170461fm05_1p0_unsorted.h5", "r")
```
```text 
🗂️ HDF5.File: (read-only) (complete the path)/X_2p170461fm05_1p0_unsorted.h5
├─ 🔢 redshifts
├─ 🔢 spectra_matrix
├─ 🔢 variance_matrix
└─ 🔢 zconf 
```
Next, we can read the `spectra_matrix` and the `variance_matrix` datasets. Replace `NaNs` and `Infs` with finite values and take the inverse of the variance matrix. 
```julia
X = replace(read(file["spectra_matrix"]), NaN32 => 0f0);
V = replace(1 ./ read(file["variance_matrix"]), NaN32 => 1f-12, Inf32 => 1f-12);
```
`Moose.jl` provides an implementation of [nearly-NMF](https://arxiv.org/abs/2311.04855) adapted from this [package](https://github.com/dylanagreen/nearly_nmf). To run nearly-NMF, we first create an instance of the `nNMF` Struct. At this step, we must provide the spectra matrix `X` and the inverse variance matrix `V` and the desired rank `k`. 
```julia
nmf = nNMF(X; V= V, k= 10)
```
The `nmf` Struct instance has several fields that hold NMF solver matrices: the weights matrix `W` and the basis vectors matrix field `H`. It also holds other run metadata, see API documentation for more details.

We then apply the `nearly!()` method on this `nNMF` instance
```julia
nearly!(nmf)
```
After the run finishes, basis vectors matrix can be accessed by just calling: 
```julia
nmf.H
``` 
## Running `Moose.jl` on FITS files
This example demonstrates how to run `Moose.jl` on a collection of one-dimensional spectra stored as individual FITS files within a single directory.
We illustrate this workflow using a sample of spectra located @ "data/spectra_sample".

Directory structure:
```bash
$ tree data/spectra_sample
data/spectra_sample/
├── udf10_00002.fits
├── udf10_00004.fits
├── udf10_00005.fits
...
└── udf10_06682.fits
1 directory, 24 files
```
Each FITS file is assumed to contain a single spectrum along with its satandard deviation vector.

#### Step 1 — Load FITS spectra into memory
The first step is to load the content of all FITS files in the directory into a single `NamedTuple` using the `leggere()` function:
```julia
path = "../data/spectra_sample"
data = leggere(path, Val(:fits)) 
```
> **ℹ️ Note:** The `leggere()` method accepts two optional keyword arguments:
> * "`DataExtName` (default: "`DATA`")" 
> * "`StatExtName` (default: "`STAT`")"
>
> These specify the FITS HDU names used to read the fluxes and satandard deviations, respectively.
> To override the defaults:
> ```julia
> data = leggere(path, Val(:fits); DataExtName="...", StatExtName="...")
>``` 
The returned object `data` is a `NamedTuple` with the following field names:`flux`, `sdev`, `awave` and `ids`. 
The ith FITS file quantities can be accessed by indexing, i.e. `data.flux[i]` is the flux vector corresponding to the ith FITS file. 
#### Step 2 — Instantiate the wavelength grid and basis
Next, define instances of `Γgrid` and `Basis` Structs.
```julia
wgrid = Γgrid(λmin = 4700f0, δζ = 0.0005f0)
basis = Basis(wgrid)
```
These structs define the rest-frame log-wavelengths grid and NMF representation used by `Moose.jl` during redshift inference.

#### Step 3 — Run the processing pipeline
Specify the output HDF5 file and run the main processing routine using `flow()`:
```julia
outPath   = "../output/chi2_files/chi2_moose.h5"
flow(wgrid, basis, data; output_path = outPath)
```
The `flow()` method sequentially processes each spectrum in `data`. For each spectrum it writes an HDF5 group named with the spectrum's `id`
into the output file. Each HDF5 group contains the following quantities:

| **Output name**  |**Symbole**| **Type**          | **Dimensions** | **Description**                                            |
|:-----------------|:---------:|:-----------------:| :------------: |:-----------------------------------------------------------|
| `id`             |  id       | `String`          | `-`            | identifier for the HDF5 group                              |
| `chi2_1`         |  χ²₁      | `Vector{Float32}` | `n`            | χ² curve                                                   |
| `r_0`            |  r₀       | `Vector{Float32}` | `l`            | interpolated spectrum                                      |
| `r_1`            |  r₁       | `Vector{Float32}` | `l`            | reconstruction at predicted redshift $z\\_1$               |
| `coeffs_1`       |  ω₁       | `Vector{Float32}` | `k`            | FNNLS decomposition coefficients                           |
| `z_1`            |  z₁       | `Float32`         | `1`            | predicted redshift.                                        |
| `z_1to5`         |  z₅₎      | `Vector{Float32}` | `5`            | first to fifth redshift predictions                        |
| `dchi2_1`        |  Δχ²₁     | `Float32`         | `1`            | significance score of the predicted redshfit               |
| `dchi2_1to5`     |  Δχ²₅₎    | `Vector{Float32}` | `5`            | significance scores of the first 5 redshift predicitions   |
| `R_1`            |  R₁       | `Float32`         | `1`            | robustness score of the predicted redshift                 |
#### Step 4 — Read output file
Read the content of the output file by running,
```julia
output = leggere(outPath, Val(:chi2file))
```
`output` is a `NamedTuple` with the same field names in the table above. However, the type of each field is now wrapped with the Vector type. `output.chi2_1` has the type `Vector{Vector{Float32}}`, and indexing `output.chi2_1` allows to acess individual chi-square curves (other quantities can be accessed by indexing the other field names).
## Running `Moose.jl` on a datacube
!!! info
    This section is under construction
## Deblending
!!! info
    This section is under construction.

