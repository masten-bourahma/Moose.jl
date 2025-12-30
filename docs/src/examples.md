## Running Non-negative Matrix Factorization (NMF)
We illustrate how to run non-negative matrix factorization on the dataset used in this [paper]().
The data file can be downloaded [here]().
```bash
wget 
```
The file, provided in HDF5 format, contains three HDF5 datasets: `redshifts`, `spectra_matrix`, `std_matrix`.
`spectra_matrix` is a two-dimensional array containing a MUSE galaxy spectrum in each row. Each spectrum represents flux densities sampled on a fixed rest-frame log-wavelengths grid. `std_matrix` is the associated per-pixel uncertainities (standard deviations). `redshifts` is a vector listing the redshift of each spectrum in the `spectra_matrix` dataset.

To read this file run the following commands
```
file = h5open("X_2p170461fm05_1p0_unsorted.h5", "r")

```
```text 
🗂️ HDF5.File: (read-only) (complete the path)/X_2p170461fm05_1p0_unsorted.h5
├─ 🔢 redshifts
├─ 🔢 spectra_matrix
└─ 🔢 std_matrix
```
Next, we can read the `spectra_matrix` and the `std_matrix` datasets.
```julia
X = read(file["spectra_matrix"])
Σ = read(file["std_matrix"])
```
(Currate step here)
...

`Moose.jl` provides an implementation of nearly-NMF [] adapted from this [package](). To run nearly-NMF, we first create an instance of the `nNMF` Struct. At this step, we must provide the spectra matrix `X` and the variance matrix `V = 1 / Σ²` and the desired rank `k`. 
```julia
nmf = nNMF(X; V= 1 ./ (Σ .^2), k= 10)

```
the `nmf` instance has several fields that hold NMF solver matrices and run metadata, most importantly the weights matrix field `.W` and the basis vectors matrix field `.H`.

We then apply the `nearly!()` method on this `nNMF` instance
```julia
nearly!(nmf)
```

After the run finishes, one can access the basis vectors matrix by just calling 
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

In this example, we demonstrate how to run `Moose.jl` on a three-dimensional astronomical datacube.
Throughout this demo, we will work with a small test datacube located @ "data/cubes/DATACUBE_test.fits".

Astronomical datacubes produced by MUSE are very large, with dimensions on the order of ($N\\_x$× $N\\_y$ × $N\\_\lambda$) ≈ (350 × 350 × 3721). Loading such datasets directly into RAM can easily saturate system memory. To address this limitation, `Moose.jl` is designed to operate efficiently on memory-mapped data, allowing only the required portions of the cube to be loaded on demand.
The recommended workflow is therefore:
(1) convert the FITS datacube into an HDF5 file.
(2) memory-map the relevant datasets using built-in functions from the HDF5.jl package.
(3) process the cube without ever fully loading it into memory.

#### **Step 1**: FITS to HDF5 format
```julia
fits_path = "data/cubes/DATACUBE_test.fits"
h5_path   = "data/cubes/DATACUBE_test.h5"
fits_to_h5(fits_path, h5_path)
```
**[Output]**
```output
⚪ Reading 'DATA' HDU
✅ FITS 'DATA' size: (4, 4, 3721)
⚪ Writing 'DATA' to HDF5 file
🧹 Freed memory from 'DATA'
⚪ Reading 'STAT' HDU
✅ FITS 'STAT' size: (4, 4, 3721)
⚪ Writing 'STAT' to HDF5 file
🧹 Freed memory from 'STAT'
✅ HDF5 file saved @:data/cubes/DATACUBE_test.h5
```
> **ℹ️ Note 1:** `h5_path` can be set to `nothing`, in which case a default path will be generated by replacing the ".fits" file extension with an ".h5" one.

> **ℹ️ Note 2:** `fits_to_h5()` has two addtional keyword arguments `DataExtName` and `StatExtName` with "DATA" and "STAT" as default values repectively. To set other values for these parameters call the function this way:
> ```julia
>  fits_to_h5(fits_path, h5_path; DataExtName="?", StatExtName="?")
> ```

> **ℹ️ Note 3:** The `fits_to_h5()` function does more than extracting data header units into an HDF5 file. It also extracts and stores all essential header information required to process the datacube, including: 
> * Spatial and spectral axis sizes
> * Spectral sampling
> * Reference wavelength of the spectral axis
> These parameters are internally abstracted by `Moose.jl`, allowing users to interact with the datacube at a high level without manually handling FITS headers.

One can check the content of "data/cubes/DATACUBE_test.h5" by running, 
```julia
using HDF5
h5open("data/cubes/DATACUBE_test.h5", "r")
```
```text
🗂️ HDF5.File: (read-only) /data/cubes/DATACUBE_test.h5
├─ 🏷️ N₁
├─ 🏷️ N₂
├─ 🏷️ N₃
├─ 🏷️ δλ
├─ 🏷️ λᵣ
├─ 🔢 data
└─ 🔢 stat
```
#### **Step 2**: Redshift predictions on memory-mapped data using `flow()` methods
At this stage, the datacube is ready to be memory-mapped and processed efficiently by `Moose.jl` methods.
`Moose.jl` provides a `flow()` method for redshift predictions in datacubes. It internally handles data reading, memory-mapping, redshift predictions and output writing.  

To call this method, we must specify two arguments `src` and `Val(?)`,  and optionally the keyword argument `δζᵣ`. The first argument is the path to the HDF5 source file created in step (1). The second argument `Val(?)` is a type-driven Dispatch, for this method we implemented two dispatches `Val(:kernel)` and `Val(:spaxels)`, calling the method with the first type will run redshift predictions on a sliding window defined by a 3X3 mean kernel whereas the second type will run them on individual spaxels. The keryword argument specifies the test redshifts spacing, default value is 0.0005.

```julia
src   = "data/cubes/DATACUBE_test.h5"
flow(src, Val(:spaxels); δζᵣ =  0.0005f0)
# to run redshift predictions with a 3X3 mean kernel call:
#flow(src, Val(:kernel); δζᵣ =  0.0005f0)
```
```text
[ Info: 📄 File generated for the output! @ ../data/cubes/DATACUBE_test_Moose.h5
✅ HDF5 file created: ../data/cubes/DATACUBE_test_Moose.h5
Progress: 100%|█████████████████████████████████████████| Time: 0:00:11
```
> **⚙️ Functionality:** The `flow()` method is interruption-safe. It persistently stores progress indices during execution and resumes processing from the last saved state as long as the output file has not been removed or relocated.

The output file is in HDF5 format with a deafult name constructed by adding the suffix"_Moose" to the source file. The output file stores different HDF5 attributes related to run metadata, an HDF5 group named "CHI2\_KERNEL" or/and "CHI2\_SPAXELS" depending on the type-dispatch used for redshift predictions. The HDF5 group then holds HDF5 datasets, each dataset is a chi-square curve named with its indices in the cube (e.g. chi-square curve of spaxel (1,1) will be named "1\_1").

Checking the content of "/data/cubes/DATACUBE_test_Moose.h5" will show,
```text
🗂️ HDF5.File: (read-only) ../data/cubes/DATACUBE_test_Moose.h5
├─ 🏷️ Date (attributes)
├─ 🏷️ Description
├─ 🏷️ Moose_version
├─ 🏷️ N1
├─ 🏷️ N2
├─ 🏷️ N_chi2
├─ 🏷️ dz
├─ 🏷️ i_t
├─ 🏷️ j_t
├─ 🏷️ zmax
├─ 🏷️ zmin
└─ 📂 CHI2_SPAXELS (HDF5 group)
   ├─ 🔢 1_1 ├─ 🔢 1_2 ├─ 🔢 1_3 ├─ 🔢 1_4 (chi-square datasets)
   ├─ 🔢 2_1 ├─ 🔢 2_2 ├─ 🔢 2_3 ├─ 🔢 2_4
   ├─ 🔢 3_1 ├─ 🔢 3_2 ├─ 🔢 3_3 ├─ 🔢 3_4
   ├─ 🔢 4_1 ├─ 🔢 4_2 ├─ 🔢 4_3 └─ 🔢 4_4
```

## Deblending
!!! info
    This section is under construction.

