## Usage example, running Moose on fits files
Here we describe how to run Moose on a collection of fits files. We suppose that all fits files are located in one folder.
This package comes with a sample of fits files located in "data/spectra_sample"

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
1. First, let's read content of these files into a data struct

```julia
path = "./data/spectra_sample"
data = leggere_fits(path) 
```
The data struct has four fields, "flux", "sdev", "awave", "ids"

2. Initialize Basis Struct, specify output file path and run 'χloop' function
```julia
basis       = Basis()
my_output   = "../output/chi2_files/chi2_moose.h5"
χloop(basis, data; output_path = my_output)
```