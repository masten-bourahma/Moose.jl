# # Usage Example
# Let's start by importing the needed packages

using Moose, FITSIO, Plots
using DisplayAs #hide
theme(:ggplot2) #hide

# Let's instantiate a rest frame wavelength grid (Γgrid) and the Basis Struct. 
#md # ```julia
#md # wgrid = Γgrid() 
#md # basis = Basis() 
#md # ```

# Let's load a spectrum from a .fits file,
hdul = FITS("../../../data/spectra_sample/udf10_00002.fits", "r") 
fλ   = read(hdul["DATA"])
σλ   = read(hdul["STAT"]) |> (x -> sqrt.(x))
λref = read_header(hdul["DATA"])["CRVAL1"]
L    = read_header(hdul["DATA"])["NAXIS1"]
δλ   = read_header(hdul["DATA"])["CDELT1"]
λ    = collect(Float32, λref .+ (0:L-1) .* δλ)
close(hdul)

# Take a look at the spectrum!
p = plot(λ, fλ, lw =2 , xlabel ="λ [Å]", ylabel ="fλ [ergs Å^-1 s^-1 cm^-2]", label ="spectrum",)# size =(500,200))
plot!(p, λ, σλ, lw =2 , alpha = 0.7, label = "standard deviations")
p = DisplayAs.PNG(p) #hide


# Usually raw data comes with many NaNs and Infs, let us clean the flux densities and the standard deviations. We assign zeros for NaNs in fluxes, and a high number for NaNs and Infs in standard deviations,

#md # ```julia
#md # replace!(fλ, NaN => 0)
#md # replace!(σλ, NaN => 1f6, Inf => 1f6)
#md # σλ[fλ .== 0] .= 1f6
#md # ```

# Now, we can interpolate the flux densities and standard deviations to the rest frame grid, for this just run,
#md # ```julia
#md # intrpfλ, intrpσλ = interpolate(basis, fλ .* λ, σλ .* λ , λ)
#md # ```

# Run a threaded NNLS to test all possible redshifts 

#md # ```julia
#md # χ2 = threaded_nnls(basis, intrpfλ, intrpσλ)
#md # ```




# Plot the obtained χ2 curve.
#md # ```julia
#md # p = plot(wgrid.ζ, χ2, xlabel = "redshift", ylabel= "χ2")
#md # ```
using HDF5 #hide
f = h5open("../../../output/chi2_files/chi2_testMoose.h5", "r") #hide
χ2 = read(f, "2") #hide
close(f) #hide
p = plot( collect(Float32, 0:0.001:7), lw= 2, χ2, xlabel = "redshift", ylabel= "χ2", legend= false) #hide
p = DisplayAs.PNG(p) #hide


