# Context 

## The Multi-Unit Spectroscopic Explorer (MUSE)
MUSE [1] is an optical integral-field Spectrograph installed on the Very Large Telescope (VLT) operated by ESO. 
MUSE delivers hyperspectral datacubes over a 1′×1′ field of view (FOV), providing a full spectrum for each pixel (called spaxel) with a spectral resolution of 
𝑅 ∼ 2000-4000 across the wavelength range ($\sim 4600-9350 \AA$). (+ sentence about sensitivity). This unique combination of spatial and spectral information and its sensitivity makes MUSE a powerful instrument for deep-field surveys and galaxy evolution studies. 

## Challenges
While multi-object spectroscopy (MOS) surveys typically target relatively bright, pre-selected sources from photometric catalogs, integral-field spectroscopy delivers spectra for all objects within the FOV without any target pre-selection, reaching depths of magnitude 28 and beyond [2,3]. In this context, MUSE presents specific challenges arising from its wavelength coverage, which enables the detection of objects spanning a wide redshift range (z = 0–6.7). This breadth introduces significant emission-line ambiguities, most notably between the [O II] λλ3727, 3729 Å doublet at z < 1.5 and Lyα λ1216 Å at z > 2.8.

Thanks to its exquisite sensitivity, MUSE can efficiently find emission line objects without any continuum. As a result, the task of
source detection and redshift identification in deep fields needs to be performed twice [3], once on continuum detected objects and once on emission line detected objects, and currently, no existing tool can perform well on both types without visual inspections and validation. In addition, in the "redshift desert" at 1.5 < z < 2.8, galaxies lack strong spectral emission features, making redshift determination heavily dependent on stellar continuum shape. These challenges call for new approaches capable of handling both continuum and emission-line dominated spectra across the full MUSE redshift range.

## `Moose.jl`
Several tools for automated galaxy redshift inference have been developed and have demonstrated robust performance [refs]. However, these methods have not been systematically validated across the full redshift range probed by MUSE, nor across the galaxy population present in deep integral-field surveys. When applied to MUSE deep fields, existing approaches typically underperform, confuse emission lines (mainly [O II]/Lyα), and require visual inspections for redshift validation [refs].

`Moose.jl` was developed to address these limitations and to provide a framework for automated redshift inference tailored to MUSE data. The package is designed to operate on all classes of spectra (continuum dominated with no emission lines, spectra with only emission lines, ...), enabling consistent redshift estimation across the full MUSE redshift range (z = 0–6.7), including the redshift desert at 1.5 < z < 2.8. `Moose.jl` aims to reduce emission-line confusion, minimize the need for visual inspection.
   
`Moose.jl` implements a method for automated galaxy redshift prediction, enabled by the availability of approximately 10,000 MUSE galaxy
spectra with redshift labels. The method is based on Non-negative Matrix Factorization (NMF), which learns a low-rank,
additive, and non-negative representation of galaxy spectra. By enforcing non-negativity, NMF provides a parts-based and more
interpretable representation compared to other techniques like principle component analysis (PCA). The method finds the best red-
shift solution by non-negatively projecting a spectrum onto this representation for a range of trial redshifts, and then selecting
the projection with the lowest reconstruction error.

The main advantages of this method are the following: 
* It uses a learned representation. In contrast to model-based representations, learning the representation allows it to naturally capture the diversity present in the dataset. 
* It uses NMF instead of the more commonly adopted PCA. Compared to PCA, NMF provides a parts-based and more interpretable representation.
* Exploits simultaneously continuum and emission/absorption spectral features for redshift inference.
* Trial redshift appraoch followed by a χ² metric and significance, robustness scores attributions provides interpretability for the method.  

## References

[1]: Bacon Roland et al., *The MUSE second-generation VLT instrument*, 2010. https://arxiv.org/pdf/2211.16795

[2]: Bacon, R., Conseil, S., Mary, D., Brinchmann, J., Shepherd, M., Akhlaghi, M., Weilbacher, P. M., Piqueras, L., Wisotzki, L., Lagattuta, D., Epinat, B., Guérou, A., Inami, H., Cantalupo, S., Courbot, J. B., Contini, T., Richard, J., Maseda, M., Bouwens, R., … Carollo, M. (2017). The MUSE Hubble Ultra Deep Field Survey. I. Survey description, data reduction, and source detection. *Astronomy and Astrophysics*, 608, A1. https://doi.org/10.1051/0004-6361/201730833 :contentReference[oaicite:0]{index=0}

[3]: Bacon, R., Brinchmann, J., Conseil, S., Maseda, M., Nanayakkara, T., Wendt, M., Bacher, R., Mary, D., Weilbacher, P. M., Krajnović, D., Boogaard, L., Bouché, N., Contini, T., Epinat, B., Feltre, A., Guo, Y., Herenz, C., Kollatschny, W., Kusakabe, H., Leclercq, F., … Wisotzki, L. (2023). The MUSE Hubble Ultra Deep Field surveys: Data release II. *Astronomy and Astrophysics*, 670, A4. https://doi.org/10.1051/0004-6361/202244187
