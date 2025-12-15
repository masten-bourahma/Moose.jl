using Statistics, Peaks, Base.Threads, LinearAlgebra, Random, PyCall

#np      = pyimport("numpy")
#mpdaf   = pyimport("mpdaf")
#ppf     = pyimport("pyplatefit")
#astropy = pyimport("astropy");

mutable struct Source
    f::Vector{Float32}
    σ::Vector{Float32}
    λ::Vector{Float32}
    f₁::Vector{Float32}
    σ₁::Vector{Float32}
    z::Float32
    zconf::Int
    function Source(data, i)
        f     = data.flux[i]
        σ     = data.sdev[i]
        λ     = data.awave[i]
        z     = data.redshift[i]
        zconf = data.zconf[i]
        
        f₁ = similar(f)
        σ₁ = copy(σ)
        new(f, σ, λ, f₁, σ₁, z, zconf)
    end
end

mutable struct Buffer
    fʳ  ::Vector{Float32}
    σʳ  ::Vector{Float32}
    fʳ₁ ::Vector{Float32}
    fʳ₂ ::Vector{Float32}
    σʳ₁ ::Vector{Float32}
    χ²₁ ::Vector{Float32}
    χ²₂ ::Vector{Float32}
    Ω₁  ::Matrix{Float32}
    Ω₂  ::Matrix{Float32}

    function Buffer(k,l,n)
        fʳ  = Vector{Float32}(undef, l)
        σʳ  = Vector{Float32}(undef, l)
        fʳ₁ = Vector{Float32}(undef, l)
        fʳ₂ = Vector{Float32}(undef, l)
        σʳ₁ = Vector{Float32}(undef, l)
        χ²₁ = Vector{Float32}(undef, n)
        χ²₂ = Vector{Float32}(undef, n)
        Ω₁  = Matrix{Float32}(undef, k, n)
        Ω₂  = Matrix{Float32}(undef, k, n)
        new(fʳ, σʳ, fʳ₁, fʳ₂, σʳ₁, χ²₁, χ²₂, Ω₁, Ω₂)
    end
end

function deblend!(wgrid::Γgrid, basis::Basis, source::Source, buffer::Buffer)
    
    interpolate!(basis, source.f, source.σ, source.λ, buffer.fʳ, buffer.σʳ)
    flow!(basis, buffer.fʳ , buffer.σʳ, buffer.χ²₁, buffer.Ω₁)

    i₁ = argmin(buffer.χ²₁) 
    z₁ = wgrid.ζ[i₁]

    ppf_fit!(source.λ, source.f, source.σ, z₁, buffer.f₁)
    
    interpolate!(basis, source.f₁ , source.σ₁, source.λ, buffer.fʳ₁, buffer.σʳ₁)
    flow!(basis, buffer.fʳ, buffer.σʳ₁, buffer.fʳ₁, buffer.χ²₂, buffer.Ω₂)

    i₂ = argmin(χ²₂)
    z₂ = wgrid.ζ[i₂]

    buffer.fʳ₂ .= vec(Ω₂[î₂]' * vcat(transpose(fʳ₁), basis.Hi[i₂]))
    
    result = Dict( "χ²₁" => buffer.χ²₁, "χ²₂" => buffer.χ²₂,
                                    "r₀" => buffer.fʳ, "r₁" => buffer.fʳ₁, "r₂" => buffer.fʳ₂,
                                    "z₀" => source.z, "z₁" => z₁, "ẑ₂" => z₂, "zconf" => source.zconf,
                                    "Δχ²₁" => Δχ²(χ²₁), "Δχ²₂" => Δχ²(χ²₂), "R₁" => R(χ²₁), "R₂" => R(χ²₂))
    return result
end

function ppf_fit!(λ::Vector{T}, f::Vector{T}, σ::Vector{T}, z::T, f₁::Vector{T}) where{T}
    # pyplatefit has verbose with no kryword to control iter_segments
    # this function calls pyplatefit and redirects verbose to devnull
    
    emcont, abs = nothing, nothing
    line_fit = ppf.Linefit()

    lines_table = astropy.table.Table.read("/home/masten/Downloads/lines_table_extended2.fits")
    emlines     = ppf.get_lines(user_linetable = lines_table)

    abslines    = [ "MgII2796", "MgII2803", "MgI2853",
                    "FeII1608", "FeII1611","FeII2344", "FeII2374", "FeII2382","FeII2586", "FeII2600",
                    #"CaK", "CaH", "CaG",
                    "AlII1671", "AlIII1854", "AlIII1862",
                    "SiII1260", "OI1302", "SiII1304", "CII1334", "CII1334", "SiIV1394", "SiIV1403",    
                    "SiII1527", "CIV1548", "CIV1550", ]

    open("/dev/null", "w") do devnull
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                wave   = mpdaf.obj.WaveCoord(cdelt=1.25, crval= λ[1], ctype= "AWAV", cunit= astropy.units.angstrom)
                spec   = mpdaf.obj.Spectrum(wave = wave, data= fλ, var = σλ .^2)
                
                if z > 0.64738f0
                    abs = line_fit.absfit(spec,z, lines = abslines)
                    abs = collect(T, abs["abs_fit"]) .- collect(T, abs["abs_cont"])
                else 
                    abs = zeros(T,length(fλ))
                end

                emcont = ppf.fit_spec(spec, major_lines=false, fitlines=true, z = z, fitabs =false, lines = emlines, dble_lyafit= true)
                emcont = collect(T, emcont["spec_fit"]) 
            end
        end
    end

    f1 .= emcont .+ abs
end

const STRONG_LINES5 = Dict( "Lyα     " => 1215.67f0,
                                          #"C IV"    => 1549.0,
                                           "C III]  " => 1908.0f0,
                                           "Mg II   " => 2798.0f0,
                                           "[O II]  " => 3727.4235f0,
                                           "[Ne III]" => 3869.0f0,
                                           "Hδ      " => 4102.0f0,
                                           "Hγ      " => 4341.0f0,
                                           "Hβ      " => 4861.0f0,
                                           "[O III]w" => 4959.0f0,
                                           "[O III]s" => 5006.843f0,
                                           "Hα      " => 6562.819f0,
                                           "[S II]1 " => 6717.0f0,
                                           "[S II]2 " => 6731.0f0)

function line_confusion(z₁::Float32, tol::Float32=10f0)
    results = []
    zconfusing = []
    for (name1, λ₁) in STRONG_LINES5
        λ_obs = (1 + z₁) * λ₁
        
        for (name2, λ₂) in STRONG_LINES5
            #name1 == name2 && continue
            # Solve for z2 that gives same observed wavelength
            z₂ = (λ_obs / λ₂) - 1
            
            Δ = abs((1+z₂)*λ₂ - λ_obs)
            if Δ ≤ tol && (0f0 ≤ z₂   ≤ 6.7f0) && (4700 < λ_obs < 9356)
                push!(results, (name1, name2, z₁, z₂, λ_obs, Δ))
                push!(zconfusing, z₂)
            end
        end
    end
    return results, zconfusing
end

function blended_test(γ::Vector{Float32}, χ²₁::Vector{Float32}, χ²₂::Vector{Float32}, t::Float32)
    blended = false
    z₁ = γ[argmin(χ²₁)]
    z₂ = γ[argmin(χ²₂)]

    Δχ²₁ = Δχ²(χ²₁)
    Δχ²₂ = Δχ²(χ²₂)

    #check for possible confusion between z₁ and z₂
    
    # check blending
    if Δχ²₂ > t
        #if (abs(z₁ - z₂) <= 0.01f0) 
        #    blended = false
        if look(g.ζ,χ²₁, χ²₂, z₁, t)
            blended = true
        else
            blended = false
        end
    else
        blended = false
    end
    return blended
end

function look(ζ::Vector{Float32}, χ²₁::Vector{Float32}, χ²₂::Vector{Float32}, z₁::Float32, t::Float32)
    confusions, zₓ = line_confusion(z₁)

    mins = findminima(χ²₂, 20).indices
    Δχ²_scores1 = modified_Δχ²(χ²₂, mins)
    Δχ²_scores2 = Δχ²(χ²₂, mins)
    #filter minima 
    zᵢ = ζ[mins[(Δχ²_scores1 .>= 3.f0 ) .&& (Δχ²_scores2 .>= t)]] 
    #println(zᵢ)
    #println(Δχ²_scores[Δχ²_scores .>= t])

    #check confusion between redshifts in zᵢ and z₁
    blended =falses(length(zᵢ))

    for (i,z) in enumerate(zᵢ)
        if any(abs.(z .- zₓ) .<= 0.01)
            blended[i] = false
            #println(confusions[abs.(z .- zₓ) .<= 0.01])
        else
            #println(z)
            blended[i] = true
        end
    end
    #println(sort(Δχ²_scores))

    return any(blended)
end