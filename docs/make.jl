
push!(LOAD_PATH, dirname(@__DIR__))

using Documenter, DocThemeIndigo, Literate, Moose
indigo_css_path = DocThemeIndigo.install(Moose) 

# Define the input and output directories for Literate
literate_src_dir   = joinpath(@__DIR__, "src", "_literate")
literate_build_dir = joinpath(@__DIR__, "src", "_generated") 

# Process your Literate script
#Literate.markdown(
#    joinpath(literate_src_dir, "Example.jl"),
#    literate_build_dir;
#    size_threshold = 400,
#    size_threshold_warn = 250 * 1024,
#    execute = true,          # Runs the code and captures output
#    documenter = true,       # Handles Documenter-specific syntax
#    name = "Example",   # Output file will be usage_example.md
#    credit = true,          # Optional: Removes "Powered by Literate.jl" footnote
#)

makedocs(  sitename="Moose.jl",
           #remotes = nothing,
           format = Documenter.HTML(
                    prettyurls = get(ENV, "CI", nothing) == "true",
                    collapselevel = 2,
                    #assets = ["assets/logo.png"],
                    #assets  = String[ indigo_css_path,] ,
                    ),
          #          canonical = "https://masten-bourahma.github.io/Moose.jl/stable"), # Your docs URL
           modules = [Moose],
           pages   = [ "Home"             => "index.md", # Main page
                       "Context"          => "context.md",
                       "Basic usage"  => "_generated/Example.md",
                       "Usage examples"   => "examples.md",
                       "API reference"    => "api_reference.md"],
           doctest = :true,
           clean = true,
           checkdocs = :all)

                       # Deploy to GitHub Pages
deploydocs(
    repo = "github.com/masten-bourahma/Moose.jl.git", # Replace with your repo URL
    push_preview = true, # Deploy previews for pull requests
    devbranch = "dev",) # Branch where development happens

