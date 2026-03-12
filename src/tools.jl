module Tools
using FITSIO

"""
    lines 

# Fields
-
-
"""
struct lines


function lines()
    table = FITS("../data/lines_table.fits", "r")
    names =
    λrest = 


    close(table)
end
    new()
end



end