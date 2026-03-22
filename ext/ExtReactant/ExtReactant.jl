module ExtReactant

using AbstractCosmologicalEmulators
using Reactant

include("reactant_splines.jl")

export r_interval_indices,
       r_akima_slopes,
       r_akima_coefficients,
       r_akima_eval,
       r_akima_interpolation,
       r_akima_slopes_mat,
       r_akima_coefficients_mat,
       r_akima_eval_mat,
       r_akima_interpolation_mat,
       r_cubic_eval,
       r_cubic_eval_mat

end # module ExtReactant