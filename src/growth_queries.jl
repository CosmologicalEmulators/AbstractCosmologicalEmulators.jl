# Parent-owned names let the optional background and Mooncake extensions load
# in either order. Settings contain only fixed numerical controls and the RHS
# selected by the growth prescription, never active cosmological parameters.
struct _GrowthSolveSettings{F,S,R,A}
    rhs!::F
    tspan::S
    reltol::R
    abstol::A
end

function _growth_states_at end
function _growth_states_rule end
