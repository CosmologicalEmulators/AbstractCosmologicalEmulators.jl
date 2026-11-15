# Private selectors live in the parent so the two extensions can load in
# either order. These select fixed physical tables, not trainable splines.
struct _NeutrinoF end
struct _NeutrinoDFdy end
const _NeutrinoLookup = Union{_NeutrinoF,_NeutrinoDFdy}

# Implementations and table initialization belong to BackgroundCosmologyExt.
function _neutrino_table end
_neutrino_lookup(tag::_NeutrinoLookup, y::T) where {T} = _neutrino_table(tag)(y)::T

function _neutrino_lookup_slope(table, y)
    j = _akima_find_interval(table.t, y)
    w = y - table.t[j]
    return (3 * table.d[j] * w + 2 * table.c[j]) * w + table.b[j]
end
