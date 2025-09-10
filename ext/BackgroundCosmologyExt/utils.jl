"""
    _transformed_weights(quadrature_rule, order, a, b)

Transforms the points and weights of a standard quadrature rule from the interval `[-1, 1]`
to a specified interval `[a, b]`.

This is a utility function used to adapt standard quadrature rules (like Gauss-Legendre)
for numerical integration over arbitrary intervals `[a, b]`.

# Arguments
- `quadrature_rule`: A function that takes an `order` and returns a tuple `(points, weights)`
                     for the standard interval `[-1, 1]`.
- `order`: The order of the quadrature rule (number of points).
- `a`: The lower bound of the target interval.
- `b`: The upper bound of the target interval.

# Returns
A tuple `(transformed_points, transformed_weights)` for the interval `[a, b]`.

# Details
The transformation is applied to the standard points `` x_i^{\\text{std}} `` and weights `` w_i^{\\text{std}} ``
obtained from the `quadrature_rule`:
- Transformed points: `` x_i = \\frac{b - a}{2} x_i^{\\text{std}} + \\frac{b + a}{2} ``
- Transformed weights: `` w_i = \\frac{b - a}{2} w_i^{\\text{std}} ``

# Formula
The transformation formulas are:
Points: `` x_i = \\frac{b - a}{2} x_i^{\\text{std}} + \\frac{b + a}{2} ``
Weights: `` w_i = \\frac{b - a}{2} w_i^{\\text{std}} ``

# See Also
- [`_r̃_z`](@ref): An example function that uses this utility for numerical integration.
"""
function _transformed_weights(quadrature_rule, order, a, b)
    x, w = quadrature_rule(order)
    x = (b - a) / 2.0 .* x .+ (b + a) / 2.0
    w = (b - a) / 2.0 .* w
    return x, w
end