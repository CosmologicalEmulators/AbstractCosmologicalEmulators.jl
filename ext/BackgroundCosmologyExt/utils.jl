function _transformed_weights(quadrature_rule, order, a, b)
    x, w = quadrature_rule(order)
    X = (b - a) / 2.0 .* x .+ (b + a) / 2.0
    W = (b - a) / 2.0 .* w
    return X, W
end
