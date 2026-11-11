module LuxMLPFixtureCases
using AbstractCosmologicalEmulators, Lux

function fixture(T, X)
    model = Lux.Chain(Lux.Dense(3 => 5, tanh), Lux.Dense(5 => 2))
    ps = (layer_1=(weight=reshape(T.(-7:7) ./ T(16), 5, 3), bias=T.(-2:2) ./ T(8)),
          layer_2=(weight=reshape(T.(-4:5) ./ T(8), 2, 5), bias=T[0.125, -0.25]))
    st = (layer_1=NamedTuple(), layer_2=NamedTuple())
    emu = AbstractCosmologicalEmulators.LuxEmulator(Model=model, Parameters=ps, States=st)
    x = reshape(X.(-2:3) ./ X(4), 3, 2)
    seed = [0.5 -0.25; 0.125 0.75]
    return emu, x, seed
end
end
