# Decimated 3-D surfacelet transform on the device.
#
# Like `ns_surfacelet_adapt.jl`, no kernel is needed and none is written here.
# Every stage is an FFT, a pointwise broadcast against precomputed tables (the
# pyramid profile, the hourglass masks, the checkerboard filter slices) or a
# block copy between array views, and the IRC fold is a broadcast between the
# two halves of a reshaped view.  All of that runs on a device array unchanged,
# so this extension only moves the tables:
#
#     pl = Adapt.adapt(CuArray, make_surfacelet(params, dims, Float32))
#     c  = surfacelet_forward(CUDA.rand(Float32, dims...), params; plan = pl)
#
# Building the tables is host work (the filter slices are evaluated in closed
# form point by point), so a device plan is built on the host and adapted once;
# `surfacelet_forward` refuses a device volume without a plan rather than doing
# that implicitly.

using Surfacelets: SurfaceletPlan, SurfaceletCoefficients, FourierPyramidLevel, IRCStage, _SurfScale,
    STISurfaceletPlan

function Adapt.adapt_structure(to, lv::FourierPyramidLevel)
    return FourierPyramidLevel(
        lv.n, lv.nlow, lv.D,
        map(v -> Adapt.adapt(to, v), lv.idx),
        map(v -> Adapt.adapt(to, v), lv.idx_low),
        map(v -> Adapt.adapt(to, v), lv.prof),
    )
end

function Adapt.adapt_structure(to, st::IRCStage)
    return IRCStage(st.m, st.b, map(a -> Adapt.adapt(to, a), st.ana), map(a -> Adapt.adapt(to, a), st.syn))
end

function Adapt.adapt_structure(to, sc::_SurfScale)
    stages = map(s -> [Adapt.adapt(to, st) for st in s], sc.stages)
    return _SurfScale(sc.dims, sc.levels, map(m -> Adapt.adapt(to, m), sc.masks), stages)
end

function Adapt.adapt_structure(to, pl::SurfaceletPlan)
    return SurfaceletPlan(
        pl.dims, pl.params,
        [Adapt.adapt(to, lv) for lv in pl.pyramid],
        [Adapt.adapt(to, sc) for sc in pl.scales],
        Adapt.adapt(to, pl.proto),
    )
end

# The STI transform shares the directional stage; its pyramid is three
# full-length 1-D profiles per scale.
function Adapt.adapt_structure(to, pl::STISurfaceletPlan)
    return STISurfaceletPlan(
        pl.dims, pl.params,
        [map(v -> Adapt.adapt(to, v), prof) for prof in pl.low],
        [Adapt.adapt(to, sc) for sc in pl.scales],
        Adapt.adapt(to, pl.proto),
    )
end

function Adapt.adapt_structure(to, c::SurfaceletCoefficients)
    coarse = Adapt.adapt(to, c.coarse)
    A = typeof(coarse)
    return SurfaceletCoefficients(
        c.dims, coarse,
        [A[Adapt.adapt(to, s) for s in lvl] for lvl in c.subbands],
    )
end
