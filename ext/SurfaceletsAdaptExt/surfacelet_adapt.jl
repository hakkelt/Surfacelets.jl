# Decimated and STI surfacelet transforms on the device.
#
# Like `ns_surfacelet_adapt.jl`, no kernel is needed and none is written here.
# Every stage is an FFT, a pointwise broadcast against precomputed tables (the
# pyramid profiles, the hourglass masks, the checkerboard filter slices) or a
# block copy between array views, and the IRC fold is a broadcast between the
# two halves of a reshaped view.  All of that runs on a device array unchanged,
# so this extension moves the tables and has the plan's buffers and FFT plans
# rebuilt next to them:
#
#     pl = Adapt.adapt(CuArray, make_surfacelet(params, dims, Float32))
#     c  = surfacelet_forward(CUDA.rand(Float32, dims...), params; plan = pl)
#
# Building the tables is host work (the filter slices are evaluated in closed
# form point by point), so a device plan is built on the host and adapted once;
# `surfacelet_forward` refuses a device volume without a plan rather than doing
# that implicitly.

using Surfacelets: SurfaceletPlan, SurfaceletCoefficients, FourierPyramidLevel, RaisedCosineLevel,
    IRCStage, _SurfScale, _surfacelet_plan

function Adapt.adapt_structure(to, lv::FourierPyramidLevel)
    return FourierPyramidLevel(
        lv.n, lv.nlow, lv.D,
        map(v -> Adapt.adapt(to, v), lv.idx),
        map(v -> Adapt.adapt(to, v), lv.idx_low),
        map(v -> Adapt.adapt(to, v), lv.prof),
    )
end

Adapt.adapt_structure(to, lv::RaisedCosineLevel) = RaisedCosineLevel(map(v -> Adapt.adapt(to, v), lv.prof))

function Adapt.adapt_structure(to, st::IRCStage)
    return IRCStage(st.m, st.b, map(a -> Adapt.adapt(to, a), st.ana), map(a -> Adapt.adapt(to, a), st.syn))
end

# `memo` maps a host masks tuple or stages tuple to its device copy, so scales
# that share one on the host share it on the device too (and then share their
# buffers, which the plan keys on the stages).
function _adapt_scale(to, sc::_SurfScale, memo::IdDict{Any, Any})
    masks = get!(() -> map(m -> Adapt.adapt(to, m), sc.masks), memo, sc.masks)
    stages = get!(() -> map(s -> [Adapt.adapt(to, st) for st in s], sc.stages), memo, sc.stages)
    return _SurfScale(sc.dims, sc.levels, masks, stages)
end

Adapt.adapt_structure(to, sc::_SurfScale) = _adapt_scale(to, sc, IdDict{Any, Any}())

function Adapt.adapt_structure(to, pl::SurfaceletPlan{T}) where {T}
    memo = IdDict{Any, Any}()
    pyramid = [Adapt.adapt(to, lv) for lv in pl.pyramid]
    scales = [_adapt_scale(to, sc, memo) for sc in pl.scales]
    proto = Adapt.adapt(to, Array{T, 3}(undef, 0, 0, 0))
    # The flags reach FFTW only when `to` is the host after all.
    return _surfacelet_plan(pl.dims, pl.params, pyramid, scales, proto, Surfacelets.FFTW.ESTIMATE)
end

function Adapt.adapt_structure(to, c::SurfaceletCoefficients)
    coarse = Adapt.adapt(to, c.coarse)
    A = typeof(coarse)
    return SurfaceletCoefficients(
        c.dims, coarse,
        [A[Adapt.adapt(to, s) for s in lvl] for lvl in c.subbands],
    )
end
