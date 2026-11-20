# The decimated surfacelet transform on the device.
#
# Like `ns_surfacelet_adapt.jl`, no kernel is needed and none is written here.
# Every stage is an FFT, a pointwise broadcast against precomputed tables (the
# pyramid profiles, the hourglass masks, the checkerboard filter slices) or a
# block copy between array views, and the IRC fold is a broadcast between the
# two halves of a reshaped view.  All of that runs on a device array unchanged,
# so this extension moves the tables and has the workspace's buffers and FFT
# plans rebuilt next to them:
#
#     ws = make_workspace(t, CUDA.rand(Float32, dims...))   # or Adapt.adapt(CuArray, host_ws)
#     c  = analyze(t, x; workspace = ws)
#
# Building the tables is host work (the filter slices are evaluated in closed
# form point by point), so a device workspace is built on the host and adapted.

using Surfacelets: FourierPyramidLevel, IRCStage, _SurfScale, _surfacelet_workspace

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

# `memo` maps a host masks tuple or stages tuple to its device copy, so scales
# that share one on the host share it on the device too (and then share their
# buffers, which the workspace keys on the stages).
function _adapt_scale(to, sc::_SurfScale, memo::IdDict{Any, Any})
    masks = get!(() -> map(m -> Adapt.adapt(to, m), sc.masks), memo, sc.masks)
    stages = get!(() -> map(s -> [Adapt.adapt(to, st) for st in s], sc.stages), memo, sc.stages)
    return _SurfScale(sc.dims, sc.levels, masks, stages)
end

Adapt.adapt_structure(to, sc::_SurfScale) = _adapt_scale(to, sc, IdDict{Any, Any}())

function Adapt.adapt_structure(to, ws::SurfaceletWorkspace{Td, T}) where {Td, T}
    memo = IdDict{Any, Any}()
    pyramid = [Adapt.adapt(to, lv) for lv in ws.pyramid]
    scales = [_adapt_scale(to, sc, memo) for sc in ws.scales]
    proto = Adapt.adapt(to, Array{T, 3}(undef, 0, 0, 0))
    # The flags and task count matter only when `to` is the host after all.
    return _surfacelet_workspace(Td, ws.dims, ws.transform, pyramid, scales, proto, FFTW.ESTIMATE, ws.tasks)
end
