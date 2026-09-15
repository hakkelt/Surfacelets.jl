# Decimated 3-D surfacelet transform on the device.
#
# Like `nsurfacelet_gpu.jl`, no kernel is needed and none is written here.  The mask
# is a pointwise function of the three frequency indices, so filling it is a
# single broadcast over the per-axis tables; and each subband's support box is
# symmetric about the origin, so extracting it is a handful of contiguous block
# copies between array views rather than a per-voxel index remap.  Both run on a
# device array unchanged, so all this extension has to do is move the buffers
# and replan the FFTs:
#
#     pl = Adapt.adapt(CuArray, make_surfacelet(params, dims, Float32))
#     c  = surfacelet_forward(CUDA.rand(Float32, dims...), params; plan = pl)
#
# Planning scans every mask once to find its support, which is the expensive
# part and is pure host bookkeeping — the boxes are integers.  So a device plan
# is built on the host and adapted, never rebuilt per call, and
# `surfacelet_forward` refuses a device volume with no plan rather than
# silently doing that.
#
# The per-box-size FFT plans are not adapted either: they are built lazily on
# first use, so the adapted plan simply starts with an empty cache, which then
# fills with device plans.

using Surfacelets: SurfaceletPlan, SurfaceletCoefficients

function Adapt.adapt_structure(to, pl::SurfaceletPlan)
    # `spec` and `work` are scratch, so they are *allocated* on the device rather
    # than adapted: nothing needs copying, and adapting a complex array through
    # an element-typed destination (`Adapt.adapt(CuArray{Float32,3}, …)`) would
    # try to narrow `Complex{T}` to `T`.
    mask = Adapt.adapt(to, pl.mask)
    spec = similar(mask, Complex{eltype(mask)}, pl.dims)
    work = similar(mask, Complex{eltype(mask)}, pl.dims)
    fwd = Surfacelets._plan_fft_out(work, pl.flags)
    inv = Surfacelets._plan_ifft_in(spec, pl.flags)
    return SurfaceletPlan(
        pl.dims, pl.params,
        map(v -> Adapt.adapt(to, v), pl.wfreq),
        map(v -> Adapt.adapt(to, v), pl.signed),
        map(v -> Adapt.adapt(to, v), pl.negsigned),
        pl.radii, pl.boxes, pl.coarse_box, fwd, inv,
        Dict{NTuple{3, Int}, Tuple{Any, Any}}(), pl.flags, spec, work, mask,
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
