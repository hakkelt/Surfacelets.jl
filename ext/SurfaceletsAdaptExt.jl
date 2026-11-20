module SurfaceletsAdaptExt

# The whole 3-D stack on the device, in `Adapt.adapt_structure` methods alone.
#
# No kernel is needed and none is written here.  Both transforms are FFT plus a
# pointwise mask, and every mask is stored as broadcastable factors, so mask
# application, the complex real/imaginary split and the pyramid's
# frequency-domain multiply are already plain broadcasts that run on a device
# array unchanged.  What the device needs is the data moved across and the FFT
# plans rebuilt, which is exactly what these methods do — hence `Adapt` is the
# only weak dependency, with no GPUArrays or KernelAbstractions in sight.

using Surfacelets: Surfacelets, HourglassFB, NDFB, NSSurfaceletCoefficients,
    NSSurfaceletWorkspace, SurfaceletCoefficients, SurfaceletWorkspace
using Adapt: Adapt
using FFTW: FFTW, plan_rfft, plan_irfft

include("SurfaceletsAdaptExt/ns_surfacelet_adapt.jl")
include("SurfaceletsAdaptExt/surfacelet_adapt.jl")

# `make_workspace(t, x)` for a device array `x`: the host workspace `ws` moved
# to `x`'s bare array type (`CuArray`, not an element-typed one, so the tables
# keep their precision).
adapt_workspace(to, ws) = Adapt.adapt(to, ws)

end # module SurfaceletsAdaptExt
