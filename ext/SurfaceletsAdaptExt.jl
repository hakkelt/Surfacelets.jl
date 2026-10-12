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

using Surfacelets: Surfacelets, Conv3DFFTPlan, HourglassFB, NDFB, NSSurfaceletCoefficients,
    NSSurfaceletWorkspace, STISurfaceletPlan, SurfaceletCoefficients, SurfaceletPlan
using Adapt: Adapt

include("SurfaceletsAdaptExt/ns_surfacelet_adapt.jl")
include("SurfaceletsAdaptExt/surfacelet_adapt.jl")

end # module SurfaceletsAdaptExt
