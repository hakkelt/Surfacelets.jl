# Semi-translation-invariant (STI) surfacelet transform: a nonsubsampled
# multiscale pyramid followed, at every scale, by the critically sampled
# directional stage of the decimated surfacelet transform (ritf hourglass bank +
# IRC filter banks, `surfacelet_scale.jl`).  It runs on the decimated
# transform's plan type and driver (`surfacelet.jl`) with the raised-cosine
# pyramid (`raised_cosine_pyramid.jl`) as stage 1.
#
# It sits between the two published transforms.  The nonsubsampled surfacelet
# transform keeps every directional subband at full size — exactly
# shift-invariant, redundancy `1 + Σ_j 3·2^{L_j}` (49 at J = 3, L = [2,2,3]).
# The decimated one resamples both stages — redundancy ≈ 4, no shift invariance.
# Here the pyramid is not resampled, so every scale is a full-size volume and the
# scale decomposition is exactly invariant under circular shifts, while the
# directional stage decimates critically inside each hourglass.  Each scale then
# costs exactly 3 volumes however many directions it has, and the redundancy is
# `1 + 3J` — 10 at J = 3.
#
# The name comes from the structure, not the construction.  It shares the
# structure of the semi-translation-invariant contourlet transform (STICT) of
# Eslami & Radha, "Translation-invariant contourlet transform and its
# application to image denoising", IEEE TIP 15(11), 2006: a nonsubsampled
# (shift-invariant) pyramid followed by a critically sampled directional stage.
# It is not a 3-D STICT.  The STICT's pyramid is the translation-invariant
# Laplacian pyramid — the decimated LP's polyphase channels for every lattice
# shift, four highpass channels per level, keeping the LP's aliasing.  This one
# has a single raised-cosine bandpass per scale (SurfBox's mode-1 band edges
# without the resampling, `raised_cosine_pyramid.jl`), and its directional stage
# is the IRC bank of Lu & Do.
#
# Neither Lu & Do's papers nor any later paper we know of describes this 3-D
# construction.  It composes two published stages, each validated separately:
# the directional stage against SurfBox (`test_irc.jl`), the pyramid by its
# tight-frame identity.

"""
    make_sti_surfacelet(params::STISurfaceletParams, dims, T=Float64; flags=FFTW.MEASURE) -> SurfaceletPlan

Precompute everything the STI transform needs for a `dims` volume.  Every
dimension must be even and divisible by `2^level` for each directional level
along it; the pyramid itself imposes nothing.  Scales with equal directional
levels share their tables and buffers.
"""
function make_sti_surfacelet(
        params::STISurfaceletParams, dims::NTuple{3, Int}, ::Type{T} = Float64; flags::Integer = FFTW.MEASURE
    ) where {T <: AbstractFloat}
    p = _convert_stiparams(T, params)
    pyramid = _raised_cosine_pyramid(T, dims, p.J)
    masks = _ritf_masks(T, dims; msize = p.msize, beta = p.beta, lambda = p.lambda)
    build(lv) = _SurfScale(dims, lv, masks, _surf_stages(T, dims, lv, p.bo))
    first_scale = build(p.levels[1])
    cache = Dict(p.levels[1] => first_scale)
    scales = map(1:p.J) do j
        lv = p.levels[j]
        get!(() -> build(lv), cache, lv)
    end
    return _surfacelet_plan(dims, p, pyramid, [sc for sc in scales], Array{T, 3}(undef, 0, 0, 0), flags)
end

make_sti_surfacelet(params::STISurfaceletParams, dims::Vararg{Int, 3}; kwargs...) =
    make_sti_surfacelet(params, dims; kwargs...)

_plan_params_type(::STISurfaceletParams) = STISurfaceletParams

"""
    sti_surfacelet_forward(volume, params::STISurfaceletParams; plan=nothing) -> SurfaceletCoefficients

Semi-translation-invariant surfacelet transform: nonsubsampled pyramid,
critically sampled directional stage.  Redundancy `1 + 3J`; the scale
decomposition is exactly shift-invariant, the directional subbands are not.  The
coarse subband is a full-size volume; directional subbands have the sizes of
[`surfacelet_forward`](@ref)'s at the finest scale, at every scale.

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(7), 32, 32, 32);

julia> p = STISurfaceletParams(J = 3, L_array = [1, 2, 2]);

julia> c = sti_surfacelet_forward(x, p);

julia> maximum(abs, sti_surfacelet_inverse(c, p) .- x) < 1e-10
true
```
"""
function sti_surfacelet_forward(volume::AbstractArray{<:Real, 3}, params::STISurfaceletParams; plan::Union{SurfaceletPlan, Nothing} = nothing)
    pl = _resolve_plan(make_sti_surfacelet, plan, params, volume)
    return _surfacelet_forward!(_surfacelet_coefficients(pl), volume, pl)
end

"""
    sti_surfacelet_forward!(coeffs, volume, params::STISurfaceletParams; plan=nothing) -> coeffs

[`sti_surfacelet_forward`](@ref) into the preallocated `coeffs`; allocates
nothing when `plan` is given.
"""
function sti_surfacelet_forward!(
        coeffs::SurfaceletCoefficients, volume::AbstractArray{<:Real, 3}, params::STISurfaceletParams;
        plan::Union{SurfaceletPlan, Nothing} = nothing
    )
    pl = _resolve_plan(make_sti_surfacelet, plan, params, volume)
    return _surfacelet_forward!(_check_coefficients(coeffs, pl), volume, pl)
end

"""
    sti_surfacelet_inverse(coeffs, params::STISurfaceletParams; plan=nothing) -> volume

Invert [`sti_surfacelet_forward`](@ref) exactly (to rounding error).
"""
function sti_surfacelet_inverse(coeffs::SurfaceletCoefficients, params::STISurfaceletParams; plan::Union{SurfaceletPlan, Nothing} = nothing)
    pl = _resolve_plan(make_sti_surfacelet, plan, params, coeffs)
    return _surfacelet_inverse!(similar(coeffs.coarse, pl.dims), coeffs, pl)
end

"""
    sti_surfacelet_inverse!(volume, coeffs, params::STISurfaceletParams; plan=nothing) -> volume

[`sti_surfacelet_inverse`](@ref) into the preallocated `volume`; allocates
nothing when `plan` is given.
"""
function sti_surfacelet_inverse!(
        volume::AbstractArray{<:Real, 3}, coeffs::SurfaceletCoefficients, params::STISurfaceletParams;
        plan::Union{SurfaceletPlan, Nothing} = nothing
    )
    pl = _resolve_plan(make_sti_surfacelet, plan, params, coeffs)
    size(volume) == pl.dims || throw(DimensionMismatch("volume has size $(size(volume)), expected $(pl.dims)"))
    return _surfacelet_inverse!(volume, coeffs, pl)
end
