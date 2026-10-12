# Decimated surfacelet transform — Lu & Do, "Multidimensional Directional
# Filter Banks and Surfacelets", IEEE TIP 16(4), 2007, Section VI, as
# implemented by their reference toolbox SurfBox (`Surfdec.m`, `Surfrec.m`).
#
# Three stages, all in the frequency domain:
#
#   1. A multiscale pyramid (`fourier_pyramid.jl`) splits the spectrum into
#      bandpass shells and a coarse lowpass, resampling by 3/2 or 2 per level.
#   2. At every scale, the undecimated "ritf" hourglass bank (`ritf.jl`) splits
#      the shell into three hourglasses, one per axis — redundancy 3, which Lu
#      & Do prove is the minimum for an hourglass partition above 2-D.
#   3. Inside each hourglass, iterated resampled checkerboard filter banks
#      (`irc.jl`) refine it into wedges *and decimate*, critically: each wedge
#      is stored on a lattice aligned with it.
#
# Stages 2 and 3 form the per-scale directional stage (`surfacelet_scale.jl`).
#
# So the redundancy is about 3 per scale times the geometric sum of the scale
# sizes — ≈ 4 for pyramid mode 1.5 — and does not grow with the number of
# directions.  The stages reproduce SurfBox to rounding error; `test_irc.jl`
# checks every stage and the whole transform against golden data generated
# with it.
#
# The checkerboard filters are biorthogonal, so the transform is a frame with
# perfect reconstruction but not a tight one: `surfacelet_inverse` is the
# inverse, not the adjoint, of `surfacelet_forward`.

"""
    SurfaceletPlan

Precomputed pyramid profiles, hourglass masks and checkerboard filter slices for
one volume size; build with [`make_surfacelet`](@ref) and pass as `plan =` to
reuse them across calls.
"""
struct SurfaceletPlan{T, L <: FourierPyramidLevel{T}, C <: _SurfScale{T}}
    dims::NTuple{3, Int}
    params::SurfaceletParams{T}
    pyramid::Vector{L}
    scales::Vector{C}
end

"""
    make_surfacelet(params::SurfaceletParams, dims, T=Float64) -> SurfaceletPlan

Precompute everything the transform needs for a `dims` volume.  The pyramid
must be able to resample every scale exactly, so with `pyr_mode = 1.5` each
dimension must be divisible by `3 · 2^J`, with `2` by `2^(J+1)`, and with `1` by
`2^J`; each scale must also allow its directional levels (axis length divisible
by `2^level`).
"""
function make_surfacelet(
        params::SurfaceletParams, dims::NTuple{3, Int}, ::Type{T} = Float64
    ) where {T <: AbstractFloat}
    p = _convert_sparams(T, params)
    pyramid = _fourier_pyramid(T, dims, p.J, p.pyr_mode)
    scales = map(1:p.J) do j
        n = pyramid[j].n
        masks = _ritf_masks(T, n; msize = p.msize, beta = p.beta, lambda = p.lambda)
        _SurfScale(n, p.levels[j], masks, _surf_stages(T, n, p.levels[j], p.bo))
    end
    return SurfaceletPlan(dims, p, pyramid, scales)
end

make_surfacelet(params::SurfaceletParams, dims::Vararg{Int, 3}) =
    make_surfacelet(params, dims)

"""
    surfacelet_redundancy(plan::SurfaceletPlan) -> Float64

Total stored coefficients divided by the number of input voxels.
"""
function surfacelet_redundancy(plan::SurfaceletPlan)
    total = prod(plan.pyramid[end].nlow) + _directional_count(plan.scales)
    return total / prod(plan.dims)
end

"""
    surfacelet_forward(volume, params::SurfaceletParams; plan=nothing) -> SurfaceletCoefficients

Decimated 3-D surfacelet transform.  Each subband is returned at its own reduced
resolution; the total coefficient count is about 4 times the input with the
default `pyr_mode = 1.5`, whatever the number of directions — see
[`surfacelet_redundancy`](@ref).

Not shift-invariant: that is what the decimation costs, and it is why
[`ns_surfacelet_forward`](@ref) still exists.

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(7), 48, 48, 48);

julia> p = SurfaceletParams(J = 2, L_array = [1, 2]);

julia> c = surfacelet_forward(x, p);

julia> maximum(abs, surfacelet_inverse(c, p) .- x) < 1e-10
true
```
"""
function surfacelet_forward(
        volume::AbstractArray{<:Real, 3}, params::SurfaceletParams;
        plan::Union{SurfaceletPlan, Nothing} = nothing
    )
    T = _filter_eltype(_data_eltype(volume))
    pl = _resolve_plan(make_surfacelet, plan, params, size(volume), T, volume)
    X = fft!(Complex{T}.(volume))   # in place on the converted copy
    subbands = Vector{Vector{typeof(similar(X, T, (0, 0, 0)))}}(undef, pl.params.J)
    for j in 1:pl.params.J
        low = _pyramid_split!(X, pl.pyramid[j])     # X now holds the bandpass
        subbands[j] = _surf_scale_forward(X, pl.scales[j])
        X = low
    end
    coarse = real.(ifft!(X))        # `X` is the last lowpass, owned here
    return SurfaceletCoefficients(pl.dims, coarse, subbands)
end

"""
    surfacelet_inverse(coeffs::SurfaceletCoefficients, params::SurfaceletParams; plan=nothing) -> volume

Invert [`surfacelet_forward`](@ref) exactly (to rounding error).
"""
function surfacelet_inverse(
        coeffs::SurfaceletCoefficients{T}, params::SurfaceletParams;
        plan::Union{SurfaceletPlan, Nothing} = nothing
    ) where {T}
    pl = _resolve_plan(make_surfacelet, plan, params, coeffs.dims, T, coeffs.coarse)
    p = pl.params
    length(coeffs.subbands) == p.J ||
        throw(ArgumentError("coefficients were produced for a different number of scales"))
    size(coeffs.coarse) == pl.pyramid[end].nlow ||
        throw(DimensionMismatch("coarse subband has size $(size(coeffs.coarse)), expected $(pl.pyramid[end].nlow)"))
    X = fft!(complex.(coeffs.coarse))
    for j in p.J:-1:1
        length(coeffs.subbands[j]) == surfacelet_nsubbands(p, j) ||
            throw(DimensionMismatch("scale $(j) has the wrong number of subbands"))
        H = _surf_scale_inverse(coeffs.subbands[j], pl.scales[j])
        X = _pyramid_merge!(H, X, pl.pyramid[j])
    end
    return real.(ifft!(X))
end

# The plan to use: `plan` after checking its size, or a fresh host plan from
# `make` (`make_surfacelet` / `make_sti_surfacelet`) when none was given.
function _resolve_plan(
        make::F, plan, params, dims::NTuple{3, Int}, ::Type{T}, proto::AbstractArray
    ) where {F, T}
    if plan === nothing
        # The plan is built on the host; a device plan is an `Adapt.adapt` of
        # it, built once rather than implicitly on every call.
        proto isa Array || throw(
            ArgumentError(
                "a device array needs an explicit plan: `plan = Adapt.adapt(ArrayType, $(nameof(make))(params, dims, T))`"
            )
        )
        return make(params, dims, T)
    end
    plan.dims == dims ||
        throw(DimensionMismatch("plan was built for $(plan.dims), got $(dims)"))
    return plan
end
