# Semi-translation-invariant (STI) surfacelet transform: a nonsubsampled
# multiscale pyramid followed, at every scale, by the critically sampled
# directional stage of the decimated surfacelet transform (ritf hourglass bank +
# IRC filter banks, `surfacelet_scale.jl`).
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
    STISurfaceletPlan

Pyramid profiles, hourglass masks and checkerboard filter slices for one volume
size; build with [`make_sti_surfacelet`](@ref).
"""
struct STISurfaceletPlan{T, V <: AbstractVector{T}, C <: _SurfScale{T}, P <: AbstractArray{T, 3}}
    dims::NTuple{3, Int}
    params::STISurfaceletParams{T}
    low::Vector{NTuple{3, V}}       # ℓ_j per axis, full-length
    scales::Vector{C}
    proto::P
end

"""
    make_sti_surfacelet(params::STISurfaceletParams, dims, T=Float64) -> STISurfaceletPlan

Precompute everything the STI transform needs for a `dims` volume.  Every
dimension must be even and divisible by `2^level` for each directional level
along it; the pyramid itself imposes nothing.
"""
function make_sti_surfacelet(
        params::STISurfaceletParams, dims::NTuple{3, Int}, ::Type{T} = Float64
    ) where {T <: AbstractFloat}
    p = params isa STISurfaceletParams{T} ? params :
        STISurfaceletParams{T}(params.J, params.L_array, params.levels, params.bo, params.msize, T(params.beta), params.lambda)
    low = [ntuple(d -> _sti_profile(T, dims[d], 1 // 2^j, 1 // (3 * 2^j)), 3) for j in 1:p.J]
    masks = _ritf_masks(T, dims; msize = p.msize, beta = p.beta, lambda = p.lambda)
    cache = Dict{NTuple{3, NTuple{3, Int}}, Any}()
    scales = map(1:p.J) do j
        lv = p.levels[j]
        stages = get!(cache, lv) do
            ntuple(3) do K
                half = ntuple(d -> d == K ? dims[K] ÷ 2 + 1 : dims[d], 3)
                _irc_stages(T, half, dims[K], K, lv[K], p.bo)
            end
        end
        _SurfScale(dims, lv, masks, stages)
    end
    return STISurfaceletPlan(dims, p, low, scales, Array{T, 3}(undef, 0, 0, 0))
end

make_sti_surfacelet(params::STISurfaceletParams, dims::Vararg{Int, 3}) =
    make_sti_surfacelet(params, dims)

"""
    sti_surfacelet_redundancy(plan::STISurfaceletPlan) -> Float64

Total stored coefficients divided by the number of input voxels — exactly
`1 + 3J`, whatever the directional levels.
"""
function sti_surfacelet_redundancy(plan::STISurfaceletPlan)
    total = prod(plan.dims)
    for sc in plan.scales, K in 1:3
        counts, bsize = _surf_blocks(sc, K)
        total += prod(counts) * prod(ntuple(d -> d == K ? sc.dims[K] : bsize[d], 3))
    end
    return total / prod(plan.dims)
end

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
function sti_surfacelet_forward(
        volume::AbstractArray{<:Real, 3}, params::STISurfaceletParams;
        plan::Union{STISurfaceletPlan, Nothing} = nothing
    )
    T = _filter_eltype(_data_eltype(volume))
    pl = _sti_surfacelet_plan(plan, params, size(volume), T, volume)
    cur = fft(Complex{T}.(volume))
    subbands = Vector{Vector{typeof(similar(cur, T, (0, 0, 0)))}}(undef, pl.params.J)
    for j in 1:pl.params.J
        L = _sti_lowpass(pl.low[j])
        band = cur .* sqrt.(1 .- L .^ 2)
        cur .*= L
        subbands[j] = _surf_scale_forward(band, pl.scales[j])
    end
    return SurfaceletCoefficients(pl.dims, real.(ifft(cur)), subbands)
end

"""
    sti_surfacelet_inverse(coeffs, params::STISurfaceletParams; plan=nothing) -> volume

Invert [`sti_surfacelet_forward`](@ref) exactly (to rounding error).
"""
function sti_surfacelet_inverse(
        coeffs::SurfaceletCoefficients{T}, params::STISurfaceletParams;
        plan::Union{STISurfaceletPlan, Nothing} = nothing
    ) where {T}
    pl = _sti_surfacelet_plan(plan, params, coeffs.dims, T, coeffs.coarse)
    p = pl.params
    length(coeffs.subbands) == p.J ||
        throw(ArgumentError("coefficients were produced for a different number of scales"))
    size(coeffs.coarse) == pl.dims ||
        throw(DimensionMismatch("coarse subband has size $(size(coeffs.coarse)), expected $(pl.dims)"))
    cur = fft(complex.(coeffs.coarse))
    for j in p.J:-1:1
        length(coeffs.subbands[j]) == sti_surfacelet_nsubbands(p, j) ||
            throw(DimensionMismatch("scale $(j) has the wrong number of subbands"))
        L = _sti_lowpass(pl.low[j])
        band = _surf_scale_inverse(coeffs.subbands[j], pl.scales[j])
        cur .= cur .* L .+ band .* sqrt.(1 .- L .^ 2)
    end
    return real.(ifft(cur))
end

function _sti_surfacelet_plan(
        plan::Union{STISurfaceletPlan, Nothing}, params::STISurfaceletParams,
        dims::NTuple{3, Int}, ::Type{T}, proto::AbstractArray
    ) where {T}
    if plan === nothing
        proto isa Array || throw(
            ArgumentError(
                "a device array needs an explicit plan: `plan = Adapt.adapt(ArrayType, make_sti_surfacelet(params, dims, T))`"
            )
        )
        return make_sti_surfacelet(params, dims, T)
    end
    plan.dims == dims ||
        throw(DimensionMismatch("plan was built for $(plan.dims), got $(dims)"))
    return plan
end
