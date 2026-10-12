# Parameter types for the three surfacelet transforms.
#
# The nonsubsampled transform's directional stage is parameterised by a
# Bernstein mapping order and a halfband split polynomial — the hourglass
# partition has no filter-pair description at all, so these parameters have no
# counterpart in a separable-lattice directional filter bank.  The decimated
# and semi-translation-invariant transforms share SurfBox's directional stage
# (hourglass diamond kernel + checkerboard filter order) instead.

# ── Nonsubsampled ─────────────────────────────────────────────────────────────

"""
    NSSurfaceletParams

Describes a nonsubsampled surfacelet decomposition.

- `J` — number of pyramid levels.
- `L_array` — directional depth per scale, length `J`.  Scale `j` produces
  `3 * 2^(L_array[j])` directional subbands.
- `lp_filters` — the pyramid filter pair (`CDF97` by default).
- `order` — Bernstein order of the hourglass mapping kernels; larger sharpens
  the boundaries between directional subbands.
- `poly` — order of the 1-D halfband split polynomial.
- `transition` — width of the overlap between neighbouring wedges, as a fraction
  of the wedge width.
- `tight` — when `true` (the default) the directional masks are `√W`, making the
  transform a Parseval frame in the directional stage; when `false` they are the
  weights themselves and the directional inverse is a bare sum.  See
  [`make_hourglass`](@ref).

```jldoctest
julia> using Surfacelets

julia> NSSurfaceletParams(J = 2, L_array = [1, 2])
NSSurfaceletParams(J=2, L_array=[1, 2], order=12, …)
```
"""
struct NSSurfaceletParams{T <: AbstractFloat}
    J::Int
    L_array::Vector{Int}
    lp_filters::FilterPair{T}
    order::Int
    poly::Int
    transition::T
    tight::Bool
end

function NSSurfaceletParams(;
        J::Int,
        L_array::AbstractVector{<:Integer},
        lp_filters::FilterPair = CDF97,
        order::Int = 12,
        poly::Int = 1,
        transition::Real = 1.0,
        tight::Bool = true
    )
    J >= 1 || throw(ArgumentError("J must be ≥ 1"))
    length(L_array) == J || throw(ArgumentError("length(L_array) must equal J"))
    all(>=(0), L_array) || throw(ArgumentError("L_array entries must be ≥ 0"))
    order >= 1 || throw(ArgumentError("order must be ≥ 1"))
    poly >= 0 || throw(ArgumentError("poly must be ≥ 0"))
    transition >= 0 || throw(ArgumentError("transition must be ≥ 0"))
    T = eltype(lp_filters)
    return NSSurfaceletParams{T}(
        J, collect(Int, L_array),
        FilterPair{T}(T.(lp_filters.h), T.(lp_filters.g)),
        order, poly, T(transition), tight
    )
end

Base.eltype(::NSSurfaceletParams{T}) where {T} = T

function Base.show(io::IO, p::NSSurfaceletParams)
    return print(io, "NSSurfaceletParams(J=$(p.J), L_array=$(p.L_array), order=$(p.order), …)")
end

# Convert to real filter precision `Tf` (no-op when already Tf).
_convert_params(::Type{Tf}, p::NSSurfaceletParams{Tf}) where {Tf} = p
_convert_params(::Type{Tf}, p::NSSurfaceletParams) where {Tf <: AbstractFloat} =
    NSSurfaceletParams{Tf}(
    p.J, p.L_array,
    FilterPair{Tf}(Tf.(p.lp_filters.h), Tf.(p.lp_filters.g)),
    p.order, p.poly, Tf(p.transition), p.tight
)

"""
    ns_surfacelet_nsubbands(params::NSSurfaceletParams, j::Int) -> Int

Number of directional subbands at scale `j`, `3 * 2^L_array[j]`.
"""
ns_surfacelet_nsubbands(p::NSSurfaceletParams, j::Int) = 3 * (1 << p.L_array[j])

# ── Decimated ─────────────────────────────────────────────────────────────────

"""
    SurfaceletParams

Describes a decimated 3-D surfacelet decomposition.

- `J` — number of scales.
- `L_array` — directional refinement per scale, length `J`.  An integer `L`
  gives `2^L` wedges in each of the three hourglasses (`3 · 2^L` subbands),
  split as evenly as possible between the two transverse axes.  A `3 × 3`
  integer matrix gives SurfBox's level matrix instead: row `K` holds the number
  of checkerboard stages along each axis inside hourglass `K` (the diagonal is
  ignored), so hourglass `K` has `2^(sum of row K off the diagonal)` wedges.
- `pyr_mode` — pyramid resampling: `1.5` (default, redundancy ≈ 4), `1`
  (≈ 6.4) or `2` (≈ 3.4).  Every scale size must divide accordingly; see
  [`make_surfacelet`](@ref).
- `bo` — order of the checkerboard filters, `4:2:18` (default 12).  Higher is
  sharper directionally, at no extra cost: the filters are evaluated in closed
  form in the frequency domain.
- `msize`, `beta`, `lambda` — the hourglass diamond kernel: its length (odd),
  Kaiser window `β`, and the even power it is raised to.

```jldoctest
julia> using Surfacelets

julia> SurfaceletParams(J = 2, L_array = [1, 2])
SurfaceletParams(J=2, L_array=[1, 2], pyr_mode=1.5, bo=12)
```
"""
struct SurfaceletParams{T <: AbstractFloat}
    J::Int
    L_array::Vector{Any}
    levels::Vector{NTuple{3, NTuple{3, Int}}}   # levels[j][K][m]; zero for m == K
    pyr_mode::Float64
    bo::Int
    msize::Int
    beta::T
    lambda::Int
end

function _surf_levels(L::Integer)
    la, lb = _wedge_levels(Int(L))
    return ntuple(
        K -> begin
            a, b = _transverse_axes(K)
            ntuple(m -> m == a ? la : (m == b ? lb : 0), 3)
        end, 3
    )
end

function _surf_levels(L::AbstractMatrix{<:Integer})
    size(L) == (3, 3) || throw(ArgumentError("a level matrix must be 3 × 3"))
    return ntuple(K -> ntuple(m -> m == K ? 0 : Int(L[K, m]), 3), 3)
end

function SurfaceletParams(;
        J::Int,
        L_array::AbstractVector,
        pyr_mode::Real = 1.5,
        bo::Int = 12,
        msize::Int = 15,
        beta::Real = 2.5,
        lambda::Int = 4,
        precision::Type{T} = Float64
    ) where {T <: AbstractFloat}
    J >= 1 || throw(ArgumentError("J must be ≥ 1"))
    length(L_array) == J || throw(ArgumentError("length(L_array) must equal J"))
    levels = map(_surf_levels, L_array)
    all(lv -> all(r -> all(>=(0), r), lv), levels) ||
        throw(ArgumentError("directional levels must be ≥ 0"))
    _pyramid_mode(pyr_mode)
    _cbd_ladder(bo)
    isodd(msize) && msize >= 3 || throw(ArgumentError("msize must be an odd integer ≥ 3"))
    beta > 0 || throw(ArgumentError("beta must be > 0"))
    iseven(lambda) && lambda > 0 || throw(ArgumentError("lambda must be a positive even integer"))
    return SurfaceletParams{T}(
        J, collect(Any, L_array), collect(levels), Float64(pyr_mode), bo, msize, T(beta), lambda
    )
end

Base.eltype(::SurfaceletParams{T}) where {T} = T

function Base.show(io::IO, p::SurfaceletParams)
    L = "[" * join((repr(l) for l in p.L_array), ", ") * "]"
    return print(io, "SurfaceletParams(J=$(p.J), L_array=$(L), pyr_mode=$(p.pyr_mode), bo=$(p.bo))")
end

"""
    surfacelet_nsubbands(p::SurfaceletParams, j::Int) -> Int

Number of directional subbands at scale `j`; `3 · 2^L` for an integer level `L`.
"""
surfacelet_nsubbands(p::SurfaceletParams, j::Int) = _nsubbands(p.levels[j])

# Subbands of one scale with level matrix `lv`: `2^(row sum)` per hourglass.
_nsubbands(lv::NTuple{3, NTuple{3, Int}}) = sum(K -> 1 << sum(lv[K]), 1:3)

_convert_sparams(::Type{T}, p::SurfaceletParams{T}) where {T} = p
_convert_sparams(::Type{T}, p::SurfaceletParams) where {T <: AbstractFloat} =
    SurfaceletParams{T}(p.J, p.L_array, p.levels, p.pyr_mode, p.bo, p.msize, T(p.beta), p.lambda)

# ── Semi-translation-invariant ────────────────────────────────────────────────

"""
    STISurfaceletParams

Describes a semi-translation-invariant (STI) surfacelet decomposition: a
nonsubsampled pyramid with the critically sampled directional stage of
[`surfacelet_forward`](@ref) at every scale.  Fields as in
[`SurfaceletParams`](@ref), without `pyr_mode`:

- `J` — number of scales; `L_array` — directional levels per scale (integers or
  `3 × 3` level matrices);
- `bo`, `msize`, `beta`, `lambda` — checkerboard filter order and hourglass
  kernel parameters.

```jldoctest
julia> using Surfacelets

julia> STISurfaceletParams(J = 3, L_array = [2, 2, 3])
STISurfaceletParams(J=3, L_array=[2, 2, 3], bo=12)
```
"""
struct STISurfaceletParams{T <: AbstractFloat}
    J::Int
    L_array::Vector{Any}
    levels::Vector{NTuple{3, NTuple{3, Int}}}
    bo::Int
    msize::Int
    beta::T
    lambda::Int
end

function STISurfaceletParams(;
        J::Int, L_array::AbstractVector, bo::Int = 12, msize::Int = 15,
        beta::Real = 2.5, lambda::Int = 4, precision::Type{T} = Float64
    ) where {T <: AbstractFloat}
    # Reuse the decimated transform's validation of the shared fields.
    p = SurfaceletParams(;
        J = J, L_array = L_array, bo = bo, msize = msize, beta = beta,
        lambda = lambda, precision = T
    )
    return STISurfaceletParams{T}(p.J, p.L_array, p.levels, p.bo, p.msize, p.beta, p.lambda)
end

Base.eltype(::STISurfaceletParams{T}) where {T} = T

function Base.show(io::IO, p::STISurfaceletParams)
    L = "[" * join((repr(l) for l in p.L_array), ", ") * "]"
    return print(io, "STISurfaceletParams(J=$(p.J), L_array=$(L), bo=$(p.bo))")
end

"""
    sti_surfacelet_nsubbands(p::STISurfaceletParams, j::Int) -> Int

Number of directional subbands at scale `j`.
"""
sti_surfacelet_nsubbands(p::STISurfaceletParams, j::Int) = _nsubbands(p.levels[j])

_convert_stiparams(::Type{T}, p::STISurfaceletParams{T}) where {T} = p
_convert_stiparams(::Type{T}, p::STISurfaceletParams) where {T <: AbstractFloat} =
    STISurfaceletParams{T}(p.J, p.L_array, p.levels, p.bo, p.msize, T(p.beta), p.lambda)
