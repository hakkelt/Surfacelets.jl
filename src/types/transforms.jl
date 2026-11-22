# The two surfacelet transforms as transform objects.
#
# The nonsubsampled transform's directional stage is parameterized by a
# Bernstein mapping order and a halfband split polynomial order — the hourglass
# partition has no filter-pair description at all, so these parameters have no
# counterpart in a separable-lattice directional filter bank.  The decimated
# transform uses SurfBox's directional stage (hourglass diamond kernel +
# checkerboard filter order) instead.
#
# A transform holds parameters only; workspaces convert them to the data's
# precision.

# ── Nonsubsampled ─────────────────────────────────────────────────────────────

"""
    NSSurfacelet(; scales, direction_levels, bernstein_order = 12, halfband_order = 1,
                 transition = 1.0, tight = true)

The nonsubsampled surfacelet transform.  Its multiscale stage is Lu and Do's
raised-cosine frequency-domain pyramid without its downsampling, a tight frame;
see the "Theory" page.  Every subband has the size of the data, so the
transform is invariant under circular shifts.

- `scales` — number of pyramid levels (`J` in the papers).
- `direction_levels` — directional depth per scale (`ℓⱼ`), length `scales`,
  from the coarsest scale to the finest.  Scale `j` produces
  `3 * 2^(direction_levels[j])` directional subbands.
- `bernstein_order` — order `N` of the Bernstein mapping kernels of the
  hourglass bank; larger sharpens the boundaries between directional subbands.
- `halfband_order` — order `m` of the 1-D halfband split polynomial.
- `transition` — the half-width of the soft step at each wedge boundary, as a
  fraction of the wedge width: `0` gives hard boundaries, `0.5` lets the two
  steps of a wedge meet at its center, and `1` (the default) ramps across a whole
  wedge on either side of each boundary, so no wedge has a flat passband — a
  plane wave at the center of a wedge keeps about 71 % of its energy there at
  two directional levels.
- `tight` — when `true` (the default) the directional masks are `√W`, which
  makes the whole transform a Parseval frame: `‖Sx‖ = ‖x‖` and the inverse is
  the adjoint.  When `false` they are the weights themselves and the
  directional inverse is a bare sum.  See [`Surfacelets.make_hourglass`](@ref).

Give `scales`, `direction_levels` or both.  `scales` alone gives parabolic
scaling: one directional level at the two coarsest scales, one more every
second scale toward the finest.  `direction_levels` alone gives
`scales = length(direction_levels)`; when both are given they must agree.

Run it with [`analyze`](@ref) and `synthesize`; repeated calls on one
size reuse a [`NSSurfaceletWorkspace`](@ref).  The transform also streams:
[`analyze(f, t, x)`](@ref analyze) hands the subbands to `f` one at a time.
Real and complex data are accepted; complex data is transformed as two real
channels, so `analyze(t, x + im * y) == complex.(analyze(t, x), analyze(t, y))`
exactly.

```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(5), 16, 16, 16);

julia> t = NSSurfacelet(direction_levels = [1, 2], bernstein_order = 6);

julia> c = analyze(t, x);

julia> nsubbands(t, size(x))   # the lowpass, then 6 + 12 directional subbands, coarse to fine
19

julia> maximum(abs, synthesize(t, c) .- x) < 1e-10
true

julia> NSSurfacelet(scales = 4)
NSSurfacelet(scales = 4, direction_levels = [1, 1, 2, 2], bernstein_order = 12, halfband_order = 1, transition = 1.0, tight = true)
```

# References
1. Y. M. Lu and M. N. Do, "Multidimensional Directional Filter Banks and
   Surfacelets", *IEEE Transactions on Image Processing*, 16(4):918–931, 2007.
   [doi:10.1109/TIP.2007.891785](https://doi.org/10.1109/TIP.2007.891785)
2. Y. M. Lu and M. N. Do, "A Mapping-Based Design for Nonsubsampled Hourglass
   Filter Banks in Arbitrary Dimensions", *IEEE Transactions on Signal Processing*,
   56(4):1466–1478, 2008.
   [doi:10.1109/TSP.2007.909340](https://doi.org/10.1109/TSP.2007.909340)
3. A. L. da Cunha, J. Zhou, and M. N. Do, "The Nonsubsampled Contourlet Transform:
   Theory, Design, and Applications", *IEEE Transactions on Image Processing*,
   15(10):3089–3101, 2006.
   [doi:10.1109/TIP.2006.877507](https://doi.org/10.1109/TIP.2006.877507)
"""
struct NSSurfacelet <: AbstractMultiscaleTransform
    scales::Int
    direction_levels::Vector{Int}
    bernstein_order::Int
    halfband_order::Int
    transition::Float64
    tight::Bool
    function NSSurfacelet(
            scales::Integer, direction_levels::AbstractVector{<:Integer}, bernstein_order::Integer,
            halfband_order::Integer, transition::Real, tight::Bool
        )
        _check_scales(scales, direction_levels)
        all(>=(0), direction_levels) || throw(ArgumentError("direction_levels entries must be ≥ 0, got $(direction_levels)"))
        bernstein_order >= 1 || throw(ArgumentError("bernstein_order must be ≥ 1, got $(bernstein_order)"))
        halfband_order >= 0 || throw(ArgumentError("halfband_order must be ≥ 0, got $(halfband_order)"))
        transition >= 0 || throw(ArgumentError("transition must be ≥ 0, got $(transition)"))
        # `+ 0.0` turns -0.0 into 0.0, so equal transforms hash alike.
        return new(scales, collect(Int, direction_levels), bernstein_order, halfband_order, Float64(transition) + 0.0, tight)
    end
end

function NSSurfacelet(;
        scales::Union{Integer, Nothing} = nothing, direction_levels::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        bernstein_order::Integer = 12, halfband_order::Integer = 1, transition::Real = 1.0, tight::Bool = true
    )
    J, levels = _scales_and_levels(scales, direction_levels)
    return NSSurfacelet(J, levels, bernstein_order, halfband_order, transition, tight)
end

Base.:(==)(a::NSSurfacelet, b::NSSurfacelet) =
    a.scales == b.scales && a.direction_levels == b.direction_levels && a.bernstein_order == b.bernstein_order &&
    a.halfband_order == b.halfband_order && a.transition == b.transition && a.tight == b.tight
Base.hash(t::NSSurfacelet, h::UInt) =
    hash(
    t.tight, hash(
        t.transition, hash(
            t.halfband_order, hash(t.bernstein_order, hash(t.direction_levels, hash(t.scales, hash(:NSSurfacelet, h))))
        )
    )
)

function Base.show(io::IO, t::NSSurfacelet)
    return print(
        io, "NSSurfacelet(scales = ", t.scales, ", direction_levels = ", t.direction_levels,
        ", bernstein_order = ", t.bernstein_order, ", halfband_order = ", t.halfband_order,
        ", transition = ", t.transition, ", tight = ", t.tight, ")"
    )
end

# The scale count and the per-scale directional levels from either or both.
function _scales_and_levels(scales, direction_levels)
    if direction_levels === nothing
        scales === nothing && throw(ArgumentError("give `scales`, `direction_levels` or both"))
        return scales, _parabolic_levels(scales)
    end
    return something(scales, length(direction_levels)), direction_levels
end

# Parabolic scaling, coarsest scale first: one directional level at the two
# coarsest scales and one more every second scale toward the finest, so that
# the number of directions doubles at every other scale.
_parabolic_levels(scales::Integer) = [1 + (j - 1) ÷ 2 for j in 1:scales]

function _check_scales(scales::Integer, direction_levels::AbstractVector)
    scales >= 1 || throw(ArgumentError("scales must be ≥ 1, got $(scales)"))
    length(direction_levels) == scales ||
        throw(ArgumentError("length(direction_levels) must equal scales = $(scales), got $(length(direction_levels))"))
    return nothing
end

# Directional subbands of scale `j`.
_ns_nbands(t::NSSurfacelet, j::Int) = 3 * (1 << t.direction_levels[j])

# ── Decimated ─────────────────────────────────────────────────────────────────

"""
    Surfacelet(; scales, direction_levels, pyramid_ratio = 1.5, checkerboard_order = 12,
               diamond_length = 15, kaiser_beta = 2.5, diamond_power = 4)

Lu and Do's decimated 3-D surfacelet transform, reproducing their reference
toolbox SurfBox to rounding error: a 3/2-resampling Fourier pyramid, SurfBox's
hourglass bank, and iterated resampled checkerboard filter banks that refine
each hourglass into wedges while decimating critically.  About 4 coefficients
per voxel with the default `pyramid_ratio`, whatever the number of directions
(see [`surfacelet_redundancy`](@ref)); not shift-invariant.  The SurfBox name
of each parameter is given in parentheses.

- `scales` — number of scales (the length of `Lev_array`).
- `direction_levels` — directional refinement per scale, length `scales`,
  from the coarsest scale to the finest (SurfBox's `Lev_array` lists the
  scales finest first, so this is `reverse(Lev_array)`).  An integer `L` gives `2^L` wedges in each of the three
  hourglasses (`3 · 2^L` subbands), split as evenly as possible between the
  two transverse axes.  A `3 × 3` integer matrix gives SurfBox's level matrix
  instead: row `K` holds the number of checkerboard stages along each axis
  inside hourglass `K` (the diagonal is ignored), so hourglass `K` has
  `2^(sum of row K off the diagonal)` wedges.
- `pyramid_ratio` — resampling factor of the pyramid (`Pyr_mode`): `1.5`
  (default, redundancy ≈ 4), `1` (≈ 6.4) or `2` (≈ 3.4).  Every scale size
  must divide accordingly; see
  [`make_workspace`](@ref make_workspace(::Surfacelet, ::AbstractArray)).
- `checkerboard_order` — order of the checkerboard filters (`bo`), one of
  [`checkerboard_orders()`](@ref checkerboard_orders) (default 12).  Higher is
  sharper directionally, at no extra cost: the filters are evaluated in closed
  form in the frequency domain.
- `diamond_length`, `kaiser_beta`, `diamond_power` — the hourglass diamond
  kernel: its length (`msize`, odd), the `β` of its Kaiser window (`beta`),
  and the even power `λ` it is raised to (`lambda`).

Give `scales`, `direction_levels` or both, as for [`NSSurfacelet`](@ref):
`scales` alone gives parabolic scaling, `direction_levels` alone gives
`scales = length(direction_levels)`.

The checkerboard filters are biorthogonal, so the transform is a frame with
perfect reconstruction but not a tight one: `synthesize` is its
inverse, not its adjoint.  Real and complex data are accepted; complex data is
transformed as two real channels, so the transform is exactly real-linear.

```jldoctest
julia> using Surfacelets, Random

julia> Surfacelet(direction_levels = [1, 2])
Surfacelet(scales = 2, direction_levels = [1, 2], pyramid_ratio = 1.5, checkerboard_order = 12, diamond_length = 15, kaiser_beta = 2.5, diamond_power = 4)

julia> x = randn(Xoshiro(7), 48, 48, 48);

julia> t = Surfacelet(direction_levels = [1, 2]);

julia> maximum(abs, synthesize(t, analyze(t, x)) .- x) < 1e-10
true
```

# References
1. Y. M. Lu and M. N. Do, "Multidimensional Directional Filter Banks and
   Surfacelets", *IEEE Transactions on Image Processing*, 16(4):918–931, 2007.
   [doi:10.1109/TIP.2007.891785](https://doi.org/10.1109/TIP.2007.891785)
"""
struct Surfacelet <: AbstractMultiscaleTransform
    scales::Int
    direction_levels::Vector{Union{Int, Matrix{Int}}}   # as given: integer levels or 3 × 3 level matrices
    levels::Vector{NTuple{3, NTuple{3, Int}}}   # levels[j][K][m]; zero for m == K
    pyramid_ratio::Float64
    checkerboard_order::Int
    diamond_length::Int
    kaiser_beta::Float64
    diamond_power::Int
    function Surfacelet(
            scales::Integer, direction_levels::AbstractVector, pyramid_ratio::Real, checkerboard_order::Integer,
            diamond_length::Integer, kaiser_beta::Real, diamond_power::Integer
        )
        _check_scales(scales, direction_levels)
        levels = map(_surf_levels, direction_levels)
        all(lv -> all(r -> all(>=(0), r), lv), levels) ||
            throw(ArgumentError("directional levels must be ≥ 0"))
        _pyramid_mode(pyramid_ratio)
        _cbd_ladder(Int(checkerboard_order))
        isodd(diamond_length) && diamond_length >= 3 ||
            throw(ArgumentError("diamond_length must be an odd integer ≥ 3, got $(diamond_length)"))
        kaiser_beta > 0 || throw(ArgumentError("kaiser_beta must be > 0, got $(kaiser_beta)"))
        iseven(diamond_power) && diamond_power > 0 ||
            throw(ArgumentError("diamond_power must be a positive even integer, got $(diamond_power)"))
        entries = Union{Int, Matrix{Int}}[_level_entry(L) for L in direction_levels]
        return new(
            scales, entries, collect(NTuple{3, NTuple{3, Int}}, levels), pyramid_ratio, checkerboard_order,
            diamond_length, kaiser_beta, diamond_power
        )
    end
end

function Surfacelet(;
        scales::Union{Integer, Nothing} = nothing, direction_levels::Union{AbstractVector, Nothing} = nothing,
        pyramid_ratio::Real = 1.5, checkerboard_order::Integer = 12, diamond_length::Integer = 15,
        kaiser_beta::Real = 2.5, diamond_power::Integer = 4
    )
    J, levels = _scales_and_levels(scales, direction_levels)
    return Surfacelet(J, levels, pyramid_ratio, checkerboard_order, diamond_length, kaiser_beta, diamond_power)
end

_level_entry(L::Integer) = Int(L)
_level_entry(L::AbstractMatrix{<:Integer}) = Matrix{Int}(L)
_level_entry(L) = throw(ArgumentError("a directional level must be an integer or a 3 × 3 integer matrix, got $(repr(L))"))

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
    size(L) == (3, 3) || throw(ArgumentError("a level matrix must be 3 × 3, got size $(size(L))"))
    return ntuple(K -> ntuple(m -> m == K ? 0 : Int(L[K, m]), 3), 3)
end

_surf_levels(L) = _level_entry(L)

# `levels` is derived from `direction_levels`, so it takes no part in equality.
Base.:(==)(a::Surfacelet, b::Surfacelet) =
    a.scales == b.scales && a.direction_levels == b.direction_levels && a.pyramid_ratio == b.pyramid_ratio &&
    a.checkerboard_order == b.checkerboard_order && a.diamond_length == b.diamond_length &&
    a.kaiser_beta == b.kaiser_beta && a.diamond_power == b.diamond_power
Base.hash(t::Surfacelet, h::UInt) =
    hash(
    t.diamond_power, hash(
        t.kaiser_beta, hash(
            t.diamond_length, hash(
                t.checkerboard_order,
                hash(t.pyramid_ratio, hash(t.direction_levels, hash(t.scales, hash(:Surfacelet, h))))
            )
        )
    )
)

function Base.show(io::IO, t::Surfacelet)
    L = "[" * join((repr(l) for l in t.direction_levels), ", ") * "]"
    return print(
        io, "Surfacelet(scales = ", t.scales, ", direction_levels = ", L, ", pyramid_ratio = ", t.pyramid_ratio,
        ", checkerboard_order = ", t.checkerboard_order, ", diamond_length = ", t.diamond_length,
        ", kaiser_beta = ", t.kaiser_beta, ", diamond_power = ", t.diamond_power, ")"
    )
end

# Subbands of one scale with level matrix `lv`: `2^(row sum)` per hourglass.
_nsubbands(lv::NTuple{3, NTuple{3, Int}}) = sum(K -> 1 << sum(lv[K]), 1:3)
