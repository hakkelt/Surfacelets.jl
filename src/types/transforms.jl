# The two surfacelet transforms as transform objects.
#
# The nonsubsampled transform's directional stage is parameterised by a
# Bernstein mapping order and a halfband split polynomial — the hourglass
# partition has no filter-pair description at all, so these parameters have no
# counterpart in a separable-lattice directional filter bank.  The decimated
# transform uses SurfBox's directional stage (hourglass diamond kernel +
# checkerboard filter order) instead.
#
# A transform holds parameters only; workspaces convert them to the data's
# precision.

# ── Nonsubsampled ─────────────────────────────────────────────────────────────

"""
    NSSurfacelet(; J, L_array, order = 12, poly = 1, transition = 1.0, tight = true)

The nonsubsampled surfacelet transform.  Its multiscale stage is Lu and Do's
raised-cosine frequency-domain pyramid without its downsampling, a tight frame;
see the "Theory" page.  Every subband has the size of the data, so the
transform is invariant under circular shifts.

- `J` — number of pyramid levels.
- `L_array` — directional depth per scale, length `J`.  Scale `j` produces
  `3 * 2^(L_array[j])` directional subbands.
- `order` — Bernstein order of the hourglass mapping kernels; larger sharpens
  the boundaries between directional subbands.
- `poly` — order of the 1-D halfband split polynomial.
- `transition` — width of the overlap between neighbouring wedges, as a fraction
  of the wedge width.
- `tight` — when `true` (the default) the directional masks are `√W`, which
  makes the whole transform a Parseval frame: `‖Sx‖ = ‖x‖` and the inverse is
  the adjoint.  When `false` they are the weights themselves and the
  directional inverse is a bare sum.  See [`Surfacelets.make_hourglass`](@ref).

Run it with [`analyze`](@ref) and `synthesize`; repeated calls on one
size reuse a [`NSSurfaceletWorkspace`](@ref).  The transform also streams:
[`analyze(f, t, x)`](@ref analyze) hands the subbands to `f` one at a time.
Real and complex data are accepted; complex data is transformed as two real
channels, so `analyze(t, x + im * y) == complex.(analyze(t, x), analyze(t, y))`
exactly.

```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(5), 16, 16, 16);

julia> t = NSSurfacelet(J = 2, L_array = [1, 2], order = 6);

julia> c = analyze(t, x);

julia> nsubbands(t, size(x))   # the lowpass, then 6 + 12 directional subbands
19

julia> maximum(abs, synthesize(t, c) .- x) < 1e-10
true

julia> NSSurfacelet(J = 2, L_array = [1, 2])
NSSurfacelet(J = 2, L_array = [1, 2], order = 12, poly = 1, transition = 1.0, tight = true)
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
    J::Int
    L_array::Vector{Int}
    order::Int
    poly::Int
    transition::Float64
    tight::Bool
    function NSSurfacelet(J::Integer, L_array::AbstractVector{<:Integer}, order::Integer, poly::Integer, transition::Real, tight::Bool)
        J >= 1 || throw(ArgumentError("J must be ≥ 1, got $(J)"))
        length(L_array) == J || throw(ArgumentError("length(L_array) must equal J = $(J), got $(length(L_array))"))
        all(>=(0), L_array) || throw(ArgumentError("L_array entries must be ≥ 0, got $(L_array)"))
        order >= 1 || throw(ArgumentError("order must be ≥ 1, got $(order)"))
        poly >= 0 || throw(ArgumentError("poly must be ≥ 0, got $(poly)"))
        transition >= 0 || throw(ArgumentError("transition must be ≥ 0, got $(transition)"))
        return new(J, collect(Int, L_array), order, poly, transition, tight)
    end
end

NSSurfacelet(; J::Integer, L_array::AbstractVector{<:Integer}, order::Integer = 12, poly::Integer = 1, transition::Real = 1.0, tight::Bool = true) =
    NSSurfacelet(J, L_array, order, poly, transition, tight)

Base.:(==)(a::NSSurfacelet, b::NSSurfacelet) =
    a.J == b.J && a.L_array == b.L_array && a.order == b.order && a.poly == b.poly &&
    a.transition == b.transition && a.tight == b.tight
Base.hash(t::NSSurfacelet, h::UInt) =
    hash(t.tight, hash(t.transition, hash(t.poly, hash(t.order, hash(t.L_array, hash(t.J, hash(:NSSurfacelet, h)))))))

function Base.show(io::IO, t::NSSurfacelet)
    return print(
        io, "NSSurfacelet(J = ", t.J, ", L_array = ", t.L_array, ", order = ", t.order, ", poly = ", t.poly,
        ", transition = ", t.transition, ", tight = ", t.tight, ")"
    )
end

# Directional subbands of scale `j`.
_ns_nbands(t::NSSurfacelet, j::Int) = 3 * (1 << t.L_array[j])

# ── Decimated ─────────────────────────────────────────────────────────────────

"""
    Surfacelet(; J, L_array, pyr_mode = 1.5, bo = 12, msize = 15, beta = 2.5, lambda = 4)

Lu and Do's decimated 3-D surfacelet transform, reproducing their reference
toolbox SurfBox to rounding error: a 3/2-resampling Fourier pyramid, SurfBox's
hourglass bank, and iterated resampled checkerboard filter banks that refine
each hourglass into wedges while decimating critically.  About 4 coefficients
per voxel with the default `pyr_mode`, whatever the number of directions (see
[`surfacelet_redundancy`](@ref)); not shift-invariant.

- `J` — number of scales.
- `L_array` — directional refinement per scale, length `J`.  An integer `L`
  gives `2^L` wedges in each of the three hourglasses (`3 · 2^L` subbands),
  split as evenly as possible between the two transverse axes.  A `3 × 3`
  integer matrix gives SurfBox's level matrix instead: row `K` holds the number
  of checkerboard stages along each axis inside hourglass `K` (the diagonal is
  ignored), so hourglass `K` has `2^(sum of row K off the diagonal)` wedges.
- `pyr_mode` — pyramid resampling: `1.5` (default, redundancy ≈ 4), `1`
  (≈ 6.4) or `2` (≈ 3.4).  Every scale size must divide accordingly; see
  [`make_workspace`](@ref make_workspace(::Surfacelet, ::AbstractArray)).
- `bo` — order of the checkerboard filters, one of [`checkerboard_orders()`](@ref checkerboard_orders)
  (default 12).  Higher is sharper directionally, at no extra cost: the filters
  are evaluated in closed form in the frequency domain.
- `msize`, `beta`, `lambda` — the hourglass diamond kernel: its length (odd),
  Kaiser window `β`, and the even power it is raised to.

The checkerboard filters are biorthogonal, so the transform is a frame with
perfect reconstruction but not a tight one: `synthesize` is its
inverse, not its adjoint.  Real and complex data are accepted; complex data is
transformed as two real channels, so the transform is exactly real-linear.

```jldoctest
julia> using Surfacelets, Random

julia> Surfacelet(J = 2, L_array = [1, 2])
Surfacelet(J = 2, L_array = [1, 2], pyr_mode = 1.5, bo = 12, msize = 15, beta = 2.5, lambda = 4)

julia> x = randn(Xoshiro(7), 48, 48, 48);

julia> t = Surfacelet(J = 2, L_array = [1, 2]);

julia> maximum(abs, synthesize(t, analyze(t, x)) .- x) < 1e-10
true
```

# References
1. Y. M. Lu and M. N. Do, "Multidimensional Directional Filter Banks and
   Surfacelets", *IEEE Transactions on Image Processing*, 16(4):918–931, 2007.
   [doi:10.1109/TIP.2007.891785](https://doi.org/10.1109/TIP.2007.891785)
"""
struct Surfacelet <: AbstractMultiscaleTransform
    J::Int
    L_array::Vector{Union{Int, Matrix{Int}}}   # as given: integer levels or 3 × 3 level matrices
    levels::Vector{NTuple{3, NTuple{3, Int}}}   # levels[j][K][m]; zero for m == K
    pyr_mode::Float64
    bo::Int
    msize::Int
    beta::Float64
    lambda::Int
    function Surfacelet(J::Integer, L_array::AbstractVector, pyr_mode::Real, bo::Integer, msize::Integer, beta::Real, lambda::Integer)
        J >= 1 || throw(ArgumentError("J must be ≥ 1, got $(J)"))
        length(L_array) == J || throw(ArgumentError("length(L_array) must equal J = $(J), got $(length(L_array))"))
        levels = map(_surf_levels, L_array)
        all(lv -> all(r -> all(>=(0), r), lv), levels) ||
            throw(ArgumentError("directional levels must be ≥ 0"))
        _pyramid_mode(pyr_mode)
        _cbd_ladder(bo)
        isodd(msize) && msize >= 3 || throw(ArgumentError("msize must be an odd integer ≥ 3, got $(msize)"))
        beta > 0 || throw(ArgumentError("beta must be > 0, got $(beta)"))
        iseven(lambda) && lambda > 0 || throw(ArgumentError("lambda must be a positive even integer, got $(lambda)"))
        entries = Union{Int, Matrix{Int}}[_level_entry(L) for L in L_array]
        return new(J, entries, collect(NTuple{3, NTuple{3, Int}}, levels), pyr_mode, bo, msize, beta, lambda)
    end
end

Surfacelet(;
    J::Integer, L_array::AbstractVector, pyr_mode::Real = 1.5, bo::Integer = 12, msize::Integer = 15,
    beta::Real = 2.5, lambda::Integer = 4
) = Surfacelet(J, L_array, pyr_mode, bo, msize, beta, lambda)

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

# `levels` is derived from `L_array`, so it takes no part in equality.
Base.:(==)(a::Surfacelet, b::Surfacelet) =
    a.J == b.J && a.L_array == b.L_array && a.pyr_mode == b.pyr_mode && a.bo == b.bo &&
    a.msize == b.msize && a.beta == b.beta && a.lambda == b.lambda
Base.hash(t::Surfacelet, h::UInt) =
    hash(t.lambda, hash(t.beta, hash(t.msize, hash(t.bo, hash(t.pyr_mode, hash(t.L_array, hash(t.J, hash(:Surfacelet, h))))))))

function Base.show(io::IO, t::Surfacelet)
    L = "[" * join((repr(l) for l in t.L_array), ", ") * "]"
    return print(
        io, "Surfacelet(J = ", t.J, ", L_array = ", L, ", pyr_mode = ", t.pyr_mode, ", bo = ", t.bo,
        ", msize = ", t.msize, ", beta = ", t.beta, ", lambda = ", t.lambda, ")"
    )
end

# Subbands of one scale with level matrix `lv`: `2^(row sum)` per hourglass.
_nsubbands(lv::NTuple{3, NTuple{3, Int}}) = sum(K -> 1 << sum(lv[K]), 1:3)
