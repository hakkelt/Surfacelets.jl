# Types for the surfacelet transforms.
#
# The directional stage is parameterised by a Bernstein mapping order and a
# halfband split polynomial — the hourglass partition has no filter-pair
# description at all, so these parameters have no counterpart in a
# separable-lattice directional filter bank.

"""
    NSurfaceletParams

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

julia> NSurfaceletParams(J = 2, L_array = [1, 2])
NSurfaceletParams(J=2, L_array=[1, 2], order=12, …)
```
"""
struct NSurfaceletParams{T <: AbstractFloat}
    J::Int
    L_array::Vector{Int}
    lp_filters::FilterPair{T}
    order::Int
    poly::Int
    transition::T
    tight::Bool
end

function NSurfaceletParams(;
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
    return NSurfaceletParams{T}(
        J, collect(Int, L_array),
        FilterPair{T}(T.(lp_filters.h), T.(lp_filters.g)),
        order, poly, T(transition), tight
    )
end

Base.eltype(::NSurfaceletParams{T}) where {T} = T

function Base.show(io::IO, p::NSurfaceletParams)
    return print(io, "NSurfaceletParams(J=$(p.J), L_array=$(p.L_array), order=$(p.order), …)")
end

# Convert to real filter precision `Tf` (no-op when already Tf).
_convert_params(::Type{Tf}, p::NSurfaceletParams{Tf}) where {Tf} = p
_convert_params(::Type{Tf}, p::NSurfaceletParams) where {Tf <: AbstractFloat} =
    NSurfaceletParams{Tf}(
    p.J, p.L_array,
    FilterPair{Tf}(Tf.(p.lp_filters.h), Tf.(p.lp_filters.g)),
    p.order, p.poly, Tf(p.transition), p.tight
)

"""
    nsurfacelet_nsubbands(params::NSurfaceletParams, j::Int) -> Int

Number of directional subbands at scale `j`, `3 * 2^L_array[j]`.
"""
nsurfacelet_nsubbands(p::NSurfaceletParams, j::Int) = 3 * (1 << p.L_array[j])

"""
    NSurfaceletCoefficients{Td, A}

Output of [`nsurfacelet_forward`](@ref).  `coarse` is the low-pass residual after `J`
pyramid levels and `subbands[j][k]` the `k`-th directional subband at scale `j`.
Every array has the spatial size of the input — the transform is nonsubsampled.

Size warning: the total coefficient volume is `1 + Σ_j 3·2^L_j` copies of the
input.  At 256³ `Float32` with `J = 3` and `L = [2, 2, 3]` that is 3.1 GB; use
the streaming form of [`nsurfacelet_forward`](@ref) when that does not fit.
"""
struct NSurfaceletCoefficients{Td <: Number, A <: AbstractArray{Td, 3}}
    coarse::A
    subbands::Vector{Vector{A}}
end

# The parameters are inferred from the field types by Julia's implicit outer
# constructor; an explicit one here would collide with it.

Base.eltype(::NSurfaceletCoefficients{Td}) where {Td} = Td

function Base.show(io::IO, c::NSurfaceletCoefficients)
    ns = sum(length, c.subbands; init = 0)
    return print(io, "NSurfaceletCoefficients(size=$(size(c.coarse)), J=$(length(c.subbands)), subbands=$(ns))")
end
