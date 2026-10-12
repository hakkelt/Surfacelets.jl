# Coefficient containers for the surfacelet transforms.
#
# The nonsubsampled transform keeps every subband at the size of the input; the
# decimated and semi-translation-invariant transforms share one container, whose
# subbands each have their own decimated size.

"""
    NSSurfaceletCoefficients{Td, A}

Output of [`ns_surfacelet_forward`](@ref).  `coarse` is the low-pass residual after `J`
pyramid levels and `subbands[j][k]` the `k`-th directional subband at scale `j`.
Every array has the spatial size of the input — the transform is nonsubsampled.

Size warning: the total coefficient volume is `1 + Σ_j 3·2^L_j` copies of the
input.  At 256³ `Float32` with `J = 3` and `L = [2, 2, 3]` that is 3.1 GB; use
the streaming form of [`ns_surfacelet_forward`](@ref) when that does not fit.
"""
struct NSSurfaceletCoefficients{Td <: Number, A <: AbstractArray{Td, 3}}
    coarse::A
    subbands::Vector{Vector{A}}
end

# The parameters are inferred from the field types by Julia's implicit outer
# constructor; an explicit one here would collide with it.

Base.eltype(::NSSurfaceletCoefficients{Td}) where {Td} = Td

function Base.show(io::IO, c::NSSurfaceletCoefficients)
    ns = sum(length, c.subbands; init = 0)
    return print(io, "NSSurfaceletCoefficients(size=$(size(c.coarse)), J=$(length(c.subbands)), subbands=$(ns))")
end

"""
    SurfaceletCoefficients{T}

Output of [`surfacelet_forward`](@ref).  `subbands[j]` lists the directional
subbands of scale `j` — hourglass 1, 2, 3 in turn, and inside each the wedges in
SurfBox's order — each at its own decimated size; `coarse` is the lowpass.
"""
struct SurfaceletCoefficients{T <: AbstractFloat, A <: AbstractArray{T, 3}}
    dims::NTuple{3, Int}     # size of the volume this came from
    coarse::A
    subbands::Vector{Vector{A}}
end

Base.eltype(::SurfaceletCoefficients{T}) where {T} = T

function Base.show(io::IO, c::SurfaceletCoefficients)
    ns = sum(length, c.subbands; init = 0)
    n = length(c.coarse) + sum(sum(length, s; init = 0) for s in c.subbands; init = 0)
    return print(
        io, "SurfaceletCoefficients(dims=$(c.dims), J=$(length(c.subbands)), ",
        "subbands=$(ns), coefficients=$(n))"
    )
end
