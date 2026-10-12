# Iteratively resampled checkerboard (IRC) filter banks — the critically
# sampled directional refinement inside each hourglass of the decimated
# surfacelet transform.  Lu & Do, "Multidimensional Directional Filter Banks and
# Surfacelets", IEEE TIP 16(4), 2007, Sections III-C/D (Fig. 8, Proposition 2)
# and V-C; reference implementation SurfBox `IRCdecF.m` / `IRCrecF.m`.
#
# Channel K of the hourglass bank arrives as a spectrum that keeps only the
# non-negative half of axis K.  For every other axis m, `levels[m]` two-channel
# checkerboard stages run on the (ω_K, ω_m) planes, each downsampling axis m by
# 2.  Everything happens in the frequency domain, in place:
#
#   * Downsampling by 2 along m folds the spectrum.  At stage `le` axis m is cut
#     into blocks of `b = n_m / 2^le` bins; each pair of neighbouring blocks
#     (A, B) is one subband at the previous stage, and the two channels' folded
#     outputs `F_c(A)·A + F_c(B)·B` overwrite A (channel 0) and B (channel 1).
#
#   * Resampling is folded into the filters.  Group `g` of `2^{le-1}` (one
#     subband of the previous stage, `2b` bins wide) uses the checkerboard filter
#     sheared by `sf = 2^{le-1} - 1 - 2g`, i.e. `H(ω_K, ω_m + sf·ω_K)`, sampled on
#     that group's own `2b`-point grid.  This is the closed form of SurfBox's
#     `resampc` + `fft2` on the spatial filter, and it is exact at any size —
#     SurfBox instead refuses blocks narrower than the spatial filter.
#
#   * The analysis filters carry the ½ of downsampling, as in SurfBox.
#
# Every step is real filtering plus rectangular downsampling along axes other
# than K, so each final block is still the half-spectrum of a real subband.

struct IRCStage{A <: AbstractArray}
    m::Int          # axis being decimated
    b::Int          # block size after this stage
    ana::NTuple{2, A}   # analysis filter slices, channel 0 and 1
    syn::NTuple{2, A}   # synthesis filter slices
end

# Axis `m` of `A` split into `(b, 2, n_m ÷ 2b)`, so that the two halves of every
# 2b-bin group are `selectdim(·, m + 1, 1)` and `selectdim(·, m + 1, 2)`.
@inline function _irc_split(A::AbstractArray, m::Int, b::Int)
    sz = size(A)
    return reshape(A, (sz[1:(m - 1)]..., b, 2, sz[m] ÷ (2b), sz[(m + 1):end]...))
end

# Filter slice of channel `chan` at stage `le` on axis m, shaped to broadcast
# against a `dims` half-spectrum of channel K.
function _irc_slice(
        f::F, t, chan::Int, ::Type{T}, dims::NTuple{3, Int}, nK::Int, K::Int, m::Int, le::Int, scale::T
    ) where {F, T}
    nKh = dims[K]
    nm = dims[m]
    W = nm >> (le - 1)              # width of one group at this stage
    S = Matrix{Complex{T}}(undef, nKh, nm)
    for c in 0:(nm - 1), k in 0:(nKh - 1)
        g, l = divrem(c, W)
        sf = (1 << (le - 1)) - 1 - 2g
        ωK = T(2π) * k / nK
        ωm = T(2π) * l / W + sf * ωK
        S[k + 1, c + 1] = scale * f(t, chan, ωK, ωm)
    end
    sz = ntuple(d -> d == K ? nKh : (d == m ? nm : 1), 3)
    # `S` is indexed (K, m); when m comes first in memory it must be transposed
    # before the reshape, not merely relabelled.
    return reshape(K < m ? S : permutedims(S), sz)
end

"""
    _irc_stages(T, dims, K, levels, bo) -> Vector{IRCStage}

Stages of the IRC bank for hourglass channel `K` on a half-spectrum of size
`dims` (axis K already halved), in analysis order: axes m ≠ K ascending, and
`levels[m]` stages on each.
"""
function _irc_stages(::Type{T}, dims::NTuple{3, Int}, nK::Int, K::Int, levels::NTuple{3, Int}, bo::Int) where {T}
    t = _cbd_ladder(bo)
    stages = IRCStage{Array{Complex{T}, 3}}[]
    for m in 1:3
        m == K && continue
        levels[m] >= 0 || throw(ArgumentError("directional levels must be ≥ 0"))
        dims[m] % (1 << levels[m]) == 0 ||
            throw(ArgumentError("axis $(m) of length $(dims[m]) cannot be split $(levels[m]) times"))
        for le in 1:levels[m]
            ana = ntuple(c -> _irc_slice(_cbd_analysis, t, c - 1, T, dims, nK, K, m, le, T(0.5)), 2)
            syn = ntuple(c -> _irc_slice(_cbd_synthesis, t, c - 1, T, dims, nK, K, m, le, one(T)), 2)
            push!(stages, IRCStage(m, dims[m] >> le, ana, syn))
        end
    end
    return stages
end

# One analysis stage, in place on `S`; `tmp` holds one half of `S`.
function _irc_analysis_stage!(S::AbstractArray, st::IRCStage, tmp::AbstractArray)
    m, b = st.m, st.b
    X = _irc_split(S, m, b)
    A, B = selectdim(X, m + 1, 1), selectdim(X, m + 1, 2)
    F0, F1 = _irc_split(st.ana[1], m, b), _irc_split(st.ana[2], m, b)
    F0a, F0b = selectdim(F0, m + 1, 1), selectdim(F0, m + 1, 2)
    F1a, F1b = selectdim(F1, m + 1, 1), selectdim(F1, m + 1, 2)
    t = reshape(tmp, size(A))
    @. t = F0a * A + F0b * B
    @. B = F1a * A + F1b * B
    A .= t      # a broadcast, not `copyto!`, so a strided device view works too
    return S
end

# One synthesis stage, in place: upsample both channels by duplication and
# filter, `X(A) = G₀(A)·c₀ + G₁(A)·c₁`, `X(B) = G₀(B)·c₀ + G₁(B)·c₁`.
function _irc_synthesis_stage!(S::AbstractArray, st::IRCStage, tmp::AbstractArray)
    m, b = st.m, st.b
    X = _irc_split(S, m, b)
    A, B = selectdim(X, m + 1, 1), selectdim(X, m + 1, 2)
    G0, G1 = _irc_split(st.syn[1], m, b), _irc_split(st.syn[2], m, b)
    G0a, G0b = selectdim(G0, m + 1, 1), selectdim(G0, m + 1, 2)
    G1a, G1b = selectdim(G1, m + 1, 1), selectdim(G1, m + 1, 2)
    t = reshape(tmp, size(A))
    @. t = G0a * A + G1a * B
    @. B = G0b * A + G1b * B
    A .= t
    return S
end

function _irc_analysis!(S::AbstractArray, stages::AbstractVector{<:IRCStage}, tmp::AbstractArray)
    for st in stages
        _irc_analysis_stage!(S, st, tmp)
    end
    return S
end

function _irc_synthesis!(S::AbstractArray, stages::AbstractVector{<:IRCStage}, tmp::AbstractArray)
    for st in Iterators.reverse(stages)
        _irc_synthesis_stage!(S, st, tmp)
    end
    return S
end
