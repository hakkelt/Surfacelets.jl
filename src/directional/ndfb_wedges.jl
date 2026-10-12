# Wedge masks of the 3-D nonsubsampled directional filter bank (`ndfb.jl`).
#
# Wedge geometry.  Inside hourglass `P_i` the two transverse ratios
#
#     s = ω_a / ω_i ∈ [-1, 1],    t = ω_b / ω_i ∈ [-1, 1]
#
# parameterise direction, and an `l`-level refinement cuts each into `2^l` equal
# slabs.  Both ratios are invariant under `ω → -ω`, so the masks stay symmetric
# and real volumes keep real subbands.  Each ratio involves only two frequency
# axes, so a wedge mask is a *matrix*, not a volume — the whole bank stores
# `2·(2^l_a + 2^l_b)` planes regardless of how large the data is.

# The two transverse axes of hourglass `i`, in increasing order.
@inline _transverse_axes(i::Int) = i == 1 ? (2, 3) : (i == 2 ? (1, 3) : (1, 2))

# How a total refinement depth `L` is split between the two transverse planes of
# an hourglass.  The decimated transform's integer levels (`_surf_levels`) use
# the same split.
@inline _wedge_levels(L::Int) = (cld(L, 2), fld(L, 2))

# Weight of wedge `idx` of `nw` along one transverse ratio: which of the `nw`
# equal slabs of [-1, 1] the ratio `s` falls in, softened by `transition` (as a
# fraction of the slab width).  The outermost edges are hard, so the weights sum
# to exactly 1 for every `s` — including |s| > 1, outside the hourglass.
@inline function _wedge_weight(s::T, idx::Int, nw::Int, transition::T, poly::Int) where {T}
    nw == 1 && return one(T)
    delta = T(2) / nw
    width = transition * delta
    lo = idx == 1 ? one(T) : _soft_step(s, -one(T) + (idx - 1) * delta, width, poly)
    hi = idx == nw ? zero(T) : _soft_step(s, -one(T) + idx * delta, width, poly)
    return lo - hi
end

# Wedge masks for one hourglass: one vector of matrices per transverse plane.
function _wedge_masks(
        fb::HourglassFB{T}, i::Int, levels::NTuple{2, Int}, poly::Int, transition::T
    ) where {T}
    a, b = _transverse_axes(i)
    return (
        _wedge_masks_plane(fb, i, a, levels[1], poly, transition),
        _wedge_masks_plane(fb, i, b, levels[2], poly, transition),
    )
end

function _wedge_masks_plane(
        fb::HourglassFB{T}, ref::Int, tr::Int, l::Int, poly::Int, transition::T
    ) where {T}
    n = fb.dims
    wref = _folded_frequencies(T, n[ref]; half = ref == 1)
    wtr = _folded_frequencies(T, n[tr]; half = tr == 1)
    nw = 1 << l
    # Oriented 3-D shape: full length on the reference and transverse axes,
    # singleton on the remaining one.
    shape = ntuple(d -> d == ref ? length(wref) : (d == tr ? length(wtr) : 1), 3)
    flat = [Matrix{T}(undef, length(wref), length(wtr)) for _ in 1:nw]
    if l == 0
        fill!(flat[1], one(T))
        return [_orient(flat[1], ref, tr, shape)]
    end

    nref, ntr = n[ref], n[tr]
    @inbounds for (q, wt) in enumerate(wtr), (p, wr) in enumerate(wref)
        # Ratio parameterising direction inside the hourglass.
        s = _wedge_ratio(wr, wt, _self_conjugate_bin(p, nref), _self_conjugate_bin(q, ntr))
        for k in 1:nw
            flat[k][p, q] = _wedge_weight(s, k, nw, transition, poly)
        end
    end
    return [_orient(m, ref, tr, shape) for m in flat]
end

# Whether bin `k` of an `n`-point DFT axis is its own negative: bin 1 (`ω = 0`)
# always, plus the Nyquist bin of an even-length axis.  Decided from the *index*,
# never from the frequency value.  `2π(k - 1)/n` misses `T(π)` by an ulp for
# roughly half of all even `n` (22, 26, 30, 44, 52, 60, 88, 120, 240, …) — either
# landing just short of π, or just past it and folding to ≈ -π — so a value
# comparison silently skips the canonicalisation below at those sizes, and the
# `tight = true` bank then loses perfect reconstruction outright.
@inline _self_conjugate_bin(k::Int, n::Int) = mod(2 * (k - 1), n) == 0

# Signed slope `ω_tr / ω_ref`, evaluated at a canonical representative of the
# ± frequency pair.
#
# A directional wedge is even — `ω` and `-ω` are the same real oscillation and
# must land in the same subband — and `ω_tr / ω_ref` is invariant under negating
# both, so nothing is needed at a generic frequency.  On the DFT grid of an
# even-length axis, though, `0` and `π` are their own negatives, and there the
# invariance fails: `(π, ω_tr)` negates to `(π, -ω_tr)`, which flips the sign of
# the ratio.  Left alone that makes the masks non-even, the masked spectrum
# non-Hermitian, and the real subband a lossy record of it — invisible while
# reconstruction is a plain sum of weights, fatal once synthesis has to reapply
# the masks (`tight = true`).
#
# The fix is to fix the sign from the first coordinate that *can* carry one:
# `ω_ref` when its bin is not self-conjugate, `ω_tr` otherwise.  When both are
# self-negating the ratio is constant anyway.  Each factor canonicalises its own
# plane, which keeps the masks separable — the property the GPU path depends on.
@inline function _wedge_ratio(wr::T, wt::T, ref_self::Bool, tr_self::Bool) where {T}
    # ω_ref = 0 is the hourglass apex; the value is arbitrary because the wedge
    # weights sum to 1 for any `s`.
    wr == 0 && return zero(T)
    # ω_ref at Nyquist: it cannot carry a sign, so ω_tr's magnitude decides.
    ref_self && return abs(wt) / abs(wr)
    # ω_tr at Nyquist: it cannot carry a sign, so ω_ref's magnitude decides.
    tr_self && return wt / abs(wr)
    return wt / wr
end

# Lay a matrix indexed (reference axis, transverse axis) into a 3-D array whose
# non-singleton axes are `ref` and `tr`, ready to broadcast against a full mask.
# The remaining axis is singleton, so this is a reshape in column-major order —
# preceded by a transpose when the transverse axis comes first.
_orient(m::Matrix, ref::Int, tr::Int, shape::NTuple{3, Int}) =
    reshape(ref < tr ? m : permutedims(m), shape)
