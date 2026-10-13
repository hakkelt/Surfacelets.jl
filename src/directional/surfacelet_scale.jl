# Critically sampled directional stage of one surfacelet scale, shared by the
# decimated and semi-translation-invariant transforms (`surfacelet.jl`).
#
# The scale's spectrum is split by the undecimated "ritf" hourglass bank
# (`ritf.jl`) into three hourglasses, each kept on the half-spectrum of its own
# axis; inside each, iterated resampled checkerboard filter banks (`irc.jl`)
# refine it into wedges and decimate critically.  A wedge block is then the
# half-spectrum of a real subband along its hourglass axis, so a real inverse FFT
# along that axis first (`brfft` over `(K, others…)`) turns it into the subband.
# The two transforms differ only in the pyramid that hands this stage its
# bandpass.

# Per-scale tables: hourglass masks and IRC stages for the three hourglasses.
struct _SurfScale{T, R <: AbstractArray{T, 3}, S <: IRCStage}
    dims::NTuple{3, Int}
    levels::NTuple{3, NTuple{3, Int}}
    masks::NTuple{3, R}             # √(3 hg_K / Σ hg) on the half-spectrum of axis K
    stages::NTuple{3, Vector{S}}
end

# IRC stages of the three hourglasses of a scale of size `n` with level matrix
# `levels`, each on the half-spectrum of its own axis.
function _surf_stages(::Type{T}, n::NTuple{3, Int}, levels::NTuple{3, NTuple{3, Int}}, bo::Int) where {T}
    return ntuple(3) do K
        _irc_stages(T, _half_dims(n, K), n[K], K, levels[K], bo)
    end
end

# Size of the half-spectrum along axis K, and its index ranges in the full one.
_half_dims(n::NTuple{3, Int}, K::Int) = ntuple(d -> d == K ? n[K] ÷ 2 + 1 : n[d], 3)
_half_ranges(n::NTuple{3, Int}, K::Int) = map(Base.OneTo, _half_dims(n, K))

# Wedge blocks of hourglass K after IRC: block counts and size along each axis.
function _surf_blocks(sc::_SurfScale, K::Int)
    n = sc.dims
    counts = ntuple(d -> 1 << sc.levels[K][d], 3)
    bsize = ntuple(d -> d == K ? n[K] ÷ 2 + 1 : n[d] ÷ counts[d], 3)
    return counts, bsize
end

# Size of the real subbands of hourglass K: full-length along K.
function _surf_subband_size(sc::_SurfScale, K::Int)
    _, bsize = _surf_blocks(sc, K)
    return ntuple(d -> d == K ? sc.dims[K] : bsize[d], 3)
end

# Real coefficients stored by the directional subbands of all `scales`.
function _directional_count(scales::AbstractVector{<:_SurfScale})
    total = 0
    for sc in scales, K in 1:3
        counts, _ = _surf_blocks(sc, K)
        total += prod(counts) * prod(_surf_subband_size(sc, K))
    end
    return total
end

# Index ranges of wedge block `i` (column-major over block indices, as SurfBox's
# `box2cell`).
function _surf_block_ranges(counts::NTuple{3, Int}, bsize::NTuple{3, Int}, i::Int)
    v = Tuple(CartesianIndices(counts)[i]) .- 1
    return ntuple(d -> (v[d] * bsize[d] + 1):((v[d] + 1) * bsize[d]), 3)
end

# ── Buffers and FFT plans ─────────────────────────────────────────────────────

# One hourglass at one scale: its half-spectrum, the IRC scratch (half of it), one
# wedge block, and the real FFTs between a block and its subband, halving axis K.
# The inverse is unnormalised; the `1/N` rides on the mask.
#
# FFTW takes the region `(K, others…)` directly.  The device FFTs (and JLArrays)
# halve only the first dimension of an increasing region, so for K ≠ 1 there the
# block is permuted to K-first around the FFT (`perm`), through `blkp` and `sbp`;
# the subbands keep the volume's axis order either way.
struct _HourglassWork{C <: AbstractArray, V <: AbstractVector, R <: AbstractArray, P, Q}
    S::C
    tmp::V
    blk::C
    blkp::C     # K-first copy of `blk` (empty when not permuting)
    sbp::R      # K-first subband (empty when not permuting)
    perm::NTuple{3, Int}
    permuted::Bool
    bwd::P      # brfft: block → subband (destroys its input)
    fwd::Q      # rfft:  subband → block
end

struct _ScaleWork{H <: Tuple}
    hg::H
end

function _scale_work(sc::_SurfScale{T}, proto::AbstractArray, flags) where {T}
    hg = ntuple(Val(3)) do K
        Tc = Complex{T}
        S = fill!(similar(proto, Tc, _half_dims(sc.dims, K)), zero(Tc))
        _, bsize = _surf_blocks(sc, K)
        blk = fill!(similar(proto, Tc, bsize), zero(Tc))
        sbsize = _surf_subband_size(sc, K)
        perm = (K, ntuple(i -> i < K ? i : i + 1, Val(2))...)
        permuted = !(proto isa Array) && K != 1
        if permuted
            blkp = fill!(similar(proto, Tc, map(d -> bsize[d], perm)), zero(Tc))
            sbp = fill!(similar(proto, T, map(d -> sbsize[d], perm)), zero(T))
            bwd = _fft_plan(plan_brfft, blkp, flags, sc.dims[K])
            fwd = _fft_plan(plan_rfft, sbp, flags)
        else
            blkp = similar(blk, 0, 0, 0)
            sbp = similar(proto, T, 0, 0, 0)
            bwd = _fft_plan(plan_brfft, blk, flags, sc.dims[K], perm)
            fwd = _fft_plan(plan_rfft, fill!(similar(proto, T, sbsize), zero(T)), flags, perm)
        end
        _HourglassWork(S, similar(S, length(S) ÷ 2), blk, blkp, sbp, perm, permuted, bwd, fwd)
    end
    return _ScaleWork(hg)
end

# subband = brfft(hw.blk) (consumes `hw.blk`)
function _block_to_subband!(sb::AbstractArray, hw::_HourglassWork)
    hw.permuted || return mul!(sb, hw.bwd, hw.blk)
    permutedims!(hw.blkp, hw.blk, hw.perm)
    mul!(hw.sbp, hw.bwd, hw.blkp)
    return permutedims!(sb, hw.sbp, invperm(hw.perm))
end

# hw.blk = rfft(sb)
function _subband_to_block!(hw::_HourglassWork, sb::AbstractArray)
    hw.permuted || return mul!(hw.blk, hw.fwd, sb)
    permutedims!(hw.sbp, sb, hw.perm)
    mul!(hw.blkp, hw.fwd, hw.sbp)
    return permutedims!(hw.blk, hw.blkp, invperm(hw.perm))
end

# A plan for `A`; FFTW flags only apply to host arrays.
_fft_plan(f::F, A::Array, flags, args...) where {F} = f(A, args...; flags = flags)
_fft_plan(f::F, A::AbstractArray, flags, args...) where {F} = f(A, args...)

# ── Analysis and synthesis ────────────────────────────────────────────────────

"""
    _surf_scale_forward!(out, X, sc, w) -> out

Directional analysis of one bandpass scale into the preallocated real subbands
`out`; `X` is the full spectrum of size `sc.dims` and is left untouched.
"""
function _surf_scale_forward!(out::AbstractVector, X::AbstractArray{Complex{T}, 3}, sc::_SurfScale{T}, w::_ScaleWork) where {T}
    i0 = 0
    for K in 1:3
        hw = w.hg[K]
        counts, bsize = _surf_blocks(sc, K)
        s = inv(T(prod(_surf_subband_size(sc, K))))
        _masked!(hw.S, view(X, _half_ranges(sc.dims, K)...), sc.masks[K], s)
        _irc_analysis!(hw.S, sc.stages[K], hw.tmp)
        for i in 1:prod(counts)
            _copy!(hw.blk, view(hw.S, _surf_block_ranges(counts, bsize, i)...))
            _block_to_subband!(out[i0 + i], hw)
        end
        i0 += prod(counts)
    end
    return out
end

"""
    _surf_scale_inverse!(X, subbands, sc, w) -> X

Directional synthesis of one bandpass scale: its full spectrum, written to `X`.
"""
function _surf_scale_inverse!(X::AbstractArray{Complex{T}, 3}, subbands::AbstractVector, sc::_SurfScale{T}, w::_ScaleWork) where {T}
    fill!(X, zero(Complex{T}))
    third = inv(T(3))
    i0 = 0
    for K in 1:3
        hw = w.hg[K]
        counts, bsize = _surf_blocks(sc, K)
        expect = _surf_subband_size(sc, K)
        for i in 1:prod(counts)
            sb = subbands[i0 + i]
            size(sb) == expect ||
                throw(DimensionMismatch("subband $(i0 + i) has size $(size(sb)), expected $(expect)"))
            _subband_to_block!(hw, sb)
            _copy!(view(hw.S, _surf_block_ranges(counts, bsize, i)...), hw.blk)
        end
        i0 += prod(counts)
        _irc_synthesis!(hw.S, sc.stages[K], hw.tmp)
        _hermitian_add!(X, hw.S, sc.masks[K], K, third)
    end
    return X
end

# X += c·M·S on the half-spectrum of axis K, and its Hermitian mirror on the rest,
# X[-ω] += c·M[ω]·conj(S[ω]), which is the spectrum of a real signal.  Index i of
# a full axis mirrors to mod(1 - i, n) + 1: 1 ↦ 1 and 2:n ↦ n:-1:2, so each of the
# two other axes splits into those two ranges and every piece is a strided view.
# All ranges are `StepRange`s, so the index tuples have one concrete type.
function _hermitian_add!(X::AbstractArray, S::AbstractArray, M::AbstractArray, K::Int, c::Real)
    n = size(X)
    h = size(S, K)
    _add_masked!(view(X, map(Base.OneTo, size(S))...), c, M, S, identity)
    o1, o2 = filter(!=(K), (1, 2, 3))
    for first1 in (true, false), first2 in (true, false)
        dst = ntuple(d -> d == K ? ((h + 1):1:n[K]) : _part(d == o1 ? first1 : first2, n[d], false), 3)
        src = ntuple(d -> d == K ? ((h - 1):-1:2) : _part(d == o1 ? first1 : first2, n[d], true), 3)
        _add_masked!(view(X, dst...), c, view(M, src...), view(S, src...), conj)
    end
    return X
end

# Index 1 alone, or the rest of an `n`-point axis (reversed for the mirror).
_part(isfirst::Bool, n::Int, mirrored::Bool) = isfirst ? (1:1:1) : (mirrored ? (n:-1:2) : (2:1:n))

# ── Host kernels ──────────────────────────────────────────────────────────────

# The block copies and masked sums above work on views with offset or reversed
# strides, where Base's broadcast walks Cartesian indices and checks every axis
# for a singleton at run time.  On host arrays Strided orders and blocks the loops
# by stride instead: at 96³ `_hermitian_add!` runs 2.6–3× faster and a block copy
# 5×, which takes 15% off a decimated inverse.  Device arrays keep the broadcast.
const _HostArray{T, N} = Union{Array{T, N}, SubArray{T, N, <:Array{T}}}

# dst = src
_copy!(dst, src) = (dst .= src)
_copy!(dst::_HostArray, src::_HostArray) = (@strided dst .= src; dst)

# dst = src · m · c
_masked!(dst, src, m, c) = (dst .= src .* m .* c)
_masked!(dst::_HostArray, src::_HostArray, m::_HostArray, c) = (@strided dst .= src .* m .* c; dst)

# dst += c · m · f(s)
_add_masked!(dst, c, m, s, f::F) where {F} = (dst .+= c .* m .* f.(s))
_add_masked!(dst::_HostArray, c, m::_HostArray, s::_HostArray, f::F) where {F} = (@strided dst .+= c .* m .* f.(s); dst)
