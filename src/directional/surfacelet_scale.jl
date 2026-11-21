# Critically sampled directional stage of one scale of the decimated surfacelet
# transform (`surfacelet.jl`).
#
# The scale's spectrum is split by the undecimated "ritf" hourglass bank
# (`ritf.jl`) into three hourglasses, each kept on the half-spectrum of its own
# axis; inside each, iterated resampled checkerboard filter banks (`irc.jl`)
# refine it into wedges and decimate critically.  A wedge block is then the
# half-spectrum of a real subband along its hourglass axis, so a real inverse FFT
# along that axis first (`brfft` over `(K, others…)`) turns it into the subband.

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
# The inverse is unnormalized; the `1/N` rides on the mask.
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

# The three hourglasses, and the blocks of the scale as `(K, i)` pairs in subband
# order.  The block FFTs run as tasks over `items` (`_foreach_block`); on the host
# each task has its own flat buffer, as long as the largest block, whose head is
# reshaped to the block at hand, because `brfft` destroys its input.  A device has
# no buffers and runs one task on `hg[K].blk`.
struct _ScaleWork{H <: Tuple, B <: AbstractVector}
    hg::H
    items::Vector{NTuple{2, Int}}
    bufs::Vector{B}
end

function _scale_work(sc::_SurfScale{T}, proto::AbstractArray, flags, tasks::Integer) where {T}
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
    items = [(K, i) for K in 1:3 for i in 1:prod(first(_surf_blocks(sc, K)))]
    len = maximum(K -> length(hg[K].blk), 1:3)
    nbuf = proto isa Array ? n_tasks(proto, min(Int(tasks), length(items))) : 0
    bufs = [similar(proto, Complex{T}, len) for _ in 1:nbuf]
    return _ScaleWork(hg, items, bufs)
end

# Block buffer of task `t` for a block of size `bsize`.
_task_block(bufs::Vector{<:Vector}, ::_HourglassWork, t::Int, bsize::NTuple{3, Int}) =
    reshape(view(bufs[t], 1:prod(bsize)), bsize)
_task_block(::AbstractVector, hw::_HourglassWork, ::Int, ::NTuple{3, Int}) = hw.blk

# `f(hw, blk, ranges, n)` for every block of the scale: `hw` is its hourglass's
# work, `blk` the task's block buffer, `ranges` the block's place in `hw.S` and
# `n` its subband index.  Blocks are split among the tasks when `threaded`.
function _foreach_block(f::F, sc::_SurfScale, w::_ScaleWork, threaded::Bool) where {F}
    n = length(w.items)
    nt = isempty(w.bufs) ? 1 : min(length(w.bufs), n)
    foreach_chunk(n, nt; threaded) do range, t
        for idx in range
            K, i = w.items[idx]
            _one_block(f, sc, w, w.hg[K], K, i, idx, t)
        end
    end
    # The block loop may run on Polyester's workers, which keep spinning on
    # Julia's threads after it ends; the Strided kernels and threaded FFTs that
    # follow run on those threads and would wait for them.
    threaded && NestedThreading.quiesce_foreign_pools()
    return nothing
end

# A function barrier: `w.hg[K]` has a different type for each K.
function _one_block(f::F, sc::_SurfScale, w::_ScaleWork, hw::_HourglassWork, K::Int, i::Int, idx::Int, t::Int) where {F}
    counts, bsize = _surf_blocks(sc, K)
    return f(hw, _task_block(w.bufs, hw, t, bsize), _surf_block_ranges(counts, bsize, i), idx)
end

# subband = brfft(src), with the block buffer `blk` (consumed; only a host block
# is not `hw.blk`).  A permuted block is copied and permuted in one pass, a
# broadcast: the devices' `permutedims!` does not take a view.
function _block_to_subband!(sb::AbstractArray, hw::_HourglassWork, blk::AbstractArray, src::AbstractArray)
    if hw.permuted
        _permuted_copy!(hw.blkp, src, hw.perm)
        mul!(hw.sbp, hw.bwd, hw.blkp)
        return permutedims!(sb, hw.sbp, invperm(hw.perm))
    end
    _block_copy!(blk, src)
    return mul!(sb, hw.bwd, blk)
end

# dst = rfft(sb), through the block buffer `blk`
function _subband_to_block!(dst::AbstractArray, hw::_HourglassWork, blk::AbstractArray, sb::AbstractArray)
    if hw.permuted
        permutedims!(hw.sbp, sb, hw.perm)
        mul!(hw.blkp, hw.fwd, hw.sbp)
        return _permuted_copy!(dst, hw.blkp, invperm(hw.perm))
    end
    mul!(blk, hw.fwd, sb)
    return _block_copy!(dst, blk)
end

# dst = permutedims(src, p).  The permutation is a type parameter of
# `PermutedDimsArray`, so each one a `perm` (K-first) or its inverse can be is
# spelled out.
function _permuted_copy!(dst::AbstractArray{<:Any, 3}, src::AbstractArray{<:Any, 3}, p::NTuple{3, Int})
    p == (2, 1, 3) && return dst .= PermutedDimsArray(src, (2, 1, 3))
    p == (3, 1, 2) && return dst .= PermutedDimsArray(src, (3, 1, 2))
    p == (2, 3, 1) && return dst .= PermutedDimsArray(src, (2, 3, 1))
    throw(ArgumentError("not a K-first permutation or its inverse"))
end

# A plan for `A`; FFTW flags only apply to host arrays.
_fft_plan(f::F, A::Array, flags, args...) where {F} = f(A, args...; flags = flags)
_fft_plan(f::F, A::AbstractArray, flags, args...) where {F} = f(A, args...)

# ── Analysis and synthesis ────────────────────────────────────────────────────

"""
    _surf_scale_forward!(out, X, sc, w, threaded) -> out

Directional analysis of one bandpass scale into the preallocated real subbands
`out`; `X` is the full spectrum of size `sc.dims` and is left untouched.
"""
function _surf_scale_forward!(
        out, X::AbstractArray{Complex{T}, 3}, sc::_SurfScale{T}, w::_ScaleWork, threaded::Bool
    ) where {T}
    for K in 1:3
        hw = w.hg[K]
        s = inv(T(prod(_surf_subband_size(sc, K))))
        _masked!(hw.S, view(X, _half_ranges(sc.dims, K)...), sc.masks[K], s)
        _irc_analysis!(hw.S, sc.stages[K], hw.tmp)
    end
    _foreach_block(sc, w, threaded) do hw, blk, ranges, n
        _block_to_subband!(out[n], hw, blk, view(hw.S, ranges...))
    end
    return out
end

"""
    _surf_scale_inverse!(X, subbands, sc, w, threaded) -> X

Directional synthesis of one bandpass scale: its full spectrum, written to `X`.
"""
function _surf_scale_inverse!(
        X::AbstractArray{Complex{T}, 3}, subbands, sc::_SurfScale{T}, w::_ScaleWork, threaded::Bool
    ) where {T}
    for (n, (K, _)) in enumerate(w.items)
        expect = _surf_subband_size(sc, K)
        size(subbands[n]) == expect ||
            throw(DimensionMismatch("subband $(n) has size $(size(subbands[n])), expected $(expect)"))
    end
    _foreach_block(sc, w, threaded) do hw, blk, ranges, n
        _subband_to_block!(view(hw.S, ranges...), hw, blk, subbands[n])
    end
    fill!(X, zero(Complex{T}))
    third = inv(T(3))
    for K in 1:3
        hw = w.hg[K]
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
# by stride instead.  Device arrays keep the broadcast.
const _HostArray{T, N} = Union{Array{T, N}, SubArray{T, N, <:Array{T}}}

# dst = src
_copy!(dst, src) = (dst .= src)
_copy!(dst::_HostArray, src::_HostArray) = (@strided dst .= src; dst)

# dst = src between a host task's block buffer and its place in the hourglass,
# inside a block task, which must not spawn Strided's own tasks: a plain loop over
# the block, first axis innermost.
const _HostBlock{T} = Base.ReshapedArray{T, 3, <:SubArray{T, 1, Vector{T}}}

_block_copy!(dst, src) = (dst .= src)
_block_copy!(dst::_HostBlock, src::_HostArray) = _loop_copy!(dst, src)
_block_copy!(dst::_HostArray, src::_HostBlock) = _loop_copy!(dst, src)

function _loop_copy!(dst::AbstractArray{T, 3}, src::AbstractArray{T, 3}) where {T}
    @inbounds for I in CartesianIndices(dst)
        dst[I] = src[I]
    end
    return dst
end

# dst = src · m · c
_masked!(dst, src, m, c) = (dst .= src .* m .* c)
_masked!(dst::_HostArray, src::_HostArray, m::_HostArray, c) = (@strided dst .= src .* m .* c; dst)

# dst += c · m · f(s)
_add_masked!(dst, c, m, s, f::F) where {F} = (dst .+= c .* m .* f.(s))
_add_masked!(dst::_HostArray, c, m::_HostArray, s::_HostArray, f::F) where {F} = (@strided dst .+= c .* m .* f.(s); dst)
