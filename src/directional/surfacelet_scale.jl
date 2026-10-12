# Critically sampled directional stage of one surfacelet scale, shared by the
# decimated (`surfacelet.jl`) and semi-translation-invariant
# (`sti_surfacelet.jl`) transforms.
#
# The scale's spectrum is split by the undecimated "ritf" hourglass bank
# (`ritf.jl`) into three hourglasses, each kept on the half-spectrum of its own
# axis; inside each, iterated resampled checkerboard filter banks (`irc.jl`)
# refine it into wedges and decimate critically.  A wedge block is then
# Hermitian-completed along its hourglass axis and inverse transformed, so every
# subband is real.  The two transforms differ only in the pyramid that hands
# this stage its bandpass.

# Per-scale state: hourglass masks and IRC stages for the three hourglasses.
struct _SurfScale{T, R <: AbstractArray{T, 3}, S <: IRCStage}
    dims::NTuple{3, Int}
    levels::NTuple{3, NTuple{3, Int}}
    masks::NTuple{3, R}             # √(3 hg_K / Σ hg) on the half-spectrum of axis K
    stages::NTuple{3, Vector{S}}
end

# Wedge blocks of hourglass K after IRC: block counts and size along each axis.
function _surf_blocks(sc::_SurfScale, K::Int)
    n = sc.dims
    counts = ntuple(d -> 1 << sc.levels[K][d], 3)
    bsize = ntuple(d -> d == K ? n[K] ÷ 2 + 1 : n[d] ÷ counts[d], 3)
    return counts, bsize
end

# Index ranges of wedge block `i` (column-major over block indices, as SurfBox's
# `box2cell`).
function _surf_block_ranges(counts::NTuple{3, Int}, bsize::NTuple{3, Int}, i::Int)
    v = Tuple(CartesianIndices(counts)[i]) .- 1
    return ntuple(d -> (v[d] * bsize[d] + 1):((v[d] + 1) * bsize[d]), 3)
end

# Hermitian completion along axis K: a half-spectrum of `h = n/2 + 1` bins
# becomes the full `n`-bin spectrum of a real signal, `X[-ω] = conj(X[ω])`.
function _hermitian_complete(x::AbstractArray{<:Complex, 3}, K::Int)
    h = size(x, K)
    n = 2 * (h - 1)
    others = Tuple(d for d in 1:3 if d != K)
    # Reverse every other axis modulo its length: index i ↦ mod(1 - i, len) + 1.
    r = circshift(reverse(x; dims = others), ntuple(d -> d == K ? 0 : 1, 3))
    tail = conj.(reverse(selectdim(r, K, 2:(h - 1)); dims = K))
    return cat(x, tail; dims = K)
end

# Analysis of one bandpass scale; `X` is the full spectrum of size `sc.dims`.
function _surf_scale_forward(X::AbstractArray{Complex{T}, 3}, sc::_SurfScale) where {T}
    n = sc.dims
    out = Vector{typeof(similar(X, T, (0, 0, 0)))}()
    for K in 1:3
        half = ntuple(d -> d == K ? (1:(n[K] ÷ 2 + 1)) : (1:n[d]), 3)
        S = X[half...] .* sc.masks[K]
        tmp = similar(S, length(S) ÷ 2)
        _irc_analysis!(S, sc.stages[K], tmp)
        counts, bsize = _surf_blocks(sc, K)
        for i in 1:prod(counts)
            blk = S[_surf_block_ranges(counts, bsize, i)...]
            push!(out, real.(ifft(_hermitian_complete(blk, K))))
        end
    end
    return out
end

# Synthesis of one bandpass scale; returns its full spectrum.
function _surf_scale_inverse(subbands::AbstractVector, sc::_SurfScale{T}) where {T}
    n = sc.dims
    proto = first(subbands)
    X = fill!(similar(proto, Complex{T}, n), zero(Complex{T}))
    i0 = 0
    for K in 1:3
        hsz = ntuple(d -> d == K ? n[K] ÷ 2 + 1 : n[d], 3)
        S = fill!(similar(proto, Complex{T}, hsz), zero(Complex{T}))
        counts, bsize = _surf_blocks(sc, K)
        for i in 1:prod(counts)
            sb = subbands[i0 + i]
            expect = ntuple(d -> d == K ? n[K] : bsize[d], 3)
            size(sb) == expect ||
                throw(DimensionMismatch("subband $(i0 + i) has size $(size(sb)), expected $(expect)"))
            F = fft(complex.(sb))
            S[_surf_block_ranges(counts, bsize, i)...] .= selectdim(F, K, 1:hsz[K])
        end
        i0 += prod(counts)
        tmp = similar(S, length(S) ÷ 2)
        _irc_synthesis!(S, sc.stages[K], tmp)
        X .+= _hermitian_complete(S .* sc.masks[K], K)
    end
    X ./= 3
    return X
end
