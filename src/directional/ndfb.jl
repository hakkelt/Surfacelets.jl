# 3-D nonsubsampled directional filter bank (NDFB) — hourglass stage refined
# into wedges.
#
# Reference: Y. M. Lu and M. N. Do, "Multidimensional Directional Filter Banks
# and Surfacelets", IEEE Trans. Image Processing 16(4):918-931, 2007.
#
# Structure (paper, Fig. 7).  Each of the three hourglass subbands `P_i` is fed
# through two 2-D filter banks operating on the two planes that contain axis `i`:
# `l_a` levels on the first, `l_b` on the second, giving `2^(l_a + l_b)` wedges
# per hourglass and `3 · 2^L` directional subbands in total for `L = l_a + l_b`.
# The ideal passbands are rectangular-based pyramids radiating from the origin.
#
# Wedge geometry: each wedge is a soft slab of a transverse ratio `ω_a / ω_i`,
# so its mask is a matrix, not a volume; the construction is in
# `ndfb_wedges.jl`.
#
# Deviation from the paper, deliberate: Lu & Do build these wedges with a chain
# of iteratively resampled checkerboard filter banks (IRC), because their
# transform is critically sampled inside each hourglass and the resampling is
# what keeps it so.  This bank is nonsubsampled, so there is nothing to resample
# for; the wedge boundaries are placed directly in the frequency domain instead.
# The resulting *ideal* partition is the same one, and getting it directly avoids
# the failure mode the 2-D `nsdfb` hit when its wedges were approximated by
# separable 1-D filtering — that yields stripe/block tilings, not wedges.
#
# Tightness follows the hourglass bank's `tight` setting, and needs nothing
# extra: the wedge weights along each ratio are a partition of unity, so the
# products `W_hourglass · W_a · W_b` are too, and `Σ (√W)² = Σ W = 1` holds over
# all `3·2^levels` subbands however deep the refinement goes.
#
# Perfect reconstruction stays structural: for each ratio the soft steps
# telescope to 1, so the wedge masks of a hourglass sum to that hourglass's mask,
# the three hourglass masks sum to 1, and reconstruction is a plain sum.

"""
    NDFB{T}

3-D nonsubsampled directional filter bank: [`HourglassFB`](@ref) followed by a
wedge refinement of each hourglass.  Build with [`make_ndfb`](@ref).
"""
struct NDFB{T <: AbstractFloat, A3 <: AbstractArray{T, 3}, FB <: HourglassFB{T}}
    fb::FB
    levels::NTuple{2, Int}          # (l_a, l_b) refinement levels per hourglass
    # Wedge masks, `[hourglass][which plane][wedge]`.  Each depends on only two
    # frequency axes, and is stored as a 3-D array with a singleton on the third
    # so a subband mask is a plain broadcast — no scalar indexing, so the bank
    # also runs on GPU arrays.  The storage is still matrix-sized.
    wedges::NTuple{3, NTuple{2, Vector{A3}}}
end

"""
    make_ndfb(dims::NTuple{3,Int}, ::Type{T}=Float64;
                levels::Int=2, order::Int=12, poly::Int=1, tight::Bool=true,
                transition::Real=1.0, threading=Auto(), flags=FFTW.MEASURE)

Build a 3-D NDFB producing `3 * 2^levels` directional subbands, each the full
size of the input.

`levels` is the total refinement depth `L`; it is split as evenly as possible
between the two transverse planes of each hourglass (`l_a = ceil(L/2)`,
`l_b = floor(L/2)`), matching the balanced expansion rule of the paper.
`levels = 0` leaves the plain three-subband hourglass bank.

`order`, `poly` and `tight` are forwarded to [`make_hourglass`](@ref); with
`tight = true` (the default) the whole bank is a Parseval frame, since the wedge
weights are themselves a partition of unity.  `transition`
is the half-width of the soft step at each wedge boundary, as a fraction of the
wedge width: `0` gives hard boundaries, `0.5` lets the two steps of a wedge meet
at its centre, and `1` (the default) ramps across a whole wedge on either side
of each boundary, so no wedge has a flat passband — a plane wave at the centre
of a wedge keeps about 71 % of its energy there at `levels = 2`.
"""
function make_ndfb(
        dims::NTuple{3, Int}, ::Type{T} = Float64;
        levels::Int = 2, order::Int = 12, poly::Int = 1, tight::Bool = true,
        transition::Real = 1.0,
        threading::ThreadingPolicy = Auto(),
        flags::Integer = FFTW.MEASURE
    ) where {T <: AbstractFloat}
    levels >= 0 || throw(ArgumentError("levels must be ≥ 0"))
    0 <= transition || throw(ArgumentError("transition must be ≥ 0"))
    fb = make_hourglass(
        dims, T; order = order, poly = poly, tight = tight,
        threading = threading, flags = flags
    )
    return ndfb_sharing(fb, levels; poly = poly, transition = transition)
end

"""
    ndfb_sharing(fb::HourglassFB, levels::Int; poly=fb.poly, transition=1.0) -> NDFB

Build a directional bank on top of an *existing* hourglass bank instead of
planning a new one.  The hourglass stage depends only on the volume size, so a
multiscale transform can plan it once and vary only the wedge refinement per
scale — the wedge masks are matrices, so an extra scale costs kilobytes.
"""
function ndfb_sharing(
        fb::HourglassFB{T}, levels::Int;
        poly::Int = fb.poly, transition::Real = 1.0
    ) where {T <: AbstractFloat}
    levels >= 0 || throw(ArgumentError("levels must be ≥ 0"))
    transition >= 0 || throw(ArgumentError("transition must be ≥ 0"))
    lev = _wedge_levels(levels)
    wedges = ntuple(i -> _wedge_masks(fb.dims, i, lev, poly, T(transition)), 3)
    return NDFB{T, typeof(wedges[1][1][1]), typeof(fb)}(fb, lev, wedges)
end

function make_ndfb(dims::NTuple{3, Int}, ::Type{Complex{T}}; kwargs...) where {T <: AbstractFloat}
    return make_ndfb(dims, T; kwargs...)
end

make_ndfb(n1::Int, n2::Int, n3::Int, ::Type{Td} = Float64; kwargs...) where {Td} =
    make_ndfb((n1, n2, n3), Td; kwargs...)

# How many wedges the bank's refinement gives on each transverse plane; the
# split itself is `_wedge_levels` (`ndfb_wedges.jl`).
@inline _wedge_counts(fb::NDFB) = (1 << fb.levels[1], 1 << fb.levels[2])

# Decompose a linear subband index into (hourglass, wedge on plane a, wedge on
# plane b) given the two wedge counts.  Subbands are ordered hourglass first,
# then `p`, then `q`.
@inline function _subband_triple(idx::Int, na::Int, nb::Int)
    z = idx - 1
    q = z % nb + 1
    z ÷= nb
    p = z % na + 1
    return (z ÷ na + 1, p, q)
end

"""
    ndfb_nsubbands(fb::NDFB) -> Int

Number of directional subbands, `3 * 2^levels`.
"""
ndfb_nsubbands(fb::NDFB) = 3 * (1 << sum(fb.levels))

"""
    ndfb_subband_index(fb::NDFB, hourglass, p, q) -> Int

Linear index of the subband refining `hourglass` (1-3) at wedge `p` of its first
transverse plane and wedge `q` of its second.  Subbands are ordered hourglass
first, then `p`, then `q`.
"""
function ndfb_subband_index(fb::NDFB, hourglass::Int, p::Int, q::Int)
    na, nb = _wedge_counts(fb)
    1 <= hourglass <= 3 || throw(ArgumentError("hourglass index must be 1, 2 or 3"))
    1 <= p <= na || throw(ArgumentError("wedge index p out of range 1:$(na)"))
    1 <= q <= nb || throw(ArgumentError("wedge index q out of range 1:$(nb)"))
    return ((hourglass - 1) * na + (p - 1)) * nb + q
end

"""
    ndfb_subband_triple(fb::NDFB, idx::Int) -> (hourglass, p, q)

Inverse of [`ndfb_subband_index`](@ref): which hourglass and which wedge of
each transverse plane subband `idx` refines.  Use it to interpret a subband
index as a direction.
"""
function ndfb_subband_triple(fb::NDFB, idx::Int)
    na, nb = _wedge_counts(fb)
    1 <= idx <= 3 * na * nb || throw(ArgumentError("subband index out of range"))
    return _subband_triple(idx, na, nb)
end

"""
    ndfb_mask(fb::NDFB, idx::Int) -> Array{T,3}

Frequency-domain analysis mask of directional subband `idx` on the rfft grid.
For inspection and tests; the transform never materialises one.

The masks sum to 1 over all subbands when the bank is not tight, and their
squares sum to 1 when it is.  [`ndfb_weight`](@ref) returns the underlying
partition of unity either way.
"""
function ndfb_mask(fb::NDFB{T}, idx::Int) where {T}
    i, p, q = ndfb_subband_triple(fb, idx)
    out = similar(fb.fb.m1)
    # The transform kernel with a unit spectrum: the mask itself.
    _mask_multiply!(out, fb, true, i, p, q)
    return out
end

"""
    ndfb_weight(fb::NDFB, idx::Int) -> Array{T,3}

Frequency-domain *weight* of directional subband `idx`, before the `tight`
normalisation: the hourglass weight times the two wedge factors.  The weights
sum to 1 over all subbands at every frequency.
"""
function ndfb_weight(fb::NDFB{T}, idx::Int) where {T}
    i, p, q = ndfb_subband_triple(fb, idx)
    hg = fb.fb
    wa = fb.wedges[i][1][p]
    wb = fb.wedges[i][2][q]
    out = similar(hg.m1)
    @. out = _stage1_weight(hg.m1, i) * _stage2_weight(hg.m2, i) * wa * wb
    return out
end

# The directional bank's mask is the hourglass mask of subband `i` times the two
# wedge factors, so both kernels are the hourglass ones from `hourglass.jl`
# with the wedge planes filled in.  `wa` and `wb` are indexed by (reference axis,
# transverse axis), so which of the three frequency axes each varies along
# depends on the hourglass — which is why they are stored already oriented and
# the whole mask stays one broadcast.
_mask_multiply!(dst, fb::NDFB, spec, i::Int, p::Int, q::Int) =
    _mask_multiply!(dst, fb.fb, spec, i, fb.wedges[i][1][p], fb.wedges[i][2][q])

# `acc += spec · mask` — the synthesis counterpart, used only when tight.
_mask_accumulate!(acc, fb::NDFB, spec, i::Int, p::Int, q::Int) =
    _mask_accumulate!(acc, fb.fb, spec, i, fb.wedges[i][1][p], fb.wedges[i][2][q])

"""
    ndfb_decompose(volume, fb::NDFB) -> Vector

All `3 * 2^levels` directional subbands, each the size of `volume`.

Memory warning: at 256³ `Float32` a single subband is 64 MB, so `levels = 3`
(24 subbands) materialises 1.5 GB.  Use the callback form below to process
subbands one at a time instead.
"""
function ndfb_decompose(volume::AbstractArray{Td, 3}, fb::NDFB) where {Td}
    out = [similar(volume) for _ in 1:ndfb_nsubbands(fb)]
    ndfb_decompose!(out, volume, fb)
    return out
end

"""
    ndfb_decompose!(dst, volume, fb::NDFB) -> dst

In-place [`ndfb_decompose`](@ref); `dst` is an indexable collection of
`ndfb_nsubbands(fb)` arrays.
"""
function ndfb_decompose!(dst, volume::AbstractArray{Td, 3}, fb::NDFB) where {Td}
    length(dst) == ndfb_nsubbands(fb) ||
        throw(DimensionMismatch("dst must hold $(ndfb_nsubbands(fb)) subbands"))
    ndfb_decompose(volume, fb) do idx, band
        copyto!(dst[idx], band)
    end
    return dst
end

"""
    ndfb_decompose(f, volume, fb::NDFB)

Streaming decomposition: calls `f(idx, subband)` once per directional subband,
reusing a single scratch buffer between calls.  `subband` is only valid until
`f` returns — copy or reduce it there.

This is the form to use on real volumes: it holds one subband at a time instead
of `3 * 2^levels` of them.

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(4), 16, 16, 16);

julia> fb = make_ndfb(size(x), Float64; levels = 2, order = 6);

julia> energy = zeros(ndfb_nsubbands(fb));

julia> ndfb_decompose(x, fb) do i, band
           energy[i] = sum(abs2, band)
       end;

julia> abs(sum(energy) / sum(abs2, x) - 1) < 2.0
true
```
"""
function ndfb_decompose(f, volume::AbstractArray{T, 3}, fb::NDFB{T}) where {T <: AbstractFloat}
    hg = fb.fb
    size(volume) == hg.dims || throw(DimensionMismatch("volume size does not match the plan"))
    copyto!(hg.rbuf, volume)
    mul!(hg.spec, hg.fwd, hg.rbuf)
    _stream_subbands(f, fb)
    return nothing
end

function ndfb_decompose(
        f, volume::AbstractArray{Complex{T}, 3}, fb::NDFB{T}
    ) where {T <: AbstractFloat}
    hg = fb.fb
    size(volume) == hg.dims || throw(DimensionMismatch("volume size does not match the plan"))
    # Both spectra are held at once and each subband is finished before moving
    # on, so streaming stays streaming for complex volumes: two spectra and one
    # complex scratch volume, not one real volume per subband.
    _load_complex_spectra!(hg, volume)
    spec2, rbuf2 = _complex_scratch!(hg)
    tmp = _volume_scratch!(hg)
    _walk_subbands(fb) do idx, i, p, q
        _mask_and_invert!(fb, hg.spec, i, p, q, hg.rbuf)
        _mask_and_invert!(fb, spec2, i, p, q, rbuf2)
        @. tmp = complex(hg.rbuf, rbuf2)
        f(idx, tmp)
    end
    return nothing
end

# Walk every (hourglass, p, q) triple, applying its mask to the spectrum already
# sitting in `hg.spec` and handing the inverse transform to `g`.
function _stream_subbands(g, fb::NDFB)
    hg = fb.fb
    _walk_subbands(fb) do idx, i, p, q
        _mask_and_invert!(fb, hg.spec, i, p, q, hg.rbuf)
        g(idx, hg.rbuf)
    end
    return nothing
end

# Iterate the subbands in index order, handing `g` the linear index and the
# (hourglass, p, q) triple that names the mask.  The mask itself is never
# materialised — see `_mask_multiply!`.
function _walk_subbands(g, fb::NDFB)
    na, nb = _wedge_counts(fb)
    for i in 1:3, p in 1:na, q in 1:nb
        g(ndfb_subband_index(fb, i, p, q), i, p, q)
    end
    return nothing
end

function _mask_and_invert!(fb::NDFB, spec, i::Int, p::Int, q::Int, dst)
    hg = fb.fb
    _mask_multiply!(hg.work, fb, spec, i, p, q)
    mul!(dst, hg.inv, hg.work)
    return dst
end

"""
    ndfb_reconstruct(subbands, fb::NDFB) -> volume

Invert [`ndfb_decompose`](@ref).  Perfect reconstruction is exact either way,
but the cost depends on the bank: when the hourglass bank is not tight the masks
are a partition of unity and this is a plain sum, and when it is tight the same
`√W` masks are reapplied, costing a forward transform per subband and one
inverse for the lot.
"""
function ndfb_reconstruct(subbands, fb::NDFB)
    out = similar(first(subbands))
    return ndfb_reconstruct!(out, subbands, fb)
end

"""
    ndfb_reconstruct!(volume, subbands, fb::NDFB) -> volume

In-place [`ndfb_reconstruct`](@ref).
"""
function ndfb_reconstruct!(
        volume::AbstractArray{<:Number, 3}, subbands::Union{AbstractVector, Tuple},
        fb::NDFB
    )
    length(subbands) == ndfb_nsubbands(fb) ||
        throw(DimensionMismatch("expected $(ndfb_nsubbands(fb)) subbands"))
    return ndfb_reconstruct!(k -> subbands[k], volume, fb)
end

"""
    ndfb_reconstruct!(f, volume, fb::NDFB) -> volume

Streaming reconstruction: `f(k)` must return directional subband `k`.  Subbands
are consumed one at a time and in index order, so a caller holding them on disk
or regenerating them never materialises the full set.
"""
function ndfb_reconstruct!(f, volume::AbstractArray{<:Number, 3}, fb::NDFB)
    size(volume) == fb.fb.dims || throw(DimensionMismatch("volume size does not match the plan"))
    hg = fb.fb
    if !hg.tight
        fill!(volume, zero(eltype(volume)))
        for k in 1:ndfb_nsubbands(fb)
            b = f(k)
            @. volume += b
        end
        return volume
    end
    return _ndfb_synthesise!(f, volume, fb)
end

function _ndfb_synthesise!(f, volume::AbstractArray{T, 3}, fb::NDFB{T}) where {T <: AbstractFloat}
    hg = fb.fb
    # `work` is the buffer the inverse plan consumes and nothing else uses during
    # synthesis, so the masked spectra accumulate straight into it — no separate
    # accumulator, and no full-spectrum copy before the transform.
    acc = hg.work
    fill!(acc, zero(eltype(acc)))
    _walk_subbands(fb) do idx, i, p, q
        _load_subband!(hg.rbuf, f(idx))
        mul!(hg.spec, hg.fwd, hg.rbuf)
        _mask_accumulate!(acc, fb, hg.spec, i, p, q)
        nothing
    end
    mul!(hg.rbuf, hg.inv, acc)
    copyto!(volume, hg.rbuf)
    return volume
end

function _ndfb_synthesise!(
        f, volume::AbstractArray{Complex{T}, 3}, fb::NDFB{T}
    ) where {T <: AbstractFloat}
    hg = fb.fb
    # Two accumulators, one per half of the complex volume; `spec` is the scratch
    # both forward transforms land in.  Both accumulators are consumed by the
    # inverse plan, so they double as its (destroyed) input.
    spec2, rbuf2 = _complex_scratch!(hg)
    acc, acc2 = hg.work, spec2
    fill!(acc, zero(eltype(acc)))
    fill!(acc2, zero(eltype(acc2)))
    _walk_subbands(fb) do idx, i, p, q
        band = f(idx)
        @. hg.rbuf = real(band)
        mul!(hg.spec, hg.fwd, hg.rbuf)
        _mask_accumulate!(acc, fb, hg.spec, i, p, q)
        @. hg.rbuf = imag(band)
        mul!(hg.spec, hg.fwd, hg.rbuf)
        _mask_accumulate!(acc2, fb, hg.spec, i, p, q)
        nothing
    end
    mul!(hg.rbuf, hg.inv, acc)
    mul!(rbuf2, hg.inv, acc2)
    @. volume = complex(hg.rbuf, rbuf2)
    return volume
end
