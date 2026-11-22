# Nonsubsampled surfacelet transform.
#
# Multiscale stage: the raised-cosine nonsubsampled pyramid — Lu & Do's
# frequency-domain pyramid without its downsampling, a tight frame
# (the nonsubsampled variant in `fourier_pyramid.jl`).
# Directional stage: the hourglass bank refined into wedges (`ndfb_decompose`).
#
# Every coefficient array keeps the spatial size of the input, so the transform
# is exactly equivariant under circular shifts of the volume, which is why both
# stages filter periodically.  Periodic filtering is a product on the DFT grid,
# so the workspace runs the whole cascade as masks on the rfft grid of the
# input: subband k of scale j is
#
#     irfft( X · Bⱼ · Wᵢ · wₐ · w_b ),
#
# with X the input spectrum, Bⱼ the pyramid's bandpass of level j, Wᵢ the
# hourglass mask and wₐ, w_b the two wedge factors (all square-rooted when the
# bank is tight; `_hg_factor`).  None of these is stored as a volume except the
# hourglass response: Bⱼ = Π_{i<j}ℓᵢ · √(1 - ℓⱼ²), ℓⱼ the separable lowpass of
# level j, is a broadcast of per-axis vectors, and the wedge factors are planes.
# Every mask is real, so the synthesis applies the same masks again: with
# `tight = true` the transform is a Parseval frame and its inverse is its
# adjoint.  The forward forms `Yᵢ = X·Bⱼ·Wᵢ` once per hourglass, so a subband costs one
# multiply by two planes and one inverse FFT; the inverse mirrors it with one
# forward FFT per subband, a per-hourglass sum, and one inverse FFT in total.
#
# The cascade is a mask bank: one group per (scale, hourglass) with the group
# mask Bⱼ·Wᵢ, a bank of wedge masks per group, and the lowpass.  The threaded
# forms run it on MultiscaleFilterBanks' `MaskBankEngine`, which splits the
# subbands into contiguous chunks, one task per FFT buffer (the FFT plans are
# single-threaded); a task forms a group's product when its chunk enters the
# group.  The streaming forms run the engine on one task.
#
# References: Y. M. Lu & M. N. Do, "Multidimensional Directional Filter Banks and
# Surfacelets", IEEE TIP 16(4), 2007 (the NDFB and the surfacelet transform);
# Y. M. Lu & M. N. Do, "A Mapping-Based Design for Nonsubsampled Hourglass Filter
# Banks in Arbitrary Dimensions", IEEE TSP 56(4), 2008 (the nonsubsampled
# hourglass bank); A. L. da Cunha, J. Zhou & M. N. Do, "The Nonsubsampled
# Contourlet Transform: Theory, Design, and Applications", IEEE TIP 15(10), 2006
# (the nonsubsampled pyramid + directional bank structure).

"""
    NSSurfaceletWorkspace

Masks, buffers and FFT plans of an [`NSSurfacelet`](@ref) transform for one
volume size, element type and array type.  Build with
[`make_workspace`](@ref make_workspace(::NSSurfacelet, ::AbstractArray)) and pass
it as `workspace` (or in place of the transform) to keep repeated calls out of
the allocator — the pattern iterative reconstruction algorithms need.

A workspace holds scratch buffers, so it must not serve two transforms at the
same time; it threads internally instead.  `Adapt.adapt(ArrayType, ws)` moves
it to a device, rebuilding its buffers and FFT plans there.
"""
struct NSSurfaceletWorkspace{
        Td <: Number, T <: AbstractFloat, A3 <: AbstractArray{T, 3}, D <: AbstractArray{Td, 3}, PM, BK, LM, E,
    } <: AbstractTransformWorkspace
    dims::NTuple{3, Int}
    transform::NSSurfacelet
    tasks::Union{Int, Nothing}                  # as requested; `nothing` is one per thread
    # Tables (moved by `Adapt`).
    m1::A3                                      # hourglass f₀(M₁), rfft grid
    m2::A3                                      # hourglass f₀(M₂), (1, n₂, n₃)
    wedges::Vector{NTuple{3, NTuple{2, Vector{A3}}}}  # [scale][hourglass][plane][wedge], √ when tight
    pa::Vector{NTuple{3, A3}}                   # Π_{i<j} ℓᵢ per axis, j = 1:J+1
    pl::Vector{NTuple{3, A3}}                   # ℓⱼ per axis, j = 1:J
    band::Vector{A3}                            # [scale] Bⱼ on the rfft grid; empty when not cached
    # The cascade as a mask bank on the tables: one group per (scale, hourglass).
    items::Vector{MaskItem{1}}                  # index (subband,), in `subband_layout` order
    pyramid::PM                                 # MaskBank of `_NSBandMask`s, one per group
    banks::BK                                   # [group] MaskBank of `_WedgeMask`s
    lowpass::LM                                 # `_NSLowpassMask`
    engine::E                                   # MaskBankEngine; a complex volume runs as two real channels
    rbuf::Vector{A3}                            # real data: a volume the FFT plans read and write; else empty
    cbuf::Vector{D}                             # complex data: the subband a streaming callback gets; else empty
end

MultiscaleTransformsBase.transform(ws::NSSurfaceletWorkspace) = ws.transform
Base.eltype(::Type{<:NSSurfaceletWorkspace{Td}}) where {Td} = Td
Base.size(ws::NSSurfaceletWorkspace) = ws.dims

Base.show(io::IO, ws::NSSurfaceletWorkspace) =
    print(io, "NSSurfaceletWorkspace{", eltype(ws), "}(", join(ws.dims, "×"), ", ", ws.transform, ", tasks = ", something(ws.tasks, "nothing"), ")")

_nchannels(::NSSurfaceletWorkspace{<:Real}) = 1
_nchannels(::NSSurfaceletWorkspace{<:Complex}) = 2

function MultiscaleTransformsBase.make_workspace(
        t::NSSurfacelet, ::Type{T0}, dims::Dims{3};
        flags::Integer = FFTW.MEASURE, tasks::Union{Integer, Nothing} = nothing, tune::Bool = false,
        prewarm::Bool = true, cache_bandpass::Bool = true
    ) where {T0 <: Number}
    ws = _ns_host_workspace(t, T0, dims; flags, tasks, cache_bandpass)
    return prewarm ? _prewarm!(ws, Array{eltype(ws), 3}(undef, 0, 0, 0)) : ws
end

# The host workspace, not yet run.
function _ns_host_workspace(
        t::NSSurfacelet, ::Type{T0}, dims::Dims{3}; flags::Integer, tasks::Union{Integer, Nothing}, cache_bandpass::Bool
    ) where {T0 <: Number}
    tasks === nothing || tasks >= 1 || throw(ArgumentError("tasks must be ≥ 1, got $(tasks)"))
    all(>=(1), dims) || throw(ArgumentError("volume size must be positive, got $(dims)"))
    Td = data_eltype(T0)
    T = filter_eltype(Td)
    m1, m2 = _hourglass_tables(T, dims, t.bernstein_order, t.halfband_order)
    wedges = [_ns_wedge_factors(T, dims, l, t) for l in t.direction_levels]
    bins(d) = d == 1 ? (1:(dims[1] ÷ 2 + 1)) : (1:dims[d])
    resp = ntuple(d -> _rc_responses(T, t.scales, dims[d], bins(d)), 3)
    # Bⱼ ≡ 0 exactly when ℓⱼ = 1 wherever Π_{i<j} ℓᵢ > 0, on every axis: the
    # factors are separable, ≤ 1, and 1 at DC.
    # Pyramid level j (1 the finest) is scale J - j + 1.
    for j in 1:t.scales
        all(d -> all(((a, l),) -> a == 0 || l == 1, zip(resp[d][1][j], resp[d][2][j])), 1:3) &&
            throw(ArgumentError("size $(dims) is too small for $(t.scales) scales: scale $(t.scales - j + 1) would be empty"))
    end
    orient(v, d) = reshape(v, ntuple(k -> k == d ? length(v) : 1, 3))
    per_level(r, J) = [ntuple(d -> orient(resp[d][r][j], d), 3) for j in 1:J]
    pa, pl = per_level(1, t.scales + 1), per_level(2, t.scales)
    band = cache_bandpass ? [_bandpass.(pa[j]..., pl[j]...) for j in 1:t.scales] : typeof(m1)[]
    proto = Array{T, 3}(undef, 0, 0, 0)
    ntasks = tasks === nothing ? nothing : Int(tasks)
    return _ns_workspace(Td, dims, t, m1, m2, wedges, pa, pl, band, proto, flags, ntasks)
end

MultiscaleTransformsBase.make_workspace(
    ::NSSurfacelet, ::Type{<:Number}, dims::Dims;
    flags::Integer = FFTW.MEASURE, tasks::Union{Integer, Nothing} = nothing, tune::Bool = false,
    prewarm::Bool = true, cache_bandpass::Bool = true
) = throw(ArgumentError("the surfacelet transforms take volumes (three axes), got size $(dims)"))

"""
    make_workspace(t::NSSurfacelet, T::Type{<:Number}, dims; flags = FFTW.MEASURE, tasks = nothing,
                   tune = false, prewarm = true, cache_bandpass = true) -> NSSurfaceletWorkspace
    make_workspace(t::NSSurfacelet, x::AbstractArray; <same keywords>) -> NSSurfaceletWorkspace

The workspace of `t` for volumes of size `dims` (three axes) and element type
`T` (real or complex; an integer type means `float(T)`).  The second form takes
size, element type and array type from `x`: for a device array the workspace is
built on the host and moved with `Adapt`.

Keywords:

- `flags`: the FFTW planner flags of the host FFT plans; ignored on a device.
- `tasks`: the most tasks a threaded call runs the subbands on, and with them the
  buffers, one set per task; `nothing` is one per Julia thread (never more than
  the threads), `1` a serial workspace.  A device runs one.
- `tune`: accepted and ignored (there is no device batch tuning).
- `prewarm`: run one analysis and synthesis of zeros at construction (on the
  device, for a device `x`), so that the first call is compiled and its buffers
  are touched.  `Adapt.adapt` of a workspace does not run it.
- `cache_bandpass`: see below.

Memory: the hourglass response is one `real(T)` array on the rfft grid,
`(n₁÷2+1) × n₂ × n₃`; the buffers are about `3·tasks + 1` such arrays for
real data and `4·tasks + 2` for complex data (`9` with one task).  With `cache_bandpass = true` the
pyramid's bandpass of each scale is precomputed as well, `t.scales` more such arrays;
it saves recomputing the non-separable factor `√(1 - ℓⱼ²)` once per hourglass
and channel in both directions.  Pass `false` to broadcast it from per-axis
vectors instead, so that everything per-scale is vectors and planes.

Throws an `ArgumentError` when `dims` is too small for `t.scales` scales: a scale
whose bandpass would be identically zero.
"""
function MultiscaleTransformsBase.make_workspace(
        t::NSSurfacelet, x::AbstractArray;
        flags::Integer = FFTW.MEASURE, tasks::Union{Integer, Nothing} = nothing, tune::Bool = false,
        prewarm::Bool = true, cache_bandpass::Bool = true
    )
    ws = make_workspace(t, eltype(x), size(x); flags, tasks, tune, prewarm = false, cache_bandpass)
    ws = _workspace_like(x, ws)
    return prewarm ? _prewarm!(ws, x) : ws
end

# The wedge factors of one scale with depth `L`, in precision `T`: the analysis
# masks' planes, square-rooted when the bank is tight (the weights are ≥ 0 up to
# rounding).
function _ns_wedge_factors(::Type{T}, dims::NTuple{3, Int}, L::Int, t::NSSurfacelet) where {T}
    lev = _wedge_levels(L)
    return ntuple(3) do i
        map(_wedge_masks(dims, i, lev, t.halfband_order, T(t.transition))) do planes
            t.tight ? [_tight_sqrt.(w) for w in planes] : planes
        end
    end
end

# The workspace around its tables, with buffers like `proto` (host or device).
# Host plans use `flags` on one FFTW thread; the engine runs at most `tasks`
# tasks (one per thread when `nothing`) on the host and one on a device.
function _ns_workspace(
        ::Type{Td}, dims::NTuple{3, Int}, t::NSSurfacelet, m1, m2, wedges, pa, pl, band,
        proto::AbstractArray, flags, tasks::Union{Int, Nothing}
    ) where {Td}
    T = eltype(m1)
    half = (dims[1] ÷ 2 + 1, dims[2], dims[3])
    X1 = fill!(similar(proto, Complex{T}, half), zero(Complex{T}))
    r = fill!(similar(proto, T, dims), zero(T))
    # The host plans also read and write the subbands of stacked coefficients,
    # slices whose memory alignment follows from the volume's byte size.
    pflags = _slices_aligned(dims, T) ? flags : flags | FFTW.UNALIGNED
    pfwd, pinv = _fftw_single_threaded(proto) do
        (_fft_plan(plan_rfft, r, pflags), _fft_plan(plan_irfft, X1, pflags, dims[1]))
    end
    items, pyramid, banks = _ns_mask_bank(t, m1, m2, wedges, pa, pl, band)
    lowpass = _NSLowpassMask(pa[end])
    engine = MaskBankEngine(
        banks, items, X1;
        pfwd, pinv, pyramid, lowpass = (lowpass, lowpass), split_complex = Td <: Complex, tasks
    )
    rbuf = Td <: Complex ? typeof(r)[] : [r]
    cbuf = [similar(proto, Td, dims) for _ in 1:(Td <: Complex ? 1 : 0)]
    return NSSurfaceletWorkspace{
        Td, T, typeof(m1), eltype(cbuf), typeof(pyramid), typeof(banks), typeof(lowpass), typeof(engine),
    }(dims, t, tasks, m1, m2, wedges, pa, pl, band, items, pyramid, banks, lowpass, engine, rbuf, cbuf)
end

# The items, group masks and wedge banks of the cascade.  Group g = 3(j - 1) + i
# is hourglass i of scale j (coarse to fine; pyramid level J - j + 1, as `pa`,
# `pl` and `band` are indexed finest first); its items are the wedges (p, q), channel
# (p - 1)·nb + q of its bank, at subband `1 + (subbands of scales < j) +
# _ns_index(i, p, q, na, nb)` of `subband_layout` (the lowpass is subband 1).
function _ns_mask_bank(t::NSSurfacelet, m1, m2, wedges, pa, pl, band)
    J = t.scales
    cached(k) = isempty(band) ? similar(m1, 0, 0, 0) : band[k]
    groups = [(j, i) for j in eachindex(wedges) for i in 1:3]
    ana = [_NSBandMask(m1, m2, cached(J - j + 1), pa[J - j + 1], pl[J - j + 1], i, true, t.tight) for (j, i) in groups]
    syn = [_NSBandMask(m1, m2, cached(J - j + 1), pa[J - j + 1], pl[J - j + 1], i, t.tight, t.tight) for (j, i) in groups]
    offsets = cumsum([1; [_ns_nbands(t, j) for j in 1:(t.scales - 1)]])
    items = MaskItem{1}[]
    banks = map(enumerate(groups)) do (g, (j, i))
        wa, wb = wedges[j][i]
        na, nb = length(wa), length(wb)
        for a in 1:na, b in 1:nb
            push!(items, MaskItem((offsets[j] + _ns_index(i, a, b, na, nb),), g, g, (a - 1) * nb + b))
        end
        MaskBank(
            [_WedgeMask(wa[a], wb[b], false) for a in 1:na for b in 1:nb],
            [_WedgeMask(wa[a], wb[b], !t.tight) for a in 1:na for b in 1:nb]
        )
    end
    return items, MaskBank(ana, syn), banks
end

# The workspace of a call: `workspace` after checking that it was built for `t`
# and data like `x` (its masks are baked in, so a mismatch would silently
# transform with the wrong ones), or a new one with at most `tasks` tasks.
function _ns_workspace_for(workspace, t::NSSurfacelet, x::AbstractArray; tasks::Union{Integer, Nothing} = nothing)
    workspace === nothing && return make_workspace(t, x; prewarm = false, tasks)
    workspace isa NSSurfaceletWorkspace ||
        throw(ArgumentError("an NSSurfacelet needs an NSSurfaceletWorkspace, got $(typeof(workspace))"))
    return check_workspace(workspace, t, data_eltype(x), size(x))
end

# Threading the subband FFTs pays for real and complex data alike, so only
# `threaded = false` turns it off.
_ns_threaded(threaded::Union{Bool, Nothing}) = threaded !== false

# ── Masks ─────────────────────────────────────────────────────────────────────
#
# Mask kinds of the engine (methods of `mask_apply!` and `mask_accumulate!`),
# held as the tables they are computed from.  Each runs on one task: the engine
# threads over the subbands.

# Hourglass factor of subband `i`: √W when tight, W otherwise.
@inline function _hg_factor(m1::T, m2::T, i::Int, tight::Bool) where {T}
    w = _stage1_weight(m1, i) * _stage2_weight(m2, i)
    return tight ? _tight_sqrt(w) : w
end

# The scale-j bandpass Bⱼ = Π_{i<j}ℓᵢ · √(1 - ℓⱼ²) from its per-axis factors.
@inline _bandpass(a1::T, a2::T, a3::T, l1::T, l2::T, l3::T) where {T} =
    a1 * a2 * a3 * sqrt(max(1 - (l1 * l2 * l3)^2, zero(T)))

# The group mask of hourglass `i` at scale j: Bⱼ times the hourglass factor
# (`hg`), or Bⱼ alone — the synthesis of a bank that is not tight, whose
# directional inverse is a bare sum.  Bⱼ is the cached volume `band`, or when
# that is empty a lazy broadcast of its per-axis factors `pa`, `pl`.
struct _NSBandMask{A3 <: AbstractArray}
    m1::A3
    m2::A3
    band::A3
    pa::NTuple{3, A3}
    pl::NTuple{3, A3}
    i::Int
    hg::Bool
    tight::Bool
end

_mask_bandpass(M::_NSBandMask) = isempty(M.band) ? Broadcast.broadcasted(_bandpass, M.pa..., M.pl...) : M.band

MultiscaleFilterBanks.mask_apply!(Z, M::_NSBandMask, X) = _ns_band!(Z, X, _mask_bandpass(M), M, false)
MultiscaleFilterBanks.mask_accumulate!(A, M::_NSBandMask, F) = _ns_band!(A, F, _mask_bandpass(M), M, true)

# dst = src · M, or dst += src · M with `add`, for the bandpass `B` of `M`.
function _ns_band!(dst, src, B, M::_NSBandMask, add::Bool)
    m1, m2, i, tight = M.m1, M.m2, M.i, M.tight
    if M.hg
        add ? (@. dst += src * (B * _hg_factor(m1, m2, i, tight))) : (@. dst = src * (B * _hg_factor(m1, m2, i, tight)))
    else
        add ? (@. dst += src * B) : (@. dst = src * B)
    end
    return dst
end

# The host method walks the columns, as `_wedge_mul!` does: everything but `m1`
# and the cached Bⱼ (or `a1`, `l1`) is a scalar per column.
function _ns_band!(dst::Array, src::Array, B, M::_NSBandMask{<:Array}, add::Bool)
    m1, m2 = M.m1, M.m2
    # The unchecked loop reads every array at dst's indices (m2 along axes 2, 3).
    axes(src) == axes(dst) == axes(m1) == axes(B) && axes(m2, 2) == axes(dst, 2) && axes(m2, 3) == axes(dst, 3) ||
        throw(DimensionMismatch("bandpass: the spectra and masks do not share the grid $(size(dst))"))
    @inbounds for k in axes(dst, 3), l in axes(dst, 2)
        μ2 = m2[1, l, k]
        d, s, μ1 = view(dst, :, l, k), view(src, :, l, k), view(m1, :, l, k)
        _band_col!(d, s, _band_column(B, l, k), μ1, μ2, M.i, add, M.hg, M.tight)
    end
    return dst
end

# One column of `_ns_band!`.  A cached column goes through FastBroadcast's
# `@..`, about twice as fast as Base's broadcast here; the lazy column of an
# uncached Bⱼ is a `Broadcasted`, which only Base's broadcast fuses.
@inline function _band_col!(d, s, b::SubArray, μ1, μ2, i::Int, add::Bool, hg::Bool, tight::Bool)
    if hg
        add ? (@.. d += s * (b * _hg_factor(μ1, μ2, i, tight))) : (@.. d = s * (b * _hg_factor(μ1, μ2, i, tight)))
    else
        add ? (@.. d += s * b) : (@.. d = s * b)
    end
    return d
end

@inline function _band_col!(d, s, b::Broadcast.Broadcasted, μ1, μ2, i::Int, add::Bool, hg::Bool, tight::Bool)
    if hg
        add ? (@. d += s * (b * _hg_factor(μ1, μ2, i, tight))) : (@. d = s * (b * _hg_factor(μ1, μ2, i, tight)))
    else
        add ? (@. d += s * b) : (@. d = s * b)
    end
    return d
end

# Column (:, l, k) of Bⱼ, cached or broadcast from its factors.
@inline _band_column(B::Array, l::Int, k::Int) = view(B, :, l, k)

@inline function _band_column(B::Broadcast.Broadcasted, l::Int, k::Int)
    a1, a2, a3, l1, l2, l3 = B.args
    return Broadcast.broadcasted(_bandpass, vec(a1), a2[1, l, 1], a3[1, 1, k], vec(l1), l2[1, l, 1], l3[1, 1, k])
end

# The lowpass of the last pyramid level, a₁a₂a₃ from its per-axis factors.
struct _NSLowpassMask{A3 <: AbstractArray}
    a::NTuple{3, A3}
end

MultiscaleFilterBanks.mask_apply!(Z, M::_NSLowpassMask, X) = _ns_lowpass!(Z, X, M.a, false)
MultiscaleFilterBanks.mask_accumulate!(A, M::_NSLowpassMask, F) = _ns_lowpass!(A, F, M.a, true)

# Y = X · a₁a₂a₃, or Y += with `add`.  Same column walk as `_ns_band!`.
_ns_lowpass!(Y, X, a, add::Bool) = add ? (Y .+= X .* a[1] .* a[2] .* a[3]) : (Y .= X .* a[1] .* a[2] .* a[3])

function _ns_lowpass!(Y::Array, X::Array, a::NTuple{3, Array}, add::Bool)
    a1, a2, a3 = a
    axes(X) == axes(Y) && length(a1) == size(Y, 1) && axes(a2, 2) == axes(Y, 2) && axes(a3, 3) == axes(Y, 3) ||
        throw(DimensionMismatch("lowpass: the spectra and factors do not share the grid $(size(Y))"))
    a1v = vec(a1)
    @inbounds for k in axes(Y, 3), l in axes(Y, 2)
        a23 = a2[1, l, 1] * a3[1, 1, k]
        y, x = view(Y, :, l, k), view(X, :, l, k)
        add ? (@.. y += x * (a1v * a23)) : (@.. y = x * (a1v * a23))
    end
    return Y
end

# Wedge (p, q) of an hourglass: the product of its two oriented planes `wa`, `wb`
# (singleton on one axis each), or with `unit` the constant 1 — the synthesis of a
# bank that is not tight.
struct _WedgeMask{A3 <: AbstractArray}
    wa::A3
    wb::A3
    unit::Bool
end

MultiscaleFilterBanks.mask_apply!(Z, M::_WedgeMask, Y) = M.unit ? copyto!(Z, Y) : _wedge_mul!(Z, Y, M.wa, M.wb, false)
MultiscaleFilterBanks.mask_accumulate!(A, M::_WedgeMask, F) = M.unit ? (@.. A += F; A) : _wedge_mul!(A, F, M.wa, M.wb, true)

# The engine passes a scale with the group and lowpass products; it is 1 here
# (the inverse plan is normalized).
function MultiscaleFilterBanks.mask_apply!(Z, M::Union{_NSBandMask, _NSLowpassMask, _WedgeMask}, X, s)
    MultiscaleFilterBanks.mask_apply!(Z, M, X)
    isone(s) || (Z .*= s)
    return Z
end

# dst = Y·wa·wb, or dst += Y·wa·wb when `add`.  On the host the 3-D broadcast
# does not vectorize across the singleton axes — it runs at a tenth of a plain
# copy — so the host method walks the columns, each a contiguous 1-D broadcast in
# which a plane contributes a column or a scalar, through FastBroadcast's `@..`
# (15–25% faster than Base's broadcast on these columns).
_wedge_mul!(dst, Y, wa, wb, add::Bool) = add ? (dst .+= Y .* wa .* wb) : (dst .= Y .* wa .* wb)

function _wedge_mul!(dst::Array{T, 3}, Y::Array{T, 3}, wa::Array{R, 3}, wb::Array{R, 3}, add::Bool) where {T, R}
    # The unchecked loop reads each plane along an axis of dst's length or a singleton one.
    fits(w) = all(d -> size(w, d) in (1, size(dst, d)), 1:3)
    axes(Y) == axes(dst) && fits(wa) && fits(wb) ||
        throw(DimensionMismatch("wedge: the spectra and planes do not share the grid $(size(dst))"))
    @inbounds for k in axes(dst, 3), j in axes(dst, 2)
        a = view(wa, :, min(j, size(wa, 2)), min(k, size(wa, 3)))
        b = view(wb, :, min(j, size(wb, 2)), min(k, size(wb, 3)))
        _wedge_col!(view(dst, :, j, k), view(Y, :, j, k), a, b, add)
    end
    return dst
end

@inline function _wedge_col!(d, y, a, b, add::Bool)
    if length(a) == 1 && length(b) == 1
        s = a[1] * b[1]
        add ? (@.. d += y * s) : (@.. d = y * s)
    elseif length(a) == 1
        s = a[1]
        add ? (@.. d += y * b * s) : (@.. d = y * b * s)
    elseif length(b) == 1
        s = b[1]
        add ? (@.. d += y * a * s) : (@.. d = y * a * s)
    else
        add ? (@.. d += y * a * b) : (@.. d = y * a * b)
    end
    return d
end

# The (hourglass, p, q) ↦ subband index map (`ndfb_subband_index`'s order).
_ns_index(i::Int, p::Int, q::Int, na::Int, nb::Int) = ((i - 1) * na + (p - 1)) * nb + q

# Stacked coefficients as the engine's container: subband i is the slice
# c[:, :, :, i], the lowpass the first.
struct _NSStack{A <: AbstractArray}
    c::A
end

MultiscaleFilterBanks.mask_bank_subband(s::_NSStack, (i,)::Tuple{Int}) = view(s.c, :, :, :, i)
MultiscaleFilterBanks.mask_bank_coarse(s::_NSStack) = view(s.c, :, :, :, 1)

# Whether the slices along a trailing axis of an array of volumes of size `dims`
# and element type `T` keep the array's 64-byte alignment, so that FFT plans
# made on aligned buffers apply to them.
_slices_aligned(dims::Dims, ::Type{T}) where {T} = (prod(dims) * sizeof(T)) % 64 == 0

# ── Drivers ───────────────────────────────────────────────────────────────────

# The threaded forms run on the engine.  It reads a real volume and writes and
# reads real subbands with the FFT plans directly, so those must be arrays of
# the plans' type; a complex volume and its subbands go through the engine's
# channel buffers.  Coefficients of another array type take the streaming path.
_ns_engine_arrays(ws::NSSurfaceletWorkspace, c::AbstractArray) = _nchannels(ws) == 2 || _is_stack_of(c, ws.rbuf[1])

# Whether the slices c[:, :, :, i] are arrays the FFT plans of `r` apply to: the
# strided views of a host stack, or device arrays (a contiguous view of a device
# array is one).
_is_stack_of(::Array{T, 4}, ::Array{T, 3}) where {T} = true
_is_stack_of(c::AbstractArray{<:Any, 4}, r::AbstractArray{<:Any, 3}) = !(r isa Array) && typeof(view(c, :, :, :, 1)) === typeof(r)
_is_stack_of(c, r) = false

# A real volume the plans can read: `x` itself, or a copy in `ws.rbuf[1]`
# (which the engine does not use).
_ns_engine_input(ws::NSSurfaceletWorkspace, x::AbstractArray) =
    _nchannels(ws) == 2 || x isa eltype(ws.rbuf) ? x : (ws.rbuf[1] .= x)

# The streaming forms run the engine on one task.  A real volume, input or
# output, and real subbands go through `ws.rbuf[1]` unless the plans can read
# or write them directly; the engine reads its input into a spectrum before
# writing a subband, and every subband before writing the output, so one buffer
# serves for all.

# Streaming forward cascade: calls `f(i, band)` for each directional subband in
# the workspace's item order (increasing `i`), then for the lowpass (`i = 1`).
function _ns_forward!(f::F, volume::AbstractArray, ws::NSSurfaceletWorkspace) where {F}
    out = _nchannels(ws) == 1 ? ws.rbuf[1] : ws.cbuf[1]
    sink(index, band) = f(index === nothing ? 1 : index[1], band)
    mask_bank_stream_forward!(sink, out, _ns_engine_input(ws, volume), ws.engine)
    return nothing
end

# Streaming inverse cascade; `source(i)` returns subband `i` (`subband_layout`
# order), once: the directional subbands in increasing `i`, then the lowpass.
function _ns_inverse!(source::F, volume::AbstractArray, ws::NSSurfaceletWorkspace) where {F}
    function band(index)
        i = index === nothing ? 1 : index[1]
        b = _ns_checked_band(source(i), ws, i)
        return _nchannels(ws) == 2 || b isa eltype(ws.rbuf) ? b : (ws.rbuf[1] .= b)
    end
    y = _nchannels(ws) == 2 || volume isa eltype(ws.rbuf) ? volume : ws.rbuf[1]
    mask_bank_stream_inverse!(band, y, ws.engine)
    y === volume || copyto!(volume, y)
    return volume
end

# The subbands arrive one at a time, so each size is checked as it comes.
function _ns_checked_band(band::AbstractArray, ws::NSSurfaceletWorkspace, i::Int)
    size(band) == ws.dims || throw(DimensionMismatch("subband $(i) has size $(size(band)), expected $(ws.dims)"))
    return band
end

# ── Interface ─────────────────────────────────────────────────────────────────

MultiscaleTransformsBase.supports_streaming(::NSSurfacelet) = true

_check_volume(dims::Dims{3}) = dims
_check_volume(dims::Dims) = throw(ArgumentError("the surfacelet transforms take volumes (three axes), got size $(dims)"))

"""
    subband_layout(t::NSSurfacelet, dims) -> Vector{@NamedTuple{scale::Int, direction::Int}}

What each subband of `t` is: the lowpass first, as `(scale = 0, direction = 0)`,
then the directional subbands of scale 1 (the coarsest), 2, …, `t.scales` (the
finest), each as `(scale = j, direction = k)` with `k` in `1:3·2^t.direction_levels[j]`: hourglass 1, 2, 3 in
turn, and inside each the wedges in `Surfacelets.ndfb_subband_index` order.
Every subband has size `dims`.
"""
function MultiscaleTransformsBase.subband_layout(t::NSSurfacelet, dims::Dims)
    _check_volume(dims)
    layout = [(scale = 0, direction = 0)]
    for j in 1:t.scales, k in 1:_ns_nbands(t, j)
        push!(layout, (scale = j, direction = k))
    end
    return layout
end

MultiscaleTransformsBase.nsubbands(t::NSSurfacelet, dims::Dims) = (_check_volume(dims); 1 + sum(j -> _ns_nbands(t, j), 1:t.scales))

"""
    similar_coefficients(t::NSSurfacelet, x::AbstractArray) -> AbstractArray

Uninitialized coefficients of `t` for volumes like `x`: an array like `x` of
size `(size(x)..., nsubbands(t, size(x)))` and element type `float(eltype(x))`,
whose slice `[:, :, :, i]` is subband `i` of
[`subband_layout`](@ref subband_layout(::NSSurfacelet, ::Dims)), the lowpass first.
"""
function MultiscaleTransformsBase.similar_coefficients(t::NSSurfacelet, x::AbstractArray)
    dims = _check_volume(size(x))
    return similar(x, data_eltype(x), (dims..., nsubbands(t, dims)))
end

MultiscaleTransformsBase.subband(::NSSurfacelet, c::AbstractArray{<:Any, 4}, i::Integer) = view(c, :, :, :, i)
MultiscaleTransformsBase.signal_size(::NSSurfacelet, c::AbstractArray{<:Any, 4}) = size(c)[1:3]

function _ns_check_coefficients(c::AbstractArray, t::NSSurfacelet, dims::NTuple{3, Int})
    expect = (dims..., nsubbands(t, dims))
    size(c) == expect || throw(DimensionMismatch("coefficients of size $(size(c)), expected $(expect)"))
    return c
end

function MultiscaleTransformsBase.analyze!(
        c::AbstractArray, t::NSSurfacelet, x::AbstractArray;
        workspace = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    ws = _ns_workspace_for(workspace, t, x)
    _ns_check_coefficients(c, t, ws.dims)
    eltype(ws) <: Complex && !(eltype(c) <: Complex) &&
        throw(ArgumentError("coefficients of element type $(eltype(c)), the workspace is for $(eltype(ws))"))
    if _ns_engine_arrays(ws, c)
        mask_bank_forward!(_NSStack(c), _ns_engine_input(ws, x), ws.engine; threaded = _ns_threaded(threaded))
    else
        _ns_forward!((i, b) -> copyto!(view(c, :, :, :, i), b), x, ws)
    end
    return c
end

function MultiscaleTransformsBase.synthesize!(
        x::AbstractArray, t::NSSurfacelet, c::AbstractArray;
        workspace = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    ws = _ns_workspace_for(workspace, t, x)
    _ns_check_coefficients(c, t, ws.dims)
    _ns_engine_arrays(ws, c) || return _ns_inverse!(i -> view(c, :, :, :, i), x, ws)
    # A real output the inverse plan cannot write is filled through `ws.rbuf[1]`.
    y = _nchannels(ws) == 2 || x isa eltype(ws.rbuf) ? x : ws.rbuf[1]
    mask_bank_inverse!(y, _NSStack(c), ws.engine; threaded = _ns_threaded(threaded))
    y === x || copyto!(x, y)
    return x
end

"""
    analyze(f, t::NSSurfacelet, x; workspace = nothing, threaded = nothing) -> nothing

Streaming analysis: calls `f(i, band)` once for every subband `i` of
[`subband_layout(t, size(x))`](@ref subband_layout), the directional subbands
in increasing `i` and then the lowpass (`i = 1`), reusing one buffer between
calls.  `band` is valid only until `f` returns — copy or reduce it there.  The
analysis runs serially: `f` is called from the calling task, one subband at a
time and in that order, so it need not be thread-safe.  `threaded` does not
apply and is ignored.

This is the form to use at working sizes.  At 256³ `Float32` with `scales = 3` and
`direction_levels = [2, 2, 3]` the full coefficient set is 3.1 GB, while streaming
holds the workspace's buffers: without a `workspace` the call builds a serial
one (`tasks = 1`), a few half-spectrum volumes plus the plans and masks (see
[`make_workspace`](@ref make_workspace(::NSSurfacelet, ::AbstractArray)) for the
count).  A workspace keeps one set of buffers per task, so build the one you
stream with as `make_workspace(t, x; tasks = 1)`.

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(6), 16, 16, 16);

julia> t = NSSurfacelet(direction_levels = [1, 1], bernstein_order = 6);

julia> energy = zeros(nsubbands(t, size(x)));

julia> analyze(t, x) do i, band
           energy[i] = sum(abs2, band)
       end

julia> sum(energy) ≈ sum(abs2, x)
true
```
"""
function MultiscaleTransformsBase.analyze(f, t::NSSurfacelet, x::AbstractArray; workspace = nothing, threaded::Union{Bool, Nothing} = nothing)
    ws = _ns_workspace_for(workspace, t, x; tasks = 1)
    return _ns_forward!(f, x, ws)
end

"""
    synthesize!(f, x, t::NSSurfacelet; workspace = nothing, threaded = nothing) -> x

Streaming synthesis into `x`: `f(i)` must return subband `i` of
[`subband_layout(t, size(x))`](@ref subband_layout).  It is called once per
subband, the directional subbands in increasing `i` and then the lowpass, so a
caller holding the subbands on disk or regenerating them never materializes the
full set.  The synthesis runs serially: `f` is called from the calling task,
one subband at a time and in that order.  `threaded` does not apply and is
ignored.
"""
function MultiscaleTransformsBase.synthesize!(f, x::AbstractArray, t::NSSurfacelet; workspace = nothing, threaded::Union{Bool, Nothing} = nothing)
    ws = _ns_workspace_for(workspace, t, x; tasks = 1)
    return _ns_inverse!(f, x, ws)
end
