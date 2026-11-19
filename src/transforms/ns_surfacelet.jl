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
# group.  The streaming forms run the same masks on one task.
#
# References: Y. M. Lu & M. N. Do, "Multidimensional Directional Filter Banks and
# Surfacelets", IEEE TIP 16(4), 2007 (the NDFB and the surfacelet transform);
# Y. M. Lu & M. N. Do, "A Mapping-Based Design for Nonsubsampled Hourglass Filter
# Banks in Arbitrary Dimensions", IEEE TSP 56(4), 2008 (the nonsubsampled
# hourglass bank); A. L. da Cunha, J. Zhou & M. N. Do, "The Nonsubsampled
# Contourlet Transform: Theory, Design, and Applications", IEEE TIP 15(10), 2006
# (the nonsubsampled pyramid + directional bank structure).

"""
    NSSurfaceletWorkspace{Td, T}

Masks, buffers and FFT plans for the nonsubsampled surfacelet transform at one
volume size and element type.  Build with [`make_ns_surfacelet_workspace`](@ref)
and pass it to `ns_surfacelet_forward` / `ns_surfacelet_inverse` to keep repeated
calls out of the allocator — the pattern iterative reconstruction algorithms
need.

A workspace holds scratch buffers, so it must not serve two transforms at the
same time; it threads internally instead, one task per Julia thread.
`Adapt.adapt(ArrayType, ws)` moves it to a device, rebuilding its buffers and FFT
plans there.
"""
struct NSSurfaceletWorkspace{
        Td <: Number, T <: AbstractFloat, A3 <: AbstractArray{T, 3}, C3 <: AbstractArray{Complex{T}, 3},
        D <: AbstractArray{Td, 3}, PM, BK, LM, E, P, Q,
    }
    dims::NTuple{3, Int}
    params::NSSurfaceletParams{T}
    # Tables (moved by `Adapt`).
    m1::A3                                      # hourglass f₀(M₁), rfft grid
    m2::A3                                      # hourglass f₀(M₂), (1, n₂, n₃)
    wedges::Vector{NTuple{3, NTuple{2, Vector{A3}}}}  # [scale][hourglass][plane][wedge], √ when tight
    pa::Vector{NTuple{3, A3}}                   # Π_{i<j} ℓᵢ per axis, j = 1:J+1
    pl::Vector{NTuple{3, A3}}                   # ℓⱼ per axis, j = 1:J
    band::Vector{A3}                            # [scale] Bⱼ on the rfft grid; empty when not cached
    # The cascade as a mask bank on the tables: one group per (scale, hourglass).
    items::Vector{MaskItem{2}}                  # index (scale, subband)
    pyramid::PM                                 # MaskBank of `_NSBandMask`s, one per group
    banks::BK                                   # [group] MaskBank of `_WedgeMask`s
    lowpass::LM                                 # `_NSLowpassMask`
    engine::E                                   # MaskBankEngine of the threaded forms
    # Buffers of the streaming forms, which run on one task.  Channel 1's
    # spectrum, group and FFT buffers are the engine's own, idle meanwhile.  A
    # complex volume is two real channels, filtered alike, which keeps
    # `T(x + i·y) == T(x) + i·T(y)` exact.
    X::Vector{C3}                               # [channel] input spectrum / synthesis sum
    Y::Vector{C3}                               # [channel] group product / its synthesis
    work::C3                                    # FFT buffer (consumed by `pinv`)
    rbuf::Vector{A3}                            # [channel] real volume
    cbuf::Vector{D}                             # complex subband handed to a callback (complex only)
    coarse::D
    pfwd::P                                     # rfft, single-threaded
    pinv::Q                                     # irfft (destroys its input)
end

Base.eltype(::NSSurfaceletWorkspace{Td}) where {Td} = Td

_nchannels(::NSSurfaceletWorkspace{<:Real}) = 1
_nchannels(::NSSurfaceletWorkspace{<:Complex}) = 2

"""
    make_ns_surfacelet_workspace(params::NSSurfaceletParams, dims::NTuple{3,Int},
                         ::Type{Td}=Float64; flags=FFTW.MEASURE, cache_bandpass=true)

Allocate the workspace for `params` at volume size `dims` and element type `Td`
(real or complex).

Memory: the hourglass response is one `real(Td)` array on the rfft grid,
`(n₁÷2+1) × n₂ × n₃`; the buffers are about `3·nthreads + 1` such arrays for
real data and `4·nthreads + 5` for complex data, one set per Julia thread.  With
`cache_bandpass = true` the pyramid's bandpass of each scale is precomputed as
well, `J` more such arrays; it saves recomputing the non-separable factor
`√(1 - ℓⱼ²)` once per hourglass and channel in both directions.  Pass `false`
to broadcast it from per-axis vectors instead, so that everything per-scale is
vectors and planes.

Throws an `ArgumentError` when `dims` is too small for `params.J` scales: a
scale whose bandpass would be identically zero.
"""
function make_ns_surfacelet_workspace(
        params::NSSurfaceletParams, dims::NTuple{3, Int}, ::Type{Td} = Float64;
        flags::Integer = FFTW.MEASURE, cache_bandpass::Bool = true
    ) where {Td <: Number}
    T = filter_eltype(Td)
    p = _convert_params(T, params)
    m1, m2 = _hourglass_tables(T, dims, p.order, p.poly)
    wedges = [_ns_wedge_factors(dims, l, p) for l in p.L_array]
    bins(d) = d == 1 ? (1:(dims[1] ÷ 2 + 1)) : (1:dims[d])
    resp = ntuple(d -> _rc_responses(T, p.J, dims[d], bins(d)), 3)
    # Bⱼ ≡ 0 exactly when ℓⱼ = 1 wherever Π_{i<j} ℓᵢ > 0, on every axis: the
    # factors are separable, ≤ 1, and 1 at DC.
    for j in 1:p.J
        all(d -> all(((a, l),) -> a == 0 || l == 1, zip(resp[d][1][j], resp[d][2][j])), 1:3) &&
            throw(ArgumentError("size $(dims) is too small for $(p.J) scales: scale $(j) would be empty"))
    end
    orient(v, d) = reshape(v, ntuple(k -> k == d ? length(v) : 1, 3))
    per_level(r, J) = [ntuple(d -> orient(resp[d][r][j], d), 3) for j in 1:J]
    pa, pl = per_level(1, p.J + 1), per_level(2, p.J)
    band = cache_bandpass ? [_bandpass.(pa[j]..., pl[j]...) for j in 1:p.J] : typeof(m1)[]
    proto = Array{T, 3}(undef, 0, 0, 0)
    return _ns_workspace(Td, dims, p, m1, m2, wedges, pa, pl, band, proto, flags)
end

make_ns_surfacelet_workspace(params::NSSurfaceletParams, dims::Vararg{Int, 3}; kwargs...) =
    make_ns_surfacelet_workspace(params, dims; kwargs...)

# The wedge factors of one scale with depth `L`: the analysis masks' planes,
# square-rooted when the bank is tight (the weights are ≥ 0 up to rounding).
function _ns_wedge_factors(dims::NTuple{3, Int}, L::Int, p::NSSurfaceletParams{T}) where {T}
    lev = _wedge_levels(L)
    return ntuple(3) do i
        map(_wedge_masks(dims, i, lev, p.poly, p.transition)) do planes
            p.tight ? [_tight_sqrt.(w) for w in planes] : planes
        end
    end
end

# The workspace around its tables, with buffers like `proto` (host or device).
# Host plans use `flags` on one FFTW thread; the engine runs one task per thread
# on the host and one on a device.
function _ns_workspace(
        ::Type{Td}, dims::NTuple{3, Int}, p::NSSurfaceletParams{T}, m1, m2, wedges, pa, pl, band,
        proto::AbstractArray, flags
    ) where {Td, T}
    nch = Td <: Complex ? 2 : 1
    half = (dims[1] ÷ 2 + 1, dims[2], dims[3])
    spectrum() = fill!(similar(proto, Complex{T}, half), zero(Complex{T}))
    volume() = fill!(similar(proto, T, dims), zero(T))
    rbuf = [volume() for _ in 1:nch]
    X1 = spectrum()
    pfwd, pinv = _fftw_single_threaded(proto) do
        (_fft_plan(plan_rfft, rbuf[1], flags), _fft_plan(plan_irfft, X1, flags, dims[1]))
    end
    items, pyramid, banks = _ns_mask_bank(p, m1, m2, wedges, pa, pl, band)
    lowpass = _NSLowpassMask(pa[end])
    engine = MaskBankEngine(
        banks, items, X1, pfwd, pinv;
        pyramid, lowpass = (lowpass, lowpass), split_complex = Td <: Complex
    )
    X = pushfirst!([spectrum() for _ in 2:nch], X1)
    Y = pushfirst!([spectrum() for _ in 2:nch], engine.groups[1])
    coarse = similar(proto, Td, dims)
    cbuf = Td <: Complex ? [similar(proto, Td, dims)] : typeof(coarse)[]
    return NSSurfaceletWorkspace{
        Td, T, typeof(m1), typeof(X1), typeof(coarse), typeof(pyramid), typeof(banks), typeof(lowpass),
        typeof(engine), typeof(pfwd), typeof(pinv),
    }(
        dims, p, m1, m2, wedges, pa, pl, band, items, pyramid, banks, lowpass, engine,
        X, Y, engine.buffers[1], rbuf, cbuf, coarse, pfwd, pinv
    )
end

# The items, group masks and wedge banks of the cascade.  Group g = 3(j - 1) + i
# is hourglass i of scale j; its items are the wedges (p, q), channel
# (p - 1)·nb + q of its bank, at subband index `_ns_index(i, p, q, na, nb)`.
function _ns_mask_bank(p::NSSurfaceletParams, m1, m2, wedges, pa, pl, band)
    cached(j) = isempty(band) ? similar(m1, 0, 0, 0) : band[j]
    groups = [(j, i) for j in eachindex(wedges) for i in 1:3]
    ana = [_NSBandMask(m1, m2, cached(j), pa[j], pl[j], i, true, p.tight) for (j, i) in groups]
    syn = [_NSBandMask(m1, m2, cached(j), pa[j], pl[j], i, p.tight, p.tight) for (j, i) in groups]
    items = MaskItem{2}[]
    banks = map(enumerate(groups)) do (g, (j, i))
        wa, wb = wedges[j][i]
        na, nb = length(wa), length(wb)
        for a in 1:na, b in 1:nb
            push!(items, MaskItem((j, _ns_index(i, a, b, na, nb)), g, g, (a - 1) * nb + b))
        end
        MaskBank(
            [_WedgeMask(wa[a], wb[b], false) for a in 1:na for b in 1:nb],
            [_WedgeMask(wa[a], wb[b], !p.tight) for a in 1:na for b in 1:nb]
        )
    end
    return items, MaskBank(ana, syn), banks
end

# Build a workspace on demand when the caller did not supply one, or check that
# the one supplied was built for `params`: its masks are baked in, so a mismatch
# would silently transform with the wrong ones.
function _ns_surfacelet_workspace(
        ws::Union{NSSurfaceletWorkspace, Nothing}, params::NSSurfaceletParams,
        dims::NTuple{3, Int}, ::Type{Td}
    ) where {Td}
    ws === nothing && return make_ns_surfacelet_workspace(params, dims, Td)
    ws.dims == dims || throw(DimensionMismatch("workspace size does not match the volume"))
    eltype(ws) === Td || throw(ArgumentError("workspace element type does not match the volume"))
    q = ws.params
    q.J == params.J || throw(ArgumentError("workspace was built for a different number of scales"))
    q.L_array == params.L_array ||
        throw(ArgumentError("workspace was built for L_array = $(q.L_array), not $(params.L_array)"))
    q.tight == params.tight ||
        throw(ArgumentError("workspace was built with tight = $(q.tight), not $(params.tight)"))
    (q.order, q.poly) == (params.order, params.poly) && q.transition == eltype(q)(params.transition) ||
        throw(ArgumentError("workspace was built for a different hourglass or wedge design"))
    return ws
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
# (the inverse plan is normalised).
function MultiscaleFilterBanks.mask_apply!(Z, M::Union{_NSBandMask, _NSLowpassMask, _WedgeMask}, X, s)
    MultiscaleFilterBanks.mask_apply!(Z, M, X)
    isone(s) || (Z .*= s)
    return Z
end

# dst = Y·wa·wb, or dst += Y·wa·wb when `add`.  On the host the 3-D broadcast
# does not vectorise across the singleton axes — it runs at a tenth of a plain
# copy — so the host method walks the columns, each a contiguous 1-D broadcast in
# which a plane contributes a column or a scalar, through FastBroadcast's `@..`
# (15–25% faster than Base's broadcast on these columns).
_wedge_mul!(dst, Y, wa, wb, add::Bool) = add ? (dst .+= Y .* wa .* wb) : (dst .= Y .* wa .* wb)

function _wedge_mul!(dst::Array{T, 3}, Y::Array{T, 3}, wa::Array{R, 3}, wb::Array{R, 3}, add::Bool) where {T, R}
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

MultiscaleFilterBanks.mask_bank_subband(c::NSSurfaceletCoefficients, (j, k)::NTuple{2, Int}) = c.subbands[j][k]

# Real channel `c` of a volume: the volume itself, or its real or imaginary part.
@inline _channel(x::Number, c::Int) = c == 1 ? real(x) : imag(x)

# `spec = rfft(channel c of x)`, through `rbuf` unless `x` can feed the plan directly.
function _ns_spectrum!(spec, x::AbstractArray, rbuf::AbstractArray, ws::NSSurfaceletWorkspace, c::Int)
    if eltype(x) <: Real && x isa typeof(rbuf)
        mul!(spec, ws.pfwd, x)
    else
        rbuf .= _channel.(x, c)
        mul!(spec, ws.pfwd, rbuf)
    end
    return spec
end

# ── Drivers ───────────────────────────────────────────────────────────────────

# The threaded forms run on the engine.  It reads a real volume and writes and
# reads real subbands with the FFT plans directly, so those must be arrays of
# the plans' type; a complex volume and its subbands go through the engine's
# channel buffers.  Coefficients of another array type take the streaming path.
_ns_engine_arrays(ws::NSSurfaceletWorkspace, coeffs) =
    _nchannels(ws) == 2 || (
    coeffs.coarse isa eltype(ws.rbuf) && all(s -> all(b -> b isa eltype(ws.rbuf), s), coeffs.subbands)
)

# A real volume the plans can read: `x` itself, or a copy in `ws.rbuf[1]`
# (which the engine does not use).
_ns_engine_input(ws::NSSurfaceletWorkspace, x::AbstractArray) =
    _nchannels(ws) == 2 || x isa eltype(ws.rbuf) ? x : (ws.rbuf[1] .= x)

# Streaming forward cascade, on one task: calls `f(j, k, band)` for each subband
# in the workspace's item order.  Returns `ws.coarse`.
function _ns_forward!(f::F, volume::AbstractArray, ws::NSSurfaceletWorkspace) where {F}
    nch = _nchannels(ws)
    Z = ws.work
    for c in 1:nch
        _ns_spectrum!(ws.X[c], volume, ws.rbuf[c], ws, c)
    end
    g = 0
    for it in ws.items
        if it.pyramid != g
            g = it.pyramid
            for c in 1:nch
                mask_apply!(ws.Y[c], ws.pyramid.ana[g], ws.X[c])
            end
        end
        H = ws.banks[it.bank].ana[it.channel]
        for c in 1:nch
            mask_apply!(Z, H, ws.Y[c])
            mul!(ws.rbuf[c], ws.pinv, Z)
        end
        j, k = it.index
        f(j, k, nch == 1 ? ws.rbuf[1] : _ns_combine!(ws.cbuf[1], ws.rbuf))
    end
    for c in 1:nch
        mask_apply!(Z, ws.lowpass, ws.X[c])
        mul!(ws.rbuf[c], ws.pinv, Z)
    end
    return _ns_combine!(ws.coarse, ws.rbuf)
end

# Streaming inverse cascade, on one task; `source(j, k)` returns subband k of
# scale j, once.
function _ns_inverse!(source::F, volume::AbstractArray, coarse::AbstractArray, ws::NSSurfaceletWorkspace) where {F}
    nch = _nchannels(ws)
    Z = ws.work
    for c in 1:nch
        fill!(ws.X[c], zero(eltype(ws.X[c])))
    end
    g = 0
    for it in ws.items
        if it.pyramid != g
            for c in 1:nch
                g == 0 || mask_accumulate!(ws.X[c], ws.pyramid.syn[g], ws.Y[c])
                fill!(ws.Y[c], zero(eltype(ws.Y[c])))
            end
            g = it.pyramid
        end
        j, k = it.index
        band = source(j, k)
        # The subbands arrive one at a time, so each size is checked here.
        size(band) == ws.dims || throw(DimensionMismatch("subband ($(j), $(k)) has the wrong size"))
        H = ws.banks[it.bank].syn[it.channel]
        for c in 1:nch
            _ns_spectrum!(Z, band, ws.rbuf[c], ws, c)
            mask_accumulate!(ws.Y[c], H, Z)
        end
    end
    for c in 1:nch
        g == 0 || mask_accumulate!(ws.X[c], ws.pyramid.syn[g], ws.Y[c])
        _ns_spectrum!(Z, coarse, ws.rbuf[c], ws, c)
        mask_accumulate!(ws.X[c], ws.lowpass, Z)
        mul!(ws.rbuf[c], ws.pinv, ws.X[c])
    end
    return _ns_combine!(volume, ws.rbuf)
end

# dst = the volume of channels `r` (one real channel, or real and imaginary).
_ns_combine!(dst::AbstractArray, r::AbstractVector) = length(r) == 1 ? (dst .= r[1]) : (dst .= complex.(r[1], r[2]))

# ── Public API ────────────────────────────────────────────────────────────────

"""
    ns_surfacelet_forward(volume, params::NSSurfaceletParams; workspace=nothing, threaded = nothing) -> NSSurfaceletCoefficients

Nonsubsampled surfacelet transform.  Invariant under circular shifts of the
input; every subband keeps the size of `volume`.  The subbands are computed in
parallel unless `threaded = false`.

See [`ns_surfacelet_forward(f, volume, params)`](@ref) for the streaming form, which
does not hold all subbands at once.

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(5), 16, 16, 16);

julia> p = NSSurfaceletParams(J = 2, L_array = [1, 2], order = 6);

julia> c = ns_surfacelet_forward(x, p);

julia> length(c.subbands[1]), length(c.subbands[2])
(6, 12)

julia> maximum(abs, ns_surfacelet_inverse(c, p) .- x) < 1e-10
true
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
function ns_surfacelet_forward(
        volume::AbstractArray{<:Number, 3}, params::NSSurfaceletParams;
        workspace::Union{NSSurfaceletWorkspace, Nothing} = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    # Pin the container element type to the *data* type the transform works in,
    # not `eltype(volume)`: an integer volume is filtered as `float(eltype)`, and
    # storing those coefficients back into an integer array would throw.
    Td = data_eltype(volume)
    subbands = [[similar(volume, Td) for _ in 1:ns_surfacelet_nsubbands(params, j)] for j in 1:params.J]
    coeffs = NSSurfaceletCoefficients(similar(volume, Td), subbands)
    return ns_surfacelet_forward!(coeffs, volume, params; workspace = workspace, threaded = threaded)
end

"""
    ns_surfacelet_forward!(coeffs, volume, params::NSSurfaceletParams; workspace=nothing, threaded = nothing) -> coeffs

In-place [`ns_surfacelet_forward`](@ref): writes into the preallocated `coeffs`
(a result of an earlier call), allocating nothing but the threads' tasks when a
`workspace` is given.
"""
function ns_surfacelet_forward!(
        coeffs::NSSurfaceletCoefficients, volume::AbstractArray{<:Number, 3}, params::NSSurfaceletParams;
        workspace::Union{NSSurfaceletWorkspace, Nothing} = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    Td = data_eltype(volume)
    ws = _ns_surfacelet_workspace(workspace, params, size(volume), Td)
    length(coeffs.subbands) == params.J ||
        throw(ArgumentError("coefficients were produced for a different number of scales"))
    for j in 1:params.J
        length(coeffs.subbands[j]) == ns_surfacelet_nsubbands(params, j) ||
            throw(DimensionMismatch("scale $(j) has the wrong number of subbands"))
        all(b -> size(b) == ws.dims, coeffs.subbands[j]) ||
            throw(DimensionMismatch("scale $(j) has a subband of the wrong size"))
    end
    size(coeffs.coarse) == ws.dims || throw(DimensionMismatch("coarse subband has the wrong size"))
    if _ns_engine_arrays(ws, coeffs)
        mask_bank_forward!(coeffs, _ns_engine_input(ws, volume), ws.engine, _ns_threaded(threaded))
    else
        copyto!(coeffs.coarse, _ns_forward!((j, k, b) -> copyto!(coeffs.subbands[j][k], b), volume, ws))
    end
    return coeffs
end

"""
    ns_surfacelet_forward(f, volume, params::NSSurfaceletParams; workspace=nothing)

Streaming nonsubsampled surfacelet transform: calls `f(j, k, subband)` for directional subband `k` of
scale `j`, reusing one buffer between calls.  `subband` is only valid until `f`
returns — copy or reduce it there.  The calls come one at a time, from the
calling task, so `f` need not be thread-safe.

This is the form to use at working sizes.  At 256³ `Float32` with `J = 3` and
`L = [2, 2, 3]` the full coefficient set is 3.1 GB, while streaming holds a few
volumes plus the plans.

Returns the coarse residual (a workspace buffer, valid until its next use).

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(6), 16, 16, 16);

julia> p = NSSurfaceletParams(J = 2, L_array = [1, 1], order = 6);

julia> total = 0.0;

julia> coarse = ns_surfacelet_forward(x, p) do j, k, band
           global total += sum(abs2, band)
       end;

julia> total > 0 && size(coarse) == size(x)
true
```
"""
function ns_surfacelet_forward(
        f, volume::AbstractArray{<:Number, 3}, params::NSSurfaceletParams;
        workspace::Union{NSSurfaceletWorkspace, Nothing} = nothing
    )
    Td = data_eltype(volume)
    ws = _ns_surfacelet_workspace(workspace, params, size(volume), Td)
    return _ns_forward!(f, volume, ws)
end

"""
    ns_surfacelet_inverse(coeffs::NSSurfaceletCoefficients, params::NSSurfaceletParams; workspace=nothing, threaded = nothing) -> volume

Invert [`ns_surfacelet_forward`](@ref).
"""
function ns_surfacelet_inverse(
        coeffs::NSSurfaceletCoefficients, params::NSSurfaceletParams;
        workspace::Union{NSSurfaceletWorkspace, Nothing} = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    out = similar(coeffs.coarse)
    return ns_surfacelet_inverse!(out, coeffs, params; workspace = workspace, threaded = threaded)
end

"""
    ns_surfacelet_inverse!(volume, coeffs, params::NSSurfaceletParams; workspace=nothing, threaded = nothing) -> volume

In-place [`ns_surfacelet_inverse`](@ref).
"""
function ns_surfacelet_inverse!(
        volume::AbstractArray{<:Number, 3}, coeffs::NSSurfaceletCoefficients,
        params::NSSurfaceletParams;
        workspace::Union{NSSurfaceletWorkspace, Nothing} = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    ws = _ns_inverse_workspace(workspace, params, volume, coeffs.coarse)
    length(coeffs.subbands) == params.J ||
        throw(ArgumentError("coefficients were produced for a different number of scales"))
    for j in 1:params.J
        length(coeffs.subbands[j]) == ns_surfacelet_nsubbands(params, j) ||
            throw(DimensionMismatch("scale $(j) has the wrong number of subbands"))
        all(b -> size(b) == ws.dims, coeffs.subbands[j]) ||
            throw(DimensionMismatch("scale $(j) has a subband of the wrong size"))
    end
    _ns_engine_arrays(ws, coeffs) || return _ns_inverse!((j, k) -> coeffs.subbands[j][k], volume, coeffs.coarse, ws)
    # A real output the inverse plan cannot write is filled through `ws.rbuf[1]`.
    y = _nchannels(ws) == 2 || volume isa eltype(ws.rbuf) ? volume : ws.rbuf[1]
    mask_bank_inverse!(y, coeffs, ws.engine, _ns_threaded(threaded))
    y === volume || copyto!(volume, y)
    return volume
end

"""
    ns_surfacelet_inverse(f, coarse, params::NSSurfaceletParams; workspace=nothing) -> volume

Streaming inverse: `f(j, k)` must return directional subband `k` of scale `j`.
The subbands are consumed one at a time, from the calling task, so a caller
holding them on disk or regenerating them never materialises the full set.
"""
function ns_surfacelet_inverse(
        f, coarse::AbstractArray{<:Number, 3}, params::NSSurfaceletParams;
        workspace::Union{NSSurfaceletWorkspace, Nothing} = nothing
    )
    out = similar(coarse)
    return ns_surfacelet_inverse!(f, out, coarse, params; workspace = workspace)
end

"""
    ns_surfacelet_inverse!(f, volume, coarse, params::NSSurfaceletParams; workspace=nothing) -> volume

In-place streaming inverse; see [`ns_surfacelet_inverse(f, coarse, params)`](@ref).
"""
function ns_surfacelet_inverse!(
        f, volume::AbstractArray{<:Number, 3}, coarse::AbstractArray{<:Number, 3},
        params::NSSurfaceletParams; workspace::Union{NSSurfaceletWorkspace, Nothing} = nothing
    )
    ws = _ns_inverse_workspace(workspace, params, volume, coarse)
    return _ns_inverse!(f, volume, coarse, ws)
end

function _ns_inverse_workspace(workspace, params::NSSurfaceletParams, volume::AbstractArray, coarse::AbstractArray)
    size(coarse) == size(volume) || throw(DimensionMismatch("coarse size does not match the output"))
    return _ns_surfacelet_workspace(workspace, params, size(volume), data_eltype(volume))
end
