# Nonsubsampled surfacelet transform.
#
# Multiscale stage: the à trous nonsubsampled pyramid (`nsp3d_decompose`).
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
# hourglass response: Bⱼ is a broadcast of per-axis vectors
# (`MultiscaleFilterBanks.nsp_responses`) and the wedge factors are planes.  The
# forward forms `Yᵢ = X·Bⱼ·Wᵢ` once per hourglass, so a subband costs one
# multiply by two planes and one inverse FFT; the inverse mirrors it with one
# forward FFT per subband, a per-hourglass sum, and one inverse FFT in total.
# This equals `nsp3d_decompose` followed by `ndfb_decompose` to rounding.
#
# The subbands are independent, so with threading the wedges of each hourglass
# are split into contiguous chunks, one task per FFT buffer
# (`MultiscaleFilterBanks.foreach_chunk`); the FFT plans are single-threaded.
# The full-volume passes between those loops (the band and lowpass products and
# the sum of the inverse's per-task partial sums) are split among the same tasks,
# by slabs of the third axis or of the linear index: left serial, at 8 threads
# they took a third of the 96³ inverse.
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
same time; it threads internally instead.  `Adapt.adapt(ArrayType, ws)` moves it
to a device, rebuilding its buffers and FFT plans there.
"""
struct NSSurfaceletWorkspace{
        Td <: Number, T <: AbstractFloat, A3 <: AbstractArray{T, 3}, C3 <: AbstractArray{Complex{T}, 3},
        D <: AbstractArray{Td, 3}, P, Q,
    }
    dims::NTuple{3, Int}
    params::NSSurfaceletParams{T}
    # Tables (moved by `Adapt`).
    m1::A3                                      # hourglass f₀(M₁), rfft grid
    m2::A3                                      # hourglass f₀(M₂), (1, n₂, n₃)
    wedges::Vector{NTuple{3, NTuple{2, Vector{A3}}}}  # [scale][hourglass][plane][wedge], √ when tight
    pa::Vector{NTuple{3, C3}}                   # Π_{i<j} Hᵢ per axis, j = 1:J+1
    pgh::Vector{NTuple{3, C3}}                  # Gⱼ Hⱼ per axis, j = 1:J
    ps::Vector{NTuple{3, C3}}                   # Π_{i<j} Gᵢ per axis, j = 1:J+1
    # Buffers.  A complex volume is two real channels, filtered alike, which
    # keeps `T(x + i·y) == T(x) + i·T(y)` exact.
    X::Vector{C3}                               # [channel] input spectrum / synthesis sum
    Y::Vector{C3}                               # [channel] hourglass product / its synthesis
    work::Vector{C3}                            # [task]
    rbuf::Vector{Vector{A3}}                    # [task][channel]
    partial::Vector{Vector{C3}}                 # [channel][task - 1], synthesis partial sums
    cbuf::Vector{D}                             # [task] a complex subband handed to a callback
    coarse::D
    pfwd::P                                     # rfft, single-threaded
    pinv::Q                                     # irfft (destroys its input)
end

Base.eltype(::NSSurfaceletWorkspace{Td}) where {Td} = Td

_nchannels(::NSSurfaceletWorkspace{<:Real}) = 1
_nchannels(::NSSurfaceletWorkspace{<:Complex}) = 2

"""
    make_ns_surfacelet_workspace(params::NSSurfaceletParams, dims::NTuple{3,Int},
                         ::Type{Td}=Float64; tasks=Threads.nthreads(), flags=FFTW.MEASURE)

Allocate the workspace for `params` at volume size `dims` and element type `Td`
(real or complex).

Memory: the hourglass response is one `real(Td)` array on the rfft grid,
`(n₁÷2+1) × n₂ × n₃`; the buffers are about `4 + 2·tasks` such arrays for real
data (twice for complex).  `tasks` bounds the threads the transform uses, and so
the buffers; everything per-scale is vectors and planes.
"""
function make_ns_surfacelet_workspace(
        params::NSSurfaceletParams, dims::NTuple{3, Int}, ::Type{Td} = Float64;
        tasks::Integer = Threads.nthreads(), flags::Integer = FFTW.MEASURE
    ) where {Td <: Number}
    T = filter_eltype(Td)
    p = _convert_params(T, params)
    m1, m2 = _hourglass_tables(T, dims, p.order, p.poly)
    wedges = [_ns_wedge_factors(dims, l, p) for l in p.L_array]
    resp = ntuple(3) do d
        nsp_responses(p.lp_filters, p.J, dims[d]; bins = d == 1 ? (1:(dims[1] ÷ 2 + 1)) : (1:dims[d]))
    end
    orient(v, d) = reshape(Complex{T}.(v), ntuple(k -> k == d ? length(v) : 1, 3))
    per_level(r, J) = [ntuple(d -> orient(resp[d][r][j], d), 3) for j in 1:J]
    pa, pgh, ps = per_level(1, p.J + 1), per_level(2, p.J), per_level(3, p.J + 1)
    proto = Array{T, 3}(undef, 0, 0, 0)
    return _ns_workspace(Td, dims, p, m1, m2, wedges, pa, pgh, ps, proto, tasks, flags)
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
# Host plans use `flags` on one FFTW thread; a device gets one task.
function _ns_workspace(
        ::Type{Td}, dims::NTuple{3, Int}, p::NSSurfaceletParams{T}, m1, m2, wedges, pa, pgh, ps,
        proto::AbstractArray, tasks::Integer, flags
    ) where {Td, T}
    tasks >= 1 || throw(ArgumentError("tasks must be ≥ 1"))
    nt = proto isa Array ? Int(tasks) : 1
    nch = Td <: Complex ? 2 : 1
    half = (dims[1] ÷ 2 + 1, dims[2], dims[3])
    spectrum() = fill!(similar(proto, Complex{T}, half), zero(Complex{T}))
    volume() = fill!(similar(proto, T, dims), zero(T))
    X = [spectrum() for _ in 1:nch]
    Y = [spectrum() for _ in 1:nch]
    work = [spectrum() for _ in 1:nt]
    rbuf = [[volume() for _ in 1:nch] for _ in 1:nt]
    partial = [[spectrum() for _ in 2:nt] for _ in 1:nch]
    coarse = similar(proto, Td, dims)
    cbuf = Td <: Complex ? [similar(proto, Td, dims) for _ in 1:nt] : typeof(coarse)[]
    pfwd, pinv = _fftw_single_threaded(proto) do
        (_fft_plan(plan_rfft, rbuf[1][1], flags), _fft_plan(plan_irfft, work[1], flags, dims[1]))
    end
    return NSSurfaceletWorkspace{Td, T, typeof(m1), eltype(X), typeof(coarse), typeof(pfwd), typeof(pinv)}(
        dims, p, m1, m2, wedges, pa, pgh, ps, X, Y, work, rbuf, partial, cbuf, coarse, pfwd, pinv
    )
end

# Equality of two filter pairs at the workspace's precision — the workspace holds
# `params`' filters already converted, so the comparison converts rather than
# promoting (and stays allocation-free).
function _same_filters(a::FilterPair{T}, b::FilterPair) where {T}
    length(a.h) == length(b.h) && length(a.g) == length(b.g) || return false
    all(i -> a.h[i] == T(b.h[i]), eachindex(a.h)) || return false
    return all(i -> a.g[i] == T(b.g[i]), eachindex(a.g))
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
    _same_filters(q.lp_filters, params.lp_filters) ||
        throw(ArgumentError("workspace was built for a different pyramid filter pair"))
    return ws
end

# Threading the subband FFTs pays for real and complex data alike, so only
# `threaded = false` turns it off.
_ns_threaded(threaded::Union{Bool, Nothing}) = threaded !== false

# ── Masks ─────────────────────────────────────────────────────────────────────

# Hourglass factor of subband `i`: √W when tight, W otherwise.
@inline function _hg_factor(m1::T, m2::T, i::Int, tight::Bool) where {T}
    w = _stage1_weight(m1, i) * _stage2_weight(m2, i)
    return tight ? _tight_sqrt(w) : w
end

# Y = X · Bⱼ · Wᵢ: the hourglass-i part of the scale-j bandpass.
function _ns_band!(Y, X, ws::NSSurfaceletWorkspace, j::Int, i::Int, threaded::Bool)
    a1, a2, a3 = ws.pa[j]
    g1, g2, g3 = ws.pgh[j]
    m1, m2, tight = ws.m1, ws.m2, ws.params.tight
    @. Y = X * (a1 * a2 * a3) * (1 - g1 * g2 * g3) * _hg_factor(m1, m2, i, tight)
    return Y
end

# The host methods walk the columns, as `_wedge_mul!` does: everything but `a1`,
# `g1` and `m1` is a scalar per column.  These full-volume passes sit between the
# threaded subband loops, so with threading they split the third axis among the
# workspace's tasks too (the forward at 96³ and 4 threads: 85 → 70 ms).
function _ns_band!(Y::Array, X::Array, ws::NSSurfaceletWorkspace, j::Int, i::Int, threaded::Bool)
    a1, a2, a3 = ws.pa[j]
    g1, g2, g3 = ws.pgh[j]
    m1, m2, tight = ws.m1, ws.m2, ws.params.tight
    a1v, g1v = vec(a1), vec(g1)
    _foreach_slab(Y, ws, threaded) do ks
        @inbounds for k in ks, l in axes(Y, 2)
            a23 = a2[1, l, 1] * a3[1, 1, k]
            g23 = g2[1, l, 1] * g3[1, 1, k]
            μ2 = m2[1, l, k]
            y, x, μ1 = view(Y, :, l, k), view(X, :, l, k), view(m1, :, l, k)
            @. y = x * (a1v * a23) * (1 - g1v * g23) * _hg_factor(μ1, μ2, i, tight)
        end
    end
    return Y
end

# X += Y · Sⱼ · Wᵢ, the synthesis counterpart (Wᵢ = 1 when not tight: the
# directional inverse is then a bare sum).
function _ns_unband!(X, Y, ws::NSSurfaceletWorkspace, j::Int, i::Int, threaded::Bool)
    s1, s2, s3 = ws.ps[j]
    m1, m2, tight = ws.m1, ws.m2, ws.params.tight
    if tight
        @. X += Y * (s1 * s2 * s3) * _hg_factor(m1, m2, i, true)
    else
        @. X += Y * (s1 * s2 * s3)
    end
    return X
end

function _ns_unband!(X::Array, Y::Array, ws::NSSurfaceletWorkspace, j::Int, i::Int, threaded::Bool)
    s1, s2, s3 = ws.ps[j]
    m1, m2, tight = ws.m1, ws.m2, ws.params.tight
    s1v = vec(s1)
    _foreach_slab(X, ws, threaded) do ks
        @inbounds for k in ks, l in axes(X, 2)
            s23 = s2[1, l, 1] * s3[1, 1, k]
            μ2 = m2[1, l, k]
            x, y, μ1 = view(X, :, l, k), view(Y, :, l, k), view(m1, :, l, k)
            if tight
                @. x += y * (s1v * s23) * _hg_factor(μ1, μ2, i, true)
            else
                @. x += y * (s1v * s23)
            end
        end
    end
    return X
end

# Y = X · a₁a₂a₃ (or Y += when `add`), the lowpass of the last pyramid level with
# its per-axis factors `a`.  Same column walk as `_ns_band!`.
_ns_lowpass!(Y, X, a, add::Bool, ws::NSSurfaceletWorkspace, threaded::Bool) =
    add ? (Y .+= X .* a[1] .* a[2] .* a[3]) : (Y .= X .* a[1] .* a[2] .* a[3])

function _ns_lowpass!(Y::Array, X::Array, a, add::Bool, ws::NSSurfaceletWorkspace, threaded::Bool)
    a1, a2, a3 = a
    a1v = vec(a1)
    _foreach_slab(Y, ws, threaded) do ks
        @inbounds for k in ks, l in axes(Y, 2)
            a23 = a2[1, l, 1] * a3[1, 1, k]
            y, x = view(Y, :, l, k), view(X, :, l, k)
            add ? (@. y += x * (a1v * a23)) : (@. y = x * (a1v * a23))
        end
    end
    return Y
end

# Call `f(ks)` on slabs `ks` of the third axis of `A`, one per workspace task when
# threaded.
_foreach_slab(f::F, A::AbstractArray, ws::NSSurfaceletWorkspace, threaded::Bool) where {F} =
    foreach_chunk((ks, _) -> f(ks), size(A, 3), min(length(ws.work), size(A, 3)), threaded)

# dst = Y·wa·wb, or dst += Y·wa·wb when `add`, with `wa`, `wb` the oriented wedge
# planes (singleton on one axis each).  On the host the 3-D broadcast does not
# vectorise across the singleton axes — it runs at a tenth of a plain copy — so
# the host method walks the columns, each a contiguous 1-D broadcast in which a
# plane contributes a column or a scalar.
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
        add ? (d .+= y .* s) : (d .= y .* s)
    elseif length(a) == 1
        add ? (d .+= y .* b .* a[1]) : (d .= y .* b .* a[1])
    elseif length(b) == 1
        add ? (d .+= y .* a .* b[1]) : (d .= y .* a .* b[1])
    else
        add ? (d .+= y .* a .* b) : (d .= y .* a .* b)
    end
    return d
end

# Wedge counts of scale j, and the (hourglass, p, q) ↦ subband index map
# (`ndfb_subband_index`'s order).
_ns_wedge_counts(ws::NSSurfaceletWorkspace, j::Int) = map(length, ws.wedges[j][1])
_ns_index(i::Int, p::Int, q::Int, na::Int, nb::Int) = ((i - 1) * na + (p - 1)) * nb + q

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

# Forward cascade.  For each subband, channel c is written to `target(j, k, t, c)`
# and then `done(j, k, t)` is called, by task `t`.  Returns `ws.coarse`.
function _ns_forward!(target::F, done::G, volume::AbstractArray, ws::NSSurfaceletWorkspace, threaded::Bool) where {F, G}
    threaded = loop_threaded(threaded)     # serial under a caller's NestedThreading budget
    nch = _nchannels(ws)
    for c in 1:nch
        _ns_spectrum!(ws.X[c], volume, ws.rbuf[1][c], ws, c)
    end
    nt = length(ws.work)
    for j in eachindex(ws.wedges)
        na, nb = _ns_wedge_counts(ws, j)
        for i in 1:3
            for c in 1:nch
                _ns_band!(ws.Y[c], ws.X[c], ws, j, i, threaded)
            end
            planes = ws.wedges[j][i]
            foreach_chunk(na * nb, nt, threaded) do range, t
                for r in range
                    p, q = divrem(r - 1, nb) .+ 1
                    k = _ns_index(i, p, q, na, nb)
                    for c in 1:nch
                        _wedge_mul!(ws.work[t], ws.Y[c], planes[1][p], planes[2][q], false)
                        mul!(target(j, k, t, c), ws.pinv, ws.work[t])
                    end
                    done(j, k, t)
                end
            end
        end
    end
    for c in 1:nch
        _ns_lowpass!(ws.work[1], ws.X[c], ws.pa[end], false, ws, threaded)
        mul!(ws.rbuf[1][c], ws.pinv, ws.work[1])
    end
    return _ns_combine!(ws.coarse, ws.rbuf[1])
end

# Inverse cascade; `source(j, k)` returns subband k of scale j.
function _ns_inverse!(source::F, volume::AbstractArray, coarse::AbstractArray, ws::NSSurfaceletWorkspace, threaded::Bool) where {F}
    threaded = loop_threaded(threaded)     # one answer for every loop of the call
    nch = _nchannels(ws)
    tight = ws.params.tight
    nt = length(ws.work)
    for c in 1:nch
        fill!(ws.X[c], zero(eltype(ws.X[c])))
    end
    for j in eachindex(ws.wedges)
        na, nb = _ns_wedge_counts(ws, j)
        for i in 1:3
            planes = ws.wedges[j][i]
            foreach_chunk(na * nb, nt, threaded) do range, t
                acc(c) = t == 1 ? ws.Y[c] : ws.partial[c][t - 1]
                for c in 1:nch
                    fill!(acc(c), zero(eltype(acc(c))))
                end
                W = ws.work[t]
                for r in range
                    p, q = divrem(r - 1, nb) .+ 1
                    band = source(j, _ns_index(i, p, q, na, nb))
                    # The threaded callers check every size up front; this is for
                    # the streaming inverse, whose subbands arrive one at a time.
                    size(band) == ws.dims ||
                        throw(DimensionMismatch("subband ($(j), $(_ns_index(i, p, q, na, nb))) has the wrong size"))
                    for c in 1:nch
                        _ns_spectrum!(W, band, ws.rbuf[t][c], ws, c)
                        if tight
                            _wedge_mul!(acc(c), W, planes[1][p], planes[2][q], true)
                        else
                            A = acc(c)
                            @.. A += W
                        end
                    end
                end
            end
            for c in 1:nch
                threaded && add_partials!(ws.Y[c], ws.partial[c], true)
                _ns_unband!(ws.X[c], ws.Y[c], ws, j, i, threaded)
            end
        end
    end
    for c in 1:nch
        _ns_spectrum!(ws.work[1], coarse, ws.rbuf[1][c], ws, c)
        _ns_lowpass!(ws.X[c], ws.work[1], ws.ps[end], true, ws, threaded)
        mul!(ws.rbuf[1][c], ws.pinv, ws.X[c])
    end
    return _ns_combine!(volume, ws.rbuf[1])
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
    # Real data is transformed straight into a subband that can take the inverse
    # FFT's output; complex data goes through the two channel buffers.
    real_out = _nchannels(ws) == 1
    target(j, k, t, c) = (b = coeffs.subbands[j][k]; real_out && b isa eltype(eltype(ws.rbuf)) ? b : ws.rbuf[t][c])
    function done(j, k, t)
        b = coeffs.subbands[j][k]
        (real_out && b isa eltype(eltype(ws.rbuf))) || _ns_combine!(b, ws.rbuf[t])
        return nothing
    end
    copyto!(coeffs.coarse, _ns_forward!(target, done, volume, ws, _ns_threaded(threaded)))
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
    target(j, k, t, c) = ws.rbuf[t][c]
    function done(j, k, t)
        band = _nchannels(ws) == 1 ? ws.rbuf[t][1] : _ns_combine!(ws.cbuf[t], ws.rbuf[t])
        f(j, k, band)
        return nothing
    end
    return _ns_forward!(target, done, volume, ws, false)
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
    return _ns_inverse!((j, k) -> coeffs.subbands[j][k], volume, coeffs.coarse, ws, _ns_threaded(threaded))
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
    return _ns_inverse!(f, volume, coarse, ws, false)
end

function _ns_inverse_workspace(workspace, params::NSSurfaceletParams, volume::AbstractArray, coarse::AbstractArray)
    size(coarse) == size(volume) || throw(DimensionMismatch("coarse size does not match the output"))
    return _ns_surfacelet_workspace(workspace, params, size(volume), data_eltype(volume))
end
