# 3-D separable convolution with configurable boundary extension.
#
# Structure mirrors the 2-D `conv2d_sep!` path: a fully generic kernel driven by
# `_get3d` (any boundary, any element type) plus a fast path for real CPU arrays
# with :symmetric / :periodic extension that splits the volume into an interior
# slab (no boundary handling needed) and two border slabs.
#
# The filtering axis is a `Val{D}` type parameter so the offset arithmetic folds
# away at compile time and one kernel serves all three axes.
#
# Only the separable API is provided — the 3-D transforms never convolve with a
# dense 3-D kernel.  The periodic-boundary FFT backend lives in `conv3d_fft.jl`
# and is selected automatically by `conv3d_sep!`; see `_conv3d_use_fft`.

@inline function _get3d(
        src::AbstractArray{T, 3}, i::Int, j::Int, k::Int,
        n1::Int, n2::Int, n3::Int, bv::Val{B}
    )::T where {T, B}
    ii = clamp_idx(i, n1, bv)
    jj = clamp_idx(j, n2, bv)
    kk = clamp_idx(k, n3, bv)
    if B === :zero && (ii < 1 || ii > n1 || jj < 1 || jj > n2 || kk < 1 || kk > n3)
        return zero(T)
    end
    return @inbounds src[ii, jj, kk]
end

# Offsets applied to (i, j, k) when filtering along axis D by `off` samples.
@inline _axis_offsets(::Val{1}, off::Int) = (off, 0, 0)
@inline _axis_offsets(::Val{2}, off::Int) = (0, off, 0)
@inline _axis_offsets(::Val{3}, off::Int) = (0, 0, off)

# Generic per-voxel kernel over an arbitrary index box — handles every boundary
# mode and every element type, at the cost of an index fold per tap.
@inline function _filter_dim_generic!(
        dst::AbstractArray{Td, 3}, src::AbstractArray{Td, 3},
        h::AbstractVector{Tf}, dv::Val{D}, bv::Val{B},
        r1::UnitRange{Int}, r2::UnitRange{Int}, r3::UnitRange{Int}
    ) where {Td, Tf, D, B}
    n1, n2, n3 = size(src)
    l = length(h)
    c = (l + 1) ÷ 2
    @inbounds for k in r3, j in r2, i in r1
        acc = zero(Td)
        for t in 1:l
            o1, o2, o3 = _axis_offsets(dv, c - t)
            acc += h[t] * _get3d(src, i + o1, j + o2, k + o3, n1, n2, n3, bv)
        end
        dst[i, j, k] = acc
    end
    return dst
end

# Interior kernel — no boundary extension, restructured as "zero then accumulate
# over taps" so the innermost loop is a stride-1 axpy.  Same shape as the 2-D
# fast path (and same reason: avoids the LoopVectorization gensym precompile bug
# triggered by 3-level `@turbo` reductions).
@inline function _filter_dim_interior!(
        dst::AbstractArray{Td, 3}, src::AbstractArray{Td, 3},
        h::AbstractVector{Tf}, dv::Val{D},
        r1::UnitRange{Int}, r2::UnitRange{Int}, r3::UnitRange{Int}
    ) where {Td, Tf, D}
    l = length(h)
    c = (l + 1) ÷ 2
    @inbounds for k in r3, j in r2
        @simd for i in r1
            dst[i, j, k] = zero(Td)
        end
    end
    @inbounds for t in 1:l
        w = h[t]
        o1, o2, o3 = _axis_offsets(dv, c - t)
        for k in r3, j in r2
            @simd for i in r1
                dst[i, j, k] += w * src[i + o1, j + o2, k + o3]
            end
        end
    end
    return dst
end

"""
    _filter_dim!(dst, src, h, Val(D), Val(B)) -> dst

Convolve `src` with the 1-D filter `h` along axis `D` (1, 2 or 3), writing into
`dst`.  `h` is treated as centred at index `(length(h) + 1) ÷ 2`.  `dst` and
`src` must not alias.
"""
function _filter_dim!(
        dst::AbstractArray{Td, 3}, src::AbstractArray{Td, 3},
        h::AbstractVector{Tf}, dv::Val{D}, bv::Val{B}
    ) where {Td, Tf, D, B}
    n1, n2, n3 = size(src)
    return _filter_dim_generic!(dst, src, h, dv, bv, 1:n1, 1:n2, 1:n3)
end

# Fast path: boundary modes whose interior needs no index folding.  Complex data
# takes it too — the interior kernel is a real-weighted axpy either way, and the
# element type only widens the accumulator.  Keeping the two in step is not just
# a performance question: `_conv3d_use_fft` picks the backend from the tap count
# alone, so a complex volume and its real and imaginary parts must reach the same
# kernel for `T(x + i·y) == T(x) + i·T(y)` to hold bit-for-bit.
function _filter_dim!(
        dst::AbstractArray{Td, 3}, src::AbstractArray{Td, 3},
        h::AbstractVector{Tf}, dv::Val{D},
        bv::Union{Val{:symmetric}, Val{:periodic}}
    ) where {Td, Tf, D}
    n1, n2, n3 = size(src)
    n = (n1, n2, n3)[D]
    l = length(h)
    c = (l + 1) ÷ 2

    # The tap loop reads `i + c - t` for `t ∈ 1:l`, i.e. `i - (l - c)` up to
    # `i + (c - 1)`.  Both bounds must stay inside `1:n`, which for an
    # even-length filter is *not* `c` (there `l - c = c`, one tap further back).
    lo = l - c + 1         # first index along D needing no extension
    hi = n - l + c         # last such index
    if hi < lo
        # Filter longer than the volume along this axis — no interior at all.
        return _filter_dim_generic!(dst, src, h, dv, bv, 1:n1, 1:n2, 1:n3)
    end

    full = (1:n1, 1:n2, 1:n3)
    lead = Base.setindex(full, 1:(lo - 1), D)
    trail = Base.setindex(full, (hi + 1):n, D)
    inner = Base.setindex(full, lo:hi, D)

    _filter_dim_generic!(dst, src, h, dv, bv, lead...)
    _filter_dim_generic!(dst, src, h, dv, bv, trail...)
    _filter_dim_interior!(dst, src, h, dv, inner...)
    return dst
end

"""
    conv3d_sep!(dst, src, h1, h2, h3; boundary=:symmetric, tmp=nothing) -> dst

Separable 3-D convolution: apply the 1-D filter `h1` along axis 1, then `h2`
along axis 2, then `h3` along axis 3.  Each filter is treated as centred at
index `(length(h) + 1) ÷ 2`.

`dst` must not alias `src`.  Pass a preallocated `tmp` (same size and element
type as `src`, distinct from both `dst` and `src`) to make the call
allocation-free.  Supported `boundary` values are `:symmetric` (default),
`:periodic` and `:zero`.

This is the workhorse for the 3-D nonsubsampled pyramid.

# Examples
```jldoctest
julia> using Surfacelets

julia> x = ones(8, 8, 8);

julia> h = [0.25, 0.5, 0.25];

julia> y = conv3d_sep(x, h, h, h);

julia> maximum(abs, y .- 1) < 1e-12
true
```
"""
function conv3d_sep!(
        dst::AbstractArray{Td, 3},
        src::AbstractArray{Td, 3},
        h1::AbstractVector{Tf},
        h2::AbstractVector{Tf},
        h3::AbstractVector{Tf};
        boundary::Symbol = :symmetric,
        tmp::Union{AbstractArray{Td, 3}, Nothing} = nothing,
        plan::Union{Conv3DFFTPlan, Nothing} = nothing,
        backend::Symbol = :auto
    ) where {Td, Tf}
    size(dst) == size(src) || throw(DimensionMismatch("dst and src must have the same size"))
    tmp === nothing || size(tmp) == size(src) ||
        throw(DimensionMismatch("tmp and src must have the same size"))
    # Reject typos rather than silently falling through to :zero / :auto — the
    # wrong boundary is a wrong answer, not a wrong performance.
    boundary in (:symmetric, :periodic, :zero) ||
        throw(ArgumentError("unsupported boundary $(boundary); use :symmetric, :periodic or :zero"))
    backend in (:auto, :direct, :fft) ||
        throw(ArgumentError("unsupported backend $(backend); use :auto, :direct or :fft"))
    backend === :fft && boundary !== :periodic &&
        throw(ArgumentError("the FFT backend is exact only for boundary = :periodic"))
    # Periodic separable filtering is exactly circular convolution, so the FFT
    # backend is not an approximation — only a different cost curve.
    if backend !== :direct && boundary === :periodic &&
            Td <: Union{AbstractFloat, Complex{<:AbstractFloat}}
        taps = max(length(h1), length(h2), length(h3))
        if plan !== nothing
            return conv3d_sep_fft!(dst, src, h1, h2, h3, plan)
        elseif backend === :fft || _conv3d_use_fft(Td, boundary, size(src), taps)
            p = make_conv3d_fft_plan(size(src), Td)
            return conv3d_sep_fft!(dst, src, h1, h2, h3, p)
        end
    end
    # Only the direct path needs the intermediate buffer, and it is a full volume
    # (64 MB at 256³ Float32) — so it is allocated here rather than as a default
    # keyword argument, which would allocate it for the FFT path too.
    scratch = tmp === nothing ? similar(src) : tmp
    if boundary === :symmetric
        _conv3d_sep_impl!(dst, src, h1, h2, h3, scratch, Val(:symmetric))
    elseif boundary === :periodic
        _conv3d_sep_impl!(dst, src, h1, h2, h3, scratch, Val(:periodic))
    else
        _conv3d_sep_impl!(dst, src, h1, h2, h3, scratch, Val(:zero))
    end
    return dst
end

# Three passes, two buffers: src → dst (axis 1) → tmp (axis 2) → dst (axis 3).
@inline function _conv3d_sep_impl!(
        dst::AbstractArray{Td, 3}, src::AbstractArray{Td, 3},
        h1::AbstractVector{Tf}, h2::AbstractVector{Tf}, h3::AbstractVector{Tf},
        tmp::AbstractArray{Td, 3}, bv::Val{B}
    ) where {Td, Tf, B}
    _filter_dim!(dst, src, h1, Val(1), bv)
    _filter_dim!(tmp, dst, h2, Val(2), bv)
    _filter_dim!(dst, tmp, h3, Val(3), bv)
    return dst
end

"""
    conv3d_sep(src, h1, h2, h3; boundary=:symmetric) -> Array{,3}

Allocating version of [`conv3d_sep!`](@ref).
"""
function conv3d_sep(
        src::AbstractArray{Td, 3},
        h1::AbstractVector{Tf},
        h2::AbstractVector{Tf},
        h3::AbstractVector{Tf};
        boundary::Symbol = :symmetric,
        plan::Union{Conv3DFFTPlan, Nothing} = nothing,
        backend::Symbol = :auto
    ) where {Td, Tf}
    dst = similar(src)
    return conv3d_sep!(dst, src, h1, h2, h3; boundary, plan, backend)
end
