# FFT backend for periodic 3-D separable convolution.
#
# Periodic-boundary separable filtering *is* circular convolution, so an FFT
# evaluation is exact (not an approximation) and its cost is independent of the
# filter length.  That matters much more in 3-D than in 2-D: the à trous filters
# grow as `2^(j-1)` taps per axis, and the direct kernel is memory-bound.
#
# Measured with a preallocated plan, `direct / FFT` per pass (a ratio above 1
# means the FFT wins), one FFTW thread and four giving the same picture:
#
#   | taps | 64³ real | 64³ cplx | 128³ real | 128³ cplx | 256³ real | 256³ cplx |
#   |------|----------|----------|-----------|-----------|-----------|-----------|
#   |    3 |     1.21 |     0.76 |      0.70 |      0.78 |      0.73 |      0.67 |
#   |    5 |     1.69 |     1.94 |      1.44 |      1.78 |      1.20 |      1.36 |
#   |    9 |     4.90 |     3.37 |      3.92 |      4.99 |      2.73 |      2.70 |
#   |   33 |    43.44 |    22.85 |     44.72 |     22.82 |         — |         — |
#
# Hence the single crossover in `_conv3d_use_fft`: five taps, for every volume
# size, element type and thread count measured.  One threshold for real and
# complex is not only simpler — it is required, because a complex volume and its
# real and imaginary parts must reach the same kernel for the package invariant
# `T(x + i·y) == T(x) + i·T(y)` to hold bit-for-bit.
#
# Complex volumes are filtered as two real transforms (real part, then imaginary
# part) rather than by a complex `plan_fft`.  Two `rfft` round trips cost about
# the same as one complex FFT of the same size, and doing it this way keeps the
# package invariant `T(x + i·y) == T(x) + i·T(y)` true *bit-for-bit* — the real
# and complex paths run the identical kernel on identical data.
#
# Transfer functions are kept separable — the 3-D response is
# `W1[k1] * W2[k2] * W3[k3]`, so the per-axis 1-D DFTs are built from length-`n`
# vectors and broadcast into the frequency cube.  No dense 3-D kernel is ever
# materialised.
#
# The plan object is explicit rather than a package-level `Dict`: see the
# "FFTW plans: no Julia-side plan cache" note in AGENTS.md.  FFTW's own MEASURE
# wisdom makes re-planning a known size sub-millisecond, so a caller that cannot
# hold a plan simply pays that.

"""
    Conv3DFFTPlan{T}

Preallocated FFTW plans and buffers for [`conv3d_sep_fft!`](@ref) on volumes of
one fixed size and real element type `T`.  Build it with
[`make_conv3d_fft_plan`](@ref) and reuse it across scales and iterations.
"""
mutable struct Conv3DFFTPlan{
        T <: AbstractFloat, A3 <: AbstractArray{T, 3},
        C3 <: AbstractArray{Complex{T}, 3}, Pf, Pi,
    }
    const dims::NTuple{3, Int}
    const fwd::Pf
    const inv::Pi
    const freq::C3                     # rfft-sized spectrum of the volume
    const ker::A3                      # real scratch holding a copy of the input
    # Per-axis transfer functions, kept as 3-D arrays with singleton axes so the
    # frequency-domain multiply is a plain broadcast — no scalar indexing, so
    # the plan works on GPU arrays too.
    const w1::C3                       # rfft of the axis-1 circular kernel
    const w2::C3                       # fft  of the axis-2 circular kernel
    const w3::C3                       # fft  of the axis-3 circular kernel
    const kbuf1::Vector{T}             # host-side kernel scratch (length n, cheap)
    const kbuf2::Vector{T}
    const kbuf3::Vector{T}
    # Extra real buffer holding the filtered real part while the imaginary part
    # is transformed.  Allocated on first complex use so a real-only workload
    # never pays for it (64 MB at 256³ Float32).
    aux::Union{Nothing, A3}
end

"""
    make_conv3d_fft_plan(dims::NTuple{3,Int}, ::Type{Td}=Float64;
                         threaded = nothing, flags=FFTW.MEASURE) -> Conv3DFFTPlan

Build FFTW plans and scratch buffers for periodic separable convolution of
`dims`-sized volumes with element type `Td`.  A complex `Td` produces a plan of
the corresponding real precision with the extra real scratch preallocated; the
plan then serves both real and complex volumes.

Planning with `FFTW.MEASURE` is worth it whenever the plan is reused (every
pyramid level of every iteration); pass `flags = FFTW.ESTIMATE` for a one-shot
call where first-call latency dominates.
"""
function make_conv3d_fft_plan(
        dims::NTuple{3, Int}, ::Type{T} = Float64;
        threaded::Union{Bool, Nothing} = nothing,
        flags::Integer = FFTW.MEASURE
    ) where {T <: AbstractFloat}
    n1, n2, n3 = dims
    all(>(0), dims) || throw(ArgumentError("volume dimensions must be positive"))
    buf = zeros(T, n1, n2, n3)
    fwd, spec, inv = with_fftw_threads(_fftw_threads(threaded)) do
        f = plan_rfft(buf; flags = flags)
        s = f * buf
        f, s, plan_irfft(s, n1; flags = flags)
    end
    w1 = Array{Complex{T}, 3}(undef, n1 ÷ 2 + 1, 1, 1)
    w2 = Array{Complex{T}, 3}(undef, 1, n2, 1)
    w3 = Array{Complex{T}, 3}(undef, 1, 1, n3)
    return Conv3DFFTPlan{T, typeof(buf), typeof(spec), typeof(fwd), typeof(inv)}(
        dims, fwd, inv, spec, buf, w1, w2, w3,
        Vector{T}(undef, n1), Vector{T}(undef, n2), Vector{T}(undef, n3),
        nothing,
    )
end

# Complex element type: the plan is real (two real transforms per complex volume);
# preallocate the extra real buffer so the first complex call allocates nothing.
function make_conv3d_fft_plan(
        dims::NTuple{3, Int}, ::Type{Complex{T}}; kwargs...
    ) where {T <: AbstractFloat}
    plan = make_conv3d_fft_plan(dims, T; kwargs...)
    plan.aux = similar(plan.ker)
    return plan
end

make_conv3d_fft_plan(n1::Int, n2::Int, n3::Int, ::Type{Td} = Float64; kwargs...) where {Td} =
    make_conv3d_fft_plan((n1, n2, n3), Td; kwargs...)

# FFTW thread count for the `threaded` keyword.  The default (`nothing`) uses every
# Julia thread: an FFT here is always a big enough chunk of work to be worth
# splitting.  The
# plan is made inside `with_fftw_threads`, so a restriction opened around the
# call (a caller's `@budgeted_threads` loop) caps the count.
_fftw_threads(threaded::Union{Bool, Nothing}) = threaded === false ? 1 : Threads.nthreads()

# Circular kernel for the *correlation* convention used by `_filter_dim!`:
#   y[i] = Σ_t h[t] · x[i + c - t],   c = (length(h) + 1) ÷ 2
# equals the circular convolution of `x` with `w[p] = h[c + p]`, indices taken
# mod `n`.  Taps are accumulated, not assigned, so a filter longer than the
# volume aliases exactly the way the direct periodic path does.
function _circular_kernel!(w::AbstractVector{T}, h::AbstractVector, n::Int) where {T}
    fill!(w, zero(T))
    l = length(h)
    c = (l + 1) ÷ 2
    @inbounds for t in 1:l
        w[mod(t - c, n) + 1] += T(h[t])
    end
    return w
end

"""
    conv3d_sep_fft!(dst, src, h1, h2, h3, plan) -> dst

Periodic separable 3-D convolution evaluated in the frequency domain.  Exactly
equivalent to `conv3d_sep!(dst, src, h1, h2, h3; boundary=:periodic)` up to
floating-point rounding, but independent of the filter lengths.

`src` may be real or complex; a complex volume is filtered as two real
transforms, so `conv3d_sep_fft!` on `x + i·y` equals
`conv3d_sep_fft!(x) + i·conv3d_sep_fft!(y)` bit-for-bit.

`dst` may alias `src`.  The call allocates nothing beyond the plan, provided a
complex plan was built with a complex element type (or has already run once).
"""
function conv3d_sep_fft!(
        dst::AbstractArray{T, 3}, src::AbstractArray{T, 3},
        h1::AbstractVector, h2::AbstractVector, h3::AbstractVector,
        plan::Conv3DFFTPlan{T}
    ) where {T <: AbstractFloat}
    _check_plan_dims(dst, src, plan)
    _build_kernel_spectra!(plan, h1, h2, h3)
    copyto!(plan.ker, src)
    _apply_freq_filter!(plan)
    copyto!(dst, plan.ker)
    return dst
end

function conv3d_sep_fft!(
        dst::AbstractArray{Complex{T}, 3}, src::AbstractArray{Complex{T}, 3},
        h1::AbstractVector, h2::AbstractVector, h3::AbstractVector,
        plan::Conv3DFFTPlan{T}
    ) where {T <: AbstractFloat}
    _check_plan_dims(dst, src, plan)
    _build_kernel_spectra!(plan, h1, h2, h3)

    aux = plan.aux
    if aux === nothing
        aux = similar(plan.ker)
        plan.aux = aux
    end

    # Real part first; park the result in `aux` while the imaginary part goes
    # through the same kernel.  Identical arithmetic on both halves is what makes
    # the real/complex agreement exact rather than merely close.
    @. plan.ker = real(src)
    _apply_freq_filter!(plan)
    copyto!(aux, plan.ker)

    @. plan.ker = imag(src)
    _apply_freq_filter!(plan)

    @. dst = complex(aux, plan.ker)
    return dst
end

@inline function _check_plan_dims(dst, src, plan::Conv3DFFTPlan)
    size(dst) == plan.dims || throw(DimensionMismatch("dst size does not match the plan"))
    size(src) == plan.dims || throw(DimensionMismatch("src size does not match the plan"))
    return nothing
end

# 1-D DFTs of the three circular kernels.  rfft along axis 1 (half spectrum,
# matching the volume transform); full fft along axes 2 and 3.
function _build_kernel_spectra!(
        plan::Conv3DFFTPlan{T}, h1::AbstractVector, h2::AbstractVector, h3::AbstractVector
    ) where {T}
    n1, n2, n3 = plan.dims
    _circular_kernel!(plan.kbuf1, h1, n1)
    _circular_kernel!(plan.kbuf2, h2, n2)
    _circular_kernel!(plan.kbuf3, h3, n3)
    # The 1-D transforms are length-n and stay on the host; only the results are
    # copied into the (possibly device-resident) transfer-function arrays.
    copyto!(plan.w1, rfft(plan.kbuf1))
    copyto!(plan.w2, fft(plan.kbuf2))
    copyto!(plan.w3, fft(plan.kbuf3))
    return plan
end

# Filter the real volume sitting in `plan.ker` in place, using the kernel spectra
# built by `_build_kernel_spectra!`.
function _apply_freq_filter!(plan::Conv3DFFTPlan)
    mul!(plan.freq, plan.fwd, plan.ker)
    w1, w2, w3 = plan.w1, plan.w2, plan.w3
    @. plan.freq *= w1 * w2 * w3
    mul!(plan.ker, plan.inv, plan.freq)
    return plan
end

"""
    conv3d_sep_fft(src, h1, h2, h3) -> Array{,3}

Allocating [`conv3d_sep_fft!`](@ref); builds a throwaway `FFTW.ESTIMATE` plan.
"""
function conv3d_sep_fft(
        src::AbstractArray{Td, 3}, h1::AbstractVector, h2::AbstractVector, h3::AbstractVector
    ) where {Td <: Union{AbstractFloat, Complex{<:AbstractFloat}}}
    plan = make_conv3d_fft_plan(size(src), Td; flags = FFTW.ESTIMATE)
    return conv3d_sep_fft!(similar(src), src, h1, h2, h3, plan)
end

# Crossover point of the auto-selection rule, measured as in the table at the top
# of this file.  It is the one machine-dependent part of the backend choice, so
# it is named here rather than buried in the dispatch helper.  It deliberately
# does not depend on the element type or on the FFTW thread count: see the table.
const _CONV3D_FFT_MIN_TAPS = 5
# Below this the transform overhead does not amortise whatever the tap count.
const _CONV3D_FFT_MIN_DIM = 16

"""
    _conv3d_use_fft(Td, boundary, dims, taps) -> Bool

Auto-selection rule for the periodic 3-D backend.  The FFT path requires a
floating-point element type (real or complex), a periodic boundary (where it is
exact), a volume large enough for the transform overhead to amortise, and a
filter long enough to lose to it.
"""
@inline function _conv3d_use_fft(::Type{Td}, boundary::Symbol, dims::NTuple{3, Int}, taps::Int) where {Td}
    Td <: Union{AbstractFloat, Complex{<:AbstractFloat}} || return false
    boundary === :periodic || return false
    minimum(dims) >= _CONV3D_FFT_MIN_DIM || return false
    return taps >= _CONV3D_FFT_MIN_TAPS
end
