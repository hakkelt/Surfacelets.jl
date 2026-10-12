# 3-D nonsubsampled hourglass filter bank.
#
# Reference: Y. M. Lu and M. N. Do, "A Mapping-Based Design for Nonsubsampled
# Hourglass Filter Banks in Arbitrary Dimensions", IEEE TSP 56(4), 2008.
#
# The bank splits the spectrum into three hourglass-shaped subbands, each aligned
# with one frequency axis:
#
#     P_i = {ω : |ω_i| ≥ |ω_j| for all j ≠ i}
#
# i.e. subband `i` keeps the frequencies closer to axis `i` than to any other.
# This is the directional stage that replaces the 2-D quincunx/shear filter bank
# — neither the quincunx lattice nor the Q2345 ladder carries over to 3-D, so
# none of `directional/quincunx_fb.jl` is reused here.
#
# Structure (Lu & Do, Fig. 3a): a cascade of two two-channel mapped filter banks.
#
#     spectrum ──┬─ P₁                                    (3-D mapping kernel)
#                └─ P₂ ∪ P₃ ──┬─ P₂                       (2-D kernel on ω₂,ω₃)
#                             └─ P₃
#
# The second kernel needs only the two remaining frequency variables — Lu & Do's
# Proposition 1 shows the optimal passband support for a node with index set `J`
# and child set `K` depends on `|J|` variables only.  In 3-D that makes the
# second stage a 2-D design, which is why `m2` below is a matrix.
#
# Implementation: frequency domain, on the DFT grid.  Lu & Do deliberately chose
# FIR spatial filters because their FFT-domain predecessor needed too much memory
# for streaming video.  That trade-off inverts here — this package's 3-D stages
# are nonsubsampled with a periodic boundary, so the FFT is already the default
# backend (see `conv3d_fft.jl`), and evaluating the mapping kernel directly on
# the grid is *exact*: no FIR truncation of a 37³ filter, no convolution.
#
# Perfect reconstruction is structural.  The 1-D polynomials satisfy
# `f₀(x) + f₁(x) = 1` identically, so the three *weights* sum to
# `f₀(M₁) + f₁(M₁)·(f₀(M₂) + f₁(M₂)) = f₀(M₁) + f₁(M₁) = 1` at every frequency.
#
# Two ways to turn those weights into filters, selected by `tight`:
#
#   * `tight = false` — the masks are the weights `W_i` themselves.  Synthesis
#     filters are all 1 and reconstruction is a plain sum, with no transforms at
#     all.  But `Σ W_i² ` ranges over `[1/3, 1]`, so the bank is a frame with
#     bound ratio 3: analysis does not preserve energy, and white noise arrives
#     with subband-dependent variance.
#
#   * `tight = true` (the default) — the masks are `√W_i`, the pointwise
#     normalisation SurfBox uses.  Then `Σ (√W_i)² = Σ W_i = 1` identically: a
#     Parseval frame, `Σ_i ‖s_i‖² = ‖x‖²`, and the analysis adjoint is its own
#     inverse.  Synthesis applies the same masks, so reconstruction costs a
#     forward transform per subband plus one inverse instead of a bare sum.
#
# (SurfBox normalises to `Σ V_i² = N` rather than 1; the two differ only by the
# constant `√N`, and bound 1 is the convention the rest of this package uses.)

"""
    HourglassFB{T}

Precomputed 3-D nonsubsampled hourglass filter bank for one volume size and real
precision `T`.  Build with [`make_hourglass`](@ref).

Holds the two mapping-kernel frequency responses (`m1` on the 3-D rfft grid,
`m2` on the 2-D `(ω₂, ω₃)` grid) rather than the three subband masks, which are
derived pointwise on the fly — a third of the memory, and the polynomial
evaluation is negligible next to the transforms.
"""
mutable struct HourglassFB{
        T <: AbstractFloat, A3 <: AbstractArray{T, 3},
        C3 <: AbstractArray{Complex{T}, 3}, Pf, Pi,
    }
    const dims::NTuple{3, Int}
    const order::Int              # Bernstein order N of the mapping kernels
    const poly::Int               # order m of the 1-D halfband split polynomial
    const tight::Bool             # masks are √W (Parseval) rather than W
    # `f₀` of the two mapping responses, cached rather than the responses
    # themselves: the split polynomial is otherwise re-evaluated per voxel for
    # every subband, and a multiscale transform asks for dozens of them.
    #
    # Both are 3-D arrays with singleton axes where they do not depend on a
    # frequency, so every mask is a plain broadcast — no scalar indexing, which
    # is what lets the whole bank run on a GPU array.
    const m1::A3                  # f₀(M₁) on the rfft grid, (n₁÷2+1, n₂, n₃)
    const m2::A3                  # f₀(M₂) on the (ω₂, ω₃) grid, (1, n₂, n₃)
    const fwd::Pf
    const inv::Pi
    const spec::C3                # spectrum of the input, preserved across subbands
    const work::C3                # scratch consumed by the inverse plan
    const rbuf::A3
    # Second spectrum and output buffer, for the imaginary half of a complex
    # volume.  Allocated on first complex use so real-only workloads never pay
    # for them.  Two spectra cost about two real volumes — far less than holding
    # every subband's real part, which is what a two-pass scheme would need.
    spec2::Union{Nothing, C3}
    rbuf2::Union{Nothing, A3}
    # Full-size complex scratch, for handing a complex subband to a streaming
    # callback.  Allocated on first complex use for the same reason as the pair
    # above (128 MB at 256³ ComplexF32).
    cbuf::Union{Nothing, C3}
end

"""
    make_hourglass(dims::NTuple{3,Int}, ::Type{T}=Float64;
                    order::Int=12, poly::Int=1, tight::Bool=true,
                    threading=Auto(), flags=FFTW.MEASURE) -> HourglassFB

Build the 3-D hourglass filter bank for `dims`-sized volumes.

`order` is the Bernstein order `N` of the mapping kernels: larger `N` sharpens
the transition between hourglass subbands (Lu & Do use `N = 12`, which in their
FIR realisation corresponds to filters of support up to 61³ — here it costs
nothing but a slightly longer design step).  `poly` is the order `m` of the 1-D
halfband split polynomial; larger `m` flattens the response near the passband
centres.  Neither affects perfect reconstruction, which is exact for any choice.

`tight` selects how the three weights become filters.  With `tight = true` (the
default) the masks are `√W_i`, so the bank is a Parseval frame — analysis
preserves energy, `Σ_i ‖s_i‖² = ‖x‖²` — and synthesis applies the same masks,
costing one forward transform per subband plus one inverse.  With
`tight = false` the masks are the weights themselves, reconstruction is a bare
sum with no transforms, and the frame-bound ratio is 3.  Perfect reconstruction
is exact either way.

A complex element type is accepted and produces a plan of the corresponding real
precision; complex volumes are filtered as two real passes, keeping
`T(x + i·y) == T(x) + i·T(y)` exact.
"""
function make_hourglass(
        dims::NTuple{3, Int}, ::Type{T} = Float64;
        order::Int = 12, poly::Int = 1, tight::Bool = true,
        threading::ThreadingPolicy = Auto(),
        flags::Integer = FFTW.MEASURE
    ) where {T <: AbstractFloat}
    n1, n2, n3 = dims
    m1, m2 = _hourglass_tables(T, dims, order, poly)
    FFTW.set_num_threads(_fftw_threads(threading))
    rbuf = zeros(T, n1, n2, n3)
    fwd = plan_rfft(rbuf; flags = flags)
    spec = fwd * rbuf
    inv = plan_irfft(spec, n1; flags = flags)
    return HourglassFB{T, typeof(m1), typeof(spec), typeof(fwd), typeof(inv)}(
        dims, order, poly, tight, m1, m2, fwd, inv, spec, similar(spec), rbuf,
        nothing, nothing, nothing
    )
end

# `f₀` of the two mapping responses on the rfft grid of `dims`: `m1` is 3-D,
# `m2` depends on (ω₂, ω₃) only and has a singleton first axis.
function _hourglass_tables(::Type{T}, dims::NTuple{3, Int}, order::Int, poly::Int) where {T}
    n1, n2, n3 = dims
    all(>(0), dims) || throw(ArgumentError("volume dimensions must be positive"))
    order >= 1 || throw(ArgumentError("Bernstein order must be ≥ 1"))

    # Stage 1 — split P₁ out of the whole spectrum with a 3-D kernel.  Axis 1 is
    # the rfft axis, so only its non-negative half of the grid is needed.
    b1 = bernstein_table(order, dft_frequencies(T, n1; half = true))
    b2 = bernstein_table(order, dft_frequencies(T, n2))
    b3 = bernstein_table(order, dft_frequencies(T, n3))
    m1 = mapping_response(mapping_coefficients(order, 3, (1,)), (b1, b2, b3))
    map!(m -> halfband_split(m, poly), m1, m1)

    # Stage 2 — split P₂ from P₃ with a 2-D kernel on the (ω₂, ω₃) plane.
    m2f = mapping_response(mapping_coefficients(order, 2, (1,)), (b2, b3))
    map!(m -> halfband_split(m, poly), m2f, m2f)
    return m1, reshape(m2f, 1, n2, n3)
end

function make_hourglass(
        dims::NTuple{3, Int}, ::Type{Complex{T}}; kwargs...
    ) where {T <: AbstractFloat}
    fb = make_hourglass(dims, T; kwargs...)
    _complex_scratch!(fb)
    return fb
end

# Ensure the imaginary-half buffers exist; returns them.
function _complex_scratch!(fb::HourglassFB{T, A3}) where {T, A3}
    s2 = fb.spec2
    if s2 === nothing
        s2 = similar(fb.spec)
        fb.spec2 = s2
    end
    r2 = fb.rbuf2
    if r2 === nothing
        r2 = similar(fb.rbuf)
        fb.rbuf2 = r2
    end
    return s2, r2
end

# Ensure the full-size complex scratch exists; returns it.
function _volume_scratch!(fb::HourglassFB)
    c = fb.cbuf
    if c === nothing
        c = similar(fb.spec, fb.dims)
        fb.cbuf = c
    end
    return c
end

# Load one subband into a real scratch buffer.  A bare `copyto!` accepts any
# array of the same total length, silently reconstructing from a wrongly shaped
# subband, so the shape is checked here — the plain-sum path gets the same check
# for free from broadcasting.
function _load_subband!(dst::AbstractArray{T, 3}, band::AbstractArray) where {T}
    size(dst) == size(band) || throw(
        DimensionMismatch("subband size $(size(band)) does not match the plan $(size(dst))")
    )
    return copyto!(dst, band)
end

# Load the real and imaginary parts of `volume` into the two spectra.
function _load_complex_spectra!(fb::HourglassFB{T}, volume::AbstractArray{Complex{T}, 3}) where {T}
    spec2, _ = _complex_scratch!(fb)
    @. fb.rbuf = real(volume)
    mul!(fb.spec, fb.fwd, fb.rbuf)
    @. fb.rbuf = imag(volume)
    mul!(spec2, fb.fwd, fb.rbuf)
    return fb
end

make_hourglass(n1::Int, n2::Int, n3::Int, ::Type{Td} = Float64; kwargs...) where {Td} =
    make_hourglass((n1, n2, n3), Td; kwargs...)

"""
    hourglass_mask(fb::HourglassFB, i::Int) -> Array{T,3}

The frequency-domain analysis mask of hourglass subband `i` (1, 2 or 3) on the
rfft grid.  Materialising it is only useful for inspection and tests — the
transform derives mask values on the fly.

The three masks sum to 1 at every frequency when `fb.tight` is false, and their
*squares* sum to 1 when it is true.  See [`hourglass_weight`](@ref) for the
underlying partition of unity in either case.
"""
function hourglass_mask(fb::HourglassFB{T}, i::Int) where {T}
    1 <= i <= 3 || throw(ArgumentError("hourglass subband index must be 1, 2 or 3"))
    out = similar(fb.m1)
    # The transform kernel with a unit spectrum: the mask itself.
    _mask_multiply!(out, fb, true, i)
    return out
end

"""
    hourglass_weight(fb::HourglassFB, i::Int) -> Array{T,3}

The frequency-domain *weight* `W_i` of hourglass subband `i` on the rfft grid,
before the `tight` normalisation.  The three weights are a partition of unity —
`Σ_i W_i = 1` at every frequency — for any `order`/`poly`.
"""
function hourglass_weight(fb::HourglassFB{T}, i::Int) where {T}
    1 <= i <= 3 || throw(ArgumentError("hourglass subband index must be 1, 2 or 3"))
    out = similar(fb.m1)
    @. out = _stage1_weight(fb.m1, i) * _stage2_weight(fb.m2, i)
    return out
end

# The weights are exact partitions of unity in real arithmetic, but rounding can
# leave one a few ulps below zero, so the tight normalisation clamps.
@inline _tight_sqrt(x::T) where {T} = sqrt(max(x, zero(T)))

# Stage weights, written as broadcastable scalar functions of the cached f₀
# values plus a small integer mode.  `f₁ = 1 - f₀` is what makes the bank a
# partition of unity.
#
# Stage 1: f₀(M₁) for subband 1, f₁(M₁) otherwise.
@inline _stage1_weight(f0::T, i::Int) where {T} = i == 1 ? f0 : one(T) - f0

# Stage 2: 1 for subband 1 (stage 2 does not touch P₁), f₀(M₂) for 2, f₁(M₂)
# for 3.
@inline function _stage2_weight(f0::T, i::Int) where {T}
    i == 1 && return one(T)
    return i == 2 ? f0 : one(T) - f0
end

"""
    hourglass_decompose(volume, fb::HourglassFB) -> (s₁, s₂, s₃)

Split `volume` into the three hourglass subbands.  Every subband keeps the full
size of the input — the bank is nonsubsampled, so the decomposition is exactly
equivariant under circular shifts and the redundancy is 3 (`N` in `N`-D, which
Lu & Do show is unavoidable: hourglass partitioning cannot be critically sampled
above 2-D).

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(2), 16, 16, 16);

julia> fb = make_hourglass(size(x), Float64; order = 6);

julia> s = hourglass_decompose(x, fb);

julia> maximum(abs, hourglass_reconstruct(s, fb) .- x) < 1e-10
true
```
"""
function hourglass_decompose(volume::AbstractArray{<:Number, 3}, fb::HourglassFB)
    out = ntuple(_ -> similar(volume), 3)
    return hourglass_decompose!(out, volume, fb)
end

"""
    hourglass_decompose!(dst, volume, fb) -> dst

In-place [`hourglass_decompose`](@ref); `dst` is a 3-tuple of arrays matching
the plan size and the element type of `volume`.
"""
function hourglass_decompose!(
        dst::NTuple{3, <:AbstractArray{T, 3}}, volume::AbstractArray{T, 3},
        fb::HourglassFB{T}
    ) where {T <: AbstractFloat}
    size(volume) == fb.dims || throw(DimensionMismatch("volume size does not match the plan"))
    copyto!(fb.rbuf, volume)
    mul!(fb.spec, fb.fwd, fb.rbuf)
    for i in 1:3
        _mask_multiply!(fb.work, fb, fb.spec, i)
        mul!(fb.rbuf, fb.inv, fb.work)
        copyto!(dst[i], fb.rbuf)
    end
    return dst
end

function hourglass_decompose!(
        dst::NTuple{3, <:AbstractArray{Complex{T}, 3}}, volume::AbstractArray{Complex{T}, 3},
        fb::HourglassFB{T}
    ) where {T <: AbstractFloat}
    size(volume) == fb.dims || throw(DimensionMismatch("volume size does not match the plan"))
    # Real and imaginary parts go through the identical real kernel — the same
    # construction `conv3d_fft.jl` uses, and for the same reason: it keeps
    # `T(x + i·y) == T(x) + i·T(y)` exact.  Both spectra are held at once so each
    # subband is finished before moving to the next.
    _load_complex_spectra!(fb, volume)
    spec2, rbuf2 = _complex_scratch!(fb)
    for i in 1:3
        _mask_multiply!(fb.work, fb, fb.spec, i)
        mul!(fb.rbuf, fb.inv, fb.work)
        _mask_multiply!(fb.work, fb, spec2, i)
        mul!(rbuf2, fb.inv, fb.work)
        @. dst[i] = complex(fb.rbuf, rbuf2)
    end
    return dst
end

# The mask kernels, shared with the directional bank in `ndfb.jl`: the wedge
# factors `wa`, `wb` are `nothing` here and oriented wedge planes there, so the
# two banks differ by which arguments they pass and not by a second copy of the
# mask expression.  That matters beyond duplication — when `tight` is set the
# same mask is reapplied on synthesis, so the two expressions drifting apart
# would break the Parseval normalisation silently.
#
# `_wedge_scale` resolves `nothing` at compile time, so the hourglass bank does
# not pay two multiplies by an all-ones factor it does not have.
#
# The mask is never materialised on the transform paths: at 256³ Float32 it is a
# 34 MB write followed immediately by a 34 MB read, per subband.  Folding it into
# the multiply removes both while keeping the expression broadcast-only, which is
# what the GPU path needs.
@inline _wedge_scale(x, w) = x * w
@inline _wedge_scale(x, ::Nothing) = x

# dst .= spec .* mask, the mask being hourglass weight `i` times the wedges.
function _mask_multiply!(dst, fb::HourglassFB, spec, i::Int, wa = nothing, wb = nothing)
    m1, m2 = fb.m1, fb.m2
    if fb.tight
        @. dst = spec * _tight_sqrt(
            _wedge_scale(_wedge_scale(_stage1_weight(m1, i) * _stage2_weight(m2, i), wa), wb)
        )
    else
        @. dst = spec * _wedge_scale(
            _wedge_scale(_stage1_weight(m1, i) * _stage2_weight(m2, i), wa), wb
        )
    end
    return dst
end

# acc .+= spec .* mask — the synthesis counterpart, used only when tight.
function _mask_accumulate!(acc, fb::HourglassFB, spec, i::Int, wa = nothing, wb = nothing)
    m1, m2 = fb.m1, fb.m2
    @. acc += spec * _tight_sqrt(
        _wedge_scale(_wedge_scale(_stage1_weight(m1, i) * _stage2_weight(m2, i), wa), wb)
    )
    return acc
end

"""
    hourglass_reconstruct(subbands, fb::HourglassFB) -> volume

Invert [`hourglass_decompose`](@ref).  Perfect reconstruction holds exactly for
any `order`/`poly` — it is structural, not an approximation that degrades with
filter sharpness.

When `fb.tight` is false the synthesis filters are all 1 and this is a plain
sum, with no transforms.  When it is true the same `√W` masks are applied on the
way back, so each subband costs a forward transform and the sum is completed by
one inverse.
"""
function hourglass_reconstruct(subbands::NTuple{3, <:AbstractArray{Td, 3}}, fb::HourglassFB) where {Td}
    out = similar(subbands[1])
    return hourglass_reconstruct!(out, subbands, fb)
end

"""
    hourglass_reconstruct!(volume, subbands, fb) -> volume

In-place [`hourglass_reconstruct`](@ref).
"""
function hourglass_reconstruct!(
        volume::AbstractArray{Td, 3}, subbands::NTuple{3, <:AbstractArray{Td, 3}},
        fb::HourglassFB
    ) where {Td}
    size(volume) == fb.dims || throw(DimensionMismatch("volume size does not match the plan"))
    if !fb.tight
        s1, s2, s3 = subbands
        @. volume = s1 + s2 + s3
        return volume
    end
    return _hourglass_synthesise!(volume, subbands, fb)
end

function _hourglass_synthesise!(
        volume::AbstractArray{T, 3}, subbands, fb::HourglassFB{T}
    ) where {T <: AbstractFloat}
    # `work` is the buffer the inverse plan consumes and nothing else uses during
    # synthesis, so the masked spectra accumulate straight into it.
    acc = fb.work
    fill!(acc, zero(eltype(acc)))
    for i in 1:3
        _load_subband!(fb.rbuf, subbands[i])
        mul!(fb.spec, fb.fwd, fb.rbuf)
        _mask_accumulate!(acc, fb, fb.spec, i)
    end
    mul!(fb.rbuf, fb.inv, acc)
    copyto!(volume, fb.rbuf)
    return volume
end

function _hourglass_synthesise!(
        volume::AbstractArray{Complex{T}, 3}, subbands, fb::HourglassFB{T}
    ) where {T <: AbstractFloat}
    # Real and imaginary parts take the identical real kernel, as everywhere
    # else in the 3-D stack, so `T(x + i·y) == T(x) + i·T(y)` stays exact.
    spec2, rbuf2 = _complex_scratch!(fb)
    acc, acc2 = fb.work, spec2
    fill!(acc, zero(eltype(acc)))
    fill!(acc2, zero(eltype(acc2)))
    for i in 1:3
        band = subbands[i]
        @. fb.rbuf = real(band)
        mul!(fb.spec, fb.fwd, fb.rbuf)
        _mask_accumulate!(acc, fb, fb.spec, i)
        @. fb.rbuf = imag(band)
        mul!(fb.spec, fb.fwd, fb.rbuf)
        _mask_accumulate!(acc2, fb, fb.spec, i)
    end
    mul!(fb.rbuf, fb.inv, acc)
    mul!(rbuf2, fb.inv, acc2)
    @. volume = complex(fb.rbuf, rbuf2)
    return volume
end
