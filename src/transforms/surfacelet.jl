# Decimated surfacelet transform — the critically-sampled-ish counterpart of the
# nonsubsampled surfacelet transform.
#
# Reference for the target frequency partitioning: Y. M. Lu and M. N. Do,
# "Multidimensional Directional Filter Banks and Surfacelets", IEEE TIP 16(4),
# 2007, Section VI.  The ideal passbands are pairs of concentric cubes radiating
# out from the origin at different scales and directions.
#
# Why this exists.  The nonsubsampled transform is exactly shift-invariant and correspondingly
# redundant: `1 + Σ_j 3·2^{L_j}` copies of the input, which is 49 volumes (3.1 GB
# at 256³ Float32) for J=3, L=[2,2,3].  The surfacelet transform trades shift
# invariance for a redundancy of a small constant, which is what makes it usable
# as a sparsifying transform rather than only as an analysis tool.
#
# How it differs from the reference construction — deliberately, and worth
# reading before comparing coefficients with SurfBox:
#
#   * Lu & Do decimate with a multiscale pyramid whose first level resamples by
#     the non-integer factor 3/2 per axis, plus a critically sampled chain of
#     iteratively resampled checkerboard filter banks inside each hourglass.  The
#     numeric filter parameters for that pyramid are given in tables in the paper
#     which are not recoverable from its text, and the reference implementation
#     (SurfBox) is no longer distributed, so reproducing it exactly is not
#     possible from the sources at hand.
#
#   * Instead, decimation here is done by frequency-domain support extraction:
#     each subband mask has *compact* support in frequency, so the masked
#     spectrum is restricted to the smallest symmetric box containing it and
#     inverse-transformed at that reduced size.  This is the same mechanism the
#     wrapping-based discrete curvelet transform uses.  The resulting redundancy
#     is measured, not assumed — see `surfacelet_redundancy`.
#
# How far that gets, honestly.  Measured against the nonsubsampled transform at
# the same parameters:
#
#   | J, L          | surfacelet | NSurfacelet |
#   |---------------|-------------|-------|
#   | 1, [1]        |  6.2        |   7   |
#   | 2, [1, 1]     |  7.7        |  13   |
#   | 2, [2, 2]     | 15.3        |  25   |
#   | 3, [2, 2, 3]  | 16.0        |  49   |
#
# So this is a partially decimated transform: about 3× leaner than the NSurfacelet at
# useful settings, not the redundancy of about 4 that Lu & Do report.  The reason
# is specific and worth knowing before trying to tune the parameters for it: an
# *axis-aligned* box recovers the radial decimation in full (a coarser shell
# shrinks by 2 per axis) but almost none of the directional decimation, because
# the bounding box of a pyramid radiating from the origin is nearly as large as
# the pyramid's enclosing cube.  Closing that gap needs the box to be aligned
# with the wedge — sheared wrapping, or equivalently Lu & Do's resampled
# checkerboard chain — which in turn needs the shear to be an integer index
# remap on the sampling grid.  It is not, at dyadic wedge boundaries, unless the
# decimation is interleaved with the refinement the way the reference
# construction does it.  That is the missing piece, not a tuning problem.
#
#   * Consequently the directional masks are *not* the Bernstein mapping kernels
#     of `hourglass.jl`.  Those are the right choice there and the wrong one
#     here: a mapping kernel is a polynomial in `cos ω`, so `f₀(M(ω))` is nonzero
#     almost everywhere and its support box is the whole grid — nothing to
#     decimate.  The requirement inverts, so the construction does too: masks
#     here are built from soft steps that are *exactly* 0 and 1 outside a
#     transition band.
#
# Perfect reconstruction.  The masks `W` form a partition of unity and the
# transform uses `V = √W`, so `Σ V² = 1` and synthesis is the analysis adjoint:
# extraction is lossless because each mask vanishes outside its box, so
# `fft(ifft(restrict(Y))) == restrict(Y)` exactly.  Reconstruction is therefore
# exact, and the frame is tight in the weighted sense
# `Σ_i N_i ‖s_i‖² = N ‖x‖²` where `N_i` is the subband's element count.

"""
    SurfaceletParams

Describes a 3-D surfacelet decomposition.

- `J` — number of scales.  Scale `j` covers the frequency shell
  `‖ω‖_∞ ∈ [π/2^j, π/2^{j-1}]`, so it decimates by roughly `2^{j-1}` per axis.
- `L_array` — directional depth per scale, length `J`; scale `j` has
  `3 · 2^{L_array[j]}` directional subbands.
- `poly` — order of the soft-step polynomial (transition smoothness).
- `transition` — width of the overlap between neighbouring wedges, as a fraction
  of the wedge width.
- `radial_transition` — width of the overlap between neighbouring scale shells,
  as a fraction of the shell width.
- `angular_transition` — width, in radians of `|ω_i| - |ω_j|`, of the overlap
  between neighbouring hourglasses.

Wider transitions mean smoother, better-localised surfacelets and larger support
boxes; that trade-off is exactly the redundancy dial.

```jldoctest
julia> using Surfacelets

julia> SurfaceletParams(J = 2, L_array = [1, 2])
SurfaceletParams(J=2, L_array=[1, 2], …)
```
"""
struct SurfaceletParams{T <: AbstractFloat}
    J::Int
    L_array::Vector{Int}
    poly::Int
    transition::T
    radial_transition::T
    angular_transition::T
end

function SurfaceletParams(;
        J::Int,
        L_array::AbstractVector{<:Integer},
        poly::Int = 1,
        transition::Real = 0.5,
        radial_transition::Real = 0.3,
        angular_transition::Real = 0.3,
        precision::Type{T} = Float64
    ) where {T <: AbstractFloat}
    J >= 1 || throw(ArgumentError("J must be ≥ 1"))
    length(L_array) == J || throw(ArgumentError("length(L_array) must equal J"))
    all(>=(0), L_array) || throw(ArgumentError("L_array entries must be ≥ 0"))
    poly >= 0 || throw(ArgumentError("poly must be ≥ 0"))
    transition >= 0 || throw(ArgumentError("transition must be ≥ 0"))
    radial_transition > 0 || throw(ArgumentError("radial_transition must be > 0"))
    angular_transition > 0 || throw(ArgumentError("angular_transition must be > 0"))
    return SurfaceletParams{T}(
        J, collect(Int, L_array), poly,
        T(transition), T(radial_transition), T(angular_transition)
    )
end

Base.eltype(::SurfaceletParams{T}) where {T} = T

function Base.show(io::IO, p::SurfaceletParams)
    return print(io, "SurfaceletParams(J=$(p.J), L_array=$(p.L_array), …)")
end

"""
    surfacelet_nsubbands(p::SurfaceletParams, j::Int) -> Int

Number of directional subbands at scale `j`, `3 * 2^L_array[j]`.
"""
surfacelet_nsubbands(p::SurfaceletParams, j::Int) = 3 * (1 << p.L_array[j])

# One subband's frequency box.
#
# The box is symmetric about the origin, so it is described completely by its
# per-axis reach in signed frequency: in FFT order it is the union of the blocks
# `1:reach+1` (the non-negative frequencies) and `n-reach+1:n` (the negative
# ones).  Storing it that way rather than as a per-voxel index map makes
# extraction and accumulation a handful of *contiguous block* copies, so both
# are plain broadcasts over array views — which is what lets the whole transform
# run unchanged on a device array.
struct SubbandBox
    size::NTuple{3, Int}
    reach::NTuple{3, Int}
    nonempty::Bool
end

# The (source, destination) index blocks of one axis, always two pairs so the
# result is type-stable; the second pair is empty when the axis is not decimated.
@inline function _axis_blocks(reach::Int, b::Int, n::Int)
    b == n && return ((1:n, 1:n), (1:0, 1:0))
    return ((1:(reach + 1), 1:(reach + 1)), ((n - reach + 1):n, (reach + 2):b))
end

@inline _box_blocks(box::SubbandBox, dims::NTuple{3, Int}) =
    ntuple(d -> _axis_blocks(box.reach[d], box.size[d], dims[d]), 3)

"""
    SurfaceletPlan{T}

Precomputed masks metadata and FFT plans for one volume size.  Build with
[`make_surfacelet`](@ref).

The per-subband masks are *not* stored — at 256³ that would be gigabytes.  Only
the 1-D frequency tables, the per-scale wedge boundaries, and each subband's
frequency box are kept; masks are evaluated on the fly.

The buffers and frequency tables are stored as abstract array types, so a plan
built here can be moved to a device with `Adapt.adapt(CuArray, plan)` (the GPU
extension replans the FFTs, which cannot be adapted).  The support boxes are
plain integers and stay shared.
"""
mutable struct SurfaceletPlan{
        T <: AbstractFloat, A <: AbstractArray{<:Complex, 3},
        Am <: AbstractArray{<:Real, 3}, Vt <: AbstractVector, Vi <: AbstractVector, P, Pi,
    }
    const dims::NTuple{3, Int}
    const params::SurfaceletParams{T}
    const wfreq::NTuple{3, Vt}             # ω_d folded to (-π, π]
    const signed::NTuple{3, Vi}            # signed frequency index per axis
    const negsigned::NTuple{3, Vi}         # signed index of the negated frequency
    const radii::Vector{T}                 # shell boundaries π, π/2, …, π/2^J
    const boxes::Vector{Vector{SubbandBox}}   # [scale][subband]
    # Filled in once the masks have been scanned; the scan needs the plan.
    coarse_box::SubbandBox
    const fwd::P
    const inv::Pi                          # in-place inverse for the full grid
    # In-place forward/inverse plans per distinct box size, built on first use.
    # Every subband of every call transforms at its box size, so planning them
    # per call — which is what a bare `fft!`/`ifft!` does — is pure overhead.
    const boxplans::Dict{NTuple{3, Int}, Tuple{Any, Any}}
    const flags::UInt32
    const spec::A
    const work::A
    const mask::Am                         # full-grid mask scratch
end

# Outer constructor: the seven type parameters follow from the arguments, and
# spelling them out at every call site (here and in the GPU extension's adapt
# method) is only a chance to get them wrong.
function SurfaceletPlan(
        dims::NTuple{3, Int}, params::SurfaceletParams{T}, wfreq::NTuple{3, Vt},
        signed::NTuple{3, Vi}, negsigned::NTuple{3, Vi}, radii::Vector{T},
        boxes::Vector{Vector{SubbandBox}}, coarse_box::SubbandBox, fwd, inv,
        boxplans::Dict{NTuple{3, Int}, Tuple{Any, Any}}, flags::UInt32, spec::A,
        work::A, mask::Am
    ) where {T, A, Am, Vt, Vi}
    return SurfaceletPlan{T, A, Am, Vt, Vi, typeof(fwd), typeof(inv)}(
        dims, params, wfreq, signed, negsigned, radii, boxes, coarse_box,
        fwd, inv, boxplans, flags, spec, work, mask
    )
end

# FFTW takes planner flags; the device backends do not have them, and passing
# the keyword through would be a `MethodError` on a `CuArray`.
_plan_fft_out(buf::Array, flags::Integer) = plan_fft(buf; flags = flags)
_plan_fft_out(buf, ::Integer) = plan_fft(buf)
_plan_fft_in(buf::Array, flags::Integer) = plan_fft!(buf; flags = flags)
_plan_fft_in(buf, ::Integer) = plan_fft!(buf)
_plan_ifft_in(buf::Array, flags::Integer) = plan_ifft!(buf; flags = flags)
_plan_ifft_in(buf, ::Integer) = plan_ifft!(buf)

"""
    make_surfacelet(params::SurfaceletParams, dims::NTuple{3,Int}, ::Type{T}=Float64;
                     threading=Auto(), flags=FFTW.MEASURE) -> SurfaceletPlan

Plan the surfacelet transform for `dims`-sized volumes.

Planning scans every subband mask once to find its frequency support, so it
costs one full-grid pass per subband.  Reuse the plan.
"""
function make_surfacelet(
        params::SurfaceletParams, dims::NTuple{3, Int}, ::Type{T} = Float64;
        threading::ThreadingPolicy = Auto(),
        flags::Integer = FFTW.MEASURE
    ) where {T <: AbstractFloat}
    all(>(0), dims) || throw(ArgumentError("volume dimensions must be positive"))
    p = _convert_sparams(T, params)
    wfreq = ntuple(d -> _folded_frequencies(T, dims[d]), 3)
    signed = ntuple(d -> _signed_indices(dims[d]), 3)
    negsigned = ntuple(d -> signed[d][_negation_indices(dims[d])], 3)
    radii = T[T(π) / 2^j for j in 0:p.J]

    FFTW.set_num_threads(_fftw_threads(threading))
    buf = zeros(Complex{T}, dims)
    fwd = plan_fft(buf; flags = flags)
    inv = plan_ifft!(buf; flags = flags)
    spec = Array{Complex{T}, 3}(undef, dims)
    work = Array{Complex{T}, 3}(undef, dims)
    mask = Array{T, 3}(undef, dims)

    plan = SurfaceletPlan(
        dims, p, wfreq, signed, negsigned, radii,
        [SubbandBox[] for _ in 1:p.J], _full_box(dims), fwd, inv,
        Dict{NTuple{3, Int}, Tuple{Any, Any}}(), UInt32(flags), spec, work, mask
    )
    # Support boxes, found by scanning each mask once.
    for j in 1:p.J
        for k in 1:surfacelet_nsubbands(p, j)
            _fill_surfacelet_mask!(mask, plan, j, k)
            push!(plan.boxes[j], _support_box(mask, plan))
        end
    end
    _fill_coarse_mask!(mask, plan)
    plan.coarse_box = _support_box(mask, plan)
    return plan
end

# In-place forward and inverse plans for one box size, built once per size.
function _box_plans(pl::SurfaceletPlan, sz::NTuple{3, Int})
    return get!(pl.boxplans, sz) do
        # Planning may scribble on `buf`; it is scratch and discarded here.
        buf = similar(pl.spec, sz)
        (_plan_fft_in(buf, pl.flags), _plan_ifft_in(buf, pl.flags))
    end
end

make_surfacelet(params::SurfaceletParams, dims::Vararg{Int, 3}; kwargs...) =
    make_surfacelet(params, dims; kwargs...)

_convert_sparams(::Type{T}, p::SurfaceletParams{T}) where {T} = p
_convert_sparams(::Type{T}, p::SurfaceletParams) where {T <: AbstractFloat} =
    SurfaceletParams{T}(
    p.J, p.L_array, p.poly,
    T(p.transition), T(p.radial_transition), T(p.angular_transition)
)

# Signed frequency index of each DFT bin: 0, 1, …, ⌊n/2⌋, then negative.
_signed_indices(n::Int) = Int[k <= n ÷ 2 ? k : k - n for k in 0:(n - 1)]

# Index of the bin holding -ω, per axis.  This is an involution, and it is *not*
# always the sign flip of the signed index: at the Nyquist bin of an even-length
# axis, π ≡ -π maps to itself.
_negation_indices(n::Int) = Int[mod(-(k - 1), n) + 1 for k in 1:n]

_full_box(dims::NTuple{3, Int}) = SubbandBox(dims, dims, true)

"""
    SurfaceletCoefficients{T}

Output of [`surfacelet_forward`](@ref).  Unlike the nonsubsampled containers the arrays
here have *different* sizes: each subband is stored at its own decimated
resolution, which is the whole point.
"""
struct SurfaceletCoefficients{T <: AbstractFloat, A <: AbstractArray{T, 3}}
    dims::NTuple{3, Int}     # size of the volume this came from
    coarse::A
    subbands::Vector{Vector{A}}
end

Base.eltype(::SurfaceletCoefficients{T}) where {T} = T

function Base.show(io::IO, c::SurfaceletCoefficients)
    ns = sum(length, c.subbands; init = 0)
    n = length(c.coarse) + sum(sum(length, s; init = 0) for s in c.subbands; init = 0)
    return print(
        io, "SurfaceletCoefficients(dims=$(c.dims), J=$(length(c.subbands)), ",
        "subbands=$(ns), coefficients=$(n))"
    )
end

"""
    surfacelet_redundancy(plan::SurfaceletPlan) -> Float64

Total stored coefficients divided by the number of input voxels.  This is
measured from the plan's actual support boxes, so it reflects the transition
widths that were chosen rather than an idealised figure.
"""
function surfacelet_redundancy(plan::SurfaceletPlan)
    total = prod(plan.coarse_box.size)
    for scale in plan.boxes, box in scale
        total += prod(box.size)
    end
    return total / prod(plan.dims)
end

# ── Mask construction ────────────────────────────────────────────────────────
#
# Every factor below is exactly 0 or 1 outside a transition band, which is what
# makes the support boxes small.  `halfband_split` on a clamped argument gives
# that: `f₀(-1) = 0` and `f₀(1) = 1` exactly, and `f₀(x) + f₀(-x) = 1` makes the
# factors telescope to a partition of unity.

# The soft step (`_step`) and the wedge partition (`_wedge_weight`) are shared
# with the nonsubsampled bank — see `filters/bernstein.jl` and
# `directional/ndfb.jl`.  Only the radial and hourglass factors below are
# specific to the decimated transform.

# 1 well *below* the shell boundary at `b`, 0 well above it: a soft low-pass
# indicator of that edge, which is what makes the differences taken in
# `_surf_mask_value` telescope to 1 over the scales plus the coarse band.
@inline _radial_edge(r::T, b::T, rt::T, poly::Int) where {T} =
    one(T) - _step((r - b) / (rt * b), poly)

# Unnormalised hourglass weight of axis `i`: 1 where |ω_i| dominates both other
# axes by more than the transition width, 0 where either other axis dominates by
# that much.  Normalising by the sum over i gives the partition of unity, and
# the sum is bounded away from zero because whichever axis is largest
# contributes at least `f₀(0)²`.
@inline function _hourglass_weight(a::NTuple{3, T}, i::Int, delta::T, poly::Int) where {T}
    w = one(T)
    for j in 1:3
        j == i && continue
        w *= _step((a[i] - a[j]) / delta, poly)
    end
    return w
end

# Everything one subband's mask needs that is not a per-voxel frequency.  It is
# `isbits`, so it rides into a broadcast — including a GPU one — as a scalar.
struct _SurfMaskParams{T}
    dims::NTuple{3, Int}
    poly::Int
    delta::T          # angular transition width
    transition::T     # wedge transition width
    b_in::T           # inner shell boundary, radii[j + 1]
    b_out::T          # outer shell boundary, radii[j]; unused when `first`
    rt::T             # radial transition width, as a fraction of the boundary
    first::Bool       # scale 1 has no outer shell edge — the weight is 1 there
    hg::Int           # hourglass axis
    ax::Int           # the two transverse axes of `hg`
    bx::Int
    ip::Int           # wedge index on each transverse plane
    iq::Int
    na::Int           # wedge counts on each transverse plane
    nb::Int
end

# W = radial_j · normalised hourglass · wedge_p · wedge_q, then V = √W, at one
# voxel.  `q` is the signed frequency index of the voxel and `n` that of its
# negation — the two are *not* related by a sign flip on the DFT grid, see below.
@inline function _surf_mask_value(
        q1::Int, q2::Int, q3::Int, n1::Int, n2::Int, n3::Int, mp::_SurfMaskParams{T}
    ) where {T}
    # Evaluate at the canonical representative of {ω, -ω}.  The wedge ratios are
    # sign-sensitive, and on the DFT grid ω → -ω is *not* simply a sign flip of
    # the folded frequency: at the Nyquist bin of an even axis, π ≡ -π maps to
    # itself, so `ω₂/ω₁` there would come out with opposite signs at two bins
    # that are each other's negation.  That asymmetry makes the masked spectrum
    # non-Hermitian and the subband complex — and since the subbands are stored
    # real, it silently destroys perfect reconstruction.  Picking one member of
    # each ± pair by a fixed rule removes the ambiguity, evaluates each
    # direction once, and keeps the mask exactly even.
    q = (q1, q2, q3)
    qn = (n1, n2, n3)
    qq = qn > q ? qn : q
    w = (
        2 * T(π) * qq[1] / mp.dims[1],
        2 * T(π) * qq[2] / mp.dims[2],
        2 * T(π) * qq[3] / mp.dims[3],
    )
    a = (abs(w[1]), abs(w[2]), abs(w[3]))
    r = max(a[1], a[2], a[3])
    inner = _radial_edge(r, mp.b_in, mp.rt, mp.poly)
    outer = mp.first ? one(T) : _radial_edge(r, mp.b_out, mp.rt, mp.poly)
    rad = outer - inner
    rad <= 0 && return zero(T)
    hw = _hourglass_weight(a, mp.hg, mp.delta, mp.poly)
    hw <= 0 && return zero(T)
    # The normalising sum includes `hw`, so only the other two are new.
    norm = hw
    for t in 1:3
        t == mp.hg && continue
        norm += _hourglass_weight(a, t, mp.delta, mp.poly)
    end
    # Signed ratios: the wedge slabs span [-1, 1], so using |ω_a|/|ω_i| here
    # would fold the two halves together and starve half the wedges.
    ref = w[mp.hg]
    s = ref == 0 ? zero(T) : w[mp.ax] / ref
    t = ref == 0 ? zero(T) : w[mp.bx] / ref
    wv = _wedge_weight(s, mp.ip, mp.na, mp.transition, mp.poly) *
        _wedge_weight(t, mp.iq, mp.nb, mp.transition, mp.poly)
    return sqrt(rad * (hw / norm) * wv)
end

# The mask is a pointwise function of the three frequency indices, so filling it
# is one broadcast over the per-axis tables reshaped onto their own dimension.
# That is what keeps the transform device-capable: no scalar indexing anywhere.
function _fill_surfacelet_mask!(
        out::AbstractArray{<:Real, 3}, plan::SurfaceletPlan{T}, j::Int, k::Int
    ) where {T}
    p = plan.params
    na, nb = _wedge_counts(p.L_array[j])
    hg, ip, iq = _subband_triple(k, na, nb)
    ax, bx = _transverse_axes(hg)
    mp = _SurfMaskParams{T}(
        plan.dims, p.poly, p.angular_transition, p.transition,
        plan.radii[j + 1], plan.radii[max(j, 2)], p.radial_transition, j == 1,
        hg, ax, bx, ip, iq, na, nb
    )
    s1, s2, s3 = plan.signed
    g1, g2, g3 = plan.negsigned
    out .= _surf_mask_value.(
        reshape(s1, :, 1, 1), reshape(s2, 1, :, 1), reshape(s3, 1, 1, :),
        reshape(g1, :, 1, 1), reshape(g2, 1, :, 1), reshape(g3, 1, 1, :), Ref(mp)
    )
    return out
end

@inline _coarse_mask_value(x1::T, x2::T, x3::T, b::T, rt::T, poly::Int) where {T} =
    sqrt(_radial_edge(max(abs(x1), abs(x2), abs(x3)), b, rt, poly))

function _fill_coarse_mask!(
        out::AbstractArray{<:Real, 3}, plan::SurfaceletPlan{T}
    ) where {T}
    p = plan.params
    w1, w2, w3 = plan.wfreq
    out .= _coarse_mask_value.(
        reshape(w1, :, 1, 1), reshape(w2, 1, :, 1), reshape(w3, 1, 1, :),
        plan.radii[p.J + 1], p.radial_transition, p.poly
    )
    return out
end

# ── Support boxes ────────────────────────────────────────────────────────────

# Smallest box symmetric about the origin containing the mask's support, as
# per-axis index maps.  Symmetry matters: the masks are even in ω, so a
# symmetric box keeps the restricted spectrum Hermitian and the subband real.
function _support_box(mask::AbstractArray{T, 3}, plan::SurfaceletPlan{T}) where {T}
    dims = plan.dims
    reach = zeros(Int, 3)
    found = false
    @inbounds for k3 in axes(mask, 3), k2 in axes(mask, 2), k1 in axes(mask, 1)
        mask[k1, k2, k3] > 0 || continue
        found = true
        q = (plan.signed[1][k1], plan.signed[2][k2], plan.signed[3][k3])
        for d in 1:3
            reach[d] = max(reach[d], abs(q[d]))
        end
    end
    found || return SubbandBox((1, 1, 1), (0, 0, 0), false)
    sizes = ntuple(d -> min(2 * reach[d] + 1, dims[d]), 3)
    return SubbandBox(sizes, ntuple(d -> reach[d], 3), true)
end

# ── Transform ────────────────────────────────────────────────────────────────

"""
    surfacelet_forward(volume, params::SurfaceletParams; plan=nothing) -> SurfaceletCoefficients

Decimated 3-D surfacelet transform.  Each subband is returned at its own reduced
resolution, so the total coefficient count is a small multiple of the input —
see [`surfacelet_redundancy`](@ref).

Not shift-invariant: that is what the decimation costs, and it is why
[`nsurfacelet_forward`](@ref) still exists.

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(7), 32, 32, 32);

julia> p = SurfaceletParams(J = 2, L_array = [1, 1]);

julia> c = surfacelet_forward(x, p);

julia> maximum(abs, surfacelet_inverse(c, p) .- x) < 1e-10
true
```
"""
function surfacelet_forward(
        volume::AbstractArray{<:Real, 3}, params::SurfaceletParams;
        plan::Union{SurfaceletPlan, Nothing} = nothing
    )
    T = _filter_eltype(_data_eltype(volume))
    pl = _surfacelet_plan(plan, params, size(volume), T, volume)
    p = pl.params

    pl.work .= volume
    mul!(pl.spec, pl.fwd, pl.work)                 # the spectrum stays in `spec`

    mask = pl.mask
    A = typeof(mask)
    subbands = Vector{Vector{A}}(undef, p.J)
    for j in 1:p.J
        subbands[j] = Vector{A}(undef, surfacelet_nsubbands(p, j))
        for k in eachindex(subbands[j])
            _fill_surfacelet_mask!(mask, pl, j, k)
            subbands[j][k] = _extract_subband(pl, mask, pl.boxes[j][k])
        end
    end
    _fill_coarse_mask!(mask, pl)
    coarse = _extract_subband(pl, mask, pl.coarse_box)
    return SurfaceletCoefficients(pl.dims, coarse, subbands)
end

"""
    surfacelet_inverse(coeffs::SurfaceletCoefficients, params::SurfaceletParams; plan=nothing) -> volume

Invert [`surfacelet_forward`](@ref).  Exact: the masks satisfy `Σ V² = 1` and
each vanishes outside its box, so restriction loses nothing.
"""
function surfacelet_inverse(
        coeffs::SurfaceletCoefficients{T}, params::SurfaceletParams;
        plan::Union{SurfaceletPlan, Nothing} = nothing
    ) where {T}
    pl = _surfacelet_plan(plan, params, coeffs.dims, T, coeffs.coarse)
    p = pl.params
    length(coeffs.subbands) == p.J ||
        throw(ArgumentError("coefficients were produced for a different number of scales"))

    fill!(pl.spec, zero(eltype(pl.spec)))
    mask = pl.mask
    for j in 1:p.J
        length(coeffs.subbands[j]) == surfacelet_nsubbands(p, j) ||
            throw(DimensionMismatch("scale $(j) has the wrong number of subbands"))
        for k in eachindex(coeffs.subbands[j])
            _fill_surfacelet_mask!(mask, pl, j, k)
            _accumulate_subband!(pl, coeffs.subbands[j][k], mask, pl.boxes[j][k])
        end
    end
    _fill_coarse_mask!(mask, pl)
    _accumulate_subband!(pl, coeffs.coarse, mask, pl.coarse_box)

    spec = pl.inv * pl.spec
    out = similar(pl.mask, pl.dims)
    out .= real.(spec)
    return out
end

function _surfacelet_plan(
        plan::Union{SurfaceletPlan, Nothing}, params::SurfaceletParams,
        dims::NTuple{3, Int}, ::Type{T}, proto::AbstractArray
    ) where {T}
    if plan === nothing
        # Planning scans every mask, so a device plan is built on the host and
        # adapted once rather than rebuilt per call; there is nothing sensible to
        # do implicitly here.
        proto isa Array || throw(
            ArgumentError(
                """
                a device-resident volume needs an explicit plan: \
                Adapt.adapt(CuArray, make_surfacelet(params, dims, T))\
                """
            )
        )
        return make_surfacelet(params, dims, T)
    end
    plan.dims == dims || throw(DimensionMismatch("plan size does not match the volume"))
    return plan
end

# Restrict the masked spectrum to the subband's box and transform back at box
# size.  The mask is even in ω and the box is symmetric, so the restricted data
# is Hermitian on the box grid and the result is real up to rounding.
function _extract_subband(
        pl::SurfaceletPlan, mask::AbstractArray{<:Real, 3}, box::SubbandBox
    )
    buf = similar(pl.spec, box.size)
    fill!(buf, zero(eltype(buf)))
    if box.nonempty
        spec = pl.spec
        bl = _box_blocks(box, pl.dims)
        for (s3, d3) in bl[3], (s2, d2) in bl[2], (s1, d1) in bl[1]
            @views buf[d1, d2, d3] .= spec[s1, s2, s3] .* mask[s1, s2, s3]
        end
    end
    buf = _box_plans(pl, box.size)[2] * buf
    out = similar(pl.mask, box.size)
    out .= real.(buf)
    return out
end

# Transform the subband back up to its box spectrum and accumulate it, masked,
# into the full spectrum.
function _accumulate_subband!(
        pl::SurfaceletPlan, band::AbstractArray{<:Real, 3},
        mask::AbstractArray{<:Real, 3}, box::SubbandBox
    )
    size(band) == box.size ||
        throw(DimensionMismatch("subband is $(size(band)), expected $(box.size)"))
    box.nonempty || return pl.spec
    buf = similar(pl.spec, box.size)
    buf .= band
    buf = _box_plans(pl, box.size)[1] * buf
    spec = pl.spec
    bl = _box_blocks(box, pl.dims)
    for (s3, d3) in bl[3], (s2, d2) in bl[2], (s1, d1) in bl[1]
        @views spec[s1, s2, s3] .+= buf[d1, d2, d3] .* mask[s1, s2, s3]
    end
    return spec
end
