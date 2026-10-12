# Decimated surfacelet transform — Lu & Do, "Multidimensional Directional
# Filter Banks and Surfacelets", IEEE TIP 16(4), 2007, Section VI, as
# implemented by their reference toolbox SurfBox (`Surfdec.m`, `Surfrec.m`).
#
# Three stages, all in the frequency domain:
#
#   1. A multiscale pyramid (`fourier_pyramid.jl`) splits the spectrum into
#      bandpass shells and a coarse lowpass, resampling by 3/2 or 2 per level.
#   2. At every scale, the undecimated "ritf" hourglass bank (`ritf.jl`) splits
#      the shell into three hourglasses, one per axis — redundancy 3, which Lu
#      & Do prove is the minimum for an hourglass partition above 2-D.
#   3. Inside each hourglass, iterated resampled checkerboard filter banks
#      (`irc.jl`) refine it into wedges *and decimate*, critically: each wedge
#      is stored on a lattice aligned with it.
#
# Stages 2 and 3 form the per-scale directional stage (`surfacelet_scale.jl`).
#
# So the redundancy is about 3 per scale times the geometric sum of the scale
# sizes — ≈ 4 for pyramid mode 1.5 — and does not grow with the number of
# directions.  The stages reproduce SurfBox to rounding error; `test_irc.jl`
# checks every stage and the whole transform against golden data generated
# with it.
#
# The checkerboard filters are biorthogonal, so the transform is a frame with
# perfect reconstruction but not a tight one: `surfacelet_inverse` is the
# inverse, not the adjoint, of `surfacelet_forward`.
#
# The semi-translation-invariant transform (`sti_surfacelet.jl`) is this one
# with a nonsubsampled pyramid, so both run on one plan type and one driver; the
# pyramid is a vector of `FourierPyramidLevel`s or of `RaisedCosineLevel`s, and
# dispatch on it picks the stage-1 functions.

"""
    SurfaceletPlan

Everything a decimated or semi-translation-invariant surfacelet transform needs
for one volume size: the pyramid profiles, hourglass masks and checkerboard
filter slices, and the spectra, block buffers and FFT plans the transform runs
in, so that the `!` transforms allocate nothing.  Build with
[`make_surfacelet`](@ref) or [`make_sti_surfacelet`](@ref) and pass as `plan =`.

A plan holds its own scratch buffers, so one plan must not run two transforms at
the same time; give each task its own.  `Adapt.adapt(ArrayType, plan)` moves it
to a device, rebuilding the buffers and FFT plans there.
"""
struct SurfaceletPlan{T, P, L, C <: _SurfScale{T}, W <: _ScaleWork, A <: AbstractArray{Complex{T}, 3}, F, G}
    dims::NTuple{3, Int}
    params::P
    pyramid::Vector{L}
    scales::Vector{C}
    work::Vector{W}         # scales with equal tables share one
    spectra::Vector{A}      # see `_pyramid_sizes`
    pfwd::F                 # in-place FFT and unnormalised inverse on spectra[1] …
    pbwd::G
    cfwd::F                 # … and on the coarse spectrum
    cbwd::G
end

Base.eltype(::SurfaceletPlan{T}) where {T} = T

# The plan around `pyramid` and `scales`, with buffers like `proto` (host or
# device).  Host FFT plans use `flags` on one FFTW thread.
function _surfacelet_plan(
        dims::NTuple{3, Int}, params, pyramid::AbstractVector, scales::AbstractVector{<:_SurfScale{T}},
        proto::AbstractArray, flags
    ) where {T}
    return _fftw_single_threaded(proto) do
        works = IdDict{Any, Any}()
        work = [get!(() -> _scale_work(sc, proto, flags), works, sc.stages) for sc in scales]
        Tc = Complex{T}
        spectra = [fill!(similar(proto, Tc, n), zero(Tc)) for n in _pyramid_sizes(pyramid)]
        pfwd, pbwd = _inplace_plans(spectra[1], flags)
        C = _coarse_spectrum(spectra, pyramid)
        cfwd, cbwd = C === spectra[1] ? (pfwd, pbwd) : _inplace_plans(C, flags)
        SurfaceletPlan(dims, params, collect(pyramid), collect(scales), [w for w in work], spectra, pfwd, pbwd, cfwd, cbwd)
    end
end

_inplace_plans(A::AbstractArray, flags) = (_fft_plan(plan_fft!, A, flags), _fft_plan(plan_bfft!, A, flags))

# Host plans are measured with one FFTW thread: the plan is applied to many
# small blocks, where FFTW's threads cost more than they gain.
_fftw_single_threaded(f::F, proto::AbstractArray) where {F} = proto isa Array ? _with_fftw_threads(f, 1) : f()

"""
    make_surfacelet(params::SurfaceletParams, dims, T=Float64; flags=FFTW.MEASURE) -> SurfaceletPlan

Precompute everything the transform needs for a `dims` volume.  The pyramid
must be able to resample every scale exactly, so with `pyr_mode = 1.5` each
dimension must be divisible by `3 · 2^J`, with `2` by `2^(J+1)`, and with `1` by
`2^J`; each scale must also allow its directional levels (axis length divisible
by `2^level`).  `flags` are the FFTW planner flags; pass `FFTW.ESTIMATE` for a
plan used once.
"""
function make_surfacelet(
        params::SurfaceletParams, dims::NTuple{3, Int}, ::Type{T} = Float64; flags::Integer = FFTW.MEASURE
    ) where {T <: AbstractFloat}
    p = _convert_sparams(T, params)
    pyramid = _fourier_pyramid(T, dims, p.J, p.pyr_mode)
    scales = map(1:p.J) do j
        n = pyramid[j].n
        masks = _ritf_masks(T, n; msize = p.msize, beta = p.beta, lambda = p.lambda)
        _SurfScale(n, p.levels[j], masks, _surf_stages(T, n, p.levels[j], p.bo))
    end
    return _surfacelet_plan(dims, p, pyramid, scales, Array{T, 3}(undef, 0, 0, 0), flags)
end

make_surfacelet(params::SurfaceletParams, dims::Vararg{Int, 3}; kwargs...) =
    make_surfacelet(params, dims; kwargs...)

"""
    surfacelet_redundancy(plan::SurfaceletPlan) -> Float64

Total stored coefficients divided by the number of input voxels; exactly
`1 + 3J` for a semi-translation-invariant plan.
"""
function surfacelet_redundancy(plan::SurfaceletPlan)
    total = length(_coarse_spectrum(plan.spectra, plan.pyramid)) + _directional_count(plan.scales)
    return total / prod(plan.dims)
end

"""
    surfacelet_forward(volume, params::SurfaceletParams; plan=nothing) -> SurfaceletCoefficients

Decimated 3-D surfacelet transform.  Each subband is returned at its own reduced
resolution; the total coefficient count is about 4 times the input with the
default `pyr_mode = 1.5`, whatever the number of directions — see
[`surfacelet_redundancy`](@ref).

Not shift-invariant: that is what the decimation costs, and it is why
[`ns_surfacelet_forward`](@ref) still exists.

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(7), 48, 48, 48);

julia> p = SurfaceletParams(J = 2, L_array = [1, 2]);

julia> c = surfacelet_forward(x, p);

julia> maximum(abs, surfacelet_inverse(c, p) .- x) < 1e-10
true
```
"""
function surfacelet_forward(volume::AbstractArray{<:Real, 3}, params::SurfaceletParams; plan::Union{SurfaceletPlan, Nothing} = nothing)
    pl = _resolve_plan(make_surfacelet, plan, params, volume)
    return _surfacelet_forward!(_surfacelet_coefficients(pl), volume, pl)
end

"""
    surfacelet_forward!(coeffs, volume, params::SurfaceletParams; plan=nothing) -> coeffs

[`surfacelet_forward`](@ref) into the preallocated `coeffs` (a result of an
earlier call with the same plan); allocates nothing when `plan` is given.
"""
function surfacelet_forward!(
        coeffs::SurfaceletCoefficients, volume::AbstractArray{<:Real, 3}, params::SurfaceletParams;
        plan::Union{SurfaceletPlan, Nothing} = nothing
    )
    pl = _resolve_plan(make_surfacelet, plan, params, volume)
    return _surfacelet_forward!(_check_coefficients(coeffs, pl), volume, pl)
end

"""
    surfacelet_inverse(coeffs::SurfaceletCoefficients, params::SurfaceletParams; plan=nothing) -> volume

Invert [`surfacelet_forward`](@ref) exactly (to rounding error).
"""
function surfacelet_inverse(coeffs::SurfaceletCoefficients, params::SurfaceletParams; plan::Union{SurfaceletPlan, Nothing} = nothing)
    pl = _resolve_plan(make_surfacelet, plan, params, coeffs)
    return _surfacelet_inverse!(similar(coeffs.coarse, pl.dims), coeffs, pl)
end

"""
    surfacelet_inverse!(volume, coeffs, params::SurfaceletParams; plan=nothing) -> volume

[`surfacelet_inverse`](@ref) into the preallocated `volume`; allocates nothing
when `plan` is given.
"""
function surfacelet_inverse!(
        volume::AbstractArray{<:Real, 3}, coeffs::SurfaceletCoefficients, params::SurfaceletParams;
        plan::Union{SurfaceletPlan, Nothing} = nothing
    )
    pl = _resolve_plan(make_surfacelet, plan, params, coeffs)
    size(volume) == pl.dims || throw(DimensionMismatch("volume has size $(size(volume)), expected $(pl.dims)"))
    return _surfacelet_inverse!(volume, coeffs, pl)
end

# ── Driver shared with the STI transform ──────────────────────────────────────

function _surfacelet_forward!(c::SurfaceletCoefficients, volume::AbstractArray{<:Real, 3}, pl::SurfaceletPlan)
    size(volume) == pl.dims || throw(DimensionMismatch("volume has size $(size(volume)), expected $(pl.dims)"))
    X = pl.spectra[1]
    X .= volume
    pl.pfwd * X
    for j in eachindex(pl.scales)
        band = _pyramid_band!(pl.spectra, pl.pyramid, j)
        _surf_scale_forward!(c.subbands[j], band, pl.scales[j], pl.work[j])
    end
    C = _coarse_spectrum(pl.spectra, pl.pyramid)
    pl.cbwd * C
    c.coarse .= real.(C) .* inv(real(eltype(C))(length(C)))
    return c
end

function _surfacelet_inverse!(volume::AbstractArray{<:Real, 3}, c::SurfaceletCoefficients, pl::SurfaceletPlan)
    _check_coefficients(c, pl)
    C = _coarse_spectrum(pl.spectra, pl.pyramid)
    C .= c.coarse
    pl.cfwd * C
    for j in reverse(eachindex(pl.scales))
        _surf_scale_inverse!(_pyramid_band_buffer(pl.spectra, pl.pyramid, j), c.subbands[j], pl.scales[j], pl.work[j])
        _pyramid_merge!(pl.spectra, pl.pyramid, j)
    end
    X = pl.spectra[1]
    pl.pbwd * X
    volume .= real.(X) .* inv(real(eltype(X))(length(X)))
    return volume
end

# Coefficients for `pl`, allocated like its buffers.
function _surfacelet_coefficients(pl::SurfaceletPlan{T}) where {T}
    proto = pl.spectra[1]
    coarse = similar(proto, T, size(_coarse_spectrum(pl.spectra, pl.pyramid)))
    subbands = map(pl.scales) do sc
        [similar(proto, T, _surf_subband_size(sc, K)) for K in 1:3 for _ in 1:prod(first(_surf_blocks(sc, K)))]
    end
    return SurfaceletCoefficients(pl.dims, coarse, subbands)
end

function _check_coefficients(c::SurfaceletCoefficients, pl::SurfaceletPlan)
    c.dims == pl.dims || throw(DimensionMismatch("coefficients are for $(c.dims), the plan for $(pl.dims)"))
    length(c.subbands) == length(pl.scales) ||
        throw(ArgumentError("coefficients were produced for a different number of scales"))
    expect = size(_coarse_spectrum(pl.spectra, pl.pyramid))
    size(c.coarse) == expect ||
        throw(DimensionMismatch("coarse subband has size $(size(c.coarse)), expected $(expect)"))
    for (j, sc) in enumerate(pl.scales)
        length(c.subbands[j]) == _nsubbands(sc.levels) ||
            throw(DimensionMismatch("scale $(j) has the wrong number of subbands"))
    end
    return c
end

# The plan to use: `plan` after checking it, or a fresh host plan from `make`
# (`make_surfacelet` / `make_sti_surfacelet`) when none was given.
function _resolve_plan(make::F, plan, params, x::Union{AbstractArray, SurfaceletCoefficients}) where {F}
    dims = x isa SurfaceletCoefficients ? x.dims : size(x)
    proto = x isa SurfaceletCoefficients ? x.coarse : x
    if plan === nothing
        # The plan is built on the host; a device plan is an `Adapt.adapt` of
        # it, built once rather than implicitly on every call.
        proto isa Array || throw(
            ArgumentError(
                "a device array needs an explicit plan: `plan = Adapt.adapt(ArrayType, $(nameof(make))(params, dims, T))`"
            )
        )
        T = _filter_eltype(_data_eltype(proto))
        return make(params, dims, T; flags = FFTW.ESTIMATE)
    end
    plan.params isa _plan_params_type(params) ||
        throw(ArgumentError("plan was built for $(nameof(typeof(plan.params))), not $(nameof(typeof(params)))"))
    plan.dims == dims ||
        throw(DimensionMismatch("plan was built for $(plan.dims), got $(dims)"))
    return plan
end

_plan_params_type(::SurfaceletParams) = SurfaceletParams
