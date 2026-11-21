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
# perfect reconstruction but not a tight one: `synthesize` is the inverse,
# not the adjoint, of `analyze`.

"""
    SurfaceletWorkspace

Everything a [`Surfacelet`](@ref) transform needs for one volume size, element
type and array type: the pyramid profiles, hourglass masks and checkerboard
filter slices, and the spectra, block buffers and FFT plans the transform runs
in, so that `analyze!` and `synthesize!` allocate nothing.  Build with
[`make_workspace`](@ref make_workspace(::Surfacelet, ::AbstractArray)) and pass it
as `workspace` (or in place of the transform).

A workspace holds its own scratch buffers, so one workspace must not run two
transforms at the same time; give each task its own.  A transform with the
workspace spreads its work over at most the `tasks` it was built for.
`Adapt.adapt(ArrayType, ws)` moves it to a device, rebuilding the buffers and
FFT plans there.
"""
struct SurfaceletWorkspace{Td <: Number, T, L <: FourierPyramidLevel{T}, C <: _SurfScale{T}, W <: _ScaleWork, A <: AbstractArray{Complex{T}, 3}, F, G, R} <: AbstractTransformWorkspace
    dims::NTuple{3, Int}
    transform::Surfacelet
    tasks::Int              # as requested; a device runs one
    pyramid::Vector{L}
    scales::Vector{C}
    work::Vector{W}         # scales with equal tables share one
    spectra::Vector{A}      # [j] the spectrum entering level j; [end] the coarse lowpass
    pfwd::F                 # in-place FFT and unnormalized inverse on spectra[1] …
    pbwd::G
    cfwd::F                 # … and on the coarse spectrum; each a pair (threaded, serial)
    cbwd::G
    layout::SubbandLayout{3}    # of the coefficients
    offsets::Vector{Int}    # [j] subbands before the first of pyramid level j (1 the finest)
    rcoef::R                # complex data: real coefficients of one channel; `nothing` otherwise
end

MultiscaleTransformsBase.transform(ws::SurfaceletWorkspace) = ws.transform
Base.eltype(::Type{<:SurfaceletWorkspace{Td}}) where {Td} = Td
Base.size(ws::SurfaceletWorkspace) = ws.dims

# The workspace around `pyramid` and `scales`, with buffers like `proto` (host or
# device) for `tasks` tasks.  Host FFT plans use `flags`; the block plans run on
# one FFTW thread each, the full-volume plans on `tasks` or, in a serial call, one.
function _surfacelet_workspace(
        ::Type{Td}, dims::NTuple{3, Int}, t::Surfacelet, pyramid::AbstractVector, scales::AbstractVector{<:_SurfScale{T}},
        proto::AbstractArray, flags, tasks::Integer
    ) where {Td, T}
    tasks >= 1 || throw(ArgumentError("tasks must be ≥ 1, got $(tasks)"))
    layout = SubbandLayout(_surf_layout_sizes(t, dims), dims)
    # The coefficients hold the scales coarse to fine, so the coarser levels come first.
    counts = [_nsubbands(sc.levels) for sc in scales]
    offsets = [1 + sum(counts[(j + 1):end]; init = 0) for j in eachindex(counts)]
    # The block FFTs read and write the subbands in place, at offsets in the
    # coefficients' one vector that need not keep its memory alignment.
    aligned = all(r -> ((first(r) - 1) * sizeof(T)) % 64 == 0, layout.ranges)
    bflags = aligned ? flags : flags | FFTW.UNALIGNED
    work = _fftw_single_threaded(proto) do
        build(sc) = _scale_work(sc, proto, bflags, tasks)
        first_work = build(first(scales))
        works = IdDict{Any, typeof(first_work)}(first(scales).stages => first_work)
        [get!(() -> build(sc), works, sc.stages) for sc in scales]
    end
    Tc = Complex{T}
    # The scales hand each other their spectra, so nothing is copied.
    sizes = [[lv.n for lv in pyramid]; [last(pyramid).nlow]]
    spectra = [fill!(similar(proto, Tc, n), zero(Tc)) for n in sizes]
    pfwd, pbwd = _inplace_plans(spectra[1], flags, tasks)
    cfwd, cbwd = _inplace_plans(spectra[end], flags, tasks)
    rcoef = Td <: Complex ? SubbandVector{T}(undef, layout, proto) : nothing
    return SurfaceletWorkspace{
        Td, T, eltype(pyramid), eltype(scales), eltype(work), eltype(spectra), typeof(pfwd), typeof(pbwd), typeof(rcoef),
    }(dims, t, Int(tasks), collect(pyramid), collect(scales), work, spectra, pfwd, pbwd, cfwd, cbwd, layout, offsets, rcoef)
end

# In-place FFT and unnormalized inverse of `A`, each as a pair: on `tasks` FFTW
# threads for a threaded call, on one for a serial call (`threaded = false`, or
# inside a caller's NestedThreading restriction).  A plan keeps the thread count
# it was made with, so a serial call needs a plan of its own.  One plan serves
# both on a device or with one task.
function _inplace_plans(A::AbstractArray, flags, tasks::Integer)
    make() = (_fft_plan(plan_fft!, A, flags), _fft_plan(plan_bfft!, A, flags))
    if !(A isa Array) || tasks == 1
        fwd, bwd = A isa Array ? with_fftw_threads(make, 1) : make()
        return (fwd, fwd), (bwd, bwd)
    end
    fwd, bwd = with_fftw_threads(make, tasks)
    fwd1, bwd1 = with_fftw_threads(make, 1)
    return (fwd, fwd1), (bwd, bwd1)
end

_pick_plan(p::Tuple, threaded::Bool) = threaded ? p[1] : p[2]

# Host plans applied to the blocks of a scale use one FFTW thread: the blocks
# themselves are split among tasks (`_foreach_block`).
_fftw_single_threaded(f::F, proto::AbstractArray) where {F} = proto isa Array ? with_fftw_threads(f, 1) : f()

function MultiscaleTransformsBase.make_workspace(
        t::Surfacelet, ::Type{T0}, dims::Dims{3};
        tasks::Integer = Threads.nthreads(), flags::Integer = FFTW.MEASURE
    ) where {T0 <: Number}
    Td = data_eltype(T0)
    T = filter_eltype(Td)
    pyramid = _fourier_pyramid(T, dims, t.scales, t.pyramid_ratio)
    # Pyramid level j (1 the finest) is scale J - j + 1.
    scales = map(1:t.scales) do j
        n = pyramid[j].n
        lv = t.levels[t.scales - j + 1]
        masks = _ritf_masks(T, n; msize = t.diamond_length, beta = T(t.kaiser_beta), lambda = t.diamond_power)
        _SurfScale(n, lv, masks, _surf_stages(T, n, lv, t.checkerboard_order))
    end
    return _surfacelet_workspace(Td, dims, t, pyramid, scales, Array{T, 3}(undef, 0, 0, 0), flags, tasks)
end

MultiscaleTransformsBase.make_workspace(::Surfacelet, ::Type{<:Number}, dims::Dims) =
    throw(ArgumentError("the surfacelet transforms take volumes (three axes), got size $(dims)"))

"""
    make_workspace(t::Surfacelet, T::Type{<:Number}, dims;
                   tasks = Threads.nthreads(), flags = FFTW.MEASURE) -> SurfaceletWorkspace
    make_workspace(t::Surfacelet, x::AbstractArray; kwargs...) -> SurfaceletWorkspace

Precompute everything `t` needs for volumes of size `dims` (three axes) and
element type `T` (real or complex; an integer type means `float(T)`).  The
second form takes size, element type and array type from `x`: for a device
array the tables are built on the host (the filter slices are evaluated point by
point in closed form) and the workspace is moved with `Adapt`.

The pyramid must be able to resample every scale exactly, so with
`pyramid_ratio = 1.5` each dimension must be divisible by `3 · 2^J`, with `2`
by `2^(J+1)`, and with `1` by `2^J`, for `J = t.scales`; each scale must also allow its directional
levels (axis length divisible by `2^level`).  `tasks` bounds the threads a
transform with the workspace uses: the blocks of a scale are split among that
many tasks, each with a buffer of one block, and the full-volume FFTs run on
that many FFTW threads (on one in a serial call).  `flags` are the FFTW planner
flags; pass `FFTW.ESTIMATE` for a workspace used once.
"""
MultiscaleTransformsBase.make_workspace(t::Surfacelet, x::AbstractArray; kwargs...) =
    _workspace_like(x, make_workspace(t, eltype(x), size(x); kwargs...))

"""
    surfacelet_redundancy(t::Surfacelet, dims) -> Float64

Coefficients of `t` on a volume of size `dims`, divided by the number of voxels.
"""
function surfacelet_redundancy(t::Surfacelet, dims::Dims)
    return sum(prod, _surf_layout_sizes(t, _check_volume(dims))) / prod(dims)
end

# The sizes of the subbands of `t` on a `dims` volume, in `subband_layout`
# order: the lowpass, then for every scale, coarse to fine, the wedges of hourglass 1, 2, 3, each
# in SurfBox's block order.  Throws as `make_workspace` does for sizes the
# pyramid or the directional levels cannot divide.
function _surf_layout_sizes(t::Surfacelet, dims::NTuple{3, Int})
    pyramid = _fourier_pyramid(Float64, dims, t.scales, t.pyramid_ratio)
    sizes = [last(pyramid).nlow]
    for j in 1:t.scales
        n = pyramid[t.scales - j + 1].n
        for K in 1:3
            counts = ntuple(d -> 1 << t.levels[j][K][d], 3)
            all(d -> n[d] % counts[d] == 0, 1:3) ||
                throw(ArgumentError("scale $(j) of size $(n) cannot be split into $(counts) wedge blocks in hourglass $(K)"))
            sb = ntuple(d -> d == K ? n[K] : n[d] ÷ counts[d], 3)
            append!(sizes, Iterators.repeated(sb, prod(counts)))
        end
    end
    return sizes
end

"""
    subband_layout(t::Surfacelet, dims) -> Vector{@NamedTuple{scale::Int, hourglass::Int, wedge::Int}}

What each subband of `t` is: the lowpass first, as
`(scale = 0, hourglass = 0, wedge = 0)`, then for scale 1 (the coarsest), 2, …,
`t.scales` (the finest) the wedges of hourglass 1, 2, 3 in turn, `(scale = j, hourglass = K, wedge = w)`
with the wedges in SurfBox's order.  Hourglass `K` keeps axis `K` at the scale's
full length and decimates the other two.
"""
function MultiscaleTransformsBase.subband_layout(t::Surfacelet, dims::Dims)
    _surf_layout_sizes(t, _check_volume(dims))
    layout = [(scale = 0, hourglass = 0, wedge = 0)]
    for j in 1:t.scales, K in 1:3, w in 1:(1 << sum(t.levels[j][K]))
        push!(layout, (scale = j, hourglass = K, wedge = w))
    end
    return layout
end

MultiscaleTransformsBase.nsubbands(t::Surfacelet, dims::Dims) = (_surf_layout_sizes(t, _check_volume(dims)); 1 + sum(j -> _nsubbands(t.levels[j]), 1:t.scales))

# ── Driver ────────────────────────────────────────────────────────────────────

# `threaded = false` runs serially; so does a call inside a caller's NestedThreading
# restriction (`loop_threaded`).  Strided's kernels thread on their own, by a
# process-wide count that a restriction lowers to one, so a serial call opens one.
_s_threaded(threaded::Union{Bool, Nothing}) = threaded !== false

function _with_threading(f::F, threaded::Bool) where {F}
    threaded = loop_threaded(threaded)
    threaded && return f(true)
    return NestedThreading.with_threads_disabled(() -> f(false); only = :strided)
end

# The workspace of a call: `workspace` after checking it, or a new one, planned
# with `FFTW.ESTIMATE` (used once) for data like `proto`.
function _s_workspace_for(workspace, t::Surfacelet, ::Type{Td}, dims::Dims, proto::AbstractArray) where {Td}
    if workspace === nothing
        return _workspace_like(proto, make_workspace(t, Td, dims; flags = FFTW.ESTIMATE))
    end
    workspace isa SurfaceletWorkspace ||
        throw(ArgumentError("a Surfacelet needs a SurfaceletWorkspace, got $(typeof(workspace))"))
    return check_workspace(workspace, t, Td, dims)
end

# Real data: one pass.  Complex data: its real and imaginary parts through the
# identical real pass, one after the other, which keeps the transform exactly
# real-linear.
function _surfacelet_forward!(c, volume::AbstractArray, ws::SurfaceletWorkspace{<:Real}, threaded::Bool)
    size(volume) == ws.dims || throw(DimensionMismatch("volume has size $(size(volume)), expected $(ws.dims)"))
    return _with_threading(t -> _surfacelet_forward_body!(c, volume, ws, t), threaded)
end

function _surfacelet_forward!(c, volume::AbstractArray, ws::SurfaceletWorkspace{<:Complex}, threaded::Bool)
    size(volume) == ws.dims || throw(DimensionMismatch("volume has size $(size(volume)), expected $(ws.dims)"))
    r = ws.rcoef
    _with_threading(t -> _surfacelet_forward_body!(r, Broadcast.broadcasted(real, volume), ws, t), threaded)
    _s_set_channel!(c, r, 1)
    _with_threading(t -> _surfacelet_forward_body!(r, Broadcast.broadcasted(imag, volume), ws, t), threaded)
    _s_set_channel!(c, r, 2)
    return c
end

function _surfacelet_inverse!(volume::AbstractArray, c, ws::SurfaceletWorkspace{<:Real}, threaded::Bool)
    _with_threading(t -> _surfacelet_inverse_body!((X, s) -> (volume .= real.(X) .* s), c, ws, t), threaded)
    return volume
end

function _surfacelet_inverse!(volume::AbstractArray, c, ws::SurfaceletWorkspace{<:Complex}, threaded::Bool)
    r = ws.rcoef
    _s_get_channel!(r, c, 1)
    _with_threading(t -> _surfacelet_inverse_body!((X, s) -> (volume .= complex.(real.(X) .* s, zero(s))), r, ws, t), threaded)
    _s_get_channel!(r, c, 2)
    _with_threading(t -> _surfacelet_inverse_body!((X, s) -> (volume .= complex.(real.(volume), real.(X) .* s)), r, ws, t), threaded)
    return volume
end

# `src` is the volume, or a lazy broadcast of one of its real channels.
function _surfacelet_forward_body!(c, src, ws::SurfaceletWorkspace, threaded::Bool)
    X = ws.spectra[1]
    X .= src
    _pick_plan(ws.pfwd, threaded) * X
    for j in eachindex(ws.scales)
        band = _pyramid_split!(ws.spectra[j + 1], ws.spectra[j], ws.pyramid[j])
        _surf_scale_forward!(_ScaleBands(c, ws.offsets[j]), band, ws.scales[j], ws.work[j], threaded)
    end
    C = ws.spectra[end]
    _pick_plan(ws.cbwd, threaded) * C
    subband(c, 1) .= real.(C) .* inv(real(eltype(C))(length(C)))
    return c
end

# `store!(X, s)` writes the volume from its unnormalized spectrum `X` and `s = 1/N`.
function _surfacelet_inverse_body!(store!::S, c, ws::SurfaceletWorkspace, threaded::Bool) where {S}
    C = ws.spectra[end]
    C .= subband(c, 1)
    _pick_plan(ws.cfwd, threaded) * C
    for j in reverse(eachindex(ws.scales))
        _surf_scale_inverse!(ws.spectra[j], _ScaleBands(c, ws.offsets[j]), ws.scales[j], ws.work[j], threaded)
        _pyramid_merge!(ws.spectra[j], ws.spectra[j + 1], ws.pyramid[j])
    end
    X = ws.spectra[1]
    _pick_plan(ws.pbwd, threaded) * X
    store!(X, inv(real(eltype(X))(length(X))))
    return nothing
end

# The subbands of one scale, indexed from 1: subband `offset + n` of `c`.
struct _ScaleBands{C <: SubbandVector}
    c::C
    offset::Int
end

Base.getindex(b::_ScaleBands, n::Int) = subband(b.c, b.offset + n)

# Channel `ch` (1 real, 2 imaginary) of the complex coefficients `c` = the real
# coefficients `r`; channel 1 also clears channel 2.
function _s_set_channel!(c::SubbandVector, r::SubbandVector, ch::Int)
    a, b = parent(c), parent(r)
    ch == 1 ? (a .= complex.(b, zero(eltype(b)))) : (a .= complex.(real.(a), b))
    return c
end

# r = channel `ch` of the complex coefficients `c`.
function _s_get_channel!(r::SubbandVector, c::SubbandVector, ch::Int)
    a, b = parent(c), parent(r)
    ch == 1 ? (b .= real.(a)) : (b .= imag.(a))
    return r
end

function _check_coefficients(c::SubbandVector, ws::SurfaceletWorkspace)
    c.layout == ws.layout ||
        throw(DimensionMismatch("the coefficients' subband layout is not that of $(ws.transform) on size $(ws.dims)"))
    eltype(c) === eltype(ws) ||
        throw(ArgumentError("coefficients of element type $(eltype(c)), the workspace is for $(eltype(ws))"))
    return c
end

# ── Interface ─────────────────────────────────────────────────────────────────

"""
    similar_coefficients(t::Surfacelet, x::AbstractArray) -> SubbandVector

Uninitialized coefficients of `t` for volumes like `x`: a `SubbandVector`
(MultiscaleTransformsBase) of element type `float(eltype(x))` stored in an array like `x` (host or device),
with the subbands of [`subband_layout`](@ref subband_layout(::Surfacelet, ::Dims)),
the lowpass first.
"""
function MultiscaleTransformsBase.similar_coefficients(t::Surfacelet, x::AbstractArray)
    dims = _check_volume(size(x))
    return SubbandVector{data_eltype(x)}(undef, SubbandLayout(_surf_layout_sizes(t, dims), dims), x)
end

MultiscaleTransformsBase.subband(::Surfacelet, c::SubbandVector, i::Integer) = subband(c, i)

function MultiscaleTransformsBase.analyze!(
        c::SubbandVector, t::Surfacelet, x::AbstractArray;
        workspace = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    ws = _s_workspace_for(workspace, t, data_eltype(x), size(x), x)
    _check_coefficients(c, ws)
    return _surfacelet_forward!(c, x, ws, _s_threaded(threaded))
end

function MultiscaleTransformsBase.synthesize!(
        x::AbstractArray, t::Surfacelet, c::SubbandVector;
        workspace = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    ws = _s_workspace_for(workspace, t, data_eltype(x), size(x), x)
    _check_coefficients(c, ws)
    return _surfacelet_inverse!(x, c, ws, _s_threaded(threaded))
end
