# Nonsubsampled surfacelet transform.
#
# Multiscale stage: `nsp3d_decompose` (à trous, no downsampling).
# Directional stage: `ndfb_decompose` (hourglass + wedge refinement).
#
# Every coefficient array keeps the spatial size of the input, so the transform
# is exactly equivariant under circular shifts of the volume, which is why both
# stages filter periodically.
#
# The directional bank depends only on the volume size, never on the scale, so
# the (large) hourglass mapping response is planned once and shared across every
# scale; only the wedge masks — matrices — differ per scale.

"""
    NSurfaceletWorkspace{Td, T}

Reusable plans and buffers for the nonsubsampled surfacelet transform at one
volume size and element type.
Build with [`make_nsurfacelet_workspace`](@ref) and pass it to `nsurfacelet_forward` /
`nsurfacelet_inverse` to keep repeated calls out of the allocator — the pattern
iterative reconstruction algorithms need.
"""
struct NSurfaceletWorkspace{
        Td <: Number, T <: AbstractFloat, B <: NDFB{T}, P,
        A3 <: AbstractArray{Td, 3},
    }
    dims::NTuple{3, Int}
    banks::Vector{B}                 # one per scale, sharing one hourglass bank
    conv::P                          # Conv3DFFTPlan for the pyramid stage
    lp_filters::FilterPair{T}        # the pair `filters` was built from
    # à trous filter pair per scale, `g_j` already rescaled.  Cached because
    # rebuilding them per scale per call allocates inside the workspace path,
    # which is the one thing that path promises not to do.
    filters::Vector{Tuple{Vector{T}, Vector{T}}}
    input::A3                        # running input of the pyramid cascade
    coarse::A3
    bandpass::A3
    tmp::A3
    tmp2::A3
end

Base.eltype(::NSurfaceletWorkspace{Td}) where {Td} = Td

"""
    make_nsurfacelet_workspace(params::NSurfaceletParams, dims::NTuple{3,Int},
                         ::Type{Td}=Float64; threading=Auto(), flags=FFTW.MEASURE)

Allocate the workspace for `params` at volume size `dims` and element type `Td`
(real or complex).

The hourglass mapping response dominates the memory: one `real(Td)` array on the
rfft grid, `(n₁÷2+1) × n₂ × n₃`.  Everything per-scale is matrices.
"""
function make_nsurfacelet_workspace(
        params::NSurfaceletParams, dims::NTuple{3, Int}, ::Type{Td} = Float64;
        threading::ThreadingPolicy = Auto(),
        flags::Integer = FFTW.MEASURE
    ) where {Td <: Number}
    T = _filter_eltype(Td)
    p = _convert_params(T, params)
    fb = make_hourglass(
        dims, T; order = params.order, poly = params.poly, tight = params.tight,
        threading = threading, flags = flags
    )
    banks = [
        ndfb_sharing(fb, params.L_array[j]; poly = params.poly, transition = params.transition)
            for j in 1:params.J
    ]
    conv = make_conv3d_fft_plan(dims, Td; threading = threading, flags = flags)
    filters = [_nsp_atrous_filters(p.lp_filters, j, T) for j in 1:params.J]
    bufs = ntuple(_ -> Array{Td, 3}(undef, dims), 5)
    return NSurfaceletWorkspace{Td, T, eltype(banks), typeof(conv), typeof(bufs[1])}(
        dims, banks, conv, p.lp_filters, filters,
        bufs[1], bufs[2], bufs[3], bufs[4], bufs[5]
    )
end

make_nsurfacelet_workspace(params::NSurfaceletParams, dims::Vararg{Int, 3}; kwargs...) =
    make_nsurfacelet_workspace(params, dims; kwargs...)

# Equality of two filter pairs at the workspace's precision — the workspace holds
# `params`' filters already converted, so the comparison converts rather than
# promoting (and stays allocation-free).
function _same_filters(a::FilterPair{T}, b::FilterPair) where {T}
    length(a.h) == length(b.h) && length(a.g) == length(b.g) || return false
    all(i -> a.h[i] == T(b.h[i]), eachindex(a.h)) || return false
    return all(i -> a.g[i] == T(b.g[i]), eachindex(a.g))
end

# Build a workspace on demand when the caller did not supply one.
function _nsurfacelet_workspace(
        ws::Union{NSurfaceletWorkspace, Nothing}, params::NSurfaceletParams,
        dims::NTuple{3, Int}, ::Type{Td}
    ) where {Td}
    ws === nothing && return make_nsurfacelet_workspace(params, dims, Td)
    ws.dims == dims || throw(DimensionMismatch("workspace size does not match the volume"))
    eltype(ws) === Td || throw(ArgumentError("workspace element type does not match the volume"))
    length(ws.banks) == params.J ||
        throw(ArgumentError("workspace was built for a different number of scales"))
    # `tight` is baked into the banks at build time and read back off them by
    # every mask, so a workspace built the other way silently normalises the
    # coefficients differently from what `params` asks for.
    ws.banks[1].fb.tight == params.tight || throw(
        ArgumentError(
            "workspace was built with tight = $(ws.banks[1].fb.tight), not $(params.tight)"
        )
    )
    # The à trous filters are cached rather than rebuilt per call, so a workspace
    # built from a different pyramid filter pair would silently filter with the
    # wrong one.
    _same_filters(ws.lp_filters, params.lp_filters) ||
        throw(ArgumentError("workspace was built for a different pyramid filter pair"))
    # The per-scale directional depth has to match too: the banks decide how many
    # subbands are produced while `params` decides how many the caller allocates,
    # so a mismatch would either overrun the coefficient vector or leave part of
    # it undefined.
    for j in 1:params.J
        sum(ws.banks[j].levels) == params.L_array[j] || throw(
            ArgumentError(
                "workspace scale $(j) was built for L = $(sum(ws.banks[j].levels)), not $(params.L_array[j])"
            )
        )
    end
    return ws
end

"""
    nsurfacelet_forward(volume, params::NSurfaceletParams; workspace=nothing) -> NSurfaceletCoefficients

Nonsubsampled surfacelet transform.  Invariant under circular shifts of the
input; every subband keeps the size of `volume`.

See [`nsurfacelet_forward(f, volume, params)`](@ref) for the streaming form, which
does not hold all subbands at once.

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(5), 16, 16, 16);

julia> p = NSurfaceletParams(J = 2, L_array = [1, 2], order = 6);

julia> c = nsurfacelet_forward(x, p);

julia> length(c.subbands[1]), length(c.subbands[2])
(6, 12)

julia> maximum(abs, nsurfacelet_inverse(c, p) .- x) < 1e-10
true
```
"""
function nsurfacelet_forward(
        volume::AbstractArray{<:Number, 3}, params::NSurfaceletParams;
        workspace::Union{NSurfaceletWorkspace, Nothing} = nothing
    )
    # Pin the container element type to the *data* type the transform works in,
    # not `eltype(volume)`: an integer volume is filtered as `float(eltype)`, and
    # storing those coefficients back into an integer array would throw.
    A = typeof(similar(volume, _data_eltype(volume)))
    subbands = [Vector{A}(undef, nsurfacelet_nsubbands(params, j)) for j in 1:params.J]
    coarse = nsurfacelet_forward(volume, params; workspace = workspace) do j, k, band
        subbands[j][k] = copy(band)
    end
    return NSurfaceletCoefficients(copy(coarse), subbands)
end

"""
    nsurfacelet_forward(f, volume, params::NSurfaceletParams; workspace=nothing)

Streaming nonsubsampled surfacelet transform: calls `f(j, k, subband)` for directional subband `k` of
scale `j`, reusing one buffer between calls.  `subband` is only valid until `f`
returns — copy or reduce it there.

This is the form to use at working sizes.  At 256³ `Float32` with `J = 3` and
`L = [2, 2, 3]` the full coefficient set is 3.1 GB, while streaming holds two
volumes plus the plans.

Returns the coarse residual.

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(6), 16, 16, 16);

julia> p = NSurfaceletParams(J = 2, L_array = [1, 1], order = 6);

julia> total = 0.0;

julia> coarse = nsurfacelet_forward(x, p) do j, k, band
           global total += sum(abs2, band)
       end;

julia> total > 0 && size(coarse) == size(x)
true
```
"""
function nsurfacelet_forward(
        f, volume::AbstractArray{<:Number, 3}, params::NSurfaceletParams;
        workspace::Union{NSurfaceletWorkspace, Nothing} = nothing
    )
    Td = _data_eltype(volume)
    dims = size(volume)
    ws = _nsurfacelet_workspace(workspace, params, dims, Td)
    p = _convert_params(_filter_eltype(Td), params)

    copyto!(ws.input, volume)
    for j in 1:params.J
        h_j, g_j = ws.filters[j]
        nsp3d_decompose!(
            ws.coarse, ws.bandpass, ws.input, p.lp_filters, j;
            tmp = ws.tmp, tmp2 = ws.tmp2, h_j = h_j, g_j = g_j, plan = ws.conv
        )
        ndfb_decompose(ws.bandpass, ws.banks[j]) do k, band
            f(j, k, band)
        end
        copyto!(ws.input, ws.coarse)
    end
    return ws.coarse
end

"""
    nsurfacelet_inverse(coeffs::NSurfaceletCoefficients, params::NSurfaceletParams; workspace=nothing) -> volume

Invert [`nsurfacelet_forward`](@ref).
"""
function nsurfacelet_inverse(
        coeffs::NSurfaceletCoefficients, params::NSurfaceletParams;
        workspace::Union{NSurfaceletWorkspace, Nothing} = nothing
    )
    out = similar(coeffs.coarse)
    return nsurfacelet_inverse!(out, coeffs, params; workspace = workspace)
end

"""
    nsurfacelet_inverse!(volume, coeffs, params::NSurfaceletParams; workspace=nothing) -> volume

In-place [`nsurfacelet_inverse`](@ref).
"""
function nsurfacelet_inverse!(
        volume::AbstractArray{<:Number, 3}, coeffs::NSurfaceletCoefficients,
        params::NSurfaceletParams; workspace::Union{NSurfaceletWorkspace, Nothing} = nothing
    )
    return nsurfacelet_inverse!(volume, coeffs.coarse, params; workspace = workspace) do j, k
        return coeffs.subbands[j][k]
    end
end

"""
    nsurfacelet_inverse(f, coarse, params::NSurfaceletParams; workspace=nothing) -> volume

Streaming inverse: `f(j, k)` must return directional subband `k` of scale `j`.
The subbands are consumed one at a time, in reverse scale order, so a caller
holding them on disk or regenerating them never materialises the full set.
"""
function nsurfacelet_inverse(
        f, coarse::AbstractArray{<:Number, 3}, params::NSurfaceletParams;
        workspace::Union{NSurfaceletWorkspace, Nothing} = nothing
    )
    out = similar(coarse)
    return nsurfacelet_inverse!(f, out, coarse, params; workspace = workspace)
end

"""
    nsurfacelet_inverse!(f, volume, coarse, params::NSurfaceletParams; workspace=nothing) -> volume

In-place streaming inverse; see [`nsurfacelet_inverse(f, coarse, params)`](@ref).
"""
function nsurfacelet_inverse!(
        f, volume::AbstractArray{<:Number, 3}, coarse::AbstractArray{<:Number, 3},
        params::NSurfaceletParams; workspace::Union{NSurfaceletWorkspace, Nothing} = nothing
    )
    Td = _data_eltype(volume)
    dims = size(volume)
    size(coarse) == dims || throw(DimensionMismatch("coarse size does not match the output"))
    ws = _nsurfacelet_workspace(workspace, params, dims, Td)
    p = _convert_params(_filter_eltype(Td), params)

    copyto!(ws.coarse, coarse)
    for j in params.J:-1:1
        # Invert the directional stage back into the scale-j bandpass.  A bare
        # sum when the bank is not tight; a masked accumulation when it is.
        ndfb_reconstruct!(ws.bandpass, ws.banks[j]) do k
            band = f(j, k)
            size(band) == dims || throw(DimensionMismatch("subband ($(j), $(k)) has the wrong size"))
            band
        end
        g_j = ws.filters[j][2]
        nsp3d_reconstruct!(
            ws.input, ws.coarse, ws.bandpass, p.lp_filters, j;
            tmp = ws.tmp, tmp2 = ws.tmp2, g_j = g_j, plan = ws.conv
        )
        copyto!(ws.coarse, ws.input)
    end
    copyto!(volume, ws.coarse)
    return volume
end
