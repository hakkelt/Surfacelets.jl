# 3-D Nonsubsampled Pyramid (NSP3D) — à trous upsampled filters, no downsampling.
#
# Direct extension of the 2-D `nsp_decompose!` / `nsp_reconstruct!` pair: the same
# à trous filter construction, the same periodic boundary extension, one extra
# separable axis.  All outputs keep the spatial size of the input volume, so the
# stage is exactly equivariant under circular shifts.
#
# Synthesis-filter normalisation: `_NSP_SYNTH_SCALE` is applied *per axis*, so the
# 2-D constant carries over unchanged.  The biorthogonal synthesis `g` has DC gain
# 2 per axis, giving 2³ = 8 across a separable 3-D filtering; the 0.5 per-axis
# rescale brings the prediction back to unit DC gain.  Without it the "bandpass"
# would be dominated by low-pass leakage instead of being a true bandpass.
#
# Backend: every convolution here is periodic, so `conv3d_sep!` auto-selects the
# FFT backend once the à trous filters get long (level ≥ 2 single-threaded, all
# levels with FFTW threading).  A single `Conv3DFFTPlan` is built per call and
# shared by the analysis and synthesis convolutions; pass `plan = …` to reuse one
# across levels and iterations.

"""
    nsp3d_decompose(volume, fp::FilterPair, level::Int) -> (coarse, bandpass)

One level of the 3-D Nonsubsampled Pyramid at pyramid level `level` (1 = finest).
Both outputs have the same size as `volume`.

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(1), 16, 16, 16);

julia> c, bp = nsp3d_decompose(x, CDF97, 1);

julia> size(c) == size(bp) == size(x)
true
```
"""
function nsp3d_decompose(volume::AbstractArray{<:Number, 3}, fp::FilterPair, level::Int)
    level >= 1 || throw(ArgumentError("level must be ≥ 1"))
    Td = _data_eltype(volume)   # data type (preserves complex)
    Tf = _filter_eltype(Td)     # real filter precision
    vol = Td === eltype(volume) ? volume : Td.(volume)
    h_j, g_j = _nsp_atrous_filters(fp, level, Tf)
    plan = _nsp3d_plan(vol, max(length(h_j), length(g_j)))
    coarse = conv3d_sep(vol, h_j, h_j, h_j; boundary = :periodic, plan = plan)
    pred = conv3d_sep(coarse, g_j, g_j, g_j; boundary = :periodic, plan = plan)
    bp = vol .- pred
    return coarse, bp
end

"""
    nsp3d_decompose!(coarse, bandpass, volume, fp::FilterPair, level;
                    tmp=similar(volume), tmp2=similar(volume)) -> (coarse, bandpass)

In-place NSP3D decomposition.  `tmp` and `tmp2` must be distinct buffers of the
same size as `volume`; when both are supplied — together with cached `h_j`/`g_j`
à trous filters — the call is allocation-free.
"""
function nsp3d_decompose!(
        coarse::AbstractArray{<:Number, 3}, bandpass::AbstractArray{<:Number, 3},
        volume::AbstractArray{<:Number, 3}, fp::FilterPair, level::Int;
        tmp::AbstractArray = similar(volume),
        tmp2::AbstractArray = similar(volume),
        h_j::Union{AbstractVector, Nothing} = nothing,
        g_j::Union{AbstractVector, Nothing} = nothing,
        plan::Union{Conv3DFFTPlan, Nothing} = nothing
    )
    # Use caller-supplied (workspace-cached) à trous filters when given; otherwise
    # build them here.  The cached `g_j` already carries the synthesis rescale.
    if h_j === nothing || g_j === nothing
        Tf = _filter_eltype(eltype(coarse))
        h_built, g_built = _nsp_atrous_filters(fp, level, Tf)
        h_j = h_j === nothing ? h_built : h_j
        g_j = g_j === nothing ? g_built : g_j
    end
    p = plan === nothing ? _nsp3d_plan(volume, max(length(h_j), length(g_j))) : plan
    conv3d_sep!(coarse, volume, h_j, h_j, h_j; boundary = :periodic, tmp = tmp2, plan = p)
    conv3d_sep!(tmp, coarse, g_j, g_j, g_j; boundary = :periodic, tmp = tmp2, plan = p)
    @. bandpass = volume - tmp
    return coarse, bandpass
end

"""
    nsp3d_reconstruct(coarse, bandpass, fp::FilterPair, level) -> volume

Reconstruct one NSP3D level.

# Examples
```jldoctest
julia> using Surfacelets, Random

julia> x = randn(Xoshiro(1), 16, 16, 16);

julia> c, bp = nsp3d_decompose(x, CDF97, 1);

julia> rec = nsp3d_reconstruct(c, bp, CDF97, 1);

julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function nsp3d_reconstruct(
        coarse::AbstractArray{<:Number, 3}, bandpass::AbstractArray{<:Number, 3},
        fp::FilterPair, level::Int
    )
    Td = promote_type(_data_eltype(coarse), _data_eltype(bandpass))
    Tf = _filter_eltype(Td)
    coarse_T = Td === eltype(coarse) ? coarse : Td.(coarse)
    bandpass_T = Td === eltype(bandpass) ? bandpass : Td.(bandpass)
    _, g_j = _nsp_atrous_filters(fp, level, Tf)
    pred = conv3d_sep(coarse_T, g_j, g_j, g_j; boundary = :periodic, plan = _nsp3d_plan(coarse_T, length(g_j)))
    return bandpass_T .+ pred
end

"""
    nsp3d_reconstruct!(volume, coarse, bandpass, fp::FilterPair, level;
                      tmp=similar(volume), tmp2=similar(volume)) -> volume

In-place NSP3D reconstruction.  See [`nsp3d_decompose!`](@ref) for the buffer
requirements.
"""
function nsp3d_reconstruct!(
        volume::AbstractArray{<:Number, 3}, coarse::AbstractArray{<:Number, 3},
        bandpass::AbstractArray{<:Number, 3}, fp::FilterPair, level::Int;
        tmp::AbstractArray = similar(volume),
        tmp2::AbstractArray = similar(volume),
        g_j::Union{AbstractVector, Nothing} = nothing,
        plan::Union{Conv3DFFTPlan, Nothing} = nothing
    )
    if g_j === nothing
        Tf = _filter_eltype(eltype(volume))
        _, g_j = _nsp_atrous_filters(fp, level, Tf)
    end
    p = plan === nothing ? _nsp3d_plan(coarse, length(g_j)) : plan
    conv3d_sep!(tmp, coarse, g_j, g_j, g_j; boundary = :periodic, tmp = tmp2, plan = p)
    @. volume = bandpass + tmp
    return volume
end

# The à trous filter pair is `_nsp_atrous_filters` from the 2-D pyramid: the
# rescale is per axis, so the same pair serves all three.
#
# Build one FFT plan shared by the analysis and synthesis convolutions of a level,
# or `nothing` when the auto rule says the direct kernel is cheaper.  Returning
# `nothing` leaves `conv3d_sep!` on its own (direct) path.
function _nsp3d_plan(vol::AbstractArray{Td, 3}, taps::Int) where {Td}
    _conv3d_use_fft(Td, :periodic, size(vol), taps) || return nothing
    return make_conv3d_fft_plan(size(vol), Td)
end
