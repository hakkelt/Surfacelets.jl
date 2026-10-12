# Nonsubsampled raised-cosine pyramid of the semi-translation-invariant (STI)
# surfacelet transform (`sti_surfacelet.jl`).
#
# SurfBox's mode-1 band edges without the resampling: level j uses the
# separable raised-cosine lowpass `ℓ_j` with cutoff `w_j = 1/2^j` and transition
# `t_j = w_j / 3` (in units of π), on the original grid.  With
#     bandpass_j = √(1 - ℓ_j²) · Π_{i<j} ℓ_i,    coarse = Π_i ℓ_i,
# `Σ_j bandpass_j² + coarse² = 1` telescopes, so the pyramid is a tight frame,
# and every factor is a pointwise frequency weight, so it commutes with circular
# shifts.

# One level: `ℓ_j` as three full-length 1-D profiles, broadcast and never stored
# as a volume.
struct RaisedCosineLevel{T, V <: AbstractVector{T}}
    prof::NTuple{3, V}
end

# 1-D lowpass profile on a full DFT axis: √rcos ramp from 1 (|f| ≤ Q) to 0 (|f| ≥ P).
function _sti_profile(::Type{T}, n::Int, w::Rational, t::Rational) where {T}
    P, Q = _rcos_edges(n, w, t)
    P > Q || throw(ArgumentError("axis of length $(n) is too short for $(round(Float64(w); digits = 4))π pyramid cutoff"))
    return T[
        (f = abs(k <= n ÷ 2 ? k : k - n); f <= P ? _rcos_profile(T, f, P, Q) : zero(T))
            for k in 0:(n - 1)
    ]
end

_raised_cosine_pyramid(::Type{T}, dims::NTuple{3, Int}, J::Int) where {T} =
    [RaisedCosineLevel(ntuple(d -> _sti_profile(T, dims[d], 1 // 2^j, 1 // (3 * 2^j)), 3)) for j in 1:J]

# The pyramid as the transform drives it (`surfacelet.jl`): spectrum 1 is the
# running lowpass (and finally the coarse subband), spectrum 2 the bandpass.
_pyramid_sizes(levels::AbstractVector{<:RaisedCosineLevel}) = (n = map(length, levels[1].prof); [n, n])

function _pyramid_band!(spectra::AbstractVector, levels::AbstractVector{<:RaisedCosineLevel}, j::Int)
    cur, band = spectra
    l1, l2, l3 = _separable_factors(levels[j].prof)
    @. band = cur * sqrt(1 - (l1 * l2 * l3)^2)
    @. cur *= l1 * l2 * l3
    return band
end

_pyramid_band_buffer(spectra::AbstractVector, ::AbstractVector{<:RaisedCosineLevel}, ::Int) = spectra[2]

function _pyramid_merge!(spectra::AbstractVector, levels::AbstractVector{<:RaisedCosineLevel}, j::Int)
    cur, band = spectra
    l1, l2, l3 = _separable_factors(levels[j].prof)
    @. cur = cur * (l1 * l2 * l3) + band * sqrt(1 - (l1 * l2 * l3)^2)
    return cur
end

_coarse_spectrum(spectra::AbstractVector, ::AbstractVector{<:RaisedCosineLevel}) = spectra[1]
