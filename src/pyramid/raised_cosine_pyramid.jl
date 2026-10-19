# Raised-cosine nonsubsampled pyramid of the nonsubsampled surfacelet transform
# (`ns_surfacelet.jl`).
#
# Lu & Do's frequency-domain pyramid (`fourier_pyramid.jl`) without the
# resampling, with SurfBox's mode-1 band edges: level j uses the separable
# raised-cosine lowpass `ℓ_j` with cutoff `w_j = 1/2^j` and transition
# `t_j = w_j / 3` (in units of π), on the original grid.  With
#     bandpass_j = √(1 - ℓ_j²) · Π_{i<j} ℓ_i,    coarse = Π_i ℓ_i,
# `Σ_j bandpass_j² + coarse² = 1` telescopes, so the pyramid is a tight frame,
# and every factor is a pointwise frequency weight, so it commutes with circular
# shifts.

# 1-D lowpass profile on a full DFT axis: √rcos ramp from 1 (|f| ≤ Q) to 0 (|f| ≥ P).
function _rc_profile(::Type{T}, n::Int, w::Rational, t::Rational) where {T}
    P, Q = _rcos_edges(n, w, t)
    P > Q || throw(ArgumentError("axis of length $(n) is too short for $(round(Float64(w); digits = 4))π pyramid cutoff"))
    return T[
        (f = abs(k <= n ÷ 2 ? k : k - n); f <= P ? _rcos_profile(T, f, P, Q) : zero(T))
            for k in 0:(n - 1)
    ]
end

# The pyramid's per-axis factors on the DFT bins `bins` of an axis of length `n`:
# Π_{i<j} ℓᵢ for j = 1:J+1 (the lowpass products, the last one the coarse band)
# and ℓⱼ for j = 1:J.
function _rc_responses(::Type{T}, J::Int, n::Int, bins) where {T}
    prof = [_rc_profile(T, n, 1 // 2^j, 1 // (3 * 2^j))[bins] for j in 1:J]
    acc = [ones(T, length(bins))]
    for j in 1:J
        push!(acc, acc[end] .* prof[j])
    end
    return acc, prof
end
