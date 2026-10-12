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

# 1-D lowpass profile on a full DFT axis: √rcos ramp from 1 (|f| ≤ Q) to 0 (|f| ≥ P).
function _sti_profile(::Type{T}, n::Int, w::Rational, t::Rational) where {T}
    P = ceil(Int, n // 2 * (w + t))
    Q = floor(Int, n // 2 * (w - t))
    P > Q || throw(ArgumentError("axis of length $(n) is too short for $(round(Float64(w); digits = 4))π pyramid cutoff"))
    return T[
        (f = k <= n ÷ 2 ? k : k - n; abs(f) <= P ? sqrt(_rcos(T(P - abs(f)) / T(P - Q))) : zero(T))
            for k in 0:(n - 1)
    ]
end

_sti_lowpass(prof::NTuple{3, <:AbstractVector}) =
    prof[1] .* reshape(prof[2], 1, :) .* reshape(prof[3], 1, 1, :)
