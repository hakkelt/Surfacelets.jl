# Frequency-domain multiscale pyramid of the decimated surfacelet transform —
# Lu & Do, "Multidimensional Directional Filter Banks and Surfacelets", IEEE TIP
# 16(4), 2007, Section VI-A (eqs. 12–13); reference implementation SurfBox
# `PyrNDDec_mm.m` / `PrySDdec_onestep.m`.
#
# One level splits a spectrum `X` of size `n` with a separable lowpass `L(ω)`
# that is compactly supported inside `|ω_d| ≤ π (w + tbw)`:
#
#     lowpass   Lp = crop_{n/D}( L · X ) / D^{N/2}
#     highpass  Hp = √(1 - L²) · X                  (kept at size n)
#
# `L` is a product of 1-D raised-cosine ramps, flat at 1 for `|ω| ≤ π(w - tbw)`.
# Because `L` vanishes before `π/D`, cropping the spectrum to `n/D` bins is an
# exact, alias-free downsampling by the (possibly non-integer) factor `D`, and
# `L² + (1 - L²) = 1` makes each level a tight frame.
#
# Modes (SurfBox's `Pyr_mode`), level 1 then every later level:
#
#   | mode | w          | tbw         | D        |
#   |------|------------|-------------|----------|
#   | 1    | 1/2, 1/4   | 1/6, 1/12   | 1, 2     |
#   | 1.5  | 1/2, 3/8   | 1/7, 1/9    | 1.5, 2   |
#   | 2    | 1/3        | 1/7         | 2        |

struct FourierPyramidLevel{T, V <: AbstractVector{T}, I <: AbstractVector{Int}}
    n::NTuple{3, Int}       # size of the spectrum entering this level
    nlow::NTuple{3, Int}    # size of the lowpass leaving it
    D::T
    idx::NTuple{3, I}       # bins of `n` inside the lowpass support
    idx_low::NTuple{3, I}   # where they land in `nlow`
    prof::NTuple{3, V}      # 1-D lowpass profile on those bins
end

function _pyramid_mode(mode::Real)
    mode == 1 && return ((1 // 2, 1 // 4), (1 // 6, 1 // 12), (1 // 1, 2 // 1))
    mode == 1.5 && return ((1 // 2, 3 // 8), (1 // 7, 1 // 9), (3 // 2, 2 // 1))
    mode == 2 && return ((1 // 3, 1 // 3), (1 // 7, 1 // 7), (2 // 1, 2 // 1))
    throw(ArgumentError("pyramid mode must be 1, 1.5 or 2, got $(mode)"))
end

_rcos(x::T) where {T} = x <= 0 ? zero(T) : (x >= 1 ? one(T) : (1 - cospi(x)) / 2)

# Band edges, in bins, of the 1-D raised-cosine lowpass with cutoff `w` and
# transition `t` (in units of π) on an `n`-point axis: flat at 1 for `|f| ≤ Q`,
# zero for `|f| ≥ P`.
_rcos_edges(n::Int, w::Rational, t::Rational) = (ceil(Int, n // 2 * (w + t)), floor(Int, n // 2 * (w - t)))

# The profile's value at bin `|f| ≤ P`: a √rcos ramp from 1 down to 0.
_rcos_profile(::Type{T}, f::Int, P::Int, Q::Int) where {T} = sqrt(_rcos(T(P - f) / T(P - Q)))

# Separable lowpass from three 1-D profiles, `L[i,j,k] = p₁[i] p₂[j] p₃[k]`.
# `_separable_factors` gives the profiles oriented along their axes, for
# broadcasting the product without materialising it.
_separable_factors(prof::NTuple{3, <:AbstractVector}) =
    (prof[1], reshape(prof[2], 1, :), reshape(prof[3], 1, 1, :))

function _separable_lowpass(prof::NTuple{3, <:AbstractVector})
    l1, l2, l3 = _separable_factors(prof)
    return l1 .* l2 .* l3
end

function _pyramid_level(::Type{T}, n::NTuple{3, Int}, w::Rational, tbw::Rational, D::Rational) where {T}
    all(d -> (d * denominator(D)) % numerator(D) == 0, n) ||
        throw(ArgumentError("a pyramid level with factor $(D) cannot divide size $(n)"))
    nlow = map(d -> Int(d * denominator(D) ÷ numerator(D)), n)
    idx = Vector{Vector{Int}}(undef, 3)
    idx_low = Vector{Vector{Int}}(undef, 3)
    prof = Vector{Vector{T}}(undef, 3)
    for a in 1:3
        nr = n[a]
        P, Q = _rcos_edges(nr, w, tbw)
        2P < nr || throw(ArgumentError("size $(nr) is too small for the pyramid"))
        v = [_rcos_profile(T, p, P, Q) for p in 0:P]
        # non-negative bins 0:P, then the mirrored negative bins -P:-1
        idx[a] = vcat(collect(0:P), collect((nr - P):(nr - 1))) .+ 1
        idx_low[a] = vcat(collect(0:P), collect((nlow[a] - P):(nlow[a] - 1))) .+ 1
        prof[a] = vcat(v, reverse(v[2:end]))
    end
    return FourierPyramidLevel(n, nlow, T(D), Tuple(idx), Tuple(idx_low), Tuple(prof))
end

"""
    _fourier_pyramid(T, dims, J, mode) -> Vector{FourierPyramidLevel}

Levels of a `J`-level pyramid on a `dims` volume.
"""
function _fourier_pyramid(::Type{T}, dims::NTuple{3, Int}, J::Int, mode::Real) where {T}
    ws, tbws, Ds = _pyramid_mode(mode)
    levels = FourierPyramidLevel{T, Vector{T}, Vector{Int}}[]
    n = dims
    for j in 1:J
        k = j == 1 ? 1 : 2
        lv = _pyramid_level(T, n, ws[k], tbws[k], Ds[k])
        push!(levels, lv)
        n = lv.nlow
    end
    return levels
end

# Split `X` (size `lv.n`) in place into its highpass and write the lowpass to
# `low` (size `lv.nlow`).  The lowpass is separable on the support box and is
# broadcast from its three 1-D profiles.
function _pyramid_split!(low::AbstractArray{Complex{T}, 3}, X::AbstractArray{Complex{T}, 3}, lv::FourierPyramidLevel{T}) where {T}
    l1, l2, l3 = _separable_factors(lv.prof)
    box = view(X, lv.idx...)
    fill!(low, zero(Complex{T}))
    view(low, lv.idx_low...) .= (l1 .* l2 .* l3 .* inv(lv.D^T(1.5))) .* box
    box .*= sqrt.(1 .- (l1 .* l2 .* l3) .^ 2)
    return X
end

# Inverse of `_pyramid_split!`: `X` holds the highpass on entry and the merged
# spectrum on exit.
function _pyramid_merge!(X::AbstractArray{Complex{T}, 3}, low::AbstractArray{Complex{T}, 3}, lv::FourierPyramidLevel{T}) where {T}
    l1, l2, l3 = _separable_factors(lv.prof)
    box = view(X, lv.idx...)
    box .= sqrt.(1 .- (l1 .* l2 .* l3) .^ 2) .* box .+ (l1 .* l2 .* l3 .* lv.D^T(1.5)) .* view(low, lv.idx_low...)
    return X
end

# The pyramid as the transform drives it (`surfacelet.jl`): the spectra it works
# in, the bandpass of scale j, and the merge back.  Spectrum j has the size
# entering level j and the last one is the coarse lowpass, so the scales hand
# each other their spectra and nothing is copied.
_pyramid_sizes(levels::AbstractVector{<:FourierPyramidLevel}) = [[lv.n for lv in levels]; [levels[end].nlow]]

function _pyramid_band!(spectra::AbstractVector, levels::AbstractVector{<:FourierPyramidLevel}, j::Int)
    return _pyramid_split!(spectra[j + 1], spectra[j], levels[j])
end

_pyramid_band_buffer(spectra::AbstractVector, ::AbstractVector{<:FourierPyramidLevel}, j::Int) = spectra[j]

function _pyramid_merge!(spectra::AbstractVector, levels::AbstractVector{<:FourierPyramidLevel}, j::Int)
    return _pyramid_merge!(spectra[j], spectra[j + 1], levels[j])
end

_coarse_spectrum(spectra::AbstractVector, ::AbstractVector{<:FourierPyramidLevel}) = spectra[end]
