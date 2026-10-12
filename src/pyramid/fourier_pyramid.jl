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

function _pyramid_level(::Type{T}, n::NTuple{3, Int}, w::Rational, tbw::Rational, D::Rational) where {T}
    all(d -> (d * denominator(D)) % numerator(D) == 0, n) ||
        throw(ArgumentError("a pyramid level with factor $(D) cannot divide size $(n)"))
    nlow = map(d -> Int(d * denominator(D) ÷ numerator(D)), n)
    idx = Vector{Vector{Int}}(undef, 3)
    idx_low = Vector{Vector{Int}}(undef, 3)
    prof = Vector{Vector{T}}(undef, 3)
    for a in 1:3
        nr = n[a]
        P = ceil(Int, nr // 2 * (w + tbw))
        Q = floor(Int, nr // 2 * (w - tbw))
        2P < nr || throw(ArgumentError("size $(nr) is too small for the pyramid"))
        v = [sqrt(_rcos(T(P - p) / T(P - Q))) for p in 0:P]
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

# Separable lowpass on the support box, `L[i,j,k] = p₁[i] p₂[j] p₃[k]`.
_pyramid_lowpass(lv::FourierPyramidLevel) =
    lv.prof[1] .* reshape(lv.prof[2], 1, :) .* reshape(lv.prof[3], 1, 1, :)

# Split `X` (size `lv.n`, overwritten with the highpass) and return the lowpass.
function _pyramid_split!(X::AbstractArray{Complex{T}, 3}, lv::FourierPyramidLevel{T}) where {T}
    Lf = _pyramid_lowpass(lv)
    box = view(X, lv.idx...)
    low = fill!(similar(X, lv.nlow), zero(Complex{T}))
    view(low, lv.idx_low...) .= (Lf ./ lv.D^T(1.5)) .* box
    box .*= sqrt.(1 .- Lf .^ 2)
    return low
end

# Inverse of `_pyramid_split!`: `X` holds the highpass on entry and the merged
# spectrum on exit.
function _pyramid_merge!(X::AbstractArray{Complex{T}, 3}, low::AbstractArray{Complex{T}, 3}, lv::FourierPyramidLevel{T}) where {T}
    Lf = _pyramid_lowpass(lv)
    box = view(X, lv.idx...)
    box .= sqrt.(1 .- Lf .^ 2) .* box .+ (Lf .* lv.D^T(1.5)) .* view(low, lv.idx_low...)
    return X
end
