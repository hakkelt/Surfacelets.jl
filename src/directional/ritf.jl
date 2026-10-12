# Undecimated hourglass filter bank of the decimated surfacelet transform:
# SurfBox's "ritf" (rotation-invariant tight frame) construction — Lu & Do,
# "Multidimensional Directional Filter Banks and Surfacelets", IEEE TIP 16(4),
# 2007, Section V-A.
#
# This is a different realisation of the hourglass partition from the Bernstein
# mapping kernels of `hourglass.jl`, which the nonsubsampled transform uses.  It
# is kept separate so the decimated transform reproduces the reference
# implementation coefficient for coefficient.
#
# A 2-D diamond kernel `m(k₁, k₂) = w(k₁ + k₂) w(k₁ - k₂)` is built from a
# Kaiser-windowed half-band sinc `w` of length `msize`.  Its zero-phase response
# `D(ω_a, ω_b)`, raised to the even power `λ` and shifted by π along one axis,
# is a two-sided "bow tie" around that axis; the product over the other planes
# gives the hourglass weight
#
#     hg_K(ω) = Π_{m ≠ K} D(ω_K + π, ω_m)^λ,
#
# and `V_K = √(N hg_K / Σ_k hg_k)` satisfies `Σ_K V_K² = N` exactly.  Synthesis
# divides by `N`, so the stage is a tight frame of redundancy `N` (here 3).

# Modified Bessel function I₀ by the same truncated series SurfBox uses
# (`calculateI0.m`), so the window matches it to the last few ulps.
function _ritf_I0(x::T) where {T <: AbstractFloat}
    x /= 2
    y = one(T)
    z = one(T)
    t = one(T)
    k = 1
    while t > T(1.0e-10) * y
        z = z * x / k
        k += 1
        t = z * z
        y += t
    end
    return y
end

# Kaiser window times the half-band sinc, `w(u)` for u = -h:h with h = (n-1)/2.
function _ritf_window(::Type{T}, n::Int, beta::Real) where {T}
    isodd(n) || throw(ArgumentError("msize must be odd"))
    b = T(beta)
    den = _ritf_I0(b)
    half = (n - 1) ÷ 2
    val = [T(2) * i / (n - 1) - 1 for i in 0:half]
    wh = [_ritf_I0(b * sqrt(1 - v * v)) / den for v in val]
    w = vcat(wh, reverse(wh[1:(end - 1)]))
    for (i, u) in enumerate((-half):half)
        w[i] *= sinc(T(u) / 2)
    end
    return w
end

# Zero-phase response of the diamond kernel.  With u = k₁ + k₂ and v = k₁ - k₂
# (which must share parity) and `w` even, the double sum factorises into an
# even-tap and an odd-tap part:
#     D(a, b) = Cₑ(α) Cₑ(β) + Cₒ(α) Cₒ(β),   α = (a + b)/2,  β = (a - b)/2.
function _ritf_diamond(w::AbstractVector{T}, a::T, b::T) where {T}
    half = (length(w) - 1) ÷ 2
    α = (a + b) / 2
    β = (a - b) / 2
    ceα = ceβ = coα = coβ = zero(T)
    @inbounds for (i, u) in enumerate((-half):half)
        if iseven(u)
            ceα += w[i] * cos(u * α)
            ceβ += w[i] * cos(u * β)
        else
            coα += w[i] * cos(u * α)
            coβ += w[i] * cos(u * β)
        end
    end
    return ceα * ceβ + coα * coβ
end

# `D(ω_K + π, ω_m)^λ` on the DFT grid of an `n_K × n_m` plane.
function _ritf_plane(w::AbstractVector{T}, nK::Int, nm::Int, lambda::Int) where {T}
    P = Matrix{T}(undef, nK, nm)
    for l in 1:nm, k in 1:nK
        a = T(2π) * (k - 1) / nK + T(π)
        b = T(2π) * (l - 1) / nm
        P[k, l] = _ritf_diamond(w, a, b)^lambda
    end
    return P
end

# The three half-spectrum hourglass masks `V_K` on a `dims` grid.  Mask K keeps
# only the non-negative half of axis K (`n_K ÷ 2 + 1` bins), as the IRC stage
# that follows works on that half.
function _ritf_masks(::Type{T}, dims::NTuple{3, Int}; msize::Int = 15, beta::Real = 2.5, lambda::Int = 4) where {T}
    iseven(lambda) && lambda > 0 || throw(ArgumentError("lambda must be a positive even integer"))
    all(iseven, dims) || throw(ArgumentError("the hourglass stage needs even dimensions, got $(dims)"))
    w = _ritf_window(T, msize, beta)
    hg = ntuple(3) do K
        acc = ones(T, dims)
        for m in 1:3
            m == K && continue
            P = _ritf_plane(w, dims[K], dims[m], lambda)
            sz = ntuple(d -> d == K ? dims[K] : (d == m ? dims[m] : 1), 3)
            acc .*= reshape(K < m ? P : permutedims(P), sz)
        end
        acc
    end
    total = hg[1] .+ hg[2] .+ hg[3]
    return ntuple(3) do K
        half = ntuple(d -> d == K ? (1:(dims[K] ÷ 2 + 1)) : (1:dims[d]), 3)
        sqrt.(3 .* hg[K][half...] ./ total[half...])
    end
end
