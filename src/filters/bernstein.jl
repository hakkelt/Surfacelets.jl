# Multivariate Bernstein mapping kernels for multidimensional filter-bank design.
#
# Reference: Y. M. Lu and M. N. Do, "A Mapping-Based Design for Nonsubsampled
# Hourglass Filter Banks in Arbitrary Dimensions", IEEE Trans. Signal Processing
# 56(4):1466-1478, 2008 — Section V-C.
#
# The construction: an order-`N` Bernstein polynomial in `d` variables,
#
#     B(x) = Σ_j c_j Π_i binomial(N, j_i) x_i^{j_i} (1 - x_i)^{N - j_i},
#
# is turned into a zero-phase FIR filter of support (2N+1)^d by the change of
# variables x_i = (1 - cos ω_i) / 2.  Choosing the coefficients `c_j` as the sign
# of the desired response at the sample point `j / N` gives a maximally flat
# approximation of an ideal ±1 indicator, converging as `N` grows.
#
# Because `x_i` is monotonically increasing in `|ω_i|` on `[-π, π]`, an ordering
# condition on the frequency magnitudes (which is exactly what the hourglass
# passband supports are made of) becomes the same ordering condition on the
# integer sample indices `j_i` — see `mapping_coefficients`.
#
# We never materialise the FIR filter.  The 3-D transforms here are
# nonsubsampled with a periodic boundary, so the kernel is only ever needed as
# its frequency response sampled on the DFT grid, and that can be evaluated
# exactly and directly from the polynomial — no truncation, no convolution.

"""
    bernstein_table(N::Int, omega::AbstractVector{T}) -> Matrix{T}

Evaluate the order-`N` Bernstein basis at the frequencies `omega`, after the
change of variables `x = (1 - cos ω) / 2`.  Returns an `(N+1) × length(omega)`
matrix whose `[j+1, k]` entry is `binomial(N, j) x_k^j (1 - x_k)^(N-j)`.

The columns sum to 1, so any convex combination of rows stays in `[-1, 1]` when
the coefficients do — this is what bounds the mapped filter responses.
"""
function bernstein_table(N::Int, omega::AbstractVector{T}) where {T <: AbstractFloat}
    N >= 1 || throw(ArgumentError("Bernstein order must be ≥ 1"))
    tbl = Matrix{T}(undef, N + 1, length(omega))
    # The binomial row depends only on `N`, so it is built once rather than per
    # frequency: the table is `(N+1) × length(omega)` and `omega` is a grid axis.
    bin = _binomial_row(T, N)
    @inbounds for (k, w) in enumerate(omega)
        x = (one(T) - cos(w)) / 2
        x = clamp(x, zero(T), one(T))
        y = one(T) - x
        for j in 0:N
            tbl[j + 1, k] = bin[j + 1] * x^j * y^(N - j)
        end
    end
    return tbl
end

# Binomial coefficients of one row of Pascal's triangle, in `T`.
_binomial_row(::Type{T}, N::Int) where {T} = T[T(binomial(N, j)) for j in 0:N]

"""
    dft_frequencies(n::Int; half::Bool=false) -> Vector{Float64}

Angular frequencies of a length-`n` DFT grid, `ω_k = 2π(k-1)/n`.  With
`half = true` only the `n ÷ 2 + 1` frequencies kept by `rfft` are returned.
"""
function dft_frequencies(::Type{T}, n::Int; half::Bool = false) where {T <: AbstractFloat}
    m = half ? n ÷ 2 + 1 : n
    return T[2 * T(π) * (k - 1) / n for k in 1:m]
end
dft_frequencies(n::Int; half::Bool = false) = dft_frequencies(Float64, n; half = half)

# Folded angular frequencies of the DFT grid: ω ∈ (-π, π].  With `half = true`
# only the non-negative half kept by `rfft` is returned, already in [0, π].
function _folded_frequencies(::Type{T}, n::Int; half::Bool = false) where {T}
    half && return dft_frequencies(T, n; half = true)
    return T[
        let w = 2 * T(π) * (k - 1) / n
            w > T(π) ? w - 2 * T(π) : w
        end for k in 1:n
    ]
end

"""
    mapping_coefficients(N::Int, d::Int, group::NTuple) -> Array{Int,d}

Coefficient tensor of the Bernstein mapping kernel whose passband is

    S = {ω : max_{i ∈ group} |ω_i| ≥ max_{i ∉ group} |ω_i|},

the "simplest passband support" of Lu & Do (2008), Proposition 1 — the optimal
choice for splitting a node of the hourglass cascade tree.

Entries are `+1` inside `S`, `-1` outside, and `0` on the boundary, evaluated at
the Bernstein sample points `j / N`.  Only the ordering of the indices matters,
which is why the sample points never have to be converted back to frequencies.
"""
mapping_coefficients(N::Int, d::Int, group::NTuple{G, Int}) where {G} = mapping_coefficients(N, Val(d), group)

function mapping_coefficients(N::Int, ::Val{d}, group::NTuple{G, Int}) where {d, G}
    1 <= G < d || throw(ArgumentError("group must be a proper non-empty subset of 1:d"))
    all(g -> 1 <= g <= d, group) || throw(ArgumentError("group indices out of range"))
    c = Array{Int, d}(undef, ntuple(_ -> N + 1, d))
    @inbounds for idx in CartesianIndices(c)
        j = Tuple(idx) .- 1
        a = maximum(ntuple(t -> j[group[t]], Val(G)))
        b = maximum(j[t] for t in 1:d if !(t in group))
        c[idx] = a > b ? 1 : (a < b ? -1 : 0)
    end
    return c
end

"""
    mapping_response(coeffs, tables) -> Array

Evaluate a multivariate Bernstein mapping kernel on a DFT grid.  `coeffs` is the
`(N+1)^d` coefficient tensor from [`mapping_coefficients`](@ref) and `tables` is
a `d`-tuple of per-axis Bernstein tables from [`bernstein_table`](@ref), one per
grid axis.  The result is the real, zero-phase frequency response `M(ω)` on the
product grid, with `|M| ≤ 1` wherever the coefficients are in `[-1, 1]`.

Evaluated by successive axis contractions rather than by summing `(N+1)^d`
separable outer products, so the cost is `O(N · ∏ nᵢ)` and not `O(N^d · ∏ nᵢ)`.
"""
function mapping_response(coeffs::AbstractArray{<:Real, 1}, tables::Tuple{Matrix{T}}) where {T}
    return vec(transpose(tables[1]) * T.(coeffs))
end

function mapping_response(coeffs::AbstractArray{<:Real, 2}, tables::NTuple{2, Matrix{T}}) where {T}
    b1, b2 = tables
    # M[k1, k2] = Σ_{j1, j2} c[j1, j2] b1[j1, k1] b2[j2, k2]
    return transpose(b1) * (T.(coeffs) * b2)
end

function mapping_response(coeffs::AbstractArray{<:Real, 3}, tables::NTuple{3, Matrix{T}}) where {T}
    b1, b2, b3 = tables
    np1 = size(b1, 1)
    n1, n2, n3 = size(b1, 2), size(b2, 2), size(b3, 2)
    c = T.(coeffs)

    # Contract axis 1:  t1[(j2,j3), k1] = Σ_{j1} c[j1,j2,j3] b1[j1,k1]
    t1 = transpose(reshape(c, np1, np1 * np1)) * b1              # (np1², n1)
    # Contract axis 2:  t2[k2, (j3,k1)] = Σ_{j2} t1[j2,(j3,k1)] b2[j2,k2]
    t1r = reshape(t1, np1, np1 * n1)                             # [j2, (j3,k1)]
    t2 = transpose(b2) * t1r                                     # (n2, np1*n1)
    t2p = permutedims(reshape(t2, n2, np1, n1), (2, 3, 1))       # [j3, k1, k2]
    # Contract axis 3:  M[k3, (k1,k2)]
    t3 = transpose(b3) * reshape(t2p, np1, n1 * n2)              # (n3, n1*n2)
    return permutedims(reshape(t3, n3, n1, n2), (2, 3, 1))
end

"""
    halfband_split(x, m::Int) -> f₀

The maximally flat 1-D polynomial used to turn a mapping kernel into a
two-channel split:

    f₀(x) = ((1+x)/2)^(m+1) · Σ_{k=0}^{m} binomial(m+k, k) · ((1-x)/2)^k,
    f₁(x) = 1 - f₀(x).

`f₀(1) = 1`, `f₀(-1) = 0`, and `f₀(x) + f₀(-x) = 1`, so `f₁(x) = f₀(-x)` — the
two channels are mirror images and their sum is exactly 1.  With synthesis
polynomials `g₀ = g₁ = 1` the Bézout/perfect-reconstruction condition
`f₀g₀ + f₁g₁ = 1` therefore holds identically, and reconstruction is a plain
sum of the subbands.

`m` controls the transition sharpness: `m = 0` gives `(1+x)/2`, larger `m` is
flatter at `x = ±1` at the cost of a steeper polynomial.
"""
function halfband_split(x::T, m::Int) where {T <: Real}
    m >= 0 || throw(ArgumentError("polynomial order must be ≥ 0"))
    u = (one(T) + x) / 2
    v = (one(T) - x) / 2
    s = zero(T)
    # binomial(m+k, k) by the recurrence `c_{k+1} = c_k · (m+k+1) ÷ (k+1)`, which
    # is exact in integers: this runs once per voxel per subband, and
    # `binomial`'s overflow checking is not free at that rate.
    c = 1
    @inbounds for k in 0:m
        s += T(c) * v^k
        c = c * (m + k + 1) ÷ (k + 1)
    end
    return u^(m + 1) * s
end

# Soft step from 0 to 1 across `[-1, 1]`, flat outside it: `halfband_split` on a
# clamped argument, which is exactly 0 at -1 and exactly 1 at 1.  Consecutive
# steps therefore telescope to 1, which is what makes every partition built from
# them (hourglass, wedge, radial shell) a partition of unity.
@inline _step(u::T, poly::Int) where {T} = halfband_split(clamp(u, -one(T), one(T)), poly)

# The same step placed at `b` with half-width `w`.  A non-positive width degrades
# to a hard step rather than dividing by zero.
@inline function _soft_step(s::T, b::T, w::T, poly::Int) where {T}
    w <= 0 && return s >= b ? one(T) : zero(T)
    return _step((s - b) / w, poly)
end
