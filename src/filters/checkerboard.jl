# Two-channel checkerboard filter bank of the iterated resampled checkerboard
# (IRC) stage — Lu & Do, "Multidimensional Directional Filter Banks and
# Surfacelets", IEEE TIP 16(4), 2007, Sections III-D and V-B.
#
# The filters are ladder (Phoong-type) filters built from one 1-D odd filter
# `t`, applied separably.  With
#
#     b(ω) = 2 Σ_{q odd > 0} t_q sin(q ω),        B(ω₁, ω₂) = -b(ω₁) b(ω₂),
#
# the analysis and synthesis responses are, exactly,
#
#     H₀ = (1 + B) / √2                 G₀ = (2 + B - B²) / √2
#     H₁ = e^{ jω₂} (2 - B - B²) / √2   G₁ = e^{-jω₂} (1 - B) / √2
#
# where ω₁ runs along the hourglass axis K and ω₂ along the axis m being
# decimated.  Because `t` has only odd taps, B(ω₁, ω₂ + π) = -B(ω₁, ω₂), which
# is what collapses the general two-step ladder to these forms; then
# H₀G₀ + H₁G₁ = 2 and H₀(ω + πe₂)G₀ + H₁(ω + πe₂)G₁ = 0, i.e. perfect
# reconstruction after downsampling by 2 along ω₂.
#
# The coefficients reproduce SurfBox's `cbd_filters_<bo>_{dec,rec}.mat` (the 2-D
# filters it ships) to rounding error — `test_irc.jl` checks every order
# against the golden data.  Only `t_q` for q = 1, 3, …, bo - 1 is stored; the
# filter is antisymmetric, t_{-q} = -t_q.

const _CBD_LADDER = Dict{Int, Vector{Float64}}(
    4 => [-0.6262169131293146, -0.2391935764853579],
    6 => [-0.6263429050738657, -0.18302382681237633, -0.10753046674888095],
    8 => [-0.627819441355453, -0.1868110527281001, -0.08821345831338342, -0.05299463507171143],
    10 => [
        -0.6290391996524649, -0.19023436487159193, -0.09316419099049718,
        -0.04787638089257991, -0.02729049477362672,
    ],
    12 => [
        -0.6299909650161403, -0.19294788893803155, -0.09719811683520764,
        -0.05257765509290852, -0.027224347419537415, -0.014421093056370137,
    ],
    14 => [
        -0.6307470366053054, -0.19509552613399547, -0.10042251574577052,
        -0.056418179282558024, -0.03113763249657735, -0.015819995468636647,
        -0.007750295209451851,
    ],
    16 => [
        -0.631356163548133, -0.19683521105934682, -0.10304471724680653,
        -0.05956342644490844, -0.03441276144847804, -0.018849533868796398,
        -0.009297418074817033, -0.004215553479603451,
    ],
    18 => [
        -0.6318535467876545, -0.19826135647461407, -0.10520738123671233,
        -0.06218709792109189, -0.037184557113685186, -0.021472356755448885,
        -0.011535270028700011, -0.00549634585408209, -0.0023119823375261405,
    ],
)

"""
    checkerboard_orders() -> Vector{Int}

Orders `bo` for which checkerboard filters are available (4, 6, …, 18).
"""
checkerboard_orders() = sort!(collect(keys(_CBD_LADDER)))

function _cbd_ladder(bo::Int)
    t = get(_CBD_LADDER, bo, nothing)
    t === nothing &&
        throw(ArgumentError("checkerboard filter order must be one of $(checkerboard_orders()), got $(bo)"))
    return t
end

# b(ω) = 2 Σ t_q sin(q ω) over the odd q.
@inline function _cbd_b(t::AbstractVector, ω::T) where {T}
    s = zero(T)
    @inbounds for i in eachindex(t)
        s += T(t[i]) * sin((2i - 1) * ω)
    end
    return 2s
end

# Analysis response of channel `chan` (0 or 1) at (ω₁, ω₂).
@inline function _cbd_analysis(t, chan::Int, ω1::T, ω2::T) where {T}
    B = -_cbd_b(t, ω1) * _cbd_b(t, ω2)
    r = inv(sqrt(T(2)))
    return chan == 0 ? complex(r * (1 + B)) : cis(ω2) * (r * (2 - B - B^2))
end

# Synthesis response of channel `chan` at (ω₁, ω₂).
@inline function _cbd_synthesis(t, chan::Int, ω1::T, ω2::T) where {T}
    B = -_cbd_b(t, ω1) * _cbd_b(t, ω2)
    r = inv(sqrt(T(2)))
    return chan == 0 ? complex(r * (2 + B - B^2)) : cis(-ω2) * (r * (1 - B))
end
