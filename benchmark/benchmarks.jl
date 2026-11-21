"""
    benchmarks.jl — Surfacelets.jl BenchmarkTools SUITE

USAGE:
    julia --project=benchmark benchmark/benchmarks.jl

Covers the nonsubsampled and decimated surfacelet transforms and the
nonsubsampled directional filter bank.
"""

using BenchmarkTools, Surfacelets, Random
using Surfacelets: make_ndfb, ndfb_decompose, make_hourglass, hourglass_decompose!
using FFTW: FFTW

# Cap each @benchmarkable's tuning + steady-state phase: AirspeedVelocity runs this
# suite once per compared revision in CI, and revision-to-revision comparisons don't
# need BenchmarkTools' full-precision default budget.
BenchmarkTools.DEFAULT_PARAMETERS.samples = 3
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

const RNG = MersenneTwister(1234)

const SUITE = BenchmarkGroup()
SUITE["NSSurfacelet"] = BenchmarkGroup()
SUITE["Surfacelet"] = BenchmarkGroup()
SUITE["NDFB"] = BenchmarkGroup()

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ SUITE                                                                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# Volume sizes are 32³ and 64³, not the 256³ working point: this suite runs once
# per compared revision in CI, and a 256³ forward pass alone is seconds.
# Plans are built with FFTW.ESTIMATE so planning does not dominate the setup.
#
# Float32 only — the transforms treat a complex volume as two real passes through
# the same kernel, so there is no separate threading regime to measure.

for n in (32, 64)
    sz_str = "$(n)^3"
    vol = randn(RNG, Float32, n, n, n)

    # ── NSSurfacelet ────────────────────────────────────────────────────────────────
    SUITE["NSSurfacelet"][sz_str] = BenchmarkGroup()
    t = NSSurfacelet(direction_levels = [1, 2], bernstein_order = 12)
    ws = make_workspace(t, Float32, (n, n, n); flags = FFTW.ESTIMATE)
    SUITE["NSSurfacelet"][sz_str]["forward"] = @benchmarkable analyze($t, $vol)
    SUITE["NSSurfacelet"][sz_str]["forward (workspace)"] =
        @benchmarkable analyze($t, $vol; workspace = $ws)
    # The streaming form is the one that matters at working sizes: it holds one
    # subband instead of `1 + Σ 3·2^L` volumes.
    SUITE["NSSurfacelet"][sz_str]["forward (streaming)"] =
        @benchmarkable analyze((i, b) -> nothing, $t, $vol; workspace = $ws)
    coeffs = analyze(t, vol; workspace = ws)
    out = similar(vol)
    SUITE["NSSurfacelet"][sz_str]["inverse"] = @benchmarkable synthesize($t, $coeffs)
    SUITE["NSSurfacelet"][sz_str]["inverse!"] =
        @benchmarkable synthesize!($out, $t, $coeffs; workspace = $ws)

    # ── Decimated surfacelet (IRC) ───────────────────────────────────────────────
    # pyramid_ratio = 2 so the power-of-two sizes divide; mode 1.5 needs a factor 3.
    SUITE["Surfacelet"][sz_str] = BenchmarkGroup()
    st = Surfacelet(direction_levels = [1, 2], pyramid_ratio = 2)
    sws = make_workspace(st, Float32, (n, n, n))
    SUITE["Surfacelet"][sz_str]["forward"] = @benchmarkable analyze($st, $vol; workspace = $sws)
    scoeffs = analyze(st, vol; workspace = sws)
    SUITE["Surfacelet"][sz_str]["inverse"] = @benchmarkable synthesize($st, $scoeffs; workspace = $sws)

    # ── NDFB ───────────────────────────────────────────────────────────────
    SUITE["NDFB"][sz_str] = BenchmarkGroup()
    for L in (1, 3)
        fb = make_ndfb((n, n, n), Float32; levels = L, bernstein_order = 12, flags = FFTW.ESTIMATE)
        SUITE["NDFB"][sz_str]["L=$(L) forward (streaming)"] =
            @benchmarkable ndfb_decompose((k, b) -> nothing, $vol, $fb)
    end
    hg = make_hourglass((n, n, n), Float32; bernstein_order = 12, flags = FFTW.ESTIMATE)
    dsts = ntuple(_ -> similar(vol), 3)
    SUITE["NDFB"][sz_str]["hourglass forward"] =
        @benchmarkable hourglass_decompose!($dsts, $vol, $hg)
end
