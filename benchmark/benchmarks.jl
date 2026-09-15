"""
    benchmarks.jl — Surfacelets.jl BenchmarkTools SUITE

USAGE:
    julia --project=benchmark benchmark/benchmarks.jl

Covers the nonsubsampled surfacelet transform, the directional filter bank it is
built on, and the primitives underneath both.
"""

using BenchmarkTools, Surfacelets, MultiscaleFilterBanks, Random
using FFTW: FFTW

# Cap each @benchmarkable's tuning + steady-state phase: AirspeedVelocity runs this
# suite once per compared revision in CI, and revision-to-revision comparisons don't
# need BenchmarkTools' full-precision default budget.
BenchmarkTools.DEFAULT_PARAMETERS.samples = 3
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

const RNG = MersenneTwister(1234)

const SUITE = BenchmarkGroup()
SUITE["NSurfacelet"] = BenchmarkGroup()
SUITE["NDFB"] = BenchmarkGroup()
SUITE["primitives"] = BenchmarkGroup()

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

    # ── NSurfacelet ────────────────────────────────────────────────────────────────
    SUITE["NSurfacelet"][sz_str] = BenchmarkGroup()
    params = NSurfaceletParams(J = 2, L_array = [1, 2], order = 12)
    ws = make_nsurfacelet_workspace(params, (n, n, n), Float32; flags = FFTW.ESTIMATE)
    SUITE["NSurfacelet"][sz_str]["forward"] = @benchmarkable nsurfacelet_forward($vol, $params)
    SUITE["NSurfacelet"][sz_str]["forward (workspace)"] =
        @benchmarkable nsurfacelet_forward($vol, $params; workspace = $ws)
    # The streaming form is the one that matters at working sizes: it holds one
    # subband instead of `1 + Σ 3·2^L` volumes.
    SUITE["NSurfacelet"][sz_str]["forward (streaming)"] =
        @benchmarkable nsurfacelet_forward((j, k, b) -> nothing, $vol, $params; workspace = $ws)
    coeffs = nsurfacelet_forward(vol, params; workspace = ws)
    out = similar(vol)
    SUITE["NSurfacelet"][sz_str]["inverse"] = @benchmarkable nsurfacelet_inverse($coeffs, $params)
    SUITE["NSurfacelet"][sz_str]["inverse!"] =
        @benchmarkable nsurfacelet_inverse!($out, $coeffs, $params; workspace = $ws)

    # ── NDFB ───────────────────────────────────────────────────────────────
    SUITE["NDFB"][sz_str] = BenchmarkGroup()
    for L in (1, 3)
        fb = make_ndfb((n, n, n), Float32; levels = L, order = 12, flags = FFTW.ESTIMATE)
        SUITE["NDFB"][sz_str]["L=$(L) forward (streaming)"] =
            @benchmarkable ndfb_decompose((k, b) -> nothing, $vol, $fb)
    end
    hg = make_hourglass((n, n, n), Float32; order = 12, flags = FFTW.ESTIMATE)
    dsts = ntuple(_ -> similar(vol), 3)
    SUITE["NDFB"][sz_str]["hourglass forward"] =
        @benchmarkable hourglass_decompose!($dsts, $vol, $hg)

    # ── 3-D primitives ───────────────────────────────────────────────────────
    SUITE["primitives"][sz_str] = BenchmarkGroup()
    h_short = Float32[0.25, 0.5, 0.25]
    h_long = upsample_filter(Float32.(CDF97.h), 4)      # 33 taps: the FFT regime
    dst = similar(vol)
    tmp = similar(vol)
    fftplan = make_conv3d_fft_plan((n, n, n), Float32; flags = FFTW.ESTIMATE)
    SUITE["primitives"][sz_str]["conv3d_sep direct (3 taps)"] =
        @benchmarkable conv3d_sep!($dst, $vol, $h_short, $h_short, $h_short; tmp = $tmp, backend = :direct)
    SUITE["primitives"][sz_str]["conv3d_sep direct (33 taps)"] =
        @benchmarkable conv3d_sep!(
        $dst, $vol, $h_long, $h_long, $h_long;
        boundary = :periodic, tmp = $tmp, backend = :direct
    )
    SUITE["primitives"][sz_str]["conv3d_sep fft (33 taps)"] =
        @benchmarkable conv3d_sep_fft!($dst, $vol, $h_long, $h_long, $h_long, $fftplan)

    coarse = similar(vol)
    bparams = similar(vol)
    tmp2 = similar(vol)
    SUITE["primitives"][sz_str]["nsp3d_decompose!"] = @benchmarkable nsp3d_decompose!(
        $coarse, $bparams, $vol, CDF97, 3; tmp = $tmp, tmp2 = $tmp2, plan = $fftplan
    )
end
