# Surfacelets.jl

[![CI](https://github.com/hakkelt/Surfacelets.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/hakkelt/Surfacelets.jl/actions/workflows/CI.yml)
[![Lint](https://github.com/hakkelt/Surfacelets.jl/actions/workflows/lint.yml/badge.svg?branch=master)](https://github.com/hakkelt/Surfacelets.jl/actions/workflows/lint.yml)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://hakkelt.github.io/Surfacelets.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://hakkelt.github.io/Surfacelets.jl/dev/)
[![Coverage](https://codecov.io/gh/hakkelt/Surfacelets.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/hakkelt/Surfacelets.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Runic](https://img.shields.io/badge/code%20style-runic-blue.svg)](https://github.com/fredrikekre/Runic.jl)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)


*The surfacelet transform in Julia: a multiscale, multidirectional transform for
volumes, in a nonsubsampled (shift-invariant) and a decimated form.*

A surfacelet transform splits a volume by scale, like a 3-D wavelet, and each
scale further by direction, into wedges around the three frequency axes. Every
subband then holds surfaces of one scale and one orientation, so volumes with
smooth surfaces (organs in an MRI or CT scan, interfaces in a simulation) are
represented far more sparsely than by separable wavelets, which helps denoising
and compressed-sensing reconstruction.

## Installation

```julia
using Pkg
Pkg.add("Surfacelets")
```

## Quick start

```julia
using Surfacelets

vol    = randn(Float32, 128, 128, 128)
t      = NSSurfacelet(J = 3, L_array = [2, 2, 3])   # 48 directional subbands

coeffs = analyze(t, vol)
rec    = synthesize(t, coeffs)
subband(t, coeffs, 2)          # the first directional subband; subband 1 is the lowpass
subband_layout(t, size(vol))   # what each subband is
```

At working sizes the materialised coefficient set does not fit — 3.1 GB at 256³
`Float32` with those settings — so the nonsubsampled transform also has a
streaming form that hands you one subband at a time:

```julia
ws = make_workspace(t, vol)    # FFT plans and masks, reused by every call

analyze(t, vol; workspace = ws) do i, band
    # `band` is subband i, valid only until this callback returns
end
```

`Surfacelet` is the decimated counterpart — Lu and Do's surfacelet
transform, reproducing their reference toolbox SurfBox to rounding error. Each
subband sits on its own decimated grid, about 4 coefficients per voxel whatever
the number of directions, at the cost of shift invariance.

| | `NSSurfacelet` | `Surfacelet` |
|---|---|---|
| Subband sizes | all equal to the input | decimated, one grid per subband |
| Redundancy | `1 + Σ_j 3·2^{L_j}` (49 at `J=3`, `L=[2,2,3]`) | ≈ 4 (`pyr_mode = 1.5`), flat in the direction count |
| Shift invariance | exact | no — decimation breaks it |
| Frame | tight (Parseval) | biorthogonal, perfect reconstruction |
| Streaming form | yes, one subband at a time | not needed, the set fits |
| GPU | Adapt-based | Adapt-based |

## Not a 3-D contourlet transform

The surfacelet transform is a construction in its own right. Its first
directional stage splits the spectrum into three hourglasses around the
frequency axes — a partition with no quincunx-lattice description, which Lu and
Do prove cannot be critically sampled above two dimensions, so a redundancy of
at least 3 is structural. The nonsubsampled transform builds it from Bernstein
mapping kernels and cuts each hourglass into wedges with directly placed
frequency boundaries. The decimated transform refines each hourglass with
iterated resampled checkerboard (IRC) filter banks: ladder filters and shears,
like a contourlet DFB, but on the planes of each hourglass, and critically
sampled. The [theory page](https://hakkelt.github.io/Surfacelets.jl/stable/theory/)
explains the constructions with figures.

The 2-D contourlet and nonsubsampled contourlet transforms live in
[Contourlets.jl](https://github.com/hakkelt/Contourlets.jl), and 2-D and 3-D
shearlet transforms in [Shearlets.jl](https://github.com/hakkelt/Shearlets.jl).
The packages share their interface, defined in
[MultiscaleTransformsBase.jl](https://github.com/hakkelt/MultiscaleTransformsBase.jl)
and re-exported here, and the threading and mask-bank engine of
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl).

## Multithreading and GPU

The transforms run in parallel over Julia's threads (start Julia with
`-t auto`) unless `threaded = false`: the nonsubsampled transform computes its
subbands in parallel, the decimated transform splits each scale's blocks among
the threads.

The transforms are FFT plus a pointwise mask, and every mask is stored as
broadcastable factors, so the device path needs no kernels — only `Adapt` and a
GPU backend with an FFT library: CUDA.jl (NVIDIA), AMDGPU.jl (AMD) or Metal.jl
(Apple silicon, `Float32` only). The test suite runs on CUDA. OpenCL.jl is
supported but much slower than the CPU, because it has no FFT: FFTW runs the
FFTs on the CPU on `CLArray`s in shared virtual memory (OpenCL.jl's default on
devices that support it).

```julia
using CUDA, Adapt, Surfacelets

x  = CuArray(vol)
ws = make_workspace(t, x)      # built on the host, moved to the device
c  = analyze(t, x; workspace = ws)
```

## Development

```bash
julia --project=test test/runtests.jl               # all tests
julia --project=test test/runtests.jl :directional  # filter by tag or item name

runic src/
JULIA_LOAD_PATH="@relint:@stdlib" julia --startup-file=no scripts/relint.jl src/
```

To work against a local checkout of MultiscaleFilterBanks.jl, put it next to
this one and run `julia --project=. dev/bootstrap.jl` once; it `Pkg.develop`s
the sibling checkout into every environment of the workspace.

## Disclaimer

This package was created by [Claude](https://claude.com/claude-code), Anthropic's
AI model, under the direction of the author.

## License

MIT. The decimated surfacelet transform reimplements the construction of SurfBox
(© Yue Lu and Minh N. Do, MATLAB File Exchange 14485), whose source was consulted
to reproduce it; SurfBox's notice is retained in [`LICENSE`](LICENSE). The golden
data under `test/reference/` was produced by running SurfBox. No code is derived
from the MATLAB Contourlet or Nonsubsampled Contourlet toolboxes.
