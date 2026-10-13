# Surfacelets.jl

*The surfacelet transform in Julia: a multiscale, multidirectional transform for
volumes, in a nonsubsampled (shift-invariant), a decimated and a
semi-translation-invariant form.*

```julia
using Surfacelets

vol    = randn(Float32, 128, 128, 128)
params = NSSurfaceletParams(J = 3, L_array = [2, 2, 3])   # 48 directional subbands

coeffs = ns_surfacelet_forward(vol, params)
rec    = ns_surfacelet_inverse(coeffs, params)
```

At working sizes the materialised coefficient set does not fit — 3.1 GB at 256³
`Float32` with those settings — so the nonsubsampled transform also has a
streaming form that hands you one subband at a time:

```julia
ws = make_ns_surfacelet_workspace(params, size(vol), Float32)

coarse = ns_surfacelet_forward(vol, params; workspace = ws) do j, k, band
    # `band` is valid only until this callback returns
end
```

`surfacelet_forward` is the decimated counterpart — Lu and Do's surfacelet
transform, reproducing their reference toolbox SurfBox to rounding error. Each
subband sits on its own decimated grid, about 4 coefficients per voxel whatever
the number of directions, at the cost of shift invariance.

`sti_surfacelet_forward` is a semi-translation-invariant (STI) middle ground: a
nonsubsampled raised-cosine pyramid followed by the decimated transform's
critically sampled directional stage, redundancy `1 + 3J`. It shares the
*structure* of Eslami and Radha's semi-translation-invariant contourlet
transform, not its construction — it is not a 3-D STICT, and no paper we know
of describes it; its two stages are validated separately.

| | `ns_surfacelet_forward` | `sti_surfacelet_forward` | `surfacelet_forward` |
|---|---|---|---|
| Subband sizes | all equal to the input | coarse full size, directional subbands decimated | decimated, one grid per subband |
| Redundancy | `1 + Σ_j 3·2^{L_j}` (49 at `J=3`, `L=[2,2,3]`) | `1 + 3J` (10 at `J=3`) | ≈ 4 (`pyr_mode = 1.5`), flat in the direction count |
| Shift invariance | exact | scales exact; subbands on their decimation lattice | no — decimation breaks it |
| Frame | tight | perfect reconstruction | biorthogonal, perfect reconstruction |
| Streaming form | yes, one subband at a time | not needed | not needed, the set fits |
| GPU | CUDA / OpenCL verified | Adapt-based | Adapt-based |

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
sampled.

The 2-D contourlet and nonsubsampled contourlet transforms live in
[Contourlets.jl](https://github.com/hakkelt/Contourlets.jl). The two packages
share only the pyramid filter machinery, which lives in
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl).

## GPU

Both transforms are FFT plus a pointwise mask, and every mask is stored as
broadcastable factors, so the device path needs no kernels — only `Adapt`:

```julia
using CUDA, Adapt, Surfacelets

ws = Adapt.adapt(CuArray, make_ns_surfacelet_workspace(params, size(vol), Float32))
c  = ns_surfacelet_forward(CuArray(vol), params; workspace = ws)
```

## Installation

This package is not registered yet:

```julia
using Pkg
Pkg.add(url = "https://github.com/hakkelt/Surfacelets.jl")
```

To work against a local checkout of the core package instead, put it next to
this one and run `julia --project=. dev/bootstrap.jl` once; it
`Pkg.develop`s the sibling checkout into every environment of the workspace.

Before CI can go green on GitHub, this repository needs `CODECOV_TOKEN` and
`DOCUMENTER_KEY` set.

## Development

```bash
julia --project=test test/runtests.jl               # all tests
julia --project=test test/runtests.jl :directional  # filter by tag or item name

runic src/
JULIA_LOAD_PATH="@relint:@stdlib" julia --startup-file=no scripts/relint.jl src/
```

## License

MIT. The decimated surfacelet transform reimplements the construction of SurfBox
(© Yue Lu and Minh N. Do, MATLAB File Exchange 14485), whose source was consulted
to reproduce it; SurfBox's notice is retained in [`LICENSE`](LICENSE). The golden
data under `test/reference/` was produced by running SurfBox. No code is derived
from the MATLAB Contourlet or Nonsubsampled Contourlet toolboxes.
