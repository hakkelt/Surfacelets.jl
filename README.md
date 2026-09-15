# Surfacelets.jl

*The surfacelet transform in Julia: a multiscale, multidirectional transform for
volumes, in a nonsubsampled (shift-invariant) and a decimated form.*

```julia
using Surfacelets

vol    = randn(Float32, 128, 128, 128)
params = NSurfaceletParams(J = 3, L_array = [2, 2, 3])   # 48 directional subbands

coeffs = nsurfacelet_forward(vol, params)
rec    = nsurfacelet_inverse(coeffs, params)
```

At working sizes the materialised coefficient set does not fit — 3.1 GB at 256³
`Float32` with those settings — so the nonsubsampled transform also has a
streaming form that hands you one subband at a time:

```julia
ws = make_nsurfacelet_workspace(params, size(vol), Float32)

coarse = nsurfacelet_forward(vol, params; workspace = ws) do j, k, band
    # `band` is valid only until this callback returns
end
```

`surfacelet_forward` is the decimated counterpart: the same frequency partition,
but each subband at its own reduced resolution — about 3× leaner, at the cost of
shift invariance.

| | `nsurfacelet_forward` | `surfacelet_forward` |
|---|---|---|
| Subband sizes | all equal to the input | decimated, one grid per subband |
| Redundancy | `1 + Σ_j 3·2^{L_j}` (49 at `J=3`, `L=[2,2,3]`) | ≈ 16 at the same settings |
| Shift invariance | exact | no — decimation breaks it |
| Streaming form | yes, one subband at a time | not needed, the set fits |
| GPU | CUDA / OpenCL verified | CUDA / OpenCL verified |

## Not a 3-D contourlet transform

The surfacelet transform is a construction in its own right. Its directional
stage is Lu and Do's hourglass nonsubsampled directional filter bank, built from
Bernstein mapping kernels; a quincunx sampling matrix, a shear index remap and a
ladder filter — the machinery a contourlet DFB is made of — appear nowhere in
it. Lu and Do prove that an hourglass frequency partition cannot be critically
sampled above two dimensions, so the redundancy here is structural rather than a
design choice.

The 2-D contourlet and nonsubsampled contourlet transforms live in
[Contourlets.jl](https://github.com/hakkelt/Contourlets.jl). The two packages
share only the pyramid filter machinery, which lives in
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl).

## GPU

Both transforms are FFT plus a pointwise mask, and every mask is stored as
broadcastable factors, so the device path needs no kernels — only `Adapt`:

```julia
using CUDA, Adapt, Surfacelets

ws = Adapt.adapt(CuArray, make_nsurfacelet_workspace(params, size(vol), Float32))
c  = nsurfacelet_forward(CuArray(vol), params; workspace = ws)
```

## Installation

Neither this package nor its core dependency is registered yet:

```julia
using Pkg
Pkg.add(url = "https://github.com/hakkelt/MultiscaleFilterBanks.jl")
Pkg.add(url = "https://github.com/hakkelt/Surfacelets.jl")
```

Working from a local checkout, `MultiscaleFilterBanks.jl` is expected as a
sibling directory and wired in through `[sources]`. On Julia 1.10, which does not
read `[sources]`, run `julia --project=. dev/bootstrap.jl` once instead; it
`Pkg.develop`s the sibling checkout into every environment of the workspace.

Before CI can go green on GitHub, this repository needs `CODECOV_TOKEN` and
`DOCUMENTER_KEY` set, and the CI setup step needs `Pkg.develop(url = ...)` for
the core package until it is registered.

## Development

```bash
julia --project=test test/runtests.jl               # all tests
julia --project=test test/runtests.jl :directional  # filter by tag or item name

runic src/
JULIA_LOAD_PATH="@relint:@stdlib" julia --startup-file=no scripts/relint.jl src/
```

## License

MIT. Reference golden data under `test/reference/` is derived from SurfBox (Lu
and Do, MATLAB File Exchange 14485) and is used for geometric comparison only.
