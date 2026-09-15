# Surfacelets.jl

*The surfacelet transform in Julia: a multiscale, multidirectional transform for
volumes, in a nonsubsampled (shift-invariant) and a decimated form.*

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/hakkelt/MultiscaleFilterBanks.jl")
Pkg.add(url = "https://github.com/hakkelt/Surfacelets.jl")
```

## Quick start

```julia
using Surfacelets

vol    = randn(Float32, 128, 128, 128)
params = NSurfaceletParams(J = 3, L_array = [2, 2, 3])   # 48 directional subbands

coeffs = nsurfacelet_forward(vol, params)
rec    = nsurfacelet_inverse(coeffs, params)
```

At working sizes the materialised coefficient set does not fit — at 256³
`Float32` with these settings it is 3.1 GB — so the nonsubsampled transform also
has a streaming form that hands you one subband at a time:

```julia
ws = make_nsurfacelet_workspace(params, size(vol), Float32)

coarse = nsurfacelet_forward(vol, params; workspace = ws) do j, k, band
    # `band` is valid only until this callback returns — copy or reduce it here
end
```

The decimated counterpart, [`surfacelet_forward`](@ref), stores every subband at
its own reduced resolution — about 3× leaner at useful settings, at the cost of
shift invariance:

```julia
sp     = SurfaceletParams(J = 3, L_array = [2, 2, 3])
plan   = make_surfacelet(sp, size(vol), Float32)
sc     = surfacelet_forward(vol, sp; plan = plan)
vol_r  = surfacelet_inverse(sc, sp; plan = plan)
```

## What is here

| | [`nsurfacelet_forward`](@ref) | [`surfacelet_forward`](@ref) |
|---|---|---|
| Subband sizes | all equal to the input | decimated, one grid per subband |
| Redundancy | `1 + Σ_j 3·2^{L_j}` (49 at `J=3`, `L=[2,2,3]`) | ≈ 16 at the same settings |
| Shift invariance | exact | no — decimation breaks it |
| Streaming form | yes, one subband at a time | not needed, the set fits |
| GPU | CUDA / OpenCL verified | CUDA / OpenCL verified |
| Use for | analysis, shift-invariant denoising | sparsifying transform in reconstruction |

Both split each scale into `3 · 2^L` directional subbands: three hourglass-shaped
regions aligned with the frequency axes, each cut into `2^L` wedges. The
directional stage is available on its own as [`make_hourglass`](@ref) and
[`make_ndfb`](@ref).

## Relation to the contourlet transform

The surfacelet transform is **not** a contourlet transform with an extra axis.
Its directional stage is Lu and Do's hourglass nonsubsampled directional filter
bank, built from Bernstein mapping kernels; a quincunx sampling matrix, a shear
index remap and a ladder filter — the machinery the contourlet DFB is made of —
appear nowhere in it. Lu and Do prove that an hourglass frequency partition
cannot be critically sampled above two dimensions, so the redundancy here is
structural rather than a design choice. See [Theory](@ref theory).

The 2-D contourlet and nonsubsampled contourlet transforms live in
[Contourlets.jl](https://github.com/hakkelt/Contourlets.jl); the two packages
share only the pyramid filter machinery, which lives in
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl).

## GPU

Both transforms are FFT plus a pointwise mask, and every mask is stored as
broadcastable factors, so the device path needs no kernels — only `Adapt`:

```julia
using CUDA, Adapt, Surfacelets

ws  = Adapt.adapt(CuArray, make_nsurfacelet_workspace(params, size(vol), Float32))
c   = nsurfacelet_forward(CuArray(vol), params; workspace = ws)
```
