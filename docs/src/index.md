# Surfacelets.jl

*The surfacelet transform in Julia: a multiscale, multidirectional transform for
volumes, in a nonsubsampled (shift-invariant), a decimated and a
semi-translation-invariant form.*

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
params = NSSurfaceletParams(J = 3, L_array = [2, 2, 3])   # 48 directional subbands

coeffs = ns_surfacelet_forward(vol, params)
rec    = ns_surfacelet_inverse(coeffs, params)
```

At working sizes the materialised coefficient set does not fit — at 256³
`Float32` with these settings it is 3.1 GB — so the nonsubsampled transform also
has a streaming form that hands you one subband at a time:

```julia
ws = make_ns_surfacelet_workspace(params, size(vol), Float32)

coarse = ns_surfacelet_forward(vol, params; workspace = ws) do j, k, band
    # `band` is valid only until this callback returns — copy or reduce it here
end
```

The decimated counterpart, [`surfacelet_forward`](@ref), is Lu and Do's
surfacelet transform, reproducing their reference toolbox SurfBox to rounding
error. Every subband sits on its own decimated grid — about 4 coefficients per
voxel, whatever the number of directions — at the cost of shift invariance.
Sizes must suit its 3/2 pyramid — divisible by 3, then by 2 per further scale:

```julia
vol    = randn(Float32, 192, 192, 192)
sp     = SurfaceletParams(J = 3, L_array = [2, 2, 3])
plan   = make_surfacelet(sp, size(vol), Float32)
sc     = surfacelet_forward(vol, sp; plan = plan)
vol_r  = surfacelet_inverse(sc, sp; plan = plan)
```

[`sti_surfacelet_forward`](@ref) is a semi-translation-invariant (STI) middle
ground: a nonsubsampled raised-cosine pyramid followed by the decimated
transform's critically sampled directional stage, redundancy `1 + 3J`. It
shares the *structure* of Eslami and Radha's semi-translation-invariant
contourlet transform, not its construction — it is not a 3-D STICT; see
[Theory](@ref theory).

## What is here

| | [`ns_surfacelet_forward`](@ref) | [`sti_surfacelet_forward`](@ref) | [`surfacelet_forward`](@ref) |
|---|---|---|---|
| Subband sizes | all equal to the input | coarse full size, directional subbands decimated | decimated, one grid per subband |
| Redundancy | `1 + Σ_j 3·2^{L_j}` (49 at `J=3`, `L=[2,2,3]`) | `1 + 3J` (10 at `J=3`) | ≈ 4 (`pyr_mode = 1.5`), flat in the direction count |
| Shift invariance | exact | scales exact; subbands on their decimation lattice | no — decimation breaks it |
| Frame | tight | perfect reconstruction | biorthogonal, perfect reconstruction |
| Streaming form | yes, one subband at a time | not needed | not needed, the set fits |
| GPU | CUDA / OpenCL verified | Adapt-based, as the decimated one | Adapt-based |
| Use for | analysis, shift-invariant denoising | middle ground: scale invariance at modest redundancy | sparsifying transform in reconstruction |

All three split each scale into `3 · 2^L` directional subbands: three hourglass-shaped
regions aligned with the frequency axes, each cut into `2^L` wedges. The
directional stage is available on its own as [`make_hourglass`](@ref) and
[`make_ndfb`](@ref).

## Relation to the contourlet transform

The surfacelet transform is **not** a contourlet transform with an extra axis.
Its first directional stage splits the spectrum into three hourglasses around
the frequency axes — a partition with no quincunx-lattice description, which Lu
and Do prove cannot be critically sampled above two dimensions, so a redundancy
of at least 3 is structural. The nonsubsampled transform builds it from
Bernstein mapping kernels; the decimated one then refines each hourglass with
iterated resampled checkerboard filter banks — ladder filters and shears, as in
a contourlet DFB, but on the planes of each hourglass and critically sampled.
See [Theory](@ref theory).

The 2-D contourlet and nonsubsampled contourlet transforms live in
[Contourlets.jl](https://github.com/hakkelt/Contourlets.jl); the two packages
share only the pyramid filter machinery, which lives in
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl).

## GPU

Both transforms are FFT plus a pointwise mask, and every mask is stored as
broadcastable factors, so the device path needs no kernels — only `Adapt`:

```julia
using CUDA, Adapt, Surfacelets

ws  = Adapt.adapt(CuArray, make_ns_surfacelet_workspace(params, size(vol), Float32))
c   = ns_surfacelet_forward(CuArray(vol), params; workspace = ws)
```
