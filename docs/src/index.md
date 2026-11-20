# Surfacelets.jl

*The surfacelet transform in Julia: a multiscale, multidirectional transform for
volumes, in a nonsubsampled (shift-invariant) and a decimated form.*

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

At working sizes the materialised coefficient set does not fit — at 256³
`Float32` with these settings it is 3.1 GB — so the nonsubsampled transform also
has a streaming form that hands you one subband at a time:

```julia
ws = make_workspace(t, vol)    # FFT plans and masks, reused by every call

analyze(t, vol; workspace = ws) do i, band
    # `band` is subband i; it is valid only until this callback returns —
    # copy or reduce it here
end
```

The decimated counterpart, [`Surfacelet`](@ref), is Lu and Do's
surfacelet transform, reproducing their reference toolbox SurfBox to rounding
error. Every subband sits on its own decimated grid — about 4 coefficients per
voxel, whatever the number of directions — at the cost of shift invariance.
Sizes must suit its 3/2 pyramid — divisible by 3, then by 2 per further scale:

```julia
vol    = randn(Float32, 192, 192, 192)
st     = Surfacelet(J = 3, L_array = [2, 2, 3])
sws    = make_workspace(st, vol)
sc     = analyze(st, vol; workspace = sws)
vol_r  = synthesize(st, sc; workspace = sws)
```

## What is here

| | [`NSSurfacelet`](@ref) | [`Surfacelet`](@ref) |
|---|---|---|
| Subband sizes | all equal to the input | decimated, one grid per subband |
| Redundancy | `1 + Σ_j 3·2^{L_j}` (49 at `J=3`, `L=[2,2,3]`) | ≈ 4 (`pyr_mode = 1.5`), flat in the direction count |
| Shift invariance | exact | no — decimation breaks it |
| Frame | tight (Parseval) | biorthogonal, perfect reconstruction |
| Streaming form | yes, one subband at a time | not needed, the set fits |
| GPU | Adapt-based | Adapt-based |
| Use for | analysis, shift-invariant denoising | sparsifying transform in reconstruction |

Both split each scale into `3 · 2^L` directional subbands: three hourglass-shaped
regions aligned with the frequency axes, each cut into `2^L` wedges. The
directional stage is available on its own as [`Surfacelets.make_hourglass`](@ref)
and [`Surfacelets.make_ndfb`](@ref), public but not exported.

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
[Contourlets.jl](https://github.com/hakkelt/Contourlets.jl), and 2-D and 3-D
shearlet transforms in [Shearlets.jl](https://github.com/hakkelt/Shearlets.jl);
the packages share their interface, defined in
[MultiscaleTransformsBase.jl](https://github.com/hakkelt/MultiscaleTransformsBase.jl)
and re-exported here, and the threading and mask-bank engine of
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl).

## Multithreading and GPU

The transforms run in parallel over Julia's threads unless `threaded = false`:
the nonsubsampled transform computes its subbands in parallel, the decimated
transform splits each scale's blocks among the threads.

The transforms run on GPU backends with an FFT library: CUDA.jl (NVIDIA),
AMDGPU.jl (AMD) and Metal.jl (Apple silicon, `Float32` only); the test suite
runs on CUDA. OpenCL.jl is supported but much slower than the CPU, because it
has no FFT: FFTW runs the FFTs on the CPU on `CLArray`s in shared virtual memory
(OpenCL.jl's default on devices that support it). Move a workspace to the device
with `Adapt`:

```julia
using CUDA, Adapt, Surfacelets

x   = CuArray(vol)
ws  = make_workspace(t, x)    # built on the host, moved to the device
c   = analyze(t, x; workspace = ws)
```
