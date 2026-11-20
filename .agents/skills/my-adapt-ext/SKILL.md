---
name: my-adapt-ext
description: Work on SurfaceletsAdaptExt, the GPU path of Surfacelets.jl. Use when running the transforms on a device, adding or changing an Adapt.adapt_structure method, or debugging a scalar-indexing or FFT-planning failure on the GPU.
---

# GPU path (`SurfaceletsAdaptExt`)

There are no kernels in this package's GPU path, and there must not be any.

## Why there is nothing to write

Both transforms are an FFT followed by a pointwise mask. Every mask is stored as
**broadcastable factors** — the mapping responses and wedge planes are 3-D arrays
with singleton axes along the frequencies they do not depend on — so mask
application, the complex real/imaginary split and the pyramid's frequency-domain
multiply are already plain broadcasts. A broadcast over a device array runs on
the device unchanged.

What the device does need is (a) the data moved across and (b) the FFT plans
rebuilt, because an FFTW plan cannot be moved. That is exactly what
`ext/SurfaceletsAdaptExt/*.jl` does, in `Adapt.adapt_structure` methods and
nothing else. Hence `Adapt` is the only weak dependency: no GPUArrays, no
KernelAbstractions.

```julia
using CUDA, Adapt, Surfacelets

x  = CuArray(vol)
ws = make_workspace(NSSurfacelet(J = 2, L_array = [1, 2]), x)   # host build + Adapt.adapt(CuArray, ⋅)
c  = analyze(transform(ws), x; workspace = ws)

st  = Surfacelet(J = 2, L_array = [1, 2])
sws = Adapt.adapt(CuArray, make_workspace(st, Float32, dims))
sc  = analyze(st, x; workspace = sws)
```

## Invariants — never break these

- **No scalar indexing in the mask path.** `_mask_multiply!`, `_mask_accumulate!`
  and `_apply_freq_filter!` must stay single broadcasts. Introducing a `for`
  loop over voxels there silently moves the whole transform back to the host (or
  trips the backend's scalar-indexing guard). The same applies to
  `_fill_surfacelet_mask!`, whose masks are evaluated on the fly as one broadcast
  over the per-axis frequency tables.
- **Support boxes are extracted by contiguous block copies.** They are symmetric
  about the origin, so a subband is copied as the (at most eight) blocks
  `1:reach+1` / `n-reach+1:n` per axis — never a per-voxel index map, which
  would be a gather.
- **Workspaces are built on the host and adapted.** `make_workspace(t, x)` for a
  device array `x` (and a call on one without a workspace) builds the host
  workspace and calls the extension's `adapt_workspace`, through
  `Base.get_extension`; pass a workspace to avoid rebuilding it per call.
- **Adapt a `SurfaceletWorkspace` to the *bare* array type** (`Adapt.adapt(CuArray, ws)`),
  never `Adapt.adapt(typeof(x), ws)`: the workspace holds both real and complex
  buffers, and an element-typed destination (`CuArray{Float32,3}`) cannot fit
  both. The workspace and the filter banks take either form.
- **A workspace adapt moves the shared hourglass response once.** `_adapt_ndfb`
  rebuilds each scale's directional bank *around an already-adapted hourglass
  bank*, so a `J`-scale workspace holds one copy of the (large) mapping response,
  not `J`. Do not replace it with a plain per-scale `Adapt.adapt`.
- **Scratch buffers are allocated, not adapted.** In
  `Adapt.adapt_structure(to, ::SurfaceletWorkspace)`, `spec` and `work` are created
  with `similar(mask, Complex{eltype(mask)}, dims)` — nothing needs copying, and
  adapting a complex array through an element-typed destination would try to
  narrow `Complex{T}` to `T`.

## Deliberately absent

- **No CUDA graph capture.** The forward and inverse are streaming — they hand
  each subband to a caller's callback, which is arbitrary host code, and stream
  capture admits none. There is nothing to win either: this is a few full-volume
  FFTs, where launch overhead is noise.
- **No JLArrays in the tests.** It ships no FFT implementation, so `plan_rfft`
  falls back to an AbstractFFTs generic that indexes elementwise. `test/runtests.jl`
  activates the GPU overlay with `include_cpu_backends = false`, and the items iterate
  `GPUBackends.fft_backends`, which GPUEnv filters to backends with an
  FFTW-compatible library (CUDA, AMDGPU). Metal / oneAPI / OpenCL expose their
  FFTs through their own APIs rather than AbstractFFTs, so they are out too.

## Quick diagnostic

```julia
using CUDA, Adapt, Surfacelets
@assert Base.get_extension(Surfacelets, :SurfaceletsAdaptExt) !== nothing

dims = (16, 16, 16)
x    = randn(Float32, dims...)
t    = NSSurfacelet(J = 1, L_array = [1], order = 6)
ref  = analyze(t, x)

ws  = make_workspace(t, CuArray(x))
c   = analyze(t, CuArray(x); workspace = ws)
@assert maximum(abs, Array(subband(t, c, 1)) .- subband(t, ref, 1)) < 1.0f-4
```

If this errors with a scalar-indexing message, the cause is a broadcast that was
turned into a loop, or a gather that was introduced into the subband extraction —
look there first, not at the FFT.

Run the GPU items with `julia --project=test test/runtests.jl :gpu`. On a host
with no CUDA or AMDGPU they are vacuously true, so a green `:gpu` run proves
nothing unless `GPUBackends.fft_backends` is non-empty.
