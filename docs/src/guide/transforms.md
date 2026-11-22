# [How to use the transforms](@id guide)

Recipes for common tasks. [Theory](@ref theory) explains how the transforms are
built and what they guarantee; the [API reference](@ref api) lists every function.

```@setup guide
using Surfacelets, Random, Statistics
```

## Choose a transform

| | [`NSSurfacelet`](@ref) | [`Surfacelet`](@ref) |
|---|---|---|
| Subband sizes | all equal to the input | decimated, one grid per subband |
| Redundancy | `1 + Σ_j 3·2^{L_j}` (49 at `direction_levels = [2, 2, 3]`) | about 4, whatever the directions |
| Shift invariance | exact | no |
| Frame | tight (Parseval) | biorthogonal, perfect reconstruction |
| Streaming form | yes, one subband at a time | not needed, the set fits |
| Volume size | any | every scale must resample exactly |

Use `Surfacelet` when memory or iteration cost dominates and shift invariance
does not matter, for example as the regularizer in an iterative reconstruction.
Use `NSSurfacelet` when you need shift invariance or a Parseval frame and the
volume is small enough, or when the streaming form is enough.

## Transform a volume and invert it

```@example guide
vol = randn(Xoshiro(1), 32, 32, 32)
t   = NSSurfacelet(scales = 2, direction_levels = [1, 2])

c   = analyze(t, vol)
rec = synthesize(t, c)

println("reconstruction error: ", maximum(abs, rec .- vol))
@assert maximum(abs, rec .- vol) < 1e-12
```

The nonsubsampled coefficients are one array of size `(size(vol)..., n)`. The
decimated transform's subbands differ in size, so its coefficients are a
`SubbandVector`: one vector of all coefficients, each subband a reshaped view
into it. Either way broadcasting, `similar`, `copyto!`, `norm` and `dot` act on all
coefficients at once, and `Adapt` moves them to a device.

## Get one subband and see what it is

`subband(t, c, i)` is subband `i`. [`subband_layout(t, size(vol))`](@ref
subband_layout) describes each subband: the lowpass is subband 1, then come the
scales from coarsest to finest, and within a scale the hourglasses in order,
`2^L` subbands each.

```@example guide
layout = subband_layout(t, size(vol))
println(layout[1])
println(layout[2])
println(size(subband(t, c, 2)))
```

## Reuse a workspace in a loop

`make_workspace` builds the FFT plans and mask tables once. With `analyze!` and
`synthesize!` a loop then allocates almost nothing. The workspace can stand in
for the transform in every call.

```@example guide
ws     = make_workspace(t, vol)
out    = similar(vol)
coeffs = similar_coefficients(t, vol)

for _ in 1:3
    analyze!(coeffs, ws, vol)
    synthesize!(out, ws, coeffs)
end
println("error: ", maximum(abs, out .- vol))
```

A workspace holds scratch buffers for each task. To run several transforms at
the same time, give each its own workspace. The keyword `tasks` limits the
number of buffer sets, and `cache_bandpass = false` trades speed for memory
(nonsubsampled transform only).

## Stream the subbands of a large volume

At 256³ `Float32`, `direction_levels = [2, 2, 3]` produces 49 volumes of 64 MB,
3.1 GB in all. The streaming form calls `f(i, band)` once per subband, directional
subbands first and the lowpass `i = 1` last, and reuses one buffer. Build the
workspace with `tasks = 1` for the smallest memory footprint.

```@example guide
sws1   = make_workspace(t, vol; tasks = 1)
energy = zeros(nsubbands(t, size(vol)))

analyze(t, vol; workspace = sws1) do i, band
    # `band` is valid only until the callback returns: copy or reduce here.
    energy[i] = sum(abs2, band)
end
println("lowpass energy share: ", round(energy[1] / sum(energy), digits = 3))
```

The synthesis streams too: `synthesize!(f, x, t)` asks `f(i)` for each subband
when it needs it, so coefficients can come from disk or be regenerated.

```@example guide
r = synthesize!(similar(vol), t) do i
    subband(t, c, i)
end
println("error: ", maximum(abs, r .- vol))
@assert maximum(abs, r .- vol) < 1e-12
```

## Transform complex data

Complex volumes need no extra step. They run as two real channels through the same
filters, so `T(x + i·y) == T(x) + i·T(y)` holds exactly.

```@example guide
xr = randn(Xoshiro(2), 16, 16, 16)
xi = randn(Xoshiro(3), 16, 16, 16)
t1 = NSSurfacelet(scales = 1, direction_levels = [1])

cc = analyze(t1, complex.(xr, xi))
cr = analyze(t1, xr)
ci = analyze(t1, xi)
println("linearity error: ", maximum(abs,
    subband(t1, cc, 2) .- complex.(subband(t1, cr, 2), subband(t1, ci, 2))))
```

## Use the decimated transform

Sizes must suit its pyramid: with the default `pyramid_ratio = 1.5`, the first
level resamples by 3/2 and every further level by 2, and each scale must be even
and divide into its wedge blocks. A size that does not fit throws an error naming
the nearest one that does; the docstring of
[`make_workspace`](@ref make_workspace(::Surfacelet, ::AbstractArray)) gives the
exact rule. The workspace takes the keywords of the nonsubsampled one, except
`cache_bandpass`.

```@example guide
svol = randn(Xoshiro(3), 48, 48, 48)
st   = Surfacelet(scales = 2, direction_levels = [1, 2])
sws  = make_workspace(st, svol)

sc = analyze(st, svol; workspace = sws)   # allocates the coefficients once
analyze!(sc, st, svol; workspace = sws)   # then reuses them
sr = synthesize(st, sc; workspace = sws)

println("error:      ", maximum(abs, sr .- svol))
println("redundancy: ", round(surfacelet_redundancy(st, size(svol)), digits = 3))
@assert maximum(abs, sr .- svol) < 1e-11
```

The decimated transform is not a tight frame, so its inverse is not its adjoint.
Use the adjoint only where the algorithm does not need an exact inverse.

## Control threading

Both transforms use Julia's threads; start Julia with `-t auto` or a fixed
count. Pass `threaded = false` to `analyze`, `synthesize` or their `!` forms to run
serially. The streaming forms always run serially, in order, on the calling task.
When the calls are already inside a threaded loop of your own, give each task its
own workspace and pass `threaded = false`.

## Run on a GPU

Load a GPU backend with an FFT library (CUDA.jl, AMDGPU.jl, or Metal.jl with
`Float32`) together with `Adapt`. Build the workspace from a device array; it is
created on the host and moved to the device.

```julia
using CUDA, Adapt, Surfacelets

x  = CuArray(vol)
ws = make_workspace(t, x)
c  = analyze(t, x; workspace = ws)
```

OpenCL.jl works but is slower than the CPU, because its FFTs run on the CPU.

## Check which subband a direction lands in

A transform that reconstructs exactly can still have the wrong frequency
partition, so check a plane wave. A wave whose frequency is largest along axis 3
must put its energy in a subband of the third hourglass:

```@example guide
n  = 48
wave = [cospi(2 * (3 * (i - 1) + 7 * (j - 1) + 19 * (l - 1)) / n)
        for i in 1:n, j in 1:n, l in 1:n]
wt = Surfacelet(scales = 1, direction_levels = [2])
wc = analyze(wt, wave)

wl   = subband_layout(wt, size(wave))
we   = [sum(abs2, subband(wt, wc, i)) for i in 2:length(wl)]
best = 1 + argmax(we)
println("winning subband: ", best, " of ", length(wl), " → ", wl[best])
@assert wl[best].hourglass == 3
```
