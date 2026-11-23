# [How to use the transforms](@id guide)

Recipes for common tasks with the two transforms: [`NSSurfacelet`](@ref), the
nonsubsampled surfacelet transform, whose subbands all have the size of the
input, and [`Surfacelet`](@ref), the decimated transform, whose subbands are
downsampled. A transform splits a volume into *subbands*: the *lowpass* (the
coarse approximation) and, at every scale, directional subbands. Each scale is
split into three *hourglasses*, one around each frequency axis, and each
hourglass into `2^L` *wedges* (directions) for `L` direction levels.
[Theory](@ref theory) explains the construction and
[Properties of the transforms](@ref properties) what each transform guarantees.

## Choose a transform

Use `NSSurfacelet` when you need shift invariance or a Parseval frame (a
transform whose inverse is its adjoint and which preserves energy), for
example for denoising by thresholding. Its coefficients take
``1 + \sum_j 3 \cdot 2^{L_j}`` times the memory of the volume, or one subband at
a time with the [streaming form](@ref "Stream the subbands of a large volume").
Use `Surfacelet` when memory or iteration cost matters more, for example as the
regularizer of an iterative reconstruction: it stores about 4 coefficients per
voxel whatever the number of directions, but it is not shift invariant and
accepts only volume sizes that suit its pyramid.

## Transform a volume and invert it

```@example transform
using Surfacelets, Random

vol = randn(Xoshiro(1), 32, 32, 32)
t   = NSSurfacelet(scales = 2, direction_levels = [1, 2])

c   = analyze(t, vol)
rec = synthesize(t, c)

println("size of the coefficients: ", size(c))
println("reconstruction error: ", maximum(abs, rec .- vol))
@assert maximum(abs, rec .- vol) < 1e-12
```

`direction_levels` lists the direction levels per scale from the coarsest to
the finest; scale `j` has `3 · 2^direction_levels[j]` directional subbands.
Giving only `scales` chooses the levels for you (see [`NSSurfacelet`](@ref)).
The nonsubsampled coefficients are one array of size `(size(vol)..., n)` for `n`
subbands. The decimated coefficients are a `SubbandVector`: one vector of all
coefficients, with each subband a reshaped view into it. Either kind works with
broadcasting, `similar`, `copyto!`, `norm` and `dot` over all coefficients.

Complex volumes need no extra step. They are transformed as two real volumes,
so `analyze(t, x + im * y)` equals `complex.(analyze(t, x), analyze(t, y))`
exactly:

```@example transform
x = randn(Xoshiro(2), 16, 16, 16)
y = randn(Xoshiro(3), 16, 16, 16)
t1 = NSSurfacelet(scales = 1, direction_levels = [1])
@assert analyze(t1, complex.(x, y)) == complex.(analyze(t1, x), analyze(t1, y))
```

## Get one subband and find out what it is

`subband(t, c, i)` returns subband `i`, and [`subband_layout`](@ref) describes
every subband. The lowpass is subband 1; then come the scales from the coarsest
to the finest, and inside each scale hourglass 1, 2 and 3 in turn.

```@example transform
layout = subband_layout(t, size(vol))
println(layout[1], "  ", layout[2], "  ", layout[end])
println(size(subband(t, c, 2)))
```

## Use the decimated transform

Every axis must suit the pyramid of [`Surfacelet`](@ref): with the default
`pyramid_ratio = 1.5` the first scale is resampled by 3/2 and every coarser one
by 2, and each scale's length must be even and divisible by its number of
wedges along that axis. A size that does not fit throws an `ArgumentError`
naming the nearest size that does; the docstring of
[`make_workspace`](@ref make_workspace(::Surfacelet, ::AbstractArray)) gives
the rule.

```@example decimated
using Surfacelets, Random

vol = randn(Xoshiro(3), 48, 48, 48)
t   = Surfacelet(scales = 2, direction_levels = [1, 2])

c   = analyze(t, vol)
rec = synthesize(t, c)

println("reconstruction error: ", maximum(abs, rec .- vol))
println("redundancy: ", round(surfacelet_redundancy(t, size(vol)), digits = 3))
println("subband 2: ", subband_layout(t, size(vol))[2], ", size ", size(subband(t, c, 2)))
@assert maximum(abs, rec .- vol) < 1e-10
```

`synthesize` is the inverse of this transform but not its adjoint, because the
transform is not a tight frame. Iterative algorithms that need the adjoint of
the analysis should not use `synthesize` in its place.

## Reuse a workspace in a loop

A workspace holds the masks, FFT plans and scratch buffers of a transform for
one volume size, element type and array type. [`make_workspace`](@ref) builds
it once; pass it as `workspace`, or in place of the transform. Without one,
every call builds its own. With `analyze!`
and `synthesize!` a loop then reuses the coefficients and the output as well.

```@example workspace
using Surfacelets, Random

vol = randn(Xoshiro(1), 32, 32, 32)
t   = NSSurfacelet(scales = 2, direction_levels = [1, 2])

ws     = make_workspace(t, vol)
coeffs = similar_coefficients(t, vol)
out    = similar(vol)

for _ in 1:3
    analyze!(coeffs, ws, vol)
    synthesize!(out, ws, coeffs)
end
println("reconstruction error: ", maximum(abs, out .- vol))
@assert maximum(abs, out .- vol) < 1e-12
```

A workspace must not serve two calls at the same time; give each concurrent
task its own. Its keywords:

- `tasks`: the most tasks one call runs on, with one set of buffers per task.
  The default is one per Julia thread; `tasks = 1` builds the smallest
  workspace.
- `cache_bandpass = false` (`NSSurfacelet` only): uses less memory and
  recomputes part of each scale's mask on every call.
- `flags`: the FFTW planner flags, `FFTW.MEASURE` by default.
- `prewarm = false`: skips the trial run at construction that compiles the
  first call.

## Stream the subbands of a large volume

The coefficients of `NSSurfacelet` can be larger than memory: at 256³
`Float32` with `direction_levels = [2, 2, 3]` they are 49 volumes of 64 MiB,
3.1 GiB in all. The streaming analysis `analyze(f, t, vol)` calls
`f(i, band)` once per subband, the directional subbands first and the lowpass
(`i = 1`) last, and reuses one buffer for `band`. Copy or reduce `band` inside
`f`: it is overwritten after `f` returns.

```@example stream
using Surfacelets, Random

vol = randn(Xoshiro(1), 32, 32, 32)
t   = NSSurfacelet(scales = 2, direction_levels = [1, 2])

ws     = make_workspace(t, vol; tasks = 1)   # one set of buffers
energy = zeros(nsubbands(t, size(vol)))

analyze(t, vol; workspace = ws) do i, band
    energy[i] = sum(abs2, band)
end
println("share of the energy in the lowpass: ", round(energy[1] / sum(energy), digits = 3))
@assert sum(energy) ≈ sum(abs2, vol)
```

The streaming synthesis `synthesize!(f, x, t)` asks `f(i)` for subband `i`
when it needs it, so the subbands can come from disk or be computed on demand:

```@example stream
c = analyze(t, vol)
r = synthesize!(similar(vol), t; workspace = ws) do i
    subband(t, c, i)
end
println("reconstruction error: ", maximum(abs, r .- vol))
@assert maximum(abs, r .- vol) < 1e-12
```

Both streaming forms call `f` from the calling task, one subband at a time and
in order, so `f` need not be thread-safe.

## Control threading

Both transforms run on Julia's threads; start Julia with `--threads=auto` or a
fixed count. Pass `threaded = false` to `analyze`, `synthesize` or their `!`
forms to run serially; the streaming forms always run serially. When you call
the transforms from your own threaded loop, give each task its own workspace
and pass `threaded = false`.

## Run on a GPU

Load `Adapt` together with a GPU array package that provides FFT plans
through AbstractFFTs, such as CUDA.jl; the package is tested with CUDA.jl.
`make_workspace` called with a device array builds the workspace on the host
and moves it to the device; the transform then runs there. Without a
workspace, `analyze` builds one the same way on every call.

```julia
using CUDA, Adapt, Surfacelets

t  = NSSurfacelet(scales = 3, direction_levels = [2, 2, 3])
x  = CUDA.randn(Float32, 128, 128, 128)
ws = make_workspace(t, x)
c  = analyze(t, x; workspace = ws)
```

## Check which subband a direction lands in

A transform with the wrong directional partition still reconstructs exactly,
so check directions with a plane wave. A wave whose frequency vector is largest
along axis 3 must put most of its energy in a subband of the third hourglass:

```@example directions
using Surfacelets

n    = 48
wave = [cospi(2 * (3 * (i - 1) + 7 * (j - 1) + 19 * (l - 1)) / n)
        for i in 1:n, j in 1:n, l in 1:n]
t = Surfacelet(scales = 1, direction_levels = [2])
c = analyze(t, wave)

layout = subband_layout(t, size(wave))
energy = [sum(abs2, subband(t, c, i)) for i in 2:length(layout)]
best   = 1 + argmax(energy)
println("strongest subband: ", best, " → ", layout[best])
@assert layout[best].hourglass == 3
```

For `NSSurfacelet`, `subband_layout` gives each subband's scale and direction
index, which counts hourglass 1, 2 and 3 in turn, each with `2^L` wedges.
