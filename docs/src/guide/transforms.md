# [Transforms](@id guide)

Two transforms share the same frequency partition — three hourglasses around
the frequency axes, each cut into wedges. The hourglass split cannot be
critically sampled above two dimensions ([Theory](@ref theory)). They differ in
which stages decimate:

```
nonsubsampled  volume ─ raised-cosine pyramid (no decimation) ─ NDFB masks (no decimation)
decimated      volume ─ Fourier pyramid (resampled by 3/2, 2) ─ ritf + IRC (critical)
```

| | [`NSSurfacelet`](@ref) | [`Surfacelet`](@ref) |
|---|---|---|
| Subband sizes | all equal to the input | decimated, one grid per subband |
| Redundancy | `1 + Σ_j 3·2^{L_j}` (49 at `direction_levels = [2, 2, 3]`) | ≈ 4 (`pyramid_ratio = 1.5`), whatever the directions |
| Shift invariance | exact | no — decimation breaks it |
| Frame | tight (Parseval) | biorthogonal, perfect reconstruction |
| Streaming form | yes, one subband at a time | not needed, the set fits |
| Size constraints | none | every scale must resample exactly — see [`make_workspace`](@ref make_workspace(::Surfacelet, ::AbstractArray)) |
| Use for | analysis, shift-invariant denoising | sparsifying transform in reconstruction |

Choosing: take the decimated transform when memory or iteration cost dominates
and shift invariance does not matter (e.g. as a regularizer inside an iterative
reconstruction), and the nonsubsampled one when exact shift invariance or a
Parseval frame is required and the volume is small enough (or the streaming form
suffices).

Both are transform objects: [`analyze`](@ref) computes the coefficients and
`synthesize` inverts it.  Both split each scale into `3 · 2^L`
directional subbands: three hourglasses aligned with the frequency axes, each
cut into `2^L` wedges.  `subband(t, c, i)` is subband `i` of the
coefficients `c`, and [`subband_layout(t, size(x))`](@ref subband_layout) says
what each one is: the lowpass is subband 1, then come the scales from the coarsest
to the finest, each hourglass-major — the first `2^L` subbands of a scale
belong to the first hourglass (``|\omega_1|`` dominant), the next `2^L` to the
second, the last `2^L` to the third.

The coefficients are arrays.  The nonsubsampled transform stacks its subbands
into one array of size `(size(x)..., n)`, subband `i` being `c[:, :, :, i]`; the
decimated transform's subbands differ in size, so its coefficients are a
`SubbandVector`: one vector of all coefficients, each subband a reshaped view
into it.  Either way broadcasting, `similar`, `copyto!`, `norm` and `dot` act on
all coefficients at once, which is what an iterative algorithm needs, and
`Adapt` moves them to a device.

```@setup guide
using Surfacelets, Plots, Random, Statistics
Plots.default(show = false)
```

## Nonsubsampled surfacelets

```@example guide
using Surfacelets, Random, Statistics

vol = randn(Xoshiro(1), 32, 32, 32)
t   = NSSurfacelet(scales = 2, direction_levels = [1, 2])

c   = analyze(t, vol)
rec = synthesize(t, c)

println("PR error: ", maximum(abs, rec .- vol))
layout = subband_layout(t, size(vol))
for j in 1:t.scales
    println("scale $j: $(count(d -> d.scale == j, layout)) subbands of $(size(subband(t, c, 2)))")
end
@assert maximum(abs, rec .- vol) < 1e-12
```

With the default `tight = true` the transform is a Parseval frame: the
coefficients carry exactly the energy of the volume, ``\|Sx\| = \|x\|``, and
the inverse is the adjoint. That is what an analysis-sparsity regularizer or a
thresholding denoiser assumes: modified coefficients are synthesized by the
adjoint of the analysis, and an error added to them reaches the image with no
more energy than it had.

```@example guide
E = sum(i -> sum(abs2, subband(t, c, i)), 1:nsubbands(t, size(vol)))
println("‖Sx‖² / ‖x‖²: ", E / sum(abs2, vol))
@assert isapprox(E, sum(abs2, vol); rtol = 1e-10)
```

### Workspace

[`make_workspace`](@ref make_workspace(::NSSurfacelet, ::AbstractArray)) builds the FFT plans and mask tables once. This
is the entry point for reconstruction loops, where the same volume size is
transformed hundreds of times. The whole cascade runs as masks on the input's
spectrum — one forward FFT, then one inverse FFT per subband — and the subbands
are computed in parallel on Julia's threads unless `threaded = false`.
With `analyze!` and `synthesize!` a loop
allocates nothing but the threads' tasks; the `tasks` keyword caps the threads,
and with them the buffers.  By default the workspace runs one analysis and
synthesis at construction (`prewarm = true`), so that the first call in a loop
is already compiled.  The workspace caches each scale's
pyramid bandpass as one real array on the rfft grid; pass
`cache_bandpass = false` to save that memory at some speed.

```@example guide
ws     = make_workspace(t, vol)       # or make_workspace(t, Float64, size(vol))
out    = similar(vol)
coeffs = similar_coefficients(t, vol)

for _ in 1:3
    analyze!(coeffs, ws, vol)          # the workspace stands in for the transform
    synthesize!(out, ws, coeffs)
end
println("PR error with workspace: ", maximum(abs, out .- vol))
@assert maximum(abs, out .- vol) < 1e-12
```

### Streaming — the form to use at working sizes

At 256³ `Float32` one subband is 64 MB, so `direction_levels = [2, 2, 3]`
materializes 49 volumes — 3.1 GB — before anything is done with them. The
streaming form `analyze(f, t, x)` calls `f(i, band)` once per subband (the
directional subbands first, the lowpass `i = 1` last) and reuses a single
buffer, holding two volumes plus the plans.

```@example guide
energy = zeros(nsubbands(t, size(vol)))

analyze(t, vol) do i, band
    # `band` is valid only until this callback returns — copy or reduce here.
    energy[i] = sum(abs2, band)
end

println("coarse energy:      ", round(energy[1], digits = 2))
println("directional energy: ", round(sum(energy[2:end]), digits = 2))
```

The synthesis streams too: `synthesize!(f, x, t)` asks `f(i)` for each subband on
demand, so a caller that keeps coefficients on disk — or regenerates them — never
holds the full set.

```@example guide
r = synthesize!(similar(vol), t) do i
    subband(t, c, i)
end
println("streaming inverse PR error: ", maximum(abs, r .- vol))
@assert maximum(abs, r .- vol) < 1e-12
```

### Shift invariance

```@example guide
s = (3, -2, 5)
a = analyze(t, vol)
b = analyze(t, circshift(vol, s))

err = maximum(
    maximum(abs, subband(t, b, i) .- circshift(subband(t, a, i), s))
        for i in 1:nsubbands(t, size(vol))
)
println("shift equivariance error: ", err)
@assert err < 1e-10
```

### Complex volumes

Complex data is filtered as two real passes through the identical kernel, so the
transform is exactly real-linear — `T(x + i·y) == T(x) + i·T(y)`, bit for bit.
That makes it a free correctness check on code built on top.

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

## Surfacelets — the decimated transform

The surfacelet transform of Lu and Do stores each subband at its own reduced
resolution. It gives up shift invariance and gets back a coefficient set about 4
times the input, which is what makes it usable as a sparsifying transform rather
than only as an analysis tool. It follows their reference implementation,
SurfBox, stage by stage, and reproduces its coefficients to rounding error:

1. a frequency-domain pyramid that resamples by 3/2 or 2 per level,
2. the undecimated hourglass bank at every scale (redundancy 3), and
3. iterated resampled checkerboard (IRC) filter banks inside each hourglass,
   which refine it into wedges *and decimate critically* — each wedge is sampled
   on a lattice aligned with it.

Sizes must suit the pyramid: with the default `pyramid_ratio = 1.5` the first level
resamples by 3/2, so every dimension must be divisible by 3 and then by 2 once
per further scale. [`make_workspace`](@ref make_workspace(::Surfacelet, ::AbstractArray))
builds the pyramid profiles, hourglass masks and checkerboard filter slices, and
the buffers and FFT plans the transform runs in; build it once and reuse it.
With a workspace, `analyze!` and `synthesize!` write into
existing coefficients and volumes and allocate nothing but a few array headers
and the threads' tasks. The blocks of each scale are split among Julia's
threads, up to the workspace's `tasks` (one per thread by default), unless
`threaded = false`. Its workspace takes the keywords of the nonsubsampled
transform's, except `cache_bandpass`. A workspace
holds scratch buffers, so give each task its own.  Complex volumes are
transformed as two real channels, exactly real-linearly.

```@example guide
svol = randn(Xoshiro(3), 48, 48, 48)
st   = Surfacelet(scales = 2, direction_levels = [1, 2])
sws  = make_workspace(st, svol)

sc = analyze(st, svol; workspace = sws)   # allocates the coefficients once
analyze!(sc, st, svol; workspace = sws)   # then reuses them
sr = synthesize(st, sc; workspace = sws)

println("PR error:   ", maximum(abs, sr .- svol))
println("redundancy: ", round(surfacelet_redundancy(st, size(svol)), digits = 3))
@assert maximum(abs, sr .- svol) < 1e-11
```

### Subbands have different sizes

This is the structural difference from the nonsubsampled transform: the
subbands have *different* shapes, because a coarser scale covers a smaller
frequency shell.

```@example guide
slayout = subband_layout(st, size(svol))
println("volume:  ", size(svol))
println("coarse:  ", size(subband(st, sc, 1)))
for j in 1:st.scales
    idx = findall(d -> d.scale == j, slayout)
    println("scale $j: ", length(idx), " subbands, sizes ",
        unique(size(subband(st, sc, i)) for i in idx))
end

stored = sum(i -> length(subband(st, sc, i)), eachindex(slayout))
println("stored coefficients: ", stored, " for ", length(svol), " voxels")
```

Hourglass `K` keeps axis `K` at the scale's full length and halves the other two
axes once per checkerboard stage, so each hourglass stores exactly as many
coefficients as its scale has voxels, however finely it is cut. That is why the
redundancy does not grow with the direction count:

```@example guide
for L in ([1, 1], [2, 2], [3, 3], [4, 4])
    sp = Surfacelet(scales = 2, direction_levels = L, checkerboard_order = 4)
    println("L = $L: ", lpad(3 * 2^L[1], 3), " + ", lpad(3 * 2^L[2], 3),
        " subbands → redundancy ", round(surfacelet_redundancy(sp, (96, 96, 96)), digits = 3),
        "   (nonsubsampled: ", 1 + sum(3 * 2^l for l in L), ")")
end
```

The pyramid mode is the redundancy dial — 1, 1.5 and 2 give about 6.4, 4.0 and
3.4, the figures Lu and Do report:

```@example guide
for mode in (1, 1.5, 2)
    sp = Surfacelet(scales = 3, direction_levels = [2, 2, 2], pyramid_ratio = mode, checkerboard_order = 4)
    println("pyramid_ratio = $mode → redundancy ", round(surfacelet_redundancy(sp, (96, 96, 96)), digits = 2))
end
```

### Perfect reconstruction, not a tight frame

The pyramid and the hourglass stage are tight, but the checkerboard filters are
biorthogonal: analysis and synthesis use different filters, so the transform is
a frame with exact reconstruction whose inverse is not its adjoint. Energy is
preserved only approximately:

```@example guide
energy = sum(i -> sum(abs2, subband(st, sc, i)), eachindex(slayout))
println("Σ‖sᵢ‖² / ‖x‖² = ", round(energy / sum(abs2, svol), digits = 3))
```

Higher filter orders `checkerboard_order` (4 to 18, default 12) are sharper directionally. They
cost nothing extra here: the filters are evaluated in closed form on the
frequency grid, which also lifts SurfBox's restriction that every block be wider
than the spatial filter.

### What a surfacelet looks like

Put a single unit coefficient into one subband and invert: the result is that
subband's synthesis atom — a plate-like function, elongated transverse to its
wedge and oscillating along the wedge axis.

```@example guide
using Plots
Plots.default(show = false)

atom_t  = Surfacelet(scales = 1, direction_levels = [1])
atom_ws = make_workspace(atom_t, Float64, (48, 48, 48))

function atom(k)
    c = analyze(atom_t, zeros(48, 48, 48); workspace = atom_ws)
    subband(atom_t, c, 1 + k)[1, 1, 1] = 1.0     # one unit coefficient, at the band's origin
    return circshift(synthesize(atom_t, c; workspace = atom_ws), (24, 24, 24))
end

function panel(k)
    slice = atom(k)[:, :, 25]
    m = maximum(abs, slice)
    return heatmap(slice, color = :balance, clims = (-m, m), aspect_ratio = :equal,
        axis = false, colorbar = false, title = "wedge $k")
end

plot([panel(k) for k in (1, 3, 5)]...; layout = (1, 3), size = (900, 300))
savefig("surfacelet_atoms.png"); nothing # hide
```

![](surfacelet_atoms.png)

Each panel is the central ``x_3`` slice of a different directional subband's
atom (the `k`-th wedge of the scale, subband `1 + k`), on its own symmetric
color scale. Wedges 1 and 3 belong to the first two hourglasses and oscillate
along their respective axes; wedge 5 belongs to the third, whose axis is
perpendicular to the slice, so only its transverse envelope remains in-plane.

### Not shift-invariant — and that is the trade

```@example guide
s = (3, -2, 5)
sa = analyze(st, svol; workspace = sws)
sb = analyze(st, circshift(svol, s); workspace = sws)

rel(x, y) = maximum(abs, x .- y) / maximum(abs, x)
println("shifted vs circshifted coefficients, scale 1 wedge 1: ",
    round(rel(circshift(subband(st, sa, 2), s), subband(st, sb, 2)), digits = 3))
```

The decimated grid does not carry the shift. Reconstruction is unaffected; only
the coefficient values move. If a downstream algorithm needs shift invariance,
use [`NSSurfacelet`](@ref) and pay the redundancy.

## Directional selectivity

Perfect reconstruction proves *nothing* about the directional partition — the
masks are a partition of unity whatever shape they have, so a wrong tiling
reconstructs just as exactly. The check that matters is *which* subband a plane
wave lands in. Here the frequency vector is steepest along axis 3, so the energy
must land in the third hourglass.

```@example guide
n = 32
kvec = (2, 5, 13)
wave = [
    cospi(2 * (kvec[1] * (i - 1) + kvec[2] * (j - 1) + kvec[3] * (l - 1)) / n)
        for i in 1:n, j in 1:n, l in 1:n
]

using Surfacelets: make_ndfb, ndfb_nsubbands, ndfb_decompose, ndfb_subband_triple
fb = make_ndfb(Float64, (n, n, n); levels = 4, transition = 0.5)
e  = zeros(ndfb_nsubbands(fb))
ndfb_decompose(wave, fb) do idx, band
    e[idx] = sum(abs2, band)
end

best = argmax(e)
hourglass, p, q = ndfb_subband_triple(fb, best)
println("winning subband $best: hourglass $hourglass, wedge ($p, $q)")
println("its share of the energy: ", round(e[best] / sum(e), digits = 3))
@assert hourglass == 3
```

The surfacelet transform partitions the same way, through different filters —
SurfBox's hourglass and checkerboard banks rather than the Bernstein masks:

```@example guide
m48  = 48
wave48 = [
    cospi(2 * (3 * (i - 1) + 7 * (j - 1) + 19 * (l - 1)) / m48)
        for i in 1:m48, j in 1:m48, l in 1:m48
]
wt = Surfacelet(scales = 1, direction_levels = [2])
wc = analyze(wt, wave48)

wlayout = subband_layout(wt, size(wave48))
we      = [sum(abs2, subband(wt, wc, i)) for i in 2:length(wlayout)]
wbest   = 1 + argmax(we)
println("winning subband: ", wbest, " of ", length(wlayout), " → ", wlayout[wbest])
@assert wlayout[wbest].hourglass == 3
```

The nonsubsampled stages are regression-tested against SurfBox's partition, and
the decimated transform against its coefficients, stage by stage — see
[Validation against SurfBox](@ref theory).
