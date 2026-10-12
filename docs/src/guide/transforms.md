# [Transforms](@id guide)

Three transforms share the same frequency partition — three hourglasses around
the frequency axes, each cut into wedges. The hourglass split cannot be
critically sampled above two dimensions ([Theory](@ref theory)). They differ in
which stages decimate:

```
nonsubsampled  volume ─ à trous pyramid (no decimation) ─ NDFB masks (no decimation)
STI            volume ─ raised-cosine pyramid (no decimation) ─ ritf + IRC (critical)
decimated      volume ─ Fourier pyramid (resampled by 3/2, 2) ─ ritf + IRC (critical)
```

| | [`ns_surfacelet_forward`](@ref) | [`sti_surfacelet_forward`](@ref) | [`surfacelet_forward`](@ref) |
|---|---|---|---|
| Subband sizes | all equal to the input | coarse full size; directional subbands decimated, same sizes at every scale | decimated, one grid per subband |
| Redundancy | `1 + Σ_j 3·2^{L_j}` (49 at `J=3`, `L=[2,2,3]`) | `1 + 3J` (10 at `J=3`), whatever the directions | ≈ 4 (`pyr_mode = 1.5`), whatever the directions |
| Shift invariance | exact | scales exact; subbands only for shifts on their decimation lattice | no — decimation breaks it |
| Frame | tight (Parseval) | perfect reconstruction (biorthogonal IRC) | biorthogonal, perfect reconstruction |
| Streaming form | yes, one subband at a time | not needed | not needed, the set fits |
| Size constraints | none | even, divisible by `2^level` per directional level | every scale must resample exactly — see [`make_surfacelet`](@ref) |
| Use for | analysis, shift-invariant denoising | denoising / regularisation where 49× is too much but scale invariance matters | sparsifying transform in reconstruction |

Choosing: take the decimated transform when memory or iteration cost dominates
and shift invariance does not matter (e.g. as a regulariser inside an iterative
reconstruction), the nonsubsampled one when exact shift invariance or a
Parseval frame is required and the volume is small enough (or the streaming form
suffices), and the STI transform in between.

All three split each scale into `3 · 2^L` directional subbands: three hourglasses
aligned with the frequency axes, each cut into `2^L` wedges. Subbands are
ordered hourglass-major — indices `1:2^L` belong to the first hourglass
(``|\omega_1|`` dominant), the next `2^L` to the second, the last `2^L` to the
third.

```@setup guide
using Surfacelets, Plots, Random, Statistics
Plots.default(show = false)
```

## Nonsubsampled surfacelets

```@example guide
using Surfacelets, Random, Statistics

vol    = randn(Xoshiro(1), 32, 32, 32)
params = NSSurfaceletParams(J = 2, L_array = [1, 2])

c   = ns_surfacelet_forward(vol, params)
rec = ns_surfacelet_inverse(c, params)

println("PR error: ", maximum(abs, rec .- vol))
for j in 1:params.J
    println("scale $j: $(length(c.subbands[j])) subbands of $(size(c.subbands[j][1]))")
end
@assert maximum(abs, rec .- vol) < 1e-12
```

### Workspace

[`make_ns_surfacelet_workspace`](@ref) builds the FFT plans and mask tables once. This
is the entry point for reconstruction loops, where the same volume size is
transformed hundreds of times. The whole cascade runs as masks on the input's
spectrum — one forward FFT, then one inverse FFT per subband — and the subbands
are computed in parallel on Julia's threads unless `threading = Disabled()`.
With [`ns_surfacelet_forward!`](@ref) and [`ns_surfacelet_inverse!`](@ref) a loop
allocates nothing but the threads' tasks.

```@example guide
ws     = make_ns_surfacelet_workspace(params, size(vol), Float64)
out    = similar(vol)
coeffs = ns_surfacelet_forward(vol, params; workspace = ws)

for _ in 1:3
    ns_surfacelet_forward!(coeffs, vol, params; workspace = ws)
    ns_surfacelet_inverse!(out, coeffs, params; workspace = ws)
end
println("PR error with workspace: ", maximum(abs, out .- vol))
@assert maximum(abs, out .- vol) < 1e-12
```

### Streaming — the form to use at working sizes

At 256³ `Float32` one subband is 64 MB, so `J = 3` with `L = [2, 2, 3]`
materialises 49 volumes — 3.1 GB — before anything is done with them. The
streaming form calls back once per subband and reuses a single buffer, holding
two volumes plus the plans.

```@example guide
energy = Dict{Tuple{Int, Int}, Float64}()

coarse = ns_surfacelet_forward(vol, params) do j, k, band
    # `band` is valid only until this callback returns — copy or reduce here.
    energy[(j, k)] = sum(abs2, band)
end

println("coarse energy:      ", round(sum(abs2, coarse), digits = 2))
println("directional energy: ", round(sum(values(energy)), digits = 2))
```

The inverse streams too: `f(j, k)` returns each subband on demand, so a caller
that keeps coefficients on disk — or regenerates them — never holds the full set.

```@example guide
r = ns_surfacelet_inverse(c.coarse, params) do j, k
    c.subbands[j][k]
end
println("streaming inverse PR error: ", maximum(abs, r .- vol))
@assert maximum(abs, r .- vol) < 1e-12
```

### Shift invariance

```@example guide
s = (3, -2, 5)
a = ns_surfacelet_forward(vol, params)
b = ns_surfacelet_forward(circshift(vol, s), params)

err = maximum(
    maximum(abs, b.subbands[j][k] .- circshift(a.subbands[j][k], s))
        for j in 1:params.J for k in eachindex(a.subbands[j])
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
p  = NSSurfaceletParams(J = 1, L_array = [1])

cc = ns_surfacelet_forward(complex.(xr, xi), p)
cr = ns_surfacelet_forward(xr, p)
ci = ns_surfacelet_forward(xi, p)

println("linearity error: ", maximum(abs,
    cc.subbands[1][1] .- complex.(cr.subbands[1][1], ci.subbands[1][1])))
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

Sizes must suit the pyramid: with the default `pyr_mode = 1.5` the first level
resamples by 3/2, so every dimension must be divisible by 3 and then by 2 once
per further scale. [`make_surfacelet`](@ref) builds the pyramid profiles,
hourglass masks and checkerboard filter slices, and the buffers and FFT plans the
transform runs in; plan once and reuse. With a plan,
[`surfacelet_forward!`](@ref) and [`surfacelet_inverse!`](@ref) write into
existing coefficients and volumes and allocate nothing but a few array headers.
A plan holds scratch buffers, so give each task its own.

```@example guide
svol    = randn(Xoshiro(3), 48, 48, 48)
sparams = SurfaceletParams(J = 2, L_array = [1, 2])
plan    = make_surfacelet(sparams, size(svol), Float64)

sc = surfacelet_forward(svol, sparams; plan = plan)   # allocates the coefficients once
surfacelet_forward!(sc, svol, sparams; plan = plan)   # then reuses them
sr = surfacelet_inverse(sc, sparams; plan = plan)

println("PR error:   ", maximum(abs, sr .- svol))
println("redundancy: ", round(surfacelet_redundancy(plan), digits = 3))
@assert maximum(abs, sr .- svol) < 1e-11
```

### Subbands have different sizes

This is the one structural difference from every other container in the package:
[`SurfaceletCoefficients`](@ref) holds arrays of *different* shapes, because a
coarser scale covers a smaller frequency shell.

```@example guide
println("volume:  ", size(svol))
println("coarse:  ", size(sc.coarse))
for j in 1:sparams.J
    println("scale $j: ", length(sc.subbands[j]), " subbands, sizes ",
        unique(size.(sc.subbands[j])))
end

stored = length(sc.coarse) + sum(sum(length, s) for s in sc.subbands)
println("stored coefficients: ", stored, " for ", length(svol), " voxels")
```

Hourglass `K` keeps axis `K` at the scale's full length and halves the other two
axes once per checkerboard stage, so each hourglass stores exactly as many
coefficients as its scale has voxels, however finely it is cut. That is why the
redundancy does not grow with the direction count:

```@example guide
for L in ([1, 1], [2, 2], [3, 3], [4, 4])
    sp = SurfaceletParams(J = 2, L_array = L, bo = 4)
    pl = make_surfacelet(sp, (96, 96, 96), Float64)
    println("L = $L: ", lpad(3 * 2^L[1], 3), " + ", lpad(3 * 2^L[2], 3),
        " subbands → redundancy ", round(surfacelet_redundancy(pl), digits = 3),
        "   (nonsubsampled: ", 1 + sum(3 * 2^l for l in L), ")")
end
```

The pyramid mode is the redundancy dial — 1, 1.5 and 2 give about 6.4, 4.0 and
3.4, the figures Lu and Do report:

```@example guide
for mode in (1, 1.5, 2)
    sp = SurfaceletParams(J = 3, L_array = [2, 2, 2], pyr_mode = mode, bo = 4)
    pl = make_surfacelet(sp, (96, 96, 96), Float64)
    println("pyr_mode = $mode → redundancy ", round(surfacelet_redundancy(pl), digits = 2))
end
```

### Perfect reconstruction, not a tight frame

The pyramid and the hourglass stage are tight, but the checkerboard filters are
biorthogonal: analysis and synthesis use different filters, so the transform is
a frame with exact reconstruction whose inverse is not its adjoint. Energy is
preserved only approximately:

```@example guide
energy = sum(abs2, sc.coarse) + sum(sum(abs2, b) for s in sc.subbands for b in s)
println("Σ‖sᵢ‖² / ‖x‖² = ", round(energy / sum(abs2, svol), digits = 3))
```

Higher filter orders `bo` (4 to 18, default 12) are sharper directionally. They
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

atom_params = SurfaceletParams(J = 1, L_array = [1])
atom_plan   = make_surfacelet(atom_params, (48, 48, 48), Float64)

function atom(k)
    c = surfacelet_forward(zeros(48, 48, 48), atom_params; plan = atom_plan)
    c.subbands[1][k][1, 1, 1] = 1.0     # one unit coefficient, at the band's origin
    return circshift(surfacelet_inverse(c, atom_params; plan = atom_plan),
        (24, 24, 24))
end

function panel(k)
    slice = atom(k)[:, :, 25]
    m = maximum(abs, slice)
    return heatmap(slice, color = :balance, clims = (-m, m), aspect_ratio = :equal,
        axis = false, colorbar = false, title = "subband $k")
end

plot([panel(k) for k in (1, 3, 5)]...; layout = (1, 3), size = (900, 300))
savefig("surfacelet_atoms.png"); nothing # hide
```

![](surfacelet_atoms.png)

Each panel is the central ``x_3`` slice of a different subband's atom, on its own
symmetric colour scale. Subbands 1 and 3 belong to the first two hourglasses and
oscillate along their respective axes; subband 5 belongs to the third, whose axis
is perpendicular to the slice, so only its transverse envelope remains in-plane.

### Not shift-invariant — and that is the trade

```@example guide
s = (3, -2, 5)
sa = surfacelet_forward(svol, sparams; plan = plan)
sb = surfacelet_forward(circshift(svol, s), sparams; plan = plan)

rel(x, y) = maximum(abs, x .- y) / maximum(abs, x)
println("shifted vs circshifted coefficients, scale 1 subband 1: ",
    round(rel(circshift(sa.subbands[1][1], s), sb.subbands[1][1]), digits = 3))
```

The decimated grid does not carry the shift. Reconstruction is unaffected; only
the coefficient values move. If a downstream algorithm needs shift invariance,
use [`ns_surfacelet_forward`](@ref) and pay the redundancy — or the
semi-translation-invariant transform below, which keeps most of it at a fraction
of the cost.

## Semi-translation-invariant (STI) surfacelets — a middle ground

The nonsubsampled transform is exactly shift-invariant but stores
`1 + Σ_j 3·2^{L_j}` volumes; the decimated one stores about 4 but is not
shift-invariant at all. [`sti_surfacelet_forward`](@ref) combines a
*nonsubsampled* pyramid with the *critically sampled* directional stage of the
decimated transform. Every scale is a full-size volume, so the scale
decomposition is exactly shift-invariant, and inside each scale the three
hourglasses are refined and decimated by IRC — so each scale costs exactly three
volumes, however many directions it has. The redundancy is `1 + 3J`:

```@example guide
stivol    = randn(Xoshiro(4), 32, 32, 32)
stiparams = STISurfaceletParams(J = 3, L_array = [2, 2, 3])
stiplan   = make_sti_surfacelet(stiparams, size(stivol))

stic = sti_surfacelet_forward(stivol, stiparams; plan = stiplan)
println("PR error:   ", maximum(abs, sti_surfacelet_inverse(stic, stiparams; plan = stiplan) .- stivol))
println("redundancy: ", surfacelet_redundancy(stiplan),
    "   (nonsubsampled at the same settings: ", 1 + sum(3 * 2^l for l in stiparams.L_array), ")")
```

| | nonsubsampled | STI | decimated |
|---|---|---|---|
| Redundancy at `J=3`, `L=[2,2,3]` | 49 | 10 | ≈ 4 |
| Scale decomposition | shift-invariant | shift-invariant | resampled |
| Directional subbands | shift-invariant | invariant to shifts on their decimation lattice | not invariant |
| Size constraints | none | even, divisible by `2^level` | the 3/2 pyramid's |

The coarse band commutes with any circular shift; a directional subband of
hourglass `K` is decimated by `2^l` along each other axis, so a shift by a
multiple of that moves its coefficients exactly:

```@example guide
s  = (8, 8, 16)
stia = sti_surfacelet_forward(stivol, stiparams; plan = stiplan)
stib = sti_surfacelet_forward(circshift(stivol, s), stiparams; plan = stiplan)
println("coarse: ", maximum(abs, circshift(stia.coarse, s) .- stib.coarse))
lv = stiparams.levels[1][1]                  # hourglass 1 of scale 1
sh = (s[1], s[2] ÷ 2^lv[2], s[3] ÷ 2^lv[3])
println("subband 1: ", maximum(abs, circshift(stia.subbands[1][1], sh) .- stib.subbands[1][1]))
```

The name comes from the structure this shares with Eslami and Radha's
semi-translation-invariant contourlet transform (STICT): a nonsubsampled,
shift-invariant pyramid followed by a critically sampled directional stage. It
is not a 3-D STICT. The STICT uses the translation-invariant Laplacian pyramid —
the decimated pyramid's polyphase channels for every lattice shift, four
highpass channels per level, keeping its aliasing — where this transform uses
one raised-cosine bandpass per scale (SurfBox's mode-1 band edges without the
resampling) and Lu and Do's IRC directional stage. Neither Lu and Do's papers
nor any later paper we know of describes this 3-D construction; it composes two
published stages, each validated on its own: the directional stage against
SurfBox, the pyramid by its tight-frame identity.

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

fb = make_ndfb((n, n, n), Float64; levels = 4, transition = 0.5)
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
wp = SurfaceletParams(J = 1, L_array = [2])
wc = surfacelet_forward(wave48, wp; plan = make_surfacelet(wp, (m48, m48, m48), Float64))

we    = [sum(abs2, b) for b in wc.subbands[1]]
wbest = argmax(we)
println("winning subband: ", wbest, " of ", length(we),
    " → hourglass ", (wbest - 1) ÷ 2^wp.L_array[1] + 1)
@assert (wbest - 1) ÷ 2^wp.L_array[1] + 1 == 3
```

The nonsubsampled stages are regression-tested against SurfBox's partition, and
the decimated transform against its coefficients, stage by stage — see
[Validation against SurfBox](@ref theory).
