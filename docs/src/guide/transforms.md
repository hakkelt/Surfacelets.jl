# [Transforms](@id guide)

Two transforms share the same frequency partition — three hourglasses around the
frequency axes, each cut into wedges, which cannot be critically sampled above
two dimensions ([Theory](@ref theory)):

| | [`nsurfacelet_forward`](@ref) | [`surfacelet_forward`](@ref) |
|---|---|---|
| Subband sizes | all equal to the input | decimated, one grid per subband |
| Redundancy | `1 + Σ_j 3·2^{L_j}` (49 at `J=3`, `L=[2,2,3]`) | ≈ 16 at the same settings |
| Shift invariance | exact | no — decimation breaks it |
| Streaming form | yes, one subband at a time | not needed, the set fits |
| GPU | CUDA / OpenCL verified | CUDA / OpenCL verified |
| Use for | analysis, shift-invariant denoising | sparsifying transform in reconstruction |

Both split each scale into `3 · 2^L` directional subbands: three hourglasses
aligned with the frequency axes, each cut into `2^L` wedges. Subbands are
ordered hourglass-major — indices `1:2^L` belong to the first hourglass
(``|\omega_1|`` dominant), the next `2^L` to the second, the last `2^L` to the
third.

```@setup guide
using Surfacelets, Plots, Random, Statistics
Plots.default(show = false)
```

## 3-D NSCT

```@example guide
using Surfacelets, Random, Statistics

vol    = randn(Xoshiro(1), 32, 32, 32)
params = NSurfaceletParams(J = 2, L_array = [1, 2])

c   = nsurfacelet_forward(vol, params)
rec = nsurfacelet_inverse(c, params)

println("PR error: ", maximum(abs, rec .- vol))
for j in 1:params.J
    println("scale $j: $(length(c.subbands[j])) subbands of $(size(c.subbands[j][1]))")
end
@assert maximum(abs, rec .- vol) < 1e-12
```

### Workspace

[`make_nsurfacelet_workspace`](@ref) builds the FFT plans and mask tables once. This
is the entry point for reconstruction loops, where the same volume size is
transformed hundreds of times.

```@example guide
ws  = make_nsurfacelet_workspace(params, size(vol), Float64)
out = similar(vol)

for _ in 1:3
    coeffs = nsurfacelet_forward(vol, params; workspace = ws)
    nsurfacelet_inverse!(out, coeffs, params; workspace = ws)
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

coarse = nsurfacelet_forward(vol, params) do j, k, band
    # `band` is valid only until this callback returns — copy or reduce here.
    energy[(j, k)] = sum(abs2, band)
end

println("coarse energy:      ", round(sum(abs2, coarse), digits = 2))
println("directional energy: ", round(sum(values(energy)), digits = 2))
```

The inverse streams too: `f(j, k)` returns each subband on demand, so a caller
that keeps coefficients on disk — or regenerates them — never holds the full set.

```@example guide
r = nsurfacelet_inverse(c.coarse, params) do j, k
    c.subbands[j][k]
end
println("streaming inverse PR error: ", maximum(abs, r .- vol))
@assert maximum(abs, r .- vol) < 1e-12
```

### Shift invariance

```@example guide
s = (3, -2, 5)
a = nsurfacelet_forward(vol, params)
b = nsurfacelet_forward(circshift(vol, s), params)

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
p  = NSurfaceletParams(J = 1, L_array = [1])

cc = nsurfacelet_forward(complex.(xr, xi), p)
cr = nsurfacelet_forward(xr, p)
ci = nsurfacelet_forward(xi, p)

println("linearity error: ", maximum(abs,
    cc.subbands[1][1] .- complex.(cr.subbands[1][1], ci.subbands[1][1])))
```

## Surfacelets — the decimated transform

The surfacelet transform stores each subband at its own reduced resolution. It
gives up shift invariance and gets back a much smaller coefficient set, which is
what makes it usable as a sparsifying transform rather than only as an analysis
tool.

Planning is the expensive step: [`make_surfacelet`](@ref) evaluates every
subband mask over the full grid to find its frequency support box. Plan once and
reuse — both transforms build a plan on demand, but a per-call plan dominates
the run time.

```@example guide
sparams = SurfaceletParams(J = 2, L_array = [1, 1])
plan    = make_surfacelet(sparams, size(vol), Float64)

sc = surfacelet_forward(vol, sparams; plan = plan)
sr = surfacelet_inverse(sc, sparams; plan = plan)

println("PR error:   ", maximum(abs, sr .- vol))
println("redundancy: ", round(surfacelet_redundancy(plan), digits = 3))
@assert maximum(abs, sr .- vol) < 1e-11
```

### Subbands have different sizes

This is the one structural difference from every other container in the package:
[`SurfaceletCoefficients`](@ref) holds arrays of *different* shapes, because a
coarser scale covers a smaller frequency shell.

```@example guide
println("volume:  ", size(vol))
println("coarse:  ", size(sc.coarse))
for j in 1:sparams.J
    println("scale $j: ", length(sc.subbands[j]), " subbands, sizes ",
        unique(size.(sc.subbands[j])))
end

stored = length(sc.coarse) + sum(sum(length, s) for s in sc.subbands)
println("stored coefficients: ", stored, " for ", length(vol), " voxels")
```

[`surfacelet_redundancy`](@ref) reports exactly that ratio, *measured* from the
plan's boxes rather than derived from an idealised partition — so it reflects the
transition widths in use. Against the nonsubsampled transform at matching
parameters:

```@example guide
for (J, L) in [(1, [1]), (2, [1, 1]), (2, [2, 2])]
    sp = SurfaceletParams(J = J, L_array = L)
    pl = make_surfacelet(sp, (32, 32, 32), Float64)
    println("J=$J, L=$L: surfacelet ",
        lpad(round(surfacelet_redundancy(pl), digits = 1), 5),
        "   NSurfacelet ", 1 + sum(3 * 2^l for l in L))
end
```

About 3× leaner at useful settings — not the redundancy of roughly 4 that Lu and
Do report for the reference construction. The gap is structural, not a tuning
problem: an axis-aligned support box recovers the radial decimation but almost
none of the directional one. [Theory](@ref theory) has the details.

Wider transition bands give smoother, better-localised surfacelets and larger
support boxes, so they are the redundancy dial. At 32³ the effect is monotone but
small, because the boxes are already quantised to whole frequency bins:

```@example guide
for tr in (0.2, 0.5, 0.8)
    sp = SurfaceletParams(J = 1, L_array = [1], transition = tr,
        radial_transition = tr * 0.6, angular_transition = tr * 0.6)
    pl = make_surfacelet(sp, (32, 32, 32), Float64)
    println("transition = $tr → redundancy ",
        round(surfacelet_redundancy(pl), digits = 2))
end
```

### A tight frame, in the weighted sense

The masks are square roots of a partition of unity, ``\sum_i V_i^2 = 1``, so
analysis preserves energy. Because each subband lives on its own grid, the
Parseval identity carries the grid sizes with it:

```math
\sum_i N_i \, \|s_i\|^2 = N \, \|x\|^2
```

with ``N_i`` the subband's element count and ``N`` the volume's.

```@example guide
N = length(vol)
energy = length(sc.coarse) * sum(abs2, sc.coarse)
for j in 1:sparams.J, band in sc.subbands[j]
    global energy += length(band) * sum(abs2, band)
end
println("Σ Nᵢ‖sᵢ‖² / N‖x‖² = ", round(energy / (N * sum(abs2, vol)), digits = 12))
```

That identity is what thresholding wants: noise of variance ``\sigma^2`` arrives
in every subband with the same variance, so one threshold is meaningful across
the whole coefficient set.

### What a surfacelet looks like

Put a single unit coefficient into one subband and invert: the result is that
subband's synthesis atom — a plate-like function, elongated transverse to its
wedge and oscillating along the wedge axis.

```@example guide
using Plots
Plots.default(show = false)

atom_params = SurfaceletParams(J = 1, L_array = [1])
atom_plan   = make_surfacelet(atom_params, (32, 32, 32), Float64)

function atom(k)
    c = surfacelet_forward(zeros(32, 32, 32), atom_params; plan = atom_plan)
    c.subbands[1][k][1, 1, 1] = 1.0     # one unit coefficient, at the band's origin
    return circshift(surfacelet_inverse(c, atom_params; plan = atom_plan),
        (16, 16, 16))
end

function panel(k)
    slice = atom(k)[:, :, 17]
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
sa = surfacelet_forward(vol, sparams; plan = plan)
sb = surfacelet_forward(circshift(vol, s), sparams; plan = plan)

rel(x, y) = maximum(abs, x .- y) / maximum(abs, x)
println("shifted vs circshifted coefficients, scale 1 subband 1: ",
    round(rel(circshift(sa.subbands[1][1], s), sb.subbands[1][1]), digits = 3))
```

The decimated grid does not carry the shift. Reconstruction is unaffected; only
the coefficient values move. If a downstream algorithm needs shift invariance,
use [`nsurfacelet_forward`](@ref) and pay the redundancy.

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

The surfacelet transform partitions the same way; weighting by subband size
accounts for the differing grids:

```@example guide
wp = SurfaceletParams(J = 1, L_array = [1])
wc = surfacelet_forward(wave, wp; plan = make_surfacelet(wp, (n, n, n), Float64))

we    = [length(b) * sum(abs2, b) for b in wc.subbands[1]]
wbest = argmax(we)
println("winning subband: ", wbest, " of ", length(we),
    " → hourglass ", (wbest - 1) ÷ 2^wp.L_array[1] + 1)
@assert (wbest - 1) ÷ 2^wp.L_array[1] + 1 == 3
```

Both stages are regression-tested against SurfBox, Lu and Do's reference
implementation — see [Validation against SurfBox](@ref theory).
