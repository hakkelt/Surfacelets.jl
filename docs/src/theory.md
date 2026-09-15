# [Theory](@id theory)

```@meta
CurrentModule = Surfacelets
```

This page covers the two surfacelet transforms and the filter bank they share.

The surfacelet transform is a construction in its own right rather than a
contourlet transform with an extra axis.  Its directional stage is built from
Bernstein mapping kernels that carve the spectrum into hourglass-shaped regions
around the three frequency axes — a partition with no separable-lattice
description, so there is no sampling matrix, no shear index remap and no ladder
filter anywhere in it.  Lu and Do prove that hourglass-shaped frequency
partitioning *cannot* be achieved by a critically sampled filter bank above two
dimensions, so the redundancy of ``N`` in ``N`` dimensions is unavoidable rather
than a design choice.

## Nonsubsampled pyramid (NSP3D)

The multiscale stage is the direct extension of the 2-D à trous pyramid.  At
level ``j`` the filters are upsampled by ``2^{j-1}`` along each axis and nothing
is decimated:

```
coarse   = h_j ⊛ volume
bandpass = volume − g_j ⊛ coarse
```

The synthesis rescale `_NSP_SYNTH_SCALE` is applied **per axis**, so the 2-D
constant carries over unchanged.  This is worth stating explicitly because the
obvious guess is wrong in an instructive way: the biorthogonal synthesis ``g``
has DC gain 2 per axis, giving ``2^3 = 8`` across a separable 3-D filtering, and
one might expect the correction to become ``1/8``.  It does not — the ``1/2``
applies once per axis, and ``(1/2)^3`` is exactly the ``1/8`` needed.  A wrong
normalisation here does not break perfect reconstruction, it merely fills the
"bandpass" with low-pass leakage, which is why the test suite checks that a
constant volume leaves the bandpass at zero.

All filtering is periodic, so the stage is exactly equivariant under circular
shifts and — since periodic separable filtering *is* circular convolution — an
FFT evaluation of it is exact rather than approximate.  See
[`conv3d_sep_fft!`](@ref).

## Hourglass filter bank

The directional stage starts by splitting the spectrum into three
hourglass-shaped subbands, each aligned with one frequency axis:

```math
P_i = \{\omega : |\omega_i| \ge |\omega_j| \ \text{for all } j \ne i\}
```

so subband ``i`` keeps the frequencies closer to axis ``i`` than to any other.

The design follows Lu and Do's mapping-based construction.  One-dimensional
polynomials ``f_0, f_1`` are composed with a multidimensional *mapping kernel*
``M(\omega)`` — a zero-phase FIR filter designed so that ``M \approx +1`` on the
desired passband and ``\approx -1`` outside it — giving multidimensional filters
``f_0(M(\omega))`` and ``f_1(M(\omega))`` that inherit both the FIR property and
the perfect-reconstruction relation between ``f_0`` and ``f_1``.

The kernels come from multivariate Bernstein polynomials under the change of
variables ``x_i = (1 - \cos\omega_i)/2``, which maps ``|\omega_i|`` monotonically
onto ``[0, 1]``.  That is the reason the construction is tractable: an hourglass
passband is a statement about the *ordering* of frequency magnitudes, and after
the change of variables it becomes the same ordering statement about the integer
Bernstein sample indices — see [`mapping_coefficients`](@ref).

Three subbands are produced by a cascade of two two-channel banks:

```
spectrum ──┬─ P₁                             (3-D mapping kernel)
           └─ P₂ ∪ P₃ ──┬─ P₂                (2-D kernel on ω₂, ω₃)
                        └─ P₃
```

The second stage needs only two frequency variables.  That is not an
optimisation but a theorem: Lu and Do's Proposition 1 shows the optimal passband
support for a tree node with index set ``J`` involves ``|J|`` frequency
variables, so in 3-D the second cut is a 2-D design.

### Two deviations from the reference construction

**Frequency domain rather than FIR.**  Lu and Do deliberately chose FIR spatial
filters because their earlier frequency-domain design needed too much memory for
streaming video.  That trade-off inverts here.  These stages are nonsubsampled
with a periodic boundary, so the FFT is already the default backend, and
evaluating the mapping kernel directly on the DFT grid is *exact* — there is no
truncation of the 37³ and 61³ FIR filters the paper reports.

**A partition of unity, normalised into a Parseval frame.**  The 1-D
polynomials used here satisfy ``f_0(x) + f_1(x) = 1`` identically, so the three
*weights* ``W_k`` sum to 1 at every frequency, for every choice of `order` and
`poly`.  What the filters do with that partition is the `tight` setting of
[`make_hourglass`](@ref):

  * `tight = false` — the masks are the weights themselves.  The synthesis
    filters are all 1 and reconstruction is a plain sum, with no transforms at
    all.  But ``\sum_k W_k^2`` ranges over ``[1/3, 1]``, so the bank is a frame
    with bound ratio 3: analysis does not preserve energy, and white noise
    arrives with subband-dependent variance.

  * `tight = true` (the default) — the masks are ``\sqrt{W_k}``, the pointwise
    normalisation SurfBox uses.  Then ``\sum_k (\sqrt{W_k})^2 = \sum_k W_k = 1``
    identically: a Parseval frame, ``\sum_k \|s_k\|^2 = \|x\|^2``, with the
    analysis adjoint as its own inverse.  Synthesis reapplies the masks, so it
    costs a forward transform per subband plus one inverse rather than a bare
    sum.

The price is smaller than the extra transforms suggest, because the pyramid
dominates a 3-D NSCT.  At 128³ `Float32` with `J = 2`, `L = [2, 2]` and 8 FFTW
threads: the directional inverse costs 0.294 s tight against 0.096 s loose
(3.1×), but the full round trip only moves from 0.66 s to 0.76 s (1.15×).

Both are exact.  SurfBox normalises to ``\sum_k V_k^2 = N`` rather than 1 — the
same construction up to the constant ``\sqrt{N}`` — and
`test/items/test_surfbox_reference.jl` checks both against it.  Whichever
setting is in use, the deviation from Lu and Do's optimised lifting/ladder
coefficients costs nothing in frame conditioning; the pointwise normalisation
reaches tightness on its own.

The normalisation exposes a constraint the plain-sum path hides: each mask must
be **even** in frequency, or the masked spectrum is not Hermitian and the real
subband is a lossy record of it.  A plain sum never notices, since the
individual asymmetries cancel before the inverse transform.  See
"[A sign trap on the DFT grid](@ref sign-trap)" below — the same trap, in the same place,
for the same reason.

`order` (the Bernstein order ``N``) controls how sharply the subbands separate.
Larger ``N`` converges to the ideal indicator; the deviation from it falls
monotonically, from about 0.095 mean absolute error at ``N = 4`` to 0.024 at
``N = 20``.

## Directional filter bank (NDFB)

Each hourglass is then refined into wedges.  Inside ``P_i`` the two transverse
ratios

```math
s = \omega_a / \omega_i \in [-1, 1], \qquad t = \omega_b / \omega_i \in [-1, 1]
```

parameterise direction, and ``l`` levels of refinement cut each into ``2^l``
equal slabs.  With ``l_a + l_b = L`` this gives ``2^L`` wedges per hourglass and
``3 \cdot 2^L`` directional subbands in total, whose ideal passbands are
rectangular-based pyramids radiating from the origin.

Both ratios are invariant under ``\omega \to -\omega``, so the masks stay
symmetric and a real volume keeps real subbands.  Each ratio involves only two
frequency axes, so a wedge mask is a *matrix*: the whole bank stores
``2(2^{l_a} + 2^{l_b})`` planes no matter how large the volume is.

Lu and Do build these wedges from a chain of iteratively resampled checkerboard
filter banks, because their transform is critically sampled inside each
hourglass and the resampling is what keeps it so.  This bank is nonsubsampled,
so there is nothing to resample for and the wedge boundaries are placed directly
in the frequency domain.  The ideal partition is the same one.  Constructing it
directly also avoids the failure mode that a separable approximation invites:
filtering the transverse directions with 1-D filters produces stripe and block
tilings of the frequency plane, not wedges, and the resulting transform is still
perfectly reconstructing — so perfect reconstruction alone does not detect the
error.  The test suite therefore checks the wedge *index* a plane wave lands in,
not merely that some subband is large.

## 3-D NSCT

Composing the two stages gives the transform:

```
volume ─ NSP3D ─┬─ coarse ─→ next scale
               └─ bandpass ─ NDFB ─→ 3·2^L directional subbands
```

Every coefficient array keeps the size of the input, so the transform is exactly
shift-invariant — and correspondingly redundant: ``1 + \sum_j 3 \cdot 2^{L_j}``
copies of the input.  For ``J = 3`` and ``L = [2, 2, 3]`` that is 49 volumes,
3.1 GB at 256³ `Float32`.  The streaming form of [`nsurfacelet_forward`](@ref) exists
for this reason: it holds one subband at a time.

Because the directional bank depends only on the volume size and never on the
scale, a workspace plans the hourglass mapping response once and shares it
across every scale, with only the (matrix-sized) wedge masks differing.

## Running on a GPU

The 3-D stack needs no device kernels.  Both transforms are FFT plus a pointwise
mask, and every mask is stored as broadcastable factors — the mapping responses
and wedge planes are 3-D arrays with singleton axes where they do not depend on
a frequency — so applying a mask, splitting a complex volume into real and
imaginary halves, and the pyramid's frequency-domain multiply are all plain
broadcasts.  Moving a plan to the device is therefore just moving its arrays and
replanning the transforms:

```julia
using CUDA, Adapt, Surfacelets
using CUDA: CuArray
import Adapt

params = NSurfaceletParams(J = 2, L_array = [2, 2])
ws = Adapt.adapt(CuArray, make_nsurfacelet_workspace(params, (128, 128, 128), Float32))
c  = nsurfacelet_forward(CuArray(vol), params; workspace = ws)
```

A workspace adapt moves the shared hourglass response once, not once per scale.
Verified against the CPU path on CUDA and OpenCL.

The surfacelet transform runs on the device under the same rule, with one extra
constraint: its masks are evaluated on the fly, so `_fill_surfacelet_mask!` is a
*single* broadcast over the per-axis frequency tables, and a subband is extracted
by copying the (at most eight) contiguous corner blocks of its support box rather
than by a per-voxel index map.  Its plan is built on the host — planning scans
every mask to find its support — and then adapted to the *bare* array type
(`Adapt.adapt(CuArray, pl)`, never `typeof(x)`, since the plan holds both real
and complex buffers).  A device volume with no plan is an `ArgumentError`, not an
implicit rebuild.

## Surfacelet transform (decimated)

The NSurfacelet buys exact shift invariance with redundancy.  The surfacelet transform
trades the invariance back for a smaller coefficient set, which is what makes it
usable as a sparsifying transform rather than only as an analysis tool.

The target frequency partitioning is Lu and Do's: pairs of concentric cubes
radiating from the origin, at different scales and directions.  Scale ``j``
covers the shell ``\|\omega\|_\infty \in [\pi/2^j, \pi/2^{j-1}]``.

Decimation is done by frequency-domain support extraction.  Every mask here has
*compact* support in frequency, so the masked spectrum is restricted to the
smallest symmetric box containing it and inverse-transformed at that reduced
size — the mechanism the wrapping-based discrete curvelet transform uses.  Two
consequences:

* The directional masks are **not** the Bernstein mapping kernels used by the
  nonsubsampled bank.  Those are the right choice there and the wrong one here:
  a mapping kernel is a polynomial in ``\cos\omega``, so ``f_0(M(\omega))`` is
  nonzero almost everywhere and its bounding box is the whole grid.  The
  requirement inverts, so the construction does: masks are built from soft steps
  that are *exactly* 0 and 1 outside a transition band.

* Perfect reconstruction comes from ``\sum V_i^2 = 1`` (the masks are the square
  roots of a partition of unity) rather than ``\sum W_i = 1``.  Synthesis is then
  the analysis adjoint, and the frame is tight in the weighted sense
  ``\sum_i N_i \|s_i\|^2 = N \|x\|^2``, where ``N_i`` is the subband's element
  count.

### [A sign trap on the DFT grid](@id sign-trap)

On the DFT grid, ``\omega \to -\omega`` is *not* a sign flip of the folded
frequency: at the Nyquist bin of an even-length axis ``\pi \equiv -\pi`` maps to
itself.  A sign-sensitive quantity like the wedge ratio ``\omega_2/\omega_1``
therefore comes out with opposite signs at two bins that are each other's
negation.  That makes the masked spectrum non-Hermitian, the subband complex,
and — because subbands are stored real — destroys perfect reconstruction, with
no error message.  The masks are evaluated at a canonical representative of each
``\pm`` pair to remove the ambiguity.

The same trap bit `ndfb`, and stayed invisible for longer.  With the masks
applied once and reconstruction a plain sum, the individual asymmetries cancel
before any inverse transform: ``\sum_k W_k X = X`` whatever each ``W_k`` does at
Nyquist, so perfect reconstruction held while the masks were not even.  Turning
on `tight` makes synthesis reapply the masks, which requires each subband to
survive an `irfft`/`rfft` round trip, which requires the mask to be even on the
two self-conjugate planes of the rfft grid.  It was not, and reconstruction
failed by 0.27.

`_wedge_ratio` fixes the sign from the first coordinate that can carry one:
``\omega_\mathrm{ref}`` when it is neither 0 nor ``\pi``, and
``\omega_\mathrm{tr}`` otherwise.  Each factor canonicalises its own plane
independently, so the masks stay separable — the property the GPU path depends
on — and `ndfb — masks are even in frequency` pins the result.

### How much it actually saves

Measured redundancy against the nonsubsampled transform at the same parameters:

| `J`, `L`     | surfacelet | NSurfacelet |
|--------------|-------------|-------|
| 1, [1]       |  6.2        |   7   |
| 2, [1, 1]    |  7.7        |  13   |
| 2, [2, 2]    | 15.3        |  25   |
| 3, [2, 2, 3] | 16.0        |  49   |

So this is a *partially* decimated transform — about 3× leaner at useful
settings, not the redundancy of roughly 4 that Lu and Do report.  The reason is
worth knowing before trying to tune it away: an axis-aligned box recovers the
radial decimation in full (a coarser shell shrinks by 2 per axis) but almost
none of the directional decimation, because the bounding box of a pyramid
radiating from the origin is nearly as large as the pyramid's enclosing cube.
Closing that gap needs a box aligned with the wedge — sheared wrapping, or
equivalently Lu and Do's resampled checkerboard chain — and the shear has to be
an integer index remap on the sampling grid, which at dyadic wedge boundaries it
is not unless the decimation is interleaved with the refinement the way the
reference construction does it.  That is a missing piece, not a tuning problem.

That analysis is confirmed by the reference implementation.  Running SurfBox's
`Surfdec` on a 96³ volume at `Pyr_mode = 1.5` gives a redundancy of 3.9259 for
24 directional subbands and *the same* 3.9259 for 96 — flat in the direction
count, as sheared boxes predict.  This package's `surfacelet` gives 15.4 and
46.4 at the matching settings, growing roughly linearly.  So the gap is not a
constant factor; it widens exactly where a directional transform earns its keep.

One further limit worth stating plainly: the reference multiscale pyramid
resamples by the non-integer factor 3/2 per axis, which this package's cube-shell
octave partition does not do.

### What has been checked against SurfBox

SurfBox is available again (MATLAB File Exchange submission 14485), so the
directional stages are now regression-tested against it in
`test/items/test_surfbox_reference.jl`, with `test/reference/` holding the golden
data and the MATLAB scripts that regenerate it.  What the comparison establishes:

  * The tight normalisation matches.  SurfBox's masks satisfy
    ``\sum_k V_k^2 = 3`` to machine precision; this package's satisfy
    ``\sum_k V_k^2 = 1``, the same construction scaled by ``\sqrt{3}``.
  * The hourglass geometry agrees.  Over a 32³ grid the two constructions pick
    the same dominant hourglass at 94.6% of frequencies, with a mean absolute
    weight difference of 0.037; every disagreement is a near-tie in the
    reference (median gap 0.005 between the top two weights, against 0.887
    averaged over all frequencies), so the two differ only in how they round the
    transition bands.
  * The wedge partition agrees.  Across 80 plane waves at 12 directional
    subbands, the confusion matrix between SurfBox's subband labels and this
    package's is a bijection, 84% of directions match exactly under it, and for
    *every* one of the 80 the reference's subband is this package's first or
    second choice by energy.  This is the check that matters, because perfect
    reconstruction alone would pass for any structurally-PR bank, including one
    with the wrong tiling.

What it does not establish is coefficient-level agreement: SurfBox's hourglass
raises the magnitude response of a Kaiser-windowed diamond kernel to a power,
this package composes Bernstein mapping polynomials, and its NDFB is a decimated
resampled-checkerboard chain against directly-placed frequency boundaries here.
Same ideal partition, different discrete realisation.

## References

1. Y. M. Lu and M. N. Do, "Multidimensional Directional Filter Banks and
   Surfacelets", *IEEE Transactions on Image Processing*, 16(4):918–931, 2007.
2. Y. M. Lu and M. N. Do, "A Mapping-Based Design for Nonsubsampled Hourglass
   Filter Banks in Arbitrary Dimensions", *IEEE Transactions on Signal
   Processing*, 56(4):1466–1478, 2008.
3. G. G. Lorentz, *Bernstein Polynomials*, 2nd ed., Chelsea, 1986 — the
   multivariate Bernstein approximation behind the mapping kernels.
