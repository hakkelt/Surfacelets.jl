# AGENTS.md — Surfacelets.jl

This file guides AI agents (and human contributors) through the repository
structure and the conventions that are *specific to this package*. General Julia
best practices live in the skills under `.agents/skills/` (see below) and are not
repeated here.

## API Usage Pattern

```julia
using Surfacelets

vol    = randn(Float32, 128, 128, 128)
params = NSSurfaceletParams(J = 3, L_array = [2, 2, 3])   # 48 directional subbands

coeffs = ns_surfacelet_forward(vol, params)
rec    = ns_surfacelet_inverse(coeffs, params)
```

The nonsubsampled transform also has a **streaming** form, because the
materialised coefficient set does not fit at working sizes (3.1 GB at 256³
`Float32` with these settings):

```julia
ws = make_ns_surfacelet_workspace(params, size(vol), Float32)

coarse = ns_surfacelet_forward(vol, params; workspace = ws) do j, k, band
    # `band` is valid only until this returns
end
```

`surfacelet_forward` / `surfacelet_inverse` (and `!` variants) are the decimated
counterpart: same frequency partition, each subband at its own reduced
resolution, built through `make_surfacelet`.

`sti_surfacelet_forward` / `sti_surfacelet_inverse` (and `!` variants) are the
semi-translation-invariant (STI) transform: a nonsubsampled raised-cosine
pyramid with the decimated transform's critically sampled directional stage at
every scale (`make_sti_surfacelet`), redundancy `1 + 3J`. The name is for the
structure it shares with Eslami & Radha's STICT; it is not a 3-D STICT.

Both run on one `SurfaceletPlan` and one driver (`transforms/surfacelet.jl`);
the pyramid is a vector of `FourierPyramidLevel`s or `RaisedCosineLevel`s and
dispatch picks the stage-1 functions (`_pyramid_band!`, `_pyramid_merge!`). The
plan owns its spectra, IRC scratch, block buffers and FFT plans, so the `!`
paths allocate only array headers — and a plan must not serve two tasks at once.
A wedge block becomes its real subband through one `brfft` over `(K, others…)`,
the hourglass axis halved; the `1/N` rides on the hourglass mask.

## Quick Start

1. **Read this file** for repository layout and project-specific invariants.
2. **Load the matching skill** from `.agents/skills/` for the kind of work you
   are doing (performance, benchmarking, JET, packaging, docs, GPU).
3. **Honour the invariants** below — every change must keep them.

## Package family

```
MultiscaleFilterBanks.jl   (shared core: FilterPair, CDF97, à trous filters, threading)
        ├── Contourlets.jl   (2-D CT / NSCT)
        └── Surfacelets.jl   (this package)
```

This package depends on the core only. It does **not** depend on Contourlets.jl,
and the surfacelet transform is not a dimension-raised contourlet transform:
there is no quincunx sampling matrix, no shear index remap and no ladder filter
anywhere in it. The directional stage is Lu and Do's hourglass NDFB, built from
Bernstein mapping kernels, and Lu and Do prove that an hourglass partition
cannot be critically sampled above two dimensions — so the redundancy is
structural, not a design choice.

The shared surface with the core is small and explicit: `FilterPair`, `CDF97`,
`upsample_filter`, the threading policy types, and the internals
`_data_eltype`, `_filter_eltype`, `_use_threading`, `_clamp_idx`,
`_nsp_atrous_filters`.

## Commit Rules

- **Commit Message Formatting**: Follow the "Conventional Commits" format with the
  structure `type(scope): subject` and a detailed description (including
  performance numbers or context) separated by a blank line. For simple commits
  affecting only a couple lines, a detailed description is optional. Always append
  a co-author line referencing yourself. Do **not** add a `Claude-Session:`
  trailer line.
- **Committing Changes**: ONLY commit code upon explicit instruction from the user.

---

## Skills (`.agents/skills/`)

| Skill | Use when |
|---|---|
| `julia-perf` | Diagnosing slow code, reducing allocations, fixing type instabilities. |
| `julia-bench` | Writing/running benchmarks, comparing revisions, benchmark CI. |
| `julia-jet` | Running JET inference/type analysis on the package. |
| `julia-package-dev` | Environments, dependencies, extensions, multi-package workspace. |
| `julia-docs` | Documenter.jl site, docstrings, doctests, citations. |
| `my-adapt-ext` | Modifying anything under `ext/`, or running the transforms on a device. |
| `my-localcoverage` | Measuring coverage percentages, generating HTML reports. |

---

## Repository Layout

```
Surfacelets/
├── src/
│   ├── Surfacelets.jl                 # Main module — all exports here
│   ├── precompile.jl                  # PrecompileTools workload
│   ├── types/
│   │   ├── params.jl                  # NSSurfaceletParams, SurfaceletParams, STISurfaceletParams
│   │   └── coefficients.jl            # NSSurfaceletCoefficients, SurfaceletCoefficients
│   ├── filters/
│   │   ├── bernstein.jl               # Bernstein mapping kernels for the hourglass FB
│   │   └── checkerboard.jl            # closed-form checkerboard (IRC) filters
│   ├── primitives/
│   │   ├── conv3d.jl                  # separable convolution (direct)
│   │   └── conv3d_fft.jl              # periodic convolution via FFT (auto-selected)
│   ├── pyramid/
│   │   ├── nonsubsampled_pyramid3d.jl # nsp3d_decompose / nsp3d_reconstruct
│   │   ├── fourier_pyramid.jl         # decimated transform's resampling pyramid
│   │   └── raised_cosine_pyramid.jl   # STI transform's nonsubsampled pyramid
│   ├── directional/
│   │   ├── hourglass.jl               # nonsubsampled hourglass FB (3 subbands)
│   │   ├── ndfb_wedges.jl             # NDFB wedge masks (ratio slabs, sign canonicalisation)
│   │   ├── ndfb.jl                    # NDFB: hourglass + wedges (3·2^L subbands)
│   │   ├── ritf.jl                    # SurfBox's hourglass bank (decimated / STI)
│   │   ├── irc.jl                     # iterated resampled checkerboard filter banks
│   │   └── surfacelet_scale.jl        # one critically sampled directional scale (ritf + IRC)
│   └── transforms/
│       ├── ns_surfacelet.jl           # NS transform as DFT masks + workspace (threaded subbands)
│       ├── surfacelet.jl              # SurfaceletPlan, shared driver, decimated transform
│       └── sti_surfacelet.jl          # STI transform (nonsubsampled pyramid + IRC)
├── ext/
│   ├── SurfaceletsAdaptExt.jl
│   └── SurfaceletsAdaptExt/{ns_surfacelet_adapt.jl, surfacelet_adapt.jl}
├── test/
│   ├── runtests.jl                    # TestItemRunner entry point
│   ├── items/
│   └── reference/                     # SurfBox golden data + MATLAB generators
├── benchmark/benchmarks.jl
├── docs/
├── dev/bootstrap.jl                   # Pkg.develop the sibling core checkout
├── .agents/skills/
└── .github/workflows/
```

**Dimension tags in names are deliberate where they survive.** `conv3d_sep`,
`conv3d_sep_fft`, `Conv3DFFTPlan`, `make_conv3d_fft_plan` and `nsp3d_*` keep the
`3d`: dropping it would make them collide with Contourlets' `conv2d_sep` /
`nsp_decompose` when a user loads both packages. Everything at the transform and
filter-bank level has no tag, because the package name already says it.

---

## Implementation Conventions

- **Element types: real filters, real-or-complex data.** Filters are always real
  (`Tf <: AbstractFloat`); volume data may be complex (`Td <: Number`). The two
  flow independently — `Td = _data_eltype(volume)`,
  `Tf = _filter_eltype(Td) = real(float(Td))` — so a complex volume is filtered as
  complex·real, never complex·complex. The transform is real-linear, so
  `T(x + i·y)` equals `T(x) + i·T(y)` **bit-for-bit**; the suite asserts that with
  `== 0.0`, and it is a free correctness check for code built on top.
- **Complex is two real passes, everywhere.** `plan_rfft` is real-only, so a
  complex volume is filtered as its real part then its imaginary part through the
  identical kernel. Two real transforms cost about one complex transform, and it
  is what keeps the bit-for-bit linearity above true.
- **Periodic filtering goes through the FFT by default.** Periodic separable
  filtering is circular convolution, so an FFT evaluation is exact, not an
  approximation, and its cost is independent of tap count — which matters because
  à trous filters grow as `2^(j-1)` per axis. `conv3d_sep!` auto-selects it
  (`_conv3d_use_fft`) from a single crossover — five taps, the same for real and
  complex data and for any FFTW thread count. One threshold for both element types
  is load-bearing, not just tidy: a complex volume and its real and imaginary
  parts must reach the same kernel or the bit-for-bit linearity stops holding. At
  64³ Float32 with 33 taps: 183 ms direct vs 4.2 ms via FFT.
- **FFTW plans: no Julia-side plan cache.** FFTW already caches `MEASURE` wisdom
  globally in the C library, so re-planning a previously-seen size is
  sub-millisecond. A `Dict` of plan objects only adds a lock and a type-unstable
  cache. Use `FFTW.MEASURE` for repeated transforms, `FFTW.ESTIMATE` when a plan
  is used once and first-call latency matters.
- **Formatting.** Runic is mandatory — run `runic src/` before committing.
- **Static analysis.** Run ReLint after Runic before committing:
  ```bash
  runic src/
  JULIA_LOAD_PATH="@relint:@stdlib" julia --startup-file=no scripts/relint.jl src/
  ```

---

## Testing Conventions

- All tests are self-contained `@testitem` blocks (TestItemRunner): each block
  must import all dependencies except `Surfacelets` and `Test` inside the block.
- Tag taxonomy: `:filters`, `:primitives`, `:pyramid`, `:directional`,
  `:ns_surfacelet`, `:surfacelet` (also the STI transform), `:surfbox`, `:quality`, `:jet`, `:docs`, `:gpu`.
- Seed every random input (`Random.seed!(42)`); never rely on global RNG state.
- **Directional tests must assert the *index* of the subband a plane wave lands
  in.** See the design decision on perfect reconstruction below.

---

## Long-Running Commands

```bash
# All tests
julia --project=test test/runtests.jl
# Filter by tag (leading ':') or by test-item name
julia --project=test test/runtests.jl :directional,:gpu
julia --project=test test/runtests.jl "ns_surfacelet — shift invariance"

# Benchmarks: smoke test, then revision comparison
julia --project=benchmark --startup-file=no -e '
include("benchmark/benchmarks.jl"); using BenchmarkTools
BenchmarkTools.save(".temp/benchmark_results.json", run(SUITE; verbose=true, samples=3, seconds=5))
'
benchpkg Surfacelets --path . --rev=master,HEAD --script=benchmark/benchmarks.jl \
    --output-dir=.temp/bench --exeflags="--threads=2"
```

All generated output (test runs, coverage, benchmark JSON, profiles) goes under
`.temp/`, which is git-ignored.

**First checkout**: `MultiscaleFilterBanks.jl` is unregistered, so it is wired in
by path. `[sources]` handles that on Julia ≥ 1.11; on 1.10 run
`julia --project=. dev/bootstrap.jl` once, which `Pkg.develop`s the sibling
checkout into every environment of the workspace.

---

## Design Decisions

- **No downsampling in the nonsubsampled transform** — all subbands have the same
  spatial size as the input, and all NSP/NDFB filtering uses periodic (circular)
  convolution, so the transform is exactly invariant under circular shifts.
  Periodic filtering is a product on the DFT grid, so the workspace runs the whole
  cascade as masks on the input's rfft: pyramid bandpasses broadcast from per-axis
  responses (`MultiscaleFilterBanks._nsp_responses`), the hourglass factor applied
  once per hourglass (`Yᵢ = X·Bⱼ·√Wᵢ`), then per subband a multiply by two wedge
  planes (stored square-rooted when tight) and one irfft. The subbands are
  threaded over with `_foreach_chunk`, single-threaded plans; only
  `threading = Disabled()` and the streaming forms (whose callbacks need not be
  thread-safe) run on one task. `nsp3d_*`, `ndfb_*` and `Conv3DFFTPlan` remain as
  public building blocks; the transform no longer calls them.
- **The directional bank is built as frequency-domain masks.** This is exact for
  the periodic, nonsubsampled case, and it avoids the FIR truncation the papers
  need for streaming video. Perfect reconstruction is therefore *structural* — the
  weights are a partition of unity, so synthesis is either a plain sum or the same
  masks reapplied, depending on `tight`. Consequence worth remembering: **PR alone
  proves nothing about wedge geometry.** A wrong directional partition still
  reconstructs perfectly. Directional tests must assert the *index* of the subband
  a plane wave lands in.
- **The directional bank is a Parseval frame by default.** The three hourglass
  weights are a partition of unity for any `order`/`poly`, and so are the wedge
  weights, so the products are too. `tight = true` (the default on
  `make_hourglass` / `make_ndfb` / `NSSurfaceletParams`) takes the masks to be `√W`,
  which makes `Σ V² = 1` identically — the pointwise normalisation SurfBox uses,
  scaled to bound 1 instead of `N`. Analysis then preserves energy, which is what
  thresholding and iterative reconstruction want. The price is synthesis: the masks
  have to be reapplied, so the directional inverse costs a forward transform per
  subband plus one inverse instead of a bare sum. Pass `tight = false` to get the
  partition-of-unity bank back, with a free inverse and a frame-bound ratio of 3.
  **The normalisation makes mask evenness load-bearing.** A plain sum hides an
  asymmetric mask — `Σ W_k X = X` regardless — but reapplying `√W` requires each
  subband to survive an `irfft`/`rfft` round trip, which requires the mask to be
  even on the self-conjugate planes of the rfft grid. It was not, and PR failed by
  0.27 with no error. `_wedge_ratio` now fixes the sign from the first coordinate
  that can carry one; `ndfb — masks are even in frequency` pins it.
- **GPU needs no kernels.** Every transform is FFT plus pointwise masks, and
  every mask is stored as *broadcastable factors* — the mapping responses and wedge
  planes are 3-D arrays with singleton axes where they do not depend on a frequency
  — so mask application, the complex real/imaginary split, and the pyramid's
  frequency multiply are all plain broadcasts. Keep it that way: introducing scalar
  indexing into `_mask_multiply!`, `_mask_accumulate!` or `_apply_freq_filter!`
  silently breaks the device path. `ext/` therefore contains only
  `Adapt.adapt_structure` methods, which move the data and replan the FFTs (plans
  cannot be adapted), which is why `Adapt` is the only weak dependency — no
  GPUArrays, no KernelAbstractions. A workspace adapt moves the shared hourglass
  response **once**, not once per scale. JLArrays is excluded from the GPU tests: it
  has no FFT implementation, so `plan_rfft` falls back to a scalar-indexing
  generic. CUDA and OpenCL are verified.
  `surfacelet` and `sti_surfacelet` follow the same rule: the pyramid is a
  broadcast against its separable profile plus a crop through index views, the
  hourglass stage a broadcast against precomputed masks, and each IRC stage a
  broadcast between the two halves of a reshaped view. Their plan is built on the
  host — the checkerboard filter slices are evaluated point by point in closed
  form — and then adapted, which moves the tables and rebuilds the buffers and FFT
  plans on the device; a device volume with no plan is an `ArgumentError`, not
  an implicit rebuild. Adapt the plan to the *bare* array type
  (`Adapt.adapt(CuArray, pl)`), never `typeof(x)`: it holds real, complex and
  integer tables, and an element-typed destination cannot fit them all. Both are
  verified on CUDA.
- **Deliberately not CUDA-graph-captured.** The forward/inverse are streaming —
  they hand each subband to a caller's callback, which is arbitrary host code, and
  stream capture admits none — and there is nothing to win anyway: this is a few
  full-volume FFTs, where launch overhead is noise.
- **The directional stages are validated against SurfBox.** SurfBox (Lu and Do's
  reference implementation, MATLAB File Exchange 14485) is available and runs under
  `module load matlab`. `test/reference/` holds golden data extracted from it plus
  the MATLAB scripts that regenerate it, and `test/items/test_surfbox_reference.jl`
  checks this package against it. The agreement is on *geometry*, not coefficients:
  94.6% of frequencies pick the same dominant hourglass (every disagreement a
  near-tie), and across 80 plane waves the wedge labellings are related by a
  bijection with the reference's choice always this package's top-2. The decimated
  transform is checked coefficient for coefficient instead
  (`test/items/test_irc.jl`): it is SurfBox's IRC construction, agrees to about
  1e-13 of the array norm at every stage, and its redundancy matches SurfBox's own
  count (4.0046 at 96³ with 24 subbands per scale over three scales).
