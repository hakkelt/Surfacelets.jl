# Surfacelets.jl

[![CI](https://github.com/hakkelt/Surfacelets.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/hakkelt/Surfacelets.jl/actions/workflows/CI.yml)
[![Lint](https://github.com/hakkelt/Surfacelets.jl/actions/workflows/lint.yml/badge.svg?branch=master)](https://github.com/hakkelt/Surfacelets.jl/actions/workflows/lint.yml)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://hakkelt.github.io/Surfacelets.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://hakkelt.github.io/Surfacelets.jl/dev/)
[![Coverage](https://codecov.io/gh/hakkelt/Surfacelets.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/hakkelt/Surfacelets.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Runic](https://img.shields.io/badge/code%20style-runic-blue.svg)](https://github.com/fredrikekre/Runic.jl)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)


3-D surfacelet transforms in Julia, in a nonsubsampled (shift-invariant) and a
decimated form.

A surfacelet transform splits a volume by scale, like a 3-D wavelet transform,
and each scale further by direction. Each directional subband then holds the
surfaces of one scale and one orientation, so volumes with smooth surfaces,
such as organs in an MRI or CT scan, have sparse representations, which helps
denoising and compressed-sensing reconstruction.

- `NSSurfacelet`: every subband has the size of the volume; shift invariant; a
  Parseval frame; can stream one subband at a time.
- `Surfacelet`: Lu and Do's decimated transform; about 4 coefficients per voxel
  whatever the number of directions; not shift invariant.

## Installation

Surfacelets.jl and its dependencies MultiscaleFilterBanks.jl and
MultiscaleTransformsBase.jl are not registered yet. Clone the three repositories
side by side and develop them into your environment:

```bash
git clone https://github.com/hakkelt/MultiscaleTransformsBase.jl MultiscaleTransformsBase
git clone https://github.com/hakkelt/MultiscaleFilterBanks.jl MultiscaleFilterBanks
git clone https://github.com/hakkelt/Surfacelets.jl Surfacelets
```

```julia
using Pkg
Pkg.develop([PackageSpec(path = "MultiscaleTransformsBase"),
             PackageSpec(path = "MultiscaleFilterBanks"),
             PackageSpec(path = "Surfacelets")])
```

## Example

```julia
using Surfacelets

vol = randn(Float32, 96, 96, 96)
t   = NSSurfacelet(scales = 3, direction_levels = [2, 2, 3])   # 48 directional subbands

c   = analyze(t, vol)              # 96 × 96 × 96 × 49
rec = synthesize(t, c)             # equals vol up to rounding
subband(t, c, 2)                   # the first directional subband; subband 1 is the lowpass
subband_layout(t, size(vol))[2]    # (scale = 1, direction = 1)
```

The [documentation](https://hakkelt.github.io/Surfacelets.jl/stable/) covers
the decimated transform, workspaces, streaming large volumes, threads and GPUs,
and the theory. Surfacelets.jl shares its interface, defined in
[MultiscaleTransformsBase.jl](https://github.com/hakkelt/MultiscaleTransformsBase.jl),
with [Contourlets.jl](https://github.com/hakkelt/Contourlets.jl) (2-D) and
[Shearlets.jl](https://github.com/hakkelt/Shearlets.jl) (2-D and 3-D).

## License

MIT. The decimated surfacelet transform reimplements the construction of SurfBox
(© Yue Lu and Minh N. Do, MATLAB File Exchange 14485), whose source was consulted
to reproduce it; SurfBox's notice is retained in [`NOTICE`](NOTICE). The golden
data under `test/reference/` was produced by running SurfBox. No code is derived
from the MATLAB Contourlet or Nonsubsampled Contourlet toolboxes.

## Disclaimer

This package was created by [Claude](https://claude.com/claude-code), Anthropic's
AI model, under the direction of the author.
