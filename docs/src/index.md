# Surfacelets.jl

Surfacelets.jl computes 3-D surfacelet transforms. A surfacelet transform splits
a volume by scale, like a 3-D wavelet transform, and each scale further by
direction. Each directional subband then holds the surfaces of one scale and one
orientation, so volumes with smooth surfaces, such as organs in an MRI or CT
scan, have sparse representations. The package has two transforms:

- [`NSSurfacelet`](@ref), the nonsubsampled transform: every subband has the
  size of the volume, the transform commutes with circular shifts, and it is a
  Parseval frame (it preserves energy, and its inverse is its adjoint).
- [`Surfacelet`](@ref), Lu and Do's decimated transform: about 4 coefficients
  per voxel whatever the number of directions, but not shift invariant.

## Installation

Surfacelets.jl and its dependencies MultiscaleFilterBanks.jl and
AbstractMultiscaleTransforms.jl are not registered yet. Clone the three repositories
side by side and develop them into your environment:

```bash
git clone https://github.com/hakkelt/AbstractMultiscaleTransforms.jl AbstractMultiscaleTransforms
git clone https://github.com/hakkelt/MultiscaleFilterBanks.jl MultiscaleFilterBanks
git clone https://github.com/hakkelt/Surfacelets.jl Surfacelets
```

```julia
using Pkg
Pkg.develop([PackageSpec(path = "AbstractMultiscaleTransforms"),
             PackageSpec(path = "MultiscaleFilterBanks"),
             PackageSpec(path = "Surfacelets")])
```

## Example

```jldoctest
julia> using Surfacelets

julia> vol = [sin(i + 2j) * cos(k) for i in 1:16, j in 1:16, k in 1:16];

julia> t = NSSurfacelet(scales = 2, direction_levels = [1, 2]);

julia> coeffs = analyze(t, vol);

julia> size(coeffs)   # the lowpass and 6 + 12 directional subbands
(16, 16, 16, 19)

julia> subband_layout(t, size(vol))[2]
(scale = 1, direction = 1)

julia> maximum(abs, synthesize(t, coeffs) .- vol) < 1e-10
true
```

## Where to go next

- [How to use the transforms](@ref guide): choosing a transform, workspaces,
  streaming large volumes, threads and GPUs.
- [Theory](@ref theory): how the transforms are built.
- [Properties of the transforms](@ref properties): what they guarantee,
  checked when this documentation is built.
- [API reference](@ref api).

The package shares its interface (`analyze`, `synthesize`, `make_workspace`,
`subband_layout` and the rest), defined in
[AbstractMultiscaleTransforms.jl](https://github.com/hakkelt/AbstractMultiscaleTransforms.jl)
and re-exported here, with
[Contourlets.jl](https://github.com/hakkelt/Contourlets.jl) (2-D contourlet
transforms) and [Shearlets.jl](https://github.com/hakkelt/Shearlets.jl) (2-D and
3-D shearlet transforms).
