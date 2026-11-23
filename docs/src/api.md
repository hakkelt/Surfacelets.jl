# [API Reference](@id api)

```@meta
CurrentModule = Surfacelets
```

Every public name of Surfacelets.jl, by topic.

## Module

```@docs
Surfacelets
```

## The interface

Surfacelets re-exports the interface of
[AbstractMultiscaleTransforms](https://github.com/hakkelt/AbstractMultiscaleTransforms.jl):
`analyze`, `analyze!`, `synthesize`, `synthesize!`, `make_workspace`,
`transform`, `similar_coefficients`, `subband_layout`, `nsubbands`, `subband`,
`SubbandVector`, `SubbandLayout`, `AbstractMultiscaleTransform` and
`AbstractTransformWorkspace`. Every function that takes a transform also takes
a workspace in its place: `analyze(ws, x)`, `analyze!(c, ws, x)`,
`synthesize(ws, c)`, `synthesize!(x, ws, c)`. The methods specific to the two
transforms are listed with them below.

## Nonsubsampled surfacelet transform

```@docs
NSSurfacelet
NSSurfaceletWorkspace
make_workspace(::NSSurfacelet, ::AbstractArray)
subband_layout(::NSSurfacelet, ::Dims)
similar_coefficients(::NSSurfacelet, ::AbstractArray)
analyze(::Any, ::NSSurfacelet, ::AbstractArray)
synthesize!(::Any, ::AbstractArray, ::NSSurfacelet)
```

## Decimated surfacelet transform

```@docs
Surfacelet
SurfaceletWorkspace
make_workspace(::Surfacelet, ::AbstractArray)
subband_layout(::Surfacelet, ::Dims)
similar_coefficients(::Surfacelet, ::AbstractArray)
surfacelet_redundancy
checkerboard_orders
```

## Directional filter bank

The directional stage of the nonsubsampled transform on its own, without the
pyramid. These names are public but not exported: call them as
`Surfacelets.make_ndfb`, or import them with
`using Surfacelets: make_ndfb, ndfb_decompose`.

```@docs
HourglassFB
make_hourglass
hourglass_mask
hourglass_weight
hourglass_decompose
hourglass_decompose!
hourglass_reconstruct
hourglass_reconstruct!
NDFB
make_ndfb
ndfb_sharing
ndfb_nsubbands
ndfb_subband_index
ndfb_subband_triple
ndfb_mask
ndfb_weight
ndfb_decompose
ndfb_decompose!
ndfb_reconstruct
ndfb_reconstruct!
```

## Hourglass filter design

The Bernstein mapping kernels the hourglass bank is built from, for designing
other kernels. Public but not exported, like the names in the section above.

```@docs
bernstein_table
dft_frequencies
mapping_coefficients
mapping_response
halfband_split
```
