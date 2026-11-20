# [API Reference](@id api)

```@meta
CurrentModule = Surfacelets
```

## Module

```@docs
Surfacelets
```

## The interface

Surfacelets re-exports the interface of
[MultiscaleTransformsBase](https://github.com/hakkelt/MultiscaleTransformsBase.jl):
`analyze`, `analyze!`, `synthesize`, `synthesize!`, `make_workspace`,
`transform`, `similar_coefficients`, `subband_layout`, `nsubbands`, `subband`,
`SubbandVector`, `SubbandLayout`, `AbstractMultiscaleTransform` and
`AbstractTransformWorkspace`.  Every verb also takes a workspace in place of the
transform: `analyze(ws, x)`, `analyze!(c, ws, x)`, `synthesize(ws, c)`,
`synthesize!(x, ws, c)`.  The methods specific to the two transforms are listed
with them below.

## Nonsubsampled surfacelet transform

```@docs
NSSurfacelet
NSSurfaceletWorkspace
NSSurfaceletCoefficients
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
SurfaceletCoefficients
make_workspace(::Surfacelet, ::AbstractArray)
subband_layout(::Surfacelet, ::Dims)
similar_coefficients(::Surfacelet, ::AbstractArray)
surfacelet_redundancy
checkerboard_orders
```

## Directional filter bank

Public but not exported (call them as `Surfacelets.make_ndfb` etc., or bring
them in with `using Surfacelets: make_ndfb, ndfb_decompose`): the directional
stage on its own, without the pyramid.

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

## Filter design internals

Public but not exported (call them as `Surfacelets.bernstein_table` etc.) —
these are the pieces the hourglass bank is built from, useful if you want to
design your own mapping kernels.

```@docs
bernstein_table
dft_frequencies
mapping_coefficients
mapping_response
halfband_split
```
