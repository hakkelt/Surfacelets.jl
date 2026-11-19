# [API Reference](@id api)

```@meta
CurrentModule = Surfacelets
```

## Module

```@docs
Surfacelets
```

## Nonsubsampled surfacelet transform

```@docs
ns_surfacelet_forward
ns_surfacelet_forward!
ns_surfacelet_inverse
ns_surfacelet_inverse!
ns_surfacelet_nsubbands
make_ns_surfacelet_workspace
NSSurfaceletParams
NSSurfaceletCoefficients
NSSurfaceletWorkspace
```

## Decimated surfacelet transform

```@docs
surfacelet_forward
surfacelet_forward!
surfacelet_inverse
surfacelet_inverse!
surfacelet_nsubbands
surfacelet_redundancy
make_surfacelet
SurfaceletParams
SurfaceletCoefficients
SurfaceletPlan
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
